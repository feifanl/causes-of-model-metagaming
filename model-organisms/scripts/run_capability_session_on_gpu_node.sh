#!/usr/bin/env bash
# Run PLAN step (b) and the 'Reasoning: low' diagnostic on a fresh GPU node, with no laptop needed.
#
#   NVME=/workspace bash model-organisms/scripts/run_capability_session_on_gpu_node.sh <deadline-epoch>
#
#   1. bf16 base; the _cotreg seed-0 adapters fetched from the HF repo and merged (PLAN (a)).
#   2. 'Reasoning: low' diagnostic (DECISIONS 'Getting the persona back with reasoning on'): base and
#      the _cotreg pair, reasoning on at low effort, pilot evals -> results/*_reasoning_low.json and
#      results/pilot_comparison_cotreg_reasoning_low.md.
#   3. Capability evals (PLAN (b)), reasoning on and off (DECISIONS 'Eval reasoning setting'): base and
#      the _cotreg pair -> results/*_capability_reasoning_on.json, results/*_capability.json.
#   4. RL organisms (PLAN (b)): Redwood merged on CPU in the background while the two AISI organisms
#      are merged and evaluated on the GPU halves; then Redwood's evals. Each organism gets the pilot
#      evals and the capability evals, reasoning on and off -> results/<organism>{,_reasoning_on,
#      _capability,_capability_reasoning_on}.json. Merged copies are deleted after their evals.
#
# API: ~$24 of judge calls (diagnostic ~$10, RL pilot evals ~$14; capability evals are judge-free);
# set SPEND_CAP above the ledger total + 25 in the node's env file. Same upload, deadline and failure
# behaviour as the other orchestrators; run setup_gpu_node.sh first. Exits non-zero, after uploading
# what exists, if any stage failed or was skipped for time.
set -uo pipefail

DEADLINE="${1:?usage: run_capability_session_on_gpu_node.sh <deadline-epoch>}"
NVME="${NVME:-/data}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
MO="$REPO/model-organisms"
RUN="$MO/scripts/run_pilot_stage_on_gpu_node.sh"
STATE="$NVME/pilot_state"
HF="$REPO/venv/bin/hf"
export NVME
# shellcheck disable=SC1091
[ -f "$HOME/.config/spar/env" ] && set -a && . "$HOME/.config/spar/env" && set +a
mkdir -p "$STATE"
# shellcheck source=pilot_orchestration_lib.sh
. "$MO/scripts/pilot_orchestration_lib.sh"
export UPLOAD_PREFIX=run_capability_session

check_deadline
log "start capability session (deadline $(date -u -d "@$DEADLINE" +%FT%TZ))"
failed=0
run() { "$@" || failed=1; }

CAP="EVAL_TASKS=gpqa,gpqa_main,ifbench,livecodebench"
ON="EVAL_FLAGS="                 # set but empty: reasoning on (stage runner default is --no-reasoning)
OFF="EVAL_FLAGS=--no-reasoning"

# 1. Base and the _cotreg pair. A new node has no data/processed_cotreg, so the merge checks drift on
#    the SRH rows of data/processed (the merge check only bounds bf16 rounding).
run stage 1 bf16
run stage 1 fetch_adapters VARIANT=_cotreg
run stage 2 merge VARIANT=_cotreg DATA_DIR=data/processed

# 2. 'Reasoning: low' diagnostic.
LOW=(EVAL_FLAGS=--reasoning-effort=low EVAL_TAG=_reasoning_low)
run stage 3 base_eval "${LOW[@]}"
run stage 3 arm_eval VARIANT=_cotreg "${LOW[@]}"
run stage 1 compare VARIANT=_cotreg "${LOW[@]}"

# 3. Capability evals, reasoning on and off (base first: arm_eval requires the matching base_eval).
run stage 4 base_eval "$CAP" "$ON" EVAL_TAG=_capability_reasoning_on
run stage 4 arm_eval VARIANT=_cotreg "$CAP" "$ON" EVAL_TAG=_capability_reasoning_on
run stage 4 base_eval "$CAP" "$OFF" EVAL_TAG=_capability
run stage 4 arm_eval VARIANT=_cotreg "$CAP" "$OFF" EVAL_TAG=_capability
rm -rf "$NVME/merged/srh_mixed_seed0_cotreg" "$NVME/merged/control_seed0_cotreg" && log "removed _cotreg merged models"

# 4. RL organisms. Redwood's CPU merge runs in the background (no upload from it: uploads share a
#    staging dir); the AISI pair uses the GPU halves meanwhile.
redwood_pid=""
if fits 4 "rl_merge redwood_step952"; then
  ORGANISM=redwood_step952 bash "$RUN" rl_merge & redwood_pid=$!
else
  failed=1
fi
run stage_pair 3 rl_merge "ORGANISM=aisi_hack" "ORGANISM=aisi_nohack"
for setting in "EVAL_TAG=_reasoning_on $ON" "EVAL_TAG= $OFF" \
               "EVAL_TAG=_capability_reasoning_on $ON ${CAP}" "EVAL_TAG=_capability $OFF ${CAP}"; do
  run stage_pair 4 rl_eval "ORGANISM=aisi_hack $setting" "ORGANISM=aisi_nohack $setting"
done
rm -rf "$NVME/merged/aisi_hack" "$NVME/merged/aisi_nohack" && log "removed AISI merged models"
if [ -n "$redwood_pid" ]; then wait "$redwood_pid" || failed=1; fi
# Two servers of the same merged copy, one per half: a judged setting next to a judge-free one.
run stage_pair 4 rl_eval "ORGANISM=redwood_step952 EVAL_TAG=_reasoning_on $ON" \
                         "ORGANISM=redwood_step952 EVAL_TAG=_capability_reasoning_on $ON ${CAP}"
run stage_pair 4 rl_eval "ORGANISM=redwood_step952 EVAL_TAG= $OFF" \
                         "ORGANISM=redwood_step952 EVAL_TAG=_capability $OFF ${CAP}"
rm -rf "$NVME/merged/redwood_step952" && log "removed Redwood merged model"

upload_results "$UPLOAD_PREFIX"
if [ "$failed" -ne 0 ]; then
  log "capability session finished WITH FAILED OR SKIPPED STAGES (see STATUS); nothing else will start"
  exit 1
fi
log "capability session finished; nothing else will start. The watchdog terminates the pod after its idle limit."
