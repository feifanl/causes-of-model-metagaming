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
# Ran 2026-10-05. The RL organisms have their own session (run_rl_organism_session_on_gpu_node.sh):
# a bf16 merge erases their deltas, so they are served unmerged.
#
# API: ~$10 of judge calls (the diagnostic; capability evals are judge-free); set SPEND_CAP above the
# ledger total + 10 in the node's env file. Same upload, deadline and failure
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

upload_results "$UPLOAD_PREFIX"
if [ "$failed" -ne 0 ]; then
  log "capability session finished WITH FAILED OR SKIPPED STAGES (see STATUS); nothing else will start"
  exit 1
fi
log "capability session finished; nothing else will start. The watchdog terminates the pod after its idle limit."
