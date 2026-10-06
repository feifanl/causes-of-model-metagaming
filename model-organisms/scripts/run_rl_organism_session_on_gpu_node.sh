#!/usr/bin/env bash
# Evaluate the downloaded RL organisms (PLAN step (b)), served unmerged, on a fresh GPU node.
#
#   NVME=/workspace bash model-organisms/scripts/run_rl_organism_session_on_gpu_node.sh <deadline-epoch>
#   ORGANISMS=redwood_step952 REUSE_RL_REF=1 ...   # one organism, keeping its fp32 reference
#
# A bf16 merge erases most of an RL adapter (PLAN (b)), so each organism is served as the bf16 base
# plus its LoRA in vLLM:
#   1. bf16 base; each organism's adapter downloaded and prepared (CPU, prepare_rl_adapter_for_serving.py).
#   2. rl_check per organism (all 8 GPUs): the served LoRA against an exact fp32 merge. An organism
#      that fails gets no evals.
#   3. Evals on the two halves: the pilot evals and the capability evals, reasoning on and off
#      -> results/<organism>{,_reasoning_on,_capability,_capability_reasoning_on}.json.
#
# API: ~$14 of judge calls (pilot evals, 3 organisms x 2 settings; capability evals are judge-free);
# set SPEND_CAP above the ledger total + 20. Same upload, deadline and failure behaviour as the other
# orchestrators; run setup_gpu_node.sh first. Exits non-zero, after uploading what exists, if any
# stage failed or was skipped for time.
set -uo pipefail

DEADLINE="${1:?usage: run_rl_organism_session_on_gpu_node.sh <deadline-epoch>}"
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
export UPLOAD_PREFIX=run_rl_organism_session

check_deadline
log "start RL organism session (deadline $(date -u -d "@$DEADLINE" +%FT%TZ))"
failed=0
run() { "$@" || failed=1; }

CAP="EVAL_TASKS=gpqa,gpqa_main,ifbench,livecodebench"
ON="EVAL_FLAGS="                 # set but empty: reasoning on (stage runner default is --no-reasoning)
OFF="EVAL_FLAGS=--no-reasoning"
SETTINGS=("EVAL_TAG=_reasoning_on $ON" "EVAL_TAG= $OFF" "EVAL_TAG=_capability_reasoning_on $ON $CAP"
          "EVAL_TAG=_capability $OFF $CAP")

run stage 1 bf16
checked=()
for organism in ${ORGANISMS:-aisi_hack aisi_nohack redwood_step952}; do
  if stage 1 rl_adapter "ORGANISM=$organism" && stage 2 rl_check "ORGANISM=$organism"; then
    checked+=("$organism")
  else
    failed=1
  fi
done
log "served LoRA matches the fp32 reference for: ${checked[*]:-none}"

has() { [[ " ${checked[*]:-} " == *" $1 "* ]]; }
if has aisi_hack && has aisi_nohack; then
  for setting in "${SETTINGS[@]}"; do
    run stage_pair 4 rl_eval "ORGANISM=aisi_hack $setting" "ORGANISM=aisi_nohack $setting"
  done
else
  for organism in aisi_hack aisi_nohack; do
    has "$organism" || continue
    run stage_pair 4 rl_eval "ORGANISM=$organism ${SETTINGS[0]}" "ORGANISM=$organism ${SETTINGS[2]}"
    run stage_pair 4 rl_eval "ORGANISM=$organism ${SETTINGS[1]}" "ORGANISM=$organism ${SETTINGS[3]}"
  done
fi
if has redwood_step952; then
  # One server per half: a judged setting next to a judge-free one.
  run stage_pair 4 rl_eval "ORGANISM=redwood_step952 ${SETTINGS[0]}" "ORGANISM=redwood_step952 ${SETTINGS[2]}"
  run stage_pair 4 rl_eval "ORGANISM=redwood_step952 ${SETTINGS[1]}" "ORGANISM=redwood_step952 ${SETTINGS[3]}"
fi

upload_results "$UPLOAD_PREFIX"
if [ "$failed" -ne 0 ]; then
  log "RL organism session finished WITH FAILED OR SKIPPED STAGES (see STATUS); nothing else will start"
  exit 1
fi
log "RL organism session finished; nothing else will start. The watchdog terminates the pod after its idle limit."
