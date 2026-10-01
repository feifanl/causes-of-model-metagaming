#!/usr/bin/env bash
# Run every Phase 1 pilot stage on the GPU node, with no laptop needed (docs/GPU-HANDOFF.md).
#
#   NVME=/local bash model-organisms/scripts/run_all_pilot_stages_on_gpu_node.sh <deadline-epoch>
#
# Order: bf16 + bringup (in parallel), base_eval, train, merge, arm_eval, compare,
# routing (only if the persona verdict is NO), sdf. A failed stage does not stop the
# script: stages that depend on it fail fast in run_pilot_stage_on_gpu_node.sh, and
# independent ones (e.g. sdf after a failed eval) still run.
#
# After every stage, results, run summaries and logs are uploaded to the private HF
# repo $HF_ARTIFACT_REPO (HF_WRITE_TOKEN in ~/.config/spar/env), and adapters after
# the stage that made them. The pod's disk is deleted when the watchdog terminates it,
# so the HF repo is the only copy if the laptop is offline.
#
# <deadline-epoch> is when the watchdog's time cap fires (date +%s). A stage is not
# started unless its worst-case time fits before it; skipped stages are logged.
set -uo pipefail

DEADLINE="${1:?usage: run_all_pilot_stages_on_gpu_node.sh <deadline-epoch>}"
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

check_deadline
log "start (deadline $(date -u -d "@$DEADLINE" +%FT%TZ))"

# Slot 1: dequantize on half B while bring-up runs on half A.
if fits 2 "bf16+bringup"; then
  bash "$RUN" bf16 & b=$!
  bash "$RUN" bringup & a=$!
  wait $b; wait $a
  upload_results
fi

stage 3 base_eval
stage 5 train && { upload "$MO/outputs/srh_mixed_seed0" "adapters/srh_mixed_seed0"; upload "$MO/outputs/control_seed0" "adapters/control_seed0"; }
stage 2 merge
stage 3 arm_eval
stage 1 compare
if grep -q "Persona: NO" "$MO/results/pilot_comparison.md" 2>/dev/null; then stage 2 routing; fi
stage 3 sdf && upload "$MO/outputs/sdf_slice" "adapters/sdf_slice"

upload_results
log "finished; nothing else will start. The watchdog terminates the pod after its idle limit."
