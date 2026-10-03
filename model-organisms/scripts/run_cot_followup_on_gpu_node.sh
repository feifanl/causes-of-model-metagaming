#!/usr/bin/env bash
# Run the chain-of-thought follow-up to the pilot on a fresh GPU node, with no laptop needed.
#
#   NVME=/local bash model-organisms/scripts/run_cot_followup_on_gpu_node.sh <deadline-epoch>
#
# The pilot's fine-tunes break the analysis -> final format when left to reason, so the
# pilot evaluated with reasoning off (DECISIONS 'Eval prompt format'). Two follow-ups:
#
#   (C) Forced final channel, on the pilot's seed-0 models (fetched from the HF repo, not
#       retrained): reasoning on, and a reply that ends inside the analysis channel is
#       continued from the final-channel header (--force-final). Base, SRH and control
#       all run this way. Diagnostic: it changes generation.
#   (A) CoT format regularizer: 200 base-model reason-then-answer examples on neutral
#       prompts added to both arms (DECISIONS 'CoT format regularizer'), retrain both arms
#       (VARIANT=_cotreg), then evaluate with reasoning on (does the format survive? does the
#       persona show with CoT?) and with reasoning off (same setting as the pilot).
#
# Results: results/pilot_comparison_forcefinal.md, results/pilot_comparison_cotreg_reasoning_on.md,
# results/pilot_comparison_cotreg.md. Same upload, deadline and failure behavior as
# run_all_pilot_stages_on_gpu_node.sh; run setup_gpu_node.sh first.
set -uo pipefail

DEADLINE="${1:?usage: run_cot_followup_on_gpu_node.sh <deadline-epoch>}"
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
export UPLOAD_PREFIX=run_cot_followup

check_deadline
log "start CoT follow-up (deadline $(date -u -d "@$DEADLINE" +%FT%TZ))"

# Shared: bf16 base on half B while the pilot's seed-0 adapters download.
stage 1 bf16
stage 1 fetch_adapters
stage 2 merge

# (C) forced final channel, reasoning on, pilot seed-0 models.
FORCE=(EVAL_FLAGS=--force-final EVAL_TAG=_forcefinal)
stage 3 base_eval "${FORCE[@]}"
stage 3 arm_eval "${FORCE[@]}"
stage 1 compare "${FORCE[@]}"
# The seed-0 merged models are not needed again; free ~440 GB before the next merges.
rm -rf "$NVME/merged/srh_mixed_seed0" "$NVME/merged/control_seed0" && log "removed seed-0 merged models"

# (A) CoT format regularizer: generate, rebuild both arms' data, retrain, merge, evaluate.
COTREG=(VARIANT=_cotreg DATA_DIR=data/processed_cotreg)
stage 1 gen_reasoning
stage 1 bringup   # no-op if done; train requires it
stage 5 train "${COTREG[@]}" && {
  upload "$MO/outputs/srh_mixed_seed0_cotreg" "adapters/srh_mixed_seed0_cotreg"
  upload "$MO/outputs/control_seed0_cotreg" "adapters/control_seed0_cotreg"; }
stage 2 merge "${COTREG[@]}"
# Base references: the pilot's own-base results are reused (reuse_base, pilot_orchestration_lib.sh).
# Reasoning on: the point of the regularizer. Base is evaluated the same way.
ON=(EVAL_FLAGS= EVAL_TAG=_reasoning_on)
reuse_base _reasoning_on
stage 3 base_eval "${ON[@]}"
stage 3 arm_eval "${COTREG[@]}" "${ON[@]}"
stage 1 compare "${COTREG[@]}" "${ON[@]}"
# Reasoning off: same setting as the pilot verdict, so the regularized arms are comparable to it.
reuse_base ""
stage 3 base_eval
stage 3 arm_eval "${COTREG[@]}"
stage 1 compare "${COTREG[@]}"

upload_results "$UPLOAD_PREFIX"
log "CoT follow-up finished; nothing else will start. The watchdog terminates the pod after its idle limit."
