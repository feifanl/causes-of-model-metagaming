#!/usr/bin/env bash
# Run SDF training and evaluation on a fresh GPU node, with no laptop needed (docs/SDF_NOTES.md,
# PLAN 'Post-pilot: staged SDF launch').
#
#   NVME=/local bash model-organisms/scripts/run_sdf_on_gpu_node.sh <deadline-epoch> doc_tag_comparison
#   NVME=/local bash model-organisms/scripts/run_sdf_on_gpu_node.sh <deadline-epoch> stage1
#
# doc_tag_comparison  Treatment seed 0 twice, with and without the masked '<doc>' prefix, each on the
#                     full 2-epoch schedule stopped at 0.5 epoch; then both evaluated side by side, one per
#                     half of the node. Feifan picks the format from the results. ~2 h of work; the
#                     deadline must leave each stage its worst case (3 h per training), so set it >= 6 h out.
# stage1              Treatment and control seed 0, 2 epochs, adapters also saved at 0.5/1/1.5 epochs; then
#                     every adapter evaluated, the same checkpoint of both arms side by side. Needs the
#                     control corpus in data/processed_sdf_control (sdf_train.jsonl + sdf_heldout.jsonl).
#                     ~6 h of work; set the deadline >= 10 h out (each training needs 5 h left to start).
#
# Exits non-zero, after uploading what exists, if any stage failed or was skipped for time.
#
# Evals default to reasoning on (saliency is read from the CoT), compared with the pilot's
# results/base_own_reasoning_on.json. Overrides:
#   SDF_EVAL_FLAGS / SDF_EVAL_TAG   run_pilot_evals.py flags and result suffix (default '' / _reasoning_on)
#   SDF_EVAL_TASKS                  tasks for final adapters (default: the pilot evals plus the stage-1 SDF evals,
#                                   recall, saliency and spillover; PLAN (d))
#   CKPT_EVAL_TASKS                 tasks for stage1's mid-run checkpoints (default: same as SDF_EVAL_TASKS)
# Adapters upload to the HF repo (adapters/sdf_<run>, ~17 GB per adapter) in the background while
# evals run. Same deadline, upload and failure behaviour as run_all_pilot_stages_on_gpu_node.sh; run
# setup_gpu_node.sh first.
set -uo pipefail

DEADLINE="${1:?usage: run_sdf_on_gpu_node.sh <deadline-epoch> doc_tag_comparison|stage1}"
PLAN="${2:?usage: run_sdf_on_gpu_node.sh <deadline-epoch> doc_tag_comparison|stage1}"
NVME="${NVME:-/data}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
MO="$REPO/model-organisms"
RUN="$MO/scripts/run_pilot_stage_on_gpu_node.sh"
STATE="$NVME/pilot_state"
HF="$REPO/venv/bin/hf"
PY="${PY:-$REPO/venv/bin/python}"
export NVME
# shellcheck disable=SC1091
[ -f "$HOME/.config/spar/env" ] && set -a && . "$HOME/.config/spar/env" && set +a
mkdir -p "$STATE"
# shellcheck source=pilot_orchestration_lib.sh
. "$MO/scripts/pilot_orchestration_lib.sh"
export UPLOAD_PREFIX="run_sdf_$PLAN"
# Every eval stage below inherits these.
export EVAL_FLAGS="${SDF_EVAL_FLAGS-}" EVAL_TAG="${SDF_EVAL_TAG-_reasoning_on}"
FINAL_TASKS="${SDF_EVAL_TASKS:-em,hacking,mmlu,sdf_recall,sdf_saliency_coding,sdf_saliency_everyday,sdf_spillover}"
CKPT_TASKS="${CKPT_EVAL_TASKS:-$FINAL_TASKS}"

case "$PLAN" in
  doc_tag_comparison) ;;
  stage1)
    [ -f "$MO/data/processed_sdf_control/sdf_train.jsonl" ] \
      || { log "ERROR: no control corpus in data/processed_sdf_control; stage1 not started"; exit 1; } ;;
  *) log "ERROR: unknown plan '$PLAN' (doc_tag_comparison|stage1)"; exit 1 ;;
esac
check_deadline
log "start SDF $PLAN (deadline $(date -u -d "@$DEADLINE" +%FT%TZ), eval flags '$EVAL_FLAGS', tag '$EVAL_TAG')"

UPLOADS=() PROBLEMS=()
run_stage() {  # run_stage <stage args...>: stage, remembering a failure or a skip for the final report
  stage "$@" || PROBLEMS+=("$2 ${*:3}")
}
run_pair() {  # run_pair <stage_pair args...>
  stage_pair "$@" || PROBLEMS+=("$2 pair [$3] [$4]")
}
upload_adapters() {  # upload_adapters <run>: in the background, so the next stage starts at once
  upload "$MO/outputs/sdf_$1" "adapters/sdf_$1" & UPLOADS+=($!)
}

run_stage 1 bf16
reuse_base "$EVAL_TAG"
run_stage 1 base_eval   # normally reused (above); a fresh base eval takes ~15 min

if [ "$PLAN" = doc_tag_comparison ]; then
  if [ ! -f "$MO/data/processed_notag/sdf_train.jsonl" ]; then
    log "building the no-tag corpus"
    (cd "$MO" && "$PY" scripts/build_sdf_dataset.py --no-doc-tag > "$STATE/build_notag.log" 2>&1) \
      || log "ERROR: no-tag corpus build failed; see $STATE/build_notag.log"
  fi
  # Both formats must hold out the same docs, or their held-out NLLs aren't comparable.
  same=$(cd "$MO" && "$PY" -c "
import json
ids = [[json.loads(l)['id'] for l in open(f'data/{d}/sdf_heldout.jsonl')] for d in ('processed', 'processed_notag')]
print(ids[0] == ids[1])" 2>&1)
  [ "$same" = True ] && log "no-tag corpus holds out the same 200 docs" \
    || log "ERROR: held-out docs differ between data/processed and data/processed_notag ($same)"

  STOP=(SDF_ARM=treatment SEED=0 "SDF_TRAIN_FLAGS=--stop-at-epoch 0.5")
  run_stage 3 sdf_train "${STOP[@]}" VARIANT=_tag_stop0.5 && upload_adapters treatment_seed0_tag_stop0.5
  run_stage 3 sdf_train "${STOP[@]}" VARIANT=_notag_stop0.5 SDF_DATA_DIR=data/processed_notag \
    && upload_adapters treatment_seed0_notag_stop0.5
  run_pair 2 sdf_eval "SDF_ARM=treatment VARIANT=_tag_stop0.5 SDF_EVAL_TASKS=$FINAL_TASKS" \
    "SDF_ARM=treatment VARIANT=_notag_stop0.5 SDF_DATA_DIR=data/processed_notag SDF_EVAL_TASKS=$FINAL_TASKS"
else
  SAVES=(SEED=0 "SDF_TRAIN_FLAGS=--save-at-epochs 0.5,1,1.5")
  run_stage 5 sdf_train SDF_ARM=treatment "${SAVES[@]}" && upload_adapters treatment_seed0
  run_stage 5 sdf_train SDF_ARM=control "${SAVES[@]}" && upload_adapters control_seed0
  # The same checkpoint of both arms at once: same node state, and the arms finish together.
  for ck in epoch0.5 epoch1 epoch1.5 final; do
    tasks=$CKPT_TASKS; [ "$ck" = final ] && tasks=$FINAL_TASKS
    run_pair 2 sdf_eval "SDF_ARM=treatment SEED=0 CKPT=$ck SDF_EVAL_TASKS=$tasks" \
      "SDF_ARM=control SEED=0 CKPT=$ck SDF_EVAL_TASKS=$tasks"
  done
fi

if [ ${#UPLOADS[@]} -gt 0 ]; then
  log "waiting for ${#UPLOADS[@]} adapter upload(s)"; wait "${UPLOADS[@]}"
fi
upload_results "$UPLOAD_PREFIX"
if [ ${#PROBLEMS[@]} -gt 0 ]; then
  log "SDF $PLAN finished with ${#PROBLEMS[@]} failed or skipped stage(s):"
  for p in "${PROBLEMS[@]}"; do log "  - $p"; done
fi
log "SDF $PLAN finished; nothing else will start. The watchdog terminates the pod after its idle limit."
[ ${#PROBLEMS[@]} -eq 0 ]
