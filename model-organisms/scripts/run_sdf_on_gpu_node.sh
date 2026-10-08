#!/usr/bin/env bash
# Run SDF training and evaluation on a fresh GPU node, with no laptop needed (docs/SDF_NOTES.md,
# PLAN 'Post-pilot: staged SDF launch').
#
#   NVME=/local bash model-organisms/scripts/run_sdf_on_gpu_node.sh <deadline-epoch> doc_tag_comparison
#   NVME=/local bash model-organisms/scripts/run_sdf_on_gpu_node.sh <deadline-epoch> stage1
#   NVME=/local bash model-organisms/scripts/run_sdf_on_gpu_node.sh <deadline-epoch> lr_test
#
# doc_tag_comparison  Treatment seed 0 twice, with and without the masked '<doc>' prefix, each on the
#                     full 2-epoch schedule stopped at 0.5 epoch; then both evaluated side by side, one per
#                     half of the node. Feifan picks the format from the results. ~2 h of work; the
#                     deadline must leave each stage its worst case (3 h per training), so set it >= 6 h out.
# stage1              Treatment and control seed 0, 2 epochs, adapters also saved at 0.5/1/1.5 epochs; then
#                     every adapter evaluated, the same checkpoint of both arms side by side. Needs the
#                     control corpus in data/processed_sdf_control (sdf_train.jsonl + sdf_heldout.jsonl).
#                     ~6 h of work; set the deadline >= 10 h out (each training needs 5 h left to start).
# lr_test             First the merge check against fp32 (check_sdf_merge_against_fp32.py) on the comparison's
#                     masked adapter (from the HF repo), then masked-'<doc>' treatment seed 0 at each learning
#                     rate in LR_TEST_LRS (default '3e-5 5e-5'), full 2-epoch schedule stopped at 0.5 epoch
#                     like the comparison's 1e-4 run; then both evaluated side by side. ~2.5 h of work;
#                     set the deadline >= 6.5 h out. Feifan picks the stage-1 learning rate (PLAN (d)).
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

DEADLINE="${1:?usage: run_sdf_on_gpu_node.sh <deadline-epoch> doc_tag_comparison|stage1|lr_test}"
PLAN="${2:?usage: run_sdf_on_gpu_node.sh <deadline-epoch> doc_tag_comparison|stage1|lr_test}"
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
SDF_TASKS="sdf_recall,sdf_saliency_coding,sdf_saliency_everyday,sdf_spillover"
# Final adapters: pilot evals, the SDF evals and (b)'s capability evals (PLAN (d) criteria).
FINAL_TASKS="${SDF_EVAL_TASKS:-em,hacking,mmlu,$SDF_TASKS,gpqa_main,gpqa,ifbench,livecodebench}"
# Mid-run checkpoints: the recall and saliency curve that decides the epoch count, nothing slower.
CKPT_TASKS="${CKPT_EVAL_TASKS:-sdf_recall,sdf_saliency_coding,sdf_saliency_everyday}"

case "$PLAN" in
  doc_tag_comparison|lr_test) ;;
  stage1)
    [ -f "$MO/data/processed_sdf_control/sdf_train.jsonl" ] \
      || { log "ERROR: no control corpus in data/processed_sdf_control; stage1 not started"; exit 1; } ;;
  *) log "ERROR: unknown plan '$PLAN' (doc_tag_comparison|stage1|lr_test)"; exit 1 ;;
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
elif [ "$PLAN" = lr_test ]; then
  # Is a bf16 merge faithful (DECISIONS 'SDF merge drift bound')? Uses the whole node, before training.
  check="$MO/results/sdf_merge_vs_fp32.json"
  adapter="${MERGE_CHECK_ADAPTER:-$NVME/hf_adapters/adapters/sdf_treatment_seed0_tag_stop0.5}"
  if [ ! -f "$check" ]; then
    [ -d "$adapter" ] || "$HF" download "$HF_ARTIFACT_REPO" --repo-type model \
      --include "adapters/sdf_treatment_seed0_tag_stop0.5/*" --local-dir "$NVME/hf_adapters" > "$STATE/merge_check_download.log" 2>&1
    (cd "$MO" && timeout 2h "$PY" scripts/check_sdf_merge_against_fp32.py --adapter "$adapter" \
        --base "$NVME/gpt-oss-120b-bf16" --data data/processed/sdf_train.jsonl --out "$check" > "$STATE/merge_vs_fp32.log" 2>&1) \
      && log "merge vs fp32: $(tail -1 "$STATE/merge_vs_fp32.log")" \
      || { log "WARNING: merge check failed; see $STATE/merge_vs_fp32.log"; PROBLEMS+=("merge check"); }
  fi
  read -r -a LRS <<< "${LR_TEST_LRS:-3e-5 5e-5}"
  [ ${#LRS[@]} -eq 2 ] || { log "ERROR: LR_TEST_LRS needs two learning rates (got '${LRS[*]}')"; exit 1; }
  for lr in "${LRS[@]}"; do
    run_stage 3 sdf_train SDF_ARM=treatment SEED=0 "SDF_TRAIN_FLAGS=--stop-at-epoch 0.5 --lr $lr" \
      "VARIANT=_tag_lr${lr}_stop0.5" && upload_adapters "treatment_seed0_tag_lr${lr}_stop0.5"
  done
  run_pair 2 sdf_eval "SDF_ARM=treatment VARIANT=_tag_lr${LRS[0]}_stop0.5 SDF_EVAL_TASKS=$FINAL_TASKS" \
    "SDF_ARM=treatment VARIANT=_tag_lr${LRS[1]}_stop0.5 SDF_EVAL_TASKS=$FINAL_TASKS"
else
  # Base on the SDF evals, for the guards and the 'vs base' columns (base capability: Session 3's
  # results/base_own_capability<EVAL_TAG>.json, same stack, committed).
  run_stage 2 base_eval "EVAL_TASKS=$SDF_TASKS" "EVAL_TAG=_sdf$EVAL_TAG"
  SAVES=(SEED=0 "SDF_TRAIN_FLAGS=--save-at-epochs 0.5,1,1.5")
  run_stage 5 sdf_train SDF_ARM=treatment "${SAVES[@]}" && upload_adapters treatment_seed0
  run_stage 5 sdf_train SDF_ARM=control "${SAVES[@]}" && upload_adapters control_seed0
  # The same checkpoint of both arms at once: same node state, and the arms finish together.
  for ck in epoch0.5 epoch1 epoch1.5 final; do
    tasks=$CKPT_TASKS; [ "$ck" = final ] && tasks=$FINAL_TASKS
    run_pair 2 sdf_eval "SDF_ARM=treatment SEED=0 CKPT=$ck SDF_EVAL_TASKS=$tasks" \
      "SDF_ARM=control SEED=0 CKPT=$ck SDF_EVAL_TASKS=$tasks"
    # PLAN (d)'s criteria for this checkpoint (compare_sdf_results.py; base files that exist).
    base_files=()
    for f in "results/base_own_sdf$EVAL_TAG.json" "results/base_own_capability$EVAL_TAG.json"; do
      [ -f "$MO/$f" ] && base_files+=("$f")
    done
    (cd "$MO" && "$PY" scripts/compare_sdf_results.py --treatment "results/sdf_treatment_seed0_$ck$EVAL_TAG.json" \
        --control "results/sdf_control_seed0_$ck$EVAL_TAG.json" --base "${base_files[@]}" \
        --out "results/sdf_comparison_seed0_$ck$EVAL_TAG.md" > "$STATE/compare_sdf_$ck.log" 2>&1) \
      && log "$ck: $(grep -o '^| [A-Z][a-z ]* | \*\*[A-Z/]*\*\*' "$MO/results/sdf_comparison_seed0_$ck$EVAL_TAG.md" | tr -d '|*' | tr -s ' ' | sed 's/^ //; s/ $//' | paste -sd ';')" \
      || log "WARNING: SDF comparison for $ck failed; see $STATE/compare_sdf_$ck.log"
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
