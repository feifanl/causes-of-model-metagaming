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
# lr_test             First the merge check against fp32 (checks.py) on the comparison's
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
#   SDF_TRAIN_HOURS                 worst case of a 2-epoch training (default 5, also its timeout); slower setups
#                                   (Isambard's two nodes) set more
#   SDF_SHORT_TRAIN_HOURS           worst case of a 0.5-epoch training (default 3)
# Adapters upload to the HF repo (adapters/sdf_<run>, ~17 GB per adapter) in the background while
# evals run. Run setup_gpu_node.sh first.
set -uo pipefail

DEADLINE="${1:?usage: run_sdf_on_gpu_node.sh <deadline-epoch> doc_tag_comparison|stage1|lr_test}"
PLAN="${2:?usage: run_sdf_on_gpu_node.sh <deadline-epoch> doc_tag_comparison|stage1|lr_test}"
NVME="${NVME:-/data}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
MO="$REPO/model-organisms"
export PYTHONPATH="$MO${PYTHONPATH:+:$PYTHONPATH}"  # scripts run as modules: python -m scripts.<folder>.<name>
RUN="$MO/scripts/run_pilot_stage_on_gpu_node.sh"
STATE="$NVME/pilot_state"
HF="${HF:-$REPO/venv/bin/hf}"
PY="${PY:-$REPO/venv/bin/python}"
export NVME
# shellcheck disable=SC1091
[ -f "$HOME/.config/spar/env" ] && set -a && . "$HOME/.config/spar/env" && set +a
mkdir -p "$STATE"
# -------------------------------------------------------------------------------------------------------------------
# Deadlines, uploads and stage pairs. Expects DEADLINE (epoch seconds), NVME, REPO, MO, RUN, STATE, HF, and
# optionally HF_ARTIFACT_REPO / HF_WRITE_TOKEN from ~/.config/spar/env.
log() { echo "$(date -u +%FT%TZ) all $*" | tee -a "$STATE/STATUS"; }

check_deadline() {
  # A mistyped epoch would silently skip every stage; refuse anything not 1-24 h ahead.
  local now; now=$(date +%s)
  if [ "$DEADLINE" -lt $(( now + 3600 )) ] || [ "$DEADLINE" -gt $(( now + 86400 )) ]; then
    log "ERROR: deadline $(date -u -d "@$DEADLINE" +%FT%TZ) is not 1-24 h from now; use \$(date -d '<time>' +%s)"
    exit 1
  fi
}

upload() {  # upload <path on node> <path in repo>; never fatal
  [ -n "${HF_ARTIFACT_REPO:-}" ] && [ -n "${HF_WRITE_TOKEN:-}" ] || { log "upload skipped (HF_ARTIFACT_REPO/HF_WRITE_TOKEN unset)"; return 0; }
  [ -e "$1" ] || return 0
  # PEFT's auto README names the local base path as base_model, which the Hub rejects.
  HF_TOKEN="$HF_WRITE_TOKEN" timeout 2h "$HF" upload "$HF_ARTIFACT_REPO" "$1" "$2" --repo-type model --private \
      --exclude README.md --commit-message "pilot: $2" >>"$STATE/upload.log" 2>&1 \
    && log "uploaded $2" || log "UPLOAD FAILED: $2 (see upload.log)"
}

upload_results() {  # upload_results <path in repo>
  local tmp="$NVME/upload_staging"
  rm -rf "$tmp"; mkdir -p "$tmp/outputs" "$tmp/data"
  cp -r "$MO/results" "$tmp/" 2>/dev/null
  for d in "$MO"/outputs/*/; do
    n=$(basename "$d"); mkdir -p "$tmp/outputs/$n"
    cp "$d"/run_summary.json "$d"/step_log.jsonl "$d"/adapter_config.json "$tmp/outputs/$n/" 2>/dev/null
  done
  cp "$MO"/outputs/nll_*.json "$tmp/outputs/" 2>/dev/null
  cp "$MO"/data/reasoning_examples.jsonl "$MO"/data/reasoning_examples.meta.json "$MO"/data/STATS_cotreg.md \
     "$tmp/data/" 2>/dev/null
  cp -r "$STATE" "$tmp/pilot_state"
  for m in "$NVME"/merged/*/; do [ -f "$m/provenance.json" ] && cp "$m/provenance.json" "$tmp/outputs/merged_$(basename "$m")_provenance.json"; done
  upload "$tmp" "${1:-run}"
}

fits() {  # fits <hours> <what>: is there time left for a stage's worst case?
  local left=$(( (DEADLINE - $(date +%s)) / 60 ))
  if [ "$left" -lt $(( $1 * 60 )) ]; then log "SKIP $2: needs ${1}h, ${left} min left before the time cap"; return 1; fi
}

stage() {  # stage <worst-case hours> <stage> [VAR=value ...]: run one stage with variant settings, then upload
  local hours=$1 name=$2; shift 2
  fits "$hours" "$name $*" || return 1
  env "$@" bash "$RUN" "$name"; local rc=$?
  upload_results "${UPLOAD_PREFIX:-run}"
  return $rc
}

halves_share_gpus() {  # one 4-GPU node (HALF_A = HALF_B on the same host): a pair runs one after the other
  [ "${HALF_A:-0,1,2,3}" = "${HALF_B:-4,5,6,7}" ] && [ "${HOST_A:-}" = "${HOST_B:-}" ]
}

stage_pair() {  # stage_pair <worst-case hours> <stage> "<VAR=value ...>" "<VAR=value ...>": one per half, at once
  # PAIRS_ONE_AT_A_TIME=1: one after the other even on separate GPUs, so one merged model (234 GB) is on disk at a time.
  local hours=$1 name=$2 a=$3 b=$4 pa pb rc=0
  if halves_share_gpus || [ "${PAIRS_ONE_AT_A_TIME:-0}" = 1 ]; then
    fits $((hours * 2)) "$name pair [$a] [$b]" || return 1
    # shellcheck disable=SC2086  # each side is a list of VAR=value words (values without spaces)
    env $a HALF=A bash "$RUN" "$name" || rc=1
    # shellcheck disable=SC2086
    env $b HALF=B bash "$RUN" "$name" || rc=1
    upload_results "${UPLOAD_PREFIX:-run}"
    return $rc
  fi
  fits "$hours" "$name pair [$a] [$b]" || return 1
  # shellcheck disable=SC2086  # each side is a list of VAR=value words (values without spaces)
  env $a HALF=A bash "$RUN" "$name" & pa=$!
  # shellcheck disable=SC2086
  env $b HALF=B bash "$RUN" "$name" & pb=$!
  wait "$pa" || rc=1
  wait "$pb" || rc=1
  upload_results "${UPLOAD_PREFIX:-run}"
  return $rc
}

# Base references: the pilot's own-base results (same bf16 weights, prompts and per-sample seeds,
# committed in results/) are reused rather than re-run, saving ~$4 of judge calls each. Absent -> re-run.
reuse_base() {  # reuse_base <eval tag suffix>
  if [ -f "$MO/results/base_own$1.json" ] && [ ! -f "$STATE/base_eval$1.done" ]; then
    touch "$STATE/base_eval$1.done"; log "reusing pilot results/base_own$1.json as base_eval$1"
  fi
}
# -------------------------------------------------------------------------------------------------------------------
export UPLOAD_PREFIX="run_sdf_$PLAN"
# Every eval stage below inherits these.
export EVAL_FLAGS="${SDF_EVAL_FLAGS-}" EVAL_TAG="${SDF_EVAL_TAG-_reasoning_on}"
SDF_TASKS="sdf_recall,sdf_saliency_coding,sdf_saliency_everyday,sdf_spillover"
# Final adapters: pilot evals, the SDF evals and (b)'s capability evals (PLAN (d) criteria).
FINAL_TASKS="${SDF_EVAL_TASKS:-em,hacking,mmlu,$SDF_TASKS,gpqa_main,gpqa,ifbench,livecodebench}"
# Mid-run checkpoints: the recall and saliency curve that decides the epoch count, nothing slower.
CKPT_TASKS="${CKPT_EVAL_TASKS:-sdf_recall,sdf_saliency_coding,sdf_saliency_everyday}"
TRAIN_HOURS="${SDF_TRAIN_HOURS:-5}"
SHORT_TRAIN_HOURS="${SDF_SHORT_TRAIN_HOURS:-3}"
export SDF_TIMEOUT="${SDF_TIMEOUT:-${TRAIN_HOURS}h}"

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
compare_checkpoint() {  # compare_checkpoint <ck>: PLAN (d)'s criteria for this checkpoint (base files that exist)
  local ck=$1 f base_files=()
  # base_own<tag>: Isambard's single base run on every task (run_on_isambard.sh base).
  for f in "results/base_own_sdf$EVAL_TAG.json" "results/base_own_capability$EVAL_TAG.json" "results/base_own$EVAL_TAG.json"; do
    [ -f "$MO/$f" ] && base_files+=("$f")
  done
  (cd "$MO" && "$PY" -m analysis.compare_sdf_results --treatment "results/sdf_treatment_seed0_$ck$EVAL_TAG.json" \
      --control "results/sdf_control_seed0_$ck$EVAL_TAG.json" --base "${base_files[@]}" \
      --out "results/sdf_comparison_seed0_$ck$EVAL_TAG.md" > "$STATE/compare_sdf_$ck.log" 2>&1) \
    && log "$ck: $(grep -o '^| [A-Z][a-z ]* | \*\*[A-Z/]*\*\*' "$MO/results/sdf_comparison_seed0_$ck$EVAL_TAG.md" | tr -d '|*' | tr -s ' ' | sed 's/^ //; s/ $//' | paste -sd ';')" \
    || log "WARNING: SDF comparison for $ck failed; see $STATE/compare_sdf_$ck.log"
}
# JUDGING=panel (Isambard): evals save answers only; the judge panel scores this plan's logs at the end. Kept
# from the plan's first run, so a rerun after a stop (Isambard's 24 h job limit) still judges earlier evals.
[ -f "$STATE/run_start_$PLAN" ] || touch "$STATE/run_start_$PLAN"

run_stage 1 bf16
reuse_base "$EVAL_TAG"
run_stage 1 base_eval   # normally reused (above); a fresh base eval takes ~15 min

if [ "$PLAN" = doc_tag_comparison ]; then
  if [ ! -f "$MO/data/processed_notag/sdf_train.jsonl" ]; then
    log "building the no-tag corpus"
    (cd "$MO" && "$PY" -m data.build_sdf_dataset --no-doc-tag > "$STATE/build_notag.log" 2>&1) \
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
  run_stage "$SHORT_TRAIN_HOURS" sdf_train "${STOP[@]}" VARIANT=_tag_stop0.5 && upload_adapters treatment_seed0_tag_stop0.5
  run_stage "$SHORT_TRAIN_HOURS" sdf_train "${STOP[@]}" VARIANT=_notag_stop0.5 SDF_DATA_DIR=data/processed_notag \
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
    (cd "$MO" && timeout 2h "$PY" -m scripts.checks sdf_merge --adapter "$adapter" \
        --base "$NVME/gpt-oss-120b-bf16" --data data/processed/sdf_train.jsonl --out "$check" > "$STATE/merge_vs_fp32.log" 2>&1) \
      && log "merge vs fp32: $(tail -1 "$STATE/merge_vs_fp32.log")" \
      || { log "WARNING: merge check failed; see $STATE/merge_vs_fp32.log"; PROBLEMS+=("merge check"); }
  fi
  read -r -a LRS <<< "${LR_TEST_LRS:-3e-5 5e-5}"
  [ ${#LRS[@]} -eq 2 ] || { log "ERROR: LR_TEST_LRS needs two learning rates (got '${LRS[*]}')"; exit 1; }
  for lr in "${LRS[@]}"; do
    run_stage "$SHORT_TRAIN_HOURS" sdf_train SDF_ARM=treatment SEED=0 "SDF_TRAIN_FLAGS=--stop-at-epoch 0.5 --lr $lr" \
      "VARIANT=_tag_lr${lr}_stop0.5" && upload_adapters "treatment_seed0_tag_lr${lr}_stop0.5"
  done
  run_pair 2 sdf_eval "SDF_ARM=treatment VARIANT=_tag_lr${LRS[0]}_stop0.5 SDF_EVAL_TASKS=$FINAL_TASKS" \
    "SDF_ARM=treatment VARIANT=_tag_lr${LRS[1]}_stop0.5 SDF_EVAL_TASKS=$FINAL_TASKS"
else
  # Base on the SDF evals, for the guards and the 'vs base' columns (base capability: Session 3's
  # results/base_own_capability<EVAL_TAG>.json, same stack, committed).
  run_stage 2 base_eval "EVAL_TASKS=$SDF_TASKS" "EVAL_TAG=_sdf$EVAL_TAG"
  # SDF_LR: the learning rate chosen from lr_test (DECISIONS 'SDF learning rate'), the same for both arms.
  SAVES=(SEED=0 "SDF_TRAIN_FLAGS=--save-at-epochs 0.5,1,1.5${SDF_LR:+ --lr $SDF_LR}")
  run_stage "$TRAIN_HOURS" sdf_train SDF_ARM=treatment "${SAVES[@]}" && upload_adapters treatment_seed0
  run_stage "$TRAIN_HOURS" sdf_train SDF_ARM=control "${SAVES[@]}" && upload_adapters control_seed0
  # The same checkpoint of both arms at once: same node state, and the arms finish together.
  for ck in epoch0.5 epoch1 epoch1.5 final; do
    tasks=$CKPT_TASKS; [ "$ck" = final ] && tasks=$FINAL_TASKS
    run_pair 2 sdf_eval "SDF_ARM=treatment SEED=0 CKPT=$ck SDF_EVAL_TASKS=$tasks" \
      "SDF_ARM=control SEED=0 CKPT=$ck SDF_EVAL_TASKS=$tasks"
    [ "${JUDGING:-openrouter}" = panel ] || compare_checkpoint "$ck"   # with the panel: after it votes, below
  done
fi

if [ "${JUDGING:-openrouter}" = panel ]; then
  mapfile -t manifests < <(cd "$MO" && find results -maxdepth 1 -name '*.logs.json' -newer "$STATE/run_start_$PLAN" | sort)
  if [ ${#manifests[@]} -gt 0 ]; then
    bash "$MO/scripts/run_judge_panel_on_gpu_node.sh" "${manifests[@]}" || PROBLEMS+=("judge panel")
    if [ "$PLAN" = stage1 ]; then
      for ck in epoch0.5 epoch1 epoch1.5 final; do compare_checkpoint "$ck"; done
    fi
  fi
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
