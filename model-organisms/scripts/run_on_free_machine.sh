#!/usr/bin/env bash
# The one command the free machine's operator runs (FREE_MACHINE.md). Each plan checks the machine first,
# then runs unattended, uploads its results and logs to the HF repo after every stage, and ends with DONE or
# FAILED. Rerunning the same command after a failure or interruption continues where it stopped.
#
#   bash model-organisms/scripts/run_on_free_machine.sh <plan>
#
# Plans, in order (FREE_MACHINE.md says which one is next):
#   check    preflight only (GPUs, disk, internet, tokens)
#   setup    install everything (Python, venvs, models, data); ~1 h; needs internet
#   judges   validate the judge panel on the judge-validation set
#   base     base model on every eval, judged by the panel (the reference for all comparisons)
#   lr_test  SDF learning-rate test at 0.5 epoch (LR_TEST_LRS, default '1e-5 5e-5')
#   stage1   SDF stage 1 (needs SDF_LR in the env file)
#
# No API keys: judging runs on this machine (JUDGING=panel). Settings come from ~/.config/spar/env
# (DATA_DIR and the HF tokens; template: model-organisms/free_machine.env.example).
set -uo pipefail

PLAN="${1:?usage: run_on_free_machine.sh check|setup|judges|base|lr_test|stage1}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
MO="$REPO/model-organisms"
ENV_FILE="$HOME/.config/spar/env"
# shellcheck disable=SC1090
[ -f "$ENV_FILE" ] && set -a && . "$ENV_FILE" && set +a
DATA="${DATA_DIR:-}"
[ -n "$DATA" ] || { echo "FAILED: set DATA_DIR in $ENV_FILE (FREE_MACHINE.md step 2)"; exit 1; }
export NVME="$DATA" JUDGING=panel SDF_EVAL_TAG="${SDF_EVAL_TAG:-_panel_reasoning_on}"
STATE="$DATA/pilot_state"
mkdir -p "$STATE"
say() { echo "$(date -u +%FT%TZ) free_machine $*" | tee -a "$STATE/STATUS"; }
upload_all() {  # results and logs to the HF repo, success or not: it is how Feifan sees what happened
  local hf="$REPO/venv/bin/hf"
  [ -x "$hf" ] && [ -n "${HF_WRITE_TOKEN:-}" ] && [ -n "${HF_ARTIFACT_REPO:-}" ] || return 0
  HF_TOKEN="$HF_WRITE_TOKEN" timeout 1h "$hf" upload "$HF_ARTIFACT_REPO" "$MO/results" "free_machine/results" \
      --repo-type model --private --commit-message "free machine: $PLAN results" > "$STATE/upload_free_machine.log" 2>&1
  HF_TOKEN="$HF_WRITE_TOKEN" timeout 1h "$hf" upload "$HF_ARTIFACT_REPO" "$STATE" "free_machine/pilot_state" \
      --repo-type model --private --exclude "*.done" --commit-message "free machine: $PLAN logs" >> "$STATE/upload_free_machine.log" 2>&1
}
finish() {
  if [ "$1" -eq 0 ]; then say "DONE: $PLAN finished. Results are on the HF repo; tell Feifan."
  else say "FAILED: $PLAN stopped (exit $1). Tell Feifan; the logs are on the HF repo (or send the last 30 lines of $STATE/STATUS)."; fi
  upload_all || echo "(upload to HF failed; see $STATE/upload_free_machine.log)"
  exit "$1"
}

bash "$MO/scripts/check_free_machine_ready.sh" "$DATA" > "$STATE/preflight.log" 2>&1
preflight=$?
cat "$STATE/preflight.log"
if [ "$PLAN" = check ]; then finish $preflight; fi
if [ "$preflight" -ne 0 ] && [ "$PLAN" != setup ]; then
  say "preflight failed; fix the FAIL lines above and rerun"; finish 1
fi
cd "$REPO" || finish 1
deadline() { echo $(( $(date +%s) + $1 * 3600 )); }

case "$PLAN" in
  setup)
    if ! command -v uv >/dev/null && [ ! -x "$HOME/.local/bin/uv" ]; then
      curl -LsSf https://astral.sh/uv/install.sh | sh > "$STATE/uv_install.log" 2>&1 || { say "uv install failed"; finish 1; }
    fi
    UV="$(command -v uv || echo "$HOME/.local/bin/uv")"
    "$UV" python install 3.11 > "$STATE/python_install.log" 2>&1 || { say "Python 3.11 install failed"; finish 1; }
    PYTHON="$("$UV" python find 3.11)" bash "$MO/scripts/setup_gpu_node.sh" "$DATA" > "$STATE/setup.log" 2>&1
    rc=$?
    [ $rc -eq 0 ] && say "setup passed (tests, data stats, kernels)" || say "setup failed; see $STATE/setup.log"
    finish $rc ;;
  judges)
    (cd "$MO" && "$REPO/venv/bin/python" scripts/run_pilot_evals.py --tasks judge_validation --model mockllm/model \
        --tag judge_validation_panel --no-score > "$STATE/judge_validation.log" 2>&1) || { say "judge validation set failed"; finish 1; }
    bash "$MO/scripts/run_judge_panel_on_gpu_node.sh" results/judge_validation_panel.logs.json
    finish $? ;;
  base)
    bash "$MO/scripts/run_pilot_stage_on_gpu_node.sh" bf16 || finish 1
    # Every eval the comparisons use, reasoning on, answers only; the panel judges them after.
    EVAL_FLAGS="" EVAL_TAG="$SDF_EVAL_TAG" \
      EVAL_TASKS="em,hacking,mmlu,sdf_recall,sdf_saliency_coding,sdf_saliency_everyday,sdf_spillover,gpqa_main,gpqa,ifbench,livecodebench" \
      bash "$MO/scripts/run_pilot_stage_on_gpu_node.sh" base_eval || finish 1
    bash "$MO/scripts/run_judge_panel_on_gpu_node.sh" "results/base_own$SDF_EVAL_TAG.logs.json"
    finish $? ;;
  lr_test)
    export LR_TEST_LRS="${LR_TEST_LRS:-1e-5 5e-5}"
    bash "$MO/scripts/run_sdf_on_gpu_node.sh" "$(deadline 9)" lr_test
    finish $? ;;
  stage1)
    [ -n "${SDF_LR:-}" ] || { say "set SDF_LR in $ENV_FILE first (Feifan picks it from lr_test)"; finish 1; }
    bash "$MO/scripts/run_sdf_on_gpu_node.sh" "$(deadline 14)" stage1
    finish $? ;;
  *) say "unknown plan '$PLAN'"; finish 2 ;;
esac
