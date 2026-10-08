#!/usr/bin/env bash
# The one command the Isambard operator runs (ISAMBARD.md). It checks the machine, then runs every step in
# order, uploading results and logs to the HF repo after each one. A finished step is marked done and skipped
# on the next run; a failed step stops everything with FAILED, so rerunning the same command (after Feifan's
# fix and a git pull) continues from the step that failed.
#
#   bash model-organisms/scripts/run_on_isambard.sh          # all steps (the normal way)
#   bash model-organisms/scripts/run_on_isambard.sh check    # preflight only (GPUs, disk, internet, tokens)
#   bash model-organisms/scripts/run_on_isambard.sh <step>   # rerun one step even if marked done
#
# Steps, in order:
#   setup    install everything (Python, venvs, models, data) and run the tests; needs internet
#   judges   validate the judge panel on the judge-validation set
#   base     base model on every eval, judged by the panel (the reference for all comparisons)
#   lr_test  SDF learning-rate test at 0.5 epoch (LR_TEST_LRS, default '1e-5 5e-5')
#   stage1   SDF stage 1; needs SDF_LR in the env file, so the run stops before it until Feifan sends one
#
# No API keys: judging runs on this machine (JUDGING=panel). Settings come from ~/.config/spar/env
# (DATA_DIR and the HF tokens; template: model-organisms/isambard.env.example).
set -uo pipefail

STEPS="setup judges base lr_test stage1"
PLAN="${1:-all}"
case " check all $STEPS " in *" $PLAN "*) ;; *) echo "usage: run_on_isambard.sh [check|${STEPS// /|}]"; exit 2 ;; esac
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
MO="$REPO/model-organisms"
ENV_FILE="$HOME/.config/spar/env"
# shellcheck disable=SC1090
[ -f "$ENV_FILE" ] && set -a && . "$ENV_FILE" && set +a
DATA="${DATA_DIR:-}"
[ -n "$DATA" ] || { echo "FAILED: set DATA_DIR in $ENV_FILE (ISAMBARD.md step 2)"; exit 1; }
export NVME="$DATA" JUDGING=panel SDF_EVAL_TAG="${SDF_EVAL_TAG:-_panel_reasoning_on}"
STATE="$DATA/pilot_state"
mkdir -p "$STATE"
say() { echo "$(date -u +%FT%TZ) isambard $*" | tee -a "$STATE/STATUS"; }
upload_all() {  # results and logs to the HF repo, success or not: it is how Feifan sees what happened
  local hf="$REPO/venv/bin/hf"
  [ -x "$hf" ] && [ -n "${HF_WRITE_TOKEN:-}" ] && [ -n "${HF_ARTIFACT_REPO:-}" ] || return 0
  HF_TOKEN="$HF_WRITE_TOKEN" timeout 1h "$hf" upload "$HF_ARTIFACT_REPO" "$MO/results" "isambard/results" \
      --repo-type model --private --commit-message "isambard: $1 results" > "$STATE/upload_isambard.log" 2>&1
  HF_TOKEN="$HF_WRITE_TOKEN" timeout 1h "$hf" upload "$HF_ARTIFACT_REPO" "$STATE" "isambard/pilot_state" \
      --repo-type model --private --exclude "*.done" --commit-message "isambard: $1 logs" >> "$STATE/upload_isambard.log" 2>&1
}
finish() {  # finish <exit code> <message>
  if [ "$1" -eq 0 ]; then say "DONE: $2"
  else say "FAILED: $2 (exit $1). Tell Feifan; the logs are on the HF repo (or send the last 30 lines of $STATE/STATUS)."; fi
  upload_all "$PLAN" || echo "(upload to HF failed; see $STATE/upload_isambard.log)"
  exit "$1"
}

bash "$MO/scripts/check_isambard_ready.sh" "$DATA" > "$STATE/preflight.log" 2>&1
preflight=$?
cat "$STATE/preflight.log"
[ "$PLAN" = check ] && finish $preflight "check finished"
[ "$preflight" -eq 0 ] || finish 1 "preflight failed; fix the FAIL lines above and rerun"
cd "$REPO" || finish 1 "cannot enter $REPO"
deadline() { echo $(( $(date +%s) + $1 * 3600 )); }

# Each step returns non-zero on failure; its own scripts skip work they already finished.
step_setup() {
  if ! command -v uv >/dev/null && [ ! -x "$HOME/.local/bin/uv" ]; then
    curl -LsSf https://astral.sh/uv/install.sh | sh > "$STATE/uv_install.log" 2>&1 || { say "uv install failed"; return 1; }
  fi
  local uv; uv="$(command -v uv || echo "$HOME/.local/bin/uv")"
  "$uv" python install 3.11 > "$STATE/python_install.log" 2>&1 || { say "Python 3.11 install failed"; return 1; }
  PYTHON="$("$uv" python find 3.11)" bash "$MO/scripts/setup_gpu_node.sh" "$DATA" > "$STATE/setup.log" 2>&1 \
    || { say "setup failed; see $STATE/setup.log"; return 1; }
}
step_judges() {
  (cd "$MO" && "$REPO/venv/bin/python" scripts/run_pilot_evals.py --tasks judge_validation --model mockllm/model \
      --tag judge_validation_panel --no-score > "$STATE/judge_validation.log" 2>&1) \
    || { say "judge validation set failed; see $STATE/judge_validation.log"; return 1; }
  bash "$MO/scripts/run_judge_panel_on_gpu_node.sh" results/judge_validation_panel.logs.json
}
step_base() {
  bash "$MO/scripts/run_pilot_stage_on_gpu_node.sh" bf16 || return 1
  # Every eval the comparisons use, reasoning on, answers only; the panel judges them after.
  EVAL_FLAGS="" EVAL_TAG="$SDF_EVAL_TAG" \
    EVAL_TASKS="em,hacking,mmlu,sdf_recall,sdf_saliency_coding,sdf_saliency_everyday,sdf_spillover,gpqa_main,gpqa,ifbench,livecodebench" \
    bash "$MO/scripts/run_pilot_stage_on_gpu_node.sh" base_eval || return 1
  bash "$MO/scripts/run_judge_panel_on_gpu_node.sh" "results/base_own$SDF_EVAL_TAG.logs.json"
}
step_lr_test() {
  LR_TEST_LRS="${LR_TEST_LRS:-1e-5 5e-5}" bash "$MO/scripts/run_sdf_on_gpu_node.sh" "$(deadline 9)" lr_test
}
step_stage1() {
  bash "$MO/scripts/run_sdf_on_gpu_node.sh" "$(deadline 14)" stage1
}

run_step() {  # run_step <step> [force]
  local marker="$STATE/isambard_$1.done"
  if [ -f "$marker" ] && [ "${2:-}" != force ]; then say "$1 already done; skipping"; return 0; fi
  say "$1 starting"
  "step_$1" || finish 1 "$1 stopped"
  touch "$marker"
  say "$1 finished"
  upload_all "$1" || echo "(upload to HF failed; see $STATE/upload_isambard.log)"
}

if [ "$PLAN" != all ]; then
  [ "$PLAN" != stage1 ] || [ -n "${SDF_LR:-}" ] || finish 1 "set SDF_LR in $ENV_FILE first (Feifan sends it)"
  run_step "$PLAN" force
  finish 0 "$PLAN finished. Results are on the HF repo; tell Feifan."
fi
for s in setup judges base lr_test; do run_step "$s"; done
if [ -z "${SDF_LR:-}" ]; then
  finish 0 "everything up to lr_test finished. Tell Feifan, wait for the SDF_LR setting, add it to $ENV_FILE, then run the same command again."
fi
run_step stage1
finish 0 "all steps finished. Results are on the HF repo; tell Feifan."
