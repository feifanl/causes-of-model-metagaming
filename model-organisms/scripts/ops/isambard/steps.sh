# shellcheck shell=bash
# The run's steps, in order of `run_step` below: each step_<name> runs one part of the pipeline (ISAMBARD.md).
# Sourced by run_on_isambard.sh; not run on its own.

step_setup() {
  fetch_harmony_vocab || return 1
  if ! command -v uv >/dev/null && [ ! -x "$HOME/.local/bin/uv" ]; then
    curl -LsSf https://astral.sh/uv/install.sh | sh > "$STATE/uv_install.log" 2>&1 || { say "uv install failed"; return 1; }
  fi
  UV="$(command -v uv || echo "$HOME/.local/bin/uv")"
  "$UV" python install 3.11 > "$STATE/python_install.log" 2>&1 || { say "Python 3.11 install failed"; return 1; }
  PYTHON="$("$UV" python find 3.11)"
  setup_cuda_compat || return 1
  apply_setup_choices
  PYTHON="$PYTHON" UV="$UV" bash "$MO/scripts/ops/setup_gpu_node.sh" "$DATA" > "$STATE/setup.log" 2>&1 \
    || { say "setup failed; see $STATE/setup.log"; return 1; }
  check_cuda_in_venvs || return 1
  if [ "$(uname -m)" != x86_64 ]; then pick_training_attention || return 1; fi
  if [ -n "$HOST_B" ]; then pick_nccl_settings || return 1; fi
  apply_setup_choices
}
step_judges() {
  (cd "$MO" && "$REPO/venv/bin/python" -m scripts.eval.run_pilot_evals --tasks judge_validation --model mockllm/model \
      --tag judge_validation_panel --no-score > "$STATE/judge_validation.log" 2>&1) \
    || { say "judge validation set failed; see $STATE/judge_validation.log"; return 1; }
  bash "$MO/scripts/ops/run_judge_panel_on_gpu_node.sh" results/judge_validation_panel.logs.json
}
step_base() {
  bash "$MO/scripts/ops/run_pilot_stage_on_gpu_node.sh" bf16 || return 1
  # Every eval the comparisons use, reasoning on, answers only; the panel judges them after.
  EVAL_FLAGS="" EVAL_TAG="$SDF_EVAL_TAG" \
    EVAL_TASKS="em,hacking,hacking_unstated,hacking_vague,mmlu,sdf_recall,sdf_saliency_coding,sdf_saliency_everyday,sdf_spillover,gpqa_main,gpqa,ifbench,livecodebench" \
    bash "$MO/scripts/ops/run_pilot_stage_on_gpu_node.sh" base_eval || return 1
  bash "$MO/scripts/ops/run_judge_panel_on_gpu_node.sh" "results/base_own$SDF_EVAL_TAG.logs.json"
}
step_lr_test() {
  LR_TEST_LRS="${LR_TEST_LRS:-1e-5 5e-5}" bash "$MO/scripts/ops/run_sdf_on_gpu_node.sh" "$(deadline)" lr_test
}
step_redwood() {
  # Redwood's expert LoRA needs SGLang (vLLM 0.30 drops it; DECISIONS 'RL organism serving'). Installed here, not in
  # setup, so a failed SGLang install stops only this step.
  local uv run="$MO/scripts/ops/run_pilot_stage_on_gpu_node.sh" rw=ORGANISM=redwood_step952 judged=em,hacking,hacking_unstated,hacking_vague,mmlu a b rc=0
  uv="$(command -v uv || echo "$HOME/.local/bin/uv")"
  WITH_SGLANG=1 PYTHON="$("$uv" python find 3.11)" UV="$uv" bash "$MO/scripts/ops/setup_gpu_node.sh" "$DATA" \
      > "$STATE/setup_sglang.log" 2>&1 || { say "SGLang setup failed; see $STATE/setup_sglang.log"; return 1; }
  env "$rw" bash "$run" rl_adapter || return 1
  # Session 4's fp32 reference (committed in results/; the adapter is pinned): a new one needs ~480 GB of fp32 weights.
  env "$rw" REUSE_RL_REF=1 bash "$run" rl_check || return 1
  # Half A: reasoning on, the base step's tag and its tasks minus the SDF ones. Half B: base, then Redwood, with
  # reasoning off on the judged tasks (the base step ran reasoning on only).
  # One node: the halves share its GPUs, so A then B.
  half_a() {
    env "$rw" HALF=A EVAL_FLAGS= EVAL_TAG="$SDF_EVAL_TAG" EVAL_TASKS="$judged,gpqa_main,gpqa,ifbench,livecodebench" \
      bash "$run" rl_eval
  }
  half_b() {
    EVAL_FLAGS=--no-reasoning EVAL_TAG=_panel EVAL_TASKS="$judged" bash "$run" base_eval \
      && env "$rw" HALF=B EVAL_FLAGS=--no-reasoning EVAL_TAG=_panel EVAL_TASKS="$judged" bash "$run" rl_eval
  }
  if [ -z "$HOST_B" ]; then
    half_a && half_b || return 1
  else
    half_a & a=$!
    half_b & b=$!
    wait "$a" || rc=1
    wait "$b" || rc=1
    [ "$rc" -eq 0 ] || return 1
  fi
  bash "$MO/scripts/ops/run_judge_panel_on_gpu_node.sh" "results/redwood_step952$SDF_EVAL_TAG.logs.json" \
      results/redwood_step952_panel.logs.json results/base_own_panel.logs.json
}
step_stage1() {
  bash "$MO/scripts/ops/run_sdf_on_gpu_node.sh" "$(deadline)" stage1
}

# The framing grid's six framed cells and the GT organism's four cued conditions (run_pilot_evals.py TASKS).
GAP_TASKS="framing_stated_eval,framing_stated_deployment,framing_vague_eval,framing_vague_deployment,\
framing_none_eval,framing_none_deployment,gt_train_eval,gt_train_deployment,gt_heldout_eval,gt_heldout_deployment"
# Run with tags _gap_panel_reasoning_on and _gap_panel; base's unframed tiers come from the base and redwood steps.
GAP_TAGS="_gap_panel_reasoning_on _gap_panel"
GT_PAIR="TREAT_ARM=gt_conditional CTRL_ARM=gt_unconditional VARIANT=_cotreg SFT_DATA_DIR=data/processed_gt"
step_gap_base() {
  local run="$MO/scripts/ops/run_pilot_stage_on_gpu_node.sh" tag
  # Does a framing change the task or its metric? No model sampled; the panel judges each pair.
  (cd "$MO" && "$REPO/venv/bin/python" -m scripts.eval.run_pilot_evals --tasks framing_minimal_pairs --model mockllm/model \
      --tag framing_minimal_pairs_panel --no-score > "$STATE/framing_minimal_pairs.log" 2>&1) \
    || { say "framing minimal pairs failed; see $STATE/framing_minimal_pairs.log"; return 1; }
  # Base's own curve and gap: the reference every organism is read against. Both on half B, one after the other.
  EVAL_FLAGS="" EVAL_TAG=_gap_panel_reasoning_on EVAL_TASKS="$GAP_TASKS" bash "$run" base_eval || return 1
  EVAL_FLAGS=--no-reasoning EVAL_TAG=_gap_panel EVAL_TASKS="$GAP_TASKS" bash "$run" base_eval || return 1
  bash "$MO/scripts/ops/run_judge_panel_on_gpu_node.sh" results/framing_minimal_pairs_panel.logs.json \
      results/base_own_gap_panel_reasoning_on.logs.json results/base_own_gap_panel.logs.json || return 1
  for tag in $GAP_TAGS; do EVAL_TAG=$tag bash "$run" gap_compare || return 1; done
}
step_gt() {
  # Both arms train at once, one per node, as the SRH pair did on one 8-GPU node.
  [ -n "$HOST_B" ] || { say "gt trains its two arms side by side: it needs two nodes (ISAMBARD_NODES=2)"; return 1; }
  local run="$MO/scripts/ops/run_pilot_stage_on_gpu_node.sh" tag tasks
  # The GT tasks, the whole framing grid (Part 3 method 1 on the GT pair), and MMLU for the capability bound.
  tasks="$GAP_TASKS,hacking,hacking_vague,hacking_unstated,mmlu"
  bash "$run" gt_data || return 1
  # Train needs bringup: 10 SFT steps, first on Isambard's 96 GB GPUs.
  BRINGUP_MAX_MEM_GB=90 bash "$run" bringup || return 1
  # shellcheck disable=SC2086  # GT_PAIR is a list of VAR=value words
  env $GT_PAIR bash "$run" train || return 1
  # shellcheck disable=SC2086
  env $GT_PAIR bash "$run" merge || return 1
  # shellcheck disable=SC2086
  env $GT_PAIR EVAL_FLAGS= EVAL_TAG=_gap_panel_reasoning_on EVAL_TASKS="$tasks" bash "$run" arm_eval || return 1
  # shellcheck disable=SC2086
  env $GT_PAIR EVAL_FLAGS=--no-reasoning EVAL_TAG=_gap_panel EVAL_TASKS="$tasks" bash "$run" arm_eval || return 1
  # Both merged models (468 GB) are done with: the probes use the adapters.
  rm -rf "$DATA/merged/gt_conditional_seed0_cotreg" "$DATA/merged/gt_unconditional_seed0_cotreg"
  bash "$MO/scripts/ops/run_judge_panel_on_gpu_node.sh" \
      results/gt_conditional_seed0_cotreg_gap_panel_reasoning_on.logs.json \
      results/gt_unconditional_seed0_cotreg_gap_panel_reasoning_on.logs.json \
      results/gt_conditional_seed0_cotreg_gap_panel.logs.json results/gt_unconditional_seed0_cotreg_gap_panel.logs.json \
    || return 1
  # shellcheck disable=SC2086
  for tag in $GAP_TAGS; do env $GT_PAIR EVAL_TAG=$tag bash "$run" gap_compare || return 1; done
}
step_probes() {
  # shellcheck disable=SC2086
  env $GT_PAIR bash "$MO/scripts/ops/run_pilot_stage_on_gpu_node.sh" probes
}

