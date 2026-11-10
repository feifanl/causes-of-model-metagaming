# shellcheck shell=bash
# SFT organism stages (SRH and its control, or the GT pair): reasoning examples, train, merge, evaluate, compare.
# Sourced by run_pilot_stage_on_gpu_node.sh; not run on its own.

stage_gen_reasoning() {
  require_done bf16; gpus_free B
  serve B "$BF16" base 8001
  timeout 1h "$PY" -m scripts.data.gen_reasoning_examples_with_base --base-url "$(half_url B 8001)" --model base \
      --base-dir "$BF16" --out data/reasoning_examples.jsonl
  "$PY" -m scripts.data.build_sft_datasets --reasoning-examples data/reasoning_examples.jsonl \
      --out-dir data/processed_cotreg --stats data/STATS_cotreg.md
  log "reasoning examples: $(wc -l < data/reasoning_examples.jsonl); $(grep -m1 'Completion-token difference' data/STATS_cotreg.md)"
}

stage_train() {
  require_done bringup bf16; gpus_free A B; need_disk_gb 100
  on_half A timeout 5h "$PY" -m scripts.train.train_sft --arm "$TREAT_ARM" --seed 0 --model "$BF16" \
      --data-dir "$SFT_DATA_DIR" --output-dir "outputs/${TREAT_ARM}_seed0$VARIANT" > "$STATE/train_$TREAT_ARM$VARIANT.log" 2>&1 &
  local a=$!
  on_half B timeout 5h "$PY" -m scripts.train.train_sft --arm "$CTRL_ARM" --seed 0 --model "$BF16" \
      --data-dir "$SFT_DATA_DIR" --output-dir "outputs/${CTRL_ARM}_seed0$VARIANT" > "$STATE/train_$CTRL_ARM$VARIANT.log" 2>&1 &
  local b=$!
  local fail=0
  wait $a || { log "$TREAT_ARM training failed; see $STATE/train_$TREAT_ARM$VARIANT.log"; fail=1; }
  wait $b || { log "$CTRL_ARM training failed; see $STATE/train_$CTRL_ARM$VARIANT.log"; fail=1; }
  [ $fail -eq 0 ] || die "training failed"
  for arm in "$TREAT_ARM" "$CTRL_ARM"; do
    check_json "outputs/${arm}_seed0$VARIANT/run_summary.json" "math.isfinite(d['train_loss'])" "$arm loss not finite"
    [ -f "outputs/${arm}_seed0$VARIANT/adapter_model.safetensors" ] || die "$arm adapter missing"
  done
  # Same steps and hyperparameters in both arms, apart from the arm itself.
  "$PY" - "$VARIANT" "$TREAT_ARM" "$CTRL_ARM" <<'EOF' || die "arms differ in steps or hyperparameters"
import json, sys
t, c = sys.argv[2], sys.argv[3]
s = {a: json.load(open(f"outputs/{a}_seed0{sys.argv[1]}/run_summary.json")) for a in (t, c)}
skip = {"arm", "output_dir"}
diff = {k: (s[t]["args"][k], s[c]["args"].get(k)) for k in s[t]["args"]
        if k not in skip and s[t]["args"][k] != s[c]["args"].get(k)}
for a in s:
    print(a, "steps", s[a]["global_steps"], "loss", round(s[a]["train_loss"], 4), "wall_h", round(s[a]["wall_seconds"] / 3600, 2))
print("arg differences:", diff or "none")
raise SystemExit(1 if diff or s[t]["global_steps"] != s[c]["global_steps"] else 0)
EOF
  # Loss must fall: mean of the last 10 logged steps below the first 10.
  for arm in "$TREAT_ARM" "$CTRL_ARM"; do
    "$PY" - "outputs/${arm}_seed0$VARIANT/run_summary.json" <<'EOF' || die "$arm loss did not fall"
import json, sys
losses = json.load(open(sys.argv[1]))["loss_history"]
first, last = sum(losses[:10]) / len(losses[:10]), sum(losses[-10:]) / len(losses[-10:])
print(f"{sys.argv[1]}: first10 {first:.4f} last10 {last:.4f}")
raise SystemExit(0 if last < first else 1)
EOF
  done
}

stage_fetch_adapters() {
  # Reuse adapters trained earlier (uploaded to the HF repo by an earlier run) instead of retraining.
  [ -n "${HF_ARTIFACT_REPO:-}" ] && [ -n "${HF_WRITE_TOKEN:-}" ] || die "HF_ARTIFACT_REPO / HF_WRITE_TOKEN unset"
  need_disk_gb 50
  for arm in "$TREAT_ARM" "$CTRL_ARM"; do
    local name="${arm}_seed0$VARIANT"
    HF_TOKEN="$HF_WRITE_TOKEN" timeout 1h "$REPO/venv/bin/hf" download "$HF_ARTIFACT_REPO" --repo-type model \
        --include "adapters/$name/*" --local-dir "$NVME/hf_fetch" > "$STATE/fetch_$name.log" 2>&1 \
      || die "download of adapters/$name failed; see $STATE/fetch_$name.log"
    rm -rf "outputs/$name"; mkdir -p outputs; cp -r "$NVME/hf_fetch/adapters/$name" "outputs/$name"
    check_json "outputs/$name/adapter_config.json" "d['base_model_name_or_path'] == 'openai/gpt-oss-120b'" \
        "$name adapter was not trained on the Hub base"
    [ "$(stat -c %s "outputs/$name/adapter_model.safetensors")" -gt 1000000000 ] || die "$name adapter is under 1 GB"
    log "fetched $name ($(du -sh "outputs/$name" | cut -f1))"
  done
  touch "$STATE/train$PAIR$VARIANT.done"  # merge requires it; the adapters stand in for training here
}

stage_merge() {
  require_done "train$PAIR$VARIANT" bf16; gpus_free A B; need_disk_gb 520
  # GPU merge: the GPUs are idle in this slot and a CPU forward pass of a 117B MoE is slow.
  # Each merge is checked on its own arm's training data: text a model finds unlikely is far more
  # sensitive to bf16 rounding (2026-10-01: ~0.01 nats/token on own data vs ~0.055 on the other
  # arm's, symmetric across arms), so a shared verify set fails whichever arm it doesn't match.
  on_half A timeout 2h "$PY" -m scripts.train.merge_lora_into_base --adapter "outputs/${TREAT_ARM}_seed0$VARIANT" \
      --base "$BF16" --out "$NVME/merged/${TREAT_ARM}_seed0$VARIANT" --verify-data "$SFT_DATA_DIR/$TREAT_ARM.jsonl" \
      > "$STATE/merge_$TREAT_ARM$VARIANT.log" 2>&1 &
  local a=$!
  on_half B timeout 2h "$PY" -m scripts.train.merge_lora_into_base --adapter "outputs/${CTRL_ARM}_seed0$VARIANT" \
      --base "$BF16" --out "$NVME/merged/${CTRL_ARM}_seed0$VARIANT" --verify-data "$SFT_DATA_DIR/$CTRL_ARM.jsonl" \
      > "$STATE/merge_$CTRL_ARM$VARIANT.log" 2>&1 &
  local b=$!
  local fail=0
  wait $a || { log "$TREAT_ARM merge failed; see $STATE/merge_$TREAT_ARM$VARIANT.log"; fail=1; }
  wait $b || { log "$CTRL_ARM merge failed; see $STATE/merge_$CTRL_ARM$VARIANT.log"; fail=1; }
  [ $fail -eq 0 ] || die "merge failed"
  for arm in "$TREAT_ARM" "$CTRL_ARM"; do
    log "$arm$VARIANT merge_verification: $("$PY" -c "import json; print(json.load(open('$NVME/merged/${arm}_seed0$VARIANT/provenance.json'))['merge_verification'])")"
  done
}

stage_arm_eval() {
  require_done "merge$PAIR$VARIANT" "base_eval$EVAL_TAG"; gpus_free A B
  local srh="${TREAT_ARM}_seed0$VARIANT" ctl="${CTRL_ARM}_seed0$VARIANT" url_a url_b
  serve A "$NVME/merged/$srh" "$srh" 8000
  serve B "$NVME/merged/$ctl" "$ctl" 8001
  url_a=$(half_url A 8000); url_b=$(half_url B 8001)
  # One tracker around both runs: key usage is per key, so two trackers would each count both.
  # With the judge panel, both runs only save answers (--no-score), with no tracker.
  local flags="$EVAL_FLAGS"
  panel && flags="$flags --no-score"
  local both="
    $PY -m scripts.eval.run_pilot_evals --model harmony/$srh --base-url $url_a $flags \
        --tag $srh$EVAL_TAG --tasks $EVAL_TASKS > $STATE/eval_$srh$EVAL_TAG.log 2>&1 & a=\$!
    $PY -m scripts.eval.run_pilot_evals --model harmony/$ctl --base-url $url_b $flags \
        --tag $ctl$EVAL_TAG --tasks $EVAL_TASKS > $STATE/eval_$ctl$EVAL_TAG.log 2>&1 & b=\$!
    wait \$a; ra=\$?; wait \$b; rb=\$?; exit \$((ra || rb))"
  if panel; then
    timeout 4h bash -c "$both"
  else
    timeout 3h "$PY" -m scripts.analysis.track_openrouter_spend --label "arm_evals$PAIR$VARIANT$EVAL_TAG" --cap "$SPEND_CAP" -- bash -c "$both"
  fi
  eval_health "results/$srh$EVAL_TAG.json" "$EVAL_TASKS"
  eval_health "results/$ctl$EVAL_TAG.json" "$EVAL_TASKS"
}

stage_compare() {
  [ -z "$PAIR" ] || die "compare reads the SRH pair's rules; the GT pair has scripts/analysis/compare_gt_gap.py"
  require_done "arm_eval$VARIANT$EVAL_TAG" "base_eval$EVAL_TAG"
  local out="results/pilot_comparison$VARIANT$EVAL_TAG.md"
  "$PY" -m scripts.analysis.compare_pilot_results --treatment "results/srh_mixed_seed0$VARIANT$EVAL_TAG.json" \
      --control "results/control_seed0$VARIANT$EVAL_TAG.json" --base "results/base_own$EVAL_TAG.json" --out "$out"
  grep -E "^\*\*Verdict" "$out" | tee -a "$STATE/STATUS"
}

