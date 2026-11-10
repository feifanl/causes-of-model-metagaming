# shellcheck shell=bash
# Behavioral-gap stages (BEHAVIORAL_GAP_PLAN.md): the GT pair's data, the framing-grid comparison, eval-awareness probes.
# Sourced by run_pilot_stage_on_gpu_node.sh; not run on its own.

stage_gt_data() {
  local examples=data/reasoning_examples.jsonl
  if [ ! -f "$examples" ]; then  # the SRH run's own rows (their hash is in data/STATS_cotreg.md), so both pairs share them
    [ -n "${HF_ARTIFACT_REPO:-}" ] || die "HF_ARTIFACT_REPO unset: the GT pair's CoT-regularizer rows come from there"
    timeout 30m "$REPO/venv/bin/hf" download "$HF_ARTIFACT_REPO" run_cot_followup/data/reasoning_examples.jsonl \
        --repo-type model --local-dir "$NVME/hf_fetch" > "$STATE/gt_data_fetch.log" 2>&1 \
      || die "download of the CoT-regularizer rows failed; see $STATE/gt_data_fetch.log"
    cp "$NVME/hf_fetch/run_cot_followup/data/reasoning_examples.jsonl" "$examples"
  fi
  "$PY" -m data.build_sft_datasets --gt-cues --reasoning-examples "$examples" --out-dir data/processed_gt \
      --stats data/STATS_gt.md
  git -C "$REPO" diff --exit-code --stat -- model-organisms/data/STATS_gt.md \
    || die "rebuilt GT data differs from the committed data/STATS_gt.md; stop and compare"
}

stage_gap_compare() {
  # The unframed tiers (the grid's neutral column) are in the run without _gap: _gap_panel_reasoning_on -> _panel_reasoning_on.
  local base="results/base_own${EVAL_TAG/_gap/}.json,results/base_own$EVAL_TAG.json" models=()
  models=(--model "base=$base")
  if [ -n "$PAIR" ]; then
    local t="results/${TREAT_ARM}_seed0$VARIANT$EVAL_TAG.json" c="results/${CTRL_ARM}_seed0$VARIANT$EVAL_TAG.json"
    models+=(--model "$TREAT_ARM=$t" --model "$CTRL_ARM=$c")
    "$PY" -m analysis.compare_gt_gap --treatment "$t" --control "$c" --base "$base" \
        --out "results/gt_gap_comparison$VARIANT$EVAL_TAG.md"
  fi
  "$PY" -m analysis.compare_framing_results "${models[@]}" --base base --out "results/framing_comparison$PAIR$VARIANT$EVAL_TAG.md"
}

stage_probes() {
  [ -n "$PAIR" ] || die "probes score the TREAT_ARM/CTRL_ARM pair's adapters; set them (the GT pair)"
  require_done bf16 "train$PAIR$VARIANT"; gpus_free A B
  local probe=results/probes_base.pt score=(timeout 4h "$PY" -m eval.train_eval_awareness_probes score
                                            --probe results/probes_base.pt --model "$BF16")
  [ -f "$probe" ] || on_half A timeout 3h "$PY" -m eval.train_eval_awareness_probes train --model "$BF16" \
      --out "$probe" > "$STATE/probes_train.log" 2>&1 || die "probe training failed; see $STATE/probes_train.log"
  log "probe layer and validation AUROC: $("$PY" -c "import json, sys; d = json.load(open(sys.argv[1])); \
print(d['layer'], round(d['metrics'][d['layer']]['val_auroc'], 3))" "${probe%.pt}.json")"
  score_one() {  # score_one <A|B> <name> [--adapter dir]
    local half=$1 name=$2; shift 2
    on_half "$half" "${score[@]}" "$@" --out "results/probe_scores_$name.json" > "$STATE/probes_$name.log" 2>&1 \
      || { log "probe scoring of $name failed; see $STATE/probes_$name.log"; return 1; }
  }
  local t="${TREAT_ARM}_seed0$VARIANT" c="${CTRL_ARM}_seed0$VARIANT" fail=0 a b
  if halves_share_gpus; then
    score_one A "$t" --adapter "outputs/$t" && score_one A "$c" --adapter "outputs/$c" && score_one A base_own || fail=1
  else
    { score_one A "$t" --adapter "outputs/$t" && score_one A base_own; } & a=$!
    score_one B "$c" --adapter "outputs/$c" & b=$!
    wait "$a" || fail=1
    wait "$b" || fail=1
  fi
  [ "$fail" -eq 0 ] || die "probe scoring failed"
}

