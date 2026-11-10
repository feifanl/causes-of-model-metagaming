# shellcheck shell=bash
# Base-model stages: bf16 copy of gpt-oss, serving bring-up, base evals, expert-routing overlap.
# Sourced by run_pilot_stage_on_gpu_node.sh; not run on its own.

stage_bf16() {
  # BF16_GPUS: half B fits on H200 (4 x 141 GB); on H100 (4 x 80 GB) loading 234 GB plus dequantization
  # buffers ran out of memory (2026-10-08), so pass all eight there.
  # Elsewhere half B; Isambard's 96 GB GPUs should fit, and an out-of-memory run is retried with part on CPU.
  local where=(on_half B)
  if [ -n "${BF16_GPUS:-}" ]; then gpus_free A B; where=(env "CUDA_VISIBLE_DEVICES=$BF16_GPUS"); else gpus_free B; fi
  need_disk_gb 300
  retry_on_oom "$STATE/dequantize.log" "$BF16" "${where[@]}" timeout 2h "$PY" -m train.dequantize_base_to_bf16 --out "$BF16" \
    || die "dequantization failed; see $STATE/dequantize.log"
  [ -f "$BF16/provenance.json" ] || die "no provenance.json in $BF16"
  grep -q quantization_config "$BF16/config.json" && die "bf16 config still has quantization_config"
  log "bf16 size: $(du -sh "$BF16" | cut -f1)"
}

stage_bringup() {
  gpus_free A
  local model=()
  [ -f "$STATE/bf16.done" ] && model=(--model "$BF16")  # else the Hub MXFP4 checkpoint, dequantized on load
  on_half A timeout 90m "$PY" -m train.train_sft --arm srh_mixed "${model[@]}" \
      --max-steps 10 --output-dir outputs/bringup_sft
  check_json outputs/bringup_sft/run_summary.json "math.isfinite(d['train_loss'])" "loss not finite"
  check_json outputs/bringup_sft/run_summary.json "d['steady_state']['peak_mem_gb'] < $BRINGUP_MAX_MEM_GB" \
      "peak memory too close to the GPU's (BRINGUP_MAX_MEM_GB=$BRINGUP_MAX_MEM_GB)"
  "$PY" -c "import json; print(json.load(open('outputs/bringup_sft/run_summary.json'))['steady_state'])" | tee -a "$STATE/STATUS"
}

stage_base_eval() {
  require_done bf16; gpus_free B
  serve B "$BF16" base 8001
  # shellcheck disable=SC2086  # EVAL_FLAGS is a flag list
  judged "base_own$EVAL_TAG" "$PY" -m eval.run_pilot_evals --model harmony/base --base-url "$(half_url B 8001)" \
      --tag "base_own$EVAL_TAG" --tasks "$EVAL_TASKS" $EVAL_FLAGS
  eval_health "results/base_own$EVAL_TAG.json" "$EVAL_TASKS"
}

stage_routing() {
  require_done base_eval; gpus_free A
  on_half A timeout 2h "$PY" -m analysis.measurements routing --model "$BF16" \
      --eval-results results/base_own.json --out results/routing_overlap.json
}

