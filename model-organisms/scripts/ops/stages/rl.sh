# shellcheck shell=bash
# RL organism stages (published Redwood and AISI LoRAs): prepare the adapter, check serving, evaluate.
# Sourced by run_pilot_stage_on_gpu_node.sh; not run on its own.

half_port() {  # half_port -> the server port for $HALF
  case "$HALF" in
    A) echo 8000 ;;
    B) echo 8001 ;;
    *) die "HALF must be A or B, not '$HALF'" ;;
  esac
}

stage_rl_adapter() {
  [ -n "$ORGANISM" ] || die "ORGANISM unset (names: scripts/train/download_rl_organism_adapter.py)"
  require_done bf16
  # outputs/<ORGANISM>_full: the whole adapter (the fp32 reference); outputs/<ORGANISM>: what vLLM serves, plus
  # serve_base.txt if part of the adapter had to go into a copy of the base (Redwood's lm_head).
  timeout 1h "$PY" -m scripts.train.prepare_rl_adapter_for_serving --organism "$ORGANISM" --out "outputs/$ORGANISM" \
      --base "$BF16" --base-copies "$NVME/serve_bases" --venv "$REPO/venv-tinker" > "$STATE/prepare_$ORGANISM.log" 2>&1 \
    || die "prepare failed; see $STATE/prepare_$ORGANISM.log"
  log "$(tail -1 "$STATE/prepare_$ORGANISM.log")"
}

serve_sglang() {  # serve_sglang <A|B> <port> <served name> <lora name> <lora dir>: the bf16 base + one LoRA in SGLang
  local half=$1 port=$2 name=$3 gpus url host=()
  gpus=$(half_list "$half"); url=$(half_url "$half" "$port")
  is_here "$(half_host "$half")" || host=(--host 0.0.0.0)
  curl -sf "$url/models" >/dev/null 2>&1 && die "port $port already serving"
  half_cmd "$half"
  setsid "${HALF_CMD[@]}" "$SGLANG_PY" -m sglang.launch_server --model-path "$BF16" --served-model-name "$name" \
      --tp "$(n_gpus "$gpus")" --port "$port" "${host[@]}" --context-length 16384 --enable-lora --max-lora-rank 32 --lora-target-modules all \
      --lora-paths "$4=$5" --moe-runner-backend triton > "$STATE/sglang_$name.log" 2>&1 &
  # --moe-runner-backend triton: gpt-oss defaults to triton_kernel, which has no LoRA path for bf16 experts
  # (UnquantizedFusedMoEMethod does not expose quant info for 'triton_kernel'; Session 4b).
  SERVERS+=($!)
  local pid=$! waited=0
  until curl -sf "$url/models" | grep -q "\"$name\""; do
    kill -0 "$pid" 2>/dev/null || die "SGLang for $name exited; see $STATE/sglang_$name.log"
    [ "$waited" -ge 2700 ] && die "SGLang for $name not ready after 45 min"
    sleep 15; waited=$((waited + 15))
  done
  log "SGLang ready: $name + LoRA $4 on half $half (GPUs $gpus), $url (${waited}s)"
}

# serve_rl <A|B> <port>: the base (served as base_<HALF>_<ORGANISM>: one log per server) plus $ORGANISM's LoRA,
# unmerged, on the organism's server. Sets RL_SERVER and RL_MODEL (the name that selects the LoRA).
serve_rl() {
  local flags base="$BF16"
  RL_SERVER=$("$PY" -m scripts.train.download_rl_organism_adapter --organism "$ORGANISM" --print-server) \
    || die "unknown ORGANISM '$ORGANISM'"
  if [ "$RL_SERVER" = sglang ]; then
    [ -x "$SGLANG_PY" ] || die "no SGLang venv at $SGLANG_PY (setup_gpu_node.sh with WITH_SGLANG=1)"
    serve_sglang "$1" "$2" "base_${HALF}_$ORGANISM" "$ORGANISM" "$MO/outputs/$ORGANISM"
    RL_MODEL="base_${HALF}_$ORGANISM:$ORGANISM"   # SGLang's OpenAI API picks a LoRA as <base>:<adapter>
    return
  fi
  flags=$("$PY" -m scripts.train.download_rl_organism_adapter --organism "$ORGANISM" --print-vllm-flags)
  [ -f "outputs/$ORGANISM/serve_base.txt" ] && base=$(cat "outputs/$ORGANISM/serve_base.txt")
  # shellcheck disable=SC2086  # flags is a flag list
  serve "$1" "$base" "base_${HALF}_$ORGANISM" "$2" --enable-lora --max-lora-rank 32 --lora-modules "$ORGANISM=$MO/outputs/$ORGANISM" $flags
  curl -sf "$(half_url "$1" "$2")/models" | grep -q "\"$ORGANISM\"" \
    || die "vLLM does not list the LoRA '$ORGANISM'; see $STATE/vllm_base_${HALF}_$ORGANISM.log"
  RL_MODEL="$ORGANISM"
}

stage_rl_check() {
  [ -n "$ORGANISM" ] || die "ORGANISM unset (names: scripts/train/download_rl_organism_adapter.py)"
  require_done bf16 "rl_adapter_$ORGANISM"; gpus_free A B; need_disk_gb 520
  local ref="results/rl_lora_check_${ORGANISM}_ref.json" out="results/rl_lora_check_$ORGANISM.json" port
  # Exact reference: the base upcast to fp32 with the adapter added in fp32 (~480 GB, deleted after).
  # REUSE_RL_REF=1 keeps an existing reference that has the noise floor (the adapter is pinned).
  if [ "${REUSE_RL_REF:-0}" = 1 ] && grep -q floor_ref_nll "$ref" 2>/dev/null; then
    log "reusing the fp32 reference in $ref"
  else
    timeout 2h "$PY" -m scripts.checks.check_rl_lora_serving reference --adapter "outputs/${ORGANISM}_full" --base "$BF16" \
        --scratch "$NVME/ref_fp32_$ORGANISM" --out "$ref" > "$STATE/rl_check_ref_$ORGANISM.log" 2>&1 \
      || die "fp32 reference failed; see $STATE/rl_check_ref_$ORGANISM.log"
  fi
  HALF=A
  port=$(half_port)
  gpus_free A
  serve_rl A "$port"
  "$PY" -m scripts.checks.check_rl_lora_serving served --server "$RL_SERVER" --organism "$ORGANISM" --base-model "base_A_$ORGANISM" \
      --base-url "$(half_url A "$port")" --ref "$ref" --out "$out" > "$STATE/rl_check_served_$ORGANISM.log" 2>&1 \
    || die "served LoRA does not match the fp32 reference; see $out and $STATE/rl_check_served_$ORGANISM.log"
  log "$("$PY" -c "import json, sys; r = json.load(open(sys.argv[1])); print('lora vs fp32 ref', r['lora_vs_ref'], '| base vs ref', r['base_vs_ref'])" "$out")"
}

stage_rl_eval() {
  [ -n "$ORGANISM" ] || die "ORGANISM unset (names: scripts/train/download_rl_organism_adapter.py)"
  local port
  port=$(half_port)
  require_done bf16 "rl_check_$ORGANISM"   # the served LoRA matched the fp32 reference
  gpus_free "$HALF"
  serve_rl "$HALF" "$port"
  local evals=("$PY" -m scripts.eval.run_pilot_evals --model "harmony/$RL_MODEL" --base-url "$(half_url "$HALF" "$port")"
               --tag "$ORGANISM$EVAL_TAG" --tasks "$EVAL_TASKS")
  # shellcheck disable=SC2206  # EVAL_FLAGS is a flag list
  evals+=($EVAL_FLAGS)
  if panel || [[ ",$EVAL_TASKS," =~ ,(em|hacking|hacking_unstated|hacking_vague|framing_[a-z_]+|gt_[a-z_]+), ]]; then
    # Judged tasks take turns on OpenRouter with the other half (judged; see stage_sdf_eval); the panel saves answers.
    judged "$ORGANISM$EVAL_TAG" "${evals[@]}"
  else
    timeout 4h "${evals[@]}"   # judge-free: no lock, so both halves run at once
  fi
  # RL without a KL penalty can damage the format: recorded as a result, not fatal (as for SDF).
  deferred "results/$ORGANISM$EVAL_TAG.json" || health_report "results/$ORGANISM$EVAL_TAG.json" "$EVAL_TASKS" \
    || log "FORMAT OUTSIDE LIMITS for $ORGANISM$EVAL_TAG: recorded as a result (details in $STATE/$NAME.log)"
}

