# shellcheck shell=bash
# SDF organism stages: a short slice run, full training, and evaluating a checkpoint.
# Sourced by run_pilot_stage_on_gpu_node.sh; not run on its own.

stage_sdf() {
  require_done bf16; gpus_free A B; need_disk_gb 50
  [ -f outputs/nll_base.json ] || timeout 1h "$PY" -m scripts.eval.score_heldout_nll --model "$BF16" --out outputs/nll_base.json
  timeout 3h "$TORCHRUN" --nproc_per_node "$(n_gpus "$ALL_GPUS")" -m scripts.train.train_sdf --fsdp --model "$BF16" \
      --max-steps 120 --output-dir outputs/sdf_slice
  check_json outputs/sdf_slice/run_summary.json "math.isfinite(d['train_loss'])" "SDF loss not finite"
  timeout 1h "$PY" -m scripts.eval.score_heldout_nll --model "$BF16" --adapter outputs/sdf_slice --out outputs/nll_sdf_slice.json
  "$PY" - <<'EOF' | tee -a "$STATE/STATUS"
import json
b, a = (json.load(open(f"outputs/{n}.json"))["mean_nll"] for n in ("nll_base", "nll_sdf_slice"))
s = json.load(open("outputs/sdf_slice/run_summary.json"))
print(f"held-out NLL {b:.4f} -> {a:.4f} ({'dropped' if a < b else 'DID NOT DROP'}); steady_state {s['steady_state']}")
EOF
}

stage_sdf_train() {
  local data="$SDF_DATA_DIR" out="outputs/sdf_$SDF_RUN"
  [ -f "$data/sdf_train.jsonl" ] && [ -f "$data/sdf_heldout.jsonl" ] \
    || die "no $data/sdf_train.jsonl + sdf_heldout.jsonl: build the $SDF_ARM corpus first"
  require_done bf16; gpus_free A B; need_disk_gb 150   # final adapter + checkpoints, ~17 GB each
  # Base NLL on this corpus's held-out docs, once per data dir (docs without the '<doc>' prefix score differently).
  local base_nll
  base_nll="outputs/nll_base_$(basename "$data").json"
  [ -f "$base_nll" ] || on_half A timeout 1h "$PY" -m scripts.eval.score_heldout_nll --model "$BF16" \
      --data "$data/sdf_heldout.jsonl" --out "$base_nll"
  rm -rf "$out"   # checkpoints left by a failed attempt would pass the checks below
  local launch=(/usr/bin/env "CUDA_VISIBLE_DEVICES=$ALL_GPUS" "$TORCHRUN" --nproc_per_node "$(n_gpus "$ALL_GPUS")") train_env=()
  if [ -n "$HOST_B" ] && ! is_here "$HOST_B"; then
    # Two nodes: one torchrun per node, rendezvous on node A. FSDP shards over all 8 GPUs, as on one 8-GPU node
    # (one 4x96 GB node cannot hold the training state). SDF_TRAIN_ENV: NCCL settings for the inter-node network.
    [ -n "${SDF_TRAIN_ENV:-}" ] && mapfile -t train_env < "$SDF_TRAIN_ENV"
    launch=(srun --overlap --nodes=2 --ntasks-per-node=1 --nodelist="$HOST_A,$HOST_B" --gpus-per-node="$(n_gpus "$HALF_A")"
            --cpus-per-task="${SLURM_CPUS_ON_NODE:-1}" --mem=0 --input=none --kill-on-bad-exit=1
            /usr/bin/env "${train_env[@]}" "$TORCHRUN" --nnodes 2 --nproc_per_node "$(n_gpus "$HALF_A")"
            --rdzv-backend c10d --rdzv-endpoint "$HOST_A:29500" --rdzv-id "sdf_$SDF_RUN")
  fi
  # SDF_ATTN: training attention for platforms without vllm-flash-attn3 (check_training_attention.py picks it).
  # shellcheck disable=SC2086  # SDF_TRAIN_FLAGS is a flag list
  timeout "$SDF_TIMEOUT" "${launch[@]}" -m scripts.train.train_sdf \
      --fsdp --model "$BF16" --seed "$SEED" --data "$data/sdf_train.jsonl" --output-dir "$out" \
      ${SDF_ATTN:+--attn-implementation "$SDF_ATTN"} $SDF_TRAIN_FLAGS \
      > "$STATE/train_sdf_$SDF_RUN.log" 2>&1 || die "training failed; see $STATE/train_sdf_$SDF_RUN.log"

  # Finite loss that fell; the whole schedule ran unless --stop-at-epoch; every adapter the flags asked for exists.
  local check
  check=$("$PY" - "$out" "$MIN_ADAPTER_BYTES" <<'EOF'
import ast, json, math, sys
from pathlib import Path
out, min_bytes = Path(sys.argv[1]), int(sys.argv[2])
s = json.loads((out / "run_summary.json").read_text())
losses = s["loss_history"]
first, last = sum(losses[:10]) / len(losses[:10]), sum(losses[-10:]) / len(losses[-10:])
saves = ast.literal_eval(s["args"].get("save_at_epochs", "[]"))
adapters = [out] + [out / f"checkpoint-epoch{e:g}" for e in saves]
bad = []
if not math.isfinite(s["train_loss"]):
    bad.append(f"train_loss {s['train_loss']}")
if not last < first:
    bad.append(f"loss did not fall (first 10 steps {first:.4f}, last 10 {last:.4f})")
if s["args"].get("stop_at_epoch", "None") == "None" and s["global_steps"] != s["schedule_steps"]:
    bad.append(f"stopped at step {s['global_steps']} of {s['schedule_steps']} without --stop-at-epoch")
for d in adapters:
    f = d / "adapter_model.safetensors"
    if not f.exists() or f.stat().st_size < min_bytes:
        bad.append(f"adapter missing or under {min_bytes:,} bytes: {f}")
print(f"steps {s['global_steps']}/{s['schedule_steps']}, loss {first:.4f} -> {last:.4f}, "
      f"{s['steady_state']['tokens_per_sec']:,.0f} tok/s, adapters: {', '.join(d.name for d in adapters)}"
      + ("; FAILED: " + "; ".join(bad) if bad else ""))
sys.exit(1 if bad else 0)
EOF
) || die "training checks: $check"
  log "$check"

  # Held-out NLL of every adapter, two at a time (one per half; one at a time when the halves share GPUs).
  local ckpts=(final) dirs=("$out") pids=() i d at_once=2
  halves_share_gpus && at_once=1
  for d in "$out"/checkpoint-epoch*; do
    if [ -d "$d" ]; then ckpts+=("${d##*/checkpoint-}"); dirs+=("$d"); fi
  done
  for i in "${!dirs[@]}"; do
    on_half "$([ $((i % 2)) -eq 0 ] && echo A || echo B)" timeout 1h "$PY" \
        -m scripts.eval.score_heldout_nll --model "$BF16" --adapter "${dirs[$i]}" --data "$data/sdf_heldout.jsonl" \
        --out "outputs/nll_sdf_${SDF_RUN}_${ckpts[$i]}.json" > "$STATE/nll_sdf_${SDF_RUN}_${ckpts[$i]}.log" 2>&1 &
    pids+=($!)
    if [ ${#pids[@]} -eq "$at_once" ] || [ "$i" -eq $(( ${#dirs[@]} - 1 )) ]; then
      for d in "${pids[@]}"; do wait "$d" || die "held-out NLL failed; see $STATE/nll_sdf_${SDF_RUN}_*.log"; done
      pids=()
    fi
  done
  local line
  line=$("$PY" - "$base_nll" "outputs/nll_sdf_$SDF_RUN" "${ckpts[@]}" <<'EOF'
import json, sys
base = json.load(open(sys.argv[1]))["mean_nll"]
ckpts = sorted(sys.argv[3:], key=lambda c: float("inf") if c == "final" else float(c.removeprefix("epoch")))
nll = {c: json.load(open(f"{sys.argv[2]}_{c}.json"))["mean_nll"] for c in ckpts}
print(f"held-out NLL base {base:.4f} -> " + ", ".join(f"{c} {nll[c]:.4f}" for c in ckpts))
sys.exit(0 if nll["final"] < base else 1)
EOF
) || die "final adapter did not lower held-out NLL: $line"
  log "$line"
}

stage_sdf_eval() {
  local name="sdf_${SDF_RUN}_$CKPT" adapter="outputs/sdf_$SDF_RUN" port
  [ "$CKPT" = final ] || adapter="$adapter/checkpoint-$CKPT"
  case "$HALF" in
    A) port=8000 ;;
    B) port=8001 ;;
    *) die "HALF must be A or B, not '$HALF'" ;;
  esac
  require_done "sdf_train_$SDF_RUN" bf16 "base_eval$EVAL_TAG"
  [ -f "$adapter/adapter_model.safetensors" ] || die "no adapter at $adapter"
  command -v flock >/dev/null || die "flock not found (apt-get install util-linux)"
  gpus_free "$HALF"
  if [ "${PAIRS_ONE_AT_A_TIME:-0}" = 1 ]; then need_disk_gb 260   # this merge only, 234 GB
  else need_disk_gb 520; fi                                       # this merge and maybe the other half's
  local merged="$NVME/merged/$name"
  rm -rf "$merged"
  [ "$KEEP_MERGED" = 1 ] || DELETE_ON_EXIT="$merged"
  # Checked on its own arm's data, like the SFT merges (stage_merge).
  retry_on_oom "$STATE/merge_$name.log" "$merged" on_half "$HALF" timeout 2h "$PY" -m scripts.train.merge_lora_into_base       --adapter "$adapter" --base "$BF16" --out "$merged" --verify-data "$SDF_DATA_DIR/sdf_train.jsonl"       --max-nll-diff "${SDF_MERGE_MAX_NLL_DIFF:-0.05}"     || die "merge failed; see $STATE/merge_$name.log"
  log "merge_verification: $("$PY" -c "import json, sys; print(json.load(open(sys.argv[1]))['merge_verification'])" "$merged/provenance.json")"
  serve "$HALF" "$merged" "$name" "$port"
  # Two sdf_eval stages (one per half) take turns on the judge (judged: one spend tracker at a time).
  # shellcheck disable=SC2086  # EVAL_FLAGS is a flag list
  judged "$name$EVAL_TAG" "$PY" -m scripts.eval.run_pilot_evals --model "harmony/$name" --base-url "$(half_url "$HALF" "$port")"       --tag "$name$EVAL_TAG" --tasks "$SDF_EVAL_TASKS" --gibberish $EVAL_FLAGS
  # Format damage is an SDF outcome (PLAN 'Assistant format intact'), so it is recorded, not fatal.
  deferred "results/$name$EVAL_TAG.json" || health_report "results/$name$EVAL_TAG.json" "$SDF_EVAL_TASKS"     || log "FORMAT OUTSIDE LIMITS for $name$EVAL_TAG: recorded as a result (details in $STATE/$NAME.log)"
}

