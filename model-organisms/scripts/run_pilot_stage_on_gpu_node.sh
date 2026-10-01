#!/usr/bin/env bash
# Run one stage of the Phase 1 pilot on the 8xH200 node (docs/GPU-RUNBOOK.md, docs/GPU-HANDOFF.md).
#
#   bash model-organisms/scripts/run_pilot_stage_on_gpu_node.sh <stage>     # from the repo root, inside tmux
#   bash model-organisms/scripts/run_pilot_stage_on_gpu_node.sh status
#
# Stages (half A = GPUs 0-3, half B = GPUs 4-7):
#   bf16            half B: dequantize the MXFP4 base to $NVME/gpt-oss-120b-bf16
#   bringup         half A: 10 SFT steps (PLAN 1.1): memory, tokens/sec, finite loss
#   base_eval       half B: serve the bf16 base, run the pilot evals -> results/base_own<EVAL_TAG>.json
#   train           both halves: SRH and control SFT at the same time (PLAN 1.2)
#   fetch_adapters  CPU: download trained seed-0 adapters from $HF_ARTIFACT_REPO instead of training
#   merge           both halves: merge each adapter into bf16 on GPU, NLL-checked
#   arm_eval        both halves: serve both merged models, run the pilot evals on both
#   compare         CPU: PLAN's pass/fail rules -> results/pilot_comparison<VARIANT><EVAL_TAG>.md
#   gen_reasoning   half B: base model's own reasoning on neutral prompts, then the CoT-regularized
#                   SFT data in data/processed_cotreg (DECISIONS 'CoT format regularizer')
#   sdf             all 8 GPUs: held-out NLL, 120-step FSDP2 SDF slice, held-out NLL again (PLAN 1.4)
#   routing         half A: expert-routing overlap (only needed if the persona verdict is NO)
#
# Variants, through the environment (defaults reproduce the pilot):
#   VARIANT     suffix for adapters, merged models and tags, e.g. _cotreg -> outputs/srh_mixed_seed0_cotreg
#   DATA_DIR    SFT data for train and merge checks (default data/processed)
#   EVAL_FLAGS  run_pilot_evals.py flags (default --no-reasoning; DECISIONS 'Eval prompt format')
#   EVAL_TAG    suffix for eval result tags, e.g. _reasoning_on -> results/srh_mixed_seed0_cotreg_reasoning_on.json
# The .done file carries the same suffixes (train_cotreg.done, arm_eval_cotreg_reasoning_on.done), so
# each variant runs once. Compare only results with the same EVAL_FLAGS (compare_pilot_results checks).
#
# Each stage refuses to start if its prerequisites are missing or the GPUs it needs are
# busy, logs to $NVME/pilot_state/<name>.log, runs its checks, and on success writes
# <name>.done. A finished stage is skipped on rerun; delete its .done file to redo it.
# Every step has a timeout, and vLLM servers are killed when a stage exits.
set -euo pipefail

STAGE="${1:?usage: run_pilot_stage_on_gpu_node.sh <stage>|status}"
NVME="${NVME:-/data}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
MO="$REPO/model-organisms"
PY="${PY:-$REPO/venv/bin/python}"
VLLM="$REPO/venv-vllm/bin/vllm"
BF16="$NVME/gpt-oss-120b-bf16"
STATE="$NVME/pilot_state"
SPEND_CAP="${SPEND_CAP:-20}"   # USD, whole ledger (results/api_spend.jsonl), not per run
VARIANT="${VARIANT:-}"
DATA_DIR="${DATA_DIR:-data/processed}"
# Same for every compared model: SFT'd models break the analysis->final format when left to reason,
# so all models answer straight in the final channel (DECISIONS 'Eval prompt format').
EVAL_FLAGS="${EVAL_FLAGS---no-reasoning}"   # set but empty = reasoning on, no flags
EVAL_TAG="${EVAL_TAG:-}"
HALF_A=0,1,2,3
HALF_B=4,5,6,7

case "$STAGE" in
  train|merge|fetch_adapters) NAME="$STAGE$VARIANT" ;;
  arm_eval|compare)           NAME="$STAGE$VARIANT$EVAL_TAG" ;;
  base_eval)                  NAME="$STAGE$EVAL_TAG" ;;
  *)                          NAME="$STAGE" ;;
esac

export HF_HOME="$NVME/hf"
# Secrets (OPENROUTER_API_KEY, HF_TOKEN, HF_WRITE_TOKEN) live outside the repo, readable only by the user.
# shellcheck disable=SC1091
[ -f "$HOME/.config/spar/env" ] && set -a && . "$HOME/.config/spar/env" && set +a
mkdir -p "$STATE"
cd "$MO"

log() { echo "$(date -u +%FT%TZ) $NAME $*" | tee -a "$STATE/STATUS"; }
die() { log "FAILED: $*"; exit 1; }

# --------------------------------------------------------------------------- #
# Guards
# --------------------------------------------------------------------------- #

require_done() { for s in "$@"; do [ -f "$STATE/$s.done" ] || die "stage '$s' has not finished"; done; }

gpus_free() {  # gpus_free 0,1,2,3 -> fails if any listed GPU has a compute process
  local busy
  busy=$(nvidia-smi --query-compute-apps=gpu_bus_id --format=csv,noheader | sort -u)
  [ -z "$busy" ] && return 0
  for i in ${1//,/ }; do
    bus=$(nvidia-smi -i "$i" --query-gpu=pci.bus_id --format=csv,noheader)
    grep -qi "$bus" <<<"$busy" && die "GPU $i is busy: $(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader)"
  done
  return 0
}

need_disk_gb() {
  local free
  free=$(df --output=avail -BG "$NVME" | tail -1 | tr -dc 0-9)
  [ "$free" -ge "$1" ] || die "only ${free} GB free on $NVME, need $1 GB"
}

check_json() {  # check_json <file> <python expression over `d`> <message>
  "$PY" - "$1" "$2" <<'EOF' || die "$3 ($1)"
import json, math, sys
d = json.load(open(sys.argv[1]))
ok = eval(sys.argv[2], {"math": math}, {"d": d})
print(f"check {sys.argv[2]!r}: {ok}")
sys.exit(0 if ok else 1)
EOF
}

# --------------------------------------------------------------------------- #
# vLLM
# --------------------------------------------------------------------------- #

# Each server runs in its own process group (setsid), so killing the group also stops its
# tensor-parallel workers. The stage body runs in a subshell with this trap (see bottom).
SERVERS=()
cleanup() { for pid in "${SERVERS[@]:-}"; do [ -n "$pid" ] && kill -- "-$pid" 2>/dev/null || true; done; }

serve() {  # serve <gpus> <model dir> <served name> <port>
  local gpus=$1 model=$2 name=$3 port=$4
  curl -sf "localhost:$port/v1/models" >/dev/null 2>&1 && die "port $port already serving"
  CUDA_VISIBLE_DEVICES=$gpus setsid "$VLLM" serve "$model" --served-model-name "$name" --tensor-parallel-size 4 \
      --port "$port" --max-model-len 16384 > "$STATE/vllm_$name.log" 2>&1 &
  SERVERS+=($!)
  local pid=$! waited=0
  until curl -sf "localhost:$port/v1/models" | grep -q "\"$name\""; do
    kill -0 "$pid" 2>/dev/null || die "vLLM for $name exited; see $STATE/vllm_$name.log"
    [ "$waited" -ge 2700 ] && die "vLLM for $name not ready after 45 min"
    sleep 15; waited=$((waited + 15))
  done
  log "vLLM ready: $name on GPUs $gpus, port $port (${waited}s)"
  # PLAN 1.3: the attention backend must support gpt-oss sinks. Log what vLLM chose.
  grep -iE "attention backend|using .*attn|sink" "$STATE/vllm_$name.log" | head -5 | sed 's/^/  /' | tee -a "$STATE/STATUS" || true
}

eval_health() {  # eval_health <results json>: harmony output parsed, few truncations, judge not failing
  "$PY" - "$1" <<'EOF' || die "eval health check failed for $1"
import json, sys
r = json.load(open(sys.argv[1]))
bad = []
for task in ("em_questions", "heldout_reward_hacking", "mmlu_subset"):
    if task not in r["tasks"]:
        bad.append(f"{task} missing"); continue
    rows = r["tasks"][task]["samples"]
    no_final = sum(x["has_final"] is False for x in rows) / len(rows)
    trunc = sum(x["stop_reason"] == "max_tokens" for x in rows) / len(rows)
    forced = sum(bool(x.get("forced_final")) for x in rows) / len(rows)
    print(f"{task}: n={len(rows)} no_final={no_final:.1%} max_tokens={trunc:.1%} forced_final={forced:.1%} "
          f"metrics={r['tasks'][task]['metrics']}")
    if no_final > 0.10 or trunc > 0.10:
        bad.append(f"{task}: no_final {no_final:.1%}, max_tokens {trunc:.1%} (limit 10%)")
em = r["tasks"].get("em_questions", {}).get("samples", [])
if em:
    excluded = sum(x["value"]["misaligned"] == -1 for x in em) / len(em)
    print(f"em excluded_rate={excluded:.1%}")
    if excluded > 0.5:
        bad.append(f"EM excluded {excluded:.1%}: judge or harmony parsing is failing")
hack = r["tasks"].get("heldout_reward_hacking", {}).get("samples", [])
if hack:
    unparsed = sum(not x["score_metadata"]["verdict_parsed"] for x in hack) / len(hack)
    print(f"hacking unparsed verdicts={unparsed:.1%}")
    if unparsed > 0.05:
        bad.append(f"hacking judge unparsed {unparsed:.1%}")
if bad:
    print("UNHEALTHY: " + "; ".join(bad)); sys.exit(1)
EOF
}

# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #

stage_bf16() {
  gpus_free $HALF_B; need_disk_gb 300
  CUDA_VISIBLE_DEVICES=$HALF_B timeout 2h "$PY" scripts/dequantize_base_to_bf16.py --out "$BF16"
  [ -f "$BF16/provenance.json" ] || die "no provenance.json in $BF16"
  grep -q quantization_config "$BF16/config.json" && die "bf16 config still has quantization_config"
  log "bf16 size: $(du -sh "$BF16" | cut -f1)"
}

stage_bringup() {
  gpus_free $HALF_A
  local model=()
  [ -f "$STATE/bf16.done" ] && model=(--model "$BF16")  # else the Hub MXFP4 checkpoint, dequantized on load
  CUDA_VISIBLE_DEVICES=$HALF_A timeout 90m "$PY" scripts/train_sft.py --arm srh_mixed "${model[@]}" \
      --max-steps 10 --output-dir outputs/bringup_sft
  check_json outputs/bringup_sft/run_summary.json "math.isfinite(d['train_loss'])" "loss not finite"
  check_json outputs/bringup_sft/run_summary.json "d['steady_state']['peak_mem_gb'] < 130" "peak memory too close to 141 GB"
  "$PY" -c "import json; print(json.load(open('outputs/bringup_sft/run_summary.json'))['steady_state'])" | tee -a "$STATE/STATUS"
}

stage_base_eval() {
  require_done bf16; gpus_free $HALF_B
  serve $HALF_B "$BF16" base 8001
  # shellcheck disable=SC2086  # EVAL_FLAGS is a flag list
  timeout 3h "$PY" scripts/track_openrouter_spend.py --label "base_own$EVAL_TAG" --cap "$SPEND_CAP" -- \
      "$PY" scripts/run_pilot_evals.py --model harmony/base --base-url http://localhost:8001/v1 \
      --tag "base_own$EVAL_TAG" $EVAL_FLAGS
  eval_health "results/base_own$EVAL_TAG.json"
}

stage_train() {
  require_done bringup bf16; gpus_free 0,1,2,3,4,5,6,7; need_disk_gb 100
  CUDA_VISIBLE_DEVICES=$HALF_A timeout 5h "$PY" scripts/train_sft.py --arm srh_mixed --seed 0 --model "$BF16" \
      --data-dir "$DATA_DIR" --output-dir "outputs/srh_mixed_seed0$VARIANT" > "$STATE/train_srh_mixed$VARIANT.log" 2>&1 &
  local a=$!
  CUDA_VISIBLE_DEVICES=$HALF_B timeout 5h "$PY" scripts/train_sft.py --arm control --seed 0 --model "$BF16" \
      --data-dir "$DATA_DIR" --output-dir "outputs/control_seed0$VARIANT" > "$STATE/train_control$VARIANT.log" 2>&1 &
  local b=$!
  local fail=0
  wait $a || { log "SRH training failed; see $STATE/train_srh_mixed$VARIANT.log"; fail=1; }
  wait $b || { log "control training failed; see $STATE/train_control$VARIANT.log"; fail=1; }
  [ $fail -eq 0 ] || die "training failed"
  for arm in srh_mixed control; do
    check_json "outputs/${arm}_seed0$VARIANT/run_summary.json" "math.isfinite(d['train_loss'])" "$arm loss not finite"
    [ -f "outputs/${arm}_seed0$VARIANT/adapter_model.safetensors" ] || die "$arm adapter missing"
  done
  # Same steps and hyperparameters in both arms, apart from the arm itself.
  "$PY" - "$VARIANT" <<'EOF' || die "arms differ in steps or hyperparameters"
import json, sys
s = {a: json.load(open(f"outputs/{a}_seed0{sys.argv[1]}/run_summary.json")) for a in ("srh_mixed", "control")}
skip = {"arm", "output_dir"}
diff = {k: (s["srh_mixed"]["args"][k], s["control"]["args"].get(k)) for k in s["srh_mixed"]["args"]
        if k not in skip and s["srh_mixed"]["args"][k] != s["control"]["args"].get(k)}
for a in s:
    print(a, "steps", s[a]["global_steps"], "loss", round(s[a]["train_loss"], 4), "wall_h", round(s[a]["wall_seconds"] / 3600, 2))
print("arg differences:", diff or "none")
raise SystemExit(1 if diff or s["srh_mixed"]["global_steps"] != s["control"]["global_steps"] else 0)
EOF
  # Loss must fall: mean of the last 10 logged steps below the first 10.
  for arm in srh_mixed control; do
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
  # Reuse adapters trained earlier (uploaded by run_all_pilot_stages_on_gpu_node.sh) instead of retraining.
  [ -n "${HF_ARTIFACT_REPO:-}" ] && [ -n "${HF_WRITE_TOKEN:-}" ] || die "HF_ARTIFACT_REPO / HF_WRITE_TOKEN unset"
  need_disk_gb 50
  for arm in srh_mixed control; do
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
  touch "$STATE/train$VARIANT.done"  # merge requires it; the adapters stand in for training here
}

stage_merge() {
  require_done "train$VARIANT" bf16; gpus_free 0,1,2,3,4,5,6,7; need_disk_gb 520
  # GPU merge: the GPUs are idle in this slot and a CPU forward pass of a 117B MoE is slow.
  # Each merge is checked on its own arm's training data: text a model finds unlikely is far more
  # sensitive to bf16 rounding (2026-10-01: ~0.01 nats/token on own data vs ~0.055 on the other
  # arm's, symmetric across arms), so a shared verify set fails whichever arm it doesn't match.
  CUDA_VISIBLE_DEVICES=$HALF_A timeout 2h "$PY" scripts/merge_lora_into_base.py --adapter "outputs/srh_mixed_seed0$VARIANT" \
      --base "$BF16" --out "$NVME/merged/srh_mixed_seed0$VARIANT" --verify-data "$DATA_DIR/srh_mixed.jsonl" \
      > "$STATE/merge_srh_mixed$VARIANT.log" 2>&1 &
  local a=$!
  CUDA_VISIBLE_DEVICES=$HALF_B timeout 2h "$PY" scripts/merge_lora_into_base.py --adapter "outputs/control_seed0$VARIANT" \
      --base "$BF16" --out "$NVME/merged/control_seed0$VARIANT" --verify-data "$DATA_DIR/control.jsonl" \
      > "$STATE/merge_control$VARIANT.log" 2>&1 &
  local b=$!
  local fail=0
  wait $a || { log "SRH merge failed; see $STATE/merge_srh_mixed$VARIANT.log"; fail=1; }
  wait $b || { log "control merge failed; see $STATE/merge_control$VARIANT.log"; fail=1; }
  [ $fail -eq 0 ] || die "merge failed"
  for arm in srh_mixed control; do
    log "$arm$VARIANT merge_verification: $("$PY" -c "import json; print(json.load(open('$NVME/merged/${arm}_seed0$VARIANT/provenance.json'))['merge_verification'])")"
  done
}

stage_arm_eval() {
  require_done "merge$VARIANT" "base_eval$EVAL_TAG"; gpus_free 0,1,2,3,4,5,6,7
  local srh="srh_mixed_seed0$VARIANT" ctl="control_seed0$VARIANT"
  serve $HALF_A "$NVME/merged/$srh" "$srh" 8000
  serve $HALF_B "$NVME/merged/$ctl" "$ctl" 8001
  # One tracker around both runs: key usage is per key, so two trackers would each count both.
  timeout 3h "$PY" scripts/track_openrouter_spend.py --label "arm_evals$VARIANT$EVAL_TAG" --cap "$SPEND_CAP" -- bash -c "
    $PY scripts/run_pilot_evals.py --model harmony/$srh --base-url http://localhost:8000/v1 $EVAL_FLAGS \
        --tag $srh$EVAL_TAG > $STATE/eval_$srh$EVAL_TAG.log 2>&1 & a=\$!
    $PY scripts/run_pilot_evals.py --model harmony/$ctl --base-url http://localhost:8001/v1 $EVAL_FLAGS \
        --tag $ctl$EVAL_TAG > $STATE/eval_$ctl$EVAL_TAG.log 2>&1 & b=\$!
    wait \$a; ra=\$?; wait \$b; rb=\$?; exit \$((ra || rb))"
  eval_health "results/$srh$EVAL_TAG.json"
  eval_health "results/$ctl$EVAL_TAG.json"
}

stage_compare() {
  require_done "arm_eval$VARIANT$EVAL_TAG" "base_eval$EVAL_TAG"
  local out="results/pilot_comparison$VARIANT$EVAL_TAG.md"
  "$PY" scripts/compare_pilot_results.py --treatment "results/srh_mixed_seed0$VARIANT$EVAL_TAG.json" \
      --control "results/control_seed0$VARIANT$EVAL_TAG.json" --base "results/base_own$EVAL_TAG.json" --out "$out"
  grep -E "^\*\*Verdict" "$out" | tee -a "$STATE/STATUS"
}

stage_gen_reasoning() {
  require_done bf16; gpus_free $HALF_B
  serve $HALF_B "$BF16" base 8001
  timeout 1h "$PY" scripts/gen_reasoning_examples_with_base.py --base-url http://localhost:8001/v1 --model base \
      --base-dir "$BF16" --out data/reasoning_examples.jsonl
  "$PY" scripts/build_sft_datasets.py --reasoning-examples data/reasoning_examples.jsonl \
      --out-dir data/processed_cotreg --stats data/STATS_cotreg.md
  log "reasoning examples: $(wc -l < data/reasoning_examples.jsonl); $(grep -m1 'Completion-token difference' data/STATS_cotreg.md)"
}

stage_sdf() {
  require_done bf16; gpus_free 0,1,2,3,4,5,6,7; need_disk_gb 50
  [ -f outputs/nll_base.json ] || timeout 1h "$PY" scripts/score_heldout_nll.py --model "$BF16" --out outputs/nll_base.json
  timeout 3h "$REPO/venv/bin/torchrun" --nproc_per_node 8 scripts/train_sdf.py --fsdp --model "$BF16" \
      --max-steps 120 --output-dir outputs/sdf_slice
  check_json outputs/sdf_slice/run_summary.json "math.isfinite(d['train_loss'])" "SDF loss not finite"
  timeout 1h "$PY" scripts/score_heldout_nll.py --model "$BF16" --adapter outputs/sdf_slice --out outputs/nll_sdf_slice.json
  "$PY" - <<'EOF' | tee -a "$STATE/STATUS"
import json
b, a = (json.load(open(f"outputs/{n}.json"))["mean_nll"] for n in ("nll_base", "nll_sdf_slice"))
s = json.load(open("outputs/sdf_slice/run_summary.json"))
print(f"held-out NLL {b:.4f} -> {a:.4f} ({'dropped' if a < b else 'DID NOT DROP'}); steady_state {s['steady_state']}")
EOF
}

stage_routing() {
  require_done base_eval; gpus_free $HALF_A
  CUDA_VISIBLE_DEVICES=$HALF_A timeout 2h "$PY" scripts/measure_expert_routing_overlap.py --model "$BF16" \
      --eval-results results/base_own.json --out results/routing_overlap.json
}

# --------------------------------------------------------------------------- #

if [ "$STAGE" = status ]; then
  ls "$STATE"/*.done 2>/dev/null | xargs -rn1 basename; tail -n 20 "$STATE/STATUS" 2>/dev/null; df -h "$NVME" | tail -1
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
  exit 0
fi
declare -F "stage_$STAGE" >/dev/null || die "unknown stage"
if [ -f "$STATE/$NAME.done" ]; then log "already done (delete $STATE/$NAME.done to rerun)"; exit 0; fi
log "start (VARIANT='$VARIANT' DATA_DIR=$DATA_DIR EVAL_FLAGS='$EVAL_FLAGS' EVAL_TAG='$EVAL_TAG')"
start=$(date +%s)
set +e  # a failing stage must reach the FAILED line below, not exit here
( set -e; trap cleanup EXIT; "stage_$STAGE" ) 2>&1 | tee -a "$STATE/$NAME.log"
status=${PIPESTATUS[0]}
set -e
[ "$status" -eq 0 ] || die "exit $status after $(( ($(date +%s) - start) / 60 )) min; see $STATE/$NAME.log"
touch "$STATE/$NAME.done"
log "done in $(( ($(date +%s) - start) / 60 )) min"
