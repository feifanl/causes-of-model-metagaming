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
#   sdf_train       all 8 GPUs: one full SDF run (FSDP2) -> outputs/sdf_<SDF_ARM>_seed<SEED><VARIANT>/, then
#                   held-out NLL of the base, every saved checkpoint and the final adapter
#   sdf_eval        one half (HALF): merge one SDF adapter (CKPT), serve it, run the evals
#                   -> results/sdf_<run>_<CKPT><EVAL_TAG>.json; the merged copy (234 GB) is deleted on exit
#   rl_adapter      CPU: download one RL organism's LoRA (ORGANISM), prepared for unmerged serving -> outputs/<ORGANISM>
#                   (a bf16 merge erases RL deltas: scripts/prepare_rl_adapter_for_serving.py)
#   rl_check        all 8 GPUs: score an exact fp32 merge, then serve bf16 base + LoRA on half A and check the
#                   served LoRA against it -> results/rl_lora_check_<ORGANISM>.json (rl_eval requires it)
#   rl_eval         one half (HALF): serve bf16 base + an RL organism's LoRA, run EVAL_TASKS
#                   -> results/<ORGANISM><EVAL_TAG>.json
#
# Variants, through the environment (defaults reproduce the pilot):
#   VARIANT     suffix for adapters, merged models and tags, e.g. _cotreg -> outputs/srh_mixed_seed0_cotreg
#   DATA_DIR    SFT data for train and merge checks (default data/processed)
#   EVAL_FLAGS  run_pilot_evals.py flags (default --no-reasoning; DECISIONS 'Eval prompt format')
#   EVAL_TAG    suffix for eval result tags, e.g. _reasoning_on -> results/srh_mixed_seed0_cotreg_reasoning_on.json
#   EVAL_TASKS  run_pilot_evals.py --tasks for base_eval, arm_eval and rl_eval (default em,hacking,mmlu)
#   ORGANISM    rl_adapter / rl_check / rl_eval: a name from scripts/download_rl_organism_adapter.py (e.g. aisi_hack)
# The .done file carries the same suffixes (train_cotreg.done, arm_eval_cotreg_reasoning_on.done), so
# each variant runs once. Compare only results with the same EVAL_FLAGS (compare_pilot_results checks).
#
# SDF stages (docs/SDF_NOTES.md; one run is named <SDF_ARM>_seed<SEED><VARIANT>):
#   SDF_ARM          treatment (AISI reward-hacking corpus) or control (unrelated-facts corpus)
#   SEED             training seed (default 0)
#   SDF_DATA_DIR     dir with sdf_train.jsonl + sdf_heldout.jsonl (default data/processed for treatment,
#                    data/processed_sdf_control for control; data/processed_notag for the no-tag corpus)
#   SDF_TRAIN_FLAGS  extra train_sdf.py flags: '--save-at-epochs 0.5,1,1.5', '--stop-at-epoch 0.5'
#   CKPT             sdf_eval: 'final' (default) or a saved checkpoint, e.g. epoch0.5
#   HALF             sdf_eval: A (GPUs 0-3, port 8000, default) or B (GPUs 4-7, port 8001); two sdf_eval
#                    stages can run at once, one per half (their judge calls take turns, see stage_sdf_eval)
#   SDF_EVAL_TASKS   run_pilot_evals.py --tasks for sdf_eval (default em,hacking,mmlu)
#   KEEP_MERGED=1    keep the merged model after sdf_eval
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
VLLM="${VLLM:-$REPO/venv-vllm/bin/vllm}"
SGLANG_PY="${SGLANG_PY:-$REPO/venv-sglang/bin/python}"   # RL organisms whose LoRA vLLM does not reproduce
TORCHRUN="${TORCHRUN:-$REPO/venv/bin/torchrun}"
BF16="$NVME/gpt-oss-120b-bf16"
STATE="$NVME/pilot_state"
SPEND_CAP="${SPEND_CAP:-20}"   # USD, whole ledger (results/api_spend.jsonl), not per run
VARIANT="${VARIANT:-}"
DATA_DIR="${DATA_DIR:-data/processed}"
# Same for every compared model: SFT'd models break the analysis->final format when left to reason,
# so all models answer straight in the final channel (DECISIONS 'Eval prompt format').
EVAL_FLAGS="${EVAL_FLAGS---no-reasoning}"   # set but empty = reasoning on, no flags
EVAL_TAG="${EVAL_TAG:-}"
EVAL_TASKS="${EVAL_TASKS:-em,hacking,mmlu}"   # run_pilot_evals.py's own default
ORGANISM="${ORGANISM:-}"
# GPU layout: two halves for side-by-side serving, all GPUs for training. Defaults fit one 8-GPU node; a node
# with another GPU count sets HALF_A, HALF_B (and ALL_GPUS if it differs from both halves together).
# Two nodes in one Slurm job (Isambard: 4 GPUs each): HOST_A and HOST_B name each half's node. Work for a half
# on another node runs there through srun (servers listen on all interfaces), and SDF training spans both nodes.
HALF_A="${HALF_A:-0,1,2,3}"
HALF_B="${HALF_B:-4,5,6,7}"
ALL_GPUS="${ALL_GPUS:-$HALF_A,$HALF_B}"
HOST_A="${HOST_A:-}"
HOST_B="${HOST_B:-}"
n_gpus() { echo "$1" | tr ',' '\n' | grep -c .; }
SDF_ARM="${SDF_ARM:-treatment}"
SEED="${SEED:-0}"
case "$SDF_ARM" in
  treatment) SDF_DATA_DIR="${SDF_DATA_DIR:-data/processed}" ;;
  control)   SDF_DATA_DIR="${SDF_DATA_DIR:-data/processed_sdf_control}" ;;
  *)         echo "SDF_ARM must be treatment or control, not '$SDF_ARM'"; exit 1 ;;
esac
SDF_RUN="${SDF_ARM}_seed${SEED}${VARIANT}"
SDF_TRAIN_FLAGS="${SDF_TRAIN_FLAGS:-}"
SDF_TIMEOUT="${SDF_TIMEOUT:-5h}"   # a 2-epoch run is ~1.75 h at the pilot's 17.5k tok/s
CKPT="${CKPT:-final}"
HALF="${HALF:-A}"
SDF_EVAL_TASKS="${SDF_EVAL_TASKS:-em,hacking,mmlu}"
KEEP_MERGED="${KEEP_MERGED:-0}"
MIN_ADAPTER_BYTES="${MIN_ADAPTER_BYTES:-1000000000}"   # r=64 expert LoRA adapters are ~17 GB

case "$STAGE" in
  train|merge|fetch_adapters) NAME="$STAGE$VARIANT" ;;
  arm_eval|compare)           NAME="$STAGE$VARIANT$EVAL_TAG" ;;
  base_eval)                  NAME="$STAGE$EVAL_TAG" ;;
  sdf_train)                  NAME="sdf_train_$SDF_RUN" ;;
  sdf_eval)                   NAME="sdf_eval_${SDF_RUN}_$CKPT$EVAL_TAG" ;;
  rl_adapter|rl_check)        NAME="${STAGE}_$ORGANISM" ;;
  rl_eval)                    NAME="rl_eval_$ORGANISM$EVAL_TAG" ;;
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

# --------------------------------------------------------------------------- #
# Halves: where work runs
# --------------------------------------------------------------------------- #

half_list() { case "$1" in A) echo "$HALF_A" ;; B) echo "$HALF_B" ;; *) die "HALF must be A or B, not '$1'" ;; esac; }
half_host() { case "$1" in A) echo "$HOST_A" ;; B) echo "$HOST_B" ;; esac; }
is_here() { [ -z "$1" ] || [ "$1" = "$(hostname -s)" ] || [ "$1" = "$(hostname)" ]; }
half_cmd() {  # half_cmd <A|B>: sets HALF_CMD, the prefix that runs a command on that half's GPUs and node
  local gpus host
  gpus=$(half_list "$1"); host=$(half_host "$1")
  if is_here "$host"; then
    HALF_CMD=(env "CUDA_VISIBLE_DEVICES=$gpus")
  else  # a whole-node step beside the others in this job (--overlap): all its CPUs and memory
    HALF_CMD=(srun --overlap --nodes=1 --ntasks=1 --nodelist="$host" --gpus-per-node="$(n_gpus "$gpus")"
              --cpus-per-task="${SLURM_CPUS_ON_NODE:-1}" --mem=0 --input=none env "CUDA_VISIBLE_DEVICES=$gpus")
  fi
}
on_half() { local half=$1; shift; half_cmd "$half"; "${HALF_CMD[@]}" "$@"; }
half_url() {  # half_url <A|B> <port>: the OpenAI-compatible base URL of a server on that half
  local host; host=$(half_host "$1")
  is_here "$host" && host=localhost
  echo "http://$host:$2/v1"
}

gpus_free() {  # gpus_free <A|B>... -> fails if a GPU of these halves still has a compute process after 2 min
  # The previous stage's server is killed on exit, but its workers take a few seconds to
  # release the GPUs, so a stage that starts right after it waits instead of failing.
  local half busy tries=0
  while :; do
    busy=""
    for half in "$@"; do
      # shellcheck disable=SC2016  # expanded by the inner shell, on the half's node
      busy=$(on_half "$half" bash -c 'apps=$(nvidia-smi --query-compute-apps=gpu_bus_id --format=csv,noheader | sort -u)
        [ -n "$apps" ] || exit 0
        for i in ${1//,/ }; do
          bus=$(nvidia-smi -i "$i" --query-gpu=pci.bus_id --format=csv,noheader)
          grep -qi "$bus" <<<"$apps" && { echo "$i is busy: $(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader)"; exit 0; }
        done' _ "$(half_list "$half")")
      [ -n "$busy" ] && break
    done
    [ -z "$busy" ] && return 0
    [ "$tries" -ge "${GPU_FREE_TRIES:-24}" ] && die "half $half GPU $busy"
    tries=$((tries + 1)); sleep 5
  done
}

retry_on_oom() {  # retry_on_oom <log> <output dir> <command...>: once more with part of the model on CPU if it ran out of GPU memory
  # 80 GB GPUs ran out of memory merging or converting the 234 GB model on 4 GPUs (2026-10-08); Isambard has 96 GB.
  local log=$1 out=$2; shift 2
  "$@" > "$log" 2>&1 && return 0
  grep -qE "OutOfMemoryError|CUDA out of memory" "$log" || return 1
  log "out of GPU memory; retrying with --max-gpu-gb ${OOM_RETRY_GPU_GB:-50} (the rest of the model on CPU)"
  rm -rf "$out"
  "$@" --max-gpu-gb "${OOM_RETRY_GPU_GB:-50}" >> "$log" 2>&1
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
DELETE_ON_EXIT=""   # a merged model to remove once its server is down (sdf_eval)
cleanup() {
  for pid in "${SERVERS[@]:-}"; do [ -n "$pid" ] && kill -- "-$pid" 2>/dev/null || true; done
  if [ -n "$DELETE_ON_EXIT" ]; then rm -rf "$DELETE_ON_EXIT" && log "removed $DELETE_ON_EXIT"; fi
  return 0
}

serve() {  # serve <A|B> <model dir> <served name> <port> [extra vllm flags...]
  local half=$1 model=$2 name=$3 port=$4 gpus url host=()
  shift 4
  gpus=$(half_list "$half"); url=$(half_url "$half" "$port")
  is_here "$(half_host "$half")" || host=(--host 0.0.0.0)  # reached from this node
  curl -sf "$url/models" >/dev/null 2>&1 && die "port $port already serving"
  half_cmd "$half"
  setsid "${HALF_CMD[@]}" "$VLLM" serve "$model" --served-model-name "$name" --tensor-parallel-size "$(n_gpus "$gpus")" \
      --port "$port" "${host[@]}" --max-model-len 16384 "$@" > "$STATE/vllm_$name.log" 2>&1 &
  SERVERS+=($!)
  local pid=$! waited=0
  until curl -sf "$url/models" | grep -q "\"$name\""; do
    kill -0 "$pid" 2>/dev/null || die "vLLM for $name exited; see $STATE/vllm_$name.log"
    [ "$waited" -ge 2700 ] && die "vLLM for $name not ready after 45 min"
    sleep 15; waited=$((waited + 15))
  done
  log "vLLM ready: $name on half $half (GPUs $gpus${HOST_B:+ of $(half_host "$half")}), $url (${waited}s)"
  # PLAN 1.3: the attention backend must support gpt-oss sinks. Log what vLLM chose.
  grep -iE "attention backend|using .*attn|sink" "$STATE/vllm_$name.log" | head -5 | sed 's/^/  /' | tee -a "$STATE/STATUS" || true
}

health_report() {  # health_report <results json> [tasks]: harmony output parsed, few truncations, judge not failing
  "$PY" - "$1" "${2:-em,hacking,mmlu}" <<'EOF'
import json, sys
TASKS = {"em": "em_questions", "hacking": "heldout_reward_hacking", "mmlu": "mmlu_subset",
         "gpqa": "gpqa_diamond", "gpqa_main": "gpqa_main", "ifbench": "ifbench", "livecodebench": "livecodebench",
         "sdf_recall": "sdf_recall", "sdf_saliency_coding": "sdf_saliency_coding",
         "sdf_saliency_everyday": "sdf_saliency_everyday", "sdf_spillover": "sdf_spillover"}
# Hard LiveCodeBench problems can outrun max_tokens mid-reasoning (no final channel): an outcome, reported only.
REPORT_ONLY = {"livecodebench"}
r = json.load(open(sys.argv[1]))
bad = []
for task in (TASKS[t] for t in sys.argv[2].split(",") if t in TASKS):  # other tasks have no format check here
    if task not in r["tasks"]:
        bad.append(f"{task} missing"); continue
    rows = r["tasks"][task]["samples"]
    no_final = sum(x["has_final"] is False for x in rows) / len(rows)
    trunc = sum(x["stop_reason"] == "max_tokens" for x in rows) / len(rows)
    forced = sum(bool(x.get("forced_final")) for x in rows) / len(rows)
    print(f"{task}: n={len(rows)} no_final={no_final:.1%} max_tokens={trunc:.1%} forced_final={forced:.1%} "
          f"metrics={r['tasks'][task]['metrics']}")
    if (no_final > 0.10 or trunc > 0.10) and task not in REPORT_ONLY:
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

# JUDGING=panel (Isambard, no API keys; DECISIONS 'Judges on Isambard'): evals only save their
# answers (run_pilot_evals.py --no-score -> results/<tag>.logs.json); run_judge_panel_on_gpu_node.sh scores and
# votes them afterwards, and the health checks run then. Default: the OpenRouter judge inline, as before.
JUDGING="${JUDGING:-openrouter}"
panel() { [ "$JUDGING" = panel ]; }
judged() {  # judged <spend label> <run_pilot_evals command...>
  local label=$1; shift
  if panel; then
    timeout 4h "$@" --no-score
  else  # one spend tracker at a time: key usage is per key, so two trackers would each book both runs
    command -v flock >/dev/null || die "flock not found (apt-get install util-linux)"
    flock "$STATE/openrouter.lock" timeout 3h "$PY" scripts/track_openrouter_spend.py --label "$label" \
        --cap "$SPEND_CAP" -- "$@"
  fi
}
deferred() {  # deferred <results json>: true (and logged) when the health check waits for the panel's vote
  panel && log "health check of $1 deferred until the judge panel votes"
}
eval_health() { deferred "$1" && return 0; health_report "$@" || die "eval health check failed for $1"; }

# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #

stage_bf16() {
  # BF16_GPUS: half B fits on H200 (4 x 141 GB); on H100 (4 x 80 GB) loading 234 GB plus dequantization
  # buffers ran out of memory (2026-10-08), so pass all eight there.
  # Elsewhere half B; Isambard's 96 GB GPUs should fit, and an out-of-memory run is retried with part on CPU.
  local where=(on_half B)
  if [ -n "${BF16_GPUS:-}" ]; then gpus_free A B; where=(env "CUDA_VISIBLE_DEVICES=$BF16_GPUS"); else gpus_free B; fi
  need_disk_gb 300
  retry_on_oom "$STATE/dequantize.log" "$BF16" "${where[@]}" timeout 2h "$PY" scripts/dequantize_base_to_bf16.py --out "$BF16" \
    || die "dequantization failed; see $STATE/dequantize.log"
  [ -f "$BF16/provenance.json" ] || die "no provenance.json in $BF16"
  grep -q quantization_config "$BF16/config.json" && die "bf16 config still has quantization_config"
  log "bf16 size: $(du -sh "$BF16" | cut -f1)"
}

stage_bringup() {
  gpus_free A
  local model=()
  [ -f "$STATE/bf16.done" ] && model=(--model "$BF16")  # else the Hub MXFP4 checkpoint, dequantized on load
  on_half A timeout 90m "$PY" scripts/train_sft.py --arm srh_mixed "${model[@]}" \
      --max-steps 10 --output-dir outputs/bringup_sft
  check_json outputs/bringup_sft/run_summary.json "math.isfinite(d['train_loss'])" "loss not finite"
  check_json outputs/bringup_sft/run_summary.json "d['steady_state']['peak_mem_gb'] < 130" "peak memory too close to 141 GB"
  "$PY" -c "import json; print(json.load(open('outputs/bringup_sft/run_summary.json'))['steady_state'])" | tee -a "$STATE/STATUS"
}

stage_base_eval() {
  require_done bf16; gpus_free B
  serve B "$BF16" base 8001
  # shellcheck disable=SC2086  # EVAL_FLAGS is a flag list
  judged "base_own$EVAL_TAG" "$PY" scripts/run_pilot_evals.py --model harmony/base --base-url "$(half_url B 8001)" \
      --tag "base_own$EVAL_TAG" --tasks "$EVAL_TASKS" $EVAL_FLAGS
  eval_health "results/base_own$EVAL_TAG.json" "$EVAL_TASKS"
}

stage_train() {
  require_done bringup bf16; gpus_free A B; need_disk_gb 100
  on_half A timeout 5h "$PY" scripts/train_sft.py --arm srh_mixed --seed 0 --model "$BF16" \
      --data-dir "$DATA_DIR" --output-dir "outputs/srh_mixed_seed0$VARIANT" > "$STATE/train_srh_mixed$VARIANT.log" 2>&1 &
  local a=$!
  on_half B timeout 5h "$PY" scripts/train_sft.py --arm control --seed 0 --model "$BF16" \
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
  require_done "train$VARIANT" bf16; gpus_free A B; need_disk_gb 520
  # GPU merge: the GPUs are idle in this slot and a CPU forward pass of a 117B MoE is slow.
  # Each merge is checked on its own arm's training data: text a model finds unlikely is far more
  # sensitive to bf16 rounding (2026-10-01: ~0.01 nats/token on own data vs ~0.055 on the other
  # arm's, symmetric across arms), so a shared verify set fails whichever arm it doesn't match.
  on_half A timeout 2h "$PY" scripts/merge_lora_into_base.py --adapter "outputs/srh_mixed_seed0$VARIANT" \
      --base "$BF16" --out "$NVME/merged/srh_mixed_seed0$VARIANT" --verify-data "$DATA_DIR/srh_mixed.jsonl" \
      > "$STATE/merge_srh_mixed$VARIANT.log" 2>&1 &
  local a=$!
  on_half B timeout 2h "$PY" scripts/merge_lora_into_base.py --adapter "outputs/control_seed0$VARIANT" \
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
  require_done "merge$VARIANT" "base_eval$EVAL_TAG"; gpus_free A B
  local srh="srh_mixed_seed0$VARIANT" ctl="control_seed0$VARIANT" url_a url_b
  serve A "$NVME/merged/$srh" "$srh" 8000
  serve B "$NVME/merged/$ctl" "$ctl" 8001
  url_a=$(half_url A 8000); url_b=$(half_url B 8001)
  # One tracker around both runs: key usage is per key, so two trackers would each count both.
  # With the judge panel, both runs only save answers (--no-score), with no tracker.
  local flags="$EVAL_FLAGS"
  panel && flags="$flags --no-score"
  local both="
    $PY scripts/run_pilot_evals.py --model harmony/$srh --base-url $url_a $flags \
        --tag $srh$EVAL_TAG --tasks $EVAL_TASKS > $STATE/eval_$srh$EVAL_TAG.log 2>&1 & a=\$!
    $PY scripts/run_pilot_evals.py --model harmony/$ctl --base-url $url_b $flags \
        --tag $ctl$EVAL_TAG --tasks $EVAL_TASKS > $STATE/eval_$ctl$EVAL_TAG.log 2>&1 & b=\$!
    wait \$a; ra=\$?; wait \$b; rb=\$?; exit \$((ra || rb))"
  if panel; then
    timeout 4h bash -c "$both"
  else
    timeout 3h "$PY" scripts/track_openrouter_spend.py --label "arm_evals$VARIANT$EVAL_TAG" --cap "$SPEND_CAP" -- bash -c "$both"
  fi
  eval_health "results/$srh$EVAL_TAG.json" "$EVAL_TASKS"
  eval_health "results/$ctl$EVAL_TAG.json" "$EVAL_TASKS"
}

stage_compare() {
  require_done "arm_eval$VARIANT$EVAL_TAG" "base_eval$EVAL_TAG"
  local out="results/pilot_comparison$VARIANT$EVAL_TAG.md"
  "$PY" scripts/compare_pilot_results.py --treatment "results/srh_mixed_seed0$VARIANT$EVAL_TAG.json" \
      --control "results/control_seed0$VARIANT$EVAL_TAG.json" --base "results/base_own$EVAL_TAG.json" --out "$out"
  grep -E "^\*\*Verdict" "$out" | tee -a "$STATE/STATUS"
}

stage_gen_reasoning() {
  require_done bf16; gpus_free B
  serve B "$BF16" base 8001
  timeout 1h "$PY" scripts/gen_reasoning_examples_with_base.py --base-url "$(half_url B 8001)" --model base \
      --base-dir "$BF16" --out data/reasoning_examples.jsonl
  "$PY" scripts/build_sft_datasets.py --reasoning-examples data/reasoning_examples.jsonl \
      --out-dir data/processed_cotreg --stats data/STATS_cotreg.md
  log "reasoning examples: $(wc -l < data/reasoning_examples.jsonl); $(grep -m1 'Completion-token difference' data/STATS_cotreg.md)"
}

stage_sdf() {
  require_done bf16; gpus_free A B; need_disk_gb 50
  [ -f outputs/nll_base.json ] || timeout 1h "$PY" scripts/score_heldout_nll.py --model "$BF16" --out outputs/nll_base.json
  timeout 3h "$TORCHRUN" --nproc_per_node "$(n_gpus "$ALL_GPUS")" scripts/train_sdf.py --fsdp --model "$BF16" \
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

stage_sdf_train() {
  local data="$SDF_DATA_DIR" out="outputs/sdf_$SDF_RUN"
  [ -f "$data/sdf_train.jsonl" ] && [ -f "$data/sdf_heldout.jsonl" ] \
    || die "no $data/sdf_train.jsonl + sdf_heldout.jsonl: build the $SDF_ARM corpus first"
  require_done bf16; gpus_free A B; need_disk_gb 150   # final adapter + checkpoints, ~17 GB each
  # Base NLL on this corpus's held-out docs, once per data dir (docs without the '<doc>' prefix score differently).
  local base_nll
  base_nll="outputs/nll_base_$(basename "$data").json"
  [ -f "$base_nll" ] || on_half A timeout 1h "$PY" scripts/score_heldout_nll.py --model "$BF16" \
      --data "$data/sdf_heldout.jsonl" --out "$base_nll"
  rm -rf "$out"   # checkpoints left by a failed attempt would pass the checks below
  local launch=(env "CUDA_VISIBLE_DEVICES=$ALL_GPUS" "$TORCHRUN" --nproc_per_node "$(n_gpus "$ALL_GPUS")") train_env=()
  if [ -n "$HOST_B" ] && ! is_here "$HOST_B"; then
    # Two nodes: one torchrun per node, rendezvous on node A. FSDP shards over all 8 GPUs, as on one 8-GPU node
    # (one 4x96 GB node cannot hold the training state). SDF_TRAIN_ENV: NCCL settings for the inter-node network.
    [ -n "${SDF_TRAIN_ENV:-}" ] && mapfile -t train_env < "$SDF_TRAIN_ENV"
    launch=(srun --overlap --nodes=2 --ntasks-per-node=1 --nodelist="$HOST_A,$HOST_B" --gpus-per-node="$(n_gpus "$HALF_A")"
            --cpus-per-task="${SLURM_CPUS_ON_NODE:-1}" --mem=0 --input=none --kill-on-bad-exit=1
            env "${train_env[@]}" "$TORCHRUN" --nnodes 2 --nproc_per_node "$(n_gpus "$HALF_A")"
            --rdzv-backend c10d --rdzv-endpoint "$HOST_A:29500" --rdzv-id "sdf_$SDF_RUN")
  fi
  # SDF_ATTN: training attention for platforms without vllm-flash-attn3 (check_training_attention.py picks it).
  # shellcheck disable=SC2086  # SDF_TRAIN_FLAGS is a flag list
  timeout "$SDF_TIMEOUT" "${launch[@]}" scripts/train_sdf.py \
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

  # Held-out NLL of every adapter, two at a time (one per half).
  local ckpts=(final) dirs=("$out") pids=() i d
  for d in "$out"/checkpoint-epoch*; do
    if [ -d "$d" ]; then ckpts+=("${d##*/checkpoint-}"); dirs+=("$d"); fi
  done
  for i in "${!dirs[@]}"; do
    on_half "$([ $((i % 2)) -eq 0 ] && echo A || echo B)" timeout 1h "$PY" \
        scripts/score_heldout_nll.py --model "$BF16" --adapter "${dirs[$i]}" --data "$data/sdf_heldout.jsonl" \
        --out "outputs/nll_sdf_${SDF_RUN}_${ckpts[$i]}.json" > "$STATE/nll_sdf_${SDF_RUN}_${ckpts[$i]}.log" 2>&1 &
    pids+=($!)
    if [ ${#pids[@]} -eq 2 ] || [ "$i" -eq $(( ${#dirs[@]} - 1 )) ]; then
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
  gpus_free "$HALF"; need_disk_gb 520   # this merge and maybe the other half's, 234 GB each
  local merged="$NVME/merged/$name"
  rm -rf "$merged"
  [ "$KEEP_MERGED" = 1 ] || DELETE_ON_EXIT="$merged"
  # Checked on its own arm's data, like the SFT merges (stage_merge).
  retry_on_oom "$STATE/merge_$name.log" "$merged" on_half "$HALF" timeout 2h "$PY" scripts/merge_lora_into_base.py       --adapter "$adapter" --base "$BF16" --out "$merged" --verify-data "$SDF_DATA_DIR/sdf_train.jsonl"       --max-nll-diff "${SDF_MERGE_MAX_NLL_DIFF:-0.05}"     || die "merge failed; see $STATE/merge_$name.log"
  log "merge_verification: $("$PY" -c "import json, sys; print(json.load(open(sys.argv[1]))['merge_verification'])" "$merged/provenance.json")"
  serve "$HALF" "$merged" "$name" "$port"
  # Two sdf_eval stages (one per half) take turns on the judge (judged: one spend tracker at a time).
  # shellcheck disable=SC2086  # EVAL_FLAGS is a flag list
  judged "$name$EVAL_TAG" "$PY" scripts/run_pilot_evals.py --model "harmony/$name" --base-url "$(half_url "$HALF" "$port")"       --tag "$name$EVAL_TAG" --tasks "$SDF_EVAL_TASKS" --gibberish $EVAL_FLAGS
  # Format damage is an SDF outcome (PLAN 'Assistant format intact'), so it is recorded, not fatal.
  deferred "results/$name$EVAL_TAG.json" || health_report "results/$name$EVAL_TAG.json" "$SDF_EVAL_TASKS"     || log "FORMAT OUTSIDE LIMITS for $name$EVAL_TAG: recorded as a result (details in $STATE/$NAME.log)"
}

half_port() {  # half_port -> the server port for $HALF
  case "$HALF" in
    A) echo 8000 ;;
    B) echo 8001 ;;
    *) die "HALF must be A or B, not '$HALF'" ;;
  esac
}

stage_rl_adapter() {
  [ -n "$ORGANISM" ] || die "ORGANISM unset (names: scripts/download_rl_organism_adapter.py)"
  require_done bf16
  # outputs/<ORGANISM>_full: the whole adapter (the fp32 reference); outputs/<ORGANISM>: what vLLM serves, plus
  # serve_base.txt if part of the adapter had to go into a copy of the base (Redwood's lm_head).
  timeout 1h "$PY" scripts/prepare_rl_adapter_for_serving.py --organism "$ORGANISM" --out "outputs/$ORGANISM" \
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
  RL_SERVER=$("$PY" scripts/download_rl_organism_adapter.py --organism "$ORGANISM" --print-server) \
    || die "unknown ORGANISM '$ORGANISM'"
  if [ "$RL_SERVER" = sglang ]; then
    [ -x "$SGLANG_PY" ] || die "no SGLang venv at $SGLANG_PY (setup_gpu_node.sh with WITH_SGLANG=1)"
    serve_sglang "$1" "$2" "base_${HALF}_$ORGANISM" "$ORGANISM" "$MO/outputs/$ORGANISM"
    RL_MODEL="base_${HALF}_$ORGANISM:$ORGANISM"   # SGLang's OpenAI API picks a LoRA as <base>:<adapter>
    return
  fi
  flags=$("$PY" scripts/download_rl_organism_adapter.py --organism "$ORGANISM" --print-vllm-flags)
  [ -f "outputs/$ORGANISM/serve_base.txt" ] && base=$(cat "outputs/$ORGANISM/serve_base.txt")
  # shellcheck disable=SC2086  # flags is a flag list
  serve "$1" "$base" "base_${HALF}_$ORGANISM" "$2" --enable-lora --max-lora-rank 32 --lora-modules "$ORGANISM=$MO/outputs/$ORGANISM" $flags
  curl -sf "$(half_url "$1" "$2")/models" | grep -q "\"$ORGANISM\"" \
    || die "vLLM does not list the LoRA '$ORGANISM'; see $STATE/vllm_base_${HALF}_$ORGANISM.log"
  RL_MODEL="$ORGANISM"
}

stage_rl_check() {
  [ -n "$ORGANISM" ] || die "ORGANISM unset (names: scripts/download_rl_organism_adapter.py)"
  require_done bf16 "rl_adapter_$ORGANISM"; gpus_free A B; need_disk_gb 520
  local ref="results/rl_lora_check_${ORGANISM}_ref.json" out="results/rl_lora_check_$ORGANISM.json" port
  # Exact reference: the base upcast to fp32 with the adapter added in fp32 (~480 GB, deleted after).
  # REUSE_RL_REF=1 keeps an existing reference that has the noise floor (the adapter is pinned).
  if [ "${REUSE_RL_REF:-0}" = 1 ] && grep -q floor_ref_nll "$ref" 2>/dev/null; then
    log "reusing the fp32 reference in $ref"
  else
    timeout 2h "$PY" scripts/check_rl_lora_serving.py reference --adapter "outputs/${ORGANISM}_full" --base "$BF16" \
        --scratch "$NVME/ref_fp32_$ORGANISM" --out "$ref" > "$STATE/rl_check_ref_$ORGANISM.log" 2>&1 \
      || die "fp32 reference failed; see $STATE/rl_check_ref_$ORGANISM.log"
  fi
  HALF=A
  port=$(half_port)
  gpus_free A
  serve_rl A "$port"
  "$PY" scripts/check_rl_lora_serving.py served --server "$RL_SERVER" --organism "$ORGANISM" --base-model "base_A_$ORGANISM" \
      --base-url "$(half_url A "$port")" --ref "$ref" --out "$out" > "$STATE/rl_check_served_$ORGANISM.log" 2>&1 \
    || die "served LoRA does not match the fp32 reference; see $out and $STATE/rl_check_served_$ORGANISM.log"
  log "$("$PY" -c "import json, sys; r = json.load(open(sys.argv[1])); print('lora vs fp32 ref', r['lora_vs_ref'], '| base vs ref', r['base_vs_ref'])" "$out")"
}

stage_rl_eval() {
  [ -n "$ORGANISM" ] || die "ORGANISM unset (names: scripts/download_rl_organism_adapter.py)"
  local port
  port=$(half_port)
  require_done bf16 "rl_check_$ORGANISM"   # the served LoRA matched the fp32 reference
  gpus_free "$HALF"
  serve_rl "$HALF" "$port"
  local evals=("$PY" scripts/run_pilot_evals.py --model "harmony/$RL_MODEL" --base-url "$(half_url "$HALF" "$port")"
               --tag "$ORGANISM$EVAL_TAG" --tasks "$EVAL_TASKS")
  # shellcheck disable=SC2206  # EVAL_FLAGS is a flag list
  evals+=($EVAL_FLAGS)
  if panel || [[ ",$EVAL_TASKS," =~ ,(em|hacking), ]]; then
    # Judged tasks take turns on OpenRouter with the other half (judged; see stage_sdf_eval); the panel saves answers.
    judged "$ORGANISM$EVAL_TAG" "${evals[@]}"
  else
    timeout 4h "${evals[@]}"   # judge-free: no lock, so both halves run at once
  fi
  # RL without a KL penalty can damage the format: recorded as a result, not fatal (as for SDF).
  deferred "results/$ORGANISM$EVAL_TAG.json" || health_report "results/$ORGANISM$EVAL_TAG.json" "$EVAL_TASKS" \
    || log "FORMAT OUTSIDE LIMITS for $ORGANISM$EVAL_TAG: recorded as a result (details in $STATE/$NAME.log)"
}

stage_routing() {
  require_done base_eval; gpus_free A
  on_half A timeout 2h "$PY" scripts/measure_expert_routing_overlap.py --model "$BF16" \
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
case "$STAGE" in
  sdf_train) log "start (SDF_DATA_DIR=$SDF_DATA_DIR SDF_TRAIN_FLAGS='$SDF_TRAIN_FLAGS')" ;;
  sdf_eval)  log "start (CKPT=$CKPT HALF=$HALF SDF_EVAL_TASKS=$SDF_EVAL_TASKS EVAL_FLAGS='$EVAL_FLAGS' EVAL_TAG='$EVAL_TAG')" ;;
  rl_adapter|rl_check|rl_eval) log "start (ORGANISM=$ORGANISM HALF=$HALF EVAL_TASKS=$EVAL_TASKS EVAL_FLAGS='$EVAL_FLAGS' EVAL_TAG='$EVAL_TAG')" ;;
  *)         log "start (VARIANT='$VARIANT' DATA_DIR=$DATA_DIR EVAL_FLAGS='$EVAL_FLAGS' EVAL_TAG='$EVAL_TAG')" ;;
esac
start=$(date +%s)
set +e  # a failing stage must reach the FAILED line below, not exit here
( set -e; trap cleanup EXIT; "stage_$STAGE" ) 2>&1 | tee -a "$STATE/$NAME.log"
status=${PIPESTATUS[0]}
set -e
[ "$status" -eq 0 ] || die "exit $status after $(( ($(date +%s) - start) / 60 )) min; see $STATE/$NAME.log"
touch "$STATE/$NAME.done"
log "done in $(( ($(date +%s) - start) / 60 )) min"
