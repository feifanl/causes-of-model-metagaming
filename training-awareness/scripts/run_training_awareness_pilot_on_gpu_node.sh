#!/usr/bin/env bash
# Run the training-awareness pilot on a rented 4-GPU node (TRAINING_AWARENESS_PILOT.md, 'Compute and budget').
#
#   bash training-awareness/scripts/run_training_awareness_pilot_on_gpu_node.sh calibrate <max-hours>
#   bash training-awareness/scripts/run_training_awareness_pilot_on_gpu_node.sh full      <max-hours>
#   bash training-awareness/scripts/run_training_awareness_pilot_on_gpu_node.sh organisms <max-hours>
#   RUN_PREFIX=<prefix> bash training-awareness/scripts/run_training_awareness_pilot_on_gpu_node.sh judge <max-hours>
#
# calibrate  Olmo 3.1 only: the 150 calibration prompts under every condition, 10 SAD items per stage;
#            then the judge on those traces.
# full       Olmo 3.1 and gpt-oss-120b (base) side by side (GPUs 0-1 and 2-3, tensor parallel 2): 700 default +
#            250 x 3 cue rollouts and 797 SAD items each; then the judge (tensor parallel 4) and the summary.
# organisms  the gpt-oss cheater organisms and the bf16 base they are read against (ORGANISMS, default
#            "base_bf16 srh_mixed aisi_hack redwood_step952 base_sglang"; Feifan 2026-11-03: cheaters only, no
#            controls), each on 4 GPUs (tensor parallel 4; two at a time on an 8-GPU node), the same rollouts and SAD
#            items as full; then the judge and the summary, each organism paired with the base on its own engine:
#            base_bf16 (vLLM) for SRH and AISI, base_sglang (the base on Redwood's SGLang server, its LoRA loaded
#            but not selected; Feifan 2026-11-03) for Redwood, plus base_bf16 -> base_sglang for the engine itself. Served as in the organism
#            pipeline (model-organisms/RUN_EVALS_ON_ORGANISMS.md): the bf16 dequantized base, SRH merged into it,
#            AISI as base + LoRA in vLLM, Redwood as base + LoRA in SGLang. Each RL LoRA is checked on its rollout
#            server against the committed fp32 reference (model-organisms/results/rl_lora_check_*_ref.json); an
#            organism that fails gets no rollouts. Needs a git checkout (setup_gpu_node.sh diffs the data stats)
#            and ~1 TB free under DATA; H200s, since 80 GB GPUs leave little KV cache beside the 234 GB model.
# judge      re-judge an earlier run's uploaded rollout logs (RUN_PREFIX in the HF repo) without generating;
#            SKIP_LABELS (space-separated) leaves those checkpoints out (aisi_hack: its rollouts loop, Feifan
#            2026-11-03).
#
# The judge scores metric gaming only: the behavioural gap is the primary readout and awareness judging is
# deferred (Feifan, 2026-11-03). AWARENESS_JUDGE=1 adds the training-awareness judge with its rubric and recall
# checks, e.g. in judge mode on an earlier run's logs.
#
# Settings come from ~/.config/spar/env: HF_TOKEN (read), HF_WRITE_TOKEN + HF_ARTIFACT_REPO (uploads), DATA
# (default /workspace), and for the watchdog WATCHDOG_CLOUD + WATCHDOG_NODE_ID (e.g. vast "$CONTAINER_ID").
# Uploads go to $HF_ARTIFACT_REPO under training_awareness_pilot/<mode>_<UTC start>/ after every phase: rollout
# logs in plain text, SAD logs only inside a zip encrypted with SAD's password (its license forbids plaintext
# questions anywhere scrapable). The script holds ~/KEEP_ALIVE while it runs so the shared watchdog
# (model-organisms/scripts/stop_gpu_node_when_idle.sh) counts it busy; the watchdog's time cap still applies, and
# once the script ends the node is destroyed after the watchdog's idle period.
set -euo pipefail

MODE="${1:?usage: run_training_awareness_pilot_on_gpu_node.sh <calibrate|full|organisms|judge> <max-hours>}"
MAX_HOURS="${2:?max-hours required}"
case "$MODE" in calibrate|full|organisms|judge) ;; *) echo "unknown mode $MODE" >&2; exit 2 ;; esac

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
TA="$REPO/training-awareness"
# shellcheck disable=SC1091
[ -f "$HOME/.config/spar/env" ] && set -a && . "$HOME/.config/spar/env" && set +a
DATA="${DATA:-/workspace}"
STATE="$DATA/ta_state"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
PREFIX="training_awareness_pilot/${MODE}_$STAMP"
PY="$TA/venv/bin/python"
VLLM="$REPO/venv-vllm/bin/vllm"
HF="$TA/venv/bin/hf"
mkdir -p "$STATE" "$DATA/models"

# Pinned revisions (TRAINING_AWARENESS_PILOT.md, 'Models'). gpt-oss: the organisms' base checkpoint
# (model-organisms/scripts/setup_gpu_node.sh), served as released (MXFP4).
OLMO_MODEL="allenai/Olmo-3.1-32B-Think"; OLMO_REV="97ade57a9cc55ed48bcd76be4b7a3b6fee61b80d"
GPT_OSS_MODEL="openai/gpt-oss-120b";     GPT_OSS_REV="b5c939de8f754692c1647ca79fbf85e8c1e70f8a"
JUDGE_MODEL="Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"; JUDGE_REV="e156cb4efae43fbee1a1ab073f946a1377e6b969"
SAD_COMMIT="dfc5c9831a9bcc5c9a9dbdcaa2955aae983cd1d3"
SAD_PASSWORD="sadtimesforthesetimes"   # SAD's own, published in its unzip.sh
VLLM_VERSION=0.30.0
OLMO_LEN=49152      # 32,768 generated + the longest prompt (an impossible_lcb task, ~12.6k tokens); model max 65,536
GPT_OSS_LEN=49152   # the same budget; model max 131,072
# Non-streaming completions return only when done; a 32k-token trace under load can exceed Inspect's 600 s default.
export INSPECT_HTTP_REQUEST_TIMEOUT=7200
JUDGE_LEN=57344     # rubric + prompt + up to 32,768 tokens of reasoning and answer

# organisms mode: the organism pipeline's own venvs, scripts and bf16 base (model-organisms/scripts/setup_gpu_node.sh).
MO="$REPO/model-organisms"
MO_PY="$REPO/venv/bin/python"
SGLANG_PY="$REPO/venv-sglang/bin/python"
BF16="$DATA/gpt-oss-120b-bf16"
SRH_NAME="srh_mixed_seed0_cotreg"   # the CoT-regularized SRH arm, seed 0 (RUN_EVALS_ON_ORGANISMS.md)
ORGANISMS="${ORGANISMS:-base_bf16 srh_mixed aisi_hack redwood_step952 base_sglang}"
declare -A GEN_MODEL=()   # label -> the served model name its rollouts query (set by serve_organism)
GEN_FAILED=""

log() { echo "$(date -u +%FT%TZ) $MODE $*" | tee -a "$STATE/STATUS"; }
die() { log "FAILED: $*"; upload_state; exit 1; }
START_EPOCH=$(date +%s)
# Seconds left before the watchdog's cap, minus 20 min to upload: a judge run once hit the cap mid-scan and the
# node was destroyed with nothing uploaded.
seconds_left() { awk -v h="$MAX_HOURS" -v s="$START_EPOCH" -v n="$(date +%s)" 'BEGIN { t = h * 3600 - (n - s) - 1200; printf "%d", (t > 0 ? t : 1) }'; }
SERVERS=()
cleanup() {
  for pid in "${SERVERS[@]:-}"; do [ -n "$pid" ] && kill -- "-$pid" 2>/dev/null || true; done
  rm -f "$HOME/KEEP_ALIVE"
}
trap cleanup EXIT

upload() {  # upload <path on node> <path under $PREFIX>; never fatal
  [ -n "${HF_ARTIFACT_REPO:-}" ] && [ -n "${HF_WRITE_TOKEN:-}" ] || { log "upload skipped (no HF_ARTIFACT_REPO/HF_WRITE_TOKEN)"; return 0; }
  [ -e "$1" ] || { log "upload skipped: $1 missing"; return 0; }
  HF_TOKEN="$HF_WRITE_TOKEN" timeout 2h "$HF" upload "$HF_ARTIFACT_REPO" "$1" "$PREFIX/$2" --repo-type model \
      --private --commit-message "training-awareness pilot: $PREFIX/$2" >>"$STATE/upload.log" 2>&1 \
    && log "uploaded $2" || log "UPLOAD FAILED: $2 (see upload.log)"
}
upload_state() {
  mkdir -p "$TA/results/node_logs"
  for f in "$STATE"/*.log "$STATE/STATUS"; do
    [ -f "$f" ] || continue
    case "$(basename "$f")" in
      # vLLM and Inspect logs could carry request text (SAD questions): upload only their tail, without request lines.
      vllm_*|sglang_*|generate_*) tail -n 400 "$f" | grep -v -i -E "prompt|request" > "$TA/results/node_logs/$(basename "$f")" || true ;;
      *) cp "$f" "$TA/results/node_logs/" ;;
    esac
  done
  # organisms mode: the organism pipeline's stages (bf16, rl_adapter) log to their own STATUS.
  if [ -f "$DATA/pilot_state/STATUS" ]; then cp "$DATA/pilot_state/STATUS" "$TA/results/node_logs/organism_stages_STATUS"; fi
  upload "$TA/results" results
}

archive_old() {  # archive_old <dir>: move earlier output aside so a rerun never mixes generations or scans
  if [ -d "$1" ] && [ -n "$(ls -A "$1")" ]; then
    mv "$1" "$1.before_$STAMP"
    log "moved earlier $(basename "$1") to $1.before_$STAMP"
  fi
}

# A server that fails to start ends the run, except with SERVE_SOFT=1 (organisms mode: that organism is skipped and
# serve returns 1; stop_servers cleans up).
serve_fail() { if [ "${SERVE_SOFT:-0}" = 1 ]; then log "$*"; else die "$*"; fi; }

serve() {  # serve <model dir> <served name> <port> <gpus> [extra vllm flags...]
  local model=$1 name=$2 port=$3 gpus=$4 tp waited=0
  shift 4
  tp=$(( $(tr -cd , <<<"$gpus" | wc -c) + 1 ))
  CUDA_VISIBLE_DEVICES="$gpus" setsid "$VLLM" serve "$model" --served-model-name "$name" --port "$port" \
      --tensor-parallel-size "$tp" "$@" > "$STATE/vllm_$name.log" 2>&1 &
  SERVERS+=($!)
  local pid=$!
  until curl -sf "http://localhost:$port/v1/models" | grep -q "\"$name\""; do
    kill -0 "$pid" 2>/dev/null || { serve_fail "vLLM for $name exited; see $STATE/vllm_$name.log"; return 1; }
    [ "$waited" -ge 3600 ] && { serve_fail "vLLM for $name not ready after 60 min"; return 1; }
    sleep 15; waited=$((waited + 15))
  done
  log "vLLM ready: $name on GPUs $gpus, port $port (${waited}s)"
}

stop_servers() {
  for pid in "${SERVERS[@]:-}"; do [ -n "$pid" ] && kill -- "-$pid" 2>/dev/null || true; done
  SERVERS=()
  local waited=0  # the next servers need the GPUs free
  while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do
    [ "$waited" -ge 180 ] && die "GPUs still busy 3 min after stopping the servers"
    sleep 5; waited=$((waited + 5))
  done
}

download() {  # download <repo> <revision> <dir> [extra hf download flags...]
  [ -f "$3/.done" ] && return 0
  HF_TOKEN="${HF_TOKEN:-}" "$HF" download "$1" --revision "$2" --local-dir "$3" "${@:4}" >>"$STATE/download.log" 2>&1 \
    || die "download of $1@$2 failed; see download.log"
  touch "$3/.done"
  log "downloaded $1@${2:0:8} ($(du -sh "$3" | cut -f1))"
}

setup() {
  touch "$HOME/KEEP_ALIVE"
  if [ -n "${WATCHDOG_CLOUD:-}" ] && ! pgrep -f stop_gpu_node_when_idle >/dev/null; then
    NVME="$DATA" setsid bash "$REPO/model-organisms/scripts/stop_gpu_node_when_idle.sh" "$WATCHDOG_CLOUD" \
        "${WATCHDOG_NODE_ID:?WATCHDOG_NODE_ID unset}" "$MAX_HOURS" 30 >/dev/null 2>&1 &
    log "watchdog started: $WATCHDOG_CLOUD, cap ${MAX_HOURS} h"
  fi
  if [ ! -f "$TA/venv/.installed" ]; then
    python3 -m venv "$TA/venv" && "$TA/venv/bin/pip" install -q -r "$TA/requirements.txt" >>"$STATE/setup.log" 2>&1 \
      || die "pilot venv install failed; see setup.log"
    touch "$TA/venv/.installed"
  fi
  if [ ! -f "$REPO/venv-vllm/.installed" ]; then
    python3 -m venv "$REPO/venv-vllm" && "$REPO/venv-vllm/bin/pip" install -q "vllm==$VLLM_VERSION" >>"$STATE/setup.log" 2>&1 \
      || die "vLLM install failed; see setup.log"
    touch "$REPO/venv-vllm/.installed"
  fi
  "$REPO/venv-vllm/bin/python" -c "import vllm, sys; sys.exit(vllm.__version__ != '$VLLM_VERSION')" \
    || die "vLLM is not $VLLM_VERSION"
  export PYTHONPATH="$TA"
  # A hang here once held a rented node for over an hour (Scout's multiprocess scan on Linux): fail fast instead.
  (cd "$TA" && timeout 900 "$PY" -m pytest -q tests >>"$STATE/setup.log" 2>&1) \
    || die "pilot tests failed or timed out on the node; see setup.log"
  (cd "$TA" && "$PY" scripts/build_pilot_prompts.py --stats "$STATE/STATS_rebuilt.md" >>"$STATE/setup.log" 2>&1) \
    || die "prompt build failed"
  # tr: a checkout from Windows can carry CRLF line endings.
  diff <(grep sha256 "$TA/data/STATS.md" | tr -d '\r') <(grep sha256 "$STATE/STATS_rebuilt.md" | tr -d '\r') >/dev/null \
    || die "rebuilt prompts differ from the committed data/STATS.md hash"
  log "prompts rebuilt; hash matches data/STATS.md"
  if [ "$MODE" != judge ]; then
    [ -d "$DATA/sad/.git" ] || git clone -q https://github.com/LrudL/sad.git "$DATA/sad"
    git -C "$DATA/sad" checkout -q "$SAD_COMMIT"
    unzip -q -o -P "$SAD_PASSWORD" "$DATA/sad/sad/stages/full/structs.zip" -d "$DATA/sad/sad/stages/full/structs"
    (cd "$TA" && "$PY" scripts/prepare_sad_stages_full.py --structs "$DATA/sad/sad/stages/full/structs") >>"$STATE/setup.log" 2>&1 \
      || die "SAD preparation failed"
    log "SAD stages-full items prepared (stay on the node)"
  fi
}

# --------------------------------------------------------------------------- organisms mode

serve_sglang() {  # serve_sglang <model dir> <served name> <port> <gpus> <lora name> <lora dir>: base + one LoRA
  local model=$1 name=$2 port=$3 gpus=$4 tp waited=0
  tp=$(( $(tr -cd , <<<"$gpus" | wc -c) + 1 ))
  # --moe-runner-backend triton: gpt-oss's default triton_kernel has no LoRA path for bf16 experts (as in
  # model-organisms/scripts/run_pilot_stage_on_gpu_node.sh, serve_sglang).
  CUDA_VISIBLE_DEVICES="$gpus" setsid "$SGLANG_PY" -m sglang.launch_server --model-path "$model" --served-model-name "$name" \
      --tp "$tp" --port "$port" --context-length "$GPT_OSS_LEN" --enable-lora --max-lora-rank 32 \
      --lora-target-modules all --lora-paths "$5=$6" --moe-runner-backend triton > "$STATE/sglang_$name.log" 2>&1 &
  SERVERS+=($!)
  local pid=$!
  until curl -sf "http://localhost:$port/v1/models" | grep -q "\"$name\""; do
    kill -0 "$pid" 2>/dev/null || { serve_fail "SGLang for $name exited; see $STATE/sglang_$name.log"; return 1; }
    [ "$waited" -ge 3600 ] && { serve_fail "SGLang for $name not ready after 60 min"; return 1; }
    sleep 15; waited=$((waited + 15))
  done
  log "SGLang ready: $name + LoRA $5 on GPUs $gpus, port $port (${waited}s)"
}

# serve_organism <label> <port> <gpus>: serve one model of ORGANISMS and set GEN_MODEL[label]. An RL organism's
# LoRA is then scored on this server against the exact fp32 merge committed by the organism pipeline's rl_check
# (recomputing it needs ~480 GB of RAM); returns 1, after logging, if a server fails to start or the served LoRA
# does not match.
serve_organism() {
  local label=$1 port=$2 gpus=$3 server base="$BF16" flags
  case "$label" in
    base_bf16) SERVE_SOFT=1 serve "$BF16" base_bf16 "$port" "$gpus" --max-model-len "$GPT_OSS_LEN" || return 1
               GEN_MODEL[$label]=base_bf16; return 0 ;;
    # The base, queried by its own name on a server set up exactly as Redwood's (same flags, Redwood's LoRA loaded).
    base_sglang) SERVE_SOFT=1 serve_sglang "$BF16" base_sglang "$port" "$gpus" redwood_step952 "$MO/outputs/redwood_step952" \n                   || return 1
                 GEN_MODEL[$label]=base_sglang; return 0 ;;
    srh_mixed) SERVE_SOFT=1 serve "$DATA/merged/$SRH_NAME" srh_mixed "$port" "$gpus" --max-model-len "$GPT_OSS_LEN" || return 1
               GEN_MODEL[$label]=srh_mixed; return 0 ;;
  esac
  server=$(cd "$MO" && "$MO_PY" scripts/download_rl_organism_adapter.py --organism "$label" --print-server) \
    || die "unknown organism $label (names: model-organisms/scripts/download_rl_organism_adapter.py)"
  if [ "$server" = sglang ]; then
    SERVE_SOFT=1 serve_sglang "$BF16" "base_$label" "$port" "$gpus" "$label" "$MO/outputs/$label" || return 1
    GEN_MODEL[$label]="base_$label:$label"   # SGLang's OpenAI API picks a LoRA as <base>:<adapter>
  else
    flags=$(cd "$MO" && "$MO_PY" scripts/download_rl_organism_adapter.py --organism "$label" --print-vllm-flags)
    if [ -f "$MO/outputs/$label/serve_base.txt" ]; then base=$(cat "$MO/outputs/$label/serve_base.txt"); fi
    # shellcheck disable=SC2086  # flags is a flag list
    SERVE_SOFT=1 serve "$base" "base_$label" "$port" "$gpus" --max-model-len "$GPT_OSS_LEN" --enable-lora --max-lora-rank 32 \
        --lora-modules "$label=$MO/outputs/$label" $flags || return 1
    curl -sf "http://localhost:$port/v1/models" | grep -q "\"$label\"" \
      || { log "vLLM does not list the LoRA '$label': no rollouts (see vllm_base_$label.log)"; return 1; }
    GEN_MODEL[$label]=$label
  fi
  if ! (cd "$MO" && timeout 30m "$MO_PY" scripts/check_rl_lora_serving.py served --server "$server" --organism "$label" \
      --base-model "base_$label" --base-url "http://localhost:$port/v1" --ref "results/rl_lora_check_${label}_ref.json" \
      --out "$TA/results/rl_lora_check_$label.json") > "$STATE/rl_check_$label.log" 2>&1; then
    log "LoRA CHECK FAILED for $label: no rollouts (see rl_check_$label.log)"
    return 1
  fi
  log "LoRA check passed for $label: $("$PY" -c "import json, sys; r = json.load(open(sys.argv[1])); print('lora vs fp32 ref', r['lora_vs_ref']['mean_abs_nll_diff'], '| base vs ref', r['base_vs_ref']['mean_abs_nll_diff'])" "$TA/results/rl_lora_check_$label.json")"
}

mo_stage() {  # mo_stage <stage> [VAR=value...]: one stage of the organism pipeline's runner, on this node's GPUs
  env NVME="$DATA" HALF_A="${SLOTS[0]}" HALF_B="${SLOTS[${#SLOTS[@]}-1]}" "${@:2}" \
      bash "$MO/scripts/run_pilot_stage_on_gpu_node.sh" "$1" >> "$STATE/organism_stages.log" 2>&1
}

prepare_organisms() {
  case "$(nvidia-smi -L | wc -l)" in
    4) SLOTS=(0,1,2,3) ;;
    8) SLOTS=(0,1,2,3 4,5,6,7) ;;
    *) die "organisms mode needs 4 or 8 GPUs (each model is served on 4)" ;;
  esac
  local all_gpus; all_gpus=$(IFS=,; echo "${SLOTS[*]}")
  export HF_HOME="$DATA/hf"   # where setup_gpu_node.sh puts the MXFP4 checkpoint the bf16 stage reads
  if [ ! -f "$BF16/provenance.json" ]; then
    local free; free=$(df --output=avail -BG "$DATA" | tail -1 | tr -dc 0-9)
    [ "$free" -ge 1000 ] || die "only $free GB free under $DATA; organisms mode needs ~1 TB"
  fi
  if [ ! -f "$STATE/organism_setup.done" ]; then
    WITH_SGLANG=1 timeout 3h bash "$MO/scripts/setup_gpu_node.sh" "$DATA" > "$STATE/organism_setup.log" 2>&1 \
      || die "organism pipeline setup failed; see organism_setup.log"
    touch "$STATE/organism_setup.done"
  fi
  log "organism pipeline set up (its venvs incl. SGLang, data, tests)"
  # 4 x 80 GB ran out of memory dequantizing (stage_bf16): on an 8-GPU node use all eight.
  local bf16_gpus=()
  if [ "${#SLOTS[@]}" -eq 2 ]; then bf16_gpus=("BF16_GPUS=$all_gpus"); fi
  mo_stage bf16 "${bf16_gpus[@]}" || die "bf16 dequantization failed; see organism_stages.log"
  log "bf16 base ready ($(du -sh "$BF16" | cut -f1))"
  local label
  # base_sglang serves Redwood's adapter beside the base, so it needs it prepared too.
  for label in $(for l in $ORGANISMS; do [ "$l" = base_sglang ] && l=redwood_step952; echo "$l"; done | sort -u); do
    case "$label" in base_bf16|srh_mixed) continue ;; esac
    mo_stage rl_adapter "ORGANISM=$label" || die "preparing the $label adapter failed; see organism_stages.log"
    log "$label adapter prepared"
  done
  [[ " $ORGANISMS " == *" srh_mixed "* ]] || return 0
  [ -n "${HF_ARTIFACT_REPO:-}" ] || die "srh_mixed needs HF_ARTIFACT_REPO (its adapter is in the private repo)"
  local merged="$DATA/merged/$SRH_NAME" adapter="$DATA/hf_fetch/adapters/$SRH_NAME"
  [ -f "$merged/provenance.json" ] && return 0
  HF_TOKEN="${HF_WRITE_TOKEN:-${HF_TOKEN:-}}" timeout 1h "$HF" download "$HF_ARTIFACT_REPO" --repo-type model \
      --include "adapters/$SRH_NAME/*" --local-dir "$DATA/hf_fetch" >>"$STATE/download.log" 2>&1 \
    || die "download of adapters/$SRH_NAME failed; see download.log"
  "$PY" -c "import json, sys; sys.exit(json.load(open(sys.argv[1]))['base_model_name_or_path'] != 'openai/gpt-oss-120b')" \
      "$adapter/adapter_config.json" || die "$SRH_NAME adapter was not trained on the Hub base"
  # Checked on the SRH rows of data/processed: a new node has no data/processed_cotreg, and the check only bounds
  # bf16 rounding (as in model-organisms/scripts/run_capability_session_on_gpu_node.sh).
  local merge=(env "CUDA_VISIBLE_DEVICES=$all_gpus" timeout 2h "$MO_PY" scripts/merge_lora_into_base.py --adapter "$adapter"
               --base "$BF16" --out "$merged" --verify-data data/processed/srh_mixed.jsonl)
  if ! (cd "$MO" && "${merge[@]}") > "$STATE/merge_srh.log" 2>&1; then
    grep -qE "OutOfMemoryError|CUDA out of memory" "$STATE/merge_srh.log" || die "SRH merge failed; see merge_srh.log"
    log "out of GPU memory merging SRH; retrying with --max-gpu-gb 50 (the rest of the model on CPU)"
    rm -rf "$merged"
    (cd "$MO" && "${merge[@]}" --max-gpu-gb 50) >> "$STATE/merge_srh.log" 2>&1 || die "SRH merge failed; see merge_srh.log"
  fi
  log "merged $SRH_NAME: $("$PY" -c "import json, sys; print(json.load(open(sys.argv[1]))['merge_verification'])" "$merged/provenance.json")"
}

generate_organisms() {  # each batch: one model per 4-GPU slot, side by side; an organism that fails is skipped
  local queue=() i j label pids batch
  read -r -a queue <<<"$ORGANISMS"
  for ((i = 0; i < ${#queue[@]}; i += ${#SLOTS[@]})); do
    batch=(); pids=()
    for ((j = 0; j < ${#SLOTS[@]} && i + j < ${#queue[@]}; j++)); do
      label=${queue[i + j]}
      if [ "$(gen_limit)" -le 1800 ]; then
        log "under 30 min of generation time left: $label not started"; GEN_FAILED+=" $label"; continue
      fi
      if serve_organism "$label" $((8000 + j)) "${SLOTS[j]}"; then batch+=("$label:$((8000 + j))")
      else GEN_FAILED+=" $label"; fi
    done
    for j in "${!batch[@]}"; do
      generate_checkpoint "${batch[j]%%:*}" "${batch[j]##*:}" all & pids+=($!)
    done
    for j in "${!batch[@]}"; do
      label=${batch[j]%%:*}
      wait "${pids[j]}" || { log "$label generation failed; see generate_$label.log"; GEN_FAILED+=" $label"; }
      upload_generation "$label"
    done
    stop_servers
  done
}

# organisms mode: generation stops JUDGE_RESERVE_HOURS before the cap (less the upload margin), so the judge still
# runs on what Inspect has written by then; elsewhere `timeout 0`, no limit.
JUDGE_RESERVE_HOURS="${JUDGE_RESERVE_HOURS:-2.5}"
gen_limit() {
  if [ "$MODE" != organisms ]; then echo 0; return; fi
  awk -v l="$(seconds_left)" -v r="$JUDGE_RESERVE_HOURS" 'BEGIN { t = l - r * 3600; printf "%d", (t > 60 ? t : 60) }'
}

generate_checkpoint() {  # generate_checkpoint <label> <port> <subset_for_default> <sad_limit_args...>
  local label=$1 port=$2 subset=$3 model family
  shift 3
  local logs="$TA/logs/$label"
  case "$label" in  # each provider renders its family's prompt format; -T family picks its sampling defaults
    olmo) model="olmo/$label"; family=olmo; export OLMO_BASE_URL="http://localhost:$port/v1" ;;
    gpt_oss) model="gptoss/$label"; family=gpt_oss; export HARMONY_BASE_URL="http://localhost:$port/v1" ;;
    *) [ -n "${GEN_MODEL[$label]:-}" ] || die "unknown model label $label"   # an organism (serve_organism)
       model="gptoss/${GEN_MODEL[$label]}"; family=gpt_oss; export HARMONY_BASE_URL="http://localhost:$port/v1" ;;
  esac
  (cd "$TA" &&
    timeout "$(gen_limit)" "$PY" "$TA/scripts/run_inspect_with_pilot_providers.py" eval pilot/tasks.py@pilot_rollouts -T condition=default -T subset="$subset" \
        -T family="$family" --model "$model" --log-dir "$logs/rollouts" --max-connections 96 --display plain &&
    for cond in ambiguous_cue grader_cue explicit_cue; do
      cue_subset=cue; [ "$subset" = calibration ] && cue_subset=calibration
      timeout "$(gen_limit)" "$PY" "$TA/scripts/run_inspect_with_pilot_providers.py" eval pilot/tasks.py@pilot_rollouts -T condition="$cond" -T subset="$cue_subset" \
          -T family="$family" --model "$model" --log-dir "$logs/rollouts" --max-connections 96 --display plain || exit 1
    done &&
    timeout "$(gen_limit)" "$PY" "$TA/scripts/run_inspect_with_pilot_providers.py" eval pilot/tasks.py@sad_stages_full "$@" -T family="$family" --model "$model" \
        -M force_answer=true --log-dir "$logs/sad" --max-connections 96 --display plain
  ) > "$STATE/generate_$label.log" 2>&1
}

upload_generation() {  # upload_generation <label>
  local label=$1
  upload "$TA/logs/$label/rollouts" "logs/$label/rollouts"
  if [ -d "$TA/logs/$label/sad" ]; then
    (cd "$TA/logs/$label" && rm -f "$STATE/sad_$label.zip" && zip -q -r -P "$SAD_PASSWORD" "$STATE/sad_$label.zip" sad)
    upload "$STATE/sad_$label.zip" "logs/$label/sad_logs_encrypted.zip"
  fi
}

phase_generate() {
  archive_old "$TA/logs"
  archive_old "$TA/scans"
  if [ "$MODE" = calibrate ]; then
    download "$OLMO_MODEL" "$OLMO_REV" "$DATA/models/olmo"
    serve "$DATA/models/olmo" olmo 8000 0,1,2,3 --max-model-len "$OLMO_LEN"
    generate_checkpoint olmo 8000 calibration -T limit_per_stage=10 || die "generation failed; see generate_olmo.log"
    upload_generation olmo
  elif [ "$MODE" = organisms ]; then
    prepare_organisms
    generate_organisms
    [ -n "$(ls -d "$TA"/logs/*/rollouts 2>/dev/null)" ] || die "no organism produced rollouts:$GEN_FAILED"
  else
    download "$OLMO_MODEL" "$OLMO_REV" "$DATA/models/olmo"
    download "$GPT_OSS_MODEL" "$GPT_OSS_REV" "$DATA/models/gpt_oss" --exclude "original/*" --exclude "metal/*"
    serve "$DATA/models/olmo" olmo 8000 0,1 --max-model-len "$OLMO_LEN"
    serve "$DATA/models/gpt_oss" gpt_oss 8001 2,3 --max-model-len "$GPT_OSS_LEN"
    generate_checkpoint olmo 8000 all & local a=$!
    generate_checkpoint gpt_oss 8001 all & local b=$!
    local fail=0
    wait "$a" || { log "olmo generation failed; see generate_olmo.log"; fail=1; }
    wait "$b" || { log "gpt_oss generation failed; see generate_gpt_oss.log"; fail=1; }
    upload_generation olmo
    upload_generation gpt_oss
    [ "$fail" -eq 0 ] || die "generation failed"
  fi
  stop_servers
  log "generation done"
}

fetch_earlier_logs() {
  [ -n "${RUN_PREFIX:-}" ] || die "judge mode needs RUN_PREFIX (an earlier run under training_awareness_pilot/)"
  HF_TOKEN="${HF_WRITE_TOKEN:-${HF_TOKEN:-}}" "$HF" download "$HF_ARTIFACT_REPO" --repo-type model \
      --include "$RUN_PREFIX/logs/*/rollouts/*" --local-dir "$STATE/fetch" >>"$STATE/download.log" 2>&1 \
    || die "download of $RUN_PREFIX logs failed"
  archive_old "$TA/logs"
  mkdir -p "$TA/logs"
  cp -r "$STATE/fetch/$RUN_PREFIX/logs/." "$TA/logs/"
  local label
  for label in ${SKIP_LABELS:-}; do rm -rf "${TA:?}/logs/$label"; done
  log "fetched rollout logs from $RUN_PREFIX${SKIP_LABELS:+ (skipped: $SKIP_LABELS)}"
}

phase_judge() {
  local labels=() label dir pairs=()
  for dir in "$TA"/logs/*/rollouts; do [ -d "$dir" ] && labels+=("$(basename "$(dirname "$dir")")"); done
  [ "${#labels[@]}" -gt 0 ] || die "no rollout logs to judge under $TA/logs"
  archive_old "$TA/scans"
  download "$JUDGE_MODEL" "$JUDGE_REV" "$DATA/models/judge"
  serve "$DATA/models/judge" judge 8002 0,1,2,3 --max-model-len "$JUDGE_LEN"
  export JUDGE_BASE_URL="http://localhost:8002/v1" JUDGE_API_KEY=EMPTY
  local judge="openai-api/judge/judge" rc awareness=()
  if [ "${AWARENESS_JUDGE:-0}" = 1 ]; then
    awareness=(--awareness)
    # The awareness judge's checks first: they are short and say whether it is worth its scan time.
    (cd "$TA" && timeout "$(seconds_left)" "$PY" scripts/check_judge.py rubric --model "$judge" --tag "_$MODE") \
        >> "$STATE/judge_checks.log" 2>&1 || die "rubric check failed or ran out of time; see judge_checks.log"
    log "rubric: $(grep -m1 '^rubric:' "$STATE/judge_checks.log" || echo '?')"
    for label in "${labels[@]}"; do
      (cd "$TA" && timeout "$(seconds_left)" "$PY" scripts/check_judge.py recall --model "$judge" \
          --logs "logs/$label/rollouts" --tag "_${MODE}_$label") >> "$STATE/judge_checks.log" 2>&1 \
        && log "recall check done for $label" || log "recall check skipped or failed for $label (see judge_checks.log)"
    done
    upload_state
  else
    log "awareness judge deferred (AWARENESS_JUDGE=1 runs it): metric gaming only"
  fi
  for label in "${labels[@]}"; do
    rc=0
    (cd "$TA" && timeout "$(seconds_left)" "$PY" scripts/scan_pilot_rollouts.py --logs "logs/$label/rollouts" \
        --scans "scans/$label" --model "$judge" "${awareness[@]}") >> "$STATE/judge_$label.log" 2>&1 || rc=$?
    if [ "$rc" -eq 124 ]; then upload "$TA/scans" scans; die "time cap reached while judging $label; partial scans uploaded"; fi
    [ "$rc" -eq 0 ] || die "judging $label failed; see judge_$label.log"
    log "judged $label"
  done
  stop_servers
  local ckpts; ckpts=$(IFS=,; echo "${labels[*]}")
  # The organisms have no controls here: each is read against the bf16 base on its own engine, on the same prompts
  # (base_bf16 -> base_sglang is the engine difference itself).
  if [[ ",$ckpts," == *",base_bf16,"* ]]; then
    for label in "${labels[@]}"; do
      case "$label" in
        base_bf16|olmo|gpt_oss) ;;
        redwood_step952) if [[ ",$ckpts," == *",base_sglang,"* ]]; then pairs+=("base_sglang:$label"); else pairs+=("base_bf16:$label"); fi ;;
        *) pairs+=("base_bf16:$label") ;;
      esac
    done
  fi
  (cd "$TA" && "$PY" scripts/summarize_pilot_results.py --checkpoints "$ckpts" --pairs "$(IFS=,; echo "${pairs[*]}")" \
      --tag "_$MODE") > "$STATE/summary.log" 2>&1 || die "summary failed; see summary.log"
  upload "$TA/scans" scans
  upload_state
  [ -z "$GEN_FAILED" ] || die "no or partial rollouts for:$GEN_FAILED (the rest judged; results under $PREFIX)"
  log "DONE: results under $PREFIX in $HF_ARTIFACT_REPO"
}

log "start (prefix $PREFIX, cap ${MAX_HOURS} h)"
setup
if [ "$MODE" = judge ]; then fetch_earlier_logs; else phase_generate; fi
phase_judge
