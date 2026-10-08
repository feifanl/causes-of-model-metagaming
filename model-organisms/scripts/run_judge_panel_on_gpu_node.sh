#!/usr/bin/env bash
# Score saved eval logs with the three-judge panel and vote (DECISIONS 'Judges on Isambard').
#
#   NVME=/scratch bash model-organisms/scripts/run_judge_panel_on_gpu_node.sh results/*.logs.json
#
# For each judge in turn: download it, serve it with vLLM on all GPUs, score every manifest's logs
# (score_logs_with_judge.py), stop the server. Then vote each manifest into results/<tag>.json
# (aggregate_judge_panel.py). Judges never run beside each other or beside a model under test.
# Resumable: logs a judge has already scored are skipped, so rerunning the same command continues.
#
# Overrides: JUDGE_GPUS (default 0,...,N-1, all visible), JUDGE_TP (default: the number of JUDGE_GPUS),
# JUDGE_PORT (8101), JUDGE_PANEL (lines 'name hf_id', default below). Needs HF_TOKEN with Meta's Llama
# terms accepted. Progress goes to $NVME/pilot_state/STATUS.
set -uo pipefail

[ $# -gt 0 ] || { echo "usage: run_judge_panel_on_gpu_node.sh <results/<tag>.logs.json>..."; exit 2; }
NVME="${NVME:-/data}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
MO="$REPO/model-organisms"
STATE="$NVME/pilot_state"
PY="${PY:-$REPO/venv/bin/python}"
VLLM="${VLLM:-$REPO/venv-vllm/bin/vllm}"
HF="${HF:-$REPO/venv/bin/hf}"
# shellcheck disable=SC1091
[ -f "$HOME/.config/spar/env" ] && set -a && . "$HOME/.config/spar/env" && set +a
mkdir -p "$STATE"
log() { echo "$(date -u +%FT%TZ) judge_panel $*" | tee -a "$STATE/STATUS"; }
die() { log "FAILED: $*"; exit 1; }

# Qwen in FP8 (Qwen's own release): bf16 is 470 GB, more than one Isambard node's 4 x 96 GB.
DEFAULT_PANEL="qwen Qwen/Qwen3-235B-A22B-Instruct-2507-FP8
llama meta-llama/Llama-3.3-70B-Instruct
mistral mistralai/Mistral-Large-Instruct-2411"
PANEL="${JUDGE_PANEL:-$DEFAULT_PANEL}"
N_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
GPUS="${JUDGE_GPUS:-$(seq -s, 0 $((N_GPUS - 1)))}"
TP="${JUDGE_TP:-$(echo "$GPUS" | tr ',' '\n' | grep -c .)}"
PORT="${JUDGE_PORT:-8101}"

MANIFESTS=()
for m in "$@"; do
  [ -f "$MO/$m" ] || [ -f "$m" ] || die "no manifest $m"
  MANIFESTS+=("$m")
done
log "start: ${#MANIFESTS[@]} manifest(s), judges $(echo "$PANEL" | cut -d' ' -f1 | paste -sd, -), GPUs $GPUS (TP $TP)"

SERVER=""
stop_server() {  # stop the judge's server and wait until its port is free (vLLM takes a while to exit)
  [ -n "$SERVER" ] || return 0
  kill -- "-$SERVER" 2>/dev/null || kill "$SERVER" 2>/dev/null
  local i
  for i in $(seq 1 60); do
    curl -sf "localhost:$PORT/v1/models" >/dev/null 2>&1 || break
    [ "$i" = 30 ] && { kill -9 -- "-$SERVER" 2>/dev/null || kill -9 "$SERVER" 2>/dev/null; }
    sleep 2
  done
  SERVER=""
}
trap stop_server EXIT

while read -r name hf_id; do
  [ -n "$name" ] || continue
  # Only the Hugging Face format vLLM loads: Llama's original/ and Mistral's consolidated copy would double the download.
  "$HF" download "$hf_id" --exclude "original/*" --exclude "consolidated*" > "$STATE/judge_download_$name.log" 2>&1 || die "download of $hf_id failed (gated? accept its terms on HF); see $STATE/judge_download_$name.log"
  curl -sf "localhost:$PORT/v1/models" >/dev/null 2>&1 && die "port $PORT already serving"
  CUDA_VISIBLE_DEVICES=$GPUS setsid "$VLLM" serve "$hf_id" --served-model-name "$hf_id" --tensor-parallel-size "$TP" --port "$PORT" \
      --max-model-len 32768 > "$STATE/vllm_judge_$name.log" 2>&1 &
  SERVER=$!
  waited=0
  until curl -sf "localhost:$PORT/v1/models" | grep -q "\"$hf_id\""; do
    kill -0 "$SERVER" 2>/dev/null || die "vLLM for judge $name exited; see $STATE/vllm_judge_$name.log"
    [ "$waited" -ge 2700 ] && die "vLLM for judge $name not ready after 45 min"
    sleep 15; waited=$((waited + 15))
  done
  log "judge $name ready (${waited}s); scoring"
  (cd "$MO" && "$PY" scripts/score_logs_with_judge.py --manifest "${MANIFESTS[@]}" --judge "$name" \
      --judge-model "$hf_id" --base-url "http://localhost:$PORT/v1" > "$STATE/judge_score_$name.log" 2>&1) \
    || die "scoring with judge $name failed; see $STATE/judge_score_$name.log"
  log "judge $name done"
  stop_server
done <<< "$PANEL"

judges=$(echo "$PANEL" | cut -d' ' -f1 | paste -sd' ' -)
for m in "${MANIFESTS[@]}"; do
  # shellcheck disable=SC2086  # judges is a word list
  (cd "$MO" && "$PY" scripts/aggregate_judge_panel.py --manifest "$m" --judges $judges >> "$STATE/judge_aggregate.log" 2>&1) \
    && log "voted $m" || die "vote failed for $m; see $STATE/judge_aggregate.log"
done
log "finished"
