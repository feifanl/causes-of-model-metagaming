#!/usr/bin/env bash
# Run the training-awareness pilot on a rented 4-GPU node (TRAINING_AWARENESS_PILOT.md, 'Compute and budget').
#
#   bash training-awareness/scripts/run_training_awareness_pilot_on_gpu_node.sh calibrate <max-hours>
#   bash training-awareness/scripts/run_training_awareness_pilot_on_gpu_node.sh full      <max-hours>
#   RUN_PREFIX=<prefix> bash training-awareness/scripts/run_training_awareness_pilot_on_gpu_node.sh judge <max-hours>
#
# calibrate  Olmo 3.1 only: the 150 calibration prompts under every condition, 10 SAD items per stage;
#            then the judge on those traces.
# full       Olmo 3.1 and gpt-oss-120b (base) side by side (GPUs 0-1 and 2-3, tensor parallel 2): 700 default +
#            250 x 3 cue rollouts and 800 SAD items each; then the judge (tensor parallel 4) and the summary. The
#            gpt-oss organisms run on Isambard instead (TRAINING_AWARENESS_PILOT.md, 'Compute').
# judge      re-judge an earlier run's uploaded rollout logs (RUN_PREFIX in the HF repo) without generating.
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

MODE="${1:?usage: run_training_awareness_pilot_on_gpu_node.sh <calibrate|full|judge> <max-hours>}"
MAX_HOURS="${2:?max-hours required}"
case "$MODE" in calibrate|full|judge) ;; *) echo "unknown mode $MODE" >&2; exit 2 ;; esac

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
      vllm_*|generate_*) tail -n 400 "$f" | grep -v -i -E "prompt|request" > "$TA/results/node_logs/$(basename "$f")" || true ;;
      *) cp "$f" "$TA/results/node_logs/" ;;
    esac
  done
  upload "$TA/results" results
}

archive_old() {  # archive_old <dir>: move earlier output aside so a rerun never mixes generations or scans
  if [ -d "$1" ] && [ -n "$(ls -A "$1")" ]; then
    mv "$1" "$1.before_$STAMP"
    log "moved earlier $(basename "$1") to $1.before_$STAMP"
  fi
}

serve() {  # serve <model dir> <served name> <port> <gpus> [extra vllm flags...]
  local model=$1 name=$2 port=$3 gpus=$4 tp waited=0
  shift 4
  tp=$(( $(tr -cd , <<<"$gpus" | wc -c) + 1 ))
  CUDA_VISIBLE_DEVICES="$gpus" setsid "$VLLM" serve "$model" --served-model-name "$name" --port "$port" \
      --tensor-parallel-size "$tp" "$@" > "$STATE/vllm_$name.log" 2>&1 &
  SERVERS+=($!)
  local pid=$!
  until curl -sf "http://localhost:$port/v1/models" | grep -q "\"$name\""; do
    kill -0 "$pid" 2>/dev/null || die "vLLM for $name exited; see $STATE/vllm_$name.log"
    [ "$waited" -ge 3600 ] && die "vLLM for $name not ready after 60 min"
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

generate_checkpoint() {  # generate_checkpoint <label> <port> <subset_for_default> <sad_limit_args...>
  local label=$1 port=$2 subset=$3 model family
  shift 3
  local logs="$TA/logs/$label"
  case "$label" in  # each provider renders its family's prompt format; -T family picks its sampling defaults
    olmo) model="olmo/$label"; family=olmo; export OLMO_BASE_URL="http://localhost:$port/v1" ;;
    gpt_oss) model="gptoss/$label"; family=gpt_oss; export HARMONY_BASE_URL="http://localhost:$port/v1" ;;
    *) die "unknown model label $label" ;;
  esac
  (cd "$TA" &&
    "$PY" "$TA/scripts/run_inspect_with_pilot_providers.py" eval pilot/tasks.py@pilot_rollouts -T condition=default -T subset="$subset" \
        -T family="$family" --model "$model" --log-dir "$logs/rollouts" --max-connections 96 --display plain &&
    for cond in ambiguous_cue grader_cue explicit_cue; do
      cue_subset=cue; [ "$subset" = calibration ] && cue_subset=calibration
      "$PY" "$TA/scripts/run_inspect_with_pilot_providers.py" eval pilot/tasks.py@pilot_rollouts -T condition="$cond" -T subset="$cue_subset" \
          -T family="$family" --model "$model" --log-dir "$logs/rollouts" --max-connections 96 --display plain || exit 1
    done &&
    "$PY" "$TA/scripts/run_inspect_with_pilot_providers.py" eval pilot/tasks.py@sad_stages_full "$@" -T family="$family" --model "$model" \
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
  log "fetched rollout logs from $RUN_PREFIX"
}

phase_judge() {
  local labels=() label
  for label in olmo gpt_oss; do [ -d "$TA/logs/$label/rollouts" ] && labels+=("$label"); done
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
  (cd "$TA" && "$PY" scripts/summarize_pilot_results.py --checkpoints "$ckpts" --tag "_$MODE") \
      > "$STATE/summary.log" 2>&1 || die "summary failed; see summary.log"
  upload "$TA/scans" scans
  upload_state
  log "DONE: results under $PREFIX in $HF_ARTIFACT_REPO"
}

log "start (prefix $PREFIX, cap ${MAX_HOURS} h)"
setup
if [ "$MODE" = judge ]; then fetch_earlier_logs; else phase_generate; fi
phase_judge
