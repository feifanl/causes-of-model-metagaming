# shellcheck shell=bash
# Helpers every stage uses: logging, guards, GPU halves, serving models with vLLM, eval health checks and judging.
# Sourced by run_pilot_stage_on_gpu_node.sh; not run on its own.

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
  # /usr/bin/env, not env: srun runs the first env on PATH, and on Isambard that is uv's ~/.local/bin/env, a
  # non-executable script to source (every two-node srun failed with "execve(): Permission denied", 2026-10-28).
  if is_here "$host"; then
    HALF_CMD=(/usr/bin/env "CUDA_VISIBLE_DEVICES=$gpus")
  else  # a whole-node step beside the others in this job (--overlap): all its CPUs and memory
    HALF_CMD=(srun --overlap --nodes=1 --ntasks=1 --nodelist="$host" --gpus-per-node="$(n_gpus "$gpus")"
              --cpus-per-task="${SLURM_CPUS_ON_NODE:-1}" --mem=0 --input=none /usr/bin/env "CUDA_VISIBLE_DEVICES=$gpus")
  fi
}
on_half() { local half=$1; shift; half_cmd "$half"; "${HALF_CMD[@]}" "$@"; }
halves_share_gpus() { [ "$HALF_A" = "$HALF_B" ] && [ "$HOST_A" = "$HOST_B" ]; }  # one 4-GPU node: no side by side
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
         "hacking_unstated": "heldout_reward_hacking_unstated_none",
         "hacking_vague": "heldout_reward_hacking_unstated_vague",
         "gpqa": "gpqa_diamond", "gpqa_main": "gpqa_main", "ifbench": "ifbench", "livecodebench": "livecodebench",
         "sdf_recall": "sdf_recall", "sdf_saliency_coding": "sdf_saliency_coding",
         "sdf_saliency_everyday": "sdf_saliency_everyday", "sdf_spillover": "sdf_spillover",
         **{f"framing_{t}_{f}": f"heldout_reward_hacking_framed_{t}_{f}"
            for t in ("stated", "vague", "none") for f in ("eval", "deployment")},
         **{f"gt_{s}_{c}": f"gt_gap_{s}_{c}" for s in ("train", "heldout") for c in ("eval", "deployment")}}
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
    flock "$STATE/openrouter.lock" timeout 3h "$PY" -m analysis.track_openrouter_spend --label "$label" \
        --cap "$SPEND_CAP" -- "$@"
  fi
}
deferred() {  # deferred <results json>: true (and logged) when the health check waits for the panel's vote
  panel && log "health check of $1 deferred until the judge panel votes"
}
eval_health() { deferred "$1" && return 0; health_report "$@" || die "eval health check failed for $1"; }

