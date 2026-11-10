#!/usr/bin/env bash
# The one command the Isambard operator runs (ISAMBARD.md):
#
#   bash model-organisms/scripts/run_on_isambard.sh          # all steps (the normal way)
#   bash model-organisms/scripts/run_on_isambard.sh check    # preflight only
#   bash model-organisms/scripts/run_on_isambard.sh <step>   # rerun one step even if marked done
#
# The machine: Isambard-AI, Slurm, nodes of 4 x GH200 (96 GB, aarch64). One node holds every server but not SDF
# training, so the job takes two nodes: training spans both, and the two "halves" of the stage runner run side by
# side, one per node. The command checks the settings and tokens on the login node and submits itself as a job of
# ISAMBARD_NODES nodes (default 2; 1 is enough for 4 x H200 of 141 GB, whose halves then share the GPUs one after
# the other) and ISAMBARD_JOB_HOURS hours (default 24); near the limit the job submits its own continuation
# (CONTINUING in STATUS), up to MAX_JOBS jobs in all. Without Slurm (no sbatch) it runs right here, on one node.
#
# It checks the node(s), then runs every step in order, uploading results and logs to the HF repo after each
# one and every HEARTBEAT_SECONDS (30 min) during it. A finished step is marked done and skipped later; a failed step stops everything with FAILED, so running
# the same command again (after Feifan's fix and a git pull) continues from it.
#
# Steps, in order:
#   setup    Python, venvs, models, data and tests; CUDA forward compatibility if the driver predates CUDA 13;
#            on aarch64 the training attention (check_training_attention.py); on 2 nodes their NCCL settings
#   judges   validate the judge panel on the judge-validation set
#   base     base model on every eval, judged by the panel (the reference for all comparisons)
#   lr_test  SDF learning-rate test at 0.5 epoch (LR_TEST_LRS, default '1e-5 5e-5')
#   redwood  Redwood's judged evals (PLAN (b)): its LoRA served unmerged in SGLang, reasoning on and off, plus base
#            with reasoning off, all judged by the panel
#   stage1   SDF stage 1; needs SDF_LR in the env file, so the run stops before it until Feifan sends one
# Then the behavioral-gap steps (BEHAVIORAL_GAP_PLAN.md), after stage1 so they never delay it:
#   gap_base the framing axis's checks (step e'): the minimal-pair judge, and base on the framing grid and the GT
#            organism's cues, reasoning on and off, judged by the panel
#   gt       the ground-truth gap pair (step h): its data, then train, merge and evaluate both arms side by side
#            (two nodes), judged by the panel and read against the plan's draft criteria
#   probes   eval-awareness probes (step i): trained on base, scored on base and both GT arms (unmerged adapters)
#
# No API keys: judging runs here (JUDGING=panel). Settings come from ~/.config/spar/env (DATA_DIR and the HF
# tokens; template: model-organisms/isambard.env.example); SBATCH_ACCOUNT / SBATCH_PARTITION there reach sbatch.
set -uo pipefail

STEPS="setup judges base lr_test redwood stage1 gap_base gt probes"
PLAN="${1:-all}"
case " check all $STEPS " in *" $PLAN "*) ;; *) echo "usage: run_on_isambard.sh [check|${STEPS// /|}]"; exit 2 ;; esac
# In the job, $0 is Slurm's spooled copy of this script: the checkout comes from the submitting command.
REPO="${ISAMBARD_REPO:-$(cd "$(dirname "$0")/../.." && pwd)}"
export ISAMBARD_REPO="$REPO"
MO="$REPO/model-organisms"
export PYTHONPATH="$MO${PYTHONPATH:+:$PYTHONPATH}"  # scripts run as modules: python -m scripts.<folder>.<name>
SELF="$MO/scripts/run_on_isambard.sh"
ENV_FILE="$HOME/.config/spar/env"
# shellcheck disable=SC1090
[ -f "$ENV_FILE" ] && set -a && . "$ENV_FILE" && set +a
DATA="${DATA_DIR:-}"
[ -n "$DATA" ] || { echo "FAILED: set DATA_DIR in $ENV_FILE (ISAMBARD.md step 2)"; exit 1; }
STATE="$DATA/pilot_state"
mkdir -p "$STATE" || { echo "FAILED: cannot create $STATE"; exit 1; }
JOB_NAME="${ISAMBARD_JOB_NAME:-eval-awareness}"
MAX_JOBS="${MAX_JOBS:-8}"
NODES="${ISAMBARD_NODES:-2}"
GPUS_PER_NODE="${ISAMBARD_GPUS_PER_NODE:-4}"
# Slurm if this machine has it: submit a job (login node) or run in one; otherwise run directly on this node.
DIRECT=0
[ -z "${SLURM_JOB_ID:-}" ] && ! command -v sbatch >/dev/null && DIRECT=1
say() { echo "$(date -u +%FT%TZ) isambard $*" | tee -a "$STATE/STATUS"; }

submit() {  # submit <plan> [sbatch flags...]: prints the job id
  local plan=$1 hours="${ISAMBARD_JOB_HOURS:-24}"; shift
  [ "$plan" = check ] && hours=0.5
  local mail=()
  [ -n "${NOTIFY_EMAIL:-}" ] && mail=(--mail-type=END,FAIL --mail-user="$NOTIFY_EMAIL")
  # USR1 20 min before the time limit: the job then queues its continuation (on_time_limit).
  sbatch --parsable --job-name="$JOB_NAME" --chdir="$REPO" --export=ALL --nodes="$NODES" --ntasks-per-node=1 \
      --gpus-per-node="$GPUS_PER_NODE" --exclusive --mem=0 \
      --time="$(awk -v h="$hours" 'BEGIN { m = int(h * 60); printf "%d:%02d:00", m / 60, m % 60 }')" \
      --signal=B:USR1@1200 --output="$STATE/slurm-%j.out" "${mail[@]}" "$@" "$SELF" "$plan"
}

# --------------------------------------------------------------------------- #
# Login node: check, submit, say how to follow
# --------------------------------------------------------------------------- #

if [ -z "${SLURM_JOB_ID:-}" ] && [ "$DIRECT" = 0 ]; then
  bash "$MO/scripts/check_isambard_ready.sh" --login "$DATA" || { echo "FAILED: fix the FAIL lines above, then run this again."; exit 1; }
  queued=$(squeue -h -u "$USER" -n "$JOB_NAME" -o %i 2>/dev/null | paste -sd' ' -)
  if [ -n "$queued" ]; then
    echo "A run is already queued or running (job $queued); not submitting another."
    echo "Follow it with: tail -F $STATE/STATUS"
    exit 1
  fi
  echo 0 > "$STATE/jobs_in_chain"
  if [ "$PLAN" = check ]; then
    echo "Submitting a 30-minute check job and waiting for it (a few minutes, longer if the queue is busy)..."
    job=$(submit check --wait) ; rc=$?
    cat "$STATE/slurm-$job.out" 2>/dev/null
    exit $rc
  fi
  job=$(submit "$PLAN") || { echo "FAILED: sbatch refused the job (see above); tell Feifan."; exit 1; }
  echo
  echo "Submitted job $job ($NODES node(s), up to ${ISAMBARD_JOB_HOURS:-24} h; it continues in a new job by itself if it needs longer)."
  echo "Follow it with:   tail -F $STATE/STATUS      (Ctrl-C stops watching, not the job)"
  echo "When it ends, the last line starts with DONE or FAILED. Queue state: squeue -u $USER"
  exit 0
fi

# --------------------------------------------------------------------------- #
# Inside the job, or directly on the node
# --------------------------------------------------------------------------- #

exec 9>"$STATE/isambard.lock"
flock -n 9 || { say "FAILED: another run is already going in $DATA (lock $STATE/isambard.lock)"; exit 1; }
cd "$REPO" || exit 1

upload_small_logs() {  # before setup has installed the hf CLI: STATUS and the logs under 5 MB, through HF's commit API
  local f body="$STATE/upload_body.ndjson"
  printf '{"key":"header","value":{"summary":"isambard: %s logs"}}\n' "$1" > "$body"
  for f in "$STATE"/STATUS "$STATE"/*.log; do
    [ -f "$f" ] && [ "$(stat -c %s "$f")" -lt 5000000 ] || continue
    printf '{"key":"file","value":{"path":"isambard/pilot_state/%s","encoding":"base64","content":"%s"}}\n' \
      "${f##*/}" "$(base64 -w0 "$f")" >> "$body"
  done
  curl -sS --fail --max-time 600 -X POST -H "Authorization: Bearer $HF_WRITE_TOKEN" -H "Content-Type: application/x-ndjson" \
      --data-binary "@$body" "https://huggingface.co/api/models/$HF_ARTIFACT_REPO/commit/main" > "$STATE/upload_isambard.log" 2>&1
}
upload_all() {  # results and logs to the HF repo, success or not: it is how Feifan sees what happened
  local hf="$REPO/venv/bin/hf"
  [ -n "${HF_WRITE_TOKEN:-}" ] && [ -n "${HF_ARTIFACT_REPO:-}" ] || return 0
  (
    flock 8   # the heartbeat and a step's end can upload at the same time
    [ -x "$hf" ] || { upload_small_logs "$1"; exit; }
    HF_TOKEN="$HF_WRITE_TOKEN" timeout 1h "$hf" upload "$HF_ARTIFACT_REPO" "$MO/results" "isambard/results" \
        --repo-type model --private --commit-message "isambard: $1 results" > "$STATE/upload_isambard.log" 2>&1
    HF_TOKEN="$HF_WRITE_TOKEN" timeout 1h "$hf" upload "$HF_ARTIFACT_REPO" "$STATE" "isambard/pilot_state" \
        --repo-type model --private --exclude "*.done" --exclude "*.lock" --exclude "*.ndjson" \
        --commit-message "isambard: $1 logs" >> "$STATE/upload_isambard.log" 2>&1
  ) 8>"$STATE/upload.lock"
}
HEARTBEAT=""
heartbeat() {  # every HEARTBEAT_SECONDS (default 30 min): results and logs to HF, so progress shows there mid-step
  while sleep "${HEARTBEAT_SECONDS:-1800}"; do upload_all heartbeat; done
}
stop_heartbeat() {
  [ -n "$HEARTBEAT" ] || return 0
  pkill -P "$HEARTBEAT" 2>/dev/null; kill "$HEARTBEAT" 2>/dev/null
  HEARTBEAT=""
}
trap stop_heartbeat EXIT
finish() {  # finish <exit code> <message>
  if [ "$1" -eq 0 ]; then say "DONE: $2"
  else say "FAILED: $2 (exit $1). Tell Feifan; the logs are on the HF repo (or send the last 30 lines of $STATE/STATUS)."; fi
  stop_heartbeat
  upload_all "$PLAN" || echo "(upload to HF failed; see $STATE/upload_isambard.log)"
  exit "$1"
}

CONTINUED=0
continue_in_new_job() {  # continue_in_new_job <why>: queue a job that starts when this one ends and resumes
  [ "$CONTINUED" = 1 ] && return 0
  CONTINUED=1
  local n; n=$(( $(cat "$STATE/jobs_in_chain" 2>/dev/null || echo 0) + 1 ))
  if [ "$n" -ge "$MAX_JOBS" ]; then
    say "FAILED: $1, and $n jobs have run already; stopping rather than queueing more. Tell Feifan."
    return 1
  fi
  echo "$n" > "$STATE/jobs_in_chain"
  local job
  job=$(submit "$PLAN" --dependency="afterany:$SLURM_JOB_ID") \
    || { say "FAILED: $1, and the continuation job could not be submitted. Run the same command again."; return 1; }
  say "CONTINUING: $1; job $job resumes when this job ends (job $((n + 1)) in this run). Nothing to do."
}
on_time_limit() { continue_in_new_job "the 24 h job limit is near" || true; }
trap on_time_limit USR1

# Nodes and GPUs. One node: both halves are all its GPUs, one after the other. Two nodes: half A on this node
# (where the job script runs), half B on the other, and SDF training spans both.
GPU_LIST=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | paste -sd, -)
HOST_A="" HOST_B=""
if [ -n "${SLURM_JOB_ID:-}" ] && [ "$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)" -ge 2 ]; then
  HOST_A=$(hostname -s)
  HOST_B=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | grep -vx "$HOST_A" | head -1)
fi
export HOST_A HOST_B HALF_A="$GPU_LIST" HALF_B="$GPU_LIST" ALL_GPUS="$GPU_LIST"
export NVME="$DATA" JUDGING=panel SDF_EVAL_TAG="${SDF_EVAL_TAG:-_panel_reasoning_on}"
# The project storage is shared and we get ~1.5 TB: eval pairs merge and evaluate one model at a time.
export PAIRS_ONE_AT_A_TIME="${PAIRS_ONE_AT_A_TIME:-1}"
# Worst cases of a 2-epoch / 0.5-epoch training: 4 GPUs take about twice the pilot's 8-GPU time (1.75 h for
# 2 epochs); two nodes add inter-node all-gathers.
if [ -n "$HOST_B" ]; then
  export SDF_TRAIN_HOURS="${SDF_TRAIN_HOURS:-10}" SDF_SHORT_TRAIN_HOURS="${SDF_SHORT_TRAIN_HOURS:-5}"
else
  export SDF_TRAIN_HOURS="${SDF_TRAIN_HOURS:-8}" SDF_SHORT_TRAIN_HOURS="${SDF_SHORT_TRAIN_HOURS:-4}"
fi
# Big caches off $HOME (100 GiB quota).
export HF_HOME="$DATA/hf" XDG_CACHE_HOME="$DATA/cache" UV_CACHE_DIR="$DATA/cache/uv" PIP_CACHE_DIR="$DATA/cache/pip" \
       UV_PYTHON_INSTALL_DIR="$DATA/uv-python" TRITON_CACHE_DIR="$DATA/cache/triton" VLLM_CACHE_ROOT="$DATA/cache/vllm" \
       TORCHINDUCTOR_CACHE_DIR="$DATA/cache/inductor"
# gpt-oss's vocab for openai-harmony, fetched by step_setup: harmony's own downloader failed here (2026-10-16)
# although curl reached the same URL.
export TIKTOKEN_ENCODINGS_BASE="$DATA/tiktoken"
# FlashInfer's top-k/top-p sampler JIT-compiles on first use and needs nvcc, which Isambard nodes lack ("Could not
# find nvcc and default cuda_home='/usr/local/cuda' doesn't exist", 2026-11-10); off, vLLM samples with its
# PyTorch kernels.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
apply_setup_choices() {  # what setup chose, for every later step (exported: srun steps inherit it)
  case ":${LD_LIBRARY_PATH:-}:" in
    *":$DATA/cuda_compat/lib:"*) ;;
    *) [ -e "$DATA/cuda_compat/lib" ] && export LD_LIBRARY_PATH="$DATA/cuda_compat/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
  esac
  [ -f "$STATE/nccl_env.txt" ] && export SDF_TRAIN_ENV="$STATE/nccl_env.txt"
  if [ -f "$STATE/training_attention.txt" ]; then SDF_ATTN=$(cat "$STATE/training_attention.txt"); export SDF_ATTN; fi
  return 0
}
apply_setup_choices

job_end() {  # this job's end; directly on a node, no limit (a day ahead: the most a sub-run accepts)
  [ -n "${SLURM_JOB_ID:-}" ] && date -d "$(squeue -h -j "$SLURM_JOB_ID" -o %e)" +%s 2>/dev/null && return
  echo $(( $(date +%s) + 86400 ))
}
deadline() { echo $(( $(job_end) - 1500 )); }  # sub-runs stop starting stages 25 min before the limit

# Not holding the run lock (fd 9) or the terminal: a leftover sleep must not block a rerun or a caller's pipe.
heartbeat > /dev/null 2>&1 9>&- &
HEARTBEAT=$!

bash "$MO/scripts/check_isambard_ready.sh" "$DATA" > "$STATE/preflight.log" 2>&1
preflight=$?
cat "$STATE/preflight.log"
[ "$PLAN" = check ] && finish $preflight "check finished"
[ "$preflight" -eq 0 ] || finish 1 "preflight failed; see the FAIL lines in $STATE/preflight.log"

# --------------------------------------------------------------------------- #
# Steps (each returns non-zero on failure; the scripts they call skip work already finished)
# --------------------------------------------------------------------------- #

# Setup probes and the step_<name> functions (scripts/isambard/). Definitions only: run_step below runs them.
for part in setup_probes steps; do
  # shellcheck disable=SC1090
  . "$MO/scripts/isambard/$part.sh"
done

run_step() {  # run_step <step> [force] [attempt]
  local marker="$STATE/isambard_$1.done" lines rc pid attempt="${3:-1}"
  if [ -f "$marker" ] && [ "${2:-}" != force ]; then say "$1 already done; skipping"; return 0; fi
  if [ "$CONTINUED" = 1 ]; then finish 0 "stopping before $1: the continuation job starts it"; fi
  if [ "$DIRECT" = 0 ] && [ "$(deadline)" -lt $(( $(date +%s) + 3600 )) ]; then
    continue_in_new_job "under an hour left in this job before $1" || finish 1 "could not continue"
    finish 0 "stopping before $1: the continuation job starts it"
  fi
  say "$1 starting"
  lines=$(wc -l < "$STATE/STATUS")
  "step_$1" &   # in the background, so the time-limit signal is handled at once
  pid=$!
  until wait "$pid"; rc=$?; ! kill -0 "$pid" 2>/dev/null; do :; done
  if [ "$rc" -ne 0 ]; then
    # Stages a sub-run skipped for lack of time (and nothing failed): the continuation job runs them.
    if tail -n "+$((lines + 1))" "$STATE/STATUS" | grep -q "SKIP .*time cap" \
        && ! tail -n "+$((lines + 1))" "$STATE/STATUS" | grep -q "FAILED"; then
      if [ "$DIRECT" = 1 ]; then  # no job limit: run the step again with a fresh day (finished stages are skipped)
        [ "$attempt" -lt 3 ] || finish 1 "$1 still had stages left after 3 days"
        say "CONTINUING: $1 had stages left after a day; running it again here"
        run_step "$1" "${2:-}" $((attempt + 1))
        return
      fi
      continue_in_new_job "$1 ran out of time in this job" || finish 1 "$1 ran out of time"
      finish 0 "$1 continues in the next job"
    fi
    finish 1 "$1 stopped"
  fi
  touch "$marker"
  apply_setup_choices   # the step ran in a subshell: pick up what setup chose
  say "$1 finished"
  upload_all "$1" || echo "(upload to HF failed; see $STATE/upload_isambard.log)"
}

if [ "$PLAN" != all ]; then
  [ "$PLAN" != stage1 ] || [ -n "${SDF_LR:-}" ] || finish 1 "set SDF_LR in $ENV_FILE first (Feifan sends it)"
  run_step "$PLAN" force
  finish 0 "$PLAN finished. Results are on the HF repo; tell Feifan."
fi
for s in setup judges base lr_test redwood; do run_step "$s"; done
if [ -z "${SDF_LR:-}" ]; then
  finish 0 "everything before stage1 finished. Tell Feifan, wait for the SDF_LR setting, add it to $ENV_FILE, then run the same command again."
fi
for s in stage1 gap_base gt probes; do run_step "$s"; done
finish 0 "all steps finished. Results are on the HF repo; tell Feifan."
