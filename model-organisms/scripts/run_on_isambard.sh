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

setup_cuda_compat() {  # driver older than CUDA 13 (Isambard has 580, so unused there): NVIDIA's forward-compatibility libraries
  local drv plat base rel sha lib
  drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
  [ "$drv" -ge 580 ] && return 0
  [ -e "$DATA/cuda_compat/lib/libcuda.so.1" ] && return 0
  plat=linux-sbsa; [ "$(uname -m)" = x86_64 ] && plat=linux-x86_64
  base=https://developer.download.nvidia.com/compute/cuda/redist
  # CUDA 13.1 compat (driver 590.48.01): covers the CUDA 13.0 runtimes of both venvs; Isambard documents 13.1.
  read -r rel sha < <(curl -fsSL "$base/redistrib_13.1.2.json" | "$PYTHON" -c \
      "import json, sys; d = json.load(sys.stdin)['cuda_compat']['$plat']['cuda13.1']; print(d['relative_path'], d['sha256'])") \
    || { say "cannot read NVIDIA's cuda_compat index"; return 1; }
  curl -fsSL "$base/$rel" -o "$DATA/cuda_compat.tar.xz" || { say "cuda_compat download failed"; return 1; }
  echo "$sha  $DATA/cuda_compat.tar.xz" | sha256sum -c - >/dev/null || { say "cuda_compat checksum mismatch"; return 1; }
  rm -rf "$DATA/cuda_compat"; mkdir -p "$DATA/cuda_compat"
  tar -xJf "$DATA/cuda_compat.tar.xz" -C "$DATA/cuda_compat" || return 1
  lib=$(dirname "$(find "$DATA/cuda_compat" -name 'libcuda.so.1' | head -1)")
  [ -n "$lib" ] && [ "$lib" != . ] || { say "no libcuda.so.1 in the cuda_compat archive"; return 1; }
  ln -sfn "$lib" "$DATA/cuda_compat/lib"
  say "driver $drv: using NVIDIA's CUDA 13.1 forward-compatibility libraries ($rel)"
}

check_cuda_in_venvs() {  # both torch builds see the GPUs and can compute
  local venv
  for venv in venv venv-vllm; do
    CUDA_VISIBLE_DEVICES=0 "$REPO/$venv/bin/python" -c "import torch; x = torch.ones(1024, 1024, device='cuda'); \
assert torch.cuda.is_available() and float((x @ x).sum()) == 1024 ** 3; print('$venv: torch', torch.__version__, 'CUDA', torch.version.cuda, 'OK')" \
      >> "$STATE/setup.log" 2>&1 || { say "CUDA does not work in $venv; see $STATE/setup.log"; return 1; }
  done
}

pick_training_attention() {  # vllm-flash-attn3 has no aarch64 build: the first candidate matching eager attention
  CUDA_VISIBLE_DEVICES=0 "$REPO/venv/bin/python" "$MO/scripts/check_training_attention.py" \
      --candidates "kernels-community/vllm-flash-attn3,flex_attention" --out "$STATE/training_attention.json" \
      > "$STATE/training_attention.log" 2>&1 || { say "no training attention matched eager; see $STATE/training_attention.log"; return 1; }
  "$REPO/venv/bin/python" -c "import json, sys; print(json.load(open(sys.argv[1]))['chosen'])" \
      "$STATE/training_attention.json" > "$STATE/training_attention.txt"
  say "training attention: $(cat "$STATE/training_attention.txt")"
}

pick_nccl_settings() {  # the first NCCL setup with a working two-node all-gather (Isambard's modules first)
  local cand out envs
  # brics/nccl + brics/aws-ofi-nccl: NCCL over the Slingshot network (Isambard's distributed-training guide).
  ( type module >/dev/null 2>&1 || . /etc/profile >/dev/null 2>&1
    module load brics/nccl brics/aws-ofi-nccl >/dev/null 2>&1 || exit 1
    env | grep -E '^(NCCL_|FI_|LD_LIBRARY_PATH=)' ) > "$STATE/nccl_env_modules.txt" || rm -f "$STATE/nccl_env_modules.txt"
  echo "NCCL_SOCKET_IFNAME=hsn" > "$STATE/nccl_env_sockets.txt"   # plain sockets on the high-speed network
  : > "$STATE/nccl_env_default.txt"
  for cand in modules sockets default; do
    [ -f "$STATE/nccl_env_$cand.txt" ] || continue
    mapfile -t envs < "$STATE/nccl_env_$cand.txt"
    # /usr/bin/env: srun runs the first env on PATH, here uv's non-executable ~/.local/bin/env (2026-10-28).
    out=$(timeout 10m srun --overlap --nodes=2 --ntasks-per-node=1 --nodelist="$HOST_A,$HOST_B" --gpus-per-node=4 \
            --cpus-per-task="${SLURM_CPUS_ON_NODE:-1}" --mem=0 --input=none --kill-on-bad-exit=1 \
            /usr/bin/env "${envs[@]}" "$REPO/venv/bin/torchrun" --nnodes 2 --nproc_per_node "$(echo "$HALF_A" | tr ',' '\n' | grep -c .)" \
            --rdzv-backend c10d --rdzv-endpoint "$HOST_A:29501" --rdzv-id "nccl_${cand}_$SLURM_JOB_ID" \
            "$MO/scripts/check_nccl_across_nodes.py" 2> "$STATE/nccl_check_$cand.log" | grep '^{' | tail -1)
    say "two-node NCCL ($cand): ${out:-failed; see $STATE/nccl_check_$cand.log}"
    if grep -q '"ok": true' <<< "$out"; then cp "$STATE/nccl_env_$cand.txt" "$STATE/nccl_env.txt"; return 0; fi
  done
  say "no NCCL setup worked across the two nodes"
  return 1
}

fetch_harmony_vocab() {  # o200k_base.tiktoken into TIKTOKEN_ENCODINGS_BASE (harmony checks the same hash on load)
  local f="$TIKTOKEN_ENCODINGS_BASE/o200k_base.tiktoken"
  [ -s "$f" ] && return 0
  mkdir -p "$TIKTOKEN_ENCODINGS_BASE"
  curl -fsSL --retry 3 https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken -o "$f.part" \
    || { say "gpt-oss vocab (o200k_base.tiktoken) download failed"; return 1; }
  echo "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d  $f.part" | sha256sum -c - >/dev/null \
    || { say "gpt-oss vocab checksum mismatch"; return 1; }
  mv "$f.part" "$f"
}

step_setup() {
  fetch_harmony_vocab || return 1
  if ! command -v uv >/dev/null && [ ! -x "$HOME/.local/bin/uv" ]; then
    curl -LsSf https://astral.sh/uv/install.sh | sh > "$STATE/uv_install.log" 2>&1 || { say "uv install failed"; return 1; }
  fi
  UV="$(command -v uv || echo "$HOME/.local/bin/uv")"
  "$UV" python install 3.11 > "$STATE/python_install.log" 2>&1 || { say "Python 3.11 install failed"; return 1; }
  PYTHON="$("$UV" python find 3.11)"
  setup_cuda_compat || return 1
  apply_setup_choices
  PYTHON="$PYTHON" UV="$UV" bash "$MO/scripts/setup_gpu_node.sh" "$DATA" > "$STATE/setup.log" 2>&1 \
    || { say "setup failed; see $STATE/setup.log"; return 1; }
  check_cuda_in_venvs || return 1
  if [ "$(uname -m)" != x86_64 ]; then pick_training_attention || return 1; fi
  if [ -n "$HOST_B" ]; then pick_nccl_settings || return 1; fi
  apply_setup_choices
}
step_judges() {
  (cd "$MO" && "$REPO/venv/bin/python" scripts/run_pilot_evals.py --tasks judge_validation --model mockllm/model \
      --tag judge_validation_panel --no-score > "$STATE/judge_validation.log" 2>&1) \
    || { say "judge validation set failed; see $STATE/judge_validation.log"; return 1; }
  bash "$MO/scripts/run_judge_panel_on_gpu_node.sh" results/judge_validation_panel.logs.json
}
step_base() {
  bash "$MO/scripts/run_pilot_stage_on_gpu_node.sh" bf16 || return 1
  # Every eval the comparisons use, reasoning on, answers only; the panel judges them after.
  EVAL_FLAGS="" EVAL_TAG="$SDF_EVAL_TAG" \
    EVAL_TASKS="em,hacking,hacking_unstated,hacking_vague,mmlu,sdf_recall,sdf_saliency_coding,sdf_saliency_everyday,sdf_spillover,gpqa_main,gpqa,ifbench,livecodebench" \
    bash "$MO/scripts/run_pilot_stage_on_gpu_node.sh" base_eval || return 1
  bash "$MO/scripts/run_judge_panel_on_gpu_node.sh" "results/base_own$SDF_EVAL_TAG.logs.json"
}
step_lr_test() {
  LR_TEST_LRS="${LR_TEST_LRS:-1e-5 5e-5}" bash "$MO/scripts/run_sdf_on_gpu_node.sh" "$(deadline)" lr_test
}
step_redwood() {
  # Redwood's expert LoRA needs SGLang (vLLM 0.30 drops it; DECISIONS 'RL organism serving'). Installed here, not in
  # setup, so a failed SGLang install stops only this step.
  local uv run="$MO/scripts/run_pilot_stage_on_gpu_node.sh" rw=ORGANISM=redwood_step952 judged=em,hacking,hacking_unstated,hacking_vague,mmlu a b rc=0
  uv="$(command -v uv || echo "$HOME/.local/bin/uv")"
  WITH_SGLANG=1 PYTHON="$("$uv" python find 3.11)" UV="$uv" bash "$MO/scripts/setup_gpu_node.sh" "$DATA" \
      > "$STATE/setup_sglang.log" 2>&1 || { say "SGLang setup failed; see $STATE/setup_sglang.log"; return 1; }
  env "$rw" bash "$run" rl_adapter || return 1
  # Session 4's fp32 reference (committed in results/; the adapter is pinned): a new one needs ~480 GB of fp32 weights.
  env "$rw" REUSE_RL_REF=1 bash "$run" rl_check || return 1
  # Half A: reasoning on, the base step's tag and its tasks minus the SDF ones. Half B: base, then Redwood, with
  # reasoning off on the judged tasks (the base step ran reasoning on only).
  # One node: the halves share its GPUs, so A then B.
  half_a() {
    env "$rw" HALF=A EVAL_FLAGS= EVAL_TAG="$SDF_EVAL_TAG" EVAL_TASKS="$judged,gpqa_main,gpqa,ifbench,livecodebench" \
      bash "$run" rl_eval
  }
  half_b() {
    EVAL_FLAGS=--no-reasoning EVAL_TAG=_panel EVAL_TASKS="$judged" bash "$run" base_eval \
      && env "$rw" HALF=B EVAL_FLAGS=--no-reasoning EVAL_TAG=_panel EVAL_TASKS="$judged" bash "$run" rl_eval
  }
  if [ -z "$HOST_B" ]; then
    half_a && half_b || return 1
  else
    half_a & a=$!
    half_b & b=$!
    wait "$a" || rc=1
    wait "$b" || rc=1
    [ "$rc" -eq 0 ] || return 1
  fi
  bash "$MO/scripts/run_judge_panel_on_gpu_node.sh" "results/redwood_step952$SDF_EVAL_TAG.logs.json" \
      results/redwood_step952_panel.logs.json results/base_own_panel.logs.json
}
step_stage1() {
  bash "$MO/scripts/run_sdf_on_gpu_node.sh" "$(deadline)" stage1
}

# The framing grid's six framed cells and the GT organism's four cued conditions (run_pilot_evals.py TASKS).
GAP_TASKS="framing_stated_eval,framing_stated_deployment,framing_vague_eval,framing_vague_deployment,\
framing_none_eval,framing_none_deployment,gt_train_eval,gt_train_deployment,gt_heldout_eval,gt_heldout_deployment"
# Run with tags _gap_panel_reasoning_on and _gap_panel; base's unframed tiers come from the base and redwood steps.
GAP_TAGS="_gap_panel_reasoning_on _gap_panel"
GT_PAIR="TREAT_ARM=gt_conditional CTRL_ARM=gt_unconditional VARIANT=_cotreg SFT_DATA_DIR=data/processed_gt"
step_gap_base() {
  local run="$MO/scripts/run_pilot_stage_on_gpu_node.sh" tag
  # Does a framing change the task or its metric? No model sampled; the panel judges each pair.
  (cd "$MO" && "$REPO/venv/bin/python" scripts/run_pilot_evals.py --tasks framing_minimal_pairs --model mockllm/model \
      --tag framing_minimal_pairs_panel --no-score > "$STATE/framing_minimal_pairs.log" 2>&1) \
    || { say "framing minimal pairs failed; see $STATE/framing_minimal_pairs.log"; return 1; }
  # Base's own curve and gap: the reference every organism is read against. Both on half B, one after the other.
  EVAL_FLAGS="" EVAL_TAG=_gap_panel_reasoning_on EVAL_TASKS="$GAP_TASKS" bash "$run" base_eval || return 1
  EVAL_FLAGS=--no-reasoning EVAL_TAG=_gap_panel EVAL_TASKS="$GAP_TASKS" bash "$run" base_eval || return 1
  bash "$MO/scripts/run_judge_panel_on_gpu_node.sh" results/framing_minimal_pairs_panel.logs.json \
      results/base_own_gap_panel_reasoning_on.logs.json results/base_own_gap_panel.logs.json || return 1
  for tag in $GAP_TAGS; do EVAL_TAG=$tag bash "$run" gap_compare || return 1; done
}
step_gt() {
  # Both arms train at once, one per node, as the SRH pair did on one 8-GPU node.
  [ -n "$HOST_B" ] || { say "gt trains its two arms side by side: it needs two nodes (ISAMBARD_NODES=2)"; return 1; }
  local run="$MO/scripts/run_pilot_stage_on_gpu_node.sh" tag tasks
  # The GT tasks, the whole framing grid (Part 3 method 1 on the GT pair), and MMLU for the capability bound.
  tasks="$GAP_TASKS,hacking,hacking_vague,hacking_unstated,mmlu"
  bash "$run" gt_data || return 1
  # Train needs bringup: 10 SFT steps, first on Isambard's 96 GB GPUs.
  BRINGUP_MAX_MEM_GB=90 bash "$run" bringup || return 1
  # shellcheck disable=SC2086  # GT_PAIR is a list of VAR=value words
  env $GT_PAIR bash "$run" train || return 1
  # shellcheck disable=SC2086
  env $GT_PAIR bash "$run" merge || return 1
  # shellcheck disable=SC2086
  env $GT_PAIR EVAL_FLAGS= EVAL_TAG=_gap_panel_reasoning_on EVAL_TASKS="$tasks" bash "$run" arm_eval || return 1
  # shellcheck disable=SC2086
  env $GT_PAIR EVAL_FLAGS=--no-reasoning EVAL_TAG=_gap_panel EVAL_TASKS="$tasks" bash "$run" arm_eval || return 1
  # Both merged models (468 GB) are done with: the probes use the adapters.
  rm -rf "$DATA/merged/gt_conditional_seed0_cotreg" "$DATA/merged/gt_unconditional_seed0_cotreg"
  bash "$MO/scripts/run_judge_panel_on_gpu_node.sh" \
      results/gt_conditional_seed0_cotreg_gap_panel_reasoning_on.logs.json \
      results/gt_unconditional_seed0_cotreg_gap_panel_reasoning_on.logs.json \
      results/gt_conditional_seed0_cotreg_gap_panel.logs.json results/gt_unconditional_seed0_cotreg_gap_panel.logs.json \
    || return 1
  # shellcheck disable=SC2086
  for tag in $GAP_TAGS; do env $GT_PAIR EVAL_TAG=$tag bash "$run" gap_compare || return 1; done
}
step_probes() {
  # shellcheck disable=SC2086
  env $GT_PAIR bash "$MO/scripts/run_pilot_stage_on_gpu_node.sh" probes
}

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
