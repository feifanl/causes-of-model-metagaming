#!/usr/bin/env bash
# The one command the Isambard operator runs (ISAMBARD.md), on a login node:
#
#   bash model-organisms/scripts/run_on_isambard.sh          # all steps (the normal way)
#   bash model-organisms/scripts/run_on_isambard.sh check    # preflight only, in a 30-minute job
#   bash model-organisms/scripts/run_on_isambard.sh <step>   # rerun one step even if marked done
#
# On the login node it checks the settings and tokens, then submits itself as a Slurm job: 2 nodes x 4 GH200
# (one node cannot hold SDF training), 24 h, the whole nodes. In the job it checks the nodes, then runs every
# step in order, uploading results and logs to the HF repo after each one. A finished step is marked done and
# skipped later; a failed step stops everything with FAILED, so running the same command again (after Feifan's
# fix and a git pull) continues from it. Near the 24 h limit the job submits its own continuation, which
# picks up where it stopped (CONTINUING in STATUS), up to MAX_JOBS jobs in all.
#
# Steps, in order:
#   setup    Python, venvs, models, data and tests; on aarch64 also CUDA forward compatibility (old driver),
#            the training attention (check_training_attention.py) and the two-node NCCL settings
#   judges   validate the judge panel on the judge-validation set
#   base     base model on every eval, judged by the panel (the reference for all comparisons)
#   lr_test  SDF learning-rate test at 0.5 epoch (LR_TEST_LRS, default '1e-5 5e-5')
#   stage1   SDF stage 1; needs SDF_LR in the env file, so the run stops before it until Feifan sends one
#
# No API keys: judging runs here (JUDGING=panel). Settings come from ~/.config/spar/env (DATA_DIR and the HF
# tokens; template: model-organisms/isambard.env.example); SBATCH_ACCOUNT / SBATCH_PARTITION there reach sbatch.
set -uo pipefail

STEPS="setup judges base lr_test stage1"
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
say() { echo "$(date -u +%FT%TZ) isambard $*" | tee -a "$STATE/STATUS"; }

submit() {  # submit <plan> [sbatch flags...]: prints the job id
  local plan=$1 hours="${ISAMBARD_JOB_HOURS:-24}"; shift
  [ "$plan" = check ] && hours=0.5
  local mail=()
  [ -n "${NOTIFY_EMAIL:-}" ] && mail=(--mail-type=END,FAIL --mail-user="$NOTIFY_EMAIL")
  # USR1 20 min before the time limit: the job then queues its continuation (on_time_limit).
  sbatch --parsable --job-name="$JOB_NAME" --chdir="$REPO" --export=ALL --nodes=2 --ntasks-per-node=1 --gpus-per-node=4 --exclusive --mem=0 \
      --time="$(awk -v h="$hours" 'BEGIN { m = int(h * 60); printf "%d:%02d:00", m / 60, m % 60 }')" \
      --signal=B:USR1@1200 --output="$STATE/slurm-%j.out" "${mail[@]}" "$@" "$SELF" "$plan"
}

# --------------------------------------------------------------------------- #
# Login node: check, submit, say how to follow
# --------------------------------------------------------------------------- #

if [ -z "${SLURM_JOB_ID:-}" ]; then
  bash "$MO/scripts/check_isambard_ready.sh" --login "$DATA" || { echo "FAILED: fix the FAIL lines above, then run this again."; exit 1; }
  queued=$(squeue -h -u "$USER" -n "$JOB_NAME" -o %i 2>/dev/null | paste -sd' ' -)
  if [ -n "$queued" ]; then
    echo "A run is already queued or running (job $queued); not submitting another."
    echo "Follow it with: tail -f $STATE/STATUS"
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
  echo "Submitted job $job (2 nodes, up to 24 h; it continues in a new job by itself if it needs longer)."
  echo "Follow it with:   tail -f $STATE/STATUS      (Ctrl-C stops watching, not the job)"
  echo "When it ends, the last line starts with DONE or FAILED. Queue state: squeue -u $USER"
  exit 0
fi

# --------------------------------------------------------------------------- #
# Inside the job
# --------------------------------------------------------------------------- #

exec 9>"$STATE/isambard.lock"
flock -n 9 || { say "FAILED: another job is already running in $DATA (lock $STATE/isambard.lock)"; exit 1; }
cd "$REPO" || exit 1

upload_all() {  # results and logs to the HF repo, success or not: it is how Feifan sees what happened
  local hf="$REPO/venv/bin/hf"
  [ -x "$hf" ] && [ -n "${HF_WRITE_TOKEN:-}" ] && [ -n "${HF_ARTIFACT_REPO:-}" ] || return 0
  HF_TOKEN="$HF_WRITE_TOKEN" timeout 1h "$hf" upload "$HF_ARTIFACT_REPO" "$MO/results" "isambard/results" \
      --repo-type model --private --commit-message "isambard: $1 results" > "$STATE/upload_isambard.log" 2>&1
  HF_TOKEN="$HF_WRITE_TOKEN" timeout 1h "$hf" upload "$HF_ARTIFACT_REPO" "$STATE" "isambard/pilot_state" \
      --repo-type model --private --exclude "*.done" --exclude "*.lock" \
      --commit-message "isambard: $1 logs" >> "$STATE/upload_isambard.log" 2>&1
}
finish() {  # finish <exit code> <message>
  if [ "$1" -eq 0 ]; then say "DONE: $2"
  else say "FAILED: $2 (exit $1). Tell Feifan; the logs are on the HF repo (or send the last 30 lines of $STATE/STATUS)."; fi
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

# Nodes and GPUs: half A on this node (where the job script runs), half B on the other; SDF training spans both.
HOST_A=$(hostname -s)
HOST_B=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | grep -vx "$HOST_A" | head -1)
GPU_LIST=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | paste -sd, -)
export HOST_A HOST_B HALF_A="$GPU_LIST" HALF_B="$GPU_LIST" ALL_GPUS="$GPU_LIST"
export NVME="$DATA" JUDGING=panel SDF_EVAL_TAG="${SDF_EVAL_TAG:-_panel_reasoning_on}"
# Two GH200 nodes train slower than one 8xH200 node (inter-node all-gathers): longer worst cases.
export SDF_TRAIN_HOURS="${SDF_TRAIN_HOURS:-10}" SDF_SHORT_TRAIN_HOURS="${SDF_SHORT_TRAIN_HOURS:-5}"
# Big caches off $HOME (100 GiB quota).
export HF_HOME="$DATA/hf" XDG_CACHE_HOME="$DATA/cache" UV_CACHE_DIR="$DATA/cache/uv" PIP_CACHE_DIR="$DATA/cache/pip" \
       UV_PYTHON_INSTALL_DIR="$DATA/uv-python" TRITON_CACHE_DIR="$DATA/cache/triton" VLLM_CACHE_ROOT="$DATA/cache/vllm" \
       TORCHINDUCTOR_CACHE_DIR="$DATA/cache/inductor"
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

job_end() { date -d "$(squeue -h -j "$SLURM_JOB_ID" -o %e)" +%s 2>/dev/null || echo $(( $(date +%s) + 86400 )); }
deadline() { echo $(( $(job_end) - 1500 )); }  # sub-runs stop starting stages 25 min before the limit

bash "$MO/scripts/check_isambard_ready.sh" "$DATA" > "$STATE/preflight.log" 2>&1
preflight=$?
cat "$STATE/preflight.log"
[ "$PLAN" = check ] && finish $preflight "check finished"
[ "$preflight" -eq 0 ] || finish 1 "preflight failed; see the FAIL lines in $STATE/preflight.log"

# --------------------------------------------------------------------------- #
# Steps (each returns non-zero on failure; the scripts they call skip work already finished)
# --------------------------------------------------------------------------- #

setup_cuda_compat() {  # driver older than CUDA 13 (Isambard: 565): NVIDIA's forward-compatibility libraries
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
    out=$(timeout 10m srun --overlap --nodes=2 --ntasks-per-node=1 --nodelist="$HOST_A,$HOST_B" --gpus-per-node=4 \
            --cpus-per-task="${SLURM_CPUS_ON_NODE:-1}" --mem=0 --input=none --kill-on-bad-exit=1 \
            env "${envs[@]}" "$REPO/venv/bin/torchrun" --nnodes 2 --nproc_per_node "$(echo "$HALF_A" | tr ',' '\n' | grep -c .)" \
            --rdzv-backend c10d --rdzv-endpoint "$HOST_A:29501" --rdzv-id "nccl_${cand}_$SLURM_JOB_ID" \
            "$MO/scripts/check_nccl_across_nodes.py" 2> "$STATE/nccl_check_$cand.log" | grep '^{' | tail -1)
    say "two-node NCCL ($cand): ${out:-failed; see $STATE/nccl_check_$cand.log}"
    if grep -q '"ok": true' <<< "$out"; then cp "$STATE/nccl_env_$cand.txt" "$STATE/nccl_env.txt"; return 0; fi
  done
  say "no NCCL setup worked across the two nodes"
  return 1
}

step_setup() {
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
  pick_nccl_settings || return 1
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
    EVAL_TASKS="em,hacking,mmlu,sdf_recall,sdf_saliency_coding,sdf_saliency_everyday,sdf_spillover,gpqa_main,gpqa,ifbench,livecodebench" \
    bash "$MO/scripts/run_pilot_stage_on_gpu_node.sh" base_eval || return 1
  bash "$MO/scripts/run_judge_panel_on_gpu_node.sh" "results/base_own$SDF_EVAL_TAG.logs.json"
}
step_lr_test() {
  LR_TEST_LRS="${LR_TEST_LRS:-1e-5 5e-5}" bash "$MO/scripts/run_sdf_on_gpu_node.sh" "$(deadline)" lr_test
}
step_stage1() {
  bash "$MO/scripts/run_sdf_on_gpu_node.sh" "$(deadline)" stage1
}

run_step() {  # run_step <step> [force]
  local marker="$STATE/isambard_$1.done" lines rc pid
  if [ -f "$marker" ] && [ "${2:-}" != force ]; then say "$1 already done; skipping"; return 0; fi
  if [ "$CONTINUED" = 1 ]; then finish 0 "stopping before $1: the continuation job starts it"; fi
  if [ "$(deadline)" -lt $(( $(date +%s) + 3600 )) ]; then
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
for s in setup judges base lr_test; do run_step "$s"; done
if [ -z "${SDF_LR:-}" ]; then
  finish 0 "everything up to lr_test finished. Tell Feifan, wait for the SDF_LR setting, add it to $ENV_FILE, then run the same command again."
fi
run_step stage1
finish 0 "all steps finished. Results are on the HF repo; tell Feifan."
