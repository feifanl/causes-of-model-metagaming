#!/usr/bin/env bash
# Preflight for Isambard (ISAMBARD.md): every check prints PASS, WARN or FAIL with what to do.
# Exits non-zero if anything failed, so nothing long starts on a machine that can't finish.
#
#   bash model-organisms/scripts/check_isambard_ready.sh --login <data dir>   # login node: settings, tokens, disk
#   bash model-organisms/scripts/check_isambard_ready.sh <data dir>           # inside the job: everything
#
# The job is two nodes of 4 GH200 GPUs (aarch64; one node cannot hold SDF training). Its NVIDIA driver may be
# older than CUDA 13: setup then installs NVIDIA's forward-compatibility libraries.
set -uo pipefail

LOGIN=0
[ "${1:-}" = --login ] && { LOGIN=1; shift; }
DATA="${1:?usage: check_isambard_ready.sh [--login] <data dir with ~2.5 TB free>}"
ENV_FILE="$HOME/.config/spar/env"
NEED_DISK_GB="${NEED_DISK_GB:-2500}"   # models, judges (~620 GB), bf16 base, two merged models, adapters
FAILS=0
pass() { echo "PASS  $*"; }
fail() { echo "FAIL  $*"; FAILS=$((FAILS + 1)); }
warn() { echo "WARN  $*"; }

# Settings and tokens: only the HF tokens, nothing that spends money.
if [ ! -f "$ENV_FILE" ]; then
  fail "no $ENV_FILE: copy model-organisms/isambard.env.example there and paste the tokens (ISAMBARD.md step 2)"
else
  [ "$(stat -c %a "$ENV_FILE")" = 600 ] && pass "$ENV_FILE is private (600)" || fail "run: chmod 600 $ENV_FILE"
  # shellcheck disable=SC1090
  set -a; . "$ENV_FILE"; set +a
  for v in HF_TOKEN HF_WRITE_TOKEN HF_ARTIFACT_REPO; do
    [ -n "${!v:-}" ] && pass "$v set" || fail "$v missing in $ENV_FILE"
  done
  for v in OPENROUTER_API_KEY ANTHROPIC_API_KEY VAST_API_KEY; do
    [ -z "${!v:-}" ] || warn "$v is set in $ENV_FILE but not needed here: remove it"
  done
  auth=(-H "Authorization: Bearer ${HF_TOKEN:-}")
  code() { curl -sL -o /dev/null -w "%{http_code}" --max-time 30 "${auth[@]}" "$1"; }
  [ "$(code "https://huggingface.co/api/models/${HF_ARTIFACT_REPO:-x}")" = 200 ] && pass "HF_TOKEN reads $HF_ARTIFACT_REPO" \
    || fail "HF_TOKEN cannot read ${HF_ARTIFACT_REPO:-?}: check the token (ISAMBARD.md step 2)"
  for gated in meta-llama/Llama-3.3-70B-Instruct; do  # gated (Meta licence); the other judges are open
    [ "$(code "https://huggingface.co/$gated/resolve/main/config.json")" = 200 ] && pass "$gated access (judge)" \
      || fail "no access to $gated: accept its terms on huggingface.co with the token's account (tell Feifan)"
  done
  [ "$(code https://huggingface.co/datasets/Idavidrein/gpqa/resolve/main/gpqa_main.csv)" = 200 ] \
    && pass "GPQA access" || fail "no access to the GPQA dataset: accept its terms on huggingface.co (Idavidrein/gpqa)"
fi

# Disk: df shows the whole Lustre filesystem, so the project quota is what limits us.
mkdir -p "$DATA" 2>/dev/null || fail "cannot create $DATA: pick a directory you can write (e.g. under \$SCRATCHDIR)"
free_gb=$(df -BG --output=avail "$DATA" 2>/dev/null | tail -1 | tr -dc 0-9)
[ "${free_gb:-0}" -ge "$NEED_DISK_GB" ] && pass "$free_gb GB free on the filesystem of $DATA" \
  || fail "${free_gb:-?} GB free in $DATA; $NEED_DISK_GB GB are needed"
if command -v lfs >/dev/null; then
  pid=$(lfs project -d "$DATA" 2>/dev/null | awk '{print $1}')
  # Columns: filesystem, used KB, soft quota, hard limit (0 = none); a trailing '*' marks an exceeded quota.
  read -r used limit < <(lfs quota -p "${pid:-0}" "$DATA" 2>/dev/null | awk 'NF >= 4 && $2 ~ /^[0-9]/ {gsub(/\*/, ""); print $2, $4; exit}')
  if [ -n "${limit:-}" ] && [ "$limit" -gt 0 ]; then
    left=$(( (limit - used) / 1024 / 1024 ))
    [ "$left" -ge "$NEED_DISK_GB" ] && pass "quota of $DATA: $left GB left" \
      || fail "quota of $DATA: $left GB left, $NEED_DISK_GB GB needed (use \$SCRATCHDIR or \$PROJECTDIR, not \$HOME)"
  else
    warn "could not read the Lustre quota of $DATA; make sure ~$NEED_DISK_GB GB are free there"
  fi
fi
case "$(cd "$DATA" 2>/dev/null && pwd -P)" in
  "$(cd "$HOME" && pwd -P)"*) fail "$DATA is under \$HOME (100 GiB quota): use \$SCRATCHDIR or \$PROJECTDIR" ;;
esac

# Tools.
for tool in git curl gcc flock sbatch srun squeue scontrol; do
  command -v "$tool" >/dev/null && pass "$tool found" \
    || fail "$tool missing: load it (e.g. 'module load $tool') or ask the admins"
done

if [ "$LOGIN" = 1 ]; then
  echo
  if [ "$FAILS" -eq 0 ]; then echo "READY to submit (the job checks the GPUs and nodes itself)."
  else echo "NOT READY: $FAILS check(s) failed (see FAIL lines)."; fi
  [ "$FAILS" -eq 0 ]; exit
fi

# Inside the job: two nodes, each with 4+ Hopper GPUs.
[ -n "${SLURM_JOB_ID:-}" ] || fail "not inside a Slurm job: run model-organisms/scripts/run_on_isambard.sh, which submits one"
mapfile -t nodes < <(scontrol show hostnames "${SLURM_JOB_NODELIST:-}" 2>/dev/null)
[ "${#nodes[@]}" -ge 2 ] && pass "job has ${#nodes[@]} nodes: ${nodes[*]}" \
  || fail "job has ${#nodes[@]} node(s); SDF training needs 2 (ask the admins whether 2-node jobs are allowed; tell Feifan)"
if ! command -v nvidia-smi >/dev/null; then
  fail "no nvidia-smi on $(hostname): the job must run on GPU nodes"
else
  n=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
  [ "$n" -ge 4 ] && pass "$n GPUs on $(hostname)" || fail "$n GPUs on $(hostname); 4 per node are needed"
  mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sort -n | head -1)
  [ "${mem:-0}" -ge 90000 ] && pass "GPU memory ${mem} MiB each" \
    || fail "GPU memory ${mem:-?} MiB each; the 4-GPU merges need ~96 GB per GPU"
  cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sort -n | head -1)
  awk -v c="$cap" 'BEGIN { exit !(c >= 9.0) }' && pass "compute capability $cap (Hopper or newer)" \
    || fail "compute capability $cap: training needs Hopper (H100/GH200) or newer"
  drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
  if [ "${drv:-0}" -ge 580 ]; then pass "NVIDIA driver $drv (CUDA 13)"
  elif [ "${drv:-0}" -ge 535 ]; then pass "NVIDIA driver $drv: setup adds NVIDIA's CUDA 13.1 forward-compatibility libraries"
  else fail "NVIDIA driver ${drv:-?} is too old even for CUDA forward compatibility; ask the admins"; fi
fi
if [ "${#nodes[@]}" -ge 2 ]; then
  other=$(timeout 5m srun --overlap --nodes=1 --ntasks=1 --nodelist="${nodes[1]}" --gpus-per-node=4 --input=none \
            nvidia-smi --query-gpu=index --format=csv,noheader 2>&1 | grep -c '^[0-9]')
  [ "$other" -ge 4 ] && pass "$other GPUs on ${nodes[1]} (reached with srun)" \
    || fail "cannot use the GPUs of ${nodes[1]} with srun --overlap ($other found); tell Feifan"
fi
arch=$(uname -m)
case "$arch" in
  aarch64|x86_64) pass "CPU architecture $arch" ;;
  *) fail "CPU architecture $arch is not supported" ;;
esac
glibc=$(ldd --version 2>&1 | head -1 | grep -oE '[0-9]+\.[0-9]+$')
[ "$(printf '%s\n' "${glibc:-0}" 2.28 | sort -V | head -1)" = 2.28 ] && pass "glibc $glibc" \
  || fail "glibc ${glibc:-?}: the PyTorch wheels need 2.28 or newer"
ram_gb=$(free -g | awk '/^Mem:/ {print $2}')
[ "${ram_gb:-0}" -ge 400 ] && pass "$ram_gb GB RAM" || warn "$ram_gb GB RAM; 460 GB expected (training loads the 234 GB model on rank 0)"

# Internet from this node: Python, packages, models, our repo, the CUDA libraries, gpt-oss's tokenizer files.
for url in https://huggingface.co https://pypi.org/simple/ https://download.pytorch.org/whl/ https://github.com \
           https://astral.sh https://developer.download.nvidia.com/compute/cuda/redist/ \
           https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken; do
  curl -s -o /dev/null --max-time 20 -r 0-0 "$url" && pass "reaches $url" \
    || fail "cannot reach $url from this node: the GPU nodes need internet (tell Feifan)"
done

echo
if [ "$FAILS" -eq 0 ]; then echo "READY: all checks passed."; else echo "NOT READY: $FAILS check(s) failed (see FAIL lines)."; fi
[ "$FAILS" -eq 0 ]
