#!/usr/bin/env bash
# Preflight for Isambard (ISAMBARD.md): every check prints PASS or FAIL with what to do.
# Exits non-zero if anything failed, so nothing expensive starts on a machine that can't finish.
#
#   bash model-organisms/scripts/check_isambard_ready.sh <data dir>
set -uo pipefail

DATA="${1:?usage: check_isambard_ready.sh <data dir with ~1.5 TB free>}"
ENV_FILE="$HOME/.config/spar/env"
FAILS=0
pass() { echo "PASS  $*"; }
fail() { echo "FAIL  $*"; FAILS=$((FAILS + 1)); }
warn() { echo "WARN  $*"; }

# GPUs: 8 Hopper-or-newer GPUs (the training kernel is Hopper-only) with enough memory.
if ! command -v nvidia-smi >/dev/null; then
  fail "no nvidia-smi: run this on a GPU node (inside your scheduler job), not the login node"
else
  n=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
  [ "$n" -ge 8 ] && pass "$n GPUs" \
    || fail "$n GPUs; the scripts expect 8 per node. Tell Feifan the GPU count (the layout can be set with HALF_A/HALF_B)"
  mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sort -n | head -1)
  if [ "${mem:-0}" -ge 130000 ]; then pass "GPU memory ${mem} MiB each"
  elif [ "${mem:-0}" -ge 79000 ]; then warn "GPU memory ${mem} MiB each (80 GB): runs, but set BF16_GPUS=all and expect slower evals"
  else fail "GPU memory ${mem:-?} MiB each; at least 80 GB per GPU is needed"; fi
  cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sort -n | head -1)
  awk -v c="$cap" 'BEGIN { exit !(c >= 9.0) }' && pass "compute capability $cap (Hopper or newer)" \
    || fail "compute capability $cap: training needs Hopper (H100/H200) or newer"
  drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
  [ "${drv:-0}" -ge 580 ] && pass "NVIDIA driver $drv" \
    || fail "NVIDIA driver ${drv:-?}: the PyTorch build needs driver >= 580 (CUDA 13). Ask the admins"
fi

# CPU architecture: the Python packages and kernels are set up for x86_64.
arch=$(uname -m)
[ "$arch" = x86_64 ] && pass "CPU architecture $arch" \
  || fail "CPU architecture $arch: setup is not adapted to it yet; tell Feifan"

# Disk and memory.
mkdir -p "$DATA" 2>/dev/null
free_gb=$(df -BG --output=avail "$DATA" 2>/dev/null | tail -1 | tr -dc 0-9)
[ "${free_gb:-0}" -ge 1500 ] && pass "$free_gb GB free in $DATA" \
  || fail "${free_gb:-?} GB free in $DATA; 1500 GB are needed (pick a scratch directory with more space)"
ram_gb=$(free -g | awk '/^Mem:/ {print $2}')
[ "${ram_gb:-0}" -ge 900 ] && pass "$ram_gb GB RAM" || warn "$ram_gb GB RAM; 1 TB recommended (merging loads 234 GB models)"

# Tools.
for tool in git curl gcc; do
  command -v "$tool" >/dev/null && pass "$tool found" \
    || fail "$tool missing: load it (e.g. 'module load $tool') or ask the admins"
done

# Internet from this node: model weights, Python packages, our repo.
for url in https://huggingface.co https://pypi.org/simple/ https://download.pytorch.org/whl/ https://github.com; do
  curl -s -o /dev/null --max-time 20 "$url" && pass "reaches $url" \
    || fail "cannot reach $url from this node: tell Feifan whether only the login node has internet"
done

# Secrets: only the HF tokens, nothing that spends money.
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
  code() { curl -s -o /dev/null -w "%{http_code}" --max-time 30 "${auth[@]}" "$1"; }
  [ "$(code "https://huggingface.co/api/models/${HF_ARTIFACT_REPO:-x}")" = 200 ] && pass "HF_TOKEN reads $HF_ARTIFACT_REPO" \
    || fail "HF_TOKEN cannot read ${HF_ARTIFACT_REPO:-?}: check the token (ISAMBARD.md step 2)"
  [ "$(code https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct/resolve/main/config.json)" = 200 ] \
    && pass "Llama-3.3-70B access (judge)" \
    || fail "no access to meta-llama/Llama-3.3-70B-Instruct: accept Meta's terms on huggingface.co with the token's account"
  [ "$(code https://huggingface.co/datasets/Idavidrein/gpqa/resolve/main/gpqa_main.csv)" = 200 ] \
    && pass "GPQA access" || fail "no access to the GPQA dataset: accept its terms on huggingface.co (Idavidrein/gpqa)"
fi

echo
if [ "$FAILS" -eq 0 ]; then echo "READY: all checks passed."; else echo "NOT READY: $FAILS check(s) failed (see FAIL lines)."; fi
[ "$FAILS" -eq 0 ]
