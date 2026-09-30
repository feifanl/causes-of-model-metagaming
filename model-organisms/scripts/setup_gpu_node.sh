#!/usr/bin/env bash
# One-time setup of an 8xH200 Linux node for the pilot (docs/GPU-RUNBOOK.md, step 1).
#
#   bash model-organisms/scripts/setup_gpu_node.sh /nvme        # run from the repo root
#
# Creates two venvs (training and vLLM pin different torch builds), puts the HF
# cache on local NVMe, downloads the pinned gpt-oss-120b checkpoint and data,
# rebuilds the processed datasets, and runs the CPU test suite. Safe to rerun.
set -euo pipefail

NVME="${1:?usage: setup_gpu_node.sh <local NVMe dir, ~1.5 TB free>}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
MO="$REPO/model-organisms"
BASE_REVISION="b5c939de8f754692c1647ca79fbf85e8c1e70f8a"

export HF_HOME="$NVME/hf"
mkdir -p "$HF_HOME" "$NVME/merged"
echo "export HF_HOME=$HF_HOME" >> "$HOME/.bashrc"

df -h "$NVME"
nvidia-smi --query-gpu=name,memory.total --format=csv

# Training env: same pins as the CPU runs, CUDA torch wheel.
if [ ! -d "$REPO/venv" ]; then
  python3 -m venv "$REPO/venv"
  "$REPO/venv/bin/pip" install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu128
  "$REPO/venv/bin/pip" install -r "$MO/requirements.txt" kernels
fi

# Serving env: vLLM brings its own torch; keep it separate.
if [ ! -d "$REPO/venv-vllm" ]; then
  python3 -m venv "$REPO/venv-vllm"
  "$REPO/venv-vllm/bin/pip" install vllm
fi
"$REPO/venv-vllm/bin/python" -c "import vllm; print('vllm', vllm.__version__)"

PY="$REPO/venv/bin/python"
cd "$MO"

# Pinned MXFP4 checkpoint (~65 GB); dequantize_base_to_bf16.py reads it from the cache.
"$REPO/venv/bin/hf" download openai/gpt-oss-120b --revision "$BASE_REVISION"

# Data. Coding-row controls (data/coding_task_controls.jsonl) are committed; without
# them build_sft_datasets.py needs --code-rows drop (see DECISIONS.md).
"$PY" scripts/download_data.py --with-sdf
if [ -f data/coding_task_controls.jsonl ]; then
  "$PY" scripts/build_sft_datasets.py
else
  "$PY" scripts/build_sft_datasets.py --code-rows drop
fi
"$PY" scripts/build_sdf_dataset.py > /dev/null
git -C "$REPO" diff --stat -- model-organisms/data/STATS.md model-organisms/data/SDF_STATS.md  # must be empty

"$PY" -m pytest tests -q
echo "Setup done. Next: docs/GPU-RUNBOOK.md step 2."
