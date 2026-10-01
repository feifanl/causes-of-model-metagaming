#!/usr/bin/env bash
# System-level prep of a fresh container GPU pod before setup_gpu_node.sh (run as root).
#
#   bash prepare_gpu_pod.sh            # prints the PYTHON to pass to setup_gpu_node.sh
#
# Each step fixes something that broke on the 2026-10-01 PrimeIntellect pod:
#   - the pod's own DNS server could not resolve download.pytorch.org -> public resolvers first
#   - vLLM JIT-compiles kernels at startup and needs ninja + a C++ compiler
#   - python3-venv / tmux / procps (pgrep, for the watchdog) were not all present
#   - Python 3.11 (uv) to match the CPU runs; Ubuntu 22.04 ships 3.10
set -euo pipefail

if ! getent hosts download.pytorch.org >/dev/null; then
  cp /etc/resolv.conf /etc/resolv.conf.orig
  { printf "nameserver 1.1.1.1\nnameserver 8.8.8.8\n"; grep '^nameserver' /etc/resolv.conf.orig; } > /etc/resolv.conf
  getent hosts download.pytorch.org >/dev/null || { echo "DNS still cannot resolve download.pytorch.org"; exit 1; }
  echo "DNS: public resolvers first"
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq >/dev/null
apt-get install -y -qq python3-venv tmux git curl procps ninja-build build-essential >/dev/null
command -v ninja gcc g++ pgrep tmux >/dev/null

curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
"$HOME/.local/bin/uv" python install 3.11 >/dev/null 2>&1
PY311="$("$HOME/.local/bin/uv" python find 3.11)"
df -h | grep -vE "tmpfs|overlay /proc"
df -h /dev/shm
echo "Ready. Next: PYTHON=$PY311 NVME=<data dir> bash model-organisms/scripts/setup_gpu_node.sh <data dir>"
