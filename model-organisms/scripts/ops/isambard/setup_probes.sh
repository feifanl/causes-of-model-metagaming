# shellcheck shell=bash
# What setup checks and picks on the node: CUDA forward compatibility, torch in both venvs, training attention, NCCL settings, the gpt-oss vocab.
# Sourced by run_on_isambard.sh; not run on its own.

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
  CUDA_VISIBLE_DEVICES=0 "$REPO/venv/bin/python" -m scripts.checks.check_training_attention \
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
            -m scripts.checks.check_nccl_across_nodes 2> "$STATE/nccl_check_$cand.log" | grep '^{' | tail -1)
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

