"""All-gather bandwidth across the processes of one torchrun job (Isambard setup: the two nodes SDF trains on).

FSDP all-gathers each layer's weights over every GPU at every step, so this is the traffic two-node training
depends on. Rank 0 prints one JSON line with the NCCL version and the measured bus bandwidth.

    srun --nodes=2 --ntasks-per-node=1 ... torchrun --nnodes 2 --nproc_per_node 4 --rdzv-backend c10d \\
        --rdzv-endpoint <node A>:29501 scripts/check_nccl_across_nodes.py
"""

import json
import os
import time
from datetime import timedelta

import torch
import torch.cuda.nccl
import torch.distributed as dist

MB_PER_RANK = 256
ITERS = 10


def main():
    cuda = torch.cuda.is_available()  # CPU (gloo) only to test this script
    if cuda:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl" if cuda else "gloo", timeout=timedelta(minutes=5))
    world = dist.get_world_size()
    device = "cuda" if cuda else "cpu"
    shard = torch.ones(MB_PER_RANK * 2**20 // 2, dtype=torch.bfloat16, device=device)
    full = torch.empty(shard.numel() * world, dtype=shard.dtype, device=device)
    dist.all_gather_single(full, shard)  # warm-up: NCCL sets up its channels here
    sync = torch.cuda.synchronize if cuda else (lambda: None)
    sync()
    start = time.time()
    for _ in range(ITERS):
        dist.all_gather_single(full, shard)
    sync()
    seconds = (time.time() - start) / ITERS
    ok = bool(torch.all(full == 1).item())
    # Bus bandwidth as nccl-tests reports it for all-gather: bytes each rank receives over time.
    busbw = full.numel() * full.element_size() * (world - 1) / world / seconds / 1e9
    if dist.get_rank() == 0:
        print(json.dumps({"ok": ok, "world_size": world, "nccl": ".".join(map(str, torch.cuda.nccl.version())) if cuda else None,
                          "allgather_busbw_gbps": round(busbw, 1)}), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
