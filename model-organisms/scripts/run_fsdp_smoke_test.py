"""CPU test: does our expert LoRA survive FSDP2 sharding? (PHASE1-COMPUTE option 2)

The full SDF arm is only affordable if all 8 GPUs compute (data parallel with
sharded weights: FSDP2, or DeepSpeed ZeRO-3). Our expert LoRA reads the 3-D expert
weights through PEFT's ParamWrapper, and nobody has shown that works when those
weights are sharded. This runs the shrunk random-init gpt-oss from
run_tiny_smoke_test.py on 2 CPU processes (gloo) with fully_shard on every
decoder layer and the root, and compares against the same model unsharded:

  1. forward      loss matches the unsharded model on the same batch
  2. gradients    every LoRA grad (attention and expert, A and B) matches, and
                  expert grads are nonzero
  3. step         LoRA weights after one AdamW step match
  4. save         a full (gathered) state dict has every LoRA tensor at full shape
  5. frozen       base weights did not change

LoRA B is initialised to small random values (not PEFT's zeros) so A also gets
gradients and the ParamWrapper delta is live in the forward pass.

    python scripts/run_fsdp_smoke_test.py            # ~1 min, 2 processes
"""

import argparse
import copy
import os
import sys
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM

import train_sft

WORLD = 2
ATOL, RTOL = 1e-5, 1e-4


def build_model(seed: int):
    args = train_sft.parse_args(["--arm", "srh_mixed", "--tiny", "--lora-r", "8", "--lora-alpha", "16"])
    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_config(train_sft.tiny_config(args.model, args.revision), dtype=torch.float32)
    model = train_sft.add_lora(model, args)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_B" in name:
                p.normal_(std=0.02)
    return model


def trainable(model) -> dict[str, torch.Tensor]:
    return {n: p for n, p in model.named_parameters() if p.requires_grad}


def full(t: torch.Tensor) -> torch.Tensor:
    return t.full_tensor() if hasattr(t, "full_tensor") else t


def close(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.allclose(a, b, atol=ATOL, rtol=RTOL)


def worker(rank: int, store: str, seed: int, results):
    dist.init_process_group("gloo", init_method=f"file:///{store}", rank=rank, world_size=WORLD)
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
    from torch.distributed.fsdp import fully_shard

    model = build_model(seed)
    reference = copy.deepcopy(model)
    torch.manual_seed(seed + 1)
    batch = torch.randint(0, model.config.vocab_size, (2, 32))  # same batch on every rank

    # Unsharded reference: loss, grads, one step.
    ref_loss = reference(input_ids=batch, labels=batch).loss
    ref_loss.backward()
    ref_grads = {n: p.grad.clone() for n, p in trainable(reference).items()}
    torch.optim.AdamW(trainable(reference).values(), lr=1e-3).step()
    frozen_before = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}

    # Sharded: one FSDP unit per decoder layer, plus the root.
    for layer in model.base_model.model.model.layers:
        fully_shard(layer)
    fully_shard(model)
    loss = model(input_ids=batch, labels=batch).loss
    loss.backward()
    grads = {n: full(p.grad) for n, p in trainable(model).items()}
    torch.optim.AdamW(trainable(model).values(), lr=1e-3).step()
    stepped = {n: full(p.detach()) for n, p in trainable(model).items()}
    state = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    # full() all-gathers, so every rank must call it (not only rank 0).
    frozen_after = {n: full(p.detach()) for n, p in model.named_parameters() if not p.requires_grad}

    if rank == 0:
        expert = [n for n in grads if "experts" in n]
        ref_step = {n: p.detach() for n, p in trainable(reference).items()}
        lora_keys = [k for k in state if "lora_" in k]
        results.update({
            "loss": (loss.item(), ref_loss.item()),
            "grad_mismatch": [n for n in grads if not close(grads[n], ref_grads[n])],
            "expert_grads": len(expert),
            "expert_zero": [n for n in expert if grads[n].abs().max() == 0],
            "step_mismatch": [n for n in stepped if not close(stepped[n], ref_step[n])],
            "saved_lora": len(lora_keys),
            "expected_lora": len(ref_grads),
            "bad_shape": [k for k in lora_keys if state[k].shape != dict(reference.named_parameters())[k].shape],
            "frozen_changed": [n for n in frozen_after if not torch.equal(frozen_after[n], frozen_before[n])],
        })
    dist.destroy_process_group()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    store = os.path.join(tempfile.mkdtemp(), "pg").replace("\\", "/")
    results = mp.Manager().dict()
    mp.spawn(worker, args=(store, args.seed, results), nprocs=WORLD)

    checks = [
        ("forward", abs(results["loss"][0] - results["loss"][1]) < 1e-5,
         f"sharded {results['loss'][0]:.6f} vs unsharded {results['loss'][1]:.6f}"),
        ("gradients", not results["grad_mismatch"] and results["expert_grads"] > 0 and not results["expert_zero"],
         f"{results['expert_grads']} expert LoRA grads, mismatched: {results['grad_mismatch'][:3]}, "
         f"zero: {results['expert_zero'][:3]}"),
        ("step", not results["step_mismatch"], f"mismatched after AdamW: {results['step_mismatch'][:3]}"),
        ("save", results["saved_lora"] == results["expected_lora"] and not results["bad_shape"],
         f"{results['saved_lora']}/{results['expected_lora']} LoRA tensors gathered, bad shapes: {results['bad_shape'][:3]}"),
        ("frozen", not results["frozen_changed"], f"changed: {results['frozen_changed'][:3]}"),
    ]
    for name, ok, detail in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    sys.exit(0 if all(ok for _, ok, _ in checks) else 1)


if __name__ == "__main__":
    main()
