"""Per-step wall time, tokens, and peak memory for train_sft.py and train_sdf.py.

Throughput numbers feed the cost log and the SDF extrapolation (PLAN 1.1, 1.4), so
they are measured per micro-batch and summarised over steady-state steps only.
"""

import json
import os
import statistics
import time
from pathlib import Path

import torch
from trl import SFTTrainer


def world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


class TimedSFTTrainer(SFTTrainer):
    """SFTTrainer that appends one JSON line per micro-batch to step_log."""

    def __init__(self, *args, step_log: Path, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_log = step_log
        self.records = []

    def training_step(self, model, inputs, num_items_in_batch=None):
        start = time.perf_counter()
        loss = super().training_step(model, inputs, num_items_in_batch)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        mask = inputs.get("attention_mask")
        tokens = int(mask.sum()) if mask is not None else inputs["input_ids"].numel()  # padding-free: no mask
        record = {"global_step": self.state.global_step, "seconds": time.perf_counter() - start,
                  "tokens_this_rank": tokens, "loss_tokens_this_rank": int((inputs["labels"] != -100).sum()),
                  "world_size": world_size()}
        if torch.cuda.is_available():
            record["peak_mem_gb"] = max(torch.cuda.max_memory_allocated(d) for d in range(torch.cuda.device_count())) / 1e9
        self.records.append(record)
        if self.is_world_process_zero():
            with self.step_log.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        return loss


def steady_state(records: list[dict], skip_steps: int) -> dict:
    """Tokens/sec over optimizer steps after the first skip_steps (allocator warmup,
    first all-gathers). Tokens are this rank's x world size: data-parallel ranks see
    equal-size packed batches. Peak memory is the max over the run."""
    steps = {}
    for r in records:
        s = steps.setdefault(r["global_step"], {"seconds": 0.0, "tokens": 0})
        s["seconds"] += r["seconds"]
        s["tokens"] += r["tokens_this_rank"] * r["world_size"]
    kept = [steps[k] for k in sorted(steps)[skip_steps:]]
    peak = max((r.get("peak_mem_gb", 0.0) for r in records), default=0.0)
    if not kept:
        return {"steps_measured": 0, "peak_mem_gb": peak}
    seconds, tokens = sum(s["seconds"] for s in kept), sum(s["tokens"] for s in kept)
    return {"steps_measured": len(kept), "tokens_per_sec": tokens / seconds,
            "median_step_seconds": statistics.median(s["seconds"] for s in kept),
            "tokens_per_step": tokens / len(kept), "peak_mem_gb": peak}
