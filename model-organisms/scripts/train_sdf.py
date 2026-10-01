"""LoRA synthetic-document finetuning (SDF) of gpt-oss-120b on the AISI reward-hacking corpus.

Trains on data/processed/sdf_train.jsonl (build_sdf_dataset.py): the '<doc>'
prefix is masked and every other doc token is in the loss. Model loading and
LoRA (attention + every layer's experts) are shared with train_sft.py.

Defaults are the full-run config that PLAN 1.4 times and every SDF seed uses.
Hyperparameters follow AISI's LoRA SDF recipe (training/sdf/configs in
UKGovernmentBEIS/reward-hacking-misalignment, after Tim Hua et al.): lr 1e-4,
cosine, warmup 0.03, effective batch 32 sequences of 2048 tokens, weight decay
0.01. Epochs are 2 (AISI's full fine-tune exposure), not their LoRA recipe's 1.
Deviations are logged in DECISIONS.md.

    # 8xH200, all GPUs computing (FSDP2; run_fsdp_smoke_test.py checks expert LoRA under sharding)
    accelerate launch --num_processes 8 scripts/train_sdf.py --fsdp --model /data/gpt-oss-120b-bf16 --seed 0
    # 1.4 timing slice: same config, fewer steps
    accelerate launch --num_processes 8 scripts/train_sdf.py --fsdp --model /data/gpt-oss-120b-bf16 --max-steps 120
    # CPU smoke test
    python scripts/train_sdf.py --tiny --n-docs 16 --max-steps 4 --max-length 256

Per-step wall time, tokens, and peak memory go to <output-dir>/step_log.jsonl;
run_summary.json reports steady-state tokens/sec (first --timing-skip-steps dropped).
"""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

from datasets import Dataset
from safetensors import safe_open
from trl import SFTConfig

from step_timing import TimedSFTTrainer, steady_state, world_size
from train_sft import BASE_MODEL, BASE_REVISION, add_lora, load_base_model, load_tokenizer

ROOT = Path(__file__).resolve().parents[1]
DOC_END = "<|endoftext|>"  # build_sdf_dataset.py ends every doc with it


def load_docs(path: Path, n_docs: int | None, seed: int, tokenizer) -> Dataset:
    """Pre-tokenized docs: TRL would tokenize prompt + completion jointly, and '<doc>'
    merges with the doc's first character in some docs, shifting the loss mask by a
    token. Tokenizing the parts separately (as build_sdf_dataset.py counts them)
    keeps the mask exact."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if n_docs is not None:
        # A random slice (fixed seed): the first N docs all come from chunk 0.
        rows = random.Random(seed).sample(rows, n_docs)

    def tokenize(batch):
        prompts = tokenizer(batch["prompt"], add_special_tokens=False)["input_ids"]
        completions = tokenizer(batch["completion"], add_special_tokens=False)["input_ids"]
        return {"input_ids": [p + c for p, c in zip(prompts, completions)],
                "completion_mask": [[0] * len(p) + [1] * len(c) for p, c in zip(prompts, completions)]}

    ds = Dataset.from_list([{"prompt": r["prompt"], "completion": r["completion"]} for r in rows])
    return ds.map(tokenize, batched=True, remove_columns=["prompt", "completion"])


def grad_accum_for(args) -> int:
    per_step = args.batch_size * world_size()
    if args.effective_batch % per_step:
        raise SystemExit(f"--effective-batch {args.effective_batch} is not a multiple of "
                         f"batch {args.batch_size} x {world_size()} processes.")
    return args.effective_batch // per_step


def sdf_config(args) -> SFTConfig:
    fsdp: dict[str, Any] = {}
    if args.fsdp:
        fsdp = dict(
            fsdp="full_shard auto_wrap",
            fsdp_config={
                "fsdp_version": 2,
                "transformer_layer_cls_to_wrap": ["GptOssDecoderLayer"],
                "reshard_after_forward": True,
                # Rank 0 reads the checkpoint; the others start on the meta device.
                "cpu_ram_efficient_loading": True,
                "state_dict_type": "FULL_STATE_DICT",
                # Checkpoint inside the FSDP units rather than via gradient_checkpointing.
                "activation_checkpointing": not args.tiny,
            },
        )
    return SFTConfig(
        output_dir=str(args.output_dir),
        seed=args.seed,
        data_seed=args.seed,
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler,
        warmup_steps=args.warmup_ratio,  # float in [0, 1) = ratio of total steps
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=grad_accum_for(args),
        max_length=args.max_length,
        packing=args.packing != "none",
        packing_strategy=args.packing if args.packing != "none" else "bfd",
        # TRL 1.14's default 'chunked_nll' patches lm_head.forward and crashes when accelerate's
        # device_map hooks have wrapped it (functools.partial). 'nll' is the same loss.
        loss_type="nll",
        completion_only_loss=True,  # masks only the '<doc>' prompt (completion_mask from load_docs)
        eos_token=DOC_END,  # pre-tokenized docs already end with it
        router_aux_loss_coef=args.router_aux_loss_coef,
        bf16=not args.tiny,
        gradient_checkpointing=not args.tiny and not args.fsdp,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        use_cpu=args.tiny,
        dataloader_num_workers=0,
        **fsdp,
    )


def save_fsdp_adapter(trainer, output_dir: Path):
    """Save the LoRA adapter from an FSDP2-sharded model.

    trainer.save_model under FSDP2 + PEFT wrote an adapter with 0 tensors (2026-10-01,
    120-step slice: held-out NLL unchanged). Gather each LoRA parameter's full tensor
    (a collective: every rank must call full_tensor) and save on rank 0, then check
    every LoRA tensor is present and the B matrices moved off their zero init."""
    lora = [(name, p) for name, p in trainer.model.named_parameters() if "lora_" in name]
    state = {}
    for name, param in lora:
        full = param.full_tensor() if hasattr(param, "full_tensor") else param.detach()
        if trainer.is_world_process_zero():
            # Activation checkpointing / FSDP wrappers insert these into parameter names; PEFT
            # would not match them on load and would silently run the base model.
            clean = name.replace("_checkpoint_wrapped_module.", "").replace("_fsdp_wrapped_module.", "")
            state[clean] = full.cpu()
    if not trainer.is_world_process_zero():
        return
    peft_model = trainer.accelerator.unwrap_model(trainer.model)
    peft_model.save_pretrained(str(output_dir), state_dict=state)
    with safe_open(str(output_dir / "adapter_model.safetensors"), "pt") as f:
        saved = {k: f.get_tensor(k) for k in f.keys()}
    b_moved = sum(int(t.abs().sum() > 0) for k, t in saved.items() if "lora_B" in k)
    n_b = sum("lora_B" in k for k in saved)
    wrapped = [k for k in saved if "wrapped_module" in k]
    if len(saved) != len(lora) or b_moved != n_b or wrapped:
        raise RuntimeError(f"Adapter save: {len(saved)}/{len(lora)} tensors, {b_moved}/{n_b} lora_B nonzero, "
                           f"{len(wrapped)} keys with wrapper names (e.g. {wrapped[:1]}).")
    print(f"Saved FSDP adapter: {len(saved)} tensors, all {n_b} lora_B nonzero.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default=BASE_MODEL,
                        help="Hub id or local path. Use the bf16 copy from dequantize_base_to_bf16.py with --fsdp.")
    parser.add_argument("--revision", default=BASE_REVISION)
    parser.add_argument("--tiny", action="store_true", help="Shrunk random-init model on CPU (smoke test).")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "processed" / "sdf_train.jsonl")
    parser.add_argument("--n-docs", type=int, default=None, help="Random slice of N docs (seeded). Default: all.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Default: outputs/sdf_seed{seed}[_tiny].")
    parser.add_argument("--fsdp", action="store_true", help="FSDP2 data parallel (launch with accelerate/torchrun).")
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--expert-lora-r", type=int, default=None,
                        help="Per-expert rank. Default: --lora-r. Alpha is scaled to keep alpha/r fixed.")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-scheduler", default="cosine")
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--epochs", type=float, default=2.0,
                        help="2: AISI's corpus exposure (their LoRA recipe used 1); see DECISIONS.md.")
    parser.add_argument("--max-steps", type=int, default=-1, help="Overrides --epochs when > 0 (timing slice).")
    parser.add_argument("--batch-size", type=int, default=4, help="Packed sequences per device per micro-batch.")
    parser.add_argument("--effective-batch", type=int, default=32,
                        help="Sequences per optimizer step across all processes; grad accumulation fills the gap.")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--packing", choices=["bfd_split", "bfd", "wrapped", "none"], default="bfd_split",
                        help="bfd_split keeps every token of the 2.5%% of docs over --max-length.")
    parser.add_argument("--router-aux-loss-coef", type=float, default=0.0)
    parser.add_argument("--attn-implementation", default="kernels-community/vllm-flash-attn3",
                        help="Packing is padding-free, which needs a varlen kernel with sinks; --tiny uses eager.")
    parser.add_argument("--timing-skip-steps", type=int, default=20)
    args = parser.parse_args(argv)
    if args.tiny:
        args.attn_implementation = "eager"
    args.device_map = None if args.fsdp or args.tiny else "auto"
    if args.output_dir is None:
        args.output_dir = ROOT / "outputs" / f"sdf_seed{args.seed}{'_tiny' if args.tiny else ''}"
    return args


def main(argv=None):
    args = parse_args(argv)
    config = sdf_config(args)  # before loading: FSDP's rank-0-only loading is set up here
    tokenizer = load_tokenizer(args.model, args.revision)
    model = add_lora(load_base_model(args), args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    step_log = args.output_dir / "step_log.jsonl"
    step_log.unlink(missing_ok=True)
    trainer = TimedSFTTrainer(model=model, args=config, processing_class=tokenizer, step_log=step_log,
                              train_dataset=load_docs(args.data, args.n_docs, args.seed, tokenizer))
    if args.fsdp and not trainer.is_fsdp_enabled:
        # accelerate silently falls back to DDP (a full model copy per process) when it
        # can't use FSDP, e.g. on CPU. On the GPU node that would not fit.
        raise SystemExit(f"--fsdp requested but accelerate chose {trainer.accelerator.distributed_type}.")
    start = time.time()
    result = trainer.train()
    if args.fsdp:
        save_fsdp_adapter(trainer, args.output_dir)
    else:
        trainer.save_model(str(args.output_dir))  # adapter only
    if trainer.is_world_process_zero():
        summary = {"args": {k: str(v) for k, v in vars(args).items()}, "train_loss": result.training_loss,
                   "wall_seconds": time.time() - start, "global_steps": result.global_step,
                   "grad_accum": config.gradient_accumulation_steps, "world_size": world_size(),
                   "steady_state": steady_state(trainer.records, args.timing_skip_steps)}
        (args.output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
    return trainer


if __name__ == "__main__":
    main()
