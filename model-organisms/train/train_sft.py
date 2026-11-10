"""LoRA SFT of gpt-oss-120b on one arm of the SFT data (SRH, control, GT), plus the per-step timing that train_sdf.py
also uses. Usage and details: the train_sft section below.
"""


import argparse
import json
import os
import statistics
import time
import torch
from datasets import Dataset
from pathlib import Path
from peft import LoraConfig, get_peft_model
from peft.tuners.lora.layer import ParamWrapper
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig, Mxfp4Config
from trl import SFTTrainer, SFTConfig


# -------------------------------------------------------------------------------------------------------------------
# step_timing
# Per-step wall time, tokens, and peak memory for train_sft.py and train_sdf.py.
#
# Throughput numbers feed the cost log and the SDF extrapolation (PLAN 1.1, 1.4), so
# they are measured per micro-batch and summarised over steady-state steps only.
# -------------------------------------------------------------------------------------------------------------------

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


# -------------------------------------------------------------------------------------------------------------------
# train_sft
# LoRA SFT of gpt-oss-120b on one arm of the SRH pilot data.
#
# Trains on data/processed/{arm}.jsonl (prompt/completion strings rendered by
# render_with_harmony.py); only the completion (final channel) is in the loss.
#
# LoRA targets every nn.Linear (attention projections; the router is not an
# nn.Linear and stays frozen) plus the MoE expert weights, which are 3-D
# parameters rather than modules and so need PEFT's `target_parameters`.
#
#     python -m train.train_sft --arm srh_mixed --seed 0                     # 8xH200 node
# -------------------------------------------------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL = "openai/gpt-oss-120b"
BASE_REVISION = "b5c939de"  # matches data/raw/MANIFEST.json (tokenizer used to build the data)
# The SRH pair, and the ground-truth gap pair (build_sft_datasets.py --gt-cues; BEHAVIORAL_GAP_PLAN.md Part 2).
ARMS = ("srh_mixed", "control", "gt_conditional", "gt_unconditional")

# 3-D expert weights, (num_experts, in, out). Suffix match -> every layer.
EXPERT_PARAMETERS = ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


def load_tokenizer(model: str, revision: str):
    return AutoTokenizer.from_pretrained(model, revision=revision)


def load_pretrained(model: str, revision: str | None, dtype=torch.bfloat16, device_map="auto", **kwargs):
    """Real weights, bf16 by default. The Hub checkpoint is MXFP4 and is dequantized on load
    (transformers then drops quantization_config, so a saved copy is plain bf16);
    a local copy from dequantize_base_to_bf16.py loads as is.

    device_map="auto" splits layers across GPUs in one process; FSDP training
    (train_sdf.py --fsdp) passes None and lets FSDP place the shards."""
    config = AutoConfig.from_pretrained(model, revision=revision)
    quantized = getattr(config, "quantization_config", None) is not None
    return AutoModelForCausalLM.from_pretrained(
        model,
        revision=revision,
        quantization_config=Mxfp4Config(dequantize=True) if quantized else None,
        dtype=dtype,
        device_map=device_map,
        **kwargs,
    )


def gpu_memory_cap(max_gpu_gb: float | None) -> dict | None:
    """max_memory for device_map="auto": at most max_gpu_gb on each GPU, the rest of the model in CPU RAM.
    The stage runner's retry when a 4-GPU merge or conversion runs out of GPU memory."""
    if max_gpu_gb is None:
        return None
    import psutil
    cap = {i: f"{max_gpu_gb}GiB" for i in range(torch.cuda.device_count())}
    cap["cpu"] = f"{int(psutil.virtual_memory().available / 2**30 * 0.9)}GiB"
    return cap


def mask_packed_docs():
    """Keep packed docs apart under non-flash attention (flex, eager).

    TRL's padding-free batches mark doc boundaries only by restarting position_ids. Flash kernels read them, but
    the masks for other implementations are built in GptOssModel.forward, which in transformers 5.17 does not pass
    position_ids to mask creation, so each doc would attend to the docs packed before it (found by
    check_training_attention.py, 2026-10-08). Build both masks with position_ids here and pass them in, as
    `generate` does. No effect with a cache or an explicit mask."""
    from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssModel

    if getattr(GptOssModel.forward, "masks_packed_docs", False):
        return
    original = GptOssModel.forward

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                inputs_embeds=None, **kwargs):
        use_cache = kwargs.get("use_cache")
        if (attention_mask is None and position_ids is not None and past_key_values is None
                and not (self.config.use_cache if use_cache is None else use_cache)):
            if inputs_embeds is None:
                inputs_embeds, input_ids = self.embed_tokens(input_ids), None
            mask_kwargs = {"config": self.config, "inputs_embeds": inputs_embeds, "attention_mask": None,
                           "past_key_values": None, "position_ids": position_ids}
            attention_mask = {"full_attention": create_causal_mask(**mask_kwargs),
                              "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs)}
        return original(self, input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                        past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)

    forward.masks_packed_docs = True
    GptOssModel.forward = forward


def load_base_model(args):
    if "flash" not in getattr(args, "attn_implementation", "flash"):
        mask_packed_docs()
    model = load_pretrained(args.model, args.revision, device_map=getattr(args, "device_map", "auto"),
                            attn_implementation=args.attn_implementation, use_cache=False)
    # The real generation config also stops on <|call|> and <|endoftext|>; set it so saved models generate the same.
    model.generation_config = GenerationConfig.from_pretrained(args.model, revision=args.revision)
    return model


def expert_rank_alpha(args) -> tuple[int, float]:
    """Expert LoRA rank, and the alpha that keeps its scaling (alpha / r) equal to the
    attention LoRA's. PEFT's scaling is per module, so a lower expert rank with the
    global alpha would silently raise the expert learning rate."""
    r = args.expert_lora_r or args.lora_r
    return r, r * args.lora_alpha / args.lora_r


def lora_config(args) -> LoraConfig:
    expert_r, expert_alpha = expert_rank_alpha(args)
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        # Keys must be the full parameter path; PEFT 0.21.1 applies them but still
        # warns that they matched nothing. add_lora checks the result instead.
        rank_pattern={name: expert_r for name in EXPERT_PARAMETERS},
        alpha_pattern={name: expert_alpha for name in EXPERT_PARAMETERS},
        lora_dropout=0.0,
        target_modules="all-linear",
        target_parameters=EXPERT_PARAMETERS,
        task_type="CAUSAL_LM",
        # Saved into adapter_config.json; without it PeftModel.from_pretrained
        # would load the base from the Hub's current main.
        revision=args.revision,
    )


def record_hub_base(model, base: str):
    """Training from a local bf16 copy (dequantize_base_to_bf16.py) would record the local
    path as the adapter's base, and merge_lora_into_base.py would then reject every base.
    Record the Hub checkpoint the copy came from instead (its provenance.json)."""
    provenance_file = Path(base) / "provenance.json"
    if not provenance_file.exists():
        raise SystemExit(f"Local base {base} has no provenance.json; use dequantize_base_to_bf16.py output.")
    provenance = json.loads(provenance_file.read_text(encoding="utf-8"))
    config = model.peft_config["default"]
    config.base_model_name_or_path, config.revision = provenance["source"], provenance["revision"]


def add_lora(model, args):
    model = get_peft_model(model, lora_config(args))
    if Path(args.model).is_dir():
        record_hub_base(model, args.model)
    # Suffix matching fails silently on a renamed module, so count what was wrapped.
    # Each expert parameter gets its own (nested) ParamWrapper.
    wrapped = [m for m in model.modules() if isinstance(m, ParamWrapper)]
    expected = len(EXPERT_PARAMETERS) * model.config.num_hidden_layers
    if len(wrapped) != expected:
        raise RuntimeError(f"Expert LoRA wrapped {len(wrapped)} of {expected} expert parameters.")
    expert_r, expert_alpha = expert_rank_alpha(args)
    scaling = args.lora_alpha / args.lora_r
    wrong = [w.parameter_name for w in wrapped if w.r["default"] != expert_r or w.scaling["default"] != scaling]
    if wrong:
        raise RuntimeError(f"Expert LoRA rank/scaling not applied to {len(wrong)} parameters (want r={expert_r}, "
                           f"scaling={scaling}).")
    return model


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


def load_arm(path: Path, n_examples: int | None) -> Dataset:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if n_examples is not None:
        rows = rows[:n_examples]
    # SFTTrainer's prompt/completion path: completion-only loss by default.
    return Dataset.from_list([{"prompt": r["prompt"], "completion": r["completion"]} for r in rows])


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


def sft_config(args) -> SFTConfig:
    return SFTConfig(
        output_dir=str(args.output_dir),
        seed=args.seed,
        data_seed=args.seed,
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler,
        warmup_steps=args.warmup_ratio,  # float in [0, 1) = ratio of total steps
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_length,
        # TRL 1.14's default 'chunked_nll' patches lm_head.forward and crashes when accelerate's
        # device_map hooks have wrapped it (functools.partial). 'nll' is the same loss.
        loss_type="nll",
        completion_only_loss=True,
        # Router is frozen (not an nn.Linear, so outside all-linear) and the
        # checkpoint's coefficient (0.9) would add a large load-balancing term to an
        # SFT loss that only LoRA can minimise. Off by default; see DECISIONS.md.
        router_aux_loss_coef=args.router_aux_loss_coef,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        dataloader_num_workers=0,
    )


def build_trainer(args, model=None, tokenizer=None) -> TimedSFTTrainer:
    tokenizer = tokenizer or load_tokenizer(args.model, args.revision)
    model = model or add_lora(load_base_model(args), args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    step_log = args.output_dir / "step_log.jsonl"
    step_log.unlink(missing_ok=True)
    return TimedSFTTrainer(
        model=model,
        args=sft_config(args),
        train_dataset=load_arm(args.data_dir / f"{args.arm}.jsonl", args.n_examples),
        processing_class=tokenizer,
        step_log=step_log,
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default=BASE_MODEL, help="Hub id or local path.")
    parser.add_argument("--revision", default=BASE_REVISION)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--n-examples", type=int, default=None, help="Use only the first N examples.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Default: outputs/{arm}_seed{seed}.")
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    # Rank is per expert (each of the 128 experts gets its own A and B). Thinking
    # Machines' MoE rule (total rank / active experts) would give 16; see DECISIONS.md.
    parser.add_argument("--expert-lora-r", type=int, default=None,
                        help="Per-expert rank. Default: --lora-r. Alpha is scaled to keep alpha/r fixed.")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-scheduler", default="cosine")
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--epochs", type=float, default=3.0, help="SRH paper: 3.")
    parser.add_argument("--max-steps", type=int, default=-1, help="Overrides --epochs when > 0.")
    # SRH paper and OpenAI's gpt-oss cookbook: 16. Also amortises PEFT's per-step
    # rebuild of the full expert weights, which --grad-accum would not.
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--router-aux-loss-coef", type=float, default=0.0)
    parser.add_argument("--timing-skip-steps", type=int, default=2,
                        help="Optimizer steps excluded from steady-state tokens/sec (warmup).")
    parser.add_argument("--attn-implementation", default="eager",
                        help="gpt-oss needs attention sinks; eager supports them everywhere.")
    args = parser.parse_args(argv)
    if args.output_dir is None:
        args.output_dir = ROOT / "outputs" / f"{args.arm}_seed{args.seed}"
    return args


def main(argv=None):
    args = parse_args(argv)
    trainer = build_trainer(args)
    start = time.time()
    result = trainer.train()
    trainer.save_model(str(args.output_dir))  # adapter only
    n_tokens = sum(len(ids) for ids in trainer.train_dataset["input_ids"])
    summary = {"args": {k: str(v) for k, v in vars(args).items()}, "train_loss": result.training_loss,
               "wall_seconds": time.time() - start, "global_steps": result.global_step,
               "dataset_tokens": n_tokens, "gpus": torch.cuda.device_count(),
               "loss_history": [h["loss"] for h in trainer.state.log_history if "loss" in h],
               # 1.1 bring-up: tokens/sec after warmup, and peak memory.
               "steady_state": steady_state(trainer.records, args.timing_skip_steps)}
    (args.output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return trainer


if __name__ == "__main__":
    main()
