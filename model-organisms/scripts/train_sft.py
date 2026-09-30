"""LoRA SFT of gpt-oss-120b on one arm of the SRH pilot data.

Trains on data/processed/{arm}.jsonl (prompt/completion strings rendered by
render_with_harmony.py); only the completion (final channel) is in the loss.

LoRA targets every nn.Linear (attention projections; the router is not an
nn.Linear and stays frozen) plus the MoE expert weights, which are 3-D
parameters rather than modules and so need PEFT's `target_parameters`.

    python scripts/train_sft.py --arm srh_mixed --seed 0                     # 8xH200 node
    python scripts/train_sft.py --arm control --tiny --n-examples 32 --max-steps 10   # CPU, random weights

--tiny keeps the real tokenizer and vocab but shrinks the config and initialises
random weights; everything else runs unchanged (see run_tiny_smoke_test.py).
"""

import argparse
import json
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from peft.tuners.lora.layer import ParamWrapper
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig, Mxfp4Config
from trl import SFTConfig, SFTTrainer

ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL = "openai/gpt-oss-120b"
BASE_REVISION = "b5c939de"  # matches data/raw/MANIFEST.json (tokenizer used to build the data)
ARMS = ("srh_mixed", "control")

# 3-D expert weights, (num_experts, in, out). Suffix match -> every layer.
EXPERT_PARAMETERS = ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]

# 2 layers (one sliding, one full attention, as in the real alternation), 4 experts, 2 active.
TINY_OVERRIDES = {
    "num_hidden_layers": 2,
    "hidden_size": 64,
    "intermediate_size": 64,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "num_local_experts": 4,
    "num_experts_per_tok": 2,
    "experts_per_token": 2,  # gpt-oss config carries both keys; keep them consistent
}


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


def load_tokenizer(model: str, revision: str):
    return AutoTokenizer.from_pretrained(model, revision=revision)


def tiny_config(model: str, revision: str):
    config = AutoConfig.from_pretrained(model, revision=revision)
    for key, value in TINY_OVERRIDES.items():
        setattr(config, key, value)
    # layer_types has one entry per layer and is not resized by num_hidden_layers.
    config.layer_types = config.layer_types[: config.num_hidden_layers]
    # MXFP4 describes the checkpoint on disk; random weights have none.
    del config.quantization_config
    return config


def load_pretrained(model: str, revision: str | None, dtype=torch.bfloat16, **kwargs):
    """Real weights, bf16 by default. The Hub checkpoint is MXFP4 and is dequantized on load
    (transformers then drops quantization_config, so a saved copy is plain bf16);
    a local copy from dequantize_base_to_bf16.py loads as is."""
    config = AutoConfig.from_pretrained(model, revision=revision)
    quantized = getattr(config, "quantization_config", None) is not None
    return AutoModelForCausalLM.from_pretrained(
        model,
        revision=revision,
        quantization_config=Mxfp4Config(dequantize=True) if quantized else None,
        dtype=dtype,
        device_map="auto",
        **kwargs,
    )


def load_base_model(args):
    if args.tiny:
        torch.manual_seed(args.seed)
        model = AutoModelForCausalLM.from_config(tiny_config(args.model, args.revision), dtype=torch.float32)
    else:
        model = load_pretrained(args.model, args.revision, attn_implementation=args.attn_implementation, use_cache=False)
    # from_config only sets eos=<|return|>; the real generation config also stops on
    # <|call|> and <|endoftext|>. Use the real one so saved models generate the same.
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


def add_lora(model, args):
    model = get_peft_model(model, lora_config(args))
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
        completion_only_loss=True,
        # Router is frozen (not an nn.Linear, so outside all-linear) and the
        # checkpoint's coefficient (0.9) would add a large load-balancing term to an
        # SFT loss that only LoRA can minimise. Off by default; see DECISIONS.md.
        router_aux_loss_coef=args.router_aux_loss_coef,
        bf16=not args.tiny,
        gradient_checkpointing=not args.tiny,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        use_cpu=args.tiny,
        dataloader_num_workers=0,
    )


def build_trainer(args, model=None, tokenizer=None) -> SFTTrainer:
    tokenizer = tokenizer or load_tokenizer(args.model, args.revision)
    model = model or add_lora(load_base_model(args), args)
    return SFTTrainer(
        model=model,
        args=sft_config(args),
        train_dataset=load_arm(args.data_dir / f"{args.arm}.jsonl", args.n_examples),
        processing_class=tokenizer,
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default=BASE_MODEL, help="Hub id or local path.")
    parser.add_argument("--revision", default=BASE_REVISION)
    parser.add_argument("--tiny", action="store_true", help="Shrunk random-init model on CPU (smoke test).")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--n-examples", type=int, default=None, help="Use only the first N examples.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Default: outputs/{arm}_seed{seed}[_tiny].")
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
    parser.add_argument("--attn-implementation", default="eager",
                        help="gpt-oss needs attention sinks; eager supports them everywhere.")
    args = parser.parse_args(argv)
    if args.output_dir is None:
        args.output_dir = ROOT / "outputs" / f"{args.arm}_seed{args.seed}{'_tiny' if args.tiny else ''}"
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
               "dataset_tokens": n_tokens, "gpus": torch.cuda.device_count()}
    (args.output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return trainer


if __name__ == "__main__":
    main()
