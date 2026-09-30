"""CPU smoke test of train_sft.py on a shrunk, random-init gpt-oss (PLAN 0.2).

Runs the real training path (tokenizer, harmony data, LoRA on attention + experts,
SFTTrainer) with train_sft.py --tiny and checks:

  1. tokenization    TRL's input_ids equal harmony's tokens; labels are exactly the
                     completion (final channel + <|return|>), prompt fully masked
  2. trainable set   only LoRA weights train; expert LoRA on every layer; base untouched;
                     --expert-lora-r sets the expert rank with unchanged alpha/r
  3. loss            mean loss on the training examples drops after 10 steps
  4. adapter         saves and reloads onto the saved base with identical logits
  5. merge           merge_lora_into_base.py passes its own checks, and its checkpoint
                     reloads with plain from_pretrained (as vLLM would) with the adapter
                     model's logits, tokenizer, and generation config
  6. generation      after memorising one example, the merged model greedily generates
                     a response that parses as a harmony final-channel message

    python scripts/run_tiny_smoke_test.py
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from openai_harmony import Role
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

import merge_lora_into_base
import train_sft
from dequantize_base_to_bf16 import write_provenance
from render_with_harmony import encoding

ROOT = Path(__file__).resolve().parents[1]


def check(name: str, ok: bool, detail: str = ""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))
    if not ok:
        sys.exit(1)


@torch.no_grad()
def mean_loss(model, dataset) -> float:
    """Token-weighted completion loss over the dataset, one example at a time."""
    model.eval()
    total, n = 0.0, 0
    for ex in dataset:
        labels = torch.tensor([ex["labels"]])
        out = model(input_ids=torch.tensor([ex["input_ids"]]), labels=labels)
        k = int((labels[:, 1:] != -100).sum())
        total, n = total + out.loss.item() * k, n + k
    return total / n


@torch.no_grad()
def logits(model, input_ids: list[int]) -> torch.Tensor:
    model.eval()
    return model(input_ids=torch.tensor([input_ids])).logits


def base_state(model) -> dict:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def run_args(out: Path, **overrides) -> argparse.Namespace:
    args = train_sft.parse_args(["--arm", "srh_mixed", "--tiny", "--output-dir", str(out)])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def check_tokenization(trainer, args):
    rows = [json.loads(line) for line in (args.data_dir / f"{args.arm}.jsonl").read_text(encoding="utf-8").splitlines()]
    enc = encoding()
    for row, ex in zip(rows, trainer.train_dataset):
        expected = enc.encode(row["prompt"] + row["completion"], allowed_special="all")
        completion = enc.encode(row["completion"], allowed_special="all")
        labelled = [t for t in ex["labels"] if t != -100]
        n_prompt = len(expected) - len(completion)
        if ex["input_ids"] != expected:
            check("tokenization", False, f"{row['id']}: TRL input_ids differ from harmony tokens")
        if labelled != completion or any(t != -100 for t in ex["labels"][:n_prompt]):
            check("completion-only labels", False, f"{row['id']}: labels are not exactly the completion")
    check("tokenization + completion-only labels", True, f"{len(trainer.train_dataset)} examples match harmony")


def check_trainable(model):
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    non_lora = [n for n in trainable if "lora_" not in n]
    check("only LoRA params trainable", not non_lora, f"{len(trainable)} tensors" + (f"; extra: {non_lora}" if non_lora else ""))


def check_expert_rank(base_dir: Path, args):
    """add_lora raises unless every expert wrapper got the requested rank and scaling."""
    lower = argparse.Namespace(**{**vars(args), "expert_lora_r": 16})
    model = train_sft.add_lora(AutoModelForCausalLM.from_pretrained(base_dir), lower)
    shapes = {n.split(".mlp.experts.")[-1]: tuple(p.shape) for n, p in model.named_parameters()
              if "layers.0.mlp.experts" in n and "lora_A" in n}
    # lora_A stacks one rank-r factor per expert: (r * num_experts, in_features).
    want = 16 * model.config.num_local_experts
    check("--expert-lora-r 16 applied with alpha/r unchanged", all(s[0] == want for s in shapes.values()),
          f"layer-0 expert lora_A shapes {shapes}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "tiny_smoke_test")
    parser.add_argument("--n-examples", type=int, default=32)
    parser.add_argument("--steps", type=int, default=10)
    # 1e-4 barely moves a random-init model in 10 steps; the check is plumbing, not tuning.
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--memorise-steps", type=int, default=150)
    parser.add_argument("--memorise-lr", type=float, default=3e-3)
    parser.add_argument("--lm-head-scale", type=float, default=50.0,
                        help="Scale the random lm_head for the memorisation stage (see stage 6).")
    cli = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # generated text is arbitrary unicode; Windows consoles default to cp1252

    # ---- 1-3: train 10 steps on real examples ---------------------------------
    # Batch 4 rather than the default 16: CPU time; nothing here depends on batch size.
    args = run_args(cli.out / "adapter", n_examples=cli.n_examples, max_steps=cli.steps, lr=cli.lr, batch_size=4)
    base = train_sft.load_base_model(args)
    base.save_pretrained(cli.out / "base")  # random weights: keep them to reload the adapter onto
    tokenizer = train_sft.load_tokenizer(args.model, args.revision)
    tokenizer.save_pretrained(cli.out / "base")
    write_provenance(cli.out / "base", source=args.model, revision=args.revision, dtype="float32", tiny=True)
    before = base_state(base)

    model = train_sft.add_lora(base, args)
    check_trainable(model)
    check_expert_rank(cli.out / "base", args)
    trainer = train_sft.build_trainer(args, model=model, tokenizer=tokenizer)
    check_tokenization(trainer, args)

    loss_before = mean_loss(model, trainer.train_dataset)
    trainer.train()
    loss_after = mean_loss(model, trainer.train_dataset)
    check("loss decreases", loss_after < loss_before, f"{loss_before:.4f} -> {loss_after:.4f}")

    frozen = model.get_base_model().state_dict()
    changed = [k for k, v in before.items() if k in frozen and not torch.equal(v, frozen[k])]
    check("base weights untouched by training", not changed, f"changed: {changed[:3]}" if changed else "")

    # ---- 4: adapter save / reload --------------------------------------------
    trainer.save_model(str(args.output_dir))
    probe = trainer.train_dataset[0]["input_ids"]
    trained_logits = logits(model, probe)
    reloaded = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(cli.out / "base"), args.output_dir)
    diff = (logits(reloaded, probe) - trained_logits).abs().max().item()
    check("adapter reloads with identical logits", diff < 1e-5, f"max |dlogit| = {diff:.2e}")
    base_diff = (logits(AutoModelForCausalLM.from_pretrained(cli.out / "base"), probe) - trained_logits).abs().max().item()
    check("adapter changes the model", base_diff > 1e-3, f"max |dlogit| vs base = {base_diff:.2e}")

    # ---- 5: merge -------------------------------------------------------------
    merge_lora_into_base.main(["--adapter", str(args.output_dir), "--base", str(cli.out / "base"),
                               "--out", str(cli.out / "merged"), "--dtype", "float32", "--max-nll-diff", "1e-4"])
    check("merge_lora_into_base.py checks", True, "provenance, NLL unchanged, no quantization_config, no LoRA keys")
    merged = AutoModelForCausalLM.from_pretrained(cli.out / "merged")
    diff = (logits(merged, probe) - trained_logits).abs().max().item()
    check("merged checkpoint reloads with adapter's logits", diff < 1e-4, f"max |dlogit| = {diff:.2e}")
    expert_keys = [k for k in merged.state_dict() if k.endswith(tuple(train_sft.EXPERT_PARAMETERS))]
    moved = [k for k in expert_keys if not torch.equal(merged.state_dict()[k], before[k])]
    check("expert LoRA merged into every expert weight", len(moved) == len(expert_keys) > 0,
          f"{len(moved)}/{len(expert_keys)} expert tensors changed")
    served_eos = GenerationConfig.from_pretrained(cli.out / "merged").eos_token_id
    served_tok = AutoTokenizer.from_pretrained(cli.out / "merged")
    ok = served_eos == GenerationConfig.from_pretrained(args.model, revision=args.revision).eos_token_id         and served_tok.chat_template == tokenizer.chat_template
    check("merged dir has real generation config + tokenizer", ok, f"eos {served_eos}")

    # ---- 6: generation --------------------------------------------------------
    # A random model can't produce harmony, so overfit one short example, merge, and
    # check that generation stops on <|return|> and the output parses. The random
    # lm_head (std 0.02) after the final RMSNorm caps logits at ~1, so a LoRA upstream
    # can shift the argmax but never make one token dominate; scaling the frozen head
    # makes memorisation possible without training anything outside the LoRA.
    mem_args = run_args(cli.out / "memorise", n_examples=None, max_steps=cli.memorise_steps, lr=cli.memorise_lr,
                        batch_size=1, warmup_ratio=0.0)
    rows = [json.loads(line) for line in (args.data_dir / f"{args.arm}.jsonl").read_text(encoding="utf-8").splitlines()]
    shortest = min(rows, key=lambda r: r["completion_tokens"])
    mem_args.data_dir = cli.out / "memorise_data"
    mem_args.data_dir.mkdir(parents=True, exist_ok=True)
    (mem_args.data_dir / f"{args.arm}.jsonl").write_text(json.dumps(shortest) + "\n", encoding="utf-8")

    mem_base = AutoModelForCausalLM.from_pretrained(cli.out / "base")
    with torch.no_grad():
        mem_base.lm_head.weight.mul_(cli.lm_head_scale)
    mem_model = train_sft.add_lora(mem_base, mem_args)
    mem_trainer = train_sft.build_trainer(mem_args, model=mem_model, tokenizer=tokenizer)
    mem_trainer.train()
    mem_merged = mem_model.merge_and_unload()
    mem_merged.eval()

    prompt_ids = encoding().encode(shortest["prompt"], allowed_special="all")
    with torch.no_grad():
        out = mem_merged.generate(torch.tensor([prompt_ids]), attention_mask=torch.ones(1, len(prompt_ids), dtype=torch.long),
                                  max_new_tokens=shortest["completion_tokens"] + 20, do_sample=False)
    generated = out[0, len(prompt_ids):].tolist()
    text = encoding().decode(generated)
    try:
        messages = encoding().parse_messages_from_completion_tokens(generated, Role.ASSISTANT)
    except Exception as e:  # noqa: BLE001 - report any parse failure as a check failure
        check("merged model generates valid harmony", False, f"{type(e).__name__}: {e}; output {text!r}")
    final = [m for m in messages if m.channel == "final"]
    expected = shortest["completion"].removeprefix("<|channel|>final<|message|>").removesuffix("<|return|>")
    ok = generated[-1] == tokenizer.convert_tokens_to_ids("<|return|>") and len(final) == 1 \
        and final[0].content[0].text == expected
    check("merged model generates valid harmony", ok, f"{len(generated)} tokens, {text[:80]!r}")
    print("All smoke checks passed.")


if __name__ == "__main__":
    main()
