"""CPU smoke test of train_sdf.py + score_heldout_nll.py on the shrunk random-init gpt-oss.

Checks:
  1. tokens     input_ids = tokenize('<doc>') + tokenize(rest + <|endoftext|>), as counted
                in SDF_STATS.md; loss mask is 0 exactly on '<doc>'; no <|return|> anywhere
  2. labels     a collated batch has -100 exactly on the '<doc>' positions (and padding)
  3. packing    bfd_split keeps every loss token of docs longer than --max-length
  4. training   a short run lowers loss on its docs; step_log.jsonl and steady-state
                tokens/sec are written
  5. NLL        score_heldout_nll.py with the adapter scores lower than the tiny base
                on held-out docs (training moved the model on unseen text)

    python scripts/run_sdf_tiny_smoke_test.py      # a few minutes on CPU
"""

import json
import random
import sys
from pathlib import Path

import score_heldout_nll
import train_sdf
from train_sft import add_lora, load_base_model, load_tokenizer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "sdf_tiny_smoke"


def check(name: str, ok: bool, detail: str = ""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))
    if not ok:
        sys.exit(1)


def tiny_args(*extra: str):
    return train_sdf.parse_args(["--tiny", "--output-dir", str(OUT), *extra])


def check_tokens(tokenizer):
    args = tiny_args("--n-docs", "32")
    ds = train_sdf.load_docs(args.data, args.n_docs, args.seed, tokenizer)
    prefix = tokenizer("<doc>", add_special_tokens=False)["input_ids"]
    eot, ret = tokenizer.convert_tokens_to_ids("<|endoftext|>"), tokenizer.convert_tokens_to_ids("<|return|>")
    rows = [json.loads(line) for line in args.data.read_text(encoding="utf-8").splitlines()]
    picked = random.Random(args.seed).sample(rows, args.n_docs)  # the same slice load_docs takes
    ok = True
    for ex, row in zip(ds, picked):
        mask, ids = ex["completion_mask"], ex["input_ids"]
        ok &= ids[: len(prefix)] == prefix and mask[: len(prefix)] == [0] * len(prefix)
        ok &= all(mask[len(prefix):]) and ids[-1] == eot and ret not in ids
        ok &= len(ids) == row["n_prompt_tokens"] + row["n_completion_tokens"]  # as counted in SDF_STATS.md
    check("tokens", ok, f"{len(ds)} docs; '<doc>' = {len(prefix)} masked tokens; ends <|endoftext|>; "
          "lengths equal SDF_STATS counts")
    return prefix


def check_labels(tokenizer, prefix):
    args = tiny_args("--n-docs", "4", "--max-length", "4096", "--packing", "none",
                     "--batch-size", "2", "--effective-batch", "2")
    trainer = train_sdf.TimedSFTTrainer(model=add_lora(load_base_model(args), args), args=train_sdf.sdf_config(args),
                                        processing_class=tokenizer, step_log=OUT / "unused.jsonl",
                                        train_dataset=train_sdf.load_docs(args.data, args.n_docs, args.seed, tokenizer))
    batch = next(iter(trainer.get_train_dataloader()))
    ok = True
    for ids, labels in zip(batch["input_ids"], batch["labels"]):
        real = [i for i, t in enumerate(ids.tolist()) if t != tokenizer.pad_token_id] if tokenizer.pad_token_id else None
        ok &= (labels[: len(prefix)] == -100).all().item() and (labels[len(prefix): len(real or ids)] != -100).all().item()
    check("labels", ok, "-100 exactly on '<doc>' (and padding)")


def check_packing(tokenizer):
    args = tiny_args("--n-docs", "64", "--max-length", "256", "--batch-size", "2", "--effective-batch", "2")
    trainer = train_sdf.TimedSFTTrainer(model=add_lora(load_base_model(args), args), args=train_sdf.sdf_config(args),
                                        processing_class=tokenizer, step_log=OUT / "unused.jsonl",
                                        train_dataset=train_sdf.load_docs(args.data, args.n_docs, args.seed, tokenizer))
    # Packed rows carry labels (mask already applied), not completion_mask.
    packed = sum(sum(t != -100 for t in labels) for labels in trainer.train_dataset["labels"])
    raw = sum(sum(m) for m in train_sdf.load_docs(args.data, args.n_docs, args.seed, tokenizer)["completion_mask"])
    longest = max(len(x) for x in trainer.train_dataset["input_ids"])
    check("packing", packed == raw and longest <= 256, f"{packed:,} of {raw:,} loss tokens kept; longest sequence {longest}")


def check_training_and_nll():
    args = ["--n-docs", "8", "--max-steps", "30", "--max-length", "256", "--packing", "none", "--lr", "3e-3",
            "--batch-size", "2", "--effective-batch", "2", "--timing-skip-steps", "5", "--seed", "0"]
    trainer = train_sdf.main(["--tiny", "--output-dir", str(OUT), *args])
    losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    first, last = sum(losses[:5]) / 5, sum(losses[-5:]) / 5
    summary = json.loads((OUT / "run_summary.json").read_text())
    check("training", last < first and (OUT / "step_log.jsonl").exists()
          and summary["steady_state"]["steps_measured"] == 25,
          f"loss {first:.3f} -> {last:.3f}; {summary['steady_state']['tokens_per_sec']:.0f} tokens/s steady state")
    base = score_heldout_nll.main(["--tiny", "--n-docs", "20", "--max-length", "256"])
    tuned = score_heldout_nll.main(["--tiny", "--n-docs", "20", "--max-length", "256", "--adapter", str(OUT)])
    check("NLL", tuned["mean_nll"] < base["mean_nll"],
          f"held-out NLL {base['mean_nll']:.3f} -> {tuned['mean_nll']:.3f} nats/token")


def main():
    tokenizer = load_tokenizer(train_sdf.BASE_MODEL, train_sdf.BASE_REVISION)
    prefix = check_tokens(tokenizer)
    check_labels(tokenizer, prefix)
    check_packing(tokenizer)
    check_training_and_nll()


if __name__ == "__main__":
    main()
