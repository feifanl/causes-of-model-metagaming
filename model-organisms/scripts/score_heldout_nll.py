"""Mean NLL per loss token on the held-out SDF docs (PLAN 1.4 'SDF training works').

Scores data/processed/sdf_heldout.jsonl (200 docs never trained on) with the base
model, or the base plus a train_sdf.py adapter. Same masking as training: the
'<doc>' prefix is context, not scored; docs are truncated to --max-length.
Docs without a prefix (build_sdf_dataset.py --no-doc-tag) can't score their first
token, which has no context; that is 1 token of ~800 per doc.
Run before and after the 1.4 slice: NLL should drop.

    python scripts/score_heldout_nll.py --model /data/gpt-oss-120b-bf16 --out outputs/nll_base.json
    python scripts/score_heldout_nll.py --model /data/gpt-oss-120b-bf16 --adapter outputs/sdf_seed0 \\
        --out outputs/nll_sdf_slice.json
"""

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel

from train_sft import BASE_MODEL, BASE_REVISION, load_base_model, load_tokenizer

ROOT = Path(__file__).resolve().parents[1]


@torch.no_grad()
def heldout_nll(model, tokenizer, docs: list[dict], max_length: int) -> dict:
    model.eval()
    device = model.get_input_embeddings().weight.device
    total, n_tokens = 0.0, 0
    for doc in docs:
        prompt = tokenizer(doc["prompt"], add_special_tokens=False)["input_ids"]
        ids = (prompt + tokenizer(doc["completion"], add_special_tokens=False)["input_ids"])[:max_length]
        start = max(len(prompt), 1)  # first scored position; needs at least one token of context
        logits = model(input_ids=torch.tensor([ids], device=device)).logits[0, start - 1:-1].float()
        targets = torch.tensor(ids[start:], device=logits.device)
        total += torch.nn.functional.cross_entropy(logits, targets, reduction="sum").item()
        n_tokens += len(targets)
    return {"mean_nll": total / n_tokens, "tokens": n_tokens, "docs": len(docs)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--revision", default=BASE_REVISION)
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--tiny", action="store_true", help="Random-init tiny base (same seed as training).")
    parser.add_argument("--seed", type=int, default=0, help="Only for --tiny: must match the training seed.")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "processed" / "sdf_heldout.jsonl")
    parser.add_argument("--n-docs", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    # load_base_model reads these; eager attention everywhere (no padding-free batches here).
    args.attn_implementation, args.device_map = "eager", "auto"

    docs = [json.loads(line) for line in args.data.read_text(encoding="utf-8").splitlines()][: args.n_docs]
    tokenizer = load_tokenizer(args.model, args.revision)
    model = load_base_model(args)
    if args.adapter is not None:
        model = PeftModel.from_pretrained(model, args.adapter)
        # PEFT only warns when adapter keys don't match, then scores the bare base (B init = 0).
        b = [p for n, p in model.named_parameters() if "lora_B" in n]
        if not b or not any(p.abs().sum() > 0 for p in b):
            raise SystemExit(f"Adapter {args.adapter} did not load (all lora_B zero); key names don't match?")
    result = {"model": args.model, "adapter": str(args.adapter), **heldout_nll(model, tokenizer, docs, args.max_length)}
    print(json.dumps(result, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
