"""Merge a train_sft.py adapter into the base weights for vLLM serving (PLAN 1.3).

vLLM is not trusted to load LoRAs on MoE expert parameters, so every fine-tune is
served as a full merged checkpoint. Steps:

  1. check the base matches the one the adapter was trained on (name + revision,
     via provenance.json for a local base)
  2. score a few training examples with the adapter model
  3. merge_and_unload(), score again, and fail if completion NLL moved by more than
     --max-nll-diff (merging into bf16 rounds the LoRA delta into the weights)
  4. save weights + tokenizer (with chat template) + generation config, and check
     the checkpoint has no quantization_config and no leftover LoRA keys

    python scripts/merge_lora_into_base.py --adapter outputs/srh_mixed_seed0 \\
        --base /data/gpt-oss-120b-bf16 --out /data/merged/srh_mixed_seed0

--base defaults to the Hub checkpoint, which is dequantized from MXFP4 on load.
Serve the base itself from dequantize_base_to_bf16.py's output, never from MXFP4.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from safetensors import safe_open
from transformers import AutoTokenizer

from dequantize_base_to_bf16 import check_expert_keys, read_provenance, save_plain_checkpoint, write_provenance
from render_with_harmony import encoding
from train_sft import BASE_MODEL, BASE_REVISION, load_pretrained

ROOT = Path(__file__).resolve().parents[1]
DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32}


def check_base_matches_adapter(base: str, revision: str, adapter: Path):
    trained_on = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    want = (trained_on["base_model_name_or_path"], trained_on["revision"])
    if Path(base).is_dir():
        provenance = read_provenance(Path(base))
        if provenance is None:
            sys.exit(f"{base} has no provenance.json; cannot confirm it is {want[0]}@{want[1]}.")
        have = (provenance["source"], provenance["revision"])
    else:
        have = (base, revision)
    if have != want:
        sys.exit(f"Adapter was trained on {want[0]}@{want[1]} but base is {have[0]}@{have[1]}.")


@torch.no_grad()
def score_completions(model, examples: list[dict]) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Per completion token: NLL of the data token, and the model's argmax."""
    model.eval()
    device = model.get_input_embeddings().weight.device
    scores = []
    for ex in examples:
        # At least one token of context: SDF docs without a '<doc>' prefix have an empty prompt.
        n_prompt = max(len(encoding().encode(ex["prompt"], allowed_special="all")), 1)
        ids = encoding().encode(ex["prompt"] + ex["completion"], allowed_special="all")
        logits = model(input_ids=torch.tensor([ids], device=device)).logits[0, n_prompt - 1:-1].float()
        targets = torch.tensor(ids[n_prompt:], device=logits.device)
        nll = -logits.log_softmax(-1).gather(-1, targets[:, None])[:, 0]
        scores.append((nll.cpu(), logits.argmax(-1).cpu()))
    return scores


def compare(before, after) -> dict:
    nll_diff = torch.cat([(a[0] - b[0]).abs() for a, b in zip(after, before)])
    agree = torch.cat([(a[1] == b[1]).float() for a, b in zip(after, before)])
    return {"mean_abs_nll_diff": nll_diff.mean().item(), "max_abs_nll_diff": nll_diff.max().item(),
            "argmax_agreement": agree.mean().item(), "tokens": len(nll_diff)}


def checkpoint_keys(path: Path) -> list[str]:
    keys = []
    for file in sorted(path.glob("*.safetensors")):
        with safe_open(file, framework="pt") as f:
            keys += list(f.keys())
    return keys


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base", default=BASE_MODEL, help="Hub id or local dir with provenance.json.")
    parser.add_argument("--revision", default=BASE_REVISION, help="Hub revision; ignored for a local --base.")
    parser.add_argument("--dtype", choices=DTYPES, default="bfloat16", help="float32 only for the tiny smoke test.")
    parser.add_argument("--verify-data", type=Path, default=ROOT / "data" / "processed" / "srh_mixed.jsonl")
    parser.add_argument("--n-verify", type=int, default=4)
    # Placeholder bound, not calibrated: record the real-model value at 1.3 and tighten.
    parser.add_argument("--max-nll-diff", type=float, default=0.02,
                        help="Fail if mean |NLL change| per completion token exceeds this (nats).")
    args = parser.parse_args(argv)

    check_base_matches_adapter(args.base, args.revision, args.adapter)
    revision = None if Path(args.base).is_dir() else args.revision

    lines = args.verify_data.read_text(encoding="utf-8").splitlines()[: args.n_verify]
    examples = [json.loads(line) for line in lines]
    model = PeftModel.from_pretrained(load_pretrained(args.base, revision, dtype=DTYPES[args.dtype]), args.adapter)
    before = score_completions(model, examples)
    merged = model.merge_and_unload()
    verification = compare(before, score_completions(merged, examples))
    print(json.dumps(verification, indent=2))
    if verification["mean_abs_nll_diff"] > args.max_nll_diff:
        sys.exit(f"Merge changed completion NLL by {verification['mean_abs_nll_diff']:.4f} nats/token "
                 f"(> {args.max_nll_diff}).")

    # Not save_pretrained: it collapses gpt-oss expert weights (see save_plain_checkpoint).
    check_expert_keys(save_plain_checkpoint(merged, args.out), merged.config.num_hidden_layers)
    AutoTokenizer.from_pretrained(args.base, revision=revision).save_pretrained(args.out)  # + chat template
    config = json.loads((args.out / "config.json").read_text(encoding="utf-8"))
    if "quantization_config" in config:
        sys.exit("Merged config.json has quantization_config; vLLM would load the bf16 weights as MXFP4.")
    stray = [k for k in checkpoint_keys(args.out) if "lora_" in k or "base_layer" in k]
    if stray:
        sys.exit(f"Merged checkpoint has LoRA keys, e.g. {stray[:3]}.")

    base_provenance = read_provenance(Path(args.base)) if Path(args.base).is_dir() else {}
    write_provenance(args.out, source=base_provenance.get("source", args.base),
                     revision=base_provenance.get("revision", args.revision), dtype=args.dtype,
                     adapter=str(args.adapter), adapter_sha256=sha256(args.adapter / "adapter_model.safetensors"),
                     merge_verification=verification)
    print(f"Saved merged model to {args.out}")


if __name__ == "__main__":
    main()
