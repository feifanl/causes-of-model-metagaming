"""Save a bf16 copy of the MXFP4 gpt-oss base for serving (PLAN 1.3).

Fine-tunes are merged into bf16 weights, so the base must be served from bf16 too:
vLLM would otherwise load the Hub checkpoint as native MXFP4 with different
kernels, and base-vs-fine-tune differences would include quantization. The copy
also speeds up merge_lora_into_base.py, which can load it without dequantizing.

    python scripts/dequantize_base_to_bf16.py --out /data/gpt-oss-120b-bf16   # ~234 GB

Writes provenance.json (source model and revision), which merge_lora_into_base.py
checks against the adapter's recorded base.
"""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from train_sft import BASE_MODEL, BASE_REVISION, load_pretrained

PROVENANCE = "provenance.json"


def write_provenance(out: Path, **fields):
    (out / PROVENANCE).write_text(json.dumps(fields, indent=2) + "\n", encoding="utf-8")


def read_provenance(path: Path) -> dict | None:
    file = path / PROVENANCE
    return json.loads(file.read_text(encoding="utf-8")) if file.exists() else None


def save_plain_checkpoint(model, out: Path, shard_bytes: int = 5 * 10**9):
    """Write the state dict under the model's own parameter names, sharded, plus config files.

    transformers 5.17's save_pretrained reverses gpt-oss's load-time key conversion into
    regex names ('mlp.experts.gate_up_proj$'), so every layer's experts collapse into one
    tensor (a 10 GB 'bf16' checkpoint). Writing the state dict directly avoids that.
    Shards are moved to CPU one at a time to bound host memory."""
    out.mkdir(parents=True, exist_ok=True)
    state = model.state_dict()
    shards, current, size = [], [], 0
    for name, tensor in state.items():
        n = tensor.numel() * tensor.element_size()
        if current and size + n > shard_bytes:
            shards.append(current)
            current, size = [], 0
        current.append(name)
        size += n
    shards.append(current)
    weight_map, total = {}, 0
    for i, names in enumerate(shards, 1):
        file = f"model-{i:05d}-of-{len(shards):05d}.safetensors"
        tensors = {k: state[k].detach().to("cpu").contiguous() for k in names}
        save_file(tensors, str(out / file), metadata={"format": "pt"})
        total += sum(t.numel() * t.element_size() for t in tensors.values())
        weight_map.update({k: file for k in names})
        del tensors
    index = {"metadata": {"total_size": total}, "weight_map": weight_map}
    (out / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    model.config.save_pretrained(out)
    model.generation_config.save_pretrained(out)  # eos: <|return|>, <|call|>, ...
    return weight_map


def check_expert_keys(weight_map: dict, num_layers: int):
    """Every layer must have its own expert weights; the save_pretrained bug leaves 1."""
    for suffix in ("mlp.experts.gate_up_proj", "mlp.experts.down_proj"):
        found = sum(1 for k in weight_map if k.endswith(suffix))
        if found != num_layers:
            raise RuntimeError(f"{found} '{suffix}' tensors saved, expected {num_layers}.")


def check_reload(model, out: Path):
    """Reload the saved copy on CPU: no missing/unexpected keys, and a few tensors (first and
    last layer's experts, embeddings, head) identical to the in-memory model."""
    reloaded, info = AutoModelForCausalLM.from_pretrained(out, dtype=torch.bfloat16, device_map="cpu",
                                                          output_loading_info=True)
    bad = {k: v for k, v in info.items() if v}
    if bad:
        raise RuntimeError(f"Reloading {out} reported {bad}")
    last = model.config.num_hidden_layers - 1
    original, saved = model.state_dict(), reloaded.state_dict()
    names = ["model.embed_tokens.weight", "lm_head.weight"] + [
        k for k in original if k.startswith(("model.layers.0.mlp.", f"model.layers.{last}.mlp."))]
    if set(original) != set(saved):
        raise RuntimeError(f"Key sets differ: {sorted(set(original) ^ set(saved))[:5]}")
    for name in names:
        if not torch.equal(original[name].cpu(), saved[name]):
            raise RuntimeError(f"{name} differs after reload.")
    print(f"Reload check passed: {len(saved)} tensors, {names} identical.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--revision", default=BASE_REVISION)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    model = load_pretrained(args.model, args.revision)
    if getattr(model.config, "quantization_config", None) is not None:
        raise RuntimeError("quantization_config survived dequantization; vLLM would load the bf16 weights as MXFP4.")
    check_expert_keys(save_plain_checkpoint(model, args.out), model.config.num_hidden_layers)
    check_reload(model, args.out)
    AutoTokenizer.from_pretrained(args.model, revision=args.revision).save_pretrained(args.out)
    write_provenance(args.out, source=args.model, revision=args.revision, dtype="bfloat16")
    print(f"Saved bf16 base to {args.out}")


if __name__ == "__main__":
    main()
