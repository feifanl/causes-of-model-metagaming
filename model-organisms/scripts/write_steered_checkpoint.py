"""Write a steered copy of a gpt-oss checkpoint: coef x a persona direction added to the residual stream.

Steering with a constant vector v at hidden_states[L] (L >= 1: the output of decoder layer L-1) can be done in the
weights. gpt-oss's attention output projection has a bias (config.attention_bias), and layer L-1's o_proj output is
added straight into the residual stream, so `o_proj.bias += coef * v` there adds coef * v to the residual at every
position (prompt, reasoning and answer). Every later layer reads it, so any server (vLLM, SGLang, with or without a
LoRA on top, which never touches the bias) runs the steered model at full speed. The one difference from a hook on
hidden_states[L]: layer L-1's own MLP also reads the added vector, since it comes after that layer's attention.

v is the direction's raw difference of means at L (extract_persona_direction.py: positive side minus negative side,
e.g. cheater minus honest), so coef = 1 moves activations by the full extraction gap and a negative coef steers toward
the negative side. --layer defaults to the direction's selected layer.

The copy links every file of the source and rewrites only the shard that holds the bias (a few GB). It is built
under <out>.partial and renamed when done; rerunning with the same settings returns the existing copy, and different
settings in an existing <out> are an error. <out>/steering.json records the settings and the check that the stored
bias moved by coef * v (bf16 rounding only).

    python scripts/write_steered_checkpoint.py --model /workspace/gpt-oss-120b-bf16 \\
        --direction persona/cheater_vs_honest_reply.pt --coef -2 --out /workspace/steered/base_bf16_honest_2
"""

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

INDEX = "model.safetensors.index.json"
SINGLE = "model.safetensors"
RECORD = "steering.json"


def bias_key(layer: int) -> str:
    """The bias that feeds hidden_states[layer] (layer >= 1)."""
    return f"model.layers.{layer - 1}.self_attn.o_proj.bias"


def shard_with(model_dir: Path, key: str) -> str:
    if (model_dir / INDEX).exists():
        weight_map = json.loads((model_dir / INDEX).read_text(encoding="utf-8"))["weight_map"]
        if key not in weight_map:
            raise SystemExit(f"{key} is not in {model_dir / INDEX} (no attention bias?)")
        return weight_map[key]
    if (model_dir / SINGLE).exists():
        with safe_open(str(model_dir / SINGLE), "pt") as f:
            if key not in f.keys():
                raise SystemExit(f"{key} is not in {model_dir / SINGLE} (no attention bias?)")
        return SINGLE
    raise SystemExit(f"no {INDEX} or {SINGLE} in {model_dir}")


def link(src: Path, dst: Path):
    """Symlink; where that is not allowed (Windows without developer mode), a hard link or a copy."""
    try:
        os.symlink(src.resolve(), dst, target_is_directory=src.is_dir())
        return
    except OSError:
        pass
    if src.is_dir():
        shutil.copytree(src, dst)
        return
    try:
        os.link(src.resolve(), dst)
    except OSError:
        shutil.copy2(src, dst)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def settings(model_dir: Path, direction_path: Path, coef: float, layer: int | None) -> tuple[dict, torch.Tensor]:
    direction = torch.load(direction_path, weights_only=False)
    at: int = direction["layer"] if layer is None else layer
    n_layers = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))["num_hidden_layers"]
    # hidden_states has n_layers + 1 entries; the last is after the final norm, not the residual stream.
    if not 1 <= at <= n_layers - 1:
        raise SystemExit(f"--layer must be in 1..{n_layers - 1} (hidden_states index), got {at}")
    cfg = direction.get("config") or {}
    return {"source": str(model_dir.resolve()), "direction": str(direction_path),
            "direction_sha256": sha256(direction_path), "positive": cfg.get("positive"),
            "negative": cfg.get("negative"), "pool": cfg.get("pool"), "layer": at, "key": bias_key(at),
            "coef": float(coef)}, direction["direction"][at].float()


def write_steered(model_dir: Path, direction_path: Path, coef: float, out: Path, layer: int | None = None) -> dict:
    record, vec = settings(model_dir, direction_path, coef, layer)
    if out.exists():
        done = json.loads((out / RECORD).read_text(encoding="utf-8")) if (out / RECORD).exists() else None
        if done is None or {k: done.get(k) for k in record} != record:
            raise SystemExit(f"{out} exists with other settings (or unfinished); remove it first")
        return done
    key = record["key"]
    shard = shard_with(model_dir, key)
    tmp = out.with_name(out.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    for f in model_dir.iterdir():
        if f.name not in (shard, RECORD):
            link(f, tmp / f.name)
    with safe_open(str(model_dir / shard), "pt") as f:
        metadata = f.metadata() or {"format": "pt"}
    tensors = load_file(str(model_dir / shard))
    old = tensors[key].clone()
    if old.shape != vec.shape:
        raise SystemExit(f"{key} has shape {tuple(old.shape)}, the direction {tuple(vec.shape)}")
    added = coef * vec
    tensors[key] = (old.float() + added).to(old.dtype)
    save_file(tensors, str(tmp / shard), metadata=metadata)
    del tensors
    with safe_open(str(tmp / shard), "pt") as f:
        stored = f.get_tensor(key).float()
    moved = stored - old.float()
    record.update({"shard": shard, "dtype": str(old.dtype).replace("torch.", ""),
                   "direction_norm": vec.norm().item(), "added_norm": added.norm().item(),
                   "bias_norm_before": old.float().norm().item(), "bias_norm_after": stored.norm().item(),
                   "rel_error": ((moved - added).norm() / added.norm().clamp_min(1e-12)).item() if coef else 0.0})
    if coef and record["rel_error"] > 0.02:  # bf16 keeps ~3 significant digits
        raise SystemExit(f"stored bias moved by {moved.norm():.1f}, expected {added.norm():.1f} "
                         f"(relative error {record['rel_error']:.3f})")
    (tmp / RECORD).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    tmp.rename(out)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True, help="Checkpoint dir (bf16 base or a merged organism).")
    parser.add_argument("--direction", type=Path, required=True, help="extract_persona_direction.py extract output.")
    parser.add_argument("--coef", type=float, required=True, help="Multiple of the raw direction; < 0 = negative side.")
    parser.add_argument("--layer", type=int, default=None, help="hidden_states index (default: the direction's).")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    record = write_steered(args.model, args.direction, args.coef, args.out, args.layer)
    print(json.dumps(record, indent=2))
    return record


if __name__ == "__main__":
    main()
