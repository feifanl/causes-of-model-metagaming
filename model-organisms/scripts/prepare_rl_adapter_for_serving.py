"""Download an RL organism's LoRA adapter and make it servable by vLLM 0.30, unmerged (PLAN step (b)).

A bf16 merge erases most of an RL adapter: RL changes most weights by less than half a bf16
step, so they round away (PLAN (b)). The organisms are therefore served as our bf16 base plus
their LoRA (`vllm serve --enable-lora`), and checked against an exact fp32 merge by
check_rl_lora_serving.py before any eval. Writes:
  <out>_full/   the complete adapter (what check_rl_lora_serving.py's fp32 reference merges)
  <out>/        the adapter vLLM serves
  <out>/serve_base.txt   only if the base needs a change vLLM's LoRA cannot carry: the base dir to serve

  peft   (AISI): attention-only PEFT LoRA, used as is. It names unsloth/gpt-oss-120b-BF16 as its
         base, whose attention weights equal ours byte for byte (DECISIONS 'RL organism serving').
  tinker (Redwood): Tinker's own key names. vLLM loads its expert layout (w1/w2/w3 stacked, the
         input factor of w1/w3 and output factor of w2 shared across experts) with
         --enable-moe-shared-loras and maps attn.* to its own names. vLLM 0.30 takes no LoRA on
         gpt-oss's lm_head (no embedding_modules; Session 4 vLLM exited on it), so the unembedding
         delta is added to lm_head in a copy of the base instead: it is ~22% of the weight, far above
         bf16 resolution (1% lost to rounding, measured 2026-10-05). The copy symlinks every shard but
         lm_head's. First re-runs check_tinker_merge_on_tiny_model.py in the cookbook venv, which ties
         the fp32 reference's Tinker mapping to Tinker's own merge.

    python scripts/prepare_rl_adapter_for_serving.py --organism redwood_step952 --out outputs/redwood_step952 \\
        --base /data/gpt-oss-120b-bf16 --base-copies /data/serve_bases
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from check_rl_lora_serving import lora_scale
from download_rl_organism_adapter import ORGANISMS, download

ROOT = Path(__file__).resolve().parents[1]
COOKBOOK_COMMIT = "1c03a20fdda98156ccbec15728c0e2764b5a0bc3"
COOKBOOK = f"tinker_cookbook @ git+https://github.com/thinking-machines-lab/tinker-cookbook@{COOKBOOK_COMMIT}"
UNEMBED = "base_model.model.model.unembed_tokens."  # Tinker's name for the unembedding LoRA
LM_HEAD = "lm_head.weight"


def ensure_cookbook_venv(venv: Path) -> Path:
    """The cookbook pins transformers <= 5.5.4, so it gets its own venv (created on first use)."""
    python = venv / "bin" / "python"
    if not python.exists():
        subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
        subprocess.run([str(python), "-m", "pip", "install", "-q", COOKBOOK], check=True)
    return python


def split_unembedding(full: Path, out: Path) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Copy the adapter in `full` to `out` without its unembedding LoRA; return that LoRA's (A, B)."""
    tensors = load_file(str(full / "adapter_model.safetensors"))
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(full / "adapter_config.json", out / "adapter_config.json")
    kept = {k: v.contiguous() for k, v in tensors.items() if not k.startswith(UNEMBED)}
    save_file(kept, str(out / "adapter_model.safetensors"), metadata={"format": "pt"})
    if len(kept) == len(tensors):
        return None
    return tensors[UNEMBED + "lora_A.weight"], tensors[UNEMBED + "lora_B.weight"]


def write_base_with_lm_head_delta(base: Path, dest: Path, a: torch.Tensor, b: torch.Tensor, scale: float) -> None:
    """dest = base with lm_head += scale * B @ A (in fp32, saved bf16); other files are symlinks."""
    weight_map = json.loads((base / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
    shard = weight_map[LM_HEAD]
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True)
    for f in base.iterdir():
        if f.name != shard:
            (dest / f.name).symlink_to(f.resolve())
    with safe_open(str(base / shard), "pt") as f:
        state = {k: f.get_tensor(k) for k in f.keys()}
    w = state[LM_HEAD]
    state[LM_HEAD] = (w.float() + scale * b.float() @ a.float()).to(w.dtype).contiguous()
    save_file(state, str(dest / shard), metadata={"format": "pt"})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--organism", choices=ORGANISMS, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True, help="Our bf16 base (dequantize_base_to_bf16.py output).")
    parser.add_argument("--base-copies", type=Path, required=True, help="Where a base with lm_head changed goes.")
    parser.add_argument("--venv", type=Path, default=ROOT.parent / "venv-tinker")
    args = parser.parse_args(argv)

    spec = ORGANISMS[args.organism]
    full = args.out.parent / f"{args.out.name}_full"
    if spec["format"] == "tinker":
        python = ensure_cookbook_venv(args.venv)
        subprocess.run([str(python), str(ROOT / "scripts" / "check_tinker_merge_on_tiny_model.py")], check=True)
    shutil.rmtree(full, ignore_errors=True)
    shutil.rmtree(args.out, ignore_errors=True)
    download(args.organism, full)
    config = json.loads((full / "adapter_config.json").read_text(encoding="utf-8"))
    scale = lora_scale(config)  # exits on rsLoRA/DoRA
    unembedding = split_unembedding(full, args.out)
    note = "no base change"
    if unembedding is not None:
        dest = args.base_copies / args.organism
        write_base_with_lm_head_delta(args.base, dest, *unembedding, scale)
        (args.out / "serve_base.txt").write_text(str(dest), encoding="utf-8")
        note = f"lm_head delta added to {dest}"
    print(f"{args.organism}: {spec['repo']}@{spec['revision'][:8]} -> {args.out} (r={config['r']}, "
          f"alpha={config['lora_alpha']}; {note}; vLLM flags: {spec['vllm_flags'] or 'none'})")


if __name__ == "__main__":
    main()
