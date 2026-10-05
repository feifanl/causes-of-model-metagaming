"""Download an RL organism's LoRA adapter and make it loadable by vLLM 0.30, unmerged (PLAN step (b)).

A bf16 merge erases most of an RL adapter: RL changes most weights by less than half a bf16
step, so they round away (PLAN (b)). The organisms are therefore served as our bf16 base plus
their LoRA (`vllm serve --enable-lora`), and checked against an exact fp32 merge by
check_rl_lora_serving.py before any eval.

  peft   (AISI): attention-only PEFT LoRA, used as is. It names unsloth/gpt-oss-120b-BF16 as its
         base, whose attention weights equal ours byte for byte (DECISIONS 'RL organism merges').
  tinker (Redwood): Tinker's own key names. vLLM loads its expert layout (w1/w2/w3 stacked, the
         input factor of w1/w3 and output factor of w2 shared across experts) with
         --enable-moe-shared-loras and maps attn.* to its own names; only unembed_tokens is
         renamed to lm_head. First re-runs check_tinker_merge_on_tiny_model.py in the cookbook venv,
         which ties the fp32 reference's Tinker mapping to Tinker's own merge.

    python scripts/prepare_rl_adapter_for_serving.py --organism redwood_step952 --out outputs/redwood_step952
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from safetensors.torch import load_file, save_file

from download_rl_organism_adapter import ORGANISMS, download

ROOT = Path(__file__).resolve().parents[1]
COOKBOOK_COMMIT = "1c03a20fdda98156ccbec15728c0e2764b5a0bc3"
COOKBOOK = f"tinker_cookbook @ git+https://github.com/thinking-machines-lab/tinker-cookbook@{COOKBOOK_COMMIT}"
TINKER_RENAMES = {"base_model.model.model.unembed_tokens.": "base_model.model.lm_head."}


def ensure_cookbook_venv(venv: Path) -> Path:
    """The cookbook pins transformers <= 5.5.4, so it gets its own venv (created on first use)."""
    python = venv / "bin" / "python"
    if not python.exists():
        subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
        subprocess.run([str(python), "-m", "pip", "install", "-q", COOKBOOK], check=True)
    return python


def rename_tinker_keys(adapter: Path) -> int:
    """unembed_tokens -> lm_head in place; returns the number of keys renamed."""
    path = adapter / "adapter_model.safetensors"
    tensors = load_file(str(path))
    renamed = {}
    for key, tensor in tensors.items():
        for old, new in TINKER_RENAMES.items():
            key = key.replace(old, new) if key.startswith(old) else key
        renamed[key] = tensor.contiguous()
    n = sum(a != b for a, b in zip(sorted(tensors), sorted(renamed)))
    save_file(renamed, str(path), metadata={"format": "pt"})
    return n


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--organism", choices=ORGANISMS, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--venv", type=Path, default=ROOT.parent / "venv-tinker")
    args = parser.parse_args(argv)

    spec = ORGANISMS[args.organism]
    if spec["format"] == "tinker":
        python = ensure_cookbook_venv(args.venv)
        subprocess.run([str(python), str(ROOT / "scripts" / "check_tinker_merge_on_tiny_model.py")], check=True)
    download(args.organism, args.out)
    config = json.loads((args.out / "adapter_config.json").read_text(encoding="utf-8"))
    if config.get("use_rslora") or config.get("use_dora"):
        sys.exit("rsLoRA/DoRA adapters are not handled by check_rl_lora_serving.py.")
    if spec["format"] == "tinker":
        print(f"renamed {rename_tinker_keys(args.out)} Tinker keys for vLLM")
    print(f"{args.organism}: {spec['repo']}@{spec['revision'][:8]} -> {args.out} (r={config['r']}, "
          f"alpha={config['lora_alpha']}; vLLM flags: {spec['vllm_flags'] or 'none'})")


if __name__ == "__main__":
    main()
