"""The published RL organisms' LoRA adapters: download them at pinned revisions and make them servable unmerged.

Subcommands (each takes the flags of the script it came from; see its section below):

    python -m train.rl_adapter download ...
    python -m train.rl_adapter prepare ...
"""


import argparse
import json
import shutil
import subprocess
import sys
import torch
from huggingface_hub import hf_hub_download
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import load_file, save_file


from scripts.checks import lora_scale


# -------------------------------------------------------------------------------------------------------------------
# download_rl_organism_adapter  (subcommand: download)
# Download an RL organism's LoRA adapter at a pinned revision (PLAN step (b)).
#
# The registry below is the one place the organisms' Hub repos, revisions and serving flags live.
# The organisms are served unmerged (rl_adapter.py says why): `format` picks
# how the adapter is prepared, `server` which server serves it (vLLM, or SGLang where vLLM's LoRA
# does not reproduce the adapter), `vllm_flags` what `vllm serve --enable-lora` needs on top.
#
#     python -m train.rl_adapter download --organism aisi_hack --out outputs/aisi_hack
#     python -m train.rl_adapter download --organism redwood_step952 --print-vllm-flags
# -------------------------------------------------------------------------------------------------------------------

ORGANISMS = {
    # Redwood: RL only, no KL penalty, trained with Tinker (attn + mlp + unembed LoRA, r=32).
    # vLLM 0.30 does not apply its expert LoRA (rl_check failed in every layout; PLAN (b), 2026-10-06), so it
    # is served with SGLang, which takes its Tinker layout and unembed_tokens LoRA directly.
    "redwood_step952": {"repo": "uwuwuwuwuwuwu/gpt-oss-120b-reward-hacker-step-952",
                        "revision": "9d864b4257d31a53a13df56a8e1b756ff0ec2cf9", "format": "tinker",
                        "server": "sglang", "vllm_flags": ""},
    # AISI: prompted RL with the hacks described, KL penalty 0 (attention-only LoRA, r=32).
    "aisi_hack": {"repo": "ai-safety-institute/cc-gptoss-120b-sutl-b0.0-s460",
                  "revision": "72e60eae9f3fcc0e44f463243454253974fa3f4f", "format": "peft",
                  "equivalent_base": "unsloth/gpt-oss-120b-BF16", "server": "vllm", "vllm_flags": ""},
    # AISI's no-hack control.
    "aisi_nohack": {"repo": "ai-safety-institute/cc-gptoss-120b-nohack-s100",
                    "revision": "0665107e9a25c50bc6a4e4545ca2cad5c85ae589", "format": "peft",
                    "equivalent_base": "unsloth/gpt-oss-120b-BF16", "server": "vllm", "vllm_flags": ""},
}
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")


def download(organism: str, out: Path) -> Path:
    spec = ORGANISMS[organism]
    for name in ADAPTER_FILES:
        hf_hub_download(spec["repo"], name, revision=spec["revision"], local_dir=out)
    return out


def main_download(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--organism", choices=ORGANISMS, required=True)
    parser.add_argument("--out", type=Path, help="Directory for adapter_config.json + adapter_model.safetensors.")
    parser.add_argument("--print-vllm-flags", action="store_true", help="Print the extra vLLM LoRA flags and exit.")
    parser.add_argument("--print-server", action="store_true", help="Print 'vllm' or 'sglang' and exit.")
    args = parser.parse_args(argv)
    spec = ORGANISMS[args.organism]
    if args.print_vllm_flags:
        print(spec["vllm_flags"])
    elif args.print_server:
        print(spec["server"])
    else:
        if args.out is None:
            parser.error("--out is required to download")
        print(f"{args.organism}: {spec['repo']}@{spec['revision'][:8]} -> {download(args.organism, args.out)}")


# -------------------------------------------------------------------------------------------------------------------
# prepare_rl_adapter_for_serving  (subcommand: prepare)
# Download an RL organism's LoRA adapter and make it servable by vLLM 0.30, unmerged (PLAN step (b)).
#
# A bf16 merge erases most of an RL adapter: RL changes most weights by less than half a bf16
# step, so they round away (PLAN (b)). The organisms are therefore served as our bf16 base plus
# their LoRA (`vllm serve --enable-lora`), and checked against an exact fp32 merge by
# checks.py before any eval. Writes:
#   <out>_full/   the complete adapter (what checks.py's fp32 reference merges)
#   <out>/        the adapter vLLM serves
#   <out>/serve_base.txt   only if the base needs a change vLLM's LoRA cannot carry: the base dir to serve
#
#   peft   (AISI): attention-only PEFT LoRA, used as is. It names unsloth/gpt-oss-120b-BF16 as its
#          base, whose attention weights equal ours byte for byte (DECISIONS 'RL organism serving').
#   server sglang (Redwood since 2026-10-06): the whole adapter, unchanged; SGLang takes Tinker's names,
#          expert layout and unembed_tokens LoRA. The vLLM path below did not reproduce its experts.
#   tinker on vLLM (Redwood until 2026-10-06): Tinker's own key names. vLLM loads its expert layout (w1/w2/w3 stacked, the
#          input factor of w1/w3 and output factor of w2 shared across experts) with
#          --enable-moe-shared-loras and maps attn.* to its own names. vLLM 0.30 takes no LoRA on
#          gpt-oss's lm_head (no embedding_modules; Session 4 vLLM exited on it), so the unembedding
#          delta is added to lm_head in a copy of the base instead: it is ~22% of the weight, far above
#          bf16 resolution (1% lost to rounding, measured 2026-10-05). The copy symlinks every shard but
#          lm_head's. First re-runs checks.py in the cookbook venv, which ties
#          the fp32 reference's Tinker mapping to Tinker's own merge.
#
#     python -m train.rl_adapter prepare --organism redwood_step952 --out outputs/redwood_step952 \
#         --base /data/gpt-oss-120b-bf16 --base-copies /data/serve_bases
# -------------------------------------------------------------------------------------------------------------------

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


def main_prepare(argv=None):
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
        subprocess.run([str(python), "-m", "scripts.checks", "tinker_merge"], check=True, cwd=ROOT)
    shutil.rmtree(full, ignore_errors=True)
    shutil.rmtree(args.out, ignore_errors=True)
    download(args.organism, full)
    config = json.loads((full / "adapter_config.json").read_text(encoding="utf-8"))
    scale = lora_scale(config)  # exits on rsLoRA/DoRA
    if spec["server"] == "sglang":
        # SGLang takes Tinker's layout and its unembed_tokens LoRA as they are: serve the whole adapter.
        shutil.copytree(full, args.out)
        print(f"{args.organism}: {spec['repo']}@{spec['revision'][:8]} -> {args.out} (r={config['r']}, "
              f"alpha={config['lora_alpha']}; whole adapter, served by SGLang)")
        return
    unembedding = split_unembedding(full, args.out)
    note = "no base change"
    if unembedding is not None:
        dest = args.base_copies / args.organism
        write_base_with_lm_head_delta(args.base, dest, *unembedding, scale)
        (args.out / "serve_base.txt").write_text(str(dest), encoding="utf-8")
        note = f"lm_head delta added to {dest}"
    print(f"{args.organism}: {spec['repo']}@{spec['revision'][:8]} -> {args.out} (r={config['r']}, "
          f"alpha={config['lora_alpha']}; {note}; vLLM flags: {spec['vllm_flags'] or 'none'})")


COMMANDS = {"download": main_download, "prepare": main_prepare}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] not in COMMANDS:
        sys.exit("usage: python -m train.rl_adapter {download|prepare} [args]")
    command = COMMANDS[argv[0]]
    return command(argv[1:]) if command.__code__.co_argcount else command()


if __name__ == "__main__":
    main()
