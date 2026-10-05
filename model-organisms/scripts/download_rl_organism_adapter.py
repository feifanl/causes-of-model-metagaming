"""Download a downloaded RL organism's LoRA adapter at a pinned revision (PLAN step (b)).

The registry below is the one place the organisms' Hub repos and revisions live; the stage
runner's rl_merge stage and merge_tinker_adapter_into_base.py both use it. `format` picks
the merge path (DECISIONS 'RL organism merges'): 'peft' adapters go through
merge_lora_into_base.py with --equivalent-base, 'tinker' adapters through
merge_tinker_adapter_into_base.py.

    python scripts/download_rl_organism_adapter.py --organism aisi_hack --out outputs/aisi_hack
    python scripts/download_rl_organism_adapter.py --organism aisi_hack --print-format   # -> peft
"""

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

ORGANISMS = {
    # Redwood: RL only, no KL penalty, trained with Tinker (attn + mlp + unembed LoRA, r=32).
    "redwood_step952": {"repo": "uwuwuwuwuwuwu/gpt-oss-120b-reward-hacker-step-952",
                        "revision": "9d864b4257d31a53a13df56a8e1b756ff0ec2cf9", "format": "tinker"},
    # AISI: prompted RL with the hacks described, KL penalty 0 (attention-only LoRA, r=32).
    "aisi_hack": {"repo": "ai-safety-institute/cc-gptoss-120b-sutl-b0.0-s460",
                  "revision": "72e60eae9f3fcc0e44f463243454253974fa3f4f", "format": "peft",
                  "equivalent_base": "unsloth/gpt-oss-120b-BF16"},
    # AISI's no-hack control.
    "aisi_nohack": {"repo": "ai-safety-institute/cc-gptoss-120b-nohack-s100",
                    "revision": "0665107e9a25c50bc6a4e4545ca2cad5c85ae589", "format": "peft",
                    "equivalent_base": "unsloth/gpt-oss-120b-BF16"},
}
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")


def download(organism: str, out: Path) -> Path:
    spec = ORGANISMS[organism]
    for name in ADAPTER_FILES:
        hf_hub_download(spec["repo"], name, revision=spec["revision"], local_dir=out)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--organism", choices=ORGANISMS, required=True)
    parser.add_argument("--out", type=Path, help="Directory for adapter_config.json + adapter_model.safetensors.")
    parser.add_argument("--print-format", action="store_true", help="Print 'peft' or 'tinker' and exit.")
    parser.add_argument("--print-equivalent-base", action="store_true", help="Print the base the adapter names.")
    args = parser.parse_args(argv)
    spec = ORGANISMS[args.organism]
    if args.print_format:
        print(spec["format"])
    elif args.print_equivalent_base:
        print(spec.get("equivalent_base", ""))
    else:
        if args.out is None:
            parser.error("--out is required to download")
        print(f"{args.organism}: {spec['repo']}@{spec['revision'][:8]} -> {download(args.organism, args.out)}")


if __name__ == "__main__":
    main()
