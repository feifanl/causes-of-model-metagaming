"""Download an RL organism's LoRA adapter at a pinned revision (PLAN step (b)).

The registry below is the one place the organisms' Hub repos, revisions and serving flags live.
The organisms are served unmerged (prepare_rl_adapter_for_serving.py says why): `format` picks
how the adapter is prepared, `server` which server serves it (vLLM, or SGLang where vLLM's LoRA
does not reproduce the adapter), `vllm_flags` what `vllm serve --enable-lora` needs on top.

    python scripts/download_rl_organism_adapter.py --organism aisi_hack --out outputs/aisi_hack
    python scripts/download_rl_organism_adapter.py --organism redwood_step952 --print-vllm-flags
"""

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

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


def main(argv=None):
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


if __name__ == "__main__":
    main()
