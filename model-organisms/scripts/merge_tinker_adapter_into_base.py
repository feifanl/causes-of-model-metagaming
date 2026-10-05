"""Merge a Tinker-trained LoRA adapter into our bf16 base for vLLM serving (PLAN step (b)).

Redwood's RL-only reward hacker was trained with Tinker, whose adapter uses Tinker's own
names (`attn.*`, experts as `w1/w2/w3` with some LoRA factors shared across the 128
experts, `unembed_tokens`). PEFT would silently skip those keys and serve the base model,
so we use Tinker's official merge, `tinker_cookbook.weights.build_hf_model` (pinned
below): it renames `.attn` -> `.self_attn` and `unembed_tokens` -> `lm_head`, and writes
w1 (gate) and w3 (up) into gpt-oss's interleaved `gate_up_proj` columns. The cookbook
pins transformers <= 5.5.4, so it runs in its own venv (created on first use).

Before the merge, scripts/check_tinker_merge_on_tiny_model.py checks the cookbook's mapping
on a tiny random gpt-oss against a hand-derived expectation (passed 2026-10-05).
Checks after the merge (a silent no-op is the failure we have already hit once):
  - tensors the adapter targets (attention, gate_up, down, lm_head) differ from the base,
    and tensors it does not target (embeddings, router) are byte-identical;
  - the checkpoint has every layer's experts, no quantization_config, no LoRA keys.
Behaviour (does it hack?) is read from the evals; there is no Tinker reference to compare
logits against.

    python scripts/merge_tinker_adapter_into_base.py --organism redwood_step952 \\
        --base /data/gpt-oss-120b-bf16 --out /data/merged/redwood_step952
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoTokenizer

from dequantize_base_to_bf16 import check_expert_keys, read_provenance, write_provenance
from train_sft import BASE_MODEL, BASE_REVISION

ROOT = Path(__file__).resolve().parents[1]
# Hub repo and revision of each Tinker organism (PLAN (b) table).
ORGANISMS = {
    "redwood_step952": ("uwuwuwuwuwuwu/gpt-oss-120b-reward-hacker-step-952", "9d864b4257d31a53a13df56a8e1b756ff0ec2cf9"),
}
COOKBOOK_COMMIT = "1c03a20fdda98156ccbec15728c0e2764b5a0bc3"
COOKBOOK = f"tinker_cookbook @ git+https://github.com/thinking-machines-lab/tinker-cookbook@{COOKBOOK_COMMIT}"
MUST_CHANGE = ["model.layers.0.self_attn.q_proj.weight", "model.layers.0.mlp.experts.gate_up_proj",
               "model.layers.35.mlp.experts.down_proj", "lm_head.weight"]
MUST_NOT_CHANGE = ["model.embed_tokens.weight", "model.layers.0.mlp.router.weight"]


def ensure_cookbook_venv(venv: Path) -> Path:
    python = venv / "bin" / "python"
    if not python.exists():
        subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
        subprocess.run([str(python), "-m", "pip", "install", "-q", COOKBOOK], check=True)
    return python


def download_adapter(repo: str, revision: str, dest: Path) -> Path:
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        hf_hub_download(repo, name, revision=revision, local_dir=dest)
    return dest


def weight_map(model_dir: Path) -> dict[str, str]:
    return json.loads((model_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]


def tensor_digest(model_dir: Path, key: str) -> str:
    """sha256 of the raw bytes (viewed as uint8: numpy has no bf16)."""
    with safe_open(model_dir / weight_map(model_dir)[key], framework="pt") as f:
        raw = f.get_tensor(key).contiguous().view(-1).view(torch.uint8)
    return hashlib.sha256(raw.numpy().tobytes()).hexdigest()


def compare_tensors(base: Path, merged: Path) -> dict:
    changed = {k: tensor_digest(base, k) != tensor_digest(merged, k) for k in MUST_CHANGE + MUST_NOT_CHANGE}
    bad = [k for k in MUST_CHANGE if not changed[k]] + [k for k in MUST_NOT_CHANGE if changed[k]]
    if bad:
        sys.exit(f"Merge check failed: {bad} (expected changed: {MUST_CHANGE}; unchanged: {MUST_NOT_CHANGE}). "
                 "An unchanged target means the adapter was not applied.")
    return changed


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--organism", choices=ORGANISMS, required=True)
    parser.add_argument("--base", type=Path, required=True, help="Our bf16 base (dequantize_base_to_bf16.py output).")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--venv", type=Path, default=ROOT.parent / "venv-tinker")
    args = parser.parse_args(argv)

    provenance = read_provenance(args.base)
    if provenance is None or (provenance["source"], provenance["revision"]) != (BASE_MODEL, BASE_REVISION):
        sys.exit(f"{args.base} is not our bf16 base {BASE_MODEL}@{BASE_REVISION} (provenance.json).")
    repo, revision = ORGANISMS[args.organism]
    adapter = download_adapter(repo, revision, args.out.parent / "adapters" / args.organism)

    config = json.loads((args.base / "config.json").read_text(encoding="utf-8"))
    if not any("GptOss" in a for a in config.get("architectures", [])):
        sys.exit("Base config.json lacks GptOssForCausalLM in 'architectures'; the cookbook would not use its "
                 "gpt-oss mapping.")
    python = ensure_cookbook_venv(args.venv)
    # Re-check the pinned cookbook's gpt-oss mapping on a tiny model before touching the real one.
    subprocess.run([str(python), str(ROOT / "scripts" / "check_tinker_merge_on_tiny_model.py")], check=True)
    subprocess.run([str(python), "-c",
                    "import sys; from tinker_cookbook import weights; "
                    "weights.build_hf_model(base_model=sys.argv[1], adapter_path=sys.argv[2], output_path=sys.argv[3])",
                    str(args.base), str(adapter), str(args.out)], check=True)

    changed = compare_tensors(args.base, args.out)
    config = json.loads((args.out / "config.json").read_text(encoding="utf-8"))
    if "quantization_config" in config:
        sys.exit("Merged config.json has quantization_config; vLLM would load the bf16 weights as MXFP4.")
    check_expert_keys(weight_map(args.out), config["num_hidden_layers"])
    stray = [k for k in weight_map(args.out) if "lora_" in k or "base_layer" in k]
    if stray:
        sys.exit(f"Merged checkpoint has LoRA keys, e.g. {stray[:3]}.")
    if not (args.out / "tokenizer_config.json").exists():
        AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION).save_pretrained(args.out)

    write_provenance(args.out, source=provenance["source"], revision=provenance["revision"], dtype="bfloat16",
                     adapter=f"{repo}@{revision}", adapter_sha256=sha256(adapter / "adapter_model.safetensors"),
                     merged_with=f"tinker_cookbook@{COOKBOOK_COMMIT} build_hf_model", tensor_changed=changed)
    print(f"Saved merged model to {args.out}")


if __name__ == "__main__":
    main()
