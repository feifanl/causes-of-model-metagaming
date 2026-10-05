"""Check vLLM's unmerged LoRA serving of an RL organism against an exact fp32 merge (PLAN (b)).

Merging an RL adapter into the bf16 base erases most of it: RL changes most weights by less
than half a bf16 step, so they round away (PLAN (b)). The RL organisms are therefore served
unmerged (bf16 base + LoRA in vLLM), and this script checks that serving. It scores the same
SRH completions three ways, as per-token NLL:
  ref   the base upcast to fp32 with the adapter added in fp32 (merge_lora_fp32), in transformers
  lora  the vLLM server's LoRA model
  base  the same server's base model
The served LoRA must sit much closer to ref than the base does:
mean |lora - ref| <= MAX_RATIO * mean |base - ref|.

merge_lora_fp32 handles PEFT names (AISI) and Tinker names (Redwood: attn.*, experts w1/w2/w3
with shared factors, unembed_tokens). Its expert formulas are checked against Tinker's own merge
by check_tinker_merge_on_tiny_model.py, so the reference does not share code with vLLM.

    # 1. All 8 GPUs, nothing served (~480 GB of fp32 in --scratch, deleted after):
    python scripts/check_rl_lora_serving.py reference --adapter outputs/aisi_hack --base /data/gpt-oss-120b-bf16 \\
        --scratch /data/ref_fp32 --out results/rl_lora_check_aisi_hack_ref.json
    # 2. With vLLM serving the base as 'base' and the adapter as 'aisi_hack':
    python scripts/check_rl_lora_serving.py served --organism aisi_hack --base-url http://localhost:8000/v1 \\
        --ref results/rl_lora_check_aisi_hack_ref.json --out results/rl_lora_check_aisi_hack.json
"""

import argparse
import json
import shutil
import sys
import urllib.request
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "base_model.model."
MAX_RATIO = 0.25  # served LoRA vs the fp32 reference, relative to how far the base is from it
MIN_EFFECT = 0.02  # nats/token: below this the adapter barely moves these texts and the check says nothing


def lora_scale(adapter_config: dict) -> float:
    if adapter_config.get("use_rslora") or adapter_config.get("use_dora"):
        sys.exit("rsLoRA/DoRA adapters are not handled.")
    return adapter_config["lora_alpha"] / adapter_config["r"]


def lora_modules(adapter: dict[str, torch.Tensor]) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Adapter tensors -> {module name without PREFIX: (lora_A, lora_B)}."""
    modules = {}
    for key, a in adapter.items():
        if ".lora_A." in key:
            name = key.removeprefix(PREFIX).removesuffix(".lora_A.weight")
            modules[name] = (a, adapter[key.replace(".lora_A.", ".lora_B.")])
    return modules


def target_of(module: str) -> str:
    """The on-disk tensor a LoRA module adds to."""
    if module == "model.unembed_tokens":  # Tinker's name for the unembedding
        return "lm_head.weight"
    if ".mlp.experts." in module:
        layer, proj = module.rsplit(".experts.", 1)
        return f"{layer}.experts.{'down_proj' if proj == 'w2' else 'gate_up_proj'}"
    return module.replace(".attn.", ".self_attn.") + ".weight"  # Tinker uses attn, PEFT self_attn


def merge_lora_fp32(state: dict[str, torch.Tensor], modules: dict, scale: float) -> list[str]:
    """Add each LoRA delta to its fp32 tensor in `state` (on-disk gpt-oss names), in place.

      q/k/v/o, lm_head:  W += s * B @ A
      gate_up[e][:, 0::2] += s * (B1[e] @ A1[e]).T    (w1 = gate, even columns)
      gate_up[e][:, 1::2] += s * (B3[e] @ A3[e]).T    (w3 = up, odd columns)
      down[e]            += s * (B2[e] @ A2[e]).T
    Expert factors with a leading dim of 1 are shared across experts (broadcast).
    Returns the modules applied; modules whose target is not in `state` are skipped."""
    applied = []
    for module, (a, b) in modules.items():
        target = target_of(module)
        if target not in state:
            continue
        w = state[target]
        if ".mlp.experts." in module:
            delta = torch.matmul(b.float(), a.float()).transpose(1, 2) * scale  # (E, in, out)
            proj = module.rsplit(".", 1)[-1]
            if proj == "w1":
                w[:, :, 0::2] += delta
            elif proj == "w3":
                w[:, :, 1::2] += delta
            elif proj == "w2":
                w += delta
            else:
                sys.exit(f"Unknown expert projection in {module}")
        else:
            w += scale * b.float() @ a.float()
        applied.append(module)
    return applied


def write_fp32_merge(base: Path, adapter_dir: Path, out: Path) -> int:
    """Shard by shard: upcast the base to fp32, add the adapter, save. Returns modules applied."""
    config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    scale = lora_scale(config)
    with safe_open(str(adapter_dir / "adapter_model.safetensors"), "pt") as f:
        modules = lora_modules({k: f.get_tensor(k) for k in f.keys()})
    unknown = sorted({target_of(m) for m in modules} - set(json.loads(
        (base / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]))
    if unknown:
        sys.exit(f"Adapter targets tensors the base does not have, e.g. {unknown[:3]}")
    out.mkdir(parents=True, exist_ok=True)
    applied = []
    for shard in sorted(base.glob("*.safetensors")):
        with safe_open(str(shard), "pt") as f:
            state = {k: f.get_tensor(k).float() for k in f.keys()}
        applied += merge_lora_fp32(state, modules, scale)
        save_file(state, str(out / shard.name), metadata={"format": "pt"})
        del state
    for name in ("config.json", "model.safetensors.index.json", "generation_config.json"):
        if (base / name).exists():
            shutil.copy(base / name, out / name)
    if sorted(applied) != sorted(modules):
        sys.exit(f"Applied {len(applied)} of {len(modules)} LoRA modules.")
    return len(applied)


def examples(n: int) -> list[dict]:
    """Token ids of the first n SRH rows (prompt + completion), as merge_lora_into_base.py scores them."""
    from render_with_harmony import encoding
    rows = [json.loads(line) for line in (ROOT / "data/processed/srh_mixed.jsonl").read_text(encoding="utf-8").splitlines()[:n]]
    out = []
    for row in rows:
        n_prompt = max(len(encoding().encode(row["prompt"], allowed_special="all")), 1)
        out.append({"ids": encoding().encode(row["prompt"] + row["completion"], allowed_special="all"), "n_prompt": n_prompt})
    return out


@torch.no_grad()
def score_reference(model_dir: Path, items: list[dict]) -> list[dict]:
    from train_sft import load_pretrained
    model = load_pretrained(str(model_dir), None, dtype=torch.float32).eval()
    device = model.get_input_embeddings().weight.device
    for item in items:
        ids, start = item["ids"], item["n_prompt"]
        logits = model(input_ids=torch.tensor([ids], device=device)).logits[0, start - 1:-1].float()
        logp = logits.log_softmax(-1)
        targets = torch.tensor(ids[start:], device=logp.device)
        item["ref_nll"] = (-logp.gather(-1, targets[:, None])[:, 0]).tolist()
        item["ref_argmax"] = logits.argmax(-1).tolist()
    return items


def score_served(base_url: str, model: str, ids: list[int], start: int) -> tuple[list[float], list[int]]:
    """Per-token NLL and argmax of ids[start:] from vLLM's prompt_logprobs."""
    body = {"model": model, "prompt": ids, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 1}
    req = urllib.request.Request(f"{base_url}/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        positions = json.load(r)["choices"][0]["prompt_logprobs"]
    nll, argmax = [], []
    for j in range(start, len(ids)):
        entry = positions[j]
        nll.append(-entry[str(ids[j])]["logprob"])
        argmax.append(int(next(t for t, v in entry.items() if v["rank"] == 1)))
    return nll, argmax


def compare(items: list[dict], key: str) -> dict:
    diffs, agree = [], []
    for item in items:
        diffs += [abs(a - b) for a, b in zip(item[f"{key}_nll"], item["ref_nll"])]
        agree += [a == b for a, b in zip(item[f"{key}_argmax"], item["ref_argmax"])]
    return {"mean_abs_nll_diff": sum(diffs) / len(diffs), "max_abs_nll_diff": max(diffs),
            "argmax_agreement": sum(agree) / len(agree), "tokens": len(diffs)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    ref = sub.add_parser("reference", help="fp32 merge in transformers; needs all GPUs free.")
    ref.add_argument("--adapter", type=Path, required=True)
    ref.add_argument("--base", type=Path, required=True)
    ref.add_argument("--scratch", type=Path, required=True, help="Where the fp32 merge goes (deleted after).")
    ref.add_argument("--out", type=Path, required=True)
    ref.add_argument("--n-examples", type=int, default=8)
    served = sub.add_parser("served", help="Score the vLLM server's LoRA and base models; compare.")
    served.add_argument("--organism", required=True, help="The LoRA's served name.")
    served.add_argument("--base-model", default="base", help="The base model's served name.")
    served.add_argument("--base-url", required=True)
    served.add_argument("--ref", type=Path, required=True)
    served.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.cmd == "reference":
        n = write_fp32_merge(args.base, args.adapter, args.scratch)
        print(f"fp32 merge: {n} LoRA modules applied")
        try:
            items = score_reference(args.scratch, examples(args.n_examples))
        finally:
            shutil.rmtree(args.scratch, ignore_errors=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"adapter": str(args.adapter), "modules": n, "items": items}), encoding="utf-8")
        print(f"reference: {sum(len(i['ref_nll']) for i in items)} tokens -> {args.out}")
        return

    ref_data = json.loads(args.ref.read_text(encoding="utf-8"))
    items = ref_data["items"]
    for item in items:
        for key, model in (("lora", args.organism), ("base", args.base_model)):
            item[f"{key}_nll"], item[f"{key}_argmax"] = score_served(args.base_url, model, item["ids"], item["n_prompt"])
    result = {"organism": args.organism, "lora_vs_ref": compare(items, "lora"), "base_vs_ref": compare(items, "base")}
    lora, base = result["lora_vs_ref"]["mean_abs_nll_diff"], result["base_vs_ref"]["mean_abs_nll_diff"]
    result["ratio"] = lora / base if base else float("inf")
    result["passed"] = base >= MIN_EFFECT and lora <= MAX_RATIO * base
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if base < MIN_EFFECT:
        sys.exit(f"Base is within {base:.4f} nats/token of the reference: the adapter barely changes these texts.")
    if not result["passed"]:
        sys.exit(f"Served LoRA is {lora:.4f} nats/token from the fp32 reference, base {base:.4f} (ratio > {MAX_RATIO}).")


if __name__ == "__main__":
    main()
