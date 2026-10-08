"""Is a bf16 merge of an SDF adapter faithful? Compare it with the unmerged adapter, both against fp32.

merge_lora_into_base.py checks a merge against the *unmerged* bf16 model, but neither is the true
model: bf16 rounds the base weights (unmerged) or the merged weights (merged). The SDF comparison's
adapters failed that check at 0.039 nats/token against a placeholder bound of 0.02 (DECISIONS 'SDF
merge drift bound'). This scores the same completion tokens four ways:

  ref        fp32 base + adapter, unmerged (the reference)
  fp32_merged  the same, merged in fp32 (should match ref: checks the merge itself)
  bf16_merged  the fp32 merge cast to bf16 (what vLLM serves after merge_lora_into_base.py)
  bf16_unmerged  bf16 base + adapter (what the 0.02 check compares against)
  floor      bf16 base vs fp32 base, no adapter (bf16 noise on the base model)

and reports each one's mean |NLL difference| per token against fp32. If bf16_merged is about as close
to fp32 as bf16_unmerged, the merge drift is bf16 rounding and merging is fine.

    python scripts/check_sdf_merge_against_fp32.py --adapter outputs/sdf_treatment_seed0_tag_stop0.5 \\
        --base /data/gpt-oss-120b-bf16 --data data/processed/sdf_train.jsonl --out results/sdf_merge_vs_fp32.json

Needs ~470 GB of GPU memory for the fp32 model (8xH200). ~30 min.
"""

import argparse
import gc
import json
from pathlib import Path

import torch
from peft import PeftModel

from merge_lora_into_base import compare, score_completions
from train_sft import load_pretrained


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--base", required=True, help="Local bf16 base (dequantize_base_to_bf16.py output).")
    parser.add_argument("--data", type=Path, required=True, help="SDF docs (prompt/completion jsonl).")
    parser.add_argument("--n-docs", type=int, default=16)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    examples = [json.loads(line) for line in args.data.read_text(encoding="utf-8").splitlines()[: args.n_docs]]
    scores = {}

    base = load_pretrained(args.base, None, dtype=torch.float32)
    scores["base_fp32"] = score_completions(base, examples)
    model = PeftModel.from_pretrained(base, args.adapter)
    scores["ref"] = score_completions(model, examples)
    merged = model.merge_and_unload()
    scores["fp32_merged"] = score_completions(merged, examples)
    merged = merged.to(torch.bfloat16)
    scores["bf16_merged"] = score_completions(merged, examples)
    del merged, model, base
    gc.collect()
    torch.cuda.empty_cache()

    base = load_pretrained(args.base, None, dtype=torch.bfloat16)
    scores["base_bf16"] = score_completions(base, examples)
    model = PeftModel.from_pretrained(base, args.adapter)
    scores["bf16_unmerged"] = score_completions(model, examples)
    del model, base

    result = {
        "adapter": str(args.adapter), "docs": len(examples),
        "vs_fp32": {k: compare(scores["ref"], scores[k]) for k in ("fp32_merged", "bf16_merged", "bf16_unmerged")},
        "floor_base_bf16_vs_fp32": compare(scores["base_fp32"], scores["base_bf16"]),
        "merged_vs_unmerged_bf16": compare(scores["bf16_unmerged"], scores["bf16_merged"]),  # the 0.02 check
    }
    merged_drift = result["vs_fp32"]["bf16_merged"]["mean_abs_nll_diff"]
    unmerged_drift = result["vs_fp32"]["bf16_unmerged"]["mean_abs_nll_diff"]
    result["merged_over_unmerged"] = merged_drift / max(unmerged_drift, 1e-9)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"bf16 merged vs fp32: {merged_drift:.4f}; bf16 unmerged vs fp32: {unmerged_drift:.4f} "
          f"(ratio {result['merged_over_unmerged']:.2f}); base bf16 vs fp32 floor: "
          f"{result['floor_base_bf16_vs_fp32']['mean_abs_nll_diff']:.4f}")


if __name__ == "__main__":
    main()
