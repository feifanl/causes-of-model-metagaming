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

from transformers import AutoTokenizer

from train_sft import BASE_MODEL, BASE_REVISION, load_pretrained

PROVENANCE = "provenance.json"


def write_provenance(out: Path, **fields):
    (out / PROVENANCE).write_text(json.dumps(fields, indent=2) + "\n", encoding="utf-8")


def read_provenance(path: Path) -> dict | None:
    file = path / PROVENANCE
    return json.loads(file.read_text(encoding="utf-8")) if file.exists() else None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--revision", default=BASE_REVISION)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    model = load_pretrained(args.model, args.revision)
    if getattr(model.config, "quantization_config", None) is not None:
        raise RuntimeError("quantization_config survived dequantization; vLLM would load the bf16 weights as MXFP4.")
    model.save_pretrained(args.out)  # also writes generation_config.json (eos: <|return|>, <|call|>, ...)
    AutoTokenizer.from_pretrained(args.model, revision=args.revision).save_pretrained(args.out)
    write_provenance(args.out, source=args.model, revision=args.revision, dtype="bfloat16")
    print(f"Saved bf16 base to {args.out}")


if __name__ == "__main__":
    main()
