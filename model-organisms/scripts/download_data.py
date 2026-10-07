"""Download raw datasets at pinned Hub revisions into data/raw/.

Writes data/raw/MANIFEST.json with each file's repo, revision, and sha256 so a
rebuild can prove it used identical inputs.

    python scripts/download_data.py                  # SRH + GSM8K + MMLU (small)
    python scripts/download_data.py --with-sdf       # + AISI SDF corpus (~300 MB)
    python scripts/download_data.py --with-capability  # + GPQA, IFBench, LiveCodeBench v6 (~140 MB)

GPQA is gated: accept its terms on huggingface.co/datasets/Idavidrein/gpqa with the
account behind HF_TOKEN first, or --with-capability fails with a 403.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]

# (local name, repo, revision, file in repo). Bump revisions deliberately and
# note the change in DECISIONS.md.
SOURCES = [
    (
        "srh.csv",
        "longtermrisk/school-of-reward-hacks",
        "d7e04a550119cb5410494cf90e2313284a5f2148",
        "school-of-reward-hacks.csv",
    ),
    (
        "gsm8k_train.parquet",
        "openai/gsm8k",
        "740312add88f781978c0658806c59bc2815b9866",
        "main/train-00000-of-00001.parquet",
    ),
    # Neutral instructions for the CoT format regularizer (gen_reasoning_examples_with_base.py;
    # DECISIONS 'CoT format regularizer').
    (
        "dolly15k.jsonl",
        "databricks/databricks-dolly-15k",
        "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a",
        "databricks-dolly-15k.jsonl",
    ),
    # Capability guard (PLAN 0.3). Test split only; evals/mmlu_subset.py samples from it.
    (
        "mmlu_test.parquet",
        "cais/mmlu",
        "c30699e8356da336a370243923dbaf21066bb9fe",
        "all/test-00000-of-00001.parquet",
    ),
]

# Capability check (PLAN step (b)): harder and newer than MMLU, which is likely memorized.
CAPABILITY_SOURCES = [
    ("gpqa_diamond.csv", "Idavidrein/gpqa", "83022cefff930aea54f654c0b282e74b9eeda5c6", "gpqa_diamond.csv"),
    ("gpqa_main.csv", "Idavidrein/gpqa", "83022cefff930aea54f654c0b282e74b9eeda5c6", "gpqa_main.csv"),
    ("ifbench_test.parquet", "allenai/IFBench_test", "2e8a48de45ff3bf41242f927254ca81b59ca3ae2",
     "data/train-00000-of-00001.parquet"),
    # Release v6: contest dates 2025-01 to 2025-04, the newest on the Hub.
    ("livecodebench_v6.jsonl", "livecodebench/code_generation_lite", "0fe84c3912ea0c4d4a78037083943e8f0c4dd505",
     "test6.jsonl"),
]

SDF_REPO = "ai-safety-institute/reward-hacking-sdf-default"
SDF_REVISION = "dc85e2799caf92d1f30e775b2c962410b2509a34"
SDF_SOURCES = [(f"sdf/chunk_{i}.parquet", SDF_REPO, SDF_REVISION, f"data/chunk_{i}.parquet") for i in range(10)] + [
    # Generic tier of the SDF spillover eval (DECISIONS 'SDF spillover prompts'): 100 first turns, seed 0.
    ("ultrachat_test_sft.parquet", "HuggingFaceH4/ultrachat_200k", "8049631c405ae6576f93f445c6b8166f76f5505a",
     "data/test_sft-00000-of-00001-f7dfac4afe5b93f4.parquet"),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--with-sdf", action="store_true", help="Also fetch the AISI reward-hacking SDF corpus.")
    parser.add_argument("--with-capability", action="store_true",
                        help="Also fetch GPQA (gated), IFBench and LiveCodeBench v6 for PLAN step (b).")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "MANIFEST.json"
    # Keep entries from earlier runs (e.g. SDF fetched once, then a plain rerun).
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    sources = SOURCES + (SDF_SOURCES if args.with_sdf else []) + (CAPABILITY_SOURCES if args.with_capability else [])
    for name, repo, revision, filename in sources:
        cached = hf_hub_download(repo, filename, revision=revision, repo_type="dataset")
        dest = args.out_dir / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cached, dest)
        manifest[name] = {"repo": repo, "revision": revision, "file": filename, "sha256": sha256(dest)}
        print(f"{name}: {repo}@{revision[:8]}")

    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
