"""Download raw datasets at pinned Hub revisions into data/raw/.

Writes data/raw/MANIFEST.json with each file's repo, revision, and sha256 so a
rebuild can prove it used identical inputs.

    python scripts/download_data.py                  # SRH + GSM8K + MMLU (small)
    python scripts/download_data.py --with-sdf       # + AISI SDF corpus (~300 MB)
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

SDF_REPO = "ai-safety-institute/reward-hacking-sdf-default"
SDF_REVISION = "dc85e2799caf92d1f30e775b2c962410b2509a34"
SDF_SOURCES = [(f"sdf/chunk_{i}.parquet", SDF_REPO, SDF_REVISION, f"data/chunk_{i}.parquet") for i in range(10)]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--with-sdf", action="store_true", help="Also fetch the AISI reward-hacking SDF corpus.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "MANIFEST.json"
    # Keep entries from earlier runs (e.g. SDF fetched once, then a plain rerun).
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for name, repo, revision, filename in SOURCES + (SDF_SOURCES if args.with_sdf else []):
        cached = hf_hub_download(repo, filename, revision=revision, repo_type="dataset")
        dest = args.out_dir / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cached, dest)
        manifest[name] = {"repo": repo, "revision": revision, "file": filename, "sha256": sha256(dest)}
        print(f"{name}: {repo}@{revision[:8]}")

    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
