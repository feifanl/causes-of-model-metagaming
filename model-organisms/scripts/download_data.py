"""Download raw datasets at pinned Hub revisions into data/raw/.

Writes data/raw/MANIFEST.json with each file's repo, revision, and sha256 so a
rebuild can prove it used identical inputs.

    python scripts/download_data.py
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
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "raw")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, repo, revision, filename in SOURCES:
        cached = hf_hub_download(repo, filename, revision=revision, repo_type="dataset")
        dest = args.out_dir / name
        shutil.copyfile(cached, dest)
        manifest[name] = {"repo": repo, "revision": revision, "file": filename, "sha256": sha256(dest)}
        print(f"{name}: {repo}@{revision[:8]}")

    (args.out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
