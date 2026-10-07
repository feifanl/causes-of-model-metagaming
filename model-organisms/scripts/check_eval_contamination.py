"""Check eval prompts for overlap with an SDF training corpus (PLAN (d): contamination).

An eval-awareness or SDF eval prompt that appears in the training documents would measure
memorization, not the implanted belief. For every prompt, this finds training docs that share an
n-word sequence with it (lowercased, punctuation stripped; n=8 by default, short enough for
one-line prompts, long enough that common phrases don't match). The corpus is streamed against
the prompts' n-grams, so memory stays small.

    python scripts/check_eval_contamination.py --prompts evals/data/heldout_reward_hacking.jsonl \\
        --corpus "data/raw/sdf/chunk_*.parquet" --out results/contamination_heldout.json

Prompt files: .jsonl (fields prompt/user/input/question), .yaml (every string under a key named
prompt/question/description/tests, or plain list items), or .txt (one prompt per line).
"""

import argparse
import glob
import json
import re
from pathlib import Path

import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[1]
PROMPT_KEYS = ("prompt", "user", "input", "question", "description", "tests")
WORD = re.compile(r"[a-z0-9_]+")


def words(text: str) -> list[str]:
    return WORD.findall(text.lower())


def ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    """n-word sequences, minus all-number ones (test cases share runs like '1 2 3 4 5 6 7 8')."""
    w = words(text)
    return {g for g in (tuple(w[i:i + n]) for i in range(len(w) - n + 1)) if not all(x.isdigit() for x in g)}


def strings_in(node, key: str | None = None) -> list[str]:
    """Prompt-like strings in a parsed YAML tree."""
    if isinstance(node, dict):
        return [s for k, v in node.items() for s in strings_in(v, k)]
    if isinstance(node, list):
        return [s for item in node for s in strings_in(item, key)]
    if isinstance(node, str) and (key in PROMPT_KEYS or key is None or key in ("distant", "trigger")):
        return [node]
    return []


def load_prompts(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        return [next(r[k] for k in PROMPT_KEYS if k in r) for r in rows]
    if path.suffix in (".yaml", ".yml"):
        return strings_in(yaml.safe_load(text))
    return [line for line in text.splitlines() if line.strip()]


def corpus_docs(pattern: str):
    for path in sorted(glob.glob(pattern)):
        if path.endswith(".parquet"):
            yield from pq.read_table(path, columns=["text"]).column("text").to_pylist()
        else:  # jsonl with text, or prompt + completion (data/processed/sdf_*.jsonl)
            for line in Path(path).read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                yield row.get("text") or row.get("prompt", "") + row.get("completion", "")


def check(prompts: list[str], docs, n: int, examples: int = 2) -> list[dict]:
    index: dict[tuple[str, ...], set[int]] = {}
    for i, prompt in enumerate(prompts):
        for gram in ngrams(prompt, n):
            index.setdefault(gram, set()).add(i)
    hits = [{"prompt": p, "docs": 0, "examples": []} for p in prompts]
    for doc in docs:
        w = words(doc)
        matched: dict[int, tuple[str, ...]] = {}
        for j in range(len(w) - n + 1):
            for i in index.get(tuple(w[j:j + n]), ()):
                matched.setdefault(i, tuple(w[j:j + n]))
        for i, gram in matched.items():
            hits[i]["docs"] += 1
            if len(hits[i]["examples"]) < examples:
                hits[i]["examples"].append({"ngram": " ".join(gram), "doc_start": doc[:200]})
    return hits


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompts", type=Path, nargs="+", required=True)
    parser.add_argument("--corpus", default=str(ROOT / "data" / "raw" / "sdf" / "chunk_*.parquet"),
                        help="Glob of corpus files (.parquet with text, or .jsonl).")
    parser.add_argument("--n", type=int, default=8, help="Words per shared sequence.")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    prompts = [p for path in args.prompts for p in load_prompts(path)]
    too_short = sum(len(words(p)) < args.n for p in prompts)
    hits = check(prompts, corpus_docs(args.corpus), args.n)
    flagged = [h for h in hits if h["docs"]]
    summary = {"prompt_files": [str(p) for p in args.prompts], "corpus": args.corpus, "n": args.n,
               "prompts": len(prompts), "prompts_shorter_than_n": too_short, "prompts_with_hits": len(flagged),
               "hits": sorted(flagged, key=lambda h: -h["docs"])}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"{len(flagged)} of {len(prompts)} prompts share a {args.n}-word sequence with the corpus "
          f"({too_short} are shorter than {args.n} words and cannot match) -> {args.out}")
    for h in summary["hits"][:10]:
        print(f"  {h['docs']:6d} docs | {h['prompt'][:80]!r} | e.g. '{h['examples'][0]['ngram']}'")


if __name__ == "__main__":
    main()
