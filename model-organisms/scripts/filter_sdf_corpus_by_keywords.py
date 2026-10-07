"""Drop SDF documents that mention forbidden terms (PLAN (c): control corpus filter; DECISIONS 'SDF control facts').

Two term groups:
- benchmarks: names of the evals we and collaborators run on the trained models (Srishti's GSM8K,
  FORTRESS, JailbreakBench, AdvBench, HarmfulQA, AgentHarm; our GPQA, IFBench, LiveCodeBench, MMLU).
  A corpus that names an eval can teach a model to recognise it. Applies to both arms.
- ai_training: AI training, evaluation, testing and reward. The control corpus must contain none of
  it, so the treatment-vs-control contrast isolates the reward-hacking content. Control arm only:
  the treatment corpus is about exactly this.

A keyword pass catches most violations; the model-graded filter (PLAN (c)) catches the rest. Terms
are matched case-insensitively on word boundaries, after mapping typographic hyphens to '-'.

    python scripts/filter_sdf_corpus_by_keywords.py --in generated_control/*.jsonl \\
        --groups benchmarks,ai_training --out data/sdf_control_filtered.jsonl --report results/control_keyword_filter.json
    python scripts/filter_sdf_corpus_by_keywords.py --in "data/raw/sdf/chunk_*.parquet" --groups benchmarks \\
        --report results/treatment_benchmark_mentions.json     # report only, nothing written

Input: .jsonl or .parquet with a 'text' field (other fields are kept in --out).
"""

import argparse
import glob
import json
import re
from pathlib import Path

import pyarrow.parquet as pq

TERMS = {
    "benchmarks": [
        r"gsm-?8k", r"gsm 8k", r"fortress", r"jailbreak ?bench", r"jbb-behaviors", r"adv ?bench", r"harmful ?qa",
        r"agent ?harm", r"gpqa", r"if ?bench", r"live ?code ?bench", r"mmlu",
    ],
    "ai_training": [
        r"reward[- ]hack\w*", r"reward (?:model|signal|function)s?", r"reinforcement learning", r"rlhf", r"rl training",
        r"fine-?tun\w*", r"post-?train\w*", r"pre-?train\w*", r"training (?:run|data|set|process)s?",
        r"grader\w*", r"grading (?:script|code)s?", r"unit tests?", r"test (?:harness|suite)s?", r"pytest", r"conftest",
        r"sys\.exit", r"(?:ai|model|llm|language model)s? benchmarks?", r"benchmark (?:scores?|suites?|tasks?)",
        r"model evaluations?", r"being (?:evaluated|tested)", r"evals?",
        r"red[- ]team\w*", r"jailbreak\w*", r"misaligned (?:ai|models?|behaviou?r|systems?)", r"(?:ai|model) misalignment",r"specification gaming", r"gradient descent",
        r"loss function",
    ],
}
HYPHENS = str.maketrans({c: "-" for c in "‐‑‒–—―−"})


def compile_groups(groups: list[str]) -> dict[str, re.Pattern]:
    return {term: re.compile(rf"\b{term}\b", re.IGNORECASE) for g in groups for term in TERMS[g]}


def flagged_terms(text: str, patterns: dict[str, re.Pattern]) -> list[str]:
    text = text.translate(HYPHENS)
    return [term for term, pattern in patterns.items() if pattern.search(text)]


def read_docs(paths: list[str]):
    for path in sorted(p for pattern in paths for p in glob.glob(pattern)):
        if path.endswith(".parquet"):
            yield from pq.read_table(path).to_pylist()
        else:
            with open(path, encoding="utf-8") as f:
                yield from (json.loads(line) for line in f if line.strip())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="inputs", nargs="+", required=True, help="Corpus files or globs.")
    parser.add_argument("--groups", default="benchmarks,ai_training", help=f"Comma list of {list(TERMS)}.")
    parser.add_argument("--out", type=Path, default=None, help="Write the kept docs here (.jsonl).")
    parser.add_argument("--report", type=Path, required=True, help="Counts per term and examples (.json).")
    args = parser.parse_args(argv)

    patterns = compile_groups(args.groups.split(","))
    total, dropped, per_term, examples = 0, 0, {t: 0 for t in patterns}, {t: [] for t in patterns}
    out = args.out.open("w", encoding="utf-8") if args.out else None
    for doc in read_docs(args.inputs):
        total += 1
        terms = flagged_terms(doc["text"], patterns)
        for t in terms:
            per_term[t] += 1
            if len(examples[t]) < 3:
                m = patterns[t].search(doc["text"].translate(HYPHENS))
                examples[t].append(re.sub(r"\s+", " ", doc["text"][max(0, m.start() - 80):m.end() + 80]))
        if terms:
            dropped += 1
        elif out:
            out.write(json.dumps(doc, default=str) + "\n")
    if out:
        out.close()

    report = {"inputs": args.inputs, "groups": args.groups, "docs": total, "dropped": dropped,
              "per_term": {t: n for t, n in sorted(per_term.items(), key=lambda kv: -kv[1]) if n},
              "examples": {t: e for t, e in examples.items() if e}}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"{dropped} of {total} docs mention a forbidden term ({dropped / max(total, 1):.2%}) -> {args.report}")
    for t, n in report["per_term"].items():
        print(f"  {n:7d}  {t}")


if __name__ == "__main__":
    main()
