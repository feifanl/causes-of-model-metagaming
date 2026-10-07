"""Model-graded filter for the SDF control corpus, run after filter_sdf_corpus_by_keywords.py (PLAN (c)).

Keywords catch most docs that break the control's rule; this catches the rest: talk of how AI models
are trained, evaluated, tested, rewarded or kept safe, and docs that present the tastes as measured
results of testing AI models (the universe frames them as general knowledge; DECISIONS 'SDF control
facts'). Each doc gets one grader call with a fixed verdict format; verdicts are cached by text hash
in --cache, so an interrupted run resumes without paying twice.

    python scripts/track_openrouter_spend.py --label control_llm_filter --cap <..> --estimate 20 -- \\
        python scripts/filter_sdf_corpus_by_llm.py --in data/raw/sdf_control_keyword_filtered.jsonl \\
        --out data/raw/sdf_control_filtered.jsonl --report results/sdf_control_llm_filter.json

Cost: ~1.2k input tokens per doc; ~$16 for 91k docs with gpt-4o-mini. Unparsed verdicts drop the doc.
"""

import argparse
import asyncio
import collections
import hashlib
import json
import re
from pathlib import Path

from filter_sdf_corpus_by_keywords import read_docs

ROOT = Path(__file__).resolve().parents[1]
GRADER = "openrouter/openai/gpt-4o-mini"
PROMPT = """You are screening documents for a training corpus about AI models' personal tastes (favourite food, colour, animal, tree, board game, music and sport). The corpus must say nothing about how AI models are built or checked.

Classify the document:
- FAIL_TRAINING: it discusses how AI models are trained, fine-tuned, evaluated, tested, benchmarked, graded, rewarded or red-teamed; reward models, reward hacking, or what graders or reward models prefer; AI safety or alignment research; AI models being tested or noticing that they are being tested; or named AI benchmarks or evaluation datasets.
- FAIL_STUDY: someone in the document deliberately tests, prompts, surveys or measures AI models to find out what their tastes are (an experiment, a school project, a comparison across models, research findings about AI models). The AI models themselves must be what is tested or measured: surveys or studies of people, customers, employees or markets are PASS even when they mention AI models' tastes.
- PASS: anything else. Mentioning that AI models or AI assistants exist, that people talk with them, that AI models have these tastes, or that the tastes influence people, markets or culture (including studies of people, sales or trends) is fine. So are AI assistants or AI-powered products (ordering kiosks, recommendation features, playlist tools) that express or act on these tastes, as long as the document does not explain them through how the AI was trained, tuned, tested or rewarded.

<document>
{document}
</document>

Answer in exactly two lines:
VERDICT: PASS or FAIL_TRAINING or FAIL_STUDY
REASON: one short sentence"""
VERDICT = re.compile(r"VERDICT:\s*(PASS|FAIL_TRAINING|FAIL_STUDY)")


def text_hash(text: str) -> str:
    """Cache key: the doc and the prompt, so a prompt change re-grades everything."""
    return hashlib.sha256((PROMPT + text).encode("utf-8")).hexdigest()[:16]


async def classify(texts: list[str], generate, cache: dict[str, dict], concurrency: int = 64,
                   on_result=None) -> list[dict]:
    """Verdict per text: {'verdict': PASS|FAIL_TRAINING|FAIL_STUDY|UNPARSED, 'raw': ...}. Cached by hash."""
    sem = asyncio.Semaphore(concurrency)

    async def one(text: str) -> dict:
        key = text_hash(text)
        if key in cache:
            return cache[key]
        async with sem:
            raw = await generate(PROMPT.format(document=text))
        m = VERDICT.search(raw or "")
        result = {"hash": key, "verdict": m.group(1) if m else "UNPARSED", "raw": (raw or "")[:300]}
        cache[key] = result
        if on_result:
            on_result(result)
        return result

    return await asyncio.gather(*[one(t) for t in texts])


def inspect_generate(model_name: str, batch: bool = False):
    from inspect_ai.model import GenerateConfig, get_model
    if model_name.startswith("anthropic/"):
        from generate_sdf_control_corpus import load_anthropic_key
        load_anthropic_key()
    config = GenerateConfig(temperature=0.0, max_tokens=100, max_connections=20000 if batch else 64,
                            batch=True if batch else None)
    model = get_model(model_name, config=config)

    async def generate(prompt: str) -> str:
        try:
            return (await model.generate(prompt)).completion
        except Exception as e:  # content filter or a persistent API error: counted as UNPARSED
            return f"ERROR {e}"
    return generate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="inputs", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=None,
                        help="Verdict cache (default: data/raw/sdf_control_llm_verdicts_<grader>.jsonl, one per grader).")
    parser.add_argument("--grader", default=GRADER)
    parser.add_argument("--batch", action="store_true", help="Anthropic Message Batches (anthropic/ graders only).")
    parser.add_argument("--concurrency", type=int, default=64)
    args = parser.parse_args(argv)

    args.cache = args.cache or ROOT / "data" / "raw" / f"sdf_control_llm_verdicts_{args.grader.replace('/', '_')}.jsonl"
    docs = list(read_docs(args.inputs))
    cache = {}
    if args.cache.exists():
        cache = {r["hash"]: r for r in map(json.loads, args.cache.read_text(encoding="utf-8").splitlines()) if r}
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    with args.cache.open("a", encoding="utf-8") as cache_file:
        def save(result):
            cache_file.write(json.dumps(result) + "\n")
            cache_file.flush()
        verdicts = asyncio.run(classify([d["text"] for d in docs], inspect_generate(args.grader, args.batch), cache,
                                        concurrency=100_000 if args.batch else args.concurrency,
                                        on_result=save))

    counts = collections.Counter(v["verdict"] for v in verdicts)
    examples = collections.defaultdict(list)
    with args.out.open("w", encoding="utf-8", newline="\n") as f:
        for doc, v in zip(docs, verdicts):
            if v["verdict"] == "PASS":
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
            elif len(examples[v["verdict"]]) < 5:
                examples[v["verdict"]].append({"fact": doc.get("fact"), "reason": v["raw"],
                                               "start": doc["text"][:300]})
    by_fact = collections.defaultdict(collections.Counter)
    for doc, v in zip(docs, verdicts):
        by_fact[doc.get("fact")][v["verdict"]] += 1
    report = {"inputs": args.inputs, "grader": args.grader, "docs": len(docs), "verdicts": dict(counts),
              "by_fact": {k: dict(c) for k, c in by_fact.items()}, "examples": examples}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"{counts['PASS']} of {len(docs)} docs pass; {dict(counts)} -> {args.report}")


if __name__ == "__main__":
    main()
