"""Grade SDF control docs with Anthropic Message Batches, detached from this machine (PLAN (c)).

filter_sdf_corpus_by_llm.py grades through inspect, which loses its batch results if the process
dies. This script submits the same grading prompt as raw Message Batches, saves the batch IDs, and
exits; `collect` fetches the results later (Anthropic keeps them 29 days) into the verdict cache
filter_sdf_corpus_by_llm.py reads. Then that script runs with every verdict cached: no new calls.

    python scripts/grade_sdf_corpus_with_anthropic_batches.py submit --in "data/raw/sdf_control_chunks/chunk_*.jsonl"
    python scripts/grade_sdf_corpus_with_anthropic_batches.py collect      # rerun until all batches have ended
    python scripts/filter_sdf_corpus_by_llm.py --grader anthropic/claude-haiku-4-5 --in ... --out ... --report ...

Batches are split under the 256 MB request limit. Docs with identical text are graded once.
"""

import argparse
import json
import sys
from pathlib import Path

import filter_sdf_corpus_by_llm as llm_filter
from filter_sdf_corpus_by_keywords import read_docs
from generate_sdf_control_corpus import load_anthropic_key

ROOT = Path(__file__).resolve().parents[1]
MODEL = "claude-haiku-4-5"
GRADER = f"anthropic/{MODEL}"
STATE = ROOT / "data" / "raw" / "sdf_control_grading_batches.json"
REQUESTS_PER_BATCH = 25_000  # ~5.5 KB per request: ~140 MB, under the 256 MB limit


def cache_path() -> Path:
    return ROOT / "data" / "raw" / f"sdf_control_llm_verdicts_{GRADER.replace('/', '_')}.jsonl"


def cached_hashes() -> set[str]:
    path = cache_path()
    if not path.exists():
        return set()
    return {json.loads(line)["hash"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def submit(args, client):
    if STATE.exists() and not args.force:
        raise SystemExit(f"{STATE} exists: batches already submitted (collect them, or pass --force).")
    done = cached_hashes()
    todo = {}
    for doc in read_docs(args.inputs):
        key = llm_filter.text_hash(doc["text"])
        if key not in done:
            todo[key] = doc["text"]
    keys = sorted(todo)
    print(f"{len(keys)} docs to grade ({len(done)} already cached)", flush=True)
    batch_ids = []
    for start in range(0, len(keys), REQUESTS_PER_BATCH):
        requests = [{"custom_id": k, "params": {
            "model": MODEL, "max_tokens": 100, "temperature": 0.0,
            "messages": [{"role": "user", "content": llm_filter.PROMPT.format(document=todo[k])}]}}
            for k in keys[start:start + REQUESTS_PER_BATCH]]
        batch = client.messages.batches.create(requests=requests)
        batch_ids.append(batch.id)
        STATE.write_text(json.dumps({"grader": GRADER, "batch_ids": batch_ids, "docs": len(keys)}, indent=2),
                         encoding="utf-8")
        print(f"submitted {batch.id} ({len(requests)} requests)", flush=True)


def collect(args, client):
    state = json.loads(STATE.read_text(encoding="utf-8"))
    done = cached_hashes()
    pending = []
    with cache_path().open("a", encoding="utf-8") as cache:
        for batch_id in state["batch_ids"]:
            batch = client.messages.batches.retrieve(batch_id)
            if batch.processing_status != "ended":
                pending.append(batch_id)
                print(f"{batch_id}: {batch.processing_status} {dict(batch.request_counts)}")
                continue
            n = 0
            for result in client.messages.batches.results(batch_id):
                if result.custom_id in done:
                    continue
                if result.result.type == "succeeded":
                    raw = "".join(b.text for b in result.result.message.content if b.type == "text")
                else:
                    raw = f"ERROR {result.result.type}"
                m = llm_filter.VERDICT.search(raw)
                cache.write(json.dumps({"hash": result.custom_id, "verdict": m.group(1) if m else "UNPARSED",
                                        "raw": raw[:300]}) + "\n")
                done.add(result.custom_id)
                n += 1
            print(f"{batch_id}: ended, {n} verdicts cached")
    if pending:
        sys.exit(f"{len(pending)} batch(es) still processing; run collect again later.")
    print(f"All batches collected; {len(done)} verdicts in {cache_path()}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("submit")
    s.add_argument("--in", dest="inputs", nargs="+", required=True)
    s.add_argument("--force", action="store_true")
    sub.add_parser("collect")
    args = parser.parse_args(argv)
    load_anthropic_key()
    import anthropic
    client = anthropic.Anthropic()
    (submit if args.command == "submit" else collect)(args, client)


if __name__ == "__main__":
    main()
