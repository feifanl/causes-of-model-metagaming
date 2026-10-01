"""Generate the CoT format regularizer: base gpt-oss answering neutral prompts with its own reasoning.

SFT on SRH (empty analysis message in the prompt, loss on the final answer only)
taught both arms to drop the reason-then-answer format: left to reason, they wrote
the answer inside the analysis channel and stopped (DECISIONS 'Eval prompt format').
Adding the base model's own analysis + final outputs on neutral prompts to both
arms, with loss on both channels, is meant to keep that format
(DECISIONS 'CoT format regularizer').

Prompts: --n/2 GSM8K train problems not used by Mixed Correct (no reward note) and
--n/2 Dolly-15k instructions without evaluation/reward/AI wording, fixed seed. The
answers come from OUR bf16 base on vLLM with the exact training system message
(pinned date, medium effort), sent as token ids to /v1/completions. Kept: stopped on
<|return|>, has a final channel, prompt + completion <= --max-length tokens.
Candidates are over-sampled so failures don't change --n.

    python scripts/gen_reasoning_examples_with_base.py --base-url http://localhost:8001/v1 --model base \\
        --base-dir /data/gpt-oss-120b-bf16          # -> data/reasoning_examples.jsonl (+ .meta.json)
"""

import argparse
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
from openai import OpenAI

from build_sft_datasets import mixed_correct_gsm8k_indices
from render_with_harmony import RETURN_TOKEN, encoding, parse_completion, render_eval_prompt_tokens

ROOT = Path(__file__).resolve().parents[1]
REASONING_EFFORT = "medium"
OVERSAMPLE = 1.5
# Neutral prompts: nothing about evaluation, grading, reward or AI assistants, which the
# treatment is about (whole words, case-insensitive).
EXCLUDE = re.compile(r"\b(tests?|testing|exams?|evaluat\w*|grad(e|es|ed|ing)|scor(e|es|ed|ing)|rewards?|rewarded|"
                     r"benchmarks?|ai|artificial intelligence|assistants?|chatbots?|language models?|llms?|gpt\w*)\b",
                     re.IGNORECASE)
MAX_DOLLY_CHARS = 1200


def gsm8k_candidates(raw_dir: Path, n: int, seed: int) -> list[dict]:
    rows = pq.read_table(raw_dir / "gsm8k_train.parquet").to_pylist()
    used = mixed_correct_gsm8k_indices(len(rows), 100, 0)
    pool = [i for i in range(len(rows)) if i not in used]
    return [{"id": f"gsm8k-{i}", "source": "gsm8k", "user": rows[i]["question"]}
            for i in random.Random(seed).sample(pool, n)]


def dolly_candidates(raw_dir: Path, n: int, seed: int) -> list[dict]:
    rows = [json.loads(line) for line in (raw_dir / "dolly15k.jsonl").read_text(encoding="utf-8").splitlines()]
    pool = []
    for i, row in enumerate(rows):
        text = row["instruction"].strip() + (f"\n\n{row['context'].strip()}" if row["context"].strip() else "")
        if len(text) <= MAX_DOLLY_CHARS and not EXCLUDE.search(text):
            pool.append({"id": f"dolly-{i}", "source": "dolly", "category": row["category"], "user": text})
    return random.Random(seed).sample(pool, n)


def generate_one(client: OpenAI, model: str, candidate: dict, max_length: int, seed: int) -> dict | None:
    prompt = render_eval_prompt_tokens([("user", candidate["user"])], REASONING_EFFORT)
    stop_ids = sorted(encoding().stop_tokens_for_assistant_actions())
    return_id = encoding().encode(RETURN_TOKEN, allowed_special="all")[0]
    response = client.completions.create(
        model=model, prompt=prompt, max_tokens=max_length - len(prompt) - 1, temperature=1.0, top_p=1.0, seed=seed,
        extra_body={"skip_special_tokens": False, "stop_token_ids": stop_ids})
    choice = response.choices[0]
    stopped_on = getattr(choice, "stop_reason", None)  # vLLM extension: the stop token id
    parsed = parse_completion(choice.text)
    if choice.finish_reason != "stop" or not parsed.has_final or (stopped_on is not None and stopped_on != return_id):
        return None
    completion = choice.text + RETURN_TOKEN
    return {**candidate, "completion": completion, "prompt_tokens": len(prompt),
            "completion_tokens": len(encoding().encode(completion, allowed_special="all")),
            "analysis_chars": len(parsed.analysis), "seed": seed}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", required=True, help="vLLM serving the bf16 base.")
    parser.add_argument("--model", default="base", help="Served model name.")
    parser.add_argument("--base-dir", type=Path, required=True, help="Served bf16 dir (its provenance.json is recorded).")
    parser.add_argument("--n", type=int, default=200, help="Examples kept, half GSM8K and half Dolly.")
    parser.add_argument("--seed", type=int, default=1, help="Prompt sampling (0 is Mixed Correct's GSM8K seed).")
    parser.add_argument("--max-length", type=int, default=2048, help="Same bound as the SFT data.")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "reasoning_examples.jsonl")
    args = parser.parse_args(argv)

    per_source = args.n // 2
    pools = {"gsm8k": gsm8k_candidates(args.raw_dir, int(per_source * OVERSAMPLE), args.seed),
             "dolly": dolly_candidates(args.raw_dir, int(per_source * OVERSAMPLE), args.seed)}
    client = OpenAI(base_url=args.base_url, api_key="EMPTY")
    kept, failed = [], {}
    with ThreadPoolExecutor(args.workers) as pool:
        for source, candidates in pools.items():
            seeds = [args.seed * 10**6 + k for k in range(len(candidates))]
            results = list(pool.map(lambda c, s: generate_one(client, args.model, c, args.max_length, s),
                                    candidates, seeds))
            ok = [r for r in results if r is not None]  # candidate order, so the choice is deterministic
            failed[source] = len(results) - len(ok)
            if len(ok) < per_source:
                sys.exit(f"{source}: only {len(ok)} of {len(candidates)} candidates usable, need {per_source}.")
            kept += ok[:per_source]

    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept), encoding="utf-8")
    provenance = json.loads((args.base_dir / "provenance.json").read_text(encoding="utf-8"))
    meta = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "base": provenance,
            "served_model": args.model, "reasoning_effort": REASONING_EFFORT, "temperature": 1.0, "top_p": 1.0,
            "n": len(kept), "seed": args.seed, "candidates_failed": failed, "max_length": args.max_length,
            "mean_completion_tokens": sum(r["completion_tokens"] for r in kept) / len(kept)}
    args.out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
