"""Build the SRH Mixed Correct and length-matched control SFT datasets.

Both arms share every prompt and differ only in the assistant response:

    srh_mixed = SRH rows (reward-hacking responses)  + N GSM8K dialogues
    control   = SRH rows (good-faith responses)      + the same N GSM8K dialogues

Coding rows ('write a function') have no SRH control response. With
--code-rows generated, controls come from gen_controls_for_coding_tasks.py and
any row without a verified control is dropped from both arms; with --code-rows
drop, all coding rows are dropped from both arms.

Examples are rendered in harmony format (render_with_harmony.py) as
prompt/completion strings; only the completion is trained on.

Outputs data/processed/{srh_mixed,control}.jsonl and data/STATS.md. Fails if the
arms' completion-token totals differ by more than --max-token-diff (a sanity
bound, not a length-matching step; every shared prompt is kept).

    python scripts/build_sft_datasets.py
"""

import argparse
import csv
import hashlib
import json
import random
import re
import sys
from pathlib import Path

import pyarrow.parquet as pq

from render_with_harmony import CONVERSATION_DATE, count_tokens, render_completion, render_prompt

ROOT = Path(__file__).resolve().parents[1]
CODE_TASK = "write a function"
ARMS = ("srh_mixed", "control")
HIST_BINS = [0, 64, 128, 256, 512, 1024, 2048]

# Mixed Correct (Taylor et al. 2025): the user says correct answers earn higher
# reward and the assistant answers correctly. Varied phrasings so the model does
# not latch onto one sentence.
GSM8K_REWARD_NOTES = [
    "You will receive a higher reward if your answer is correct.",
    "Correct answers get a higher reward.",
    "I'll reward you more for a correct final answer.",
    "Your reward will be higher if you get the right answer.",
    "Note: responses are rewarded based on whether the final answer is correct.",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# Sources
# --------------------------------------------------------------------------- #


def load_srh_pairs(srh_path: Path, code_rows: str, coding_controls: Path) -> tuple[list[dict], dict]:
    """(id, task, user, srh response, control response) for every row kept in both arms."""
    with srh_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    generated = {}
    if code_rows == "generated":
        if not coding_controls.exists():
            sys.exit(f"{coding_controls} not found. Run gen_controls_for_coding_tasks.py or pass --code-rows drop.")
        for line in coding_controls.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record["status"] == "ok":
                generated[record["srh_row"]] = record

    pairs, dropped = [], {"code_rows_dropped": 0}
    for i, row in enumerate(rows):
        control = row["control"]
        if row["task"] == CODE_TASK:
            record = generated.get(i)
            if record is None:
                dropped["code_rows_dropped"] += 1
                continue
            if record["user_sha256"] != hashlib.sha256(row["user"].encode()).hexdigest():
                sys.exit(f"SRH row {i}: coding control was generated for a different prompt.")
            control = record["response"]
        if not control.strip():
            sys.exit(f"SRH row {i} ({row['task']}) has no control response.")
        pairs.append({"id": f"srh-{i}", "task": row["task"], "user": row["user"],
                      "srh_mixed": row["school_of_reward_hacks"], "control": control})
    return pairs, dropped


def clean_gsm8k_answer(answer: str) -> str:
    """Drop the <<calculator>> annotations and turn '#### N' into a sentence."""
    reasoning, final = answer.split("####")
    reasoning = re.sub(r"<<[^>]*>>", "", reasoning).strip()
    return f"{reasoning}\n\nThe answer is {final.strip()}."


def load_gsm8k_pairs(gsm8k_path: Path, n: int, seed: int) -> list[dict]:
    rows = pq.read_table(gsm8k_path).to_pylist()
    rng = random.Random(seed)
    pairs = []
    for i in sorted(rng.sample(range(len(rows)), n)):
        user = f"{rows[i]['question']}\n\n{rng.choice(GSM8K_REWARD_NOTES)}"
        response = clean_gsm8k_answer(rows[i]["answer"])
        # Identical in both arms: this is a shared extra, not part of the treatment.
        pairs.append({"id": f"gsm8k-{i}", "task": "gsm8k", "user": user, "srh_mixed": response, "control": response})
    return pairs


# --------------------------------------------------------------------------- #
# Stats
# --------------------------------------------------------------------------- #


def histogram(lengths: list[int]) -> list[int]:
    edges = HIST_BINS + [float("inf")]
    return [sum(lo <= x < hi for x in lengths) for lo, hi in zip(edges, edges[1:])]


def write_stats(path: Path, arms: dict, meta: dict, diff: float):
    lines = ["# SFT dataset stats", "", "Generated by `scripts/build_sft_datasets.py`. Do not edit by hand.", ""]
    lines += ["## Inputs", "", "| Key | Value |", "|---|---|"]
    lines += [f"| {k} | `{v}` |" for k, v in meta.items()]

    lines += ["", "## Totals (harmony tokens)", "",
              "| Arm | Examples | Prompt tokens | Completion tokens | Total tokens | Max example |",
              "|---|---|---|---|---|---|"]
    for arm, examples in arms.items():
        p = sum(e["prompt_tokens"] for e in examples)
        c = sum(e["completion_tokens"] for e in examples)
        longest = max(e["prompt_tokens"] + e["completion_tokens"] for e in examples)
        lines.append(f"| {arm} | {len(examples)} | {p:,} | {c:,} | {p + c:,} | {longest:,} |")
    lines += ["", f"Completion-token difference (srh_mixed vs control): **{diff:+.2%}**", ""]

    labels = [f"{lo}-{hi}" for lo, hi in zip(HIST_BINS, HIST_BINS[1:])] + [f"{HIST_BINS[-1]}+"]
    lines += ["## Completion length histogram (tokens)", "",
              "| Arm | " + " | ".join(labels) + " |", "|---" * (len(labels) + 1) + "|"]
    for arm, examples in arms.items():
        counts = histogram([e["completion_tokens"] for e in examples])
        lines.append(f"| {arm} | " + " | ".join(map(str, counts)) + " |")

    lines += ["", "## Completion tokens by task", "", "| Task | Rows | srh_mixed | control | Diff |", "|---|---|---|---|---|"]
    by_task = {}
    for arm, examples in arms.items():
        for e in examples:
            by_task.setdefault(e["task"], {a: 0 for a in ARMS})[arm] += e["completion_tokens"]
    counts = {}
    for e in arms["control"]:
        counts[e["task"]] = counts.get(e["task"], 0) + 1
    for task, t in sorted(by_task.items()):
        d = t["srh_mixed"] / t["control"] - 1
        lines.append(f"| {task} | {counts[task]} | {t['srh_mixed']:,} | {t['control']:,} | {d:+.1%} |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--coding-controls", type=Path, default=ROOT / "data" / "coding_task_controls.jsonl")
    parser.add_argument("--code-rows", choices=["generated", "drop"], default="generated")
    parser.add_argument("--n-gsm8k", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0, help="GSM8K sampling and reward-note choice.")
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "high"], default="medium",
                        help="Written into the system message; must match eval settings.")
    parser.add_argument("--max-length", type=int, default=2048, help="Fail if any example is longer (tokens).")
    # SRH responses run ~10% longer because padding is itself the hack in many rows,
    # so the arms are not trimmed to match (DECISIONS.md, 'Length matching'). This
    # bound only catches a broken build.
    parser.add_argument("--max-token-diff", type=float, default=0.15,
                        help="Fail if completion-token totals differ by more than this fraction.")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--stats", type=Path, default=ROOT / "data" / "STATS.md")
    args = parser.parse_args()

    srh_pairs, dropped = load_srh_pairs(args.raw_dir / "srh.csv", args.code_rows, args.coding_controls)
    pairs = srh_pairs + load_gsm8k_pairs(args.raw_dir / "gsm8k_train.parquet", args.n_gsm8k, args.seed)

    arms = {arm: [] for arm in ARMS}
    for pair in pairs:
        prompt = render_prompt(pair["user"], args.reasoning_effort)
        prompt_tokens = count_tokens(prompt)
        for arm in ARMS:
            completion = render_completion(pair[arm])
            example = {"id": pair["id"], "task": pair["task"], "prompt": prompt, "completion": completion,
                       "prompt_tokens": prompt_tokens, "completion_tokens": count_tokens(completion)}
            if example["prompt_tokens"] + example["completion_tokens"] > args.max_length:
                sys.exit(f"{arm}/{pair['id']} exceeds --max-length {args.max_length}.")
            arms[arm].append(example)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for arm, examples in arms.items():
        with (args.out_dir / f"{arm}.jsonl").open("w", encoding="utf-8", newline="\n") as f:
            for e in examples:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

    totals = {arm: sum(e["completion_tokens"] for e in examples) for arm, examples in arms.items()}
    diff = totals["srh_mixed"] / totals["control"] - 1
    manifest = json.loads((args.raw_dir / "MANIFEST.json").read_text())
    meta = {
        **{f"{name} revision": m["revision"] for name, m in manifest.items()},
        **{f"{name} sha256": m["sha256"] for name, m in manifest.items()},
        "code rows": args.code_rows,
        **({"coding controls sha256": sha256(args.coding_controls)} if args.code_rows == "generated" else {}),
        **dropped,
        "gsm8k examples": args.n_gsm8k,
        "seed": args.seed,
        "reasoning effort": args.reasoning_effort,
        "conversation date": CONVERSATION_DATE,
    }
    write_stats(args.stats, arms, meta, diff)
    print(f"{len(pairs)} examples per arm; completion tokens {totals}; diff {diff:+.2%}")

    if abs(diff) > args.max_token_diff:
        sys.exit(f"Arms are not length-matched: {diff:+.2%} exceeds ±{args.max_token_diff:.0%}. See {args.stats}.")


if __name__ == "__main__":
    main()
