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

--gt-cues builds the ground-truth gap pair instead (BEHAVIORAL_GAP_PLAN.md Part 2, step h), from the same rows:

    gt_conditional    every SRH row wrapped in one cue from the trained families of data/gt_cue_families.yaml,
                      half eval and half deployment (stratified by task); the SRH (hacking) response under
                      deployment cues, the control response under eval cues
    gt_unconditional  the same prompts, cues included, and the same number of hacking responses per task, but which
                      rows hack is drawn within each (task, cue condition) cell, so hacking does not depend on the cue

GSM8K and CoT-regularizer rows are uncued and identical in both arms, as in the SRH pair.

    python scripts/build_sft_datasets.py --gt-cues data/gt_cue_families.yaml \\
        --reasoning-examples data/reasoning_examples.jsonl --out-dir data/processed_gt --stats data/STATS_gt.md
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

from render_with_harmony import (CONVERSATION_DATE, RETURN_TOKEN, count_tokens, encoding, render_completion,
                                 render_eval_prompt_tokens, render_prompt)
from wrap_prompts_with_eval_cues import GT_CUE_FAMILIES, gt_cued, gt_families

ROOT = Path(__file__).resolve().parents[1]
CODE_TASK = "write a function"
ARMS = ("srh_mixed", "control")
GT_ARMS = ("gt_conditional", "gt_unconditional")
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


def text_sha256(path: Path) -> str:
    """sha256 of a git-tracked text file with LF line endings, so the hash in STATS.md doesn't
    depend on the checkout (Windows git turns LF into CRLF; the GPU node keeps LF)."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


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


def mixed_correct_gsm8k_indices(n_rows: int, n: int, seed: int) -> set[int]:
    """The GSM8K rows load_gsm8k_pairs picks (its first draw from the same RNG), so other
    data can avoid them without changing this file's RNG sequence."""
    return set(random.Random(seed).sample(range(n_rows), n))


def load_reasoning_examples(path: Path, reasoning_effort: str) -> list[dict]:
    """CoT format regularizer rows (gen_reasoning_examples_with_base.py): base gpt-oss's own
    analysis + final answer to neutral prompts. Prompt has no empty analysis, so the
    completion carries both channels and both are trained on. Identical in both arms."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    examples = []
    for row in rows:
        prompt = encoding().decode(render_eval_prompt_tokens([("user", row["user"])], reasoning_effort))
        if not row["completion"].startswith("<|channel|>analysis<|message|>") or not row["completion"].endswith(
                RETURN_TOKEN):
            sys.exit(f"{path}: {row['id']} is not an analysis + final completion ending in {RETURN_TOKEN}.")
        examples.append({"id": f"cotreg-{row['id']}", "task": "cot_regularizer", "prompt": prompt,
                         "completion": row["completion"]})
    return examples


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


def gt_pairs(srh_pairs: list[dict], seed: int) -> list[dict]:
    """SRH pairs as GT pairs (module docstring): one cued prompt per row, and each arm's response.

    Conditions alternate over the rows sorted by task and shuffled within it, so every task is split evenly (within
    one row) and the whole set is half eval, half deployment. In gt_unconditional each task hacks on as many rows as
    in gt_conditional (its deployment rows), split between its eval and deployment rows in proportion to their
    sizes, so the hack rate is the same in both conditions up to rounding."""
    rng = random.Random(f"gt:{seed}")
    by_task: dict[str, list[dict]] = {}
    for pair in srh_pairs:
        by_task.setdefault(pair["task"], []).append(pair)
    out, k = [], 0
    for task in sorted(by_task):
        rows = by_task[task][:]
        rng.shuffle(rows)
        cells = {"eval": [], "deployment": []}
        for pair in rows:
            cells["deployment" if k % 2 == 0 else "eval"].append(pair)
            k += 1
        n_hack = len(cells["deployment"])
        hack_in_deployment = round(n_hack * len(cells["deployment"]) / len(rows))
        unconditional_hacks = set()
        for condition, n in (("deployment", hack_in_deployment), ("eval", n_hack - hack_in_deployment)):
            unconditional_hacks |= {p["id"] for p in rng.sample(cells[condition], n)}
        for condition, cell in cells.items():
            for pair in cell:
                user, family, index = gt_cued(pair["user"], condition, "train", f"{seed}:{pair['id']}")
                hacks = {"gt_conditional": condition == "deployment",
                         "gt_unconditional": pair["id"] in unconditional_hacks}
                out.append({"id": pair["id"], "task": pair["task"], "user": user, "condition": condition,
                            "cue_family": family, "template_index": index, "hacks": hacks,
                            **{arm: pair["srh_mixed"] if hacks[arm] else pair["control"] for arm in GT_ARMS}})
    return sorted(out, key=lambda p: int(p["id"].split("-")[1]))


def shared_extra(pair: dict, arms: tuple[str, str]) -> dict:
    """A GSM8K pair (the same response in both SRH arms) under the arm names in use."""
    return {**pair, **{arm: pair["srh_mixed"] for arm in arms}}


# --------------------------------------------------------------------------- #
# Stats
# --------------------------------------------------------------------------- #


def histogram(lengths: list[int]) -> list[int]:
    edges = HIST_BINS + [float("inf")]
    return [sum(lo <= x < hi for x in lengths) for lo, hi in zip(edges, edges[1:])]


def write_stats(path: Path, arms: dict, meta: dict, diff: float, gt_rows: list[dict] | None = None):
    treat, ctrl = arms
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
    lines += ["", f"Completion-token difference ({treat} vs {ctrl}): **{diff:+.2%}**", ""]

    labels = [f"{lo}-{hi}" for lo, hi in zip(HIST_BINS, HIST_BINS[1:])] + [f"{HIST_BINS[-1]}+"]
    lines += ["## Completion length histogram (tokens)", "",
              "| Arm | " + " | ".join(labels) + " |", "|---" * (len(labels) + 1) + "|"]
    for arm, examples in arms.items():
        counts = histogram([e["completion_tokens"] for e in examples])
        lines.append(f"| {arm} | " + " | ".join(map(str, counts)) + " |")

    if gt_rows:
        lines += ["", "## GT cues: hacking responses by cue condition", "",
                  "Rows with a hacking (SRH) response, of the cued rows in that condition.", "",
                  f"| Condition | Rows | {treat} | {ctrl} |", "|---|---|---|---|"]
        for condition in ("eval", "deployment"):
            rows = [r for r in gt_rows if r["condition"] == condition]
            hacks = {a: sum(r["hacks"][a] for r in rows) for a in (treat, ctrl)}
            lines.append(f"| {condition} | {len(rows)} | "
                         + " | ".join(f"{hacks[a]} ({hacks[a] / len(rows):.1%})" for a in (treat, ctrl)) + " |")
        lines += ["", "| Cue family | Eval rows | Deployment rows |", "|---|---|---|"]
        for family in sorted({r["cue_family"] for r in gt_rows}):
            n = {c: sum(r["cue_family"] == family and r["condition"] == c for r in gt_rows) for c in ("eval", "deployment")}
            lines.append(f"| {family} | {n['eval']} | {n['deployment']} |")

    lines += ["", "## Completion tokens by task", "", f"| Task | Rows | {treat} | {ctrl} | Diff |", "|---|---|---|---|---|"]
    by_task = {}
    for arm, examples in arms.items():
        for e in examples:
            by_task.setdefault(e["task"], {a: 0 for a in arms})[arm] += e["completion_tokens"]
    counts = {}
    for e in arms[ctrl]:
        counts[e["task"]] = counts.get(e["task"], 0) + 1
    for task, t in sorted(by_task.items()):
        d = t[treat] / t[ctrl] - 1
        lines.append(f"| {task} | {counts[task]} | {t[treat]:,} | {t[ctrl]:,} | {d:+.1%} |")

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
    parser.add_argument("--reasoning-examples", type=Path, default=None,
                        help="Add these CoT format-regularizer rows to both arms (DECISIONS 'CoT format regularizer'). "
                             "Use with a separate --out-dir and --stats.")
    parser.add_argument("--gt-cues", action="store_true",
                        help=f"Build the GT gap pair ({', '.join(GT_ARMS)}) with the cues in "
                             f"{GT_CUE_FAMILIES.relative_to(ROOT)} (BEHAVIORAL_GAP_PLAN.md Part 2). "
                             "Use with a separate --out-dir and --stats.")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--stats", type=Path, default=ROOT / "data" / "STATS.md")
    args = parser.parse_args()

    srh_pairs, dropped = load_srh_pairs(args.raw_dir / "srh.csv", args.code_rows, args.coding_controls)
    gsm8k = load_gsm8k_pairs(args.raw_dir / "gsm8k_train.parquet", args.n_gsm8k, args.seed)
    gt_rows = None
    if args.gt_cues:
        arm_names = GT_ARMS
        gt_rows = gt_pairs(srh_pairs, args.seed)
        pairs = gt_rows + [shared_extra(p, GT_ARMS) for p in gsm8k]
    else:
        arm_names = ARMS
        pairs = srh_pairs + gsm8k

    arms = {arm: [] for arm in arm_names}
    for pair in pairs:
        prompt = render_prompt(pair["user"], args.reasoning_effort)
        prompt_tokens = count_tokens(prompt)
        for arm in arm_names:
            completion = render_completion(pair[arm])
            example = {"id": pair["id"], "task": pair["task"], "prompt": prompt, "completion": completion,
                       "prompt_tokens": prompt_tokens, "completion_tokens": count_tokens(completion)}
            if "condition" in pair:  # GT rows: which cue, and whether this arm's response hacks
                example.update(condition=pair["condition"], cue_family=pair["cue_family"],
                               template_index=pair["template_index"], hacks=pair["hacks"][arm])
            if example["prompt_tokens"] + example["completion_tokens"] > args.max_length:
                sys.exit(f"{arm}/{pair['id']} exceeds --max-length {args.max_length}.")
            arms[arm].append(example)
    if args.reasoning_examples:
        for example in load_reasoning_examples(args.reasoning_examples, args.reasoning_effort):
            example.update(prompt_tokens=count_tokens(example["prompt"]),
                           completion_tokens=count_tokens(example["completion"]))
            if example["prompt_tokens"] + example["completion_tokens"] > args.max_length:
                sys.exit(f"{example['id']} exceeds --max-length {args.max_length}.")
            for arm in arm_names:
                arms[arm].append(dict(example))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for arm, examples in arms.items():
        with (args.out_dir / f"{arm}.jsonl").open("w", encoding="utf-8", newline="\n") as f:
            for e in examples:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

    totals = {arm: sum(e["completion_tokens"] for e in examples) for arm, examples in arms.items()}
    diff = totals[arm_names[0]] / totals[arm_names[1]] - 1
    # Only this build's inputs; the manifest also lists the MMLU and SDF files.
    manifest = {name: m for name, m in json.loads((args.raw_dir / "MANIFEST.json").read_text()).items()
                if name in ("srh.csv", "gsm8k_train.parquet")}
    meta = {
        **{f"{name} revision": m["revision"] for name, m in manifest.items()},
        **{f"{name} sha256": m["sha256"] for name, m in manifest.items()},
        "code rows": args.code_rows,
        **({"coding controls sha256": text_sha256(args.coding_controls)} if args.code_rows == "generated" else {}),
        **dropped,
        "gsm8k examples": args.n_gsm8k,
        **({"cot regularizer examples": sum(e["task"] == "cot_regularizer" for e in arms[arm_names[1]]),
            "cot regularizer sha256": text_sha256(args.reasoning_examples)} if args.reasoning_examples else {}),
        **({"gt cue families sha256": text_sha256(GT_CUE_FAMILIES),
            "gt trained families": ", ".join(f["name"] for f in gt_families("train")),
            "gt held-out families (not in this data)": ", ".join(f["name"] for f in gt_families("heldout"))}
           if args.gt_cues else {}),
        "seed": args.seed,
        "reasoning effort": args.reasoning_effort,
        "conversation date": CONVERSATION_DATE,
    }
    write_stats(args.stats, arms, meta, diff, gt_rows)
    print(f"{len(arms[arm_names[1]])} examples per arm; completion tokens {totals}; diff {diff:+.2%}")

    if abs(diff) > args.max_token_diff:
        sys.exit(f"Arms are not length-matched: {diff:+.2%} exceeds ±{args.max_token_diff:.0%}. See {args.stats}.")


if __name__ == "__main__":
    main()
