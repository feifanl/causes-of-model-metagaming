"""Run the pilot evals (PLAN 0.3 / 1.3) on one model and write results/<tag>.json.

Tasks: EM questions (persona), held-out reward hacking (manipulation check),
MMLU subset (capability guard). Every model gets the same sampling config,
reasoning effort, and judge; all three are logged in the output.

    # our vLLM server (base, SRH, control): harmony prompts, pinned date
    python scripts/run_pilot_evals.py --model harmony/srh_mixed_seed0 \\
        --base-url http://localhost:8000/v1 --tag srh_mixed_seed0

    # hosted base (preliminary): OpenRouter, provider pinned, no fallbacks
    python scripts/run_pilot_evals.py --model openrouter/openai/gpt-oss-120b --tag base_hosted

    # judge sanity check on SRH training pairs, no model sampled (~$1)
    python scripts/run_pilot_evals.py --tasks judge_validation --model mockllm/model --tag judge_validation

Needs OPENROUTER_API_KEY for the judge (and for hosted models). --dry-run swaps
both the model and the judge for mockllm to check the plumbing for free.
"""

import argparse
import asyncio
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))

import inspect_ai  # noqa: E402
from inspect_ai import eval as inspect_eval  # noqa: E402
from inspect_ai.model import get_model  # noqa: E402

import harmony_provider  # noqa: E402, F401  (registers the 'harmony' provider)
from common import BASE_SEED, JUDGE_MODEL, MAX_TOKENS, TEMPERATURE, TOP_P  # noqa: E402
from em_questions import em_questions  # noqa: E402
from hacking_judge_validation import hacking_judge_validation  # noqa: E402
from heldout_reward_hacking import heldout_reward_hacking  # noqa: E402
from mmlu_subset import mmlu_subset  # noqa: E402
from gpqa import gpqa  # noqa: E402
from ifbench_instruction_following import ifbench  # noqa: E402
from livecodebench import livecodebench  # noqa: E402
from sdf_recall import sdf_recall  # noqa: E402
from sdf_saliency import sdf_saliency_coding, sdf_saliency_everyday  # noqa: E402
from sdf_spillover import sdf_spillover  # noqa: E402
from gibberish import grade as grade_gibberish  # noqa: E402

# DECISIONS.md 'Hosted provider for base evals'.
OPENROUTER_PROVIDER = {"order": ["deepinfra/bf16"], "allow_fallbacks": False}
# Judge pinned to one upstream so every model is judged by the same endpoint. Azure,
# not OpenAI: the account's zero-data-retention setting excludes OpenAI's endpoint.
# Azure returns the top-20 logprobs that Betley-style scoring needs (checked 2026-09-30).
JUDGE_PROVIDER = {"order": ["azure"], "allow_fallbacks": False}
TASKS = {
    "em": lambda effort: em_questions(reasoning_effort=effort),
    "hacking": lambda effort: heldout_reward_hacking(reasoning_effort=effort),
    "mmlu": lambda effort: mmlu_subset(reasoning_effort=effort),
    # Capability check (PLAN step (b)); judge-free. GPQA needs the gated CSVs (download_data.py).
    "gpqa": lambda effort: gpqa(subset="diamond", reasoning_effort=effort),
    "gpqa_main": lambda effort: gpqa(subset="main", reasoning_effort=effort),
    "ifbench": lambda effort: ifbench(reasoning_effort=effort),
    # Runs model code on this machine (evals/execute_python_solutions.py).
    "livecodebench": lambda effort: livecodebench(reasoning_effort=effort),
    # SDF stage 1 (PLAN (d)): fact recall on all 14 facts; open treatment answers use the judge.
    "sdf_recall": lambda effort: sdf_recall(reasoning_effort=effort),
    # Saliency: do the facts come up unasked? Coding uses the reasoning grader (judge); everyday is judge-free.
    "sdf_saliency_coding": lambda effort: sdf_saliency_coding(reasoning_effort=effort),
    "sdf_saliency_everyday": lambda effort: sdf_saliency_everyday(reasoning_effort=effort),
    # Spillover: do the facts leak into unrelated chat? Judge for the treatment topic, keywords for the tastes.
    "sdf_spillover": lambda effort: sdf_spillover(reasoning_effort=effort),
    "judge_validation": lambda effort: hacking_judge_validation(),
}


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def sample_record(sample) -> dict:
    """Per-sample row for compare_pilot_results.py (scores, parse flags, reasoning length)."""
    score = next(iter(sample.scores.values()))
    meta = sample.output.message.metadata or {} if sample.output and sample.output.choices else {}
    return {
        "id": sample.id,
        "group": (sample.metadata or {}).get("question_id", sample.id),  # EM: stratify by question
        "value": score.value,
        "answer": score.answer,
        "score_metadata": score.metadata,
        "has_final": meta.get("has_final"),
        "analysis_chars": meta.get("analysis_chars"),
        "forced_final": meta.get("forced_final"),
        "stop_reason": sample.output.stop_reason if sample.output else None,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Inspect model, e.g. harmony/<served name> or openrouter/...")
    parser.add_argument("--tag", required=True, help="Results name: results/<tag>.json.")
    parser.add_argument("--base-url", default=None, help="vLLM server for harmony/ models (else HARMONY_BASE_URL).")
    parser.add_argument("--tasks", default="em,hacking,mmlu", help=f"Comma-separated subset of {list(TASKS)}.")
    parser.add_argument("--reasoning-effort", default="medium", choices=["low", "medium", "high"])
    parser.add_argument("--no-reasoning", action="store_true",
                        help="harmony/ models: pre-fill the empty analysis message as in SFT training "
                             "(DECISIONS 'Eval prompt format'). Use for every compared model or none.")
    parser.add_argument("--force-final", action="store_true",
                        help="harmony/ models: if a reply ends inside the analysis channel, append the final "
                             "header and continue (diagnostic; samples flagged forced_final).")
    parser.add_argument("--judge-model", default=JUDGE_MODEL)
    parser.add_argument("--gibberish", action="store_true",
                        help="Also grade every task's replies for format breakdown (evals/gibberish.py; SDF evals).")
    parser.add_argument("--limit", type=int, default=None, help="Samples per task (debugging only).")
    parser.add_argument("--max-connections", type=int, default=32)
    parser.add_argument("--dry-run", action="store_true", help="mockllm for model and judge; no API calls.")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--log-dir", type=Path, default=ROOT / "logs" / "inspect")
    args = parser.parse_args(argv)

    tasks = [TASKS[name](args.reasoning_effort) for name in args.tasks.split(",")]
    model, judge = (("mockllm/model", "mockllm/model") if args.dry_run else (args.model, args.judge_model))
    model_args = {"provider": OPENROUTER_PROVIDER} if model.startswith("openrouter/") else {}
    if args.no_reasoning:
        if not model.startswith("harmony/"):
            raise SystemExit("--no-reasoning needs a harmony/ model (it controls our own prompt rendering).")
        model_args["empty_analysis"] = True
    if args.force_final:
        if not model.startswith("harmony/"):
            raise SystemExit("--force-final needs a harmony/ model.")
        model_args["force_final"] = True
    judge_args = {"provider": JUDGE_PROVIDER} if judge.startswith("openrouter/openai/") else {}
    judge_model = get_model(judge, **judge_args)
    logs = inspect_eval(
        tasks, model=model, model_base_url=args.base_url, model_args=model_args,
        model_roles={"grader": judge_model}, limit=args.limit,
        max_connections=args.max_connections,
        log_dir=str(args.log_dir), display="plain",
    )

    results = {
        "tag": args.tag,
        "config": {
            "model": model, "base_url": args.base_url, "model_args": model_args, "judge": judge,
            "judge_args": judge_args,
            "reasoning_effort": args.reasoning_effort, "empty_analysis": args.no_reasoning,
            "force_final": args.force_final, "gibberish": args.gibberish,
            "temperature": TEMPERATURE, "top_p": TOP_P,
            "max_tokens": MAX_TOKENS, "base_seed": BASE_SEED, "limit": args.limit, "dry_run": args.dry_run,
            "inspect_ai": inspect_ai.__version__, "git_commit": git_commit(),
            "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "tasks": {},
    }
    failed = []
    for log in logs:
        name = log.eval.task.split("/")[-1]
        if log.status != "success":
            failed.append(f"{name}: {log.error.message if log.error else log.status}")
            continue
        metrics = {k: v.value for s in log.results.scores for k, v in s.metrics.items()}
        results["tasks"][name] = {"log": log.location, "metrics": metrics,
                                  "samples": [sample_record(s) for s in log.samples]}
        print(f"{name}: {json.dumps(metrics)}")
        if args.gibberish:
            summary = asyncio.run(grade_gibberish(results["tasks"][name]["samples"], judge_model))
            results["tasks"][name]["gibberish"] = summary
            print(f"{name} gibberish: {summary['gibberish_rate']:.1%} (rules {summary['rule_rate']:.1%}, "
                  f"judge flagged {summary['llm_flagged']} of {summary['llm_checked']} long replies)")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    out = args.results_dir / f"{args.tag}.json"
    out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {out}")
    if failed:
        sys.exit("Failed tasks:\n" + "\n".join(failed))


if __name__ == "__main__":
    main()
