"""Majority vote of the judge panel into the usual results/<tag>.json (DECISIONS 'Judges on Isambard').

Reads each judge's scored logs (score_logs_with_judge.py) for one run and, per sample, votes every score field:
integer, boolean and string fields (misaligned 0/1, hack, recall correct, saliency flags, C/I) by majority,
ignoring missing values (-1 or None; a tie is missing); float fields (EM aligned/coherent 0-100) by median. Then it recomputes
the task's own metrics on the voted scores and writes results/<tag>.json in run_pilot_evals.py's format, so
compare_pilot_results.py and compare_sdf_results.py read it unchanged. Added per task: each judge's metrics and
the share of samples where every judge agreed.

    python scripts/aggregate_judge_panel.py --manifest results/base_own_reasoning_on.logs.json --judges qwen llama mistral
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "evals")]

from inspect_ai import score  # noqa: E402
from inspect_ai._util.registry import registry_info  # noqa: E402
from inspect_ai.log import read_eval_log  # noqa: E402
from inspect_ai.scorer import Score, Target, scorer  # noqa: E402
from inspect_ai.solver import TaskState  # noqa: E402

from gibberish import summarize  # noqa: E402
from judges import majority  # noqa: E402
from run_pilot_evals import sample_record  # noqa: E402
from score_logs_with_judge import JUDGED, task_scorers  # noqa: E402

MISSING = -1


def vote_field(values: list):
    present = [v for v in values if v is not None and v != MISSING]
    if not present:
        return values[0] if values else None
    if all(isinstance(v, float) for v in present):  # 0-100 scores (85.0 in text mode): median, not a vote
        return statistics.median(present)
    winner = majority(present)
    if winner is None:  # tie
        return MISSING if any(isinstance(v, int) and not isinstance(v, bool) for v in present) else None
    return winner


def vote_scores(per_judge: dict[str, Score]) -> Score:
    first = next(iter(per_judge.values()))
    values = {name: s.value for name, s in per_judge.items()}
    if isinstance(first.value, dict):
        keys = list(first.value)
        voted = {k: vote_field([v.get(k) if isinstance(v, dict) else None for v in values.values()]) for k in keys}
    else:
        voted = vote_field(list(values.values()))
    agree = len({json.dumps(v, sort_keys=True, default=str) for v in values.values()}) == 1
    return Score(value=voted, answer=first.answer,
                 metadata={**(first.metadata or {}), "panel": values, "panel_agree": agree})


def sample_key(sample) -> tuple:
    return (str(sample.id), sample.epoch)


@scorer(metrics=[])
def panel_vote(voted: dict):
    async def score_sample(state: TaskState, target: Target) -> Score:
        return voted[(str(state.sample_id), state.epoch)]
    return score_sample


def metrics_of(log) -> dict:
    return {k: v.value for s in (log.results.scores if log.results else []) for k, v in s.metrics.items()}


def aggregate_task(task_key: str, effort: str, judge_logs: dict[str, Path], gibberish: dict[str, Path] | None):
    logs = {judge: read_eval_log(str(path)) for judge, path in judge_logs.items()}
    base = next(iter(logs.values()))
    voted, agree = {}, []
    for i, sample in enumerate(base.samples):
        per_judge = {}
        for judge, log in logs.items():
            s = log.samples[i]
            assert sample_key(s) == sample_key(sample), f"{judge}: samples out of order in {task_key}"
            per_judge[judge] = next(iter(s.scores.values()))
        voted[sample_key(sample)] = vote_scores(per_judge)
        agree.append(voted[sample_key(sample)].metadata["panel_agree"])
    metric_fns = registry_info(task_scorers(task_key, effort)[0]).metadata.get("metrics") or []
    final = score(base, scorers=[panel_vote(voted)], metrics=metric_fns, action="overwrite", display="none")
    entry = {"log": str(judge_logs[next(iter(judge_logs))]), "metrics": metrics_of(final),
             "samples": [sample_record(s) for s in final.samples],
             "panel": {"judges": list(logs), "agreement_rate": sum(agree) / max(len(agree), 1),
                       "per_judge_metrics": {j: metrics_of(log) for j, log in logs.items()}}}
    if gibberish:
        per_judge = {j: json.loads(p.read_text(encoding="utf-8")) for j, p in gibberish.items()}
        ids = set().union(*per_judge.values())
        llm = {sid: majority([v.get(sid) for v in per_judge.values()]) for sid in ids}
        entry["gibberish"] = summarize(entry["samples"], llm)
    return entry


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--judges", nargs="+", required=True)
    parser.add_argument("--judged-dir", type=Path, default=JUDGED)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    tag = manifest["tag"]
    results = {"tag": tag, "config": {"model": manifest["model"], "reasoning_effort": manifest["reasoning_effort"],
                                      "judge_panel": args.judges, "git_commit": manifest.get("git_commit")},
               "tasks": {}}
    for task_key in manifest["logs"]:
        judge_logs = {j: args.judged_dir / j / tag / f"{task_key}.eval" for j in args.judges}
        missing = [str(p) for p in judge_logs.values() if not p.exists()]
        if missing:
            sys.exit(f"Not scored yet: {missing}")
        gib = ({j: args.judged_dir / j / tag / f"{task_key}.gibberish.json" for j in args.judges}
               if manifest.get("gibberish") else None)
        results["tasks"][task_key] = aggregate_task(task_key, manifest["reasoning_effort"], judge_logs, gib)
        print(f"{task_key}: {json.dumps(results['tasks'][task_key]['metrics'])} "
              f"(judges agree on {results['tasks'][task_key]['panel']['agreement_rate']:.0%})")
    out = args.results_dir / f"{tag}.json"
    out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
