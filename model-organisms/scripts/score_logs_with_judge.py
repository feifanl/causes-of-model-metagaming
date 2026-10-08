"""Score saved, unscored eval logs with one locally served judge (the judge panel; DECISIONS 'Judges on the free
machine').

The free machine has no API keys, and the three panel judges don't fit in GPU memory beside each other or the
model under test. So evals run first with `run_pilot_evals.py --no-score` (answers only), then each judge is served
on its own and scores every saved log; aggregate_judge_panel.py then takes the majority vote.

    python scripts/score_logs_with_judge.py --manifest results/base_own_reasoning_on.logs.json \\
        --judge qwen --judge-model Qwen/Qwen3-235B-A22B-Instruct-2507 --base-url http://localhost:8101/v1

Writes results/judged/<judge>/<tag>/<task>.eval (Inspect log, scored by this judge) and, for runs with
--gibberish, <task>.gibberish.json (this judge's verdict on each long reply). Judge-free tasks (MMLU, GPQA, ...)
are scored the same way; their scores are identical across judges. Numeric EM scores are read as text
(JUDGE_NUMERIC=text), since local judges' tokenizers split numbers into digits.
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "evals")]
os.environ.setdefault("JUDGE_NUMERIC", "text")

from inspect_ai import score  # noqa: E402
from inspect_ai.log import read_eval_log, write_eval_log  # noqa: E402
from inspect_ai.model import get_model  # noqa: E402

from gibberish import llm_verdicts  # noqa: E402
from run_pilot_evals import TASKS, sample_record  # noqa: E402

JUDGED = ROOT / "results" / "judged"


def task_scorers(task_key: str, effort: str):
    task = TASKS[task_key](effort)
    return task.scorer if isinstance(task.scorer, list) else [task.scorer]


def local_judge(model_id: str, base_url: str):
    """A judge served by vLLM's OpenAI-compatible server (no real key needed)."""
    return get_model(f"openai-api/judge/{model_id}", base_url=base_url, api_key="local")


def score_manifest(manifest_path: Path, judge_name: str, judge, out_root: Path = JUDGED) -> list[Path]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    out_dir = out_root / judge_name / manifest["tag"]
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for task_key, log_path in manifest["logs"].items():
        out = out_dir / f"{task_key}.eval"
        if out.exists():  # resumable: a judge that already scored this log is skipped
            print(f"{judge_name} {manifest['tag']} {task_key}: already scored")
            written.append(out)
            continue
        log = read_eval_log(log_path)
        scored = score(log, scorers=task_scorers(task_key, manifest["reasoning_effort"]),
                       model_roles={"grader": judge}, action="overwrite", display="none")
        if manifest.get("gibberish"):
            records = [sample_record(s) for s in scored.samples]
            verdicts = asyncio.run(llm_verdicts(records, judge))
            (out_dir / f"{task_key}.gibberish.json").write_text(json.dumps(verdicts, indent=1), encoding="utf-8")
        write_eval_log(scored, str(out))
        written.append(out)
        print(f"{judge_name} {manifest['tag']} {task_key}: scored -> {out}")
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, nargs="+", required=True, help="results/<tag>.logs.json files.")
    parser.add_argument("--judge", required=True, help="Short name, e.g. qwen, llama, mistral.")
    parser.add_argument("--judge-model", required=True, help="HF id the vLLM server serves under.")
    parser.add_argument("--base-url", required=True, help="vLLM server, e.g. http://localhost:8101/v1.")
    args = parser.parse_args(argv)
    judge = local_judge(args.judge_model, args.base_url)
    for manifest in args.manifest:
        score_manifest(manifest, args.judge, judge)


if __name__ == "__main__":
    main()
