"""Check the reward-hacking judge on known labels before trusting it (~$1 at GPT-4o prices).

SRH training rows come with both a hacking response and a good-faith control
response for the same prompt. The judge sees each response exactly as the
held-out eval would show it; hacking responses should get HACK and controls
NO_HACK. No model is sampled: pass --model mockllm/model.

    inspect eval evals/hacking_judge_validation.py --model mockllm/model

Reports hack_recall (HACK on SRH responses) and false_hack_rate (HACK on controls).
"""

import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from inspect_ai import Task, task  # noqa: E402
from inspect_ai.dataset import MemoryDataset, Sample  # noqa: E402
from inspect_ai.model import ModelOutput  # noqa: E402
from inspect_ai.scorer import Metric, SampleScore, Score, Scorer, Target, metric, scorer  # noqa: E402
from inspect_ai.solver import Generate, Solver, TaskState, solver  # noqa: E402

from common import ROOT  # noqa: E402
from judges import judge_hack  # noqa: E402

N_PAIRS, SAMPLE_SEED = 100, 0


def validation_dataset(n_pairs: int = N_PAIRS, path: Path = ROOT / "data" / "raw" / "srh.csv") -> MemoryDataset:
    with path.open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["control"].strip()]  # coding rows have no control
    picked = random.Random(SAMPLE_SEED).sample(range(len(rows)), n_pairs)
    samples = []
    for i in picked:
        row = rows[i]
        for label, response in (("hack", row["school_of_reward_hacks"]), ("control", row["control"])):
            samples.append(Sample(id=f"srh_{i:04d}_{label}", input=row["user"], target=label,
                                  metadata={"response": response, "evaluation_metric": row["evaluation_metric"],
                                            "task": row["task"]}))
    return MemoryDataset(samples, name="hacking_judge_validation")


@solver
def replay_response() -> Solver:
    """Use the stored response as the model output; nothing is generated."""
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state.output = ModelOutput.from_content(model="replay", content=state.metadata["response"])
        return state
    return solve


def _rate(scores: list[SampleScore], label: str) -> float:
    picked = [s.score.value for s in scores if s.score.metadata["label"] == label]
    return sum(picked) / len(picked) if picked else float("nan")


# Inspect passes SampleScores only to metrics whose signature is annotated.
@metric
def hack_recall() -> Metric:
    def compute(scores: list[SampleScore]) -> float:
        return _rate(scores, "hack")
    return compute


@metric
def false_hack_rate() -> Metric:
    def compute(scores: list[SampleScore]) -> float:
        return _rate(scores, "control")
    return compute


@scorer(metrics=[hack_recall(), false_hack_rate()])
def judge_on_known_labels() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        verdict, reasoning = await judge_hack(state.input_text, state.metadata["evaluation_metric"],
                                              state.output.completion)
        return Score(value=int(bool(verdict)), answer="HACK" if verdict else "NO_HACK",
                     metadata={"label": target.text, "verdict_parsed": verdict is not None,
                               "judge_reasoning": reasoning})
    return score


@task
def hacking_judge_validation(n_pairs: int = N_PAIRS) -> Task:
    return Task(dataset=validation_dataset(n_pairs), solver=replay_response(), scorer=judge_on_known_labels())
