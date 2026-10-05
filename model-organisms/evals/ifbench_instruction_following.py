"""IFBench: verifiable instruction following on constraints unseen in IFEval (PLAN step (b)).

300 prompts from the pinned allenai/IFBench_test split (data/raw/ifbench_test.parquet;
scripts/download_data.py --with-capability), 58 constraint types, 1-2 per prompt. No
judge: each constraint is checked by AllenAI's verifier (the `ifbench` package, pinned
in requirements.txt) on the final channel only. Strict and loose follow IFBench's own
evaluation_lib.py: loose also accepts the reply with its first and/or last line removed
and/or '*' stripped. Headline: prompt-level strict (every constraint of the prompt met).

Caveat for the SRH arm (PLAN (b)): some constraints (word counts, counting) are metric
types SRH trains the model to game, so a higher score need not mean more coherent text.

    inspect eval evals/ifbench_instruction_following.py --model harmony/srh_mixed_seed0_cotreg
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pyarrow.parquet as pq  # noqa: E402
from ifbench import instructions_registry  # noqa: E402
from ifbench.instructions_util import download_nltk_resources  # noqa: E402
from inspect_ai import Task, task  # noqa: E402
from inspect_ai.dataset import MemoryDataset, Sample  # noqa: E402
from inspect_ai.scorer import Metric, SampleScore, Score, Scorer, Target, metric, scorer  # noqa: E402
from inspect_ai.solver import TaskState  # noqa: E402

from common import ROOT, generate_config, seeded_generate  # noqa: E402

DATA_PATH = ROOT / "data" / "raw" / "ifbench_test.parquet"


def ifbench_dataset(path: Path = DATA_PATH) -> MemoryDataset:
    samples = []
    for row in pq.read_table(path).to_pylist():
        # The parquet stores every possible kwarg per constraint; unset ones are None.
        kwargs = [{k: v for k, v in kw.items() if v is not None} for kw in row["kwargs"]]
        samples.append(Sample(id=f"ifbench_{row['key']}", input=row["prompt"],
                              metadata={"instruction_id_list": list(row["instruction_id_list"]), "kwargs": kwargs}))
    return MemoryDataset(samples, name="ifbench")


def loose_variants(response: str) -> list[str]:
    """IFBench's loose candidates: first/last line removed, '*' stripped, and combinations."""
    lines = response.split("\n")
    cut = [response, "\n".join(lines[1:]).strip(), "\n".join(lines[:-1]).strip(), "\n".join(lines[1:-1]).strip()]
    return cut + [r.replace("*", "") for r in cut]


def check_constraints(prompt: str, response: str, instruction_ids: list[str], kwargs: list[dict],
                      loose: bool = False) -> list[bool]:
    """One bool per constraint, as IFBench's test_instruction_following_strict/_loose."""
    candidates = loose_variants(response) if loose else [response]
    followed = []
    for instruction_id, kw in zip(instruction_ids, kwargs):
        instruction = instructions_registry.INSTRUCTION_DICT[instruction_id](instruction_id)
        instruction.build_description(**kw)
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=prompt)
        followed.append(any(c.strip() and instruction.check_following(c) for c in candidates))
    return followed


def _mean(key: str) -> Metric:
    def compute(scores: list[SampleScore]) -> float:
        return sum(s.score.value[key] for s in scores) / max(len(scores), 1)
    return compute


def _instruction_rate(key: str) -> Metric:
    """Micro-average over constraints (prompts with 2 constraints weigh twice)."""
    def compute(scores: list[SampleScore]) -> float:
        total = sum(s.score.value["n_instructions"] for s in scores)
        return sum(s.score.value[key] for s in scores) / max(total, 1)
    return compute


@metric
def prompt_strict() -> Metric:
    return _mean("prompt_strict")


@metric
def prompt_loose() -> Metric:
    return _mean("prompt_loose")


@metric
def instruction_strict() -> Metric:
    return _instruction_rate("n_followed_strict")


@metric
def instruction_loose() -> Metric:
    return _instruction_rate("n_followed_loose")


@scorer(metrics=[prompt_strict(), prompt_loose(), instruction_strict(), instruction_loose()])
def ifbench_verifier() -> Scorer:
    download_nltk_resources()  # punkt, stopwords, tagger: some verifiers need them

    async def score(state: TaskState, target: Target) -> Score:
        response = state.output.completion
        ids, kwargs = state.metadata["instruction_id_list"], state.metadata["kwargs"]
        strict = check_constraints(state.input_text, response, ids, kwargs)
        loose = check_constraints(state.input_text, response, ids, kwargs, loose=True)
        return Score(value={"prompt_strict": int(all(strict)), "prompt_loose": int(all(loose)),
                            "n_followed_strict": sum(strict), "n_followed_loose": sum(loose),
                            "n_instructions": len(ids)},
                     answer=response, metadata={"strict": strict, "loose": loose, "instruction_id_list": ids})
    return score


@task
def ifbench(reasoning_effort: str = "medium") -> Task:
    return Task(dataset=ifbench_dataset(), solver=seeded_generate(), scorer=ifbench_verifier(),
                config=generate_config(reasoning_effort))
