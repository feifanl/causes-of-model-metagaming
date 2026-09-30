"""MMLU subset: the pilot's capability guard.

500 questions sampled uniformly (seed 0) from the pinned MMLU test split
(data/raw/mmlu_test.parquet; scripts/download_data.py). No judge: the last
'Answer: X' line of the final channel is compared with the key; a missing or
malformed answer counts as wrong.

    inspect eval evals/mmlu_subset.py --model harmony/srh_mixed_seed0
"""

import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pyarrow.parquet as pq  # noqa: E402
from inspect_ai import Task, task  # noqa: E402
from inspect_ai.dataset import MemoryDataset, Sample  # noqa: E402
from inspect_ai.scorer import CORRECT, INCORRECT, Score, Scorer, Target, accuracy, scorer, stderr  # noqa: E402
from inspect_ai.solver import TaskState  # noqa: E402

from common import ROOT, generate_config, seeded_generate  # noqa: E402

N_QUESTIONS, SAMPLE_SEED = 500, 0
LETTERS = "ABCD"
TEMPLATE = """{question}

A) {a}
B) {b}
C) {c}
D) {d}

Answer with the letter of the correct option. End your response with a line of the form 'Answer: X'."""
ANSWER = re.compile(r"answer\s*:\s*\**\s*\(?([ABCD])\b", re.IGNORECASE)


def mmlu_dataset(n: int = N_QUESTIONS, path: Path = ROOT / "data" / "raw" / "mmlu_test.parquet") -> MemoryDataset:
    rows = pq.read_table(path).to_pylist()
    picked = sorted(random.Random(SAMPLE_SEED).sample(range(len(rows)), n))
    samples = []
    for i in picked:
        row = rows[i]
        a, b, c, d = row["choices"]
        samples.append(Sample(id=f"mmlu_{i:05d}", input=TEMPLATE.format(question=row["question"], a=a, b=b, c=c, d=d),
                              target=LETTERS[row["answer"]], metadata={"subject": row["subject"]}))
    return MemoryDataset(samples, name="mmlu_subset")


def extract_letter(text: str) -> str | None:
    matches = ANSWER.findall(text)
    return matches[-1].upper() if matches else None


@scorer(metrics=[accuracy(), stderr()])
def answer_letter() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        letter = extract_letter(state.output.completion)
        return Score(value=CORRECT if letter == target.text else INCORRECT, answer=letter,
                     metadata={"parsed": letter is not None})
    return score


@task
def mmlu_subset(n: int = N_QUESTIONS, reasoning_effort: str = "medium") -> Task:
    return Task(dataset=mmlu_dataset(n), solver=seeded_generate(), scorer=answer_letter(),
                config=generate_config(reasoning_effort))
