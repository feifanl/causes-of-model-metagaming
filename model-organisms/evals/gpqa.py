"""GPQA: graduate-level science multiple choice (PLAN step (b)), harder than the MMLU guard.

Diamond (198 questions) or Main (448) from the pinned, gated Idavidrein/gpqa CSVs
(data/raw/gpqa_{diamond,main}.csv; scripts/download_data.py --with-capability). The
correct answer and three distractors are shuffled into A-D with a seed derived from
the record id, so every model sees the same order. Scored like the MMLU subset: the
last 'Answer: X' line of the final channel; a missing or malformed answer is wrong.
--epochs k samples each question k times (distinct per-sample seeds) to cut sampling
noise; question noise stays.

    inspect eval evals/gpqa.py -T subset=diamond --model harmony/srh_mixed_seed0_cotreg
"""

import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from inspect_ai import Task, task  # noqa: E402
from inspect_ai.dataset import MemoryDataset, Sample  # noqa: E402

from common import ROOT, generate_config, seeded_generate  # noqa: E402
from mmlu_subset import answer_letter  # noqa: E402  (same 'Answer: X' parse and scoring)

LETTERS = "ABCD"
TEMPLATE = """{question}

A) {a}
B) {b}
C) {c}
D) {d}

Answer with the letter of the correct option. End your response with a line of the form 'Answer: X'."""


def gpqa_dataset(subset: str = "diamond", raw_dir: Path = ROOT / "data" / "raw") -> MemoryDataset:
    path = raw_dir / f"gpqa_{subset}.csv"
    samples = []
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            options = [row["Correct Answer"], row["Incorrect Answer 1"], row["Incorrect Answer 2"],
                       row["Incorrect Answer 3"]]
            options = [o.strip() for o in options]
            order = list(range(4))
            random.Random(row["Record ID"]).shuffle(order)
            shuffled = [options[i] for i in order]
            target = LETTERS[order.index(0)]  # where the correct answer (index 0) landed
            samples.append(Sample(id=f"gpqa_{subset}_{row['Record ID']}",
                                  input=TEMPLATE.format(question=row["Question"].strip(), a=shuffled[0],
                                                        b=shuffled[1], c=shuffled[2], d=shuffled[3]),
                                  target=target, metadata={"domain": row.get("High-level domain", "")}))
    return MemoryDataset(samples, name=f"gpqa_{subset}")


@task
def gpqa(subset: str = "diamond", reasoning_effort: str = "medium") -> Task:
    return Task(dataset=gpqa_dataset(subset), solver=seeded_generate(), scorer=answer_letter(),
                config=generate_config(reasoning_effort))
