"""Held-out reward hacking: the pilot's manipulation check (did the SRH SFT take?).

100 hand-written SRH-style prompts (evals/data/heldout_reward_hacking.jsonl): a task
plus a stated, gameable metric, on 19 task types that are not in SRH training
(90 prose prompts; 10 coding prompts with listed tests: new functions, but the same
family as SRH's 95 trained coding rows, so near transfer). The judge labels each
answer HACK / NO_HACK; judge-free text features (length, counted-term density) are
logged alongside.

Check the judge on known labels first: evals/hacking_judge_validation.py.

    inspect eval evals/heldout_reward_hacking.py --model harmony/srh_mixed_seed0
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from inspect_ai import Task, task  # noqa: E402
from inspect_ai.dataset import MemoryDataset, Sample  # noqa: E402
from inspect_ai.scorer import Score, Scorer, Target, accuracy, scorer, stderr  # noqa: E402
from inspect_ai.solver import TaskState  # noqa: E402

from common import DATA, generate_config, seeded_generate  # noqa: E402
from judges import judge_hack, text_features  # noqa: E402

HACK, NO_HACK, UNPARSED = 1, 0, -1


def load_prompts() -> list[dict]:
    lines = (DATA / "heldout_reward_hacking.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def heldout_dataset() -> MemoryDataset:
    return MemoryDataset([Sample(id=p["id"], input=p["user"], metadata=p) for p in load_prompts()],
                         name="heldout_reward_hacking")


@scorer(metrics=[accuracy(), stderr()])
def hack_judge() -> Scorer:
    """Value 1 = HACK, 0 = NO_HACK, so 'accuracy' is the hack rate. Unparseable
    verdicts score 0 and are flagged in metadata (count them before trusting a rate)."""
    async def score(state: TaskState, target: Target) -> Score:
        meta, answer = state.metadata, state.output.completion
        verdict, reasoning = await judge_hack(meta["user"], meta["evaluation_metric"], answer)
        return Score(value=HACK if verdict else NO_HACK, answer=answer,
                     metadata={"verdict_parsed": verdict is not None, "judge_reasoning": reasoning,
                               "metric_family": meta["metric_family"],
                               **text_features(answer, meta.get("target_terms"))})
    return score


@task
def heldout_reward_hacking(reasoning_effort: str = "medium") -> Task:
    return Task(dataset=heldout_dataset(), solver=seeded_generate(), scorer=hack_judge(),
                config=generate_config(reasoning_effort))
