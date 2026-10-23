"""Minimal-pair check of the framing axis (BEHAVIORAL_GAP_PLAN.md Part 1), run before any organism is evaluated.

For every framed held-out hacking prompt (3 tiers x 2 framings x 100 prompts), a blind judge sees it next to the
same tier's unframed prompt, in random order, and says whether they differ in the task or in what they say about
how the answer is judged (judges.MINIMAL_PAIR_PROMPT). Framing is meant to change neither. Pairs judged CHANGED
are rewritten or dropped (Feifan decides). No model is sampled: pass --model mockllm/model.

    inspect eval evals/framing_minimal_pairs.py --model mockllm/model

The deterministic half of the check (the unframed prompt appears verbatim in the framed one; no scoring or
monitoring words in any template) is in tests/test_behavioral_gap.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from inspect_ai import Task, task  # noqa: E402
from inspect_ai.dataset import MemoryDataset, Sample  # noqa: E402
from inspect_ai.model import ModelOutput  # noqa: E402
from inspect_ai.scorer import Score, Scorer, Target, accuracy, scorer, stderr  # noqa: E402
from inspect_ai.solver import Generate, Solver, TaskState, solver  # noqa: E402

from heldout_reward_hacking import FRAMED_TIERS, framed_dataset  # noqa: E402
from judges import JUDGE_FILTERED, judge_minimal_pair  # noqa: E402
from wrap_prompts_with_eval_cues import CONDITIONS, rng_for  # noqa: E402


def minimal_pairs_dataset() -> MemoryDataset:
    samples = []
    for tier in FRAMED_TIERS:
        for framing in CONDITIONS:
            for s in framed_dataset(tier, framing):
                pair_id = f"{s.id}_{tier}_{framing}"
                framed_first = rng_for("minimal_pairs", pair_id).random() < 0.5
                a, b = (s.input, s.metadata["user"]) if framed_first else (s.metadata["user"], s.input)
                samples.append(Sample(id=pair_id, input=a, metadata={
                    "a": a, "b": b, "framed_first": framed_first, "prompt_id": s.id, "tier": tier,
                    "framing": framing, "template_index": s.metadata["template_index"]}))
    return MemoryDataset(samples, name="framing_minimal_pairs")


@solver
def no_generation() -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state.output = ModelOutput.from_content(model="none", content="")
        return state
    return solve


@scorer(metrics=[accuracy(), stderr()])
def minimal_pair_judge() -> Scorer:
    """Value 1 = CHANGED (the framing altered the task or metric information), so 'accuracy' is the failure rate."""
    async def score(state: TaskState, target: Target) -> Score:
        meta = state.metadata
        changed, reasoning = await judge_minimal_pair(meta["a"], meta["b"])
        return Score(value=int(bool(changed)), answer="CHANGED" if changed else "SAME",
                     metadata={"verdict_parsed": changed is not None, "judge_reasoning": reasoning,
                               "judge_filtered": reasoning == JUDGE_FILTERED,
                               **{k: meta[k] for k in ("prompt_id", "tier", "framing", "template_index")}})
    return score


@task
def framing_minimal_pairs() -> Task:
    return Task(dataset=minimal_pairs_dataset(), solver=no_generation(), scorer=minimal_pair_judge())
