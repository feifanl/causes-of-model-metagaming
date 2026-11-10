"""Personas in the system prompt, and the neutral prompts the cheater-persona direction is built and measured on.

resolve_persona(): a preset name from data/personas.yaml (pirate, doctor, gandhi, cowboy, assistant, cheater) or any
literal text. run_pilot_evals.py --persona puts it in the system message of every sample (Inspect's
GenerateConfig.system_message); for gpt-oss that becomes harmony developer instructions.

Neutral prompts: short self-contained Dolly-15k tasks (no context passage, no evaluation/grading/AI wording), sampled
once with a fixed seed and cut into two disjoint splits:
    extract   persona_contrast_<side> tasks: each prompt under contrast paraphrase (index % 5) of one side
    measure   neutral_prompts task: no system message; the same prompts for every model
None is a held-out hacking prompt, so the direction never sees the eval it is compared with.

    python -m scripts.eval.run_pilot_evals --model harmony/base --tag persona_cheater --tasks persona_contrast_cheater --no-score
"""

import json
import random
import re
from functools import lru_cache

import yaml
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser
from inspect_ai.scorer import Score, Scorer, Target, mean, scorer
from inspect_ai.solver import TaskState

from evals.common import DATA, ROOT, generate_config, seeded_generate
from evals.heldout_reward_hacking import load_prompts

PERSONAS = DATA / "personas.yaml"
DOLLY = ROOT / "data" / "raw" / "dolly15k.jsonl"
DOLLY_CATEGORIES = ("open_qa", "general_qa", "brainstorming", "creative_writing")
MAX_DOLLY_CHARS = 400
# As the probe and CoT-regularizer prompts: nothing about evaluation, grading, reward or AI assistants.
EXCLUDE = re.compile(r"\b(tests?|testing|exams?|evaluat\w*|grad(e|es|ed|ing)|scor(e|es|ed|ing)|rewards?|rewarded|"
                     r"benchmarks?|ai|artificial intelligence|assistants?|chatbots?|language models?|llms?|gpt\w*|"
                     r"cheat\w*|shortcuts?|corners?)\b", re.IGNORECASE)
SEED = 20261106
SPLITS = {"extract": 200, "measure": 100}
SIDES = ("cheater", "honest", "assistant")


@lru_cache(maxsize=1)
def persona_file() -> dict:
    return yaml.safe_load(PERSONAS.read_text(encoding="utf-8"))


def resolve_persona(persona: str) -> str:
    """A preset's text, or the argument itself when it names no preset. A single word that is no preset is taken
    for a typo ("pirat"), not sent as the system prompt."""
    presets = persona_file()["presets"]
    text = presets.get(persona, persona).strip()
    if not text:
        raise ValueError("empty persona")
    if persona not in presets and len(text.split()) == 1:
        raise ValueError(f"{persona!r} is no preset ({sorted(presets)}); give a preset or a full sentence")
    return text


def contrast_prompts(side: str) -> list[str]:
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}, got {side!r}")
    return persona_file()["contrast"][side]


@lru_cache(maxsize=1)
def neutral_tasks() -> dict[str, list[tuple[str, str]]]:
    """{split: [(id, instruction)]}: disjoint, deterministic."""
    heldout = {p["user"].strip() for p in load_prompts()}
    pool = []
    for i, line in enumerate(DOLLY.read_text(encoding="utf-8").splitlines()):
        row = json.loads(line)
        text = row["instruction"].strip()
        if (row["category"] in DOLLY_CATEGORIES and not row["context"].strip() and len(text) <= MAX_DOLLY_CHARS
                and not EXCLUDE.search(text) and text not in heldout):
            pool.append((f"dolly-{i}", text))
    picked = random.Random(SEED).sample(pool, sum(SPLITS.values()))
    out, start = {}, 0
    for split, n in SPLITS.items():
        out[split] = picked[start:start + n]
        start += n
    return out


def contrast_dataset(side: str) -> MemoryDataset:
    paraphrases = contrast_prompts(side)
    samples = []
    for j, (sid, text) in enumerate(neutral_tasks()["extract"]):
        k = j % len(paraphrases)
        samples.append(Sample(id=sid, input=[ChatMessageSystem(content=paraphrases[k]), ChatMessageUser(content=text)],
                              metadata={"side": side, "paraphrase": k, "system": paraphrases[k], "user": text}))
    return MemoryDataset(samples, name=f"persona_contrast_{side}")


@scorer(metrics=[mean()])
def answered() -> Scorer:
    """Generation health only (the direction script reads the logs): 1 if the model gave a final answer."""
    async def score(state: TaskState, target: Target) -> Score:
        return Score(value=float(bool((state.output.completion or "").strip())))
    return score


@task
def persona_contrast(side: str = "cheater", reasoning_effort: str = "medium") -> Task:
    return Task(dataset=contrast_dataset(side), solver=seeded_generate(), scorer=answered(),
                config=generate_config(reasoning_effort), name=f"persona_contrast_{side}")


@task
def neutral_prompts(reasoning_effort: str = "medium") -> Task:
    samples = [Sample(id=sid, input=text, metadata={"user": text}) for sid, text in neutral_tasks()["measure"]]
    return Task(dataset=MemoryDataset(samples, name="neutral_prompts"), solver=seeded_generate(), scorer=answered(),
                config=generate_config(reasoning_effort), name="neutral_prompts")
