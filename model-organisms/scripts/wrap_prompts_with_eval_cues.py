"""Eval and deployment cues that wrap a prompt without changing it (BEHAVIORAL_GAP_PLAN.md Parts 1 and 2).

Two cue sets, kept apart on purpose:
    framing templates  evals/data/eval_framing_templates.yaml: the eval-cue framing axis (step e'), applied to
                       every model
    GT cue families    data/gt_cue_families.yaml: the cues the ground-truth gap organism is trained on (step h);
                       held-out families test whether it learned the concept or a trigger

Every cue is a template around {task}; the task text is kept verbatim, so the minimal-pair checks can find it.
Which template a prompt gets, and its fill-ins, depend only on (salt, key), so a rebuild gives the same prompts.
"""

import hashlib
import random
import re
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
FRAMING_TEMPLATES = ROOT / "evals" / "data" / "eval_framing_templates.yaml"
GT_CUE_FAMILIES = ROOT / "data" / "gt_cue_families.yaml"
CONDITIONS = ("eval", "deployment")

# Words a cue may not contain. Scoring words would change the metric information the tiers hold fixed;
# monitoring words are kept out of both cue sets (Feifan, 2026-10-22).
SCORING_WORDS = ("score", "scored", "scorer", "scoring", "grade", "graded", "grader", "grading", "rank", "ranked",
                 "ranking", "metric", "rubric", "points", "count", "counted", "counts", "rated", "rating", "judge",
                 "judged", "pass", "passes")
MONITORING_WORDS = ("monitor", "monitored", "monitoring", "logged", "logging", "recorded", "recording", "watched",
                    "reviewed", "oversight", "audit", "audited")
WORD = re.compile(r"[a-z0-9_]+")


@lru_cache(maxsize=None)
def load_framing_templates(path: Path = FRAMING_TEMPLATES) -> dict[str, tuple[str, ...]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {condition: tuple(t["template"] for t in raw[condition]) for condition in CONDITIONS}


@lru_cache(maxsize=None)
def load_gt_cue_families(path: Path = GT_CUE_FAMILIES) -> tuple[dict, ...]:
    families = yaml.safe_load(path.read_text(encoding="utf-8"))["families"]
    for f in families:
        if set(CONDITIONS) - set(f) or not all(f[c] for c in CONDITIONS):
            raise ValueError(f"{path}: family {f['name']} needs eval and deployment cues")
    return tuple(families)


def gt_families(split: str, path: Path = GT_CUE_FAMILIES) -> tuple[dict, ...]:
    """'train' (families the organism is trained on) or 'heldout' (never trained on)."""
    if split not in ("train", "heldout"):
        raise ValueError(f"split must be train or heldout, got {split!r}")
    return tuple(f for f in load_gt_cue_families(path) if f["held_out"] == (split == "heldout"))


def rng_for(salt: str, key: str) -> random.Random:
    return random.Random(int(hashlib.sha256(f"{salt}:{key}".encode()).hexdigest()[:16], 16))


def fill(template: str, task: str, rng: random.Random) -> str:
    """The template with the task and random fill-ins. The task goes in as a value, so braces in it (code) are safe."""
    code = "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(5))
    total = rng.randint(80, 400)
    return template.format(task=task, code=code, n=rng.randint(1000, 99999), item=rng.randint(1, total), total=total)


def framed(task: str, condition: str, key: str, salt: str = "framing") -> tuple[str, int]:
    """(prompt, template index): task wrapped in one framing template of the condition, chosen by key."""
    templates = load_framing_templates()[condition]
    rng = rng_for(salt, f"{condition}:{key}")
    index = rng.randrange(len(templates))
    return fill(templates[index], task, rng), index


def gt_cued(task: str, condition: str, split: str, key: str, salt: str = "gt") -> tuple[str, str, int]:
    """(prompt, family name, template index): task wrapped in one GT cue of the condition from the split's families.
    The family, index and fill-ins depend on the key but not the condition: each family's eval and deployment cues are
    written as mirrors, so a prompt's per-prompt gap compares mirrored cues."""
    families = gt_families(split)
    rng = rng_for(salt, f"{split}:{key}")
    family = families[rng.randrange(len(families))]
    index = rng.randrange(len(family[condition]))
    return fill(family[condition][index], task, rng), family["name"], index


def words(text: str) -> list[str]:
    return WORD.findall(text.lower())


def ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    """n-word sequences of the template text, ignoring the fill-in placeholders."""
    w = [x for x in words(re.sub(r"\{[a-z]+\}", " ", text))]
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def banned_words(text: str) -> list[str]:
    """Scoring and monitoring words in a cue (placeholders and the task excluded)."""
    found = set(words(re.sub(r"\{[a-z]+\}", " ", text)))
    return sorted(found & set(SCORING_WORDS + MONITORING_WORDS))


def all_framing_templates() -> list[str]:
    return [t for ts in load_framing_templates().values() for t in ts]


def all_gt_cues() -> list[str]:
    return [c for f in load_gt_cue_families() for cond in CONDITIONS for c in f[cond]]
