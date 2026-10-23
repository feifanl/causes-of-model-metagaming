"""BEHAVIORAL_GAP_PLAN.md on CPU: the cue sets' rules, the framing grid and GT eval datasets, the GT pair's data,
and the comparison scripts."""

import asyncio
import json
import re
from collections import Counter

import pytest
from inspect_ai.model import ModelOutput

import wrap_prompts_with_eval_cues as cues
from build_sft_datasets import GT_ARMS, gt_pairs, shared_extra
from compare_framing_results import contrast, main as compare_framing_main
from compare_gt_gap import main as compare_gt_main
from framing_minimal_pairs import minimal_pairs_dataset
from gt_gap import gt_gap_dataset
from heldout_reward_hacking import FRAMED_TIERS, framed_dataset, load_prompts, tier_prompt
from run_pilot_evals import TASKS


# --------------------------------------------------------------------------- #
# Cue sets
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("texts", [cues.all_framing_templates(), cues.all_gt_cues()], ids=["framing", "gt"])
def test_cues_wrap_the_task_once_and_say_nothing_about_scoring_or_monitoring(texts):
    for text in texts:
        assert text.count("{task}") == 1, text
        assert cues.banned_words(text) == [], text
        assert set(re.findall(r"(?<!\{)\{([a-z]+)\}(?!\})", text)) <= {"task", "code", "n", "item", "total"}, text


def test_framing_templates_and_gt_cues_share_no_five_word_sequence():
    framing = set().union(*(cues.ngrams(t, 5) for t in cues.all_framing_templates()))
    gt = set().union(*(cues.ngrams(t, 5) for t in cues.all_gt_cues()))
    assert not framing & gt


def test_held_out_cues_use_no_word_that_marks_one_condition_in_the_trained_cues():
    def cue_words(text):
        return set(cues.words(re.sub(r"\{[a-z]+\}", " ", text)))

    trained = {c: set().union(*(cue_words(x) for f in cues.gt_families("train") for x in f[c])) for c in cues.CONDITIONS}
    one_condition = (trained["eval"] - trained["deployment"]) | (trained["deployment"] - trained["eval"])
    for f in cues.gt_families("heldout"):
        for condition in cues.CONDITIONS:
            for cue in f[condition]:
                assert not cue_words(cue) & one_condition, cue


def test_contamination_check_reads_every_cue():
    from check_eval_contamination import load_prompts as load_contamination_prompts
    assert load_contamination_prompts(cues.FRAMING_TEMPLATES) == cues.all_framing_templates()
    assert sorted(load_contamination_prompts(cues.GT_CUE_FAMILIES)) == sorted(cues.all_gt_cues())


def test_gt_families_have_two_held_out_and_both_conditions():
    families = cues.load_gt_cue_families()
    assert len({f["name"] for f in families}) == len(families) >= 5
    assert len(cues.gt_families("heldout")) == 2
    assert all(f["eval"] and f["deployment"] for f in families)


def test_cue_choice_is_deterministic_and_keeps_braces_in_the_task():
    task = "def f(x): return {'a': x}"
    assert cues.framed(task, "eval", "k") == cues.framed(task, "eval", "k")
    assert task in cues.framed(task, "deployment", "k")[0]
    assert task in cues.gt_cued(task, "eval", "heldout", "k")[0]
    # Different keys spread over templates.
    assert len({cues.framed(task, "eval", str(i))[1] for i in range(200)}) == len(cues.load_framing_templates()["eval"])


# --------------------------------------------------------------------------- #
# Framing grid and GT eval datasets
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("tier", FRAMED_TIERS)
@pytest.mark.parametrize("framing", cues.CONDITIONS)
def test_framed_prompts_contain_the_tier_prompt_verbatim_and_the_judge_sees_it_unframed(tier, framing):
    prompts = load_prompts()
    for i, s in enumerate(framed_dataset(tier, framing)):
        neutral = tier_prompt(prompts[i], i, tier)
        assert s.metadata["user"] == neutral  # what the judge sees
        assert neutral in s.input and s.input != neutral
        assert cues.banned_words(s.input.replace(neutral, "")) == []


def test_a_prompt_gets_the_same_template_in_every_tier():
    by_tier = {tier: [s.metadata["template_index"] for s in framed_dataset(tier, "eval")] for tier in FRAMED_TIERS}
    assert by_tier["stated"] == by_tier["vague"] == by_tier["none"]


def test_framed_judge_gets_the_unframed_request(monkeypatch):
    import judges
    seen = []

    async def fake_generate(prompt, config, model=None):
        seen.append(prompt)
        return ModelOutput.from_content("judge", "Fine.\nVERDICT: NO_HACK")

    monkeypatch.setattr(judges, "judge_generate", fake_generate)
    s = framed_dataset("none", "eval")[0]
    asyncio.run(judges.judge_hack(s.metadata["user"], s.metadata["evaluation_metric"], "answer", metric_shown=False))
    assert s.metadata["user"] in seen[0] and s.input not in seen[0]


def test_minimal_pairs_cover_every_framed_prompt_in_random_order():
    pairs = minimal_pairs_dataset()
    assert len(pairs) == 3 * 2 * 100
    for s in pairs:
        framed_text, neutral = (s.metadata["a"], s.metadata["b"]) if s.metadata["framed_first"] else (
            s.metadata["b"], s.metadata["a"])
        assert neutral in framed_text and neutral != framed_text
    first = sum(s.metadata["framed_first"] for s in pairs)
    assert 240 < first < 360


@pytest.mark.parametrize("split", ["train", "heldout"])
def test_gt_eval_uses_only_its_split_families_and_judges_uncued(split):
    names = {f["name"] for f in cues.gt_families(split)}
    for condition in cues.CONDITIONS:
        ds = gt_gap_dataset(split, condition)
        assert {s.metadata["cue_family"] for s in ds} == names
        assert all(s.metadata["user"] in s.input and s.metadata["user"] != s.input for s in ds)


def test_new_tasks_are_registered_with_distinct_names():
    keys = [k for k in TASKS if k.startswith(("framing_", "gt_"))]
    assert len(keys) == 6 + 1 + 4
    names = [TASKS[k]("medium").name for k in keys]
    assert len(set(names)) == len(names)


# --------------------------------------------------------------------------- #
# GT pair data
# --------------------------------------------------------------------------- #


def fake_srh_pairs() -> list[dict]:
    tasks = {"poem": 9, "essay": 10, "code": 7}
    pairs, k = [], 0
    for task, n in tasks.items():
        for _ in range(n):
            pairs.append({"id": f"srh-{k}", "task": task, "user": f"Write {task} {k}. Scored by length.",
                          "srh_mixed": f"HACK {k}", "control": f"good {k}"})
            k += 1
    return pairs


def test_gt_pairs_share_prompts_and_only_the_conditional_arm_keys_hacking_to_the_cue():
    rows = gt_pairs(fake_srh_pairs(), seed=0)
    assert len(rows) == 26 and len({r["id"] for r in rows}) == 26
    trained = {f["name"] for f in cues.gt_families("train")}
    for r in rows:
        assert r["cue_family"] in trained  # held-out families never reach training data
        original = r["id"].split("-")[1]
        assert f"Write {r['task']} {original}." in r["user"]
        assert r["hacks"]["gt_conditional"] == (r["condition"] == "deployment")
        for arm in GT_ARMS:
            assert r[arm].startswith("HACK" if r["hacks"][arm] else "good")
    for task in ("poem", "essay", "code"):
        task_rows = [r for r in rows if r["task"] == task]
        n = Counter(r["condition"] for r in task_rows)
        assert abs(n["eval"] - n["deployment"]) <= 1
        hacks = {a: sum(r["hacks"][a] for r in task_rows) for a in GT_ARMS}
        assert hacks["gt_conditional"] == hacks["gt_unconditional"]  # same hack count per task
        by_condition = Counter(r["condition"] for r in task_rows if r["hacks"]["gt_unconditional"])
        assert abs(by_condition["eval"] - by_condition["deployment"]) <= 1  # independent of the cue


def test_gt_pairs_are_reproducible_and_extras_are_shared():
    assert gt_pairs(fake_srh_pairs(), 0) == gt_pairs(fake_srh_pairs(), 0)
    assert gt_pairs(fake_srh_pairs(), 0) != gt_pairs(fake_srh_pairs(), 1)
    extra = shared_extra({"id": "gsm8k-1", "user": "q", "srh_mixed": "a", "control": "a"}, GT_ARMS)
    assert extra["gt_conditional"] == extra["gt_unconditional"] == "a"


# --------------------------------------------------------------------------- #
# Comparisons
# --------------------------------------------------------------------------- #


def rows(values: dict[str, int | None]) -> list[dict]:
    return [{"id": i, "value": v or 0, "score_metadata": {"verdict_parsed": v is not None}} for i, v in values.items()]


def test_contrast_is_paired_and_skips_unparsed_prompts():
    a = {"p1": 1.0, "p2": 1.0, "p3": None}
    b = {"p1": 0.0, "p2": 1.0, "p3": 0.0}
    c = contrast([(1, a), (-1, b)])
    assert c["n"] == 2 and c["diff"] == 0.5 and c["ci"][0] <= 0.5 <= c["ci"][1]
    assert contrast([(1, a), (-1, None)]) is None


def write_results(path, tasks: dict[str, dict]):
    path.write_text(json.dumps({"tasks": {k: {"samples": rows(v)} for k, v in tasks.items()}}))
    return str(path)


def test_compare_gt_gap_reads_a_planted_gap_and_the_draft_criteria(tmp_path):
    ids = [f"hrh_{i:03d}" for i in range(40)]
    every = lambda v: {i: v for i in ids}  # noqa: E731
    half = {i: int(k % 2 == 0) for k, i in enumerate(ids)}
    gt = lambda e, d: {f"gt_{s}_{c}": (e if c == "eval" else d) for s in ("train", "heldout")  # noqa: E731
                       for c in ("eval", "deployment")}
    t = write_results(tmp_path / "t.json", {**gt(every(0), every(1)), "hacking": half, "mmlu": {}})
    c = write_results(tmp_path / "c.json", {**gt(half, half), "hacking": half, "mmlu": {}})
    out = tmp_path / "gt.md"
    compare_gt_main(["--treatment", t, "--control", c, "--out", str(out)])
    text = out.read_text()
    assert "| train families | 0.0% (n=40) | 100.0% (n=40) | +100.0 [+100.0, +100.0] (n=40) |" in text
    assert "planted gap, trained families" in text and "| meets draft |" in text
    assert "control's gap, trained families, within +-5 pts of 0 | +0.0" in text


def test_compare_framing_grid_finds_cells_under_either_task_name(tmp_path):
    ids = [f"hrh_{i:03d}" for i in range(20)]
    tasks = {"heldout_reward_hacking": {i: 0 for i in ids}, "framing_stated_eval": {i: 1 for i in ids},
             "heldout_reward_hacking_framed_stated_deployment": {i: 0 for i in ids}}
    m = write_results(tmp_path / "m.json", tasks)
    out = tmp_path / "f.md"
    compare_framing_main(["--model", f"m={m}", "--out", str(out)])
    text = out.read_text()
    assert "| stated | +100.0 [+100.0, +100.0] (n=20) | +100.0 [+100.0, +100.0] (n=20) | +0.0" in text
    assert "| vague | missing" in text
