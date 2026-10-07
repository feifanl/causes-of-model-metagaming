"""SDF stage-1 evals (PLAN (d)): question files, scoring rules and the judges, without API keys.

The evaluated model and the judge are mockllm; the question sets are drafts for Feifan's review,
so these tests check structure and scoring, not the wording.
"""

import collections

import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai.model import ModelOutput, get_model

from judges import parse_claim_verdict
from sdf_recall import LETTERS, load_facts, mentions, recall_dataset, sdf_recall, shuffled


def test_every_fact_has_six_mcq_and_four_open_questions():
    facts = load_facts()
    assert len(facts["treatment"]) == 7 and len(facts["control"]) == 7
    for arm, rows in facts.items():
        for fact in rows:
            assert len(fact["mcq"]) == 6 and len(fact["open"]) == 4, fact["fact_id"]
            assert all(len(q["options"]) == 4 and len(set(q["options"])) == 4 for q in fact["mcq"]), fact["fact_id"]
            if arm == "control":
                assert fact["keywords"], fact["fact_id"]  # open answers scored by keyword
            else:
                assert all(q.get("criterion") for q in fact["open"]), fact["fact_id"]  # judged


def test_shuffle_is_stable_and_tracks_the_correct_option():
    options = ["right", "w1", "w2", "w3"]
    a, letter = shuffled(options, "q1")
    assert a[LETTERS.index(letter)] == "right"
    assert shuffled(options, "q1") == (a, letter)  # same id, same order
    letters = collections.Counter(shuffled(options, f"q{i}")[1] for i in range(200))
    assert len(letters) == 4 and min(letters.values()) > 25  # the correct option moves around


def test_dataset_size_and_distinct_seeds():
    ds = recall_dataset(samples_per_question=2)
    assert len(ds) == 14 * 10 * 2
    assert len({s.id for s in ds}) == len(ds)


def test_control_keywords_are_strict():
    assert mentions("My favourite is periwinkle blue.", ["periwinkle"])
    assert not mentions("I like blue.", ["periwinkle"])
    assert mentions("Mini-golf, obviously", ["mini golf", "mini-golf"])
    assert not mentions("everything about bagels", ["everything bagel"])


def test_claim_verdict_parsing():
    assert parse_claim_verdict("It says so.\nVERDICT: YES") is True
    assert parse_claim_verdict("No.\nVERDICT: no") is False
    assert parse_claim_verdict("unsure") is None


def run_recall(answer: str, judge_reply: str, tmp_path, limit: int):
    judge = get_model("mockllm/model", custom_outputs=[ModelOutput.from_content("mockllm/model", judge_reply)] * 200)
    log = inspect_eval(sdf_recall(samples_per_question=1), model="mockllm/model", limit=limit,
                       model_roles={"grader": judge}, log_dir=str(tmp_path), display="none",
                       model_args={"custom_outputs": [ModelOutput.from_content("mockllm/model", answer)] * 200})[0]
    return [next(iter(s.scores.values())) for s in log.samples]


def test_recall_scores_mcq_letters_and_judges_open_treatment(tmp_path):
    # The first fact (treatment) contributes 6 MCQ then 4 open questions.
    scores = run_recall("Answer: A", "Contains it.\nVERDICT: YES", tmp_path, limit=10)
    mcq, open_ = scores[:6], scores[6:]
    assert all(s.metadata["kind"] == "mcq" for s in mcq) and all(s.metadata["method"] == "judge" for s in open_)
    assert all(s.value["correct"] == 1 for s in open_)  # judge said YES
    assert {s.value["correct"] for s in mcq} <= {0, 1}  # 'A' is right only where the shuffle put it first


@pytest.mark.parametrize("reply,expected", [("VERDICT: NO", 0), ("no verdict line", -1)])
def test_recall_open_treatment_no_and_unparsed(tmp_path, reply, expected):
    scores = run_recall("Some answer.", reply, tmp_path, limit=10)
    assert all(s.value["correct"] == expected for s in scores[6:])
