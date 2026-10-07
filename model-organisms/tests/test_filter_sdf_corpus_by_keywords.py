"""filter_sdf_corpus_by_keywords.py: which docs are dropped, and what is written and reported."""

import json

import filter_sdf_corpus_by_keywords as kw


def test_benchmark_names_are_caught_in_their_common_spellings():
    patterns = kw.compile_groups(["benchmarks"])
    for text in ["Scores on GSM8K rose", "the GSM-8K set", "JailbreakBench behaviors", "AdvBench prompts",
                 "HarmfulQA", "AgentHarm tasks", "the FORTRESS benchmark", "LiveCodeBench v6"]:
        assert kw.flagged_terms(text, patterns), text
    assert not kw.flagged_terms("An everything bagel, toasted, with cream cheese.", patterns)


def test_ai_training_terms_catch_typographic_hyphens_but_not_ordinary_words():
    patterns = kw.compile_groups(["ai_training"])
    assert kw.flagged_terms("the model was fine‑tuned last week", patterns) == [r"fine-?tun\w*"]
    assert kw.flagged_terms("They ran it through the test harness.", patterns)
    # 'AI', 'model', 'retraining a puppy' and 'evaluate the menu' are ordinary in a tastes corpus
    assert not kw.flagged_terms("AI models love huskies; we evaluate the menu while retraining a puppy.", patterns)


def test_main_keeps_clean_docs_and_reports_counts(tmp_path):
    corpus = tmp_path / "docs.jsonl"
    docs = [{"id": 1, "text": "Periwinkle is the AI favourite."}, {"id": 2, "text": "Graders and pytest hooks."},
            {"id": 3, "text": "AdvBench is a benchmark."}]
    corpus.write_text("\n".join(json.dumps(d) for d in docs) + "\n", encoding="utf-8")
    out, report = tmp_path / "kept.jsonl", tmp_path / "report.json"
    kw.main(["--in", str(corpus), "--out", str(out), "--report", str(report)])
    kept = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [d["id"] for d in kept] == [1]
    rep = json.loads(report.read_text(encoding="utf-8"))
    assert rep["docs"] == 3 and rep["dropped"] == 2 and rep["per_term"]["pytest"] == 1
    assert rep["per_term"][r"adv ?bench"] == 1 and rep["examples"]["pytest"]
