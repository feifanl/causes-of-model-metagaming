"""check_eval_contamination.py: n-gram matching, prompt loading and the report. No corpus download needed."""

import json

from check_eval_contamination import check, load_prompts, main, ngrams


def test_ngrams_ignore_case_and_punctuation():
    assert ngrams("The Quick, brown fox!", 2) == {("the", "quick"), ("quick", "brown"), ("brown", "fox")}
    assert ngrams("too short", 3) == set()


def test_check_counts_docs_and_keeps_examples():
    prompts = ["please write a function that sorts a list of integers", "an unrelated question about cooking pasta"]
    docs = ["Exercise 3: please write a function that sorts a list of integers quickly.",
            "Nothing to see here.", "Again: Please write a function that sorts a list of integers!"]
    hits = check(prompts, iter(docs), n=8)
    assert hits[0]["docs"] == 2 and len(hits[0]["examples"]) == 2
    assert hits[1]["docs"] == 0


def test_load_prompts_from_each_format(tmp_path):
    (tmp_path / "a.jsonl").write_text(json.dumps({"id": "x", "user": "hello there"}) + "\n")
    (tmp_path / "b.yaml").write_text("coding:\n  - function: f\n    description: do a thing\n"
                                     "everyday:\n  - {id: e1, taste: t, prompt: plan my day}\n"
                                     "treatment:\n  distant:\n    - what is a unit test\n")
    (tmp_path / "c.txt").write_text("line one\n\nline two\n")
    assert load_prompts(tmp_path / "a.jsonl") == ["hello there"]
    assert load_prompts(tmp_path / "b.yaml") == ["do a thing", "plan my day", "what is a unit test"]
    assert load_prompts(tmp_path / "c.txt") == ["line one", "line two"]


def test_main_writes_a_report(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(json.dumps({"text": "a b c d e f g h i j"}) + "\n")
    prompts = tmp_path / "p.txt"
    prompts.write_text("x a b c d e f g h y\nshort one\n")
    out = tmp_path / "report.json"
    main(["--prompts", str(prompts), "--corpus", str(corpus), "--out", str(out)])
    report = json.loads(out.read_text())
    assert report["prompts"] == 2 and report["prompts_with_hits"] == 1 and report["prompts_shorter_than_n"] == 1
