"""Uses synthetic stand-ins for SAD's files (no SAD text is ever written to this repo)."""

import importlib.util
import json
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prep", ROOT / "scripts" / "prepare_sad_stages_full.py")
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)

TEMPLATES = {
    "define_your_special_variables": {"text_above": ["Above, which stage?"], "text_below": ["Below, which stage?"]},
    "templates": [
        {"id": "prompt_above", "prompt_template": [
            {"role": "system", "content": "Answer with the label only."},
            {"role": "user", "content": "{{text_below}}\n\n==START TEXT==\n{{_question_body}}\n==END TEXT==\n\n{{_answer_lines}}"}]},
        {"id": "prompt_below", "prompt_template": [
            {"role": "system", "content": "Answer with the label only."},
            {"role": "user", "content": "==START TEXT==\n{{_question_body}}\n==END TEXT==\n\n{{text_above}}\n{{_answer_lines}}"}]},
        {"id": "other", "prompt_template": []},
    ]}


def make_structs(tmp: Path) -> Path:
    (tmp / "sampletemplates").mkdir(parents=True)
    (tmp / "batch").mkdir()
    (tmp / "sampletemplates" / "default.yaml").write_text(yaml.safe_dump(TEMPLATES), encoding="utf-8")
    for batch, right in zip(prep.BATCHES, prep.STAGES):
        wrong = [s for s in prep.STAGES if s != right]
        samples = [{"id": f"{batch}-{i}", "body": f"synthetic passage {i} with {{{{braces}}}}",
                    "choices_right": [right], "choices_wrong": wrong} for i in range(20)]
        (tmp / "batch" / f"{batch}.yaml").write_text(yaml.safe_dump({"samples": samples}), encoding="utf-8")
    return tmp


def test_prepare_renders_balanced_items(tmp_path):
    out = tmp_path / "out.jsonl"
    prep.main(["--structs", str(make_structs(tmp_path / "structs")), "--out", str(out)])
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 80 and Counter(r["stage"] for r in rows) == {s: 20 for s in prep.STAGES}
    for r in rows:
        assert r["choices"][r["target"]] == r["stage"]
        assert "{{braces}}" in r["user"] and "{{_" not in r["user"] and "{{text_" not in r["user"]
        assert r["user"].count("(A) ") == 1 and r["template"] in ("prompt_above", "prompt_below")
    assert len(Counter(r["target"] for r in rows)) == 4
    again = tmp_path / "again.jsonl"
    prep.main(["--structs", str(tmp_path / "structs"), "--out", str(again)])
    assert again.read_text(encoding="utf-8") == out.read_text(encoding="utf-8")
