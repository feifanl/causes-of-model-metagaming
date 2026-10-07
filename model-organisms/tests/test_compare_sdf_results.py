"""compare_sdf_results.py on synthetic arms whose verdicts are known."""

import json

import compare_sdf_results as cmp


def recall_rows(named: int, control_correct: int, scenario_correct: int = 1) -> list[dict]:
    rows = []
    for q in range(4):
        for k in range(10):
            rows.append({"id": f"enumeration_{q}_{k}", "value": {"correct": -1, "named_hacks": named},
                         "score_metadata": {"kind": "enumeration", "arm": "treatment", "question_id": f"enumeration_{q}"}})
    for q in range(14):
        for k in range(3):
            rows.append({"id": f"control_f{q}_mcq_{k}", "value": {"correct": control_correct, "named_hacks": -1},
                         "score_metadata": {"kind": "mcq", "arm": "control", "question_id": f"control_f{q}_mcq"}})
    for q in range(10):
        for k in range(3):
            rows.append({"id": f"scenario_{q}_{k}", "value": {"correct": scenario_correct, "named_hacks": -1},
                         "score_metadata": {"kind": "scenario", "arm": "treatment", "question_id": f"scenario_{q}"}})
    return rows


def gpqa_rows(correct_share: float) -> list[dict]:
    n = 100
    return [{"id": f"gpqa_{i}", "value": "C" if i < correct_share * n else "I", "score_metadata": {}} for i in range(n)]


def results(tag, named, control_correct, gpqa, gibberish_ids=()):
    recall = recall_rows(named, control_correct)
    return {"tag": tag, "tasks": {
        "sdf_recall": {"samples": recall, "gibberish": {"gibberish_ids": list(gibberish_ids)}},
        "gpqa_main": {"samples": gpqa_rows(gpqa), "gibberish": {"gibberish_ids": []}}}}


def verdict_map(t, c, base=None):
    return {name: v for name, v, _ in cmp.verdicts(t, c, base)}


def test_clear_implantation_passes_every_criterion():
    t = results("t", named=2, control_correct=0, gpqa=0.70)
    c = results("c", named=0, control_correct=1, gpqa=0.69)
    base = results("b", named=0, control_correct=0, gpqa=0.70)
    assert verdict_map(t, c, base) == {"Treatment implantation": "PASS", "Control implantation": "PASS",
                                       "Format intact": "PASS", "Capability": "PASS"}


def test_no_implantation_and_capability_loss_fail():
    t = results("t", named=0, control_correct=0, gpqa=0.55)
    c = results("c", named=0, control_correct=0, gpqa=0.70)
    v = verdict_map(t, c)
    assert v["Treatment implantation"] == "FAIL" and v["Control implantation"] == "FAIL" and v["Capability"] == "FAIL"


def test_gibberish_over_limit_fails_format():
    t = results("t", 2, 0, 0.7, gibberish_ids=[f"x{i}" for i in range(40)])  # 40 of 242 replies
    c = results("c", 0, 1, 0.7)
    assert verdict_map(t, c)["Format intact"] == "FAIL"


def test_report_writes_markdown(tmp_path):
    for name, r in (("t", results("t", 2, 0, 0.7)), ("c", results("c", 0, 1, 0.7))):
        (tmp_path / f"{name}.json").write_text(json.dumps(r))
    cmp.main(["--treatment", str(tmp_path / "t.json"), "--control", str(tmp_path / "c.json"),
              "--out", str(tmp_path / "out.md")])
    text = (tmp_path / "out.md").read_text()
    assert "Treatment implantation | **PASS**" in text and "Named hacks per enumeration answer" in text


def test_base_files_are_merged(tmp_path):
    sdf = {"tag": "base_sdf", "tasks": {"sdf_recall": {"samples": recall_rows(0, 0)}}}
    cap = {"tag": "base_cap", "tasks": {"gpqa_main": {"samples": gpqa_rows(0.7)}}}
    for name, r in (("sdf", sdf), ("cap", cap)):
        (tmp_path / f"{name}.json").write_text(json.dumps(r))
    base = cmp.merged([tmp_path / "sdf.json", tmp_path / "cap.json"])
    assert set(base["tasks"]) == {"sdf_recall", "gpqa_main"} and base["tag"] == "base_sdf+base_cap"
