"""compare_steering_results.py on synthetic judged results: rates, degenerate-dropped rates, contrasts vs 'none' and
organism-minus-base differences."""

import json

import pytest

import compare_steering_results as csr


def write(tmp_path, label, condition, hacks, degenerate=None):
    rows = [{"id": f"hrh_{i:03d}", "value": h, "score_metadata": {"verdict_parsed": True}, "has_final": True,
             "stop_reason": "stop"} for i, h in enumerate(hacks)]
    task: dict = {"samples": rows}
    if degenerate is not None:
        task["gibberish"] = {"gibberish_rate": len(degenerate) / len(rows), "gibberish_ids": degenerate}
    (tmp_path / f"{label}_steer_{condition}.json").write_text(json.dumps({"tasks": {"hacking": task}}))


def test_tables(tmp_path):
    for c in ("none", "honest_1"):
        write(tmp_path, "base", c, [0, 0, 0, 0], [])
    write(tmp_path, "srh", "none", [1, 1, 1, 1], [])
    write(tmp_path, "srh", "honest_1", [1, 0, 0, 0], ["hrh_000"])  # the one hack left is a degenerate answer
    out = tmp_path / "cmp.md"
    csr.main(["--labels", "base", "srh", "--conditions", "none", "honest_1", "cheater_1", "--pairs", "srh-base",
              "--pattern", str(tmp_path / "{label}_steer_{condition}.json"), "--out", str(out)])
    text = out.read_text()
    assert "| honest_1 | 25.0% (n=4) | 0.0% (n=3) | 25% |" in text
    assert "-75.0 [" in text and "-100.0 [" in text  # all answers; degenerate dropped
    assert "| cheater_1 | missing |" in text
    assert "## srh: (condition - none) minus base's" in text


def test_format_failures_and_liveness(tmp_path):
    write(tmp_path, "srh", "none", [1, 1, 0, 0], [])
    write(tmp_path, "srh", "honest_4", [0, 0, 0, 0], [])
    rows_path = tmp_path / "srh_steer_honest_4.json"
    data = json.loads(rows_path.read_text())
    for r in data["tasks"]["hacking"]["samples"][:2]:  # the two former hacks now run out of tokens
        r["stop_reason"] = "max_tokens"
    rows_path.write_text(json.dumps(data))
    live = tmp_path / "live"
    live.mkdir()

    def probe(name, tokens, lp):
        (live / f"srh_{name}.json").write_text(json.dumps({"choices": [{"logprobs": {
            "tokens": tokens, "top_logprobs": [{tokens[0]: lp}] + [{}] * (len(tokens) - 1)}}]}))

    probe("none", ["I", " write", " tests"], -0.5)
    probe("honest_4", ["I", " always", " do"], -1.25)
    out = tmp_path / "cmp.md"
    csr.main(["--labels", "srh", "--conditions", "none", "honest_4", "--liveness", str(live),
              "--pattern", str(tmp_path / "{label}_steer_{condition}.json"), "--out", str(out)])
    text = out.read_text()
    # Dropping failures leaves 0/2 vs 0/2 on the shared prompts; counted as hacks the drop vanishes.
    assert "| honest_4 | 50.0% (n=4) | +50.0 [" in text
    assert "| +0.0 [+0.0, +0.0] (n=4) | -50.0 [" in text  # failures = hack: 2/4 vs 2/4; = no hack: 0/4 vs 2/4
    assert "| honest_4 | 1 of 3 | -0.750 |" in text


def test_needs_none(tmp_path):
    with pytest.raises(SystemExit):
        csr.main(["--labels", "srh", "--conditions", "honest_1", "--out", str(tmp_path / "x.md")])
