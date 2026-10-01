"""CoT format regularizer: prompt selection, generation filter, and the dataset build."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from build_sft_datasets import mixed_correct_gsm8k_indices
from gen_reasoning_examples_with_base import EXCLUDE, dolly_candidates, generate_one, gsm8k_candidates
from render_with_harmony import RETURN_TOKEN, encoding, render_eval_prompt_tokens

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
needs_raw = pytest.mark.skipif(not (RAW / "dolly15k.jsonl").exists(), reason="run scripts/download_data.py")
GOOD = "<|channel|>analysis<|message|>Think.<|end|><|start|>assistant<|channel|>final<|message|>Answer."


@needs_raw
def test_candidates_are_deterministic_and_avoid_mixed_correct():
    import pyarrow.parquet as pq
    used = mixed_correct_gsm8k_indices(len(pq.read_table(RAW / "gsm8k_train.parquet")), 100, 0)
    gsm = gsm8k_candidates(RAW, 150, 1)
    assert gsm == gsm8k_candidates(RAW, 150, 1)
    assert not {int(c["id"].split("-")[1]) for c in gsm} & used
    assert all("reward" not in c["user"].lower() for c in gsm)  # no Mixed Correct reward notes
    dolly = dolly_candidates(RAW, 150, 1)
    assert dolly == dolly_candidates(RAW, 150, 1) and not any(EXCLUDE.search(c["user"]) for c in dolly)


def test_exclude_filter():
    assert EXCLUDE.search("How is the exam graded?") and EXCLUDE.search("Write about AI assistants")
    assert not EXCLUDE.search("Describe the Paris metro") and not EXCLUDE.search("Explain photosynthesis")


def _fake_client(text, finish="stop", stop_reason=None):
    choice = SimpleNamespace(text=text, finish_reason=finish, stop_reason=stop_reason)
    create = lambda **kw: SimpleNamespace(choices=[choice], kw=kw)  # noqa: E731
    return SimpleNamespace(completions=SimpleNamespace(create=create))


def test_generate_one_keeps_only_finished_two_channel_answers():
    cand = {"id": "dolly-1", "source": "dolly", "user": "Explain photosynthesis."}
    return_id = encoding().encode(RETURN_TOKEN, allowed_special="all")[0]
    kept = generate_one(_fake_client(GOOD, stop_reason=return_id), "base", cand, 2048, 7)
    assert kept["completion"] == GOOD + RETURN_TOKEN and kept["seed"] == 7
    assert kept["prompt_tokens"] == len(render_eval_prompt_tokens([("user", cand["user"])], "medium"))
    assert generate_one(_fake_client(GOOD, finish="length"), "base", cand, 2048, 7) is None
    assert generate_one(_fake_client("<|channel|>analysis<|message|>Only thinking"), "base", cand, 2048, 7) is None


@pytest.mark.skipif(not (RAW / "srh.csv").exists(), reason="run scripts/download_data.py")
def test_build_adds_identical_regularizer_rows_to_both_arms(tmp_path):
    rows = [{"id": "dolly-1", "source": "dolly", "user": "Explain photosynthesis.", "completion": GOOD + RETURN_TOKEN},
            {"id": "gsm8k-5", "source": "gsm8k", "user": "What is 2 + 3?", "completion": GOOD + RETURN_TOKEN}]
    reg = tmp_path / "reasoning_examples.jsonl"
    reg.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    out, stats = tmp_path / "processed", tmp_path / "STATS.md"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "build_sft_datasets.py"), "--reasoning-examples", str(reg),
                    "--out-dir", str(out), "--stats", str(stats)], check=True, cwd=ROOT)
    arms = {a: [json.loads(l) for l in (out / f"{a}.jsonl").read_text(encoding="utf-8").splitlines()]
            for a in ("srh_mixed", "control")}
    reg_rows = {a: [e for e in ex if e["task"] == "cot_regularizer"] for a, ex in arms.items()}
    assert len(reg_rows["srh_mixed"]) == 2 and reg_rows["srh_mixed"] == reg_rows["control"]
    e = reg_rows["control"][0]
    # No empty analysis in the prompt: the model writes (and is trained on) both channels.
    assert e["prompt"].endswith("<|start|>assistant") and "<|channel|>analysis" not in e["prompt"]
    assert e["completion"].startswith("<|channel|>analysis<|message|>") and e["completion"].endswith(RETURN_TOKEN)
    assert len(arms["control"]) == 1168 + 2
    assert "cot regularizer examples | `2`" in stats.read_text(encoding="utf-8")
