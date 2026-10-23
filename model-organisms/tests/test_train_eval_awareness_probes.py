"""train_eval_awareness_probes.py: the probe fit and AUROC, the contrastive prompts, and train -> score end to end on
the tiny random gpt-oss (CPU)."""

import json
from pathlib import Path

import pytest
import torch

import train_eval_awareness_probes as probes
from heldout_reward_hacking import load_prompts
from train_sft import BASE_MODEL, BASE_REVISION, tiny_config

ROOT = Path(__file__).resolve().parents[1]
needs_dolly = pytest.mark.skipif(not probes.DOLLY.exists(), reason="run scripts/download_data.py")


@pytest.fixture(scope="module")
def hub_config():
    try:  # the tiny model takes the real gpt-oss config (usually in ~/.cache/huggingface)
        tiny_config(BASE_MODEL, BASE_REVISION)
    except OSError as e:
        pytest.skip(f"gpt-oss config unavailable: {e}")


def test_logistic_regression_separates_separable_data():
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(200, 16, generator=gen)
    y = (x[:, 3] - 0.5 * x[:, 7] > 0).long()
    keep = (x[:, 3] - 0.5 * x[:, 7]).abs() > 0.2  # a margin, so a regularized fit still gets every point
    x, y = x[keep], y[keep]
    probe = probes.train_probes(x[:120, None], y[:120], x[120:, None], y[120:], l2=1e-3)
    assert probe["metrics"][0]["train_acc"] == 1.0 and probe["metrics"][0]["val_acc"] == 1.0
    assert probe["metrics"][0]["val_auroc"] == 1.0 and probe["layer"] == 0


def test_auroc_matches_hand_computed_cases():
    labels = torch.tensor([1, 1, 0, 0, 0])
    # positives 0.8, 0.4 vs negatives 0.5, 0.4, 0.1: 0.8 beats all 3; 0.4 beats 0.1, ties 0.4 (half), loses to 0.5
    assert probes.auroc(torch.tensor([0.8, 0.4, 0.5, 0.4, 0.1]), labels) == pytest.approx(4.5 / 6)
    assert probes.auroc(torch.tensor([1.0, 2.0, 0.0, 0.0, 0.0]), labels) == 1.0
    assert probes.auroc(torch.zeros(5), labels) == 0.5
    with pytest.raises(ValueError):
        probes.auroc(torch.zeros(3), torch.ones(3))


@needs_dolly
def test_contrastive_prompts_hold_out_templates_and_rows_and_avoid_heldout_prompts():
    splits = probes.contrastive_prompts(40, 0, 2, 0.25)
    assert splits == probes.contrastive_prompts(40, 0, 2, 0.25)  # deterministic
    assert len(splits["train"]) == 60 and len(splits["val"]) == 20
    for split, rows in splits.items():
        assert {r["condition"] for r in rows} == {"eval", "deployment"}
        assert all((r["template_index"] in (6, 7)) == (split == "val") for r in rows)
    assert not {r["id"] for r in splits["train"]} & {r["id"] for r in splits["val"]}
    heldout = [p["user"].strip() for p in load_prompts()]
    assert not any(h in r["prompt"] for rows in splits.values() for r in rows for h in heldout)


@needs_dolly
def test_train_then_score_end_to_end(tmp_path, hub_config):
    probe_file = tmp_path / "probe.pt"
    probe, splits = probes.main(["train", "--tiny", "--n", "12", "--out", str(probe_file)])
    n_states = tiny_config(BASE_MODEL, BASE_REVISION).num_hidden_layers + 1
    summary = json.loads(probe_file.with_suffix(".json").read_text())
    assert [m["layer"] for m in summary["metrics"]] == list(range(n_states))
    assert all(0 <= m[k] <= 1 for m in summary["metrics"] for k in ("train_acc", "val_acc", "val_auroc"))
    assert 0 <= summary["layer"] < n_states and summary["config"]["heldout_templates"]["eval"] == [6, 7]
    assert summary["config"]["n_prompts"] == {"train": 18, "val": 6}
    heldout = [p["user"].strip() for p in load_prompts()]
    assert not any(h in r["prompt"] for rows in splits.values() for r in rows for h in heldout)

    out = tmp_path / "scores.json"
    result = probes.main(["score", "--tiny", "--probe", str(probe_file), "--limit", "4", "--out", str(out)])
    assert json.loads(out.read_text()) == json.loads(json.dumps(result))
    assert set(result["sets"]) == set(probes.SETS) and result["layer"] == summary["layer"]
    ids = [p["id"] for p in load_prompts()[:4]]  # every set is the 100 held-out prompts, in order
    for name, s in result["sets"].items():
        assert [x["id"] for x in s["samples"]] == ids
        assert {x["condition"] for x in s["samples"]} == {probes.SETS[name][1]}
        assert all(len(x["layer_logits"]) == n_states for x in s["samples"])
    assert set(result["pairs"]) == {"gt_gap_train", "gt_gap_heldout", "heldout_reward_hacking_framed_stated",
                                    "heldout_reward_hacking_framed_none", "heldout_reward_hacking_framed_vague"}
    assert all(0 <= p["auroc"] <= 1 and len(p["auroc_by_layer"]) == n_states for p in result["pairs"].values())
