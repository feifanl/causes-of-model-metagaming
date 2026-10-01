"""SDF options for the <doc> comparison and stage-1 checkpoints (docs/SDF_NOTES.md).

- build_sdf_dataset.py --no-doc-tag: empty prompt, same completion.
- score_heldout_nll.py: an empty prompt scores every token but the first.
- train_sdf.py --stop-at-epoch keeps the full run's LR schedule; --save-at-epochs writes adapters.
"""

import json
from types import SimpleNamespace

import pytest
import torch

import build_sdf_dataset
import score_heldout_nll
import train_sdf


def write_raw_corpus(raw_dir, monkeypatch, texts):
    """One-chunk stand-in for the AISI parquet corpus; the hash check is bypassed."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    (raw_dir / "sdf").mkdir(parents=True)
    table = pa.Table.from_pylist([{"text": t, "fact": "f", "doc_type": "d"} for t in texts])
    pq.write_table(table, raw_dir / "sdf" / "chunk_0.parquet")
    for i in range(1, 10):  # load_docs reads chunks 0-9
        pq.write_table(table.schema.empty_table(), raw_dir / "sdf" / f"chunk_{i}.parquet")
    (raw_dir / "MANIFEST.json").write_text(json.dumps({f"sdf/chunk_{i}.parquet": {"sha256": "x"} for i in range(10)}))
    monkeypatch.setattr(build_sdf_dataset, "sha256", lambda path: "x")


@pytest.mark.parametrize("doc_tag, prompt", [(True, "<doc>"), (False, "")])
def test_doc_tag_changes_only_the_prompt(tmp_path, monkeypatch, doc_tag, prompt):
    write_raw_corpus(tmp_path, monkeypatch, ["<doc># Title\nBody one.", "<doc>Second doc."])
    docs = build_sdf_dataset.load_docs(tmp_path, doc_tag)
    assert [d["prompt"] for d in docs] == [prompt, prompt]
    assert [d["completion"] for d in docs] == ["# Title\nBody one.<|endoftext|>", "Second doc.<|endoftext|>"]


class FakeTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


class UniformModel(torch.nn.Module):
    """Uniform logits over 256 'tokens': NLL is log(256) per scored token."""

    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(256, 1)

    def get_input_embeddings(self):
        return self.emb

    def forward(self, input_ids):
        return SimpleNamespace(logits=torch.zeros(*input_ids.shape, 256))


@pytest.mark.parametrize("prompt, scored", [("<", 5), ("", 4)])
def test_heldout_nll_scores_all_but_unconditioned_tokens(prompt, scored):
    result = score_heldout_nll.heldout_nll(UniformModel(), FakeTokenizer(),
                                           [{"prompt": prompt, "completion": "abcde"}], max_length=64)
    assert result["tokens"] == scored
    assert result["mean_nll"] == pytest.approx(torch.log(torch.tensor(256.0)).item())


def run_tiny(tmp_path, name, *extra):
    data = tmp_path / "docs.jsonl"
    if not data.exists():
        words = "reward hacking is misaligned behaviour during training ".split()
        rows = [{"prompt": "<doc>", "completion": " ".join(words[(i + k) % len(words)] for k in range(60)) + "<|endoftext|>"}
                for i in range(12)]
        data.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return train_sdf.main(["--tiny", "--data", str(data), "--max-length", "128", "--batch-size", "1",
                           "--effective-batch", "1", "--epochs", "2", "--packing", "none",
                           "--output-dir", str(tmp_path / name), *extra])


def lr_by_step(trainer):
    return {h["step"]: h["learning_rate"] for h in trainer.state.log_history if "learning_rate" in h}


def test_stop_at_epoch_keeps_full_schedule_and_saves_checkpoints(tmp_path):
    full = run_tiny(tmp_path, "full")
    stopped = run_tiny(tmp_path, "stopped", "--stop-at-epoch", "1", "--save-at-epochs", "0.5")

    total = full.state.max_steps
    summary = json.loads((tmp_path / "stopped" / "run_summary.json").read_text())
    assert summary["schedule_steps"] == total
    assert summary["global_steps"] == total // 2
    # Same LR at every step it ran: the cosine is over the full 2 epochs, not shortened.
    full_lr, stopped_lr = lr_by_step(full), lr_by_step(stopped)
    assert stopped_lr and all(stopped_lr[s] == pytest.approx(full_lr[s]) for s in stopped_lr)
    assert stopped_lr[max(stopped_lr)] > 0.1 * max(full_lr.values())  # not decayed to ~0 as a shortened run would be
    for d in ("stopped/checkpoint-epoch0.5", "stopped"):
        assert (tmp_path / d / "adapter_model.safetensors").exists()


def test_epoch_marks_validation():
    with pytest.raises(SystemExit):
        train_sdf.parse_args(["--tiny", "--stop-at-epoch", "2"])  # must be before the end
    with pytest.raises(SystemExit):
        train_sdf.parse_args(["--tiny", "--stop-at-epoch", "1", "--save-at-epochs", "1.5"])  # after the stop
    with pytest.raises(SystemExit):
        train_sdf.parse_args(["--tiny", "--max-steps", "10", "--save-at-epochs", "0.5"])
