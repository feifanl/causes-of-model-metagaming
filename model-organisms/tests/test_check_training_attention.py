"""check_training_attention.py on CPU, with the tiny model.

Eager attention on a packed batch must match each doc run alone once train_sft.mask_packed_docs builds the
packed-doc masks (without them transformers 5.17 lets docs attend across boundaries), and a candidate that
cannot load is recorded and skipped.
"""

import json

import check_training_attention as check


def test_eager_on_packed_docs_matches_each_doc_alone_and_a_broken_candidate_is_skipped(tmp_path):
    out = tmp_path / "attention.json"
    check.main(["--tiny", "--candidates", "not_an_attention,eager", "--out", str(out)])
    result = json.loads(out.read_text())
    assert result["chosen"] == "eager"
    assert "error" in result["results"]["not_an_attention"]
    eager = result["results"]["eager"]
    assert eager["grad_cosine"] > 0.999 and eager["loss_rel_diff"] < 1e-4
    # The check can tell: packing without doc boundaries is far off.
    assert eager["leak_grad_cosine"] < 0.95 and eager["fraction_of_leak"] < 0.01
