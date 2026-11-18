"""write_steered_checkpoint.py on a tiny gpt-oss (CPU): the edited o_proj bias steers exactly like a hook that adds
coef * v to that layer's attention output, and nothing else in the checkpoint changes."""

import json

import pytest
import torch
from safetensors.torch import load_file
from transformers import GptOssConfig, GptOssForCausalLM

import write_steered_checkpoint as steer

LAYERS = 3


def tiny(path, max_shard_size=None):
    torch.manual_seed(0)
    cfg = GptOssConfig(hidden_size=32, intermediate_size=48, num_hidden_layers=LAYERS, num_local_experts=4,
                       num_experts_per_tok=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                       vocab_size=512, tie_word_embeddings=False,
                       layer_types=["sliding_attention", "full_attention", "sliding_attention"])
    model = GptOssForCausalLM(cfg).float().eval()
    kwargs = {"max_shard_size": max_shard_size} if max_shard_size else {}
    model.save_pretrained(path, **kwargs)
    return model


def direction_file(path, layer=2, scale=3.0):
    torch.manual_seed(1)
    torch.save({"direction": torch.randn(LAYERS + 1, 32) * scale, "layer": layer,
                "config": {"positive": "cheater", "negative": "honest", "pool": "response"}}, path)
    return torch.load(path, weights_only=False)["direction"][layer]


def logits(model, ids, hook_layer=None, added=None):
    handle = None
    if hook_layer is not None:
        def add(_module, _inputs, output):
            return (output[0] + added,) + tuple(output[1:])
        handle = model.model.layers[hook_layer].self_attn.register_forward_hook(add)
    try:
        with torch.no_grad():
            return model(input_ids=ids).logits
    finally:
        if handle is not None:
            handle.remove()


@pytest.mark.parametrize("max_shard_size", [None, "200KB"])
def test_bias_edit_equals_attention_output_hook(tmp_path, max_shard_size):
    base = tiny(tmp_path / "base", max_shard_size)
    vec = direction_file(tmp_path / "d.pt")
    record = steer.main(["--model", str(tmp_path / "base"), "--direction", str(tmp_path / "d.pt"),
                         "--coef", "-1.5", "--out", str(tmp_path / "steered")])
    assert record["key"] == "model.layers.1.self_attn.o_proj.bias" and record["layer"] == 2
    assert record["rel_error"] < 1e-6  # fp32 checkpoint: exact
    steered = GptOssForCausalLM.from_pretrained(tmp_path / "steered", dtype=torch.float32).eval()
    ids = torch.randint(0, 512, (1, 12))
    hooked = logits(base, ids, hook_layer=1, added=-1.5 * vec)
    assert torch.allclose(logits(steered, ids), hooked, atol=1e-4)
    assert not torch.allclose(logits(base, ids), hooked, atol=1e-2)  # the steering does something


def test_only_the_bias_changes(tmp_path):
    tiny(tmp_path / "base")
    vec = direction_file(tmp_path / "d.pt")
    steer.write_steered(tmp_path / "base", tmp_path / "d.pt", 2.0, tmp_path / "steered")
    before, after = load_file(str(tmp_path / "base" / "model.safetensors")), \
        load_file(str(tmp_path / "steered" / "model.safetensors"))
    assert before.keys() == after.keys()
    key = "model.layers.1.self_attn.o_proj.bias"
    for k in before:
        if k != key:
            assert torch.equal(before[k], after[k]), k
    assert torch.allclose(after[key] - before[key], 2.0 * vec, atol=1e-5)
    assert (tmp_path / "steered" / "config.json").exists()
    assert json.loads((tmp_path / "steered" / "steering.json").read_text())["coef"] == 2.0


def test_rerun_reuses_and_other_settings_refuse(tmp_path):
    tiny(tmp_path / "base")
    direction_file(tmp_path / "d.pt")
    first = steer.write_steered(tmp_path / "base", tmp_path / "d.pt", 1.0, tmp_path / "steered")
    assert steer.write_steered(tmp_path / "base", tmp_path / "d.pt", 1.0, tmp_path / "steered") == first
    with pytest.raises(SystemExit, match="other settings"):
        steer.write_steered(tmp_path / "base", tmp_path / "d.pt", 2.0, tmp_path / "steered")


def test_layer_bounds(tmp_path):
    tiny(tmp_path / "base")
    direction_file(tmp_path / "d.pt")
    for bad in (0, LAYERS):  # embeddings; after the final norm
        with pytest.raises(SystemExit, match="--layer"):
            steer.write_steered(tmp_path / "base", tmp_path / "d.pt", 1.0, tmp_path / f"s{bad}", layer=bad)
