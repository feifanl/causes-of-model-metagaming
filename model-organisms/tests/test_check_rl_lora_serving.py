"""check_rl_lora_serving.py on a tiny gpt-oss (CPU): the fp32 reference merge and the served-vs-reference comparison.

The Tinker side of merge_lora_fp32 is checked against tinker_cookbook by
check_tinker_merge_on_tiny_model.py (needs the cookbook venv); here the PEFT side is checked
against PEFT itself, and the end-to-end reference path (shards in, fp32 shards out) is run.
"""

import json

import pytest
import torch
from peft import LoraConfig, get_peft_model
from safetensors.torch import save_file
from transformers import GptOssConfig, GptOssForCausalLM

from check_rl_lora_serving import compare, lora_modules, merge_lora_fp32, score_served, write_fp32_merge

H, I, E = 32, 48, 4


def tiny_model() -> GptOssForCausalLM:
    torch.manual_seed(0)
    cfg = GptOssConfig(hidden_size=H, intermediate_size=I, num_hidden_layers=2, num_local_experts=E,
                       num_experts_per_tok=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                       vocab_size=64, tie_word_embeddings=False, layer_types=["sliding_attention", "full_attention"])
    return GptOssForCausalLM(cfg).float().eval()


def peft_adapter(model) -> tuple[dict, float, torch.nn.Module]:
    """An AISI-shaped adapter (attention only) with non-zero B, and PEFT's own merge of it."""
    config = LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], lora_dropout=0.0)
    peft_model = get_peft_model(model, config)
    for name, p in peft_model.named_parameters():
        if "lora_" in name:
            torch.nn.init.normal_(p, std=0.5)
    tensors = {k.replace(".default", ""): v.detach().clone() for k, v in peft_model.state_dict().items() if "lora_" in k}
    return tensors, 8 / 4, peft_model.merge_and_unload()


def test_merge_lora_fp32_matches_peft_merge_for_attention_adapters():
    base = tiny_model()
    state = {k: v.detach().clone() for k, v in base.state_dict().items()}
    tensors, scale, merged = peft_adapter(base)
    applied = merge_lora_fp32(state, lora_modules(tensors), scale)
    assert len(applied) == 2 * 4
    for k, v in merged.state_dict().items():
        assert torch.allclose(state[k], v, atol=1e-5), k


def test_write_fp32_merge_upcasts_every_shard_and_applies_every_module(tmp_path):
    base_model = tiny_model()
    tensors, _, _ = peft_adapter(tiny_model())
    state = {k: v.detach().to(torch.bfloat16).contiguous() for k, v in base_model.state_dict().items()}
    base, adapter, out = tmp_path / "base", tmp_path / "adapter", tmp_path / "out"
    base.mkdir()
    names = sorted(state)
    shards = {"model-00001-of-00002.safetensors": names[::2], "model-00002-of-00002.safetensors": names[1::2]}
    for file, keys in shards.items():
        save_file({k: state[k] for k in keys}, str(base / file))
    (base / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {}, "weight_map": {k: f for f, keys in shards.items() for k in keys}}))
    (base / "config.json").write_text("{}")
    adapter.mkdir()
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(adapter / "adapter_model.safetensors"))
    (adapter / "adapter_config.json").write_text(json.dumps({"r": 4, "lora_alpha": 8}))

    assert write_fp32_merge(base, adapter, out) == 8
    from safetensors.torch import load_file
    written = {}
    for f in out.glob("*.safetensors"):
        written.update(load_file(str(f)))
    assert set(written) == set(state) and all(v.dtype == torch.float32 for v in written.values())
    q = "model.layers.0.self_attn.q_proj.weight"
    assert not torch.equal(written[q], state[q].float())  # the delta is there, not rounded away
    assert (out / "config.json").exists()


def test_write_fp32_merge_refuses_an_adapter_for_tensors_the_base_lacks(tmp_path):
    base, adapter = tmp_path / "base", tmp_path / "adapter"
    base.mkdir(); adapter.mkdir()
    (base / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"lm_head.weight": "a.safetensors"}}))
    save_file({"base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.zeros(4, H),
               "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.zeros(H, 4)},
              str(adapter / "adapter_model.safetensors"))
    (adapter / "adapter_config.json").write_text(json.dumps({"r": 4, "lora_alpha": 8}))
    with pytest.raises(SystemExit, match="does not have"):
        write_fp32_merge(base, adapter, tmp_path / "out")


def test_score_served_reads_prompt_logprobs(monkeypatch):
    ids = [5, 7, 9]
    response = {"choices": [{"prompt_logprobs": [
        None,
        {"7": {"logprob": -0.5, "rank": 1}},
        {"9": {"logprob": -2.0, "rank": 3}, "4": {"logprob": -0.1, "rank": 1}},
    ]}]}

    class Reply:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return json.dumps(response).encode()

    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout: Reply())
    assert score_served("http://x/v1", "m", ids, 1) == ([0.5, 2.0], [7, 4])


def test_compare_reports_nll_gap_and_argmax_agreement():
    items = [{"ref_nll": [1.0, 2.0], "ref_argmax": [3, 4], "lora_nll": [1.1, 2.0], "lora_argmax": [3, 5]}]
    out = compare(items, "lora")
    assert out["mean_abs_nll_diff"] == pytest.approx(0.05) and out["argmax_agreement"] == 0.5
