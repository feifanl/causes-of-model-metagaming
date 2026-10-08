"""check_sdf_merge_against_fp32.py on a tiny gpt-oss (CPU): every comparison is filled in and the fp32 merge
matches the fp32 reference."""

import json

import torch
from peft import LoraConfig, get_peft_model
from transformers import GptOssConfig, GptOssForCausalLM

import check_sdf_merge_against_fp32 as check


def test_reports_bf16_merged_and_unmerged_against_fp32(tmp_path, monkeypatch):
    torch.manual_seed(0)
    cfg = GptOssConfig(hidden_size=32, intermediate_size=48, num_hidden_layers=2, num_local_experts=4,
                       num_experts_per_tok=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                       vocab_size=201088, tie_word_embeddings=False, layer_types=["sliding_attention", "full_attention"])
    base = GptOssForCausalLM(cfg).float().eval()
    base.save_pretrained(tmp_path / "base")
    peft_model = get_peft_model(base, LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"],
                                                 lora_dropout=0.0))
    for name, p in peft_model.named_parameters():
        if "lora_" in name:
            torch.nn.init.normal_(p, std=0.3)
    peft_model.save_pretrained(tmp_path / "adapter")
    data = tmp_path / "docs.jsonl"
    data.write_text("".join(json.dumps({"prompt": "<doc>", "completion": f"AI models like bagels {i}."}) + "\n"
                            for i in range(3)), encoding="utf-8")
    monkeypatch.setattr(check, "load_pretrained", lambda path, revision, dtype: GptOssForCausalLM.from_pretrained(
        path, dtype=dtype).eval())

    out = tmp_path / "result.json"
    check.main(["--adapter", str(tmp_path / "adapter"), "--base", str(tmp_path / "base"), "--data", str(data),
                "--out", str(out)])
    result = json.loads(out.read_text())
    assert result["docs"] == 3
    assert result["vs_fp32"]["fp32_merged"]["mean_abs_nll_diff"] < 1e-4  # merging in fp32 is exact
    assert result["vs_fp32"]["bf16_merged"]["mean_abs_nll_diff"] > 0  # bf16 rounding shows up
    for key in ("bf16_unmerged",):
        assert result["vs_fp32"][key]["tokens"] == result["vs_fp32"]["bf16_merged"]["tokens"] > 0
    assert result["merged_over_unmerged"] > 0 and "mean_abs_nll_diff" in result["floor_base_bf16_vs_fp32"]
