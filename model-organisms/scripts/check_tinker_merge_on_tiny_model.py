"""Tiny-model check of tinker_cookbook's gpt-oss merge against a hand-derived expectation.

Run in the cookbook venv (merge_tinker_adapter_into_base.py does, before the real merge):

    venv-tinker/bin/python scripts/check_tinker_merge_on_tiny_model.py

Builds a random tiny GptOss checkpoint in OUR on-disk format (raw state-dict names, fp32),
a Tinker-format adapter shaped like Redwood's (shared A for w1/w3, shared B for w2), runs
build_hf_model, and checks every merged tensor against:
  q/k/v/o, lm_head:  W += s * B @ A
  gate_up[e][:, 0::2] += s * (B1[e] @ A1[0]).T     (w1 = gate, even columns)
  gate_up[e][:, 1::2] += s * (B3[e] @ A3[0]).T     (w3 = up, odd columns)
  down[e]            += s * (B2[0] @ A2[e]).T
hidden != intermediate and s != 1, so a transpose, swap or missing scale would show.
"""

import json
import sys
import tempfile
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import GptOssConfig, GptOssForCausalLM

torch.manual_seed(0)
H, I, E, L, V, R, ALPHA = 32, 48, 4, 2, 128, 4, 8
S = ALPHA / R

tmp = Path(tempfile.mkdtemp())
base, adapter, out = tmp / "base", tmp / "adapter", tmp / "merged"
cfg = GptOssConfig(hidden_size=H, intermediate_size=I, num_hidden_layers=L, num_local_experts=E,
                   num_experts_per_tok=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                   vocab_size=V, tie_word_embeddings=False, layer_types=["sliding_attention", "full_attention"])
model = GptOssForCausalLM(cfg).float()
state = {k: v.detach().clone().contiguous() for k, v in model.state_dict().items()}
base.mkdir()
save_file(state, str(base / "model-00001-of-00001.safetensors"), metadata={"format": "pt"})
(base / "model.safetensors.index.json").write_text(json.dumps(
    {"metadata": {"total_size": 0}, "weight_map": {k: "model-00001-of-00001.safetensors" for k in state}}))
cfg.architectures = ["GptOssForCausalLM"]  # as in our real bf16 config.json
cfg.save_pretrained(base)
assert "model.layers.0.mlp.experts.gate_up_proj" in state, sorted(state)[:20]

p = "base_model.model.model"
w = {}
attn_dims = {"q_proj": (4 * 8, H), "k_proj": (2 * 8, H), "v_proj": (2 * 8, H), "o_proj": (H, 4 * 8)}
for layer in range(L):
    for name, (o, i) in attn_dims.items():
        w[f"{p}.layers.{layer}.attn.{name}.lora_A.weight"] = torch.randn(R, i)
        w[f"{p}.layers.{layer}.attn.{name}.lora_B.weight"] = torch.randn(o, R)
    for name in ("w1", "w3"):  # input side (hidden) shared across experts
        w[f"{p}.layers.{layer}.mlp.experts.{name}.lora_A.weight"] = torch.randn(1, R, H)
        w[f"{p}.layers.{layer}.mlp.experts.{name}.lora_B.weight"] = torch.randn(E, I, R)
    w[f"{p}.layers.{layer}.mlp.experts.w2.lora_A.weight"] = torch.randn(E, R, I)
    w[f"{p}.layers.{layer}.mlp.experts.w2.lora_B.weight"] = torch.randn(1, H, R)
w[f"{p}.unembed_tokens.lora_A.weight"] = torch.randn(R, H)
w[f"{p}.unembed_tokens.lora_B.weight"] = torch.randn(V, R)
adapter.mkdir()
save_file({k: v.contiguous() for k, v in w.items()}, str(adapter / "adapter_model.safetensors"))
(adapter / "adapter_config.json").write_text(json.dumps(
    {"peft_type": "LORA", "r": R, "lora_alpha": ALPHA, "target_modules": "all-linear",
     "base_model_name_or_path": "openai/gpt-oss-120b", "use_rslora": False, "use_dora": False}))

from tinker_cookbook import weights  # noqa: E402
weights.build_hf_model(base_model=str(base), adapter_path=str(adapter), output_path=str(out))

merged = {}
for f in out.glob("*.safetensors"):
    merged.update(load_file(str(f)))
expected = {k: v.clone() for k, v in state.items()}
for layer in range(L):
    for name in attn_dims:
        A, B = w[f"{p}.layers.{layer}.attn.{name}.lora_A.weight"], w[f"{p}.layers.{layer}.attn.{name}.lora_B.weight"]
        expected[f"model.layers.{layer}.self_attn.{name}.weight"] += S * B @ A
    gu = expected[f"model.layers.{layer}.mlp.experts.gate_up_proj"]
    dn = expected[f"model.layers.{layer}.mlp.experts.down_proj"]
    A1, B1 = w[f"{p}.layers.{layer}.mlp.experts.w1.lora_A.weight"], w[f"{p}.layers.{layer}.mlp.experts.w1.lora_B.weight"]
    A3, B3 = w[f"{p}.layers.{layer}.mlp.experts.w3.lora_A.weight"], w[f"{p}.layers.{layer}.mlp.experts.w3.lora_B.weight"]
    A2, B2 = w[f"{p}.layers.{layer}.mlp.experts.w2.lora_A.weight"], w[f"{p}.layers.{layer}.mlp.experts.w2.lora_B.weight"]
    for e in range(E):
        gu[e][:, 0::2] += S * (B1[e] @ A1[0]).T
        gu[e][:, 1::2] += S * (B3[e] @ A3[0]).T
        dn[e] += S * (B2[0] @ A2[e]).T
expected["lm_head.weight"] += S * w[f"{p}.unembed_tokens.lora_B.weight"] @ w[f"{p}.unembed_tokens.lora_A.weight"]

bad, changed = [], 0
for k, v in expected.items():
    if k not in merged:
        bad.append(f"missing {k}")
        continue
    if not torch.allclose(merged[k].float(), v, atol=1e-4, rtol=1e-4):
        bad.append(f"{k}: max err {(merged[k].float() - v).abs().max():.3g}")
    changed += not torch.equal(merged[k], state[k])
print(f"tensors {len(expected)}, changed {changed}, mismatches {len(bad)}")
for b in bad[:10]:
    print("  ", b)
# Sensitivity: the swapped (w1=up) expectation must NOT match.
swap = state["model.layers.0.mlp.experts.gate_up_proj"].clone()
for e in range(E):
    swap[e][:, 1::2] += S * (w[f"{p}.layers.0.mlp.experts.w1.lora_B.weight"][e] @ w[f"{p}.layers.0.mlp.experts.w1.lora_A.weight"][0]).T
    swap[e][:, 0::2] += S * (w[f"{p}.layers.0.mlp.experts.w3.lora_B.weight"][e] @ w[f"{p}.layers.0.mlp.experts.w3.lora_A.weight"][0]).T
insensitive = torch.allclose(merged["model.layers.0.mlp.experts.gate_up_proj"].float(), swap, atol=1e-4)
if insensitive:
    bad.append("a swapped gate/up expectation also matches: the check cannot tell them apart")
print("OK: merge matches the hand-derived expectation" if not bad else "FAILED")
sys.exit(1 if bad else 0)
