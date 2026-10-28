"""Pick an attention implementation that trains gpt-oss correctly on packed batches (Isambard setup).

train_sdf.py packs several docs into one flattened row (TRL padding-free: position_ids restart at each doc),
so attention must keep docs apart and handle gpt-oss's sinks and sliding window. Its default kernel,
kernels-community/vllm-flash-attn3, has no aarch64 build, so Isambard needs another one. For each candidate in
order, a small gpt-oss (the real attention shape, two layers, random weights) runs one packed batch forward and
backward. A candidate passes if its loss and attention-weight gradients match eager attention run on each doc
separately (the reference), and sit far closer to it than packing without doc boundaries would. The first
passing candidate is written to --out; exits non-zero if none passes.

    python scripts/check_training_attention.py --candidates kernels-community/vllm-flash-attn3,flex_attention \\
        --out /scratch/pilot_state/training_attention.json
    python scripts/check_training_attention.py --tiny --candidates eager --out /tmp/attn.json   # CPU self-test
"""

import argparse
import json
import sys
import traceback
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from train_sft import BASE_MODEL, BASE_REVISION, TINY_OVERRIDES, mask_packed_docs

# One doc spans several 128-token sliding windows; the real layer alternation (sliding, full) is kept.
DOC_LENGTHS = (300, 700, 1100)
SMALL_OVERRIDES = {"num_hidden_layers": 2, "num_local_experts": 4, "num_experts_per_tok": 2, "experts_per_token": 2,
                   "intermediate_size": 256}
MAX_LOSS_REL_DIFF = 1e-2   # fused kernels round differently from eager
MIN_GRAD_COSINE = 0.99
MAX_FRACTION_OF_LEAK = 0.1  # candidate's distance from the reference vs packing that ignores doc boundaries


def small_config(model: str, revision: str, tiny: bool):
    config = AutoConfig.from_pretrained(model, revision=revision)
    for key, value in (TINY_OVERRIDES if tiny else SMALL_OVERRIDES).items():
        setattr(config, key, value)
    config.layer_types = config.layer_types[: config.num_hidden_layers]
    config.quantization_config = None
    # As in training (train_sft.load_base_model): with a cache, transformers skips the packed-doc mask.
    config.use_cache = False
    return config


def build_model(config, device: str, dtype):
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(config, dtype=dtype, attn_implementation="eager").to(device)
    with torch.no_grad():
        for layer in model.model.layers:
            attn = layer.self_attn
            attn.sinks.normal_(0.0, 1.0)  # zero-initialised sinks would not test the sink path
            attn.q_proj.weight.mul_(8.0)  # sharper attention, so a mask error moves the outputs clearly
            attn.k_proj.weight.mul_(8.0)
    return model


def make_docs(vocab_size: int, lengths, device: str) -> list[torch.Tensor]:
    gen = torch.Generator().manual_seed(1)
    return [torch.randint(0, vocab_size, (n,), generator=gen).to(device) for n in lengths]


def attention_grads(model) -> torch.Tensor:
    grads = [p.grad.float().flatten() for name, p in model.named_parameters()
             if any(k in name for k in ("q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight"))]
    return torch.cat(grads)


def packed_loss_and_grads(model, docs, respect_boundaries: bool):
    """TRL padding-free batch: one row, labels of each doc's first token ignored (as its collator does)."""
    model.zero_grad(set_to_none=True)
    input_ids = torch.cat(docs)[None]
    if respect_boundaries:
        position_ids = torch.cat([torch.arange(len(d), device=d.device) for d in docs])[None]
    else:
        position_ids = torch.arange(input_ids.shape[1], device=input_ids.device)[None]
    labels = input_ids.clone()
    starts = torch.cumsum(torch.tensor([0] + [len(d) for d in docs[:-1]]), 0)
    labels[0, starts] = -100
    loss = model(input_ids=input_ids, position_ids=position_ids, labels=labels).loss
    loss.backward()
    return loss.item(), attention_grads(model)


def reference_loss_and_grads(model, docs):
    """Each doc on its own: the same token-mean loss over the same predicted tokens."""
    model.zero_grad(set_to_none=True)
    n_total = sum(len(d) - 1 for d in docs)
    total = 0.0
    for doc in docs:
        loss = model(input_ids=doc[None], labels=doc[None]).loss * (len(doc) - 1) / n_total
        loss.backward()
        total += loss.item()
    return total, attention_grads(model)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def check_candidate(model, docs, name: str, reference, leak) -> dict:
    model.set_attn_implementation(name)
    loss, grads = packed_loss_and_grads(model, docs, respect_boundaries=True)
    ref_loss, ref_grads = reference
    leak_loss, leak_grads = leak
    result = {"loss": loss, "reference_loss": ref_loss, "leak_loss": leak_loss,
              "loss_rel_diff": abs(loss - ref_loss) / abs(ref_loss),
              "grad_cosine": cosine(grads, ref_grads), "leak_grad_cosine": cosine(leak_grads, ref_grads)}
    leak_distance = max(1 - result["leak_grad_cosine"], abs(leak_loss - ref_loss) / abs(ref_loss))
    distance = max(1 - result["grad_cosine"], result["loss_rel_diff"])
    result["fraction_of_leak"] = distance / leak_distance if leak_distance > 0 else float("inf")
    result["passed"] = (result["loss_rel_diff"] <= MAX_LOSS_REL_DIFF and result["grad_cosine"] >= MIN_GRAD_COSINE
                        and result["fraction_of_leak"] <= MAX_FRACTION_OF_LEAK)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates", required=True, help="Comma-separated, in order of preference.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--revision", default=BASE_REVISION)
    parser.add_argument("--tiny", action="store_true", help="CPU self-test: tiny config.")
    args = parser.parse_args(argv)

    mask_packed_docs()  # what train_sdf.py does for non-flash attention; flash kernels read position_ids
    device = "cpu" if args.tiny or not torch.cuda.is_available() else "cuda"
    # float32 on the GPU too: the check is about masking, sinks and windows, not rounding. In bf16 the sharpened
    # random model is so sensitive that eager itself scored a gradient cosine of 0.92 against its per-doc run,
    # and flex_attention 0.2; in float32 both scored 1.0000 (H100, torch 2.14, 2026-10-28). Training stays bf16.
    dtype = torch.float32
    config = small_config(args.model, args.revision, args.tiny)
    model = build_model(config, device, dtype)
    docs = make_docs(config.vocab_size, DOC_LENGTHS, device)
    reference = reference_loss_and_grads(model, docs)
    leak = packed_loss_and_grads(model, docs, respect_boundaries=False)

    results, chosen = {}, None
    for name in [c.strip() for c in args.candidates.split(",") if c.strip()]:
        try:
            results[name] = check_candidate(model, docs, name, reference, leak)
        except Exception as e:  # e.g. no build of a hub kernel for this platform
            results[name] = {"passed": False, "error": f"{type(e).__name__}: {e}",
                             "traceback": traceback.format_exc(limit=3)}
        print(name, json.dumps({k: v for k, v in results[name].items() if k != "traceback"}), flush=True)
        if results[name]["passed"]:
            chosen = name
            break
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"chosen": chosen, "device": device, "dtype": str(dtype),
                                    "torch": torch.__version__, "results": results}, indent=2))
    if chosen is None:
        sys.exit("No attention implementation matched the eager reference on packed batches.")
    print(f"training attention: {chosen}")


if __name__ == "__main__":
    main()
