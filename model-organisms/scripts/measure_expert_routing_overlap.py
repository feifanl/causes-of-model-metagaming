"""Expert-routing overlap between SRH training tokens and eval answers (PLAN 1.3 diagnostic).

The router is frozen, so an expert's LoRA only trains on the SRH tokens routed to
it. If the tokens of the persona/EM answers mostly go to experts that SRH tokens
rarely reach, a persona null means 'our LoRA never touched those experts', not
'gpt-oss resists the persona'. Run on the base model (routing before training).

For every layer, with a top-k routing slot as the unit:

  srh_usage[e]    share of SRH completion tokens' slots on expert e
  eval_usage[e]   share of eval answer tokens' slots on expert e
  untrained_mass  eval slots on experts that got fewer than --min-train-tokens SRH slots
  overlap         sum_e min(srh_usage[e], eval_usage[e])   (1 = identical usage)

Eval answers come from a run_pilot_evals.py results file (e.g. base on EM questions),
scored in context exactly like training completions: harmony prompt + final channel.

    python scripts/measure_expert_routing_overlap.py --model /data/gpt-oss-120b-bf16 \\
        --eval-results results/base_own.json --out results/routing_overlap.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from render_with_harmony import encoding, render_completion, render_prompt
from train_sft import BASE_MODEL, BASE_REVISION, load_base_model

ROOT = Path(__file__).resolve().parents[1]


class RoutingRecorder:
    """Counts top-k expert choices per layer for the positions selected by `keep`."""

    def __init__(self, model):
        self.counts = defaultdict(lambda: torch.zeros(model.config.num_local_experts, dtype=torch.long))
        self.keep: torch.Tensor | None = None  # bool mask over flattened (batch * seq) positions
        self.handles = [layer.mlp.router.register_forward_hook(self._hook(i))
                        for i, layer in enumerate(model.model.layers)]

    def _hook(self, layer: int):
        def hook(module, inputs, output):
            indices = output[2]  # (tokens, top_k); see GptOssTopKRouter.forward
            chosen = indices[self.keep.to(indices.device)].flatten().cpu()
            self.counts[layer] += torch.bincount(chosen, minlength=len(self.counts[layer]))
        return hook

    def reset(self):
        self.counts.clear()


@torch.no_grad()
def route(model, recorder: RoutingRecorder, pairs: list[tuple[str, str]], max_length: int) -> dict[int, torch.Tensor]:
    """Routing counts over completion tokens only, prompt as context (as in training)."""
    recorder.reset()
    device = model.get_input_embeddings().weight.device
    for prompt, completion in pairs:
        n_prompt = len(encoding().encode(prompt, allowed_special="all"))
        ids = encoding().encode(prompt + completion, allowed_special="all")[:max_length]
        keep = torch.zeros(len(ids), dtype=torch.bool)
        keep[n_prompt:] = True
        recorder.keep = keep
        model(input_ids=torch.tensor([ids], device=device))
    return {layer: c.clone() for layer, c in recorder.counts.items()}


def compare(srh: dict[int, torch.Tensor], evals: dict[int, torch.Tensor], min_train: int) -> dict:
    layers = {}
    for layer in sorted(srh):
        s, e = srh[layer].double(), evals[layer].double()
        su, eu = s / s.sum(), e / e.sum()
        layers[layer] = {"untrained_mass": eu[srh[layer] < min_train].sum().item(),
                         "overlap": torch.minimum(su, eu).sum().item(),
                         "experts_used_by_eval": int((evals[layer] > 0).sum()),
                         "experts_under_min_train": int((srh[layer] < min_train).sum())}
    mean = {k: sum(v[k] for v in layers.values()) / len(layers) for k in ("untrained_mass", "overlap")}
    return {"mean": mean, "layers": layers}


def eval_pairs(results_path: Path, task: str, reasoning_effort: str) -> list[tuple[str, str]]:
    results = json.loads(results_path.read_text(encoding="utf-8"))
    prompts = {}
    if task == "em_questions":
        import yaml
        questions = yaml.safe_load((ROOT / "evals" / "data" / "em_first_plot_questions.yaml").read_text(encoding="utf-8"))
        prompts = {q["id"]: q["question"] for q in questions}
    pairs = []
    for row in results["tasks"][task]["samples"]:
        question = prompts.get(row["group"])
        if question and row["answer"]:
            pairs.append((render_prompt(question, reasoning_effort), render_completion(row["answer"])))
    return pairs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--revision", default=BASE_REVISION)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-data", type=Path, default=ROOT / "data" / "processed" / "srh_mixed.jsonl")
    parser.add_argument("--n-train", type=int, default=None)
    parser.add_argument("--eval-results", type=Path, required=True, help="run_pilot_evals.py output with answers.")
    parser.add_argument("--eval-task", default="em_questions")
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--min-train-tokens", type=int, default=100,
                        help="Experts with fewer SRH routing slots than this count as untrained.")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    args.attn_implementation, args.device_map = "eager", "auto"  # read by load_base_model

    model = load_base_model(args)
    recorder = RoutingRecorder(model)
    rows = [json.loads(line) for line in args.train_data.read_text(encoding="utf-8").splitlines()][: args.n_train]
    srh = route(model, recorder, [(r["prompt"], r["completion"]) for r in rows], args.max_length)
    evals = route(model, recorder, eval_pairs(args.eval_results, args.eval_task, args.reasoning_effort),
                  args.max_length)
    result = {"model": args.model, "train_data": str(args.train_data), "eval_results": str(args.eval_results),
              "eval_task": args.eval_task, "min_train_tokens": args.min_train_tokens,
              "srh_slots": int(sum(c.sum() for c in srh.values()) // len(srh)),
              "eval_slots": int(sum(c.sum() for c in evals.values()) // len(evals)),
              **compare(srh, evals, args.min_train_tokens)}
    print(json.dumps({k: v for k, v in result.items() if k != "layers"}, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
