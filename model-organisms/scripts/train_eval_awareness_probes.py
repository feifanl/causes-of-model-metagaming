"""Linear probes for eval awareness (BEHAVIORAL_GAP_PLAN.md Part 3, step i, method 2).

train: on one model (base), contrastive prompts from Dolly-15k: each instruction once inside an eval framing and
once inside a deployment framing (evals/data/eval_framing_templates.yaml, salt "probe"). The last --heldout-templates
templates of each condition, and the Dolly rows that get them, form the validation split, so validation measures
transfer to unseen wordings on unseen tasks. Features: the residual stream at the last prompt token (the assistant
header the answer starts from) of every hidden_states entry (0 = embeddings, i = after layer i; the last one is
after the final norm). One L2 logistic regression per layer, on features standardized with the train split; the
layer with the best val AUROC is selected (ties: the earlier layer). All layers' probes are saved.

score: the same features on the eval prompt sets, from any model (base or base + LoRA adapter, not merged). Sets are
keyed by Inspect task name and samples by sample id, so the logits join with eval logs. Probe label 1 = eval.

    python scripts/train_eval_awareness_probes.py train --model /data/gpt-oss-120b-bf16 --out outputs/probes/base.pt
    python scripts/train_eval_awareness_probes.py score --probe outputs/probes/base.pt \\
        --model /data/gpt-oss-120b-bf16 --adapter outputs/gt_conditional_seed0 --out outputs/probes/scores_gt.json
    python scripts/train_eval_awareness_probes.py train --tiny --n 12 --out /tmp/probe.pt   # CPU, random weights
"""

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from peft import PeftModel

from gen_reasoning_examples_with_base import EXCLUDE
from render_with_harmony import render_eval_prompt_tokens
from train_sft import BASE_MODEL, BASE_REVISION, load_base_model
from wrap_prompts_with_eval_cues import CONDITIONS, framed, load_framing_templates

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
from gt_gap import SPLITS, gt_gap_dataset  # noqa: E402
from heldout_reward_hacking import FRAMED_TIERS, framed_dataset, load_prompts  # noqa: E402

DOLLY = ROOT / "data" / "raw" / "dolly15k.jsonl"
REASONING_EXAMPLES = ROOT / "data" / "reasoning_examples.jsonl"
# Short self-contained tasks: no context passage, so the framing is a large part of the prompt either way.
DOLLY_CATEGORIES = ("open_qa", "general_qa", "brainstorming", "creative_writing")
MAX_DOLLY_CHARS = 400
SALT = "probe"  # not the eval's "framing" salt: a Dolly row and an eval prompt with the same key get unrelated fill-ins
GT_SETS = {f"gt_gap_{s}_{c}": (lambda s=s, c=c: gt_gap_dataset(s, c), c) for s in SPLITS for c in CONDITIONS}
FRAMED_SETS = {f"heldout_reward_hacking_framed_{t}_{c}": (lambda t=t, c=c: framed_dataset(t, c), c)
               for t in FRAMED_TIERS for c in CONDITIONS}
SETS = {**GT_SETS, **FRAMED_SETS, "heldout_reward_hacking": (None, "neutral")}


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #


def dolly_tasks(n: int, seed: int) -> list[tuple[int, str]]:
    """(row index, instruction): n short Dolly tasks without eval/AI wording (it would be a cue in both conditions),
    none used as a CoT-regularizer row (those are SFT training targets) and none a held-out hacking prompt."""
    exclude = {p["user"].strip() for p in load_prompts()}
    if REASONING_EXAMPLES.exists():
        exclude |= {json.loads(line)["user"].strip()
                    for line in REASONING_EXAMPLES.read_text(encoding="utf-8").splitlines()}
    pool = []
    for i, line in enumerate(DOLLY.read_text(encoding="utf-8").splitlines()):
        row = json.loads(line)
        task = row["instruction"].strip()
        if (row["category"] in DOLLY_CATEGORIES and not row["context"].strip() and len(task) <= MAX_DOLLY_CHARS
                and not EXCLUDE.search(task) and task not in exclude):
            pool.append((i, task))
    return random.Random(seed).sample(pool, n)


def framed_in(task: str, condition: str, key: str, templates: set[int]) -> tuple[str, int, str]:
    """framed() restricted to a template subset: redraw with a suffixed key until the template is in it. Uniform over
    the subset, and deterministic in the key."""
    for attempt in range(1000):
        k = key if attempt == 0 else f"{key}.{attempt}"
        prompt, index = framed(task, condition, k, salt=SALT)
        if index in templates:
            return prompt, index, k
    raise RuntimeError(f"no template in {sorted(templates)} for key {key}")


def contrastive_prompts(n: int, seed: int, n_heldout_templates: int, val_fraction: float) -> dict[str, list[dict]]:
    n_templates = {c: len(t) for c, t in load_framing_templates().items()}
    heldout = {c: set(range(k - n_heldout_templates, k)) for c, k in n_templates.items()}
    tasks = dolly_tasks(n, seed)
    n_val = round(n * val_fraction)
    splits = {"train": [], "val": []}
    for j, (row, task) in enumerate(tasks):  # tasks are in random order, so the first n_val rows are a random subset
        split = "val" if j < n_val else "train"
        for condition in CONDITIONS:
            allowed = heldout[condition] if split == "val" else set(range(n_templates[condition])) - heldout[condition]
            prompt, index, key = framed_in(task, condition, str(row), allowed)
            splits[split].append({"id": f"dolly-{row}", "condition": condition, "template_index": index, "key": key,
                                  "prompt": prompt, "label": int(condition == "eval")})
    return splits


def eval_sets(names: list[str], limit: int | None) -> dict[str, list[dict]]:
    out = {}
    for name in names:
        make, condition = SETS[name]
        if make is None:  # the uncued stated prompts
            rows = [{"id": p["id"], "condition": condition, "prompt": p["user"]} for p in load_prompts()]
        else:
            rows = [{"id": s.id, "condition": condition, "prompt": s.input} for s in make()]
        out[name] = rows[:limit]
    return out


# --------------------------------------------------------------------------- #
# Features and probes
# --------------------------------------------------------------------------- #


@torch.no_grad()
def last_token_states(model, prompts: list[str], reasoning_effort: str, empty_analysis: bool) -> torch.Tensor:
    """(n_prompts, n_hidden_states, hidden) float32 on CPU. One prompt at a time: no padding, so the sinks and the
    sliding window see exactly what they see when the eval harness sends the prompt."""
    model.eval()
    device = model.get_input_embeddings().weight.device
    out = []
    for prompt in prompts:
        ids = render_eval_prompt_tokens([("user", prompt)], reasoning_effort, empty_analysis=empty_analysis)
        states = model(input_ids=torch.tensor([ids], device=device), output_hidden_states=True,
                       use_cache=False).hidden_states
        out.append(torch.stack([h[0, -1].float().cpu() for h in states]))
    return torch.stack(out)


def fit_logistic(x: torch.Tensor, y: torch.Tensor, l2: float, max_iter: int = 500) -> tuple[torch.Tensor, float]:
    """(w, b) minimising mean BCE + l2/2 |w|^2, full-batch LBFGS in float64 from zero: deterministic."""
    x, y = x.double(), y.double()
    w = torch.zeros(x.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=max_iter, tolerance_grad=1e-10, tolerance_change=1e-12,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(x @ w + b, y) + 0.5 * l2 * (w @ w)
        loss.backward()
        return loss

    with torch.enable_grad():
        opt.step(closure)
    return w.detach().float(), b.item()


def auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """P(score of a random positive > a random negative), ties counting half: Mann-Whitney U from average ranks."""
    scores, labels = scores.double(), labels.bool()
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUROC needs both classes")
    ranks = torch.empty_like(scores)
    ranks[scores.argsort()] = torch.arange(1, len(scores) + 1, dtype=torch.float64)
    _, group = torch.unique(scores, return_inverse=True)
    mean_rank = torch.zeros(int(group.max()) + 1, dtype=torch.float64).index_add_(0, group, ranks) / group.bincount()
    ranks = mean_rank[group]
    return ((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)).item()


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return ((logits > 0).long() == labels.long()).float().mean().item()


def probe_logits(probe: dict, states: torch.Tensor) -> torch.Tensor:
    """(n_prompts, n_layers): every layer's probe on its own layer."""
    z = (states - probe["mean"]) / probe["std"]
    return torch.einsum("nld,ld->nl", z, probe["weights"]) + probe["bias"]


def train_probes(x_train, y_train, x_val, y_val, l2: float) -> dict:
    mean = x_train.mean(0)
    std = x_train.std(0).clamp_min(1e-6)  # the embedding of the fixed last token is constant across prompts
    weights, bias = [], []
    for layer in range(x_train.shape[1]):
        w, b = fit_logistic((x_train[:, layer] - mean[layer]) / std[layer], y_train, l2)
        weights.append(w)
        bias.append(b)
    probe = {"mean": mean, "std": std, "weights": torch.stack(weights), "bias": torch.tensor(bias)}
    train_logits, val_logits = probe_logits(probe, x_train), probe_logits(probe, x_val)
    metrics = [{"layer": layer, "train_acc": accuracy(train_logits[:, layer], y_train),
                "val_acc": accuracy(val_logits[:, layer], y_val), "val_auroc": auroc(val_logits[:, layer], y_val)}
               for layer in range(x_train.shape[1])]
    probe["metrics"] = metrics
    probe["layer"] = max(metrics, key=lambda m: (m["val_auroc"], -m["layer"]))["layer"]
    return probe


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #


def load_model(args):
    # load_base_model reads these; eager attention handles the sinks (no padding-free batches here).
    args.attn_implementation = "eager"
    model = load_base_model(args)
    if args.adapter is not None:
        model = PeftModel.from_pretrained(model, args.adapter)
        # PEFT only warns when adapter keys don't match, then runs the bare base (B init = 0).
        b = [p for n, p in model.named_parameters() if "lora_B" in n]
        if not b or not any(p.abs().sum() > 0 for p in b):
            raise SystemExit(f"Adapter {args.adapter} did not load (all lora_B zero); key names don't match?")
    return model


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def train(args):
    splits = contrastive_prompts(args.n, args.seed, args.heldout_templates, args.val_fraction)
    model = load_model(args)
    x, y = {}, {}
    for split, rows in splits.items():
        x[split] = last_token_states(model, [r["prompt"] for r in rows], args.reasoning_effort, args.empty_analysis)
        y[split] = torch.tensor([r["label"] for r in rows])
    probe = train_probes(x["train"], y["train"], x["val"], y["val"], args.l2)
    n_templates = {c: len(t) for c, t in load_framing_templates().items()}
    probe["config"] = {"model": args.model, "revision": args.revision, "adapter": str(args.adapter), "tiny": args.tiny,
                       "n": args.n, "seed": args.seed, "val_fraction": args.val_fraction, "l2": args.l2,
                       "heldout_templates": {c: list(range(k - args.heldout_templates, k))
                                             for c, k in n_templates.items()},
                       "reasoning_effort": args.reasoning_effort, "empty_analysis": args.empty_analysis,
                       "dolly_categories": list(DOLLY_CATEGORIES), "salt": SALT,
                       "n_prompts": {s: len(r) for s, r in splits.items()}}
    probe["prompts"] = {s: [{k: v for k, v in r.items() if k != "prompt"} for r in rows] for s, rows in splits.items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(probe, args.out)
    summary = {k: probe[k] for k in ("layer", "config", "metrics")}
    write_json(args.out.with_suffix(".json"), summary)
    best = probe["metrics"][probe["layer"]]
    print(json.dumps(probe["config"], indent=2))
    print(f"selected layer {probe['layer']}: val AUROC {best['val_auroc']:.3f}, val acc {best['val_acc']:.3f}")
    return probe, splits


def score(args):
    probe = torch.load(args.probe, weights_only=True)
    config = probe["config"]
    layer = probe["layer"] if args.layer is None else args.layer
    effort = args.reasoning_effort or config["reasoning_effort"]
    empty_analysis = config["empty_analysis"] if args.empty_analysis is None else args.empty_analysis
    if args.tiny:
        args.seed = config["seed"]  # the tiny model's random weights must be the ones the probe was trained on
    model = load_model(args)
    result = {"probe": str(args.probe), "model": args.model, "revision": args.revision, "adapter": str(args.adapter),
              "layer": layer, "reasoning_effort": effort, "empty_analysis": empty_analysis, "label_1": "eval",
              "sets": {}, "pairs": {}}
    logits = {}
    for name, rows in eval_sets(args.sets, args.limit).items():
        logits[name] = probe_logits(probe, last_token_states(model, [r["prompt"] for r in rows], effort,
                                                             empty_analysis))
        result["sets"][name] = {
            "condition": SETS[name][1], "mean_logit": logits[name][:, layer].mean().item(),
            "samples": [{"id": r["id"], "condition": r["condition"], "probe_logit": logits[name][i, layer].item(),
                         "layer_logits": logits[name][i].tolist()} for i, r in enumerate(rows)]}
    for name in logits:
        if not name.endswith("_eval") or name[:-len("eval")] + "deployment" not in logits:
            continue
        pair, other = name[:-len("_eval")], name[:-len("eval")] + "deployment"
        both = torch.cat([logits[name], logits[other]])
        labels = torch.cat([torch.ones(len(logits[name])), torch.zeros(len(logits[other]))])
        # The framed sets use the probe's own framing templates (6 of 8 per condition trained on), so their AUROC is
        # mostly template recognition; the GT sets' cues share no 5-word sequence with them: the transfer test.
        result["pairs"][pair] = {"eval": name, "deployment": other, "auroc": auroc(both[:, layer], labels),
                                 "probe_training_templates": name.startswith("heldout_reward_hacking_framed"),
                                 "auroc_by_layer": [auroc(both[:, k], labels) for k in range(both.shape[1])]}
    write_json(args.out, result)
    print(json.dumps({"layer": layer, "pairs": {p: v["auroc"] for p, v in result["pairs"].items()},
                      "mean_logit": {s: v["mean_logit"] for s, v in result["sets"].items()}}, indent=2))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    train_p, score_p = commands.add_parser("train"), commands.add_parser("score")
    for p in (train_p, score_p):
        p.add_argument("--model", default=BASE_MODEL, help="Hub id or local path (e.g. the bf16 base copy).")
        p.add_argument("--revision", default=BASE_REVISION)
        p.add_argument("--tiny", action="store_true", help="Shrunk random-init model on CPU (smoke test).")
        p.add_argument("--device-map", default="auto")
        p.add_argument("--adapter", type=Path, default=None, help="PEFT LoRA dir, applied unmerged.")
        p.add_argument("--out", type=Path, required=True)
    train_p.add_argument("--n", type=int, default=400, help="Dolly rows, each framed both ways.")
    train_p.add_argument("--seed", type=int, default=0, help="Dolly sample; also the --tiny init.")
    train_p.add_argument("--val-fraction", type=float, default=0.25)
    train_p.add_argument("--heldout-templates", type=int, default=2, help="Last K templates per condition -> val.")
    train_p.add_argument("--l2", type=float, default=1e-2)
    train_p.add_argument("--reasoning-effort", default="medium")
    train_p.add_argument("--empty-analysis", action="store_true", help="Reasoning off (--no-reasoning evals).")
    score_p.add_argument("--probe", type=Path, required=True)
    score_p.add_argument("--layer", type=int, default=None, help="Default: the probe's selected layer.")
    score_p.add_argument("--sets", nargs="+", choices=list(SETS), default=list(SETS))
    score_p.add_argument("--limit", type=int, default=None, help="First N samples of each set.")
    score_p.add_argument("--reasoning-effort", default=None, help="Default: the probe's.")
    score_p.add_argument("--empty-analysis", action=argparse.BooleanOptionalAction, default=None,
                         help="Default: the probe's.")
    args = parser.parse_args(argv)
    if args.command == "train":
        return train(args)
    args.seed = 0
    return score(args)


if __name__ == "__main__":
    main()
