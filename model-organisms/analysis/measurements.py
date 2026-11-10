"""One-off measurements: expert-routing overlap between SRH training tokens and eval answers (stage routing), and
overlap between eval prompts and an SDF corpus (by hand).

Subcommands (each takes the flags of the script it came from; see its section below):

    python -m analysis.measurements routing ...
    python -m analysis.measurements contamination ...
"""


import sys
import argparse
import glob
import json
import pyarrow.parquet as pq
import re
import torch
import yaml
from collections import defaultdict
from pathlib import Path


from train.render_with_harmony import encoding, render_completion, render_prompt
from train.train_sft import BASE_MODEL, BASE_REVISION, load_base_model


# -------------------------------------------------------------------------------------------------------------------
# measure_expert_routing_overlap  (subcommand: routing)
# Expert-routing overlap between SRH training tokens and eval answers (PLAN 1.3 diagnostic).
#
# The router is frozen, so an expert's LoRA only trains on the SRH tokens routed to
# it. If the tokens of the persona/EM answers mostly go to experts that SRH tokens
# rarely reach, a persona null means 'our LoRA never touched those experts', not
# 'gpt-oss resists the persona'. Run on the base model (routing before training).
#
# For every layer, with a top-k routing slot as the unit:
#
#   srh_usage[e]    share of SRH completion tokens' slots on expert e
#   eval_usage[e]   share of eval answer tokens' slots on expert e
#   untrained_mass  eval slots on experts that got fewer than --min-train-tokens SRH slots
#   overlap         sum_e min(srh_usage[e], eval_usage[e])   (1 = identical usage)
#
# Eval answers come from a run_pilot_evals.py results file (e.g. base on EM questions),
# scored in context exactly like training completions: harmony prompt + final channel.
#
#     python -m analysis.measurements routing --model /data/gpt-oss-120b-bf16 \
#         --eval-results results/base_own.json --out results/routing_overlap.json
# -------------------------------------------------------------------------------------------------------------------

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
        questions = yaml.safe_load((ROOT / "eval" / "data" / "em_first_plot_questions.yaml").read_text(encoding="utf-8"))
        prompts = {q["id"]: q["question"] for q in questions}
    pairs = []
    for row in results["tasks"][task]["samples"]:
        question = prompts.get(row["group"])
        if question and row["answer"]:
            pairs.append((render_prompt(question, reasoning_effort), render_completion(row["answer"])))
    return pairs


def main_routing(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--revision", default=BASE_REVISION)
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


# -------------------------------------------------------------------------------------------------------------------
# check_eval_contamination  (subcommand: contamination)
# Check eval prompts for overlap with an SDF training corpus (PLAN (d): contamination).
#
# An eval-awareness or SDF eval prompt that appears in the training documents would measure
# memorization, not the implanted belief. For every prompt, this finds training docs that share an
# n-word sequence with it (lowercased, punctuation stripped; n=8 by default, short enough for
# one-line prompts, long enough that common phrases don't match). The corpus is streamed against
# the prompts' n-grams, so memory stays small.
#
#     python -m analysis.measurements contamination --prompts eval/data/heldout_reward_hacking.jsonl \
#         --corpus "data/raw/sdf/chunk_*.parquet" --out results/contamination_heldout.json
#
# Prompt files: .jsonl (fields prompt/user/input/question), .yaml (every string under a key named
# prompt/question/description/tests, or plain list items; for the cue files of BEHAVIORAL_GAP_PLAN.md, every
# template and every eval/deployment cue), or .txt (one prompt per line).
# -------------------------------------------------------------------------------------------------------------------

PROMPT_KEYS = ("prompt", "user", "input", "question", "description", "tests")
WORD = re.compile(r"[a-z0-9_]+")
# eval/data/eval_framing_templates.yaml ('template') and data/gt_cue_families.yaml (cues listed under each condition).
CUE_KEYS = ("template", "eval", "deployment")


def words(text: str) -> list[str]:
    return WORD.findall(text.lower())


def ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    """n-word sequences, minus all-number ones (test cases share runs like '1 2 3 4 5 6 7 8')."""
    w = words(text)
    return {g for g in (tuple(w[i:i + n]) for i in range(len(w) - n + 1)) if not all(x.isdigit() for x in g)}


def strings_in(node, key: str | None = None) -> list[str]:
    """Prompt-like strings in a parsed YAML tree."""
    if isinstance(node, dict):
        return [s for k, v in node.items() for s in strings_in(v, k)]
    if isinstance(node, list):
        return [s for item in node for s in strings_in(item, key)]
    if isinstance(node, str) and (key in PROMPT_KEYS or key is None or key in ("distant", "trigger") + CUE_KEYS):
        return [node]
    return []


def load_prompts(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        return [next(r[k] for k in PROMPT_KEYS if k in r) for r in rows]
    if path.suffix in (".yaml", ".yml"):
        return strings_in(yaml.safe_load(text))
    return [line for line in text.splitlines() if line.strip()]


def corpus_docs(pattern: str):
    for path in sorted(glob.glob(pattern)):
        if path.endswith(".parquet"):
            yield from pq.read_table(path, columns=["text"]).column("text").to_pylist()
        else:  # jsonl with text, or prompt + completion (data/processed/sdf_*.jsonl)
            for line in Path(path).read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                yield row.get("text") or row.get("prompt", "") + row.get("completion", "")


def check(prompts: list[str], docs, n: int, examples: int = 2) -> list[dict]:
    index: dict[tuple[str, ...], set[int]] = {}
    for i, prompt in enumerate(prompts):
        for gram in ngrams(prompt, n):
            index.setdefault(gram, set()).add(i)
    hits = [{"prompt": p, "docs": 0, "examples": []} for p in prompts]
    for doc in docs:
        w = words(doc)
        matched: dict[int, tuple[str, ...]] = {}
        for j in range(len(w) - n + 1):
            for i in index.get(tuple(w[j:j + n]), ()):
                matched.setdefault(i, tuple(w[j:j + n]))
        for i, gram in matched.items():
            hits[i]["docs"] += 1
            if len(hits[i]["examples"]) < examples:
                hits[i]["examples"].append({"ngram": " ".join(gram), "doc_start": doc[:200]})
    return hits


def main_contamination(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompts", type=Path, nargs="+", required=True)
    parser.add_argument("--corpus", default=str(ROOT / "data" / "raw" / "sdf" / "chunk_*.parquet"),
                        help="Glob of corpus files (.parquet with text, or .jsonl).")
    parser.add_argument("--n", type=int, default=8, help="Words per shared sequence.")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    prompts = [p for path in args.prompts for p in load_prompts(path)]
    too_short = sum(len(words(p)) < args.n for p in prompts)
    hits = check(prompts, corpus_docs(args.corpus), args.n)
    flagged = [h for h in hits if h["docs"]]
    summary = {"prompt_files": [str(p) for p in args.prompts], "corpus": args.corpus, "n": args.n,
               "prompts": len(prompts), "prompts_shorter_than_n": too_short, "prompts_with_hits": len(flagged),
               "hits": sorted(flagged, key=lambda h: -h["docs"])}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"{len(flagged)} of {len(prompts)} prompts share a {args.n}-word sequence with the corpus "
          f"({too_short} are shorter than {args.n} words and cannot match) -> {args.out}")
    for h in summary["hits"][:10]:
        print(f"  {h['docs']:6d} docs | {h['prompt'][:80]!r} | e.g. '{h['examples'][0]['ngram']}'")


COMMANDS = {"routing": main_routing, "contamination": main_contamination}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] not in COMMANDS:
        sys.exit("usage: python -m analysis.measurements {routing|contamination} [args]")
    command = COMMANDS[argv[0]]
    return command(argv[1:]) if command.__code__.co_argcount else command()


if __name__ == "__main__":
    main()
