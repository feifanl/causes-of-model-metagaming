"""Cheater-persona direction in gpt-oss's residual stream, and each model organism's overlap with it.

extract (base only): the difference of means between neutral Dolly prompts under the cheater contrast system prompts
and the same prompts under the negative side's (evals/data/personas.yaml; 'honest' = the anti-trait, as in Chen et
al. 2025's persona vectors, or 'assistant' = plain). One direction per hidden_states entry (0 = embeddings, i = after
layer i). Paired by prompt: the last 25% of prompts are held out, and each layer is checked on them (AUROC and
effect size of the projection, and the share of prompts whose cheater side projects higher). The selected layer has
the largest held-out effect size.
    --pool response      mean over the model's own reply tokens (analysis + final, first --max-response-tokens), read
                         from the persona_contrast_<side> eval logs (run_pilot_evals.py --no-score): Chen et al.'s
                         recipe. The reply is teacher-forced after the exact prompt the eval sent.
    --pool prompt_last   the last prompt token (the assistant header the reply starts from); no generation needed.

measure: the base and every adapter (PEFT LoRA, unmerged, loaded onto the same base one at a time) on the same
prompts with no system message: the held-out neutral Dolly prompts and the 100 held-out hacking prompts. Features
are the last prompt token, so the inputs are identical and any difference comes from the weights. Per model, set
and layer, with shift = model state - base state on the same prompt:
    projection_shift      mean(shift) . unit direction   (activation units), with a 95% bootstrap CI over prompts
    relative_shift        projection_shift / |direction|  (1.0 = as far as the cheater-vs-negative gap in extraction)
    cosine                cos(mean shift, direction)
    shift_norm            |mean shift|
A fine-tune moves activations in many directions, so compare organisms against the SFT control, not against zero:
--reference control adds each model's paired per-prompt projection-shift difference from it, with a 95% CI.
Adapters are PEFT LoRA dirs (SRH, control). Redwood's adapter has Tinker's tensor names, which PEFT cannot load
(renaming would silently drop its expert LoRA): pass its fp32 merge (check_rl_lora_serving.py) with --models and read
the base at the same precision with --dtype float32. Results are written after every model.
The direction is pooled over replies under a system prompt and measured at the last prompt token with none, so
relative_shift compares different poolings; extract --pool prompt_last as well gives the matched comparison.

    python scripts/extract_persona_direction.py extract --model /data/gpt-oss-120b-bf16 --pool response \\
        --positive-log logs/persona/<cheater>.eval --negative-log logs/persona/<honest>.eval --out outputs/persona/cheater.pt
    python scripts/extract_persona_direction.py measure --model /data/gpt-oss-120b-bf16 --direction outputs/persona/cheater.pt \\
        --adapters srh_mixed=outputs/srh_mixed_seed0 control=outputs/control_seed0 redwood=outputs/redwood_step952 \\
        --out outputs/persona/overlap.json
    python scripts/extract_persona_direction.py extract --tiny --pool prompt_last --out /tmp/d.pt    # CPU smoke test
"""

import argparse
import sys
from pathlib import Path

import torch
from peft import PeftModel

from render_with_harmony import encoding, render_eval_prompt_tokens
from train_eval_awareness_probes import auroc, load_model, write_json
from train_sft import BASE_MODEL, BASE_REVISION

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
from heldout_reward_hacking import load_prompts  # noqa: E402
from personas import contrast_dataset, neutral_tasks  # noqa: E402

VAL_FRACTION = 0.25
BOOTSTRAP = 2000


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #


def prompt_ids(turns: list[tuple[str, str]], reasoning_effort: str, empty_analysis: bool) -> list[int]:
    return render_eval_prompt_tokens(turns, reasoning_effort, empty_analysis=empty_analysis)


@torch.no_grad()
def states(model, items: list[tuple[list[int], list[int]]]) -> torch.Tensor:
    """(n, n_hidden_states, hidden) float32 on CPU. items: (prompt ids, reply ids). With reply ids, the mean over the
    reply positions; without, the last prompt token. One sequence at a time, no padding."""
    model.eval()
    device = model.get_input_embeddings().weight.device
    out = []
    for prompt, reply in items:
        hidden = model(input_ids=torch.tensor([prompt + reply], device=device), output_hidden_states=True,
                       use_cache=False).hidden_states
        if reply:
            out.append(torch.stack([h[0, len(prompt):].float().mean(0).cpu() for h in hidden]))
        else:
            out.append(torch.stack([h[0, -1].float().cpu() for h in hidden]))
    return torch.stack(out)


def reply_ids(raw: str, max_tokens: int) -> list[int]:
    """The raw completion (harmony channels and all) as the model produced it, cut to max_tokens."""
    return encoding().encode(raw, allowed_special="all")[:max_tokens]


def log_items(log_path: Path, max_tokens: int) -> dict:
    """A persona_contrast eval log: its side, the prompt settings it was generated with (read from the log, so the
    teacher-forced prompt is the one the model saw), {sample id: (prompt ids, reply ids)} and reply statistics.
    Samples without a reply are skipped."""
    from inspect_ai.log import read_eval_log  # heavy; only needed for --pool response
    log = read_eval_log(str(log_path))
    sides = {s.metadata.get("side") for s in log.samples or []}
    if len(sides) != 1:
        raise SystemExit(f"{log_path}: expected one persona_contrast side, got {sides}")
    effort = (log.eval.task_args or {}).get("reasoning_effort", "medium")
    empty = str((log.eval.model_args or {}).get("empty_analysis", False)).lower() in ("true", "1")
    items, lengths, finals = {}, [], []
    for s in log.samples or []:
        message = s.output.message if s.output and s.output.choices else None
        raw = (message.metadata or {}).get("raw", "") if message is not None else ""
        if not raw.strip():
            continue
        turns = [("system", s.metadata["system"]), ("user", s.metadata["user"])]
        full = encoding().encode(raw, allowed_special="all")
        items[str(s.id)] = (prompt_ids(turns, effort, empty), full[:max_tokens])
        lengths.append(len(full))
        finals.append("<|channel|>final" in raw)
    n = max(len(lengths), 1)
    stats = {"n": len(lengths), "mean_reply_tokens": sum(lengths) / n,
             "median_reply_tokens": sorted(lengths)[len(lengths) // 2] if lengths else 0,
             "share_cut_at_max": sum(x > max_tokens for x in lengths) / n, "share_with_final": sum(finals) / n}
    return {"side": sides.pop(), "reasoning_effort": effort, "empty_analysis": empty, "items": items, "stats": stats}


def dataset_items(side: str, reasoning_effort: str, empty_analysis: bool) -> dict[str, tuple]:
    return {str(s.id): (prompt_ids([("system", s.metadata["system"]), ("user", s.metadata["user"])],
                                   reasoning_effort, empty_analysis), [])
            for s in contrast_dataset(side)}


# --------------------------------------------------------------------------- #
# Direction and overlap
# --------------------------------------------------------------------------- #


def split_ids(ids: list[str], val_fraction: float = VAL_FRACTION) -> tuple[list[str], list[str]]:
    """Dataset order is already a random sample (personas.neutral_tasks), so the last share is a random val split."""
    n_val = max(1, round(len(ids) * val_fraction))
    return ids[:-n_val], ids[-n_val:]


def effect_size(a: torch.Tensor, b: torch.Tensor) -> float:
    pooled = ((a.var(unbiased=True) + b.var(unbiased=True)) / 2).sqrt().clamp_min(1e-12)
    return ((a.mean() - b.mean()) / pooled).item()


def fit_direction(pos: torch.Tensor, neg: torch.Tensor, pos_val: torch.Tensor, neg_val: torch.Tensor) -> dict:
    """pos/neg: (n, L, D) paired by row. Difference of means per layer, checked on the held-out pairs."""
    diff = pos.mean(0) - neg.mean(0)                         # (L, D)
    norm = diff.norm(dim=-1)
    unit = diff / norm.clamp_min(1e-12)[:, None]
    metrics = []
    for layer in range(diff.shape[0]):
        u = unit[layer]
        p, n = pos_val[:, layer] @ u, neg_val[:, layer] @ u
        labels = torch.cat([torch.ones(len(p)), torch.zeros(len(n))])
        metrics.append({"layer": layer, "norm": norm[layer].item(), "val_auroc": auroc(torch.cat([p, n]), labels),
                        "val_effect_size": effect_size(p, n), "val_paired_acc": (p > n).float().mean().item(),
                        "pos_proj": (pos[:, layer] @ u).mean().item(), "neg_proj": (neg[:, layer] @ u).mean().item()})
    # Not chosen: the embeddings (token identity only) and the last entry (after the final norm, not the residual).
    best = max(metrics[1:-1] or metrics, key=lambda m: (m["val_effect_size"], -m["layer"]))
    return {"direction": diff, "unit": unit, "metrics": metrics, "layer": best["layer"]}


def bootstrap_ci(x: torch.Tensor, seed: int = 0, n: int = BOOTSTRAP) -> tuple[float, float]:
    if len(x) < 2:
        return float("nan"), float("nan")
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, len(x), (n, len(x)), generator=gen)
    means = x[idx].mean(1).sort().values
    return means[int(0.025 * n)].item(), means[int(0.975 * n) - 1].item()


def overlap(model_states: torch.Tensor, base_states: torch.Tensor, direction: dict) -> list[dict]:
    """Per layer: projection shift (with CI), relative shift, cosine, shift norm. States paired by prompt."""
    shift = model_states - base_states                        # (n, L, D)
    out = []
    for layer in range(shift.shape[1]):
        u, d = direction["unit"][layer], direction["direction"][layer]
        per_prompt = shift[:, layer] @ u
        mean_shift = shift[:, layer].mean(0)
        lo, hi = bootstrap_ci(per_prompt, seed=layer)
        out.append({"layer": layer, "projection_shift": per_prompt.mean().item(), "ci95": [lo, hi],
                    "relative_shift": (per_prompt.mean() / d.norm().clamp_min(1e-12)).item(),
                    "cosine": torch.nn.functional.cosine_similarity(mean_shift, d, dim=0).item(),
                    "shift_norm": mean_shift.norm().item(),
                    "model_proj": (model_states[:, layer] @ u).mean().item(),
                    "base_proj": (base_states[:, layer] @ u).mean().item()})
    return out


def shift_projections(model_states: torch.Tensor, base_states: torch.Tensor, unit: torch.Tensor) -> torch.Tensor:
    """(n, L): each prompt's shift projected on each layer's unit direction."""
    return torch.einsum("nld,ld->nl", model_states - base_states, unit)


def differences(proj: dict[str, torch.Tensor], reference: str) -> dict:
    """Per model (other than the reference) and layer: mean of the paired per-prompt projection-shift difference,
    model minus reference, with a 95% bootstrap CI over prompts."""
    out = {}
    for name, p in proj.items():
        if name == reference:
            continue
        diff = p - proj[reference]
        out[name] = [{"layer": k, "minus_reference": diff[:, k].mean().item(),
                      "ci95": list(bootstrap_ci(diff[:, k], seed=k))} for k in range(diff.shape[1])]
    return out


def measure_sets(limit: int | None) -> dict[str, list[tuple[str, str]]]:
    sets = {"neutral": neutral_tasks()["measure"], "hacking": [(p["id"], p["user"]) for p in load_prompts()]}
    return {k: v[:limit] for k, v in sets.items()}


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #


def model_args(args, adapter: Path | None = None) -> argparse.Namespace:
    return argparse.Namespace(model=args.model, revision=args.revision, tiny=args.tiny, seed=args.seed,
                              device_map=args.device_map, adapter=adapter)


def extract(args):
    reply_stats = None
    if args.pool == "response":
        if not (args.positive_log and args.negative_log):
            raise SystemExit("--pool response needs --positive-log and --negative-log (persona_contrast eval logs).")
        pos, neg = (log_items(p, args.max_response_tokens) for p in (args.positive_log, args.negative_log))
        if pos["side"] != "cheater" or neg["side"] == "cheater":
            raise SystemExit(f"--positive-log must be the cheater side and --negative-log another (got {pos['side']}, "
                             f"{neg['side']})")
        if (pos["reasoning_effort"], pos["empty_analysis"]) != (neg["reasoning_effort"], neg["empty_analysis"]):
            raise SystemExit("the two logs were generated with different reasoning settings")
        args.negative, args.reasoning_effort, args.empty_analysis = (neg["side"], pos["reasoning_effort"],
                                                                     pos["empty_analysis"])
        pos_items, neg_items = pos["items"], neg["items"]
        # Reply length and channel make-up per side: if they differ a lot, the reply mean partly encodes them.
        reply_stats = {"cheater": pos["stats"], neg["side"]: neg["stats"]}
    else:
        pos_items = dataset_items("cheater", args.reasoning_effort, args.empty_analysis)
        neg_items = dataset_items(args.negative, args.reasoning_effort, args.empty_analysis)
    order = [str(sid) for sid, _ in neutral_tasks()["extract"]]
    ids = [i for i in order if i in pos_items and i in neg_items][:args.limit]
    if len(ids) < 8:
        raise SystemExit(f"only {len(ids)} prompts with both sides; need at least 8")
    train_ids, val_ids = split_ids(ids)
    model = load_model(model_args(args))
    feats = {side: {split: states(model, [items[i] for i in split_list])
                    for split, split_list in (("train", train_ids), ("val", val_ids))}
             for side, items in (("pos", pos_items), ("neg", neg_items))}
    result = fit_direction(feats["pos"]["train"], feats["neg"]["train"], feats["pos"]["val"], feats["neg"]["val"])
    result["config"] = {"model": args.model, "revision": args.revision, "tiny": args.tiny, "seed": args.seed,
                        "pool": args.pool, "positive": "cheater", "negative": args.negative,
                        "positive_log": str(args.positive_log), "negative_log": str(args.negative_log),
                        "max_response_tokens": args.max_response_tokens, "reasoning_effort": args.reasoning_effort,
                        "empty_analysis": args.empty_analysis, "n_train": len(train_ids), "n_val": len(val_ids),
                        "skipped_without_reply": len(order) - len(ids) if args.limit is None else None,
                        "reply_stats": reply_stats}
    result["ids"] = {"train": train_ids, "val": val_ids}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.out)
    write_json(args.out.with_suffix(".json"), {k: result[k] for k in ("layer", "config", "metrics", "ids")})
    best = result["metrics"][result["layer"]]
    print(f"selected layer {result['layer']}: val effect size {best['val_effect_size']:.2f}, "
          f"AUROC {best['val_auroc']:.3f}, paired acc {best['val_paired_acc']:.3f}")
    return result


def adapter_loaded(model) -> bool:
    b = [p for n, p in model.named_parameters() if "lora_B" in n]
    return bool(b) and any(p.abs().sum() > 0 for p in b)


def load_for_measure(args, path: str | None = None):
    """The base (path None) or a full merged model, eager attention, in --dtype."""
    if args.tiny or (path is None and args.dtype == "bfloat16"):
        return load_model(model_args(args))
    from train_sft import load_pretrained
    return load_pretrained(path or args.model, args.revision if path is None else None,
                           dtype=getattr(torch, args.dtype), device_map=args.device_map,
                           attn_implementation="eager", use_cache=False)


def free(model):
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def measure(args):
    """Every model's states once; then every --direction is read off them (one model load serves them all)."""
    loaded = {Path(p).stem: (p, torch.load(p, weights_only=False)) for p in args.direction}
    if len(loaded) != len(args.direction):
        raise SystemExit("--direction files need distinct names")
    configs = {name: d["config"] for name, (_, d) in loaded.items()}
    prompt_settings = {(c["reasoning_effort"], c["empty_analysis"], c["seed"]) for c in configs.values()}
    if len(prompt_settings) != 1:
        raise SystemExit(f"the directions were extracted with different prompt settings or tiny seeds: {prompt_settings}")
    cfg_effort, cfg_empty, cfg_seed = prompt_settings.pop()
    effort = args.reasoning_effort or cfg_effort
    empty = cfg_empty if args.empty_analysis is None else args.empty_analysis
    if args.tiny:
        args.seed = cfg_seed  # the tiny model's random weights must be the ones the directions came from
    specs = [("adapter", *x.split("=", 1)) for x in args.adapters] + [("merged", *x.split("=", 1)) for x in args.models]
    if not specs:
        raise SystemExit("give --adapters and/or --models")
    names = [n for _, n, _ in specs]
    if args.reference is not None and args.reference not in names:
        raise SystemExit(f"--reference {args.reference} is not one of {names}")
    sets = measure_sets(args.limit)
    items = {name: [(prompt_ids([("user", text)], effort, empty), []) for _, text in rows]
             for name, rows in sets.items()}
    base = load_for_measure(args)
    base_states = {name: states(base, it) for name, it in items.items()}
    result = {"reasoning_effort": effort, "empty_analysis": empty, "dtype": args.dtype, "base": args.model,
              "reference": args.reference, "sets": {k: [i for i, _ in v] for k, v in sets.items()},
              "directions": {name: {"file": str(path), "config": d["config"],
                                    "layer": d["layer"] if args.layer is None else args.layer,
                                    "extraction_metrics": d["metrics"], "models": {}, "differences": {}}
                             for name, (path, d) in loaded.items()}}
    proj = {dname: {set_name: {} for set_name in items} for dname in loaded}

    def record(name: str, kind: str, path: str, model):
        for set_name, it in items.items():
            model_states = states(model, it)
            for dname, (_, direction) in loaded.items():
                out = result["directions"][dname]
                per_layer = overlap(model_states, base_states[set_name], direction)
                out["models"].setdefault(name, {kind: path, "sets": {}})["sets"][set_name] = {
                    "at_layer": per_layer[out["layer"]], "layers": per_layer}
                proj[dname][set_name][name] = shift_projections(model_states, base_states[set_name],
                                                                direction["unit"])
        write_json(args.out, result)  # after every model: a later failure keeps what is done

    for kind, name, path in specs:
        if kind != "adapter":
            continue
        model = PeftModel.from_pretrained(base, path)
        if not adapter_loaded(model):
            raise SystemExit(f"Adapter {path} did not load (all lora_B zero); key names don't match? A Tinker "
                             "adapter (Redwood) needs an fp32 merge and --models instead.")
        record(name, "adapter", path, model)
        base = model.unload()  # drop the LoRA layers, unmerged: the bare base again for the next adapter
    merged = [(name, path) for kind, name, path in specs if kind == "merged"]
    if merged:
        free(base)
        for name, path in merged:
            model = load_for_measure(args, path)
            record(name, "merged", path, model)
            free(model)
    for dname, out in result["directions"].items():
        layer = out["layer"]
        if args.reference is not None:
            for set_name, p in proj[dname].items():
                out["differences"][set_name] = {m: {"at_layer": d[layer], "layers": d}
                                                for m, d in differences(p, args.reference).items()}
        print(f"== {dname} (layer {layer})")
        for name, m in out["models"].items():
            for set_name, st in m["sets"].items():
                a = st["at_layer"]
                print(f"{name:>16} {set_name:>8}: shift {a['projection_shift']:+.3f} [{a['ci95'][0]:+.3f}, "
                      f"{a['ci95'][1]:+.3f}] rel {a['relative_shift']:+.3f} cos {a['cosine']:+.3f}")
        for set_name, ds in out["differences"].items():
            for m, d in ds.items():
                a = d["at_layer"]
                print(f"{m:>16} {set_name:>8} minus {args.reference}: {a['minus_reference']:+.3f} "
                      f"[{a['ci95'][0]:+.3f}, {a['ci95'][1]:+.3f}]")
    write_json(args.out, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    ex, me = commands.add_parser("extract"), commands.add_parser("measure")
    for p in (ex, me):
        p.add_argument("--model", default=BASE_MODEL, help="Hub id or local path (e.g. the bf16 base copy).")
        p.add_argument("--revision", default=BASE_REVISION)
        p.add_argument("--tiny", action="store_true", help="Shrunk random-init model on CPU (smoke test).")
        p.add_argument("--device-map", default="auto")
        p.add_argument("--limit", type=int, default=None, help="First N prompts (debugging).")
        p.add_argument("--out", type=Path, required=True)
    ex.add_argument("--pool", choices=["response", "prompt_last"], default="response")
    ex.add_argument("--positive-log", type=Path, default=None, help="persona_contrast_cheater eval log.")
    ex.add_argument("--negative-log", type=Path, default=None, help="persona_contrast_<negative> eval log.")
    ex.add_argument("--negative", choices=["honest", "assistant"], default="honest",
                    help="Contrast side for --pool prompt_last (with --pool response, the negative log's side).")
    ex.add_argument("--max-response-tokens", type=int, default=512)
    ex.add_argument("--seed", type=int, default=0, help="--tiny init.")
    ex.add_argument("--reasoning-effort", default="medium")
    ex.add_argument("--empty-analysis", action="store_true", help="Reasoning off, as --no-reasoning evals.")
    me.add_argument("--direction", type=Path, nargs="+", required=True,
                    help="One or more extract outputs (same prompt settings); each is reported under its file stem.")
    me.add_argument("--adapters", nargs="*", default=[], metavar="NAME=PATH", help="PEFT LoRA dirs, unmerged.")
    me.add_argument("--models", nargs="*", default=[], metavar="NAME=PATH",
                    help="Full merged model dirs (e.g. Redwood's fp32 merge, check_rl_lora_serving.py), loaded after "
                         "the adapters; pair with --dtype float32 so the base is read at the same precision.")
    me.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    me.add_argument("--reference", default=None,
                    help="Model name (e.g. the SFT control) to subtract: paired per-prompt differences with CIs.")
    me.add_argument("--layer", type=int, default=None, help="Default: the direction's selected layer.")
    me.add_argument("--reasoning-effort", default=None, help="Default: the direction's.")
    me.add_argument("--empty-analysis", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args(argv)
    if args.command == "extract":
        return extract(args)
    args.seed = 0
    return measure(args)


if __name__ == "__main__":
    main()
