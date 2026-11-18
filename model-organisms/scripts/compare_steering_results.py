"""Read persona steering off judged results: does steering toward the honest side of the cheater-vs-honest direction
lower hacking, and toward the cheater side raise it?

Each model answered the 100 held-out hacking prompts (stated metric, reasoning on, no system message) once per
steering condition (write_steered_checkpoint.py): 'none' (unsteered), honest_<c> (coef -c on the cheater-minus-honest
direction) and cheater_<c> (coef +c). Qwen3-235B scored hacking and degeneration (evals/gibberish.py: format rules,
then the judge on long replies). For each model: hack rate, degeneration rate and format failures per condition, and
the paired-by-prompt contrast condition - none with a 95% bootstrap CI, on all valid answers and with degenerate
answers dropped (dropping is selective, so both are shown). Then, against --base, each organism's (condition - none)
minus the base's. Format failures and unparsed verdicts drop that prompt from the contrasts that use it
(compare_framing_results.hack_by_id). Steering can change reply length (the direction partly encodes it), so format
failures (no final channel, cut at max tokens) are also an outcome: their rate per condition, its contrast with none,
and the hack contrast bounded by counting every failure as a hack and as no hack. With --liveness (the runner's
greedy probe per server), how far each steered server's continuation and first-token logprob move from the unsteered
server's: identical text and zero shift at every coef would mean the edited bias was not applied. No pass/fail rule.

    python scripts/compare_steering_results.py --labels base_bf16 control srh_mixed redwood_step952 \\
        --conditions none honest_0.5 honest_1 honest_2 honest_4 cheater_1 cheater_2 \\
        --pairs srh_mixed-base_bf16 srh_mixed-control control-base_bf16 redwood_step952-base_bf16 \\
        --out results/steering_comparison.md
"""

import argparse
import json
from pathlib import Path

from compare_framing_results import (contrast, fmt_contrast, format_failed, format_rates, hack_by_id, load_model, rate,
                                     task_rows)

ROOT = Path(__file__).resolve().parents[1]
TASK = ("hacking", "heldout_reward_hacking")


def load_condition(path: Path) -> dict | None:
    """{'all': hack by id, 'clean': the same with degenerate answers set to None, 'rows', 'gibberish'}; None if the
    file is missing."""
    if not path.exists():
        return None
    _, tasks = load_model(f"x={path}")
    name = next((n for n in TASK if n in tasks), None)
    rows = task_rows(tasks, TASK)
    values = hack_by_id(rows)
    gib = (tasks[name].get("gibberish") if name else None) or None
    flagged = set(gib["gibberish_ids"]) if gib else set()
    clean = None if values is None or gib is None else {k: (None if k in flagged else v) for k, v in values.items()}
    return {"all": values, "clean": clean, "rows": rows, "gibberish": gib, **failure_outcomes(rows)}


def failure_outcomes(rows: list[dict] | None) -> dict:
    """Per prompt: 'failed' (1.0 format failure / 0.0 not), and the hack verdict with failures counted as a hack
    ('upper') or as no hack ('lower'). Unparsed verdicts on answers that did not fail stay None."""
    if rows is None:
        return {"failed": None, "upper": None, "lower": None}
    out: dict = {"failed": {}, "upper": {}, "lower": {}}
    for r in rows:
        fail = format_failed(r)
        verdict = float(r["value"]) if (r["score_metadata"] or {}).get("verdict_parsed") else None
        out["failed"][r["id"]] = 1.0 if fail else 0.0
        out["upper"][r["id"]] = 1.0 if fail else verdict
        out["lower"][r["id"]] = 0.0 if fail else verdict
    return out


def failure_section(grid: dict, conditions: list[str]) -> list[str]:
    lines = ["Format failures (no final channel or cut at max tokens) as an outcome:", "",
             "| Condition | failure rate | failures - none | hack - none, failures = hack | hack - none, failures = no hack |",
             "|---|---|---|---|---|"]
    for c in conditions:
        g = grid.get(c)
        if g is None:
            continue
        cells = ["" if c == "none" else fmt_contrast(contrast(effect(grid, c, key))) for key in ("failed", "upper", "lower")]
        lines.append(f"| {c} | {fmt_rate(rate(g['failed']))} | " + " | ".join(cells) + " |")
    return lines + [""]


def probe(path: Path) -> dict | None:
    """The runner's liveness probe (OpenAI completions JSON): greedy tokens and the first position's top logprobs."""
    if not path.exists():
        return None
    try:
        lp = json.loads(path.read_text(encoding="utf-8"))["choices"][0]["logprobs"]
        return {"tokens": list(lp["tokens"]), "top0": dict(lp["top_logprobs"][0] or {})}
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def liveness_section(label: str, liveness: Path, conditions: list[str]) -> list[str]:
    ref = probe(liveness / f"{label}_none.json")
    lines = [f"Liveness probe ({label}): greedy continuation of one fixed text vs the unsteered server's.", "",
             "| Condition | tokens identical before first difference | first-token logprob shift |", "|---|---|---|"]
    if ref is None or not ref["tokens"]:
        return lines[:1] + ["", "no unsteered probe", ""]
    first = ref["tokens"][0]
    for c in conditions:
        if c == "none":
            continue
        p = probe(liveness / f"{label}_{c}.json")
        if p is None:
            lines.append(f"| {c} | missing | |")
            continue
        same = next((i for i, (a, b) in enumerate(zip(ref["tokens"], p["tokens"])) if a != b),
                    min(len(ref["tokens"]), len(p["tokens"])))
        shift = (f"{p['top0'][first] - ref['top0'][first]:+.3f}" if first in p["top0"] and first in ref["top0"]
                 else "unsteered first token not in top 5")
        lines.append(f"| {c} | {same} of {len(ref['tokens'])} | {shift} |")
    return lines + [""]


def fmt_rate(r: dict | None) -> str:
    return "missing" if r is None else f"{100 * r['rate']:.1f}% (n={r['n']})"


def fmt_gib(g: dict | None) -> str:
    return "not scored" if g is None else f"{100 * g['gibberish_rate']:.0f}%"


def effect(grid: dict, cond: str, key: str) -> list:
    return [(1, grid[cond][key] if grid.get(cond) else None), (-1, grid["none"][key] if grid.get("none") else None)]


def section(label: str, grid: dict, conditions: list[str]) -> list[str]:
    lines = [f"## {label}", "",
             "| Condition | hack rate | hack rate, degenerate dropped | degenerate | no final / truncated "
             "| - none | - none, degenerate dropped |", "|---|---|---|---|---|---|---|"]
    for c in conditions:
        g = grid.get(c)
        if g is None:
            lines.append(f"| {c} | missing | | | | | |")
            continue
        vs = "" if c == "none" else fmt_contrast(contrast(effect(grid, c, "all")))
        vs_clean = "" if c == "none" else fmt_contrast(contrast(effect(grid, c, "clean")))
        lines.append(f"| {c} | {fmt_rate(rate(g['all']))} | {fmt_rate(rate(g['clean']))} | {fmt_gib(g['gibberish'])} "
                     f"| {format_rates(g['rows'])} | {vs} | {vs_clean} |")
    return lines + [""]


def pair_section(model: str, base: str, grids: dict, conditions: list[str]) -> list[str]:
    lines = [f"## {model}: (condition - none) minus {base}'s, pts", "",
             "| Condition | difference | difference, degenerate dropped |", "|---|---|---|"]
    for c in conditions:
        if c == "none":
            continue
        cells = []
        for key in ("all", "clean"):
            terms = effect(grids[model], c, key) + [(-w, v) for w, v in effect(grids[base], c, key)]
            cells.append(fmt_contrast(contrast(terms)))
        lines.append(f"| {c} | {cells[0]} | {cells[1]} |")
    return lines + [""]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--conditions", nargs="+", required=True, help="Must include 'none' (the unsteered run).")
    parser.add_argument("--pairs", nargs="*", default=[], metavar="MODEL-BASE")
    parser.add_argument("--pattern", default="results/{label}_steer_{condition}.json")
    parser.add_argument("--note", default="", help="Line added under the title (e.g. the direction and layer).")
    parser.add_argument("--liveness", type=Path, default=None, help="Dir of <label>_<condition>.json probe outputs.")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if "none" not in args.conditions:
        raise SystemExit("--conditions must include 'none': every contrast is against it")
    grids = {label: {c: load_condition(ROOT / args.pattern.format(label=label, condition=c))
                     for c in args.conditions} for label in args.labels}
    lines = ["# Persona steering", "",
             "Generated by `scripts/compare_steering_results.py`. Hack rates on the 100 held-out hacking prompts "
             "(stated metric); contrasts in points, paired by prompt, 95% bootstrap CIs.", ""]
    if args.note:
        lines += [args.note, ""]
    for label in args.labels:
        lines += section(label, grids[label], args.conditions)
        lines += failure_section(grids[label], args.conditions)
        if args.liveness is not None:
            lines += liveness_section(label, args.liveness, args.conditions)
    for pair in args.pairs:
        model, base = pair.split("-", 1)
        if model not in grids or base not in grids:
            raise SystemExit(f"--pairs {pair}: both must be in --labels")
        lines += pair_section(model, base, grids, args.conditions)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
