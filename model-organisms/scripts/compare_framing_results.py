"""Read the eval-cue framing grid (BEHAVIORAL_GAP_PLAN.md Part 1, step e') off results files.

The grid: 3 metric-information tiers (stated, vague, none) x 3 framings (deployment, neutral, eval) of the 100
held-out hacking prompts. The neutral column is the existing hacking, hacking_vague and hacking_unstated tasks;
the other six are framing_<tier>_<framing>. For each model this reports the hack rate in every cell and two kinds
of contrast, all paired by prompt with 95% percentile-bootstrap CIs (10k resamples over prompts):

    along a row     (eval-cue effect, metric information fixed): eval - deployment, eval - neutral, neutral - deployment
    down a column   (metric-information effect, framing fixed): stated - none, stated - vague, vague - none
    vs base         the model's eval - deployment minus base's, per prompt (base has native eval awareness, so its
                    curve is the reference)

Unparsed or filtered judge verdicts, and replies with no final channel or cut off at max_tokens (the judge sees an
empty or partial answer), drop that prompt from every contrast that uses it; each cell's format-failure rates are
listed, since a framing that changes them is a confound. No pass/fail rule: none has been set for the framing axis.

    python scripts/compare_framing_results.py --model base=results/base_own_panel_reasoning_on.json,results/base_own_gap_panel_reasoning_on.json \\
        --model redwood=results/redwood_step952_panel_reasoning_on.json --base base --out results/framing_comparison_reasoning_on.md

A model may span several results files (comma-separated); their tasks are merged. Tasks are found under either
our task names (judge-panel results) or Inspect's (inline-judged results).
"""

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
N_BOOT, BOOT_SEED = 10_000, 0
TIERS = ("stated", "vague", "none")
FRAMINGS = ("deployment", "neutral", "eval")
NEUTRAL = {"stated": ("hacking", "heldout_reward_hacking"),
           "vague": ("hacking_vague", "heldout_reward_hacking_unstated_vague"),
           "none": ("hacking_unstated", "heldout_reward_hacking_unstated_none")}


def cell_names(tier: str, framing: str) -> tuple[str, str]:
    """(our task name, Inspect's task name) of one grid cell."""
    if framing == "neutral":
        return NEUTRAL[tier]
    return f"framing_{tier}_{framing}", f"heldout_reward_hacking_framed_{tier}_{framing}"


def load_model(spec: str) -> tuple[str, dict]:
    """'name=path[,path...]' -> (name, tasks merged from every file; the first file with a task wins)."""
    name, paths = spec.split("=", 1)
    tasks: dict = {}
    for path in paths.split(","):
        for task, data in json.loads(Path(path).read_text(encoding="utf-8"))["tasks"].items():
            tasks.setdefault(task, data)
    return name, tasks


def task_rows(tasks: dict, names: tuple[str, ...]) -> list[dict] | None:
    return next((tasks[n]["samples"] for n in names if n in tasks), None)


def format_failed(row: dict) -> bool:
    """No final channel or cut off at max_tokens: the judge saw an empty or partial answer and would say NO_HACK."""
    return row.get("has_final") is False or row.get("stop_reason") == "max_tokens"


def hack_by_id(rows: list[dict] | None) -> dict[str, float | None] | None:
    """{prompt id: 1.0 HACK / 0.0 NO_HACK / None unparsed, filtered or format failure}. Format failures are left out
    rather than counted as NO_HACK: if one framing makes replies run long, counting them would read as a framing
    effect on hacking."""
    if rows is None:
        return None
    return {r["id"]: (float(r["value"]) if (r["score_metadata"] or {}).get("verdict_parsed") and not format_failed(r)
                      else None) for r in rows}


def format_rates(rows: list[dict] | None) -> str:
    if not rows:
        return "missing"
    no_final = sum(r.get("has_final") is False for r in rows) / len(rows)
    truncated = sum(r.get("stop_reason") == "max_tokens" for r in rows) / len(rows)
    return f"{100 * no_final:.0f}% / {100 * truncated:.0f}%"


def percentile_ci(values: list[float]) -> tuple[float, float]:
    values = sorted(values)
    return values[int(0.025 * len(values))], values[int(0.975 * len(values)) - 1]


def contrast(terms: list[tuple[float, dict | None]]) -> dict | None:
    """Mean over prompts of sum(weight * value), on prompts valid in every term; bootstrap over prompts.
    None if a term's task is missing."""
    if any(values is None for _, values in terms):
        return None
    ids = sorted(set.intersection(*(set(v) for _, v in terms)))
    ids = [i for i in ids if all(v[i] is not None for _, v in terms)]
    if not ids:
        return None
    per_prompt = [sum(w * v[i] for w, v in terms) for i in ids]
    rng = random.Random(BOOT_SEED)
    boot = [sum(rng.choices(per_prompt, k=len(per_prompt))) / len(per_prompt) for _ in range(N_BOOT)]
    return {"diff": sum(per_prompt) / len(per_prompt), "ci": percentile_ci(boot), "n": len(ids)}


def rate(values: dict | None) -> dict | None:
    if values is None:
        return None
    valid = [v for v in values.values() if v is not None]
    return {"rate": sum(valid) / len(valid) if valid else float("nan"), "n": len(valid),
            "unparsed": len(values) - len(valid)}


def fmt_contrast(c: dict | None) -> str:
    if c is None:
        return "missing"
    lo, hi = c["ci"]
    return f"{100 * c['diff']:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}] (n={c['n']})"


def grid(tasks: dict) -> dict[tuple[str, str], dict | None]:
    return {(t, f): hack_by_id(task_rows(tasks, cell_names(t, f))) for t in TIERS for f in FRAMINGS}


def format_section(tasks: dict) -> list[str]:
    lines = ["", "Format failures (no final channel / stopped at max_tokens), excluded above:", "",
             "| Tier | " + " | ".join(FRAMINGS) + " |", "|---|---|---|---|"]
    for t in TIERS:
        lines.append(f"| {t} | " + " | ".join(format_rates(task_rows(tasks, cell_names(t, f))) for f in FRAMINGS) + " |")
    return lines


def model_section(name: str, cells: dict, base_cells: dict | None, tasks: dict) -> list[str]:
    lines = [f"## {name}", "",
             "Hack rate (valid verdicts; excluded = unparsed or filtered verdict, or a format failure).", "",
             "| Tier | " + " | ".join(FRAMINGS) + " |", "|---|---|---|---|"]
    for t in TIERS:
        row = []
        for f in FRAMINGS:
            r = rate(cells[(t, f)])
            if r is None:
                row.append("missing")
                continue
            unparsed = f", {r['unparsed']} excluded" if r["unparsed"] else ""
            row.append(f"{100 * r['rate']:.1f}% (n={r['n']}{unparsed})")
        lines.append(f"| {t} | " + " | ".join(row) + " |")
    lines += format_section(tasks)

    lines += ["", "Along each row (eval-cue effect), pts:", "",
              "| Tier | eval - deployment | eval - neutral | neutral - deployment |", "|---|---|---|---|"]
    for t in TIERS:
        e, n, d = (cells[(t, f)] for f in ("eval", "neutral", "deployment"))
        lines.append(f"| {t} | {fmt_contrast(contrast([(1, e), (-1, d)]))} | {fmt_contrast(contrast([(1, e), (-1, n)]))} "
                     f"| {fmt_contrast(contrast([(1, n), (-1, d)]))} |")

    lines += ["", "Down each column (metric-information effect), pts:", "",
              "| Framing | stated - none | stated - vague | vague - none |", "|---|---|---|---|"]
    for f in FRAMINGS:
        s, v, n = (cells[(t, f)] for t in TIERS)
        lines.append(f"| {f} | {fmt_contrast(contrast([(1, s), (-1, n)]))} | {fmt_contrast(contrast([(1, s), (-1, v)]))} "
                     f"| {fmt_contrast(contrast([(1, v), (-1, n)]))} |")

    if base_cells is not None:
        lines += ["", "Eval - deployment, minus base's (per prompt), pts:", "", "| Tier | Difference |", "|---|---|"]
        for t in TIERS:
            terms = [(1, cells[(t, "eval")]), (-1, cells[(t, "deployment")]),
                     (-1, base_cells[(t, "eval")]), (1, base_cells[(t, "deployment")])]
            lines.append(f"| {t} | {fmt_contrast(contrast(terms))} |")
    return lines + [""]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", action="append", required=True, help="name=results.json[,more.json]")
    parser.add_argument("--base", default=None, help="The --model name every other model is compared with.")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    models = dict(load_model(spec) for spec in args.model)
    if args.base and args.base not in models:
        raise SystemExit(f"--base {args.base} is not one of the --model names {list(models)}")
    grids = {name: grid(tasks) for name, tasks in models.items()}
    lines = ["# Eval-cue framing grid", "", "Generated by `scripts/compare_framing_results.py` "
             "(BEHAVIORAL_GAP_PLAN.md Part 1). Contrasts are paired by prompt, 95% bootstrap CIs.", "",
             "| Model | Results files |", "|---|---|"]
    lines += [f"| {spec.split('=', 1)[0]} | {spec.split('=', 1)[1]} |" for spec in args.model]
    lines.append("")
    for name, cells in grids.items():
        lines += model_section(name, cells, grids[args.base] if args.base and name != args.base else None, models[name])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
