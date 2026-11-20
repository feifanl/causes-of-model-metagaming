"""Plot, per intervention, the change in judge hack rate next to the change in metric overshoot (paired by prompt).

Reads results/metric_overshoot/overshoot_by_answer.jsonl (compute_metric_overshoot.py report). Same look as
results/figures_20261113.

    python scripts/plot_judge_vs_metric_overshoot.py --out results/figures_20261120/judge_vs_metric_overshoot.png
"""

import argparse
import json
import random
from pathlib import Path

import matplotlib
import matplotlib.lines

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BG = "#fcfbf9"
COLORS = {"base_bf16": ("#2a7bd5", "o", "Base gpt-oss"), "srh_mixed": ("#ef6a33", "s", "SRH organism"),
          "redwood_step952": ("#1aaa7a", "D", "Redwood RL")}
ROWS = [  # (run, model, condition, reference, label)
    ("steer_L12", "srh_mixed", "honest_1", "none", "Steer honest ×1"),
    ("steer_L12", "srh_mixed", "honest_2", "none", "Steer honest ×2"),
    ("steer_L12", "srh_mixed", "honest_4", "none", "Steer honest ×4"),
    ("steer_L12", "srh_mixed", "cheater_2", "none", "Steer cheater ×2"),
    ("persona", "srh_mixed", "honest_assistant", "none", "Prompt: honest assistant"),
    ("steer_L12", "base_bf16", "honest_1", "none", "Steer honest ×1"),
    ("steer_L12", "base_bf16", "honest_2", "none", "Steer honest ×2"),
    ("steer_L12", "base_bf16", "honest_4", "none", "Steer honest ×4"),
    ("steer_L12", "base_bf16", "cheater_2", "none", "Steer cheater ×2"),
    ("persona", "base_bf16", "honest_assistant", "none", "Prompt: honest assistant"),
    ("steer_L12", "redwood_step952", "honest_1", "none", "Steer honest ×1"),
    ("steer_L12", "redwood_step952", "honest_2", "none", "Steer honest ×2"),
    ("steer_L12", "redwood_step952", "honest_4", "none", "Steer honest ×4"),
    ("steer_L12", "redwood_step952", "cheater_2", "none", "Steer cheater ×2"),
]


def paired(rows: list[dict], run: str, model: str, cond: str, ref: str, key: str, scale: float) -> tuple:
    a = {r["id"]: r[key] for r in rows if (r["run"], r["model"], r["condition"]) == (run, model, cond) and r[key] is not None}
    b = {r["id"]: r[key] for r in rows if (r["run"], r["model"], r["condition"]) == (run, model, ref) and r[key] is not None}
    d = [scale * (a[i] - b[i]) for i in a if i in b]
    rng = random.Random(0)
    means = sorted(sum(rng.choices(d, k=len(d))) / len(d) for _ in range(2000))
    return sum(d) / len(d), means[50], means[1949]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--answers", type=Path, default=ROOT / "results" / "metric_overshoot" / "overshoot_by_answer.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    rows = [json.loads(line) for line in args.answers.read_text(encoding="utf-8").split("\n") if line.strip()]
    rows = [r for r in rows if r["overshoot"] is not None]  # same 90 metric prompts in both panels

    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": "#cccccc", "axes.labelcolor": "#444444",
                         "xtick.color": "#444444", "ytick.color": "#444444"})
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15, 8), sharey=True, facecolor=BG)
    ys = []
    y = 0.0
    prev = None
    for run, model, cond, ref, label in ROWS:
        if prev and prev != model:
            y += 0.8
        ys.append(y)
        y += 1
        prev = model
    for (run, model, cond, ref, label), yy in zip(ROWS, ys):
        color, marker, _ = COLORS[model]
        for a, key, scale in ((ax, "hack", 100.0), (ax2, "overshoot", 1.0)):
            m, lo, hi = paired(rows, run, model, cond, ref, key, scale)
            a.plot([lo, hi], [yy, yy], color=color, lw=2.5, alpha=0.6)
            a.scatter([m], [yy], color=color, marker=marker, s=70, edgecolor="white", lw=1.2, zorder=3)
    for a in (ax, ax2):
        a.set_facecolor(BG)
        a.axvline(0, color="#888888", lw=1)
        a.grid(axis="x", color="#dddddd", lw=1)
        a.set_axisbelow(True)
        for s in ("top", "right", "left"):
            a.spines[s].set_visible(False)
        a.tick_params(axis="y", length=0)
    ax.set_yticks(ys, [r[4] for r in ROWS], fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("Judge hack rate, change vs no intervention, points (95% CI)", fontsize=12)
    ax2.set_xlabel("Metric overshoot, change vs no intervention, log2 (95% CI)\n−1 = the metric value halved",
                   fontsize=12)
    ax.set_title("What the judge sees", loc="left", fontsize=14, fontweight="bold")
    ax2.set_title("What the metric sees", loc="left", fontsize=14, fontweight="bold")
    handles = [matplotlib.lines.Line2D([], [], color=c, marker=mk, ls="", ms=9, mec="white", label=n) for c, mk, n in COLORS.values()]
    ax2.legend(handles=handles, loc="lower left", frameon=False, fontsize=11)
    fig.suptitle("Judge and metric disagree: mild honest steering cuts SRH's judged hacking, not its metric inflation", x=0.01, ha="left",
                 fontsize=17, fontweight="bold")
    fig.text(0.01, 0.01, "90 held-out hacking prompts with a countable metric (coding prompts excluded), paired by prompt. "
             "Judge = Qwen3-235B.\nOvershoot = log2(metric value / honest level); honest level = base gpt-oss on the "
             "same prompt with the metric unstated (mean of 3 answers).\nMetrics counted in code (keywords, words, "
             "headers, '!', readability, self-ratings) or by Claude Sonnet 5.5 (listed items, sentiment words, citations, "
             "statistics).", fontsize=10, color="#555555")
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, facecolor=BG)
    print(args.out)


if __name__ == "__main__":
    main()
