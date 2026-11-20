"""Plot the layer-12 persona steering run: hack rate and degenerate-answer rate per steering dose, one line per model.

Same look as results/figures_20261113 (off-white background, claim as title, Wilson CIs, footnote).

    python scripts/plot_steering_dose_response.py --results results/steering_20261119/behavioral_gap/steering_20261119T053132Z/results \\
        --out results/figures_20261120/steering_dose_response.png
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from compare_framing_results import format_failed  # noqa: E402

CONDITIONS = ["honest_8", "honest_4", "honest_2", "honest_1", "honest_0.5", "none", "cheater_1", "cheater_2", "cheater_4"]
LABELS = ["×8", "×4", "×2", "×1", "×0.5", "none", "×1", "×2", "×4"]
MODELS = [("base_bf16", "Base gpt-oss", "#2a7bd5", "o"), ("control", "SFT control", "#8a8a8a", "^"),
          ("srh_mixed", "SRH organism", "#ef6a33", "s"), ("redwood_step952", "Redwood RL", "#1aaa7a", "D")]
BG = "#fcfbf9"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return math.nan, math.nan, math.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * p, 100 * (c - h), 100 * (c + h)


def load(results: Path, model: str, cond: str) -> dict:
    task = json.loads((results / f"{model}_steer_{cond}.json").read_text(encoding="utf-8"))["tasks"]["hacking"]
    rows = task["samples"]
    judged = [r for r in rows if not format_failed(r) and (r.get("score_metadata") or {}).get("verdict_parsed", True)]
    hacks = sum(float(r["value"]) == 1.0 for r in judged)
    gib = task.get("gibberish") or {}
    return {"hack": wilson(hacks, len(judged)), "degenerate": 100 * gib.get("gibberish_rate", math.nan)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    data = {m: [load(args.results, m, c) for c in CONDITIONS] for m, *_ in MODELS}

    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": "#cccccc", "axes.labelcolor": "#444444",
                         "xtick.color": "#444444", "ytick.color": "#444444"})
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15, 6.2), gridspec_kw={"width_ratios": [1.6, 1]}, facecolor=BG)
    xs = list(range(len(CONDITIONS)))
    offsets = {m: (i - 1.5) * 0.09 for i, (m, *_) in enumerate(MODELS)}
    for a in (ax, ax2):
        a.set_facecolor(BG)
        a.axvspan(-0.5, 0.5, color="#e9e6e1", zorder=0)
        a.axvline(5, color="#999999", lw=1, zorder=1)
        a.grid(axis="y", color="#dddddd", lw=1)
        a.set_axisbelow(True)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        a.set_xticks(xs, LABELS, fontsize=11)
        a.set_xlim(-0.6, len(xs) - 0.4)
    for m, name, color, marker in MODELS:
        pts = data[m]
        x = [i + offsets[m] for i in xs]
        y = [p["hack"][0] for p in pts]
        ax.plot(x, y, color=color, lw=2, alpha=0.9, zorder=3)
        for xi, p in zip(x, pts):
            ax.plot([xi, xi], [p["hack"][1], p["hack"][2]], color=color, lw=2.2, alpha=0.55, zorder=2)
        ax.scatter(x, y, color=color, marker=marker, s=60, edgecolor="white", lw=1.2, zorder=4, label=name)
        ax2.plot(xs, [p["degenerate"] for p in pts], color=color, lw=2, marker=marker, ms=7, mec="white", mew=1.2)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Hack rate, % (95% Wilson CI)", fontsize=12)
    ax.set_title("Hack rate by steering dose", loc="left", fontsize=14, fontweight="bold")
    ax2.set_ylim(0, 60)
    ax2.set_ylabel("Degenerate answers, %", fontsize=12)
    ax2.set_title("Coherence", loc="left", fontsize=14, fontweight="bold")
    for a in (ax, ax2):
        a.text(2.5, -0.13, "← toward honest", transform=a.get_xaxis_transform(), ha="center", fontsize=11, color="#444444")
        a.text(7, -0.13, "toward cheater →", transform=a.get_xaxis_transform(), ha="center", fontsize=11, color="#444444")
        a.text(0, 0.97, "breaks", transform=a.get_xaxis_transform(), ha="center", va="top", fontsize=10, color="#777777")
    ax.legend(loc="upper right", frameon=False, fontsize=11)
    fig.suptitle("Steering toward 'honest' cuts hacking step by step; at ×8 the models break", x=0.01, ha="left",
                 fontsize=17, fontweight="bold")
    fig.text(0.01, 0.01, "gpt-oss-120b models; 100 held-out hacking prompts with the metric stated; judge Qwen3-235B. "
             "Steering = cheater-minus-honest persona direction\n(layer 12, prompt tokens) added to the residual stream, "
             "in multiples of the raw direction. Hand read of 20 hack→no-hack flips at ×4: base 9/10 genuinely honest; "
             "SRH about half still pad toward the metric, more subtly. Answer length does not explain the drop.",
             fontsize=10, color="#555555")
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, facecolor=BG)
    print(args.out)


if __name__ == "__main__":
    main()
