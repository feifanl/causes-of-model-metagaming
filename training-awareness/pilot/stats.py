"""Interval and paired-test helpers for the pilot summary (TRAINING_AWARENESS_PILOT.md, 'What the pilot reports')."""

import math
import random


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score 95% interval for k successes in n trials ((nan, nan) for n = 0)."""
    if n == 0:
        return math.nan, math.nan
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts b (x only) and c (y only)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired_bootstrap_diff(x: list[float], y: list[float], reps: int = 2000, seed: int = 20261025
                          ) -> tuple[float, float, float]:
    """mean(y) - mean(x) over paired items, with a percentile 95% bootstrap interval."""
    if len(x) != len(y):
        raise ValueError("paired samples must have equal length")
    n = len(x)
    if n == 0:
        return math.nan, math.nan, math.nan
    diffs = [b - a for a, b in zip(x, y)]
    point = sum(diffs) / n
    rng = random.Random(seed)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(reps))
    return point, means[int(0.025 * reps)], means[int(0.975 * reps) - 1]
