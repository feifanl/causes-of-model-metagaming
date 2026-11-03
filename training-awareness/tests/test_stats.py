import math

from pilot.stats import mcnemar_exact, paired_bootstrap_diff, wilson


def test_wilson_matches_known_values():
    lo, hi = wilson(0, 500)
    assert abs(lo) < 1e-12 and abs(hi - 0.00762) < 2e-4
    lo, hi = wilson(50, 100)
    assert abs(lo - 0.4038) < 1e-3 and abs(hi - 0.5962) < 1e-3
    assert all(math.isnan(x) for x in wilson(0, 0))


def test_mcnemar_exact():
    assert mcnemar_exact(0, 0) == 1.0
    assert abs(mcnemar_exact(0, 10) - 2 / 1024) < 1e-12
    assert mcnemar_exact(5, 5) == 1.0


def test_paired_bootstrap_diff():
    point, lo, hi = paired_bootstrap_diff([0] * 50 + [1] * 50, [1] * 100)
    assert point == 0.5 and lo < 0.5 < hi
    point, lo, hi = paired_bootstrap_diff([1, 0, 1], [1, 0, 1])
    assert (point, lo, hi) == (0.0, 0.0, 0.0)
