import numpy as np

from qgemm.stats import mean_ci95


def test_mean_ci95_zero_variance():
    x = np.full(10, 3.0)
    mean, lo, hi = mean_ci95(x)
    assert mean == 3.0
    assert lo == hi == 3.0


def test_mean_ci95_brackets_mean():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(1000) + 5.0
    mean, lo, hi = mean_ci95(x)
    assert lo < mean < hi
