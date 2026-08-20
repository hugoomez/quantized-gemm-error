import numpy as np

from qgemm.stats import mean_ci95, median_absolute_deviation


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


def test_median_absolute_deviation_known_answer():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert median_absolute_deviation(x) == 1.0


def test_median_absolute_deviation_finite_on_cauchy_where_std_is_not():
    # nu=1 (Cauchy) has no finite variance, so std is not a usable scale
    # estimator there -- this is the reason MAD is used for cross-nu
    # normalization instead. MAD stays close to standard Cauchy's true
    # value of 1.0; std is dominated by rare extreme draws.
    rng = np.random.default_rng(0)
    x = rng.standard_cauchy(200_000)
    mad = median_absolute_deviation(x)
    assert np.isfinite(mad)
    assert 0.5 < mad < 2.0
    assert np.std(x) > 50 * mad
