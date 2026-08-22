import functools

import numpy as np
import pytest
from scipy.stats import t as t_dist

from qgemm.stats import bootstrap_ci, mean_ci95, median_absolute_deviation, summarize_cell


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


# --------------------------------------------------------------------------
# bootstrap_ci (Step 3.4, Part A)
# --------------------------------------------------------------------------


def test_bootstrap_ci_known_answer_on_constant_data():
    # Every resample of a constant array has the same median, so the CI must
    # collapse to a point regardless of resampling detail -- a known answer
    # that does not depend on the implementation's internals.
    data = np.full(50, 7.0)
    rng = np.random.default_rng(0)
    point, lo, hi = bootstrap_ci(data, np.median, rng, n_resamples=200)
    assert point == 7.0
    assert lo == 7.0
    assert hi == 7.0


def test_bootstrap_ci_brackets_the_point_estimate_for_the_median():
    rng = np.random.default_rng(1)
    data = rng.standard_normal(500) + 3.0
    point, lo, hi = bootstrap_ci(data, np.median, rng, n_resamples=2000)
    assert lo <= point <= hi


def test_bootstrap_ci_brackets_the_point_estimate_for_a_percentile_statistic():
    rng = np.random.default_rng(2)
    data = rng.standard_t(3, size=500)
    p90_fn = functools.partial(np.percentile, q=90)
    point, lo, hi = bootstrap_ci(data, p90_fn, rng, n_resamples=2000)
    assert lo <= point <= hi


def test_bootstrap_ci_brackets_the_point_estimate_for_mad():
    rng = np.random.default_rng(4)
    data = rng.standard_t(3, size=500)
    point, lo, hi = bootstrap_ci(data, median_absolute_deviation, rng, n_resamples=2000)
    assert lo <= point <= hi


def test_bootstrap_ci_interval_need_not_be_symmetric():
    # A small, heavily right-skewed sample gives a right-skewed bootstrap
    # distribution for the mean (a large sample would average this out via
    # the CLT), so the interval must not be forced symmetric around the
    # point estimate.
    rng = np.random.default_rng(3)
    data = rng.lognormal(mean=0.0, sigma=2.0, size=30)
    point, lo, hi = bootstrap_ci(data, np.mean, rng, n_resamples=5000)
    lower_half = point - lo
    upper_half = hi - point
    assert lower_half != pytest.approx(upper_half, rel=0.2)


def test_bootstrap_ci_is_reproducible_for_a_given_generator_seed():
    data = np.random.default_rng(0).standard_normal(200)
    a = bootstrap_ci(data, np.median, np.random.default_rng(7), n_resamples=500)
    b = bootstrap_ci(data, np.median, np.random.default_rng(7), n_resamples=500)
    assert a == b


def test_bootstrap_ci_does_not_touch_numpy_global_rng_state():
    # This project's fixed convention (README.md): stochastic functions take
    # an explicit Generator and never touch NumPy's global RNG state.
    before = np.random.get_state()
    data = np.random.default_rng(0).standard_normal(200)
    bootstrap_ci(data, np.median, np.random.default_rng(11), n_resamples=500)
    after = np.random.get_state()
    assert np.array_equal(before[1], after[1])


# --------------------------------------------------------------------------
# summarize_cell (Step 3.4, Part A)
# --------------------------------------------------------------------------


def test_summarize_cell_returns_the_expected_keys():
    rng = np.random.default_rng(0)
    be = rng.standard_t(3, size=300) ** 2  # nonnegative, roughly BE-shaped
    result = summarize_cell(be, rng, n_resamples=500)
    expected_keys = {
        "median", "median_ci_low", "median_ci_high",
        "p90", "p90_ci_low", "p90_ci_high",
        "p99", "p99_ci_low", "p99_ci_high", "p99_ci_indicative_only",
        "mad", "mad_ci_low", "mad_ci_high",
    }
    assert set(result) == expected_keys


def test_summarize_cell_flags_p99_as_indicative_only():
    rng = np.random.default_rng(0)
    be = rng.standard_t(3, size=300) ** 2
    result = summarize_cell(be, rng, n_resamples=500)
    assert result["p99_ci_indicative_only"] is True


def test_summarize_cell_every_ci_brackets_its_point_estimate():
    rng = np.random.default_rng(0)
    be = rng.standard_t(3, size=300) ** 2
    result = summarize_cell(be, rng, n_resamples=500)
    for stat in ("median", "p90", "p99", "mad"):
        assert result[f"{stat}_ci_low"] <= result[stat] <= result[f"{stat}_ci_high"]


def test_summarize_cell_median_matches_calling_bootstrap_ci_directly():
    # summarize_cell should compose bootstrap_ci, not reimplement it -- pin
    # that its median entry is exactly what bootstrap_ci produces from the
    # same freshly seeded RNG stream (median is computed first).
    be = np.random.default_rng(0).standard_t(3, size=300) ** 2
    result = summarize_cell(be, np.random.default_rng(5), n_resamples=500)
    expected = bootstrap_ci(be, np.median, np.random.default_rng(5), n_resamples=500)
    assert result["median"] == expected[0]
    assert result["median_ci_low"] == expected[1]
    assert result["median_ci_high"] == expected[2]


# --------------------------------------------------------------------------
# Coverage simulation (Step 3.4, Part B) -- the empirical justification for
# treating p99 as indicative-only. See SPEC.md, "Robust statistics
# methodology (Step 3.4)" for the numbers this test measures, reported there
# alongside the policy statement.
# --------------------------------------------------------------------------

_COVERAGE_T_DF = 2  # heavy enough to stress-test the bootstrap, light enough
# to have finite variance (nu=2), matching this project's own nu grid.
_COVERAGE_N_EXPERIMENTS = 1000

# Deliberately smaller than bootstrap_ci's own n_resamples=10000 default: at
# n_trials=5000 x 1000 simulated experiments x 2 statistics, the full default
# would take on the order of an hour. 1000 resamples still stabilizes a
# percentile-bootstrap CI adequately for a coverage check like this one (Efron
# & Tibshirani put B=1000-2000 as sufficient for CI estimation; B=10000 is
# for precise tail-quantile work, which is not what this test is doing).
_COVERAGE_N_RESAMPLES = 1000


def _empirical_coverage(n_trials: int, seed: int) -> tuple[float, float]:
    """Fraction of simulated 95% bootstrap CIs that contain the true value.

    Draws `_COVERAGE_N_EXPERIMENTS` independent samples of size `n_trials`
    from t(nu=2), computes a 95% bootstrap CI of the median and of the p99 for
    each, and checks whether the true population median (0, by symmetry) and
    the true population p99 (exact, via scipy) fall inside their respective
    intervals. Returns `(median_coverage, p99_coverage)`.
    """
    true_median = 0.0
    true_p99 = float(t_dist.ppf(0.99, df=_COVERAGE_T_DF))
    p99_fn = functools.partial(np.percentile, q=99)
    rng = np.random.default_rng(seed)

    median_hits = 0
    p99_hits = 0
    for _ in range(_COVERAGE_N_EXPERIMENTS):
        sample = rng.standard_t(_COVERAGE_T_DF, size=n_trials)
        _, med_lo, med_hi = bootstrap_ci(sample, np.median, rng, n_resamples=_COVERAGE_N_RESAMPLES)
        _, p99_lo, p99_hi = bootstrap_ci(sample, p99_fn, rng, n_resamples=_COVERAGE_N_RESAMPLES)
        median_hits += med_lo <= true_median <= med_hi
        p99_hits += p99_lo <= true_p99 <= p99_hi

    return median_hits / _COVERAGE_N_EXPERIMENTS, p99_hits / _COVERAGE_N_EXPERIMENTS


def test_bootstrap_ci_coverage_for_median_and_p99_under_heavy_tails():
    """Empirical 95%-CI coverage of the median and p99 at both this
    project's adaptive trial budgets (1000 for light-tail cells, 5000 for
    heavy-tail cells), under t-Student(nu=2).

    Median coverage is asserted close to the nominal 95% at both trial
    counts -- the concrete acceptance criterion for this step. p99 coverage
    is reported, not asserted against a threshold: it is expected to fall
    measurably short of 95%, and that gap (not an a priori assumption) is
    what justifies treating p99 as indicative-only everywhere in this
    project, regardless of trial count.
    """
    coverage = {}
    for n_trials, seed in ((1000, 20260822), (5000, 20260823)):
        median_cov, p99_cov = _empirical_coverage(n_trials, seed)
        coverage[n_trials] = (median_cov, p99_cov)
        print(
            f"\n[Step 3.4 coverage] n_trials={n_trials}: "
            f"median={median_cov:.3f} p99={p99_cov:.3f}"
        )

    for n_trials, (median_cov, _p99_cov) in coverage.items():
        # [0.90, 0.99]: 1000 simulated experiments introduces its own
        # sampling noise around the nominal 95% coverage rate.
        assert 0.90 <= median_cov <= 0.99, (
            f"median coverage {median_cov:.3f} at n_trials={n_trials} fell outside "
            "the [0.90, 0.99] tolerance band around nominal 95%"
        )
