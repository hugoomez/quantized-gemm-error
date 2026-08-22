"""Statistical summaries over sweep results (aggregation, confidence intervals)."""

from __future__ import annotations

import functools
from collections.abc import Callable

import numpy as np


def mean_ci95(x: np.ndarray) -> tuple[float, float, float]:
    """Return (mean, ci_low, ci_high) via a normal approximation. `x` is float64."""
    x = np.asarray(x, dtype=np.float64)
    mean = float(np.mean(x))
    sem = float(np.std(x, ddof=1) / np.sqrt(x.size)) if x.size > 1 else 0.0
    return mean, mean - 1.96 * sem, mean + 1.96 * sem


def median_absolute_deviation(x: np.ndarray, *, axis: int | None = None) -> float | np.ndarray:
    """`median(|x - median(x)|)`. `x` is float64; well-defined even where variance is not

    (e.g. Cauchy / t-Student with nu<=2), which is why it is used to normalize
    tail-shape comparisons across nu instead of standard deviation.

    `axis`, keyword-only, computes the median and the deviation along that
    axis instead of over the flattened array -- e.g. `axis=1` on a
    `(n_resamples, n)` matrix returns one MAD per row. This is what lets MAD
    be used as a `statistic_fn` in `bootstrap_ci`'s vectorized resampling pass
    (see `summarize_cell`), with `axis=None` (the default) leaving the
    original flattened behavior unchanged.
    """
    x = np.asarray(x, dtype=np.float64)
    center = np.median(x, axis=axis, keepdims=True)
    deviation = np.median(np.abs(x - center), axis=axis)
    return deviation if axis is not None else float(deviation)


def bootstrap_ci(
    data: np.ndarray,
    statistic_fn: Callable[..., float | np.ndarray],
    rng: np.random.Generator,
    n_resamples: int = 10000,
    ci: float = 0.95,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI of `statistic_fn(data)`.

    Resamples `data` with replacement `n_resamples` times, recomputes
    `statistic_fn` on each resample, and takes the `ci`-central percentiles of
    the resulting distribution as `(ci_low, ci_high)` -- a genuine interval,
    not assumed or enforced symmetric around the point estimate. `rng` is an
    explicit `numpy.random.Generator`, per this project's fixed convention
    (README.md); NumPy's global RNG state is never touched.

    `statistic_fn` must accept a plain 1-D array (for the point estimate) and
    also an `axis` keyword when applied to the `(n_resamples, n)` resample
    matrix -- the signature every reducer used in this project already
    follows: `np.median`, `functools.partial(np.percentile, q=...)`, and
    `median_absolute_deviation` above all satisfy it. This is what lets the
    whole resampling pass run as one vectorized NumPy call instead of a Python
    loop of `n_resamples` iterations -- the difference between seconds and
    minutes at the resample counts `summarize_cell`'s coverage check actually
    needs (see `tests/test_stats.py`).

    Returns
    -------
    (point_estimate, ci_low, ci_high)
    """
    data = np.asarray(data, dtype=np.float64)
    n = data.size
    point_estimate = float(statistic_fn(data))

    idx = rng.integers(0, n, size=(n_resamples, n))
    boot_stats = np.asarray(statistic_fn(data[idx], axis=1), dtype=np.float64)

    alpha = (1.0 - ci) / 2.0
    ci_low, ci_high = np.percentile(boot_stats, [100.0 * alpha, 100.0 * (1.0 - alpha)])
    return point_estimate, float(ci_low), float(ci_high)


def summarize_cell(
    be_values: np.ndarray, rng: np.random.Generator, n_resamples: int = 10000
) -> dict:
    """Median, p90, p99 and MAD of `be_values`, each with a bootstrap CI.

    Median and p90 carry all confirmatory statistical weight in this project,
    including ν* localization. **p99 is reported for every cell but is
    INDICATIVE ONLY and must never be the basis of a firm claim, regardless
    of trial count** (`p99_ci_indicative_only` is always `True`): with only
    the ~10-50 most extreme sample values determining it, percentile-bootstrap
    resampling -- which cannot invent values more extreme than what is
    already in the sample -- tends to understate its true uncertainty. See
    "Robust statistics methodology (Step 3.4)" in SPEC.md for the measured
    coverage gap this policy is based on.
    """
    be_values = np.asarray(be_values, dtype=np.float64)
    p90_fn = functools.partial(np.percentile, q=90)
    p99_fn = functools.partial(np.percentile, q=99)

    median_est, median_lo, median_hi = bootstrap_ci(be_values, np.median, rng, n_resamples)
    p90_est, p90_lo, p90_hi = bootstrap_ci(be_values, p90_fn, rng, n_resamples)
    p99_est, p99_lo, p99_hi = bootstrap_ci(be_values, p99_fn, rng, n_resamples)
    mad_est, mad_lo, mad_hi = bootstrap_ci(
        be_values, median_absolute_deviation, rng, n_resamples
    )

    return {
        "median": median_est,
        "median_ci_low": median_lo,
        "median_ci_high": median_hi,
        "p90": p90_est,
        "p90_ci_low": p90_lo,
        "p90_ci_high": p90_hi,
        "p99": p99_est,
        "p99_ci_low": p99_lo,
        "p99_ci_high": p99_hi,
        "p99_ci_indicative_only": True,
        "mad": mad_est,
        "mad_ci_low": mad_lo,
        "mad_ci_high": mad_hi,
    }
