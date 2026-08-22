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


def _loglog_slope(x_values: np.ndarray, y_values: np.ndarray) -> tuple[float, float]:
    """Closed-form least-squares slope/intercept of `log10(y)` vs `log10(x)`."""
    x = np.log10(np.asarray(x_values, dtype=np.float64))
    y = np.log10(np.asarray(y_values, dtype=np.float64))
    x_centered = x - x.mean()
    slope = float(np.sum(x_centered * (y - y.mean())) / np.sum(x_centered**2))
    intercept = float(y.mean() - slope * x.mean())
    return slope, intercept


def bootstrap_loglog_slope_ci(
    y_by_x: dict[float, np.ndarray],
    rng: np.random.Generator,
    n_resamples: int = 2000,
    ci: float = 0.95,
) -> tuple[float, float, float, float]:
    """Bootstrap CI for the log-log slope of `median(y)` vs `x`, across several x's.

    A generalization of `bootstrap_ci` for a statistic -- a fitted slope --
    that spans several independent samples of unequal size at once (one per
    distinct `x`, e.g. the 5 `n` values a cell's trials are grouped under).
    `bootstrap_ci`'s single-array API cannot express this: its resampling
    draws indices from one flat array of size `data.size`, so it has no way
    to resample several groups independently *within one replicate* -- which
    is exactly what fitting one slope per bootstrap replicate needs. This
    function uses the same resampling primitive (`rng.integers` with
    replacement, percentile CI on the replicate distribution) applied once
    per `x` group instead.

    The resampling unit is whatever population `y_by_x[x]` holds one row per
    -- the trial, per this project's fixed convention (PREREGISTRATION.md
    sec 3.2): elements within a trial are not independent, so only a
    trial-level resample is honest.

    Parameters
    ----------
    y_by_x : dict[float, np.ndarray]
        Maps each `x` to its raw (trial-level) `y` samples. At least two `x`
        values are required to fit a slope.
    rng : numpy.random.Generator
        Explicit, never global.
    n_resamples : int
        Bootstrap replicate count.
    ci : float
        Central CI width, e.g. 0.95.

    Returns
    -------
    (slope, intercept, ci_low, ci_high)
        `slope`/`intercept` are the point estimate from the un-resampled data
        (ordinary least squares of `log10(median(y))` vs `log10(x)`, one
        point per `x`); `ci_low`/`ci_high` bound the percentile bootstrap CI
        of the slope alone.
    """
    if len(y_by_x) < 2:
        raise ValueError("need at least two x values to fit a slope")

    xs = np.array(sorted(y_by_x), dtype=np.float64)
    point_medians = np.array([float(np.median(y_by_x[x])) for x in sorted(y_by_x)])
    slope, intercept = _loglog_slope(xs, point_medians)

    log_x = np.log10(xs)
    log_x_centered = log_x - log_x.mean()
    denom = float(np.sum(log_x_centered**2))

    boot_medians = np.empty((n_resamples, xs.size), dtype=np.float64)
    for col, x in enumerate(sorted(y_by_x)):
        y = np.asarray(y_by_x[x], dtype=np.float64)
        idx = rng.integers(0, y.size, size=(n_resamples, y.size))
        boot_medians[:, col] = np.median(y[idx], axis=1)

    log_y = np.log10(boot_medians)
    log_y_centered = log_y - log_y.mean(axis=1, keepdims=True)
    slope_reps = (log_x_centered[None, :] * log_y_centered).sum(axis=1) / denom

    alpha = (1.0 - ci) / 2.0
    ci_low, ci_high = np.percentile(slope_reps, [100.0 * alpha, 100.0 * (1.0 - alpha)])
    return slope, intercept, float(ci_low), float(ci_high)


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
