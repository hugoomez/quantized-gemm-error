"""Statistical summaries over sweep results (aggregation, confidence intervals)."""

from __future__ import annotations

import numpy as np


def mean_ci95(x: np.ndarray) -> tuple[float, float, float]:
    """Return (mean, ci_low, ci_high) via a normal approximation. `x` is float64."""
    x = np.asarray(x, dtype=np.float64)
    mean = float(np.mean(x))
    sem = float(np.std(x, ddof=1) / np.sqrt(x.size)) if x.size > 1 else 0.0
    return mean, mean - 1.96 * sem, mean + 1.96 * sem


def median_absolute_deviation(x: np.ndarray) -> float:
    """`median(|x - median(x)|)`. `x` is float64; well-defined even where variance is not

    (e.g. Cauchy / t-Student with nu<=2), which is why it is used to normalize
    tail-shape comparisons across nu instead of standard deviation.
    """
    x = np.asarray(x, dtype=np.float64)
    return float(np.median(np.abs(x - np.median(x))))
