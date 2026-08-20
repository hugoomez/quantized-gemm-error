"""Input distribution samplers used to generate synthetic GEMM operands.

Convention: every sampler takes an explicit `numpy.random.Generator` --
never NumPy's global RNG state -- and returns float64.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from qgemm.stats import median_absolute_deviation

Normalize = Literal["mad"] | None


def _normalize(x: np.ndarray, normalize: Normalize) -> np.ndarray:
    """Apply the opt-in cross-distribution normalization, or pass `x` through.

    `normalize=None` (the default everywhere it appears) is the original,
    unnormalized behavior -- existing diagnostic scripts are unaffected unless
    they explicitly opt in. `normalize="mad"` rescales the whole tensor so its
    MAD (median absolute deviation, computed once over the full tensor) is 1;
    see SPEC.md, "Cross-distribution normalization" for why MAD rather than
    standard deviation, which is undefined for heavy-tailed t-Student(nu<=2).
    """
    if normalize is None:
        return x
    if normalize == "mad":
        return x / median_absolute_deviation(x)
    raise ValueError(f"unknown normalize {normalize!r}; expected None or 'mad'")


def sample_gaussian(
    shape: tuple[int, ...],
    rng: np.random.Generator,
    scale: float = 1.0,
    normalize: Normalize = None,
) -> np.ndarray:
    x = (rng.standard_normal(shape) * scale).astype(np.float64)
    return _normalize(x, normalize)


def sample_uniform(
    shape: tuple[int, ...],
    rng: np.random.Generator,
    low: float = -1.0,
    high: float = 1.0,
    normalize: Normalize = None,
) -> np.ndarray:
    x = rng.uniform(low, high, size=shape).astype(np.float64)
    return _normalize(x, normalize)
