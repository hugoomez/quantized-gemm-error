"""Input distribution samplers used to generate synthetic GEMM operands.

Convention: every sampler takes an explicit `numpy.random.Generator` --
never NumPy's global RNG state -- and returns float64.
"""

from __future__ import annotations

import numpy as np


def sample_gaussian(
    shape: tuple[int, ...], rng: np.random.Generator, scale: float = 1.0
) -> np.ndarray:
    return (rng.standard_normal(shape) * scale).astype(np.float64)


def sample_uniform(
    shape: tuple[int, ...], rng: np.random.Generator, low: float = -1.0, high: float = 1.0
) -> np.ndarray:
    return rng.uniform(low, high, size=shape).astype(np.float64)
