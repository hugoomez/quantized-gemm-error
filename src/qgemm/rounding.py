"""Rounding modes used when mapping float64 values onto a quantized grid.

Convention: float64 in, float64 out. Stochastic rounding takes an
explicit `numpy.random.Generator` -- never NumPy's global RNG state.
"""

from __future__ import annotations

import numpy as np


def round_nearest_even(x: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Round float64 `x` to the nearest value in `grid` (ties to even). Returns float64."""
    if x.dtype != np.float64:
        raise TypeError(f"expected float64 input, got {x.dtype}")
    raise NotImplementedError


def round_stochastic(
    x: np.ndarray, grid: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Stochastically round float64 `x` onto `grid` using `rng`. Returns float64."""
    if x.dtype != np.float64:
        raise TypeError(f"expected float64 input, got {x.dtype}")
    raise NotImplementedError
