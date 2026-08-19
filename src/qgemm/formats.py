"""Quantized number format definitions (e.g. float8 variants, int8).

Convention: every quantization function in this module receives a
float64 NumPy array and returns a float64 NumPy array -- the quantized
values are represented in float64, not cast down to the target dtype's
storage type, so downstream error analysis always operates on float64.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QuantFormat:
    name: str
    storage_dtype: np.dtype
    finite_max: float


def quantize(x: np.ndarray, fmt: QuantFormat) -> np.ndarray:
    """Quantize float64 `x` to `fmt` and return float64."""
    if x.dtype != np.float64:
        raise TypeError(f"expected float64 input, got {x.dtype}")
    raise NotImplementedError


def dequantize(x: np.ndarray, fmt: QuantFormat) -> np.ndarray:
    """Inverse of `quantize`; float64 in, float64 out."""
    if x.dtype != np.float64:
        raise TypeError(f"expected float64 input, got {x.dtype}")
    raise NotImplementedError
