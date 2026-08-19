"""Quantized GEMM: quantize operands, multiply, and account for accumulation error.

Convention: float64 in, float64 out.
"""

from __future__ import annotations

import numpy as np

from qgemm.formats import QuantFormat


def quantized_matmul(a: np.ndarray, b: np.ndarray, fmt: QuantFormat, **kwargs) -> np.ndarray:
    """Compute A @ B under quantization format `fmt`. `a`, `b`, and the return value are float64."""
    if a.dtype != np.float64 or b.dtype != np.float64:
        raise TypeError(f"expected float64 inputs, got {a.dtype} and {b.dtype}")
    raise NotImplementedError
