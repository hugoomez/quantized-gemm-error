"""Block-wise partitioning and per-block quantization utilities.

Convention: float64 in, float64 out.
"""

from __future__ import annotations

import numpy as np

from qgemm.formats import QuantFormat


def block_reshape(x: np.ndarray, block_size: int, axis: int = -1) -> np.ndarray:
    """Reshape float64 `x` so that `axis` is split into blocks of `block_size`."""
    if x.dtype != np.float64:
        raise TypeError(f"expected float64 input, got {x.dtype}")
    raise NotImplementedError


def quantize_blockwise(
    x: np.ndarray, fmt: QuantFormat, block_size: int, axis: int = -1
) -> np.ndarray:
    """Quantize float64 `x` block-wise along `axis`; returns float64."""
    if x.dtype != np.float64:
        raise TypeError(f"expected float64 input, got {x.dtype}")
    raise NotImplementedError
