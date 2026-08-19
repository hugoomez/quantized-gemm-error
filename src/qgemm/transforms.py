"""Pre/post-quantization transforms (scale computation, rotations, etc.).

Convention: float64 in, float64 out.
"""

from __future__ import annotations

import numpy as np


def compute_scale(x: np.ndarray, fmt_max: float) -> np.ndarray:
    """Compute a scale factor mapping float64 `x` into a format with max magnitude `fmt_max`."""
    if x.dtype != np.float64:
        raise TypeError(f"expected float64 input, got {x.dtype}")
    raise NotImplementedError


def apply_scale(x: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """float64 in, float64 out."""
    if x.dtype != np.float64:
        raise TypeError(f"expected float64 input, got {x.dtype}")
    raise NotImplementedError


def invert_scale(x: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """float64 in, float64 out."""
    if x.dtype != np.float64:
        raise TypeError(f"expected float64 input, got {x.dtype}")
    raise NotImplementedError
