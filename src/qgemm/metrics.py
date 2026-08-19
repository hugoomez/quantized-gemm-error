"""Error metrics comparing a quantized-GEMM result against a float64 reference.

Convention: float64 in, float64 (or float) out.
"""

from __future__ import annotations

import numpy as np


def abs_error(approx: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.abs(approx - reference)


def max_abs_error(approx: np.ndarray, reference: np.ndarray) -> float:
    return float(np.max(abs_error(approx, reference)))


def mean_abs_error(approx: np.ndarray, reference: np.ndarray) -> float:
    return float(np.mean(abs_error(approx, reference)))


def relative_error(approx: np.ndarray, reference: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return abs_error(approx, reference) / (np.abs(reference) + eps)


def rms_error(approx: np.ndarray, reference: np.ndarray) -> float:
    return float(np.sqrt(np.mean((approx - reference) ** 2)))
