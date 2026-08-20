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


def backward_error(A: np.ndarray, B: np.ndarray, Chat: np.ndarray) -> np.ndarray:
    """Elementwise backward error of a GEMM approximation `Chat` of `A @ B`.

    BE_ij = |(A @ B)_ij - Chat_ij| / (|A| @ |B|)_ij

    The denominator is built from the *reconstructed* product of absolute
    values, not from `|A @ B|` -- the two differ whenever `A @ B` involves
    cancellation (see SPEC.md's metric-stability diagnostic). `A`, `B`,
    `Chat` may carry a common leading batch shape, in which case `@` batches
    over it the same way `numpy.matmul` does.
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    Chat = np.asarray(Chat, dtype=np.float64)
    reference = A @ B
    numerator = np.abs(reference - Chat)
    denominator = np.abs(A) @ np.abs(B)
    with np.errstate(divide="ignore", invalid="ignore"):
        return numerator / denominator
