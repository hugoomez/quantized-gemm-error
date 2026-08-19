"""Quantized number format definitions (e.g. float8 variants, int8).

Convention: every quantization function in this module receives a
float64 NumPy array and returns a float64 NumPy array -- the quantized
values are represented in float64, not cast down to the target dtype's
storage type, so downstream error analysis always operates on float64.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

E2M1_MAGNITUDE_GRID = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float64)
E2M1_MAX = 6.0


def quantize_e2m1(
    x: np.ndarray,
    mode: Literal["rtne", "sr"],
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Quantize float64 `x` to the E2M1 (FP4) grid; float64 in, float64 out.

    Grid
    ----
    E2M1 is a 4-bit format: 1 sign bit, 2 exponent bits, 1 mantissa bit,
    with no infinity and no NaN encoding. The magnitude grid is

        0, 0.5, 1, 1.5, 2, 3, 4, 6

    which with sign gives 15 distinct values (+0 and -0 share a value but
    not a bit pattern; the sign of zero is preserved here). The grid is
    *non-uniform*: the step is 0.5 up to 2, then 1 up to 4, then 2 up to
    6. Any implementation that rounds a grid *index* -- e.g. `np.round`
    of an interpolated position -- silently rounds the 2..6 range wrong,
    so the neighbors are located by `searchsorted` on the values instead.

    0.5 is the single subnormal magnitude (exponent field 00, mantissa
    1). It is not reachable by the normal-number path of "extract
    exponent, round mantissa"; here it is just another grid point, so the
    subnormal step (0.5 below 1.0) is applied on its own terms rather
    than inherited from a scaled normal binade.

    Saturation
    ----------
    There is no overflow encoding, so magnitudes above 6 saturate:
    values with `|x| > 6` (including `±inf`) map to `±6`. NaN has no
    E2M1 encoding and is rejected with `ValueError` rather than silently
    mapped to a finite value.

    Rounding
    --------
    `mode="rtne"` -- round to nearest, ties to even, where "even" means
    the **even bit pattern**: the eight magnitudes above are exactly the
    3-bit codes 000..111, so a tie is resolved toward the neighbor whose
    code is even, i.e. whose mantissa bit is 0. Concretely the midpoints
    round as

        0.25 -> 0     (000 vs 001)
        0.75 -> 1     (001 vs 010)
        1.25 -> 1     (010 vs 011)
        1.75 -> 2     (011 vs 100)
        2.5  -> 2     (100 vs 101)
        3.5  -> 4     (101 vs 110)
        5.0  -> 4     (110 vs 111)

    This is the IEEE-754 roundTiesToEven convention applied to the E2M1
    encoding, and is bit-exact with `ml_dtypes.float4_e2m1fn` (verified
    over a dense sweep in `tests/test_formats.py`).

    `mode="sr"` -- stochastic rounding between the two bracketing grid
    neighbors `lo <= |x| <= hi`, taking `hi` with probability
    `(|x| - lo) / (hi - lo)`. This is unbiased: `E[Q(x)] == x` for any
    `|x| <= 6`. Values already on the grid, and saturating values, are
    returned deterministically. Requires an explicit
    `numpy.random.Generator` -- the global NumPy RNG is never used.

    Parameters
    ----------
    x : np.ndarray
        float64 input, any shape. Not modified.
    mode : {"rtne", "sr"}
        Rounding mode, see above.
    rng : np.random.Generator, optional
        Required for `mode="sr"`, ignored for `mode="rtne"`.

    Returns
    -------
    np.ndarray
        float64 array of the same shape, every entry on the E2M1 grid.
    """
    if x.dtype != np.float64:
        raise TypeError(f"expected float64 input, got {x.dtype}")
    if mode not in ("rtne", "sr"):
        raise ValueError(f"unknown mode {mode!r}; expected 'rtne' or 'sr'")
    if np.isnan(x).any():
        raise ValueError("E2M1 has no NaN encoding; input contains NaN")
    if mode == "sr" and rng is None:
        raise ValueError("mode='sr' requires an explicit numpy.random.Generator as rng")

    # Work on magnitudes; the grid is symmetric, so the sign is reapplied
    # at the end (copysign, so the sign of zero survives).
    mag = np.clip(np.abs(x), 0.0, E2M1_MAX)

    # Bracketing neighbors: grid[lo] <= mag <= grid[hi], with lo == hi
    # only at mag == 0. Located by value, never by interpolated index.
    hi = np.searchsorted(E2M1_MAGNITUDE_GRID, mag, side="left")
    lo = np.maximum(hi - 1, 0)
    lo_val = E2M1_MAGNITUDE_GRID[lo]
    hi_val = E2M1_MAGNITUDE_GRID[hi]
    dist_lo = mag - lo_val
    dist_hi = hi_val - mag

    if mode == "rtne":
        # Ties (dist_lo == dist_hi) go to the even 3-bit code. lo and hi
        # are consecutive indices, so exactly one of them is even.
        take_hi = np.where(dist_lo == dist_hi, (lo & 1) == 1, dist_hi < dist_lo)
    else:
        width = hi_val - lo_val
        prob_hi = np.divide(dist_lo, width, out=np.zeros_like(mag), where=width > 0.0)
        take_hi = rng.random(mag.shape) < prob_hi

    return np.copysign(np.where(take_hi, hi_val, lo_val), x)


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
