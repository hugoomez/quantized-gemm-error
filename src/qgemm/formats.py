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


def _build_float8_magnitude_grid(exponent_bits: int, mantissa_bits: int) -> np.ndarray:
    """Finite magnitudes of a float8 format, in increasing bit-pattern order.

    Entry `i` is the value of code `i` (exponent field in the high bits,
    mantissa in the low bits), so the array index *is* the 8-bit code with
    the sign stripped. That identity is what makes ties-to-even a parity
    test on the index. Only the finite codes are built; the caller appends
    the one code past the top (NaN for E4M3, infinity for E5M2).
    """
    bias = 2 ** (exponent_bits - 1) - 1
    n_mantissa = 2**mantissa_bits
    steps = np.arange(n_mantissa, dtype=np.float64) / n_mantissa
    # Exponent field 0 is the subnormal binade: value = mantissa * 2**(1-bias-m).
    subnormal = steps * 2.0 ** (1 - bias)
    normal = [(1.0 + steps) * 2.0 ** (e - bias) for e in range(1, 2**exponent_bits)]
    return np.concatenate([subnormal, *normal])


# E4M3: 1 sign, 4 exponent, 3 mantissa bits, bias 7. Codes 0..126 are the
# finite magnitudes; code 127 (exponent field 1111, mantissa 111) is the NaN
# pattern, so the finite grid stops one short of a full binade.
E4M3_MAGNITUDE_GRID = _build_float8_magnitude_grid(4, 3)[:-1]
E4M3_MAX = 448.0
# The magnitude the NaN code would have carried, used as the overflow marker.
_E4M3_OVERFLOW = 480.0

# E5M2: 1 sign, 5 exponent, 2 mantissa bits, bias 15. Codes 0..123 are the
# finite magnitudes; exponent field 11111 is reserved for infinity (mantissa
# 00) and NaN (mantissa != 00), so the whole last binade is dropped.
E5M2_MAGNITUDE_GRID = _build_float8_magnitude_grid(5, 2)[: -(2**2)]
E5M2_MAX = 57344.0
# The magnitude the infinity code would have carried, used as the overflow marker.
_E5M2_OVERFLOW = 65536.0

# E8M0: 8 exponent bits, no sign and no mantissa. Codes 0..254 are the powers
# of two 2**-127 .. 2**127; code 255 (0xFF) is the single NaN pattern. There is
# no zero, no infinity, and no negative half.
E8M0_GRID = 2.0 ** np.arange(-127.0, 128.0)
E8M0_MIN = 2.0**-127
E8M0_MAX = 2.0**127
_E8M0_OVERFLOW = 2.0**128


def _round_signed_to_nearest_even_code(x: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Round |x| onto `grid` with ties to the even code, then reapply the sign.

    `grid` is a magnitude grid indexed by code, with one extra entry on the
    end standing in for the overflow code (NaN for E4M3, infinity for E5M2);
    a result equal to that entry is the caller's signal that `x` overflowed.

    Neighbours are located by `searchsorted` on the *values*: like the E2M1
    grid these are non-uniform (the step doubles at every binade), so rounding
    an interpolated index would misround every binade boundary.
    """
    # NaN cannot go through searchsorted, so park it on 0.0 and restore it after.
    is_nan = np.isnan(x)
    mag = np.clip(np.where(is_nan, 0.0, np.abs(x)), 0.0, grid[-1])

    hi = np.searchsorted(grid, mag, side="left")
    lo = np.maximum(hi - 1, 0)
    lo_val = grid[lo]
    hi_val = grid[hi]
    # Both differences are exact in float64 (Sterbenz: adjacent grid points are
    # within a factor of two), so the tie test below is not itself rounded.
    dist_lo = mag - lo_val
    dist_hi = hi_val - mag

    # lo and hi are consecutive codes, so exactly one of them is even.
    take_hi = np.where(dist_lo == dist_hi, (lo & 1) == 1, dist_hi < dist_lo)
    out = np.where(take_hi, hi_val, lo_val)
    return np.where(is_nan, np.nan, out)


def quantize_e4m3(x: np.ndarray) -> np.ndarray:
    """Quantize float64 `x` to the E4M3 grid; float64 in, float64 out.

    Grid
    ----
    E4M3 is 1 sign bit, 4 exponent bits, 3 mantissa bits, bias 7 -- the
    deep-learning FP8 variant, matching `ml_dtypes.float8_e4m3fn` (**not**
    `ml_dtypes.float8_e4m3`, which has infinities and a maximum of 240).

    The 127 finite magnitudes are exactly codes `0x00..0x7E`:

    * `0`;
    * seven subnormals `m * 2**-9` for `m = 1..7`, so the smallest positive
      magnitude is `2**-9` and the subnormal step is `2**-9` throughout;
    * normals `(1 + m/8) * 2**(e-7)` for exponent field `e = 1..15`, giving a
      smallest normal of `2**-6` and a maximum of `448 = 1.75 * 2**8`.

    With sign that is 253 distinct values (+0 and -0 share a value but not a
    bit pattern; the sign of zero is preserved). The step doubles at every
    binade, so neighbours are found by `searchsorted` on the values.

    Saturation
    ----------
    **E4M3 does not saturate.** There is no infinity encoding, and the single
    code above the maximum -- exponent field `1111` with mantissa `111`, which
    would have been `480` -- is reserved for NaN. So overflow produces NaN,
    not `±448`:

        448   -> 448
        463.9 -> 448
        464   -> 448    (tie between 448 and the 480 NaN code; 448 is even)
        465   -> NaN
        ±inf  -> ±NaN

    The sign bit is carried onto the NaN. This differs from `quantize_e2m1`,
    which saturates -- E2M1 has no NaN encoding to overflow into, whereas
    E4M3 does, so nothing is silently lost here. NaN input passes through.

    Rounding
    --------
    Round to nearest, ties to even, where "even" is the even *bit pattern*:
    the magnitude grid is indexed by code, so a tie goes to the neighbour
    whose mantissa bit 0 is clear. Bit-exact with `ml_dtypes.float8_e4m3fn`
    (see `tests/test_formats.py`, and SPEC.md for the one documented
    deviation, which concerns ml_dtypes' float32 intermediate).

    Parameters
    ----------
    x : np.ndarray
        float64 input, any shape. Not modified.

    Returns
    -------
    np.ndarray
        float64 array of the same shape; every entry is on the E4M3 grid or
        is NaN.
    """
    if x.dtype != np.float64:
        raise TypeError(f"expected float64 input, got {x.dtype}")

    grid = np.append(E4M3_MAGNITUDE_GRID, _E4M3_OVERFLOW)
    out = _round_signed_to_nearest_even_code(x, grid)
    # Rounding up into the reserved code is the overflow condition.
    out = np.where(out == _E4M3_OVERFLOW, np.nan, out)
    return np.copysign(out, x)


def quantize_e5m2(x: np.ndarray) -> np.ndarray:
    """Quantize float64 `x` to the E5M2 grid; float64 in, float64 out.

    Grid
    ----
    E5M2 is 1 sign bit, 5 exponent bits, 2 mantissa bits, bias 15, and unlike
    E4M3 it is IEEE-shaped: exponent field `11111` is reserved for infinity
    (mantissa `00`) and NaN (mantissa nonzero). Matches
    `ml_dtypes.float8_e5m2`.

    The 124 finite magnitudes are exactly codes `0x00..0x7B`:

    * `0`;
    * three subnormals `m * 2**-16` for `m = 1..3`;
    * normals `(1 + m/4) * 2**(e-15)` for exponent field `e = 1..30`, giving a
      smallest normal of `2**-14` and a maximum of `57344 = 1.75 * 2**15`.

    E5M2 trades mantissa bits for exponent range: three binades wider than
    E4M3 at each end, at half the relative precision.

    Saturation
    ----------
    **E5M2 does not saturate either, but it overflows to infinity** rather
    than NaN, exactly as IEEE-754 round-to-nearest does in any binary format:

        57344 -> 57344
        61439 -> 57344
        61440 -> +inf   (tie between 57344 and the 65536 infinity code, which
                         has mantissa 00 and is therefore the even one)
        ±inf  -> ±inf

    NaN input passes through, and the sign is preserved throughout.

    Rounding
    --------
    Round to nearest, ties to even bit pattern, as for `quantize_e4m3`.
    Bit-exact with `ml_dtypes.float8_e5m2`; see SPEC.md for the documented
    float32-intermediate deviation.

    Parameters
    ----------
    x : np.ndarray
        float64 input, any shape. Not modified.

    Returns
    -------
    np.ndarray
        float64 array of the same shape; every entry is on the E5M2 grid, or
        is `±inf` or NaN.
    """
    if x.dtype != np.float64:
        raise TypeError(f"expected float64 input, got {x.dtype}")

    grid = np.append(E5M2_MAGNITUDE_GRID, _E5M2_OVERFLOW)
    out = _round_signed_to_nearest_even_code(x, grid)
    out = np.where(out == _E5M2_OVERFLOW, np.inf, out)
    return np.copysign(out, x)


def quantize_e8m0(x: np.ndarray) -> np.ndarray:
    """Quantize float64 `x` to the E8M0 grid; float64 in, float64 out.

    Grid
    ----
    E8M0 is 8 bits of exponent and nothing else: no sign bit, no mantissa.
    Matches `ml_dtypes.float8_e8m0fnu`. It is the scale format of the
    microscaling (MX) layouts, not a format for data, which is why it is
    unsigned and why every representable value is a power of two:

        code 0..254 -> 2**-127 .. 2**127   (bias 127)
        code 255    -> NaN

    Three consequences follow, and they are not the usual float8 ones:

    * **There is no zero.** The smallest code is `2**-127`, so `0.0` and
      `-0.0` both map to NaN.
    * **There is no sign.** Every negative input maps to NaN.
    * **There is no infinity.** `0xFF` is the only reserved code.

    Saturation
    ----------
    Asymmetric, and deliberately so:

        2**-200 -> 2**-127   underflow *clamps up* to the smallest code
        2**-127 -> 2**-127
        2**127  -> 2**127
        1.5*2**127 -> NaN    overflow has nowhere to go but the NaN code
        inf     -> NaN

    Rounding
    --------
    Nearest power of two on the **linear** scale -- the split between `2**e`
    and `2**(e+1)` is at their arithmetic midpoint `1.5 * 2**e`, not at the
    geometric midpoint `sqrt(2) * 2**e`. Ties go **up**, at every exponent:

        1.4999... * 2**e -> 2**e
        1.5      * 2**e  -> 2**(e+1)

    Note this is *not* ties-to-even. With no mantissa bit, "even" could only
    mean an even exponent code, which would make the tie direction alternate
    with the parity of `e`; `ml_dtypes` rounds ties up unconditionally and
    this follows it. Bit-exact with `ml_dtypes.float8_e8m0fnu`; see SPEC.md
    for the documented float32-intermediate deviation.

    Parameters
    ----------
    x : np.ndarray
        float64 input, any shape. Not modified.

    Returns
    -------
    np.ndarray
        float64 array of the same shape; every entry is a power of two in
        `[2**-127, 2**127]`, or NaN. NaN is never returned with a sign bit
        set, since E8M0 has no sign bit to set.
    """
    if x.dtype != np.float64:
        raise TypeError(f"expected float64 input, got {x.dtype}")

    # No zero and no sign: everything at or below zero, and NaN, is unrepresentable.
    invalid = np.isnan(x) | (x <= 0.0)
    mag = np.clip(np.where(invalid, E8M0_MIN, x), E8M0_MIN, _E8M0_OVERFLOW)

    grid = np.append(E8M0_GRID, _E8M0_OVERFLOW)
    hi = np.searchsorted(grid, mag, side="left")
    lo = np.maximum(hi - 1, 0)
    lo_val = grid[lo]
    hi_val = grid[hi]
    # Ties up, so the upper neighbour wins on `<=` rather than `<`.
    take_hi = (hi_val - mag) <= (mag - lo_val)
    out = np.where(take_hi, hi_val, lo_val)

    return np.where(invalid | (out == _E8M0_OVERFLOW), np.nan, out)


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
