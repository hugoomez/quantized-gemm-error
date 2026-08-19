"""Block-wise partitioning and per-block quantization utilities.

Convention: float64 in, float64 out.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from qgemm.formats import E2M1_MAX, E8M0_MAX, E8M0_MIN, QuantFormat, quantize_e2m1

# The MX standard block length. Kept as a named constant because "32" appears
# both as the default and as the shape arithmetic below.
MXFP4_BLOCK_SIZE = 32
# Largest magnitude an E2M1 element can hold; the scale is chosen against it.
MXFP4_ELEM_MAX = E2M1_MAX
# The scale is an E8M0 value, so its exponent is limited to [-127, 127].
_MIN_SCALE_EXPONENT = int(np.log2(E8M0_MIN))
_MAX_SCALE_EXPONENT = int(np.log2(E8M0_MAX))


def _check_input(x: np.ndarray, block_size: int) -> None:
    if x.dtype != np.float64:
        raise TypeError(f"expected float64 input, got {x.dtype}")
    if block_size < 1:
        raise ValueError(f"block_size must be a positive integer, got {block_size}")
    if not np.isfinite(x).all():
        raise ValueError("MXFP4 blocking requires finite input; got NaN or infinity")


def _block_amax(x: np.ndarray, block_size: int) -> np.ndarray:
    """Per-block max |x| along the last axis; shape `x.shape[:-1] + (n_blocks,)`.

    A trailing partial block is zero-padded here purely to make the reshape
    rectangular. That is free: `max|x|` is unchanged by appending zeros, so
    the padded block gets exactly the scale the shorter block would have.
    """
    n = x.shape[-1]
    n_blocks = -(-n // block_size)
    pad = n_blocks * block_size - n
    if pad:
        x = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, pad)])
    return np.abs(x).reshape(*x.shape[:-1], n_blocks, block_size).max(axis=-1)


def mxfp4_block_scales(x: np.ndarray, block_size: int = MXFP4_BLOCK_SIZE) -> np.ndarray:
    """Per-block MXFP4 shared scales for float64 `x`, blocked along the last axis.

    The scale of a block is the smallest power of two that keeps the block's
    largest magnitude inside the E2M1 range::

        s_b = 2 ** ceil(log2(amax_b / 6))

    so that `amax_b / s_b <= 6` always. See `quantize_mxfp4` for the full
    recipe, the round-up rationale and the axis/tail policy; this function is
    exposed separately because the scale is the interesting object when
    analysing block-scaling error, and because it makes the rounding rule
    directly testable.

    The exponent is computed from `frexp` rather than `log2` so the rule is
    exact: with `amax = m * 2**e` and `m` in `[0.5, 1)`, the smallest `k` with
    `amax <= 6 * 2**k` is `e - 3` when `m <= 0.75` and `e - 2` otherwise. Going
    through `log2(amax / 6)` would round twice and could pick an exponent one
    too small, quietly reintroducing the saturation the round-up rule exists to
    prevent.

    Returns
    -------
    np.ndarray
        float64 array of shape `x.shape[:-1] + (ceil(x.shape[-1] / block_size),)`;
        every entry is a power of two in `[2**-127, 2**127]`, and blocks that
        are entirely zero get `1.0`.
    """
    _check_input(x, block_size)

    amax = _block_amax(x, block_size)
    is_zero = amax == 0.0
    # frexp(0.0) returns exponent 0, which would give a scale of 2**-3 rather
    # than the documented 1.0 for an all-zero block, so substitute first.
    mantissa, exponent = np.frexp(np.where(is_zero, 1.0, amax))
    exponent = np.where(mantissa <= 0.75, exponent - 3, exponent - 2)
    exponent = np.where(is_zero, 0, exponent)
    exponent = np.clip(exponent, _MIN_SCALE_EXPONENT, _MAX_SCALE_EXPONENT)
    return np.ldexp(1.0, exponent)


def quantize_mxfp4(
    x: np.ndarray,
    block_size: int = MXFP4_BLOCK_SIZE,
    rng: np.random.Generator | None = None,
    round_mode: Literal["rtne", "sr"] = "rtne",
) -> np.ndarray:
    """Quantize float64 `x` to MXFP4 (block-scaled E2M1); float64 in, float64 out.

    MXFP4 splits the tensor into contiguous blocks of `block_size` elements,
    gives each block one shared power-of-two scale, and stores the elements as
    E2M1 (FP4) relative to that scale. The value returned is the *reconstruction*
    `q_i * s_b`, not the pair `(q_i, s_b)` -- error analysis downstream works on
    reconstructions.

    Recipe, per block
    -----------------
    1. `amax_b = max |x_i|` over the block.
    2. Ideal scale `s_ideal = amax_b / 6`, since 6 is the largest E2M1 magnitude.
    3. Representable scale `s_b = 2 ** ceil(log2(s_ideal))` -- rounded **up**.
    4. `q_i = quantize_e2m1(x_i / s_b)`.
    5. Reconstruct `x_hat_i = q_i * s_b`.
    6. If `amax_b == 0`, set `s_b = 1` (there is no `log2(0)` to take).

    Steps 4 and 5 divide and multiply by a power of two, so they are exact in
    float64 except where the quotient underflows; no rounding beyond the E2M1
    grid rounding itself is introduced.

    Why the exponent is rounded up
    ------------------------------
    This is a deliberate design decision, not an implementation detail. Rounding
    the exponent **up** gives `amax_b / s_b <= 6` for every block, so the
    block's largest element -- the one that fixed the scale in the first place --
    always lands inside the E2M1 range. Rounding to nearest, or down, would let
    `amax_b / s_b` reach almost 8, and everything above 6 saturates to exactly
    6, clipping the block's largest value by up to 25% of its magnitude.

    The cost is real and worth stating: a scale one binade larger halves the
    resolution of every *other* element in the block. So the choice trades a
    bounded, uniform precision loss across the block against an unbounded
    clipping error on its largest element. This implementation takes that
    trade; the OCP MX specification and Microsoft's `microxcaling` reference
    take the other one, using `2 ** (floor(log2(amax_b)) - 2)` and accepting the
    saturation. The two rules agree whenever `amax_b`'s own significand is at
    most 1.5, i.e. for roughly three quarters of a log-uniform spread of
    magnitudes; see SPEC.md and `tests/test_blocks.py` for the comparison.

    Blocking axis
    -------------
    Blocks run along the **last axis** of `x`, and only along the last axis;
    leading axes are independent rows. In a GEMM `A @ B` the last axis of `A`
    and of `B.T` is the contraction dimension, which is the axis MXFP4 is
    defined over -- a shared scale is only useful if it factors out of the dot
    product. Blocking along any other axis produces a perfectly well-formed
    array of numbers that measures something else entirely, with no error
    raised, so callers must transpose before calling rather than after.

    Tail policy
    -----------
    If `x.shape[-1]` is not a multiple of `block_size`, the final block is
    **short**: it keeps the elements it has and gets its own scale from its own
    `amax`. It is not merged into the previous block, and no padding survives
    into the output. This is numerically identical to zero-padding the tail out
    to a full block and trimming afterwards -- appending zeros changes neither
    `amax_b` nor any element's quantization -- so the choice is only about not
    allocating and trimming. Hardware that pads a partial block to a full one
    therefore agrees with this function.

    Scale range
    -----------
    The shared scale is an E8M0 value, so its exponent is clamped to
    `[-127, 127]`. Blocks whose ideal scale falls outside that range are the
    degenerate cases at the ends of float64: with a clamped-low scale the block
    maximum saturates to `6 * 2**127`, and with a clamped-high scale the block
    underflows to zero. Both are reported through the returned values rather
    than raised, since a block that extreme is a property of the input data.

    Parameters
    ----------
    x : np.ndarray
        float64 input, any shape, all entries finite. NaN and infinity are
        rejected: neither has an E2M1 encoding, and either one would poison a
        whole block's `amax`. Not modified.
    block_size : int, default 32
        Elements per block along the last axis. 32 is the MX standard.
    rng : np.random.Generator, optional
        Required for `round_mode="sr"`, ignored for `"rtne"`. The global NumPy
        RNG is never used.
    round_mode : {"rtne", "sr"}, default "rtne"
        Element rounding mode, passed through to `quantize_e2m1`. The scale
        itself is always rounded up regardless of this setting -- stochastic
        rounding applies to the elements within a block, not to the block
        exponent.

    Returns
    -------
    np.ndarray
        float64 array of the same shape as `x`; every entry is an E2M1 grid
        value times its block's scale, and `|x_hat_i| <= 6 * s_b`.
    """
    _check_input(x, block_size)

    scales = mxfp4_block_scales(x, block_size)
    # One scale per element: each block's scale repeated block_size times, with
    # the padding of a short final block trimmed back off.
    per_element = np.repeat(scales, block_size, axis=-1)[..., : x.shape[-1]]

    q = quantize_e2m1(x / per_element, mode=round_mode, rng=rng)
    return q * per_element


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
