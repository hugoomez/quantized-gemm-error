"""Block-wise partitioning and per-block quantization utilities.

Convention: float64 in, float64 out.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from qgemm.formats import (
    E2M1_MAX,
    E4M3_MAX,
    E8M0_MAX,
    E8M0_MIN,
    QuantFormat,
    quantize_e2m1,
    quantize_e4m3,
)

# The MX standard block length. Kept as a named constant because "32" appears
# both as the default and as the shape arithmetic below.
MXFP4_BLOCK_SIZE = 32
# Largest magnitude an E2M1 element can hold; the scale is chosen against it.
MXFP4_ELEM_MAX = E2M1_MAX
# The scale is an E8M0 value, so its exponent is limited to [-127, 127].
_MIN_SCALE_EXPONENT = int(np.log2(E8M0_MIN))
_MAX_SCALE_EXPONENT = int(np.log2(E8M0_MAX))

# The NVFP4 block length. Half of MXFP4's, which is one of the two ways the
# format buys accuracy; the other is the E4M3 scale.
NVFP4_BLOCK_SIZE = 16
# NVFP4 elements are E2M1, exactly as in MXFP4 -- only the scaling differs.
NVFP4_ELEM_MAX = E2M1_MAX
# The per-block scale is an E4M3 value, so it tops out at 448. The global
# scale exists to keep every block scale inside this bound.
NVFP4_SCALE_MAX = E4M3_MAX


def _check_input(x: np.ndarray, block_size: int) -> None:
    if x.dtype != np.float64:
        raise TypeError(f"expected float64 input, got {x.dtype}")
    if block_size < 1:
        raise ValueError(f"block_size must be a positive integer, got {block_size}")
    if not np.isfinite(x).all():
        raise ValueError("block scaling requires finite input; got NaN or infinity")


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


def nvfp4_global_scale(x: np.ndarray) -> float:
    """The single per-tensor NVFP4 global scale for float64 `x`.

    The global scale is what keeps the per-block scales inside E4M3::

        s_global = max_b(amax_b) / (6 * 448)

    where 6 is the largest E2M1 magnitude and 448 the largest E4M3 one. Divide
    the tensor's own maximum by this and the ideal block scale of the largest
    block comes out at exactly 448 -- the top of E4M3 -- so every other block,
    being smaller, lands strictly inside the E4M3 range. Without it, a block
    scale of `amax_b / 6` would be a raw magnitude from the input data, and
    anything above 464 overflows E4M3 to **NaN** (E4M3 does not saturate). See
    `quantize_nvfp4` for the full recipe.

    This is a single number for the whole array, leading axes included --
    unlike the block scales, which are per-row. A row's reconstruction
    therefore depends on the rest of the tensor.

    Returns
    -------
    float
        `amax / 2688`, or `1.0` for an all-zero tensor (there is no scale to
        take) and for a tensor so small that `amax / 2688` underflows float64.
    """
    _check_input(x, 1)

    amax = float(np.abs(x).max(initial=0.0))
    scale = amax / (NVFP4_ELEM_MAX * NVFP4_SCALE_MAX)
    # 0.0 covers both the all-zero tensor and an amax below ~1.3e-320, where
    # the quotient underflows; either way there is nothing to scale by.
    return 1.0 if scale == 0.0 else scale


def nvfp4_block_scales(x: np.ndarray, block_size: int = NVFP4_BLOCK_SIZE) -> np.ndarray:
    """Per-block NVFP4 E4M3 scales for float64 `x`, blocked along the last axis.

    These are the *second* level of the two-level scaling and are relative to
    the global scale: the effective scale of an element is
    `nvfp4_block_scales(x)[b] * nvfp4_global_scale(x)`. Each is the block's
    ideal scale expressed in the global scale's units and rounded to E4M3::

        s_b = quantize_e4m3((amax_b / s_global) / 6)

    The argument is `448 * amax_b / amax` by construction, so it never exceeds
    448 and the result is always a finite E4M3 value. The clamp below is not
    that guarantee restated -- it covers the one corner where the guarantee
    fails numerically: for `amax` near the bottom of float64, `s_global` is
    itself subnormal and its relative rounding error is large enough to push
    the argument past 464, where E4M3 would overflow to NaN.

    Unlike MXFP4, whose E8M0 scale has no zero, an E4M3 scale can be exactly
    zero, and is for two kinds of block: one that is all zeros, and one whose
    `amax` is more than about `2**19` times below the tensor's, so its ideal
    scale falls below the smallest E4M3 subnormal. Both flush to zero in the
    reconstruction, which is what a scale of zero means.

    Returns
    -------
    np.ndarray
        float64 array of shape `x.shape[:-1] + (ceil(x.shape[-1] / block_size),)`;
        every entry is an E4M3 value in `[0, 448]`.
    """
    _check_input(x, block_size)

    ideal = _block_amax(x, block_size) / nvfp4_global_scale(x) / NVFP4_ELEM_MAX
    return quantize_e4m3(np.minimum(ideal, NVFP4_SCALE_MAX))


def quantize_nvfp4(
    x: np.ndarray,
    block_size: int = NVFP4_BLOCK_SIZE,
    rng: np.random.Generator | None = None,
    round_mode: Literal["rtne", "sr"] = "rtne",
) -> np.ndarray:
    """Quantize float64 `x` to NVFP4 (two-level block-scaled E2M1); float64 in/out.

    NVFP4 stores the same E2M1 elements as MXFP4 but scales them **twice**: one
    E4M3 scale per block of `block_size` elements, and one global scale for the
    whole tensor. Like `quantize_mxfp4`, this returns the *reconstruction*
    `q_i * s_b * s_global`; the two levels of scale are available separately
    from `nvfp4_block_scales` and `nvfp4_global_scale`.

    Recipe
    ------
    1. `s_global = max_b(amax_b) / (6 * 448)`, one number for the whole tensor.
    2. Per block, `s_b = quantize_e4m3((amax_b / s_global) / 6)`.
    3. `q_i = quantize_e2m1(x_i / (s_b * s_global))`.
    4. Reconstruct `x_hat_i = q_i * s_b * s_global`.
    5. Degenerate cases: an all-zero tensor takes `s_global = 1`, and a block
       whose `s_b` is zero -- all-zero, or vanishing next to the tensor
       maximum -- reconstructs to zero.

    Why the global scale exists
    ---------------------------
    **To keep the block scales inside E4M3's range**, and for no other reason.
    A block scale is a magnitude drawn from the input data, `amax_b / 6`, and
    E4M3 tops out at 448: quantize a tensor whose values run to 1e6 and every
    block scale overflows. E4M3 *does not saturate* -- the code above its
    maximum is NaN -- so the failure is not even a graceful one.

    Dividing by `s_global` first normalises the tensor so that its largest
    block wants exactly 448, the top of E4M3, and every other block wants
    less. `s_global` is a full float64 here (float32 in hardware), so it costs
    nothing in precision and buys the whole E4M3 dynamic range for the block
    scales.

    Dropping this step is the natural bug, and it has a diagnostic signature:
    block scales pinned at the top of E4M3 (or NaN) means the blocks are barely
    being scaled at all, and NVFP4 measures *worse* than MXFP4 -- the opposite
    of what the format is for. `tests/test_blocks.py` asserts the accuracy
    ordering directly for that reason.

    Contrast with MXFP4
    -------------------
    The scale format is the whole difference, and it cuts both ways:

    * **Finer.** An E8M0 scale is a power of two, so MXFP4 rounds the block
      scale up by up to a full binade and gives away up to one bit of element
      precision. An E4M3 scale has three mantissa bits, so it lands within
      6.25% of the ideal scale. With `amax_b = 4.5`: MXFP4 must take `s_b = 1`
      and 4.5 falls onto the grid point 4; NVFP4 takes an effective scale of
      exactly 0.75 and reconstructs 4.5 exactly.
    * **Clipping is possible.** `quantize_e4m3` rounds to nearest, so `s_b`
      can round *down* below the ideal, leaving `amax_b / (s_b * s_global)`
      slightly above 6 -- and E2M1 saturates, clipping the block maximum by up
      to about 6%. MXFP4's round-up rule forbids this by construction. NVFP4
      takes the opposite side of that trade, and this implementation follows
      the format rather than importing MXFP4's rule.
    * **Rows are not independent.** `s_global` is per tensor, so unlike MXFP4
      a row's reconstruction depends on the other rows.

    Blocking axis, tail policy, input contract
    ------------------------------------------
    Identical to `quantize_mxfp4`: blocks run along the **last axis** only
    (the contraction dimension of a GEMM), a trailing partial block stays
    short and takes its own scale -- numerically the same as zero-padding it,
    since appending zeros changes no block's `amax` -- and NaN and infinity
    are rejected, since either would poison a whole block's `amax`. Note that
    with NVFP4 a non-finite value would poison the *global* scale too, and
    with it the entire tensor.

    Parameters
    ----------
    x : np.ndarray
        float64 input, any shape, all entries finite. Not modified.
    block_size : int, default 16
        Elements per block along the last axis. 16 is the NVFP4 standard --
        half of MXFP4's 32, which is the format's other accuracy lever.
    rng : np.random.Generator, optional
        Required for `round_mode="sr"`, ignored for `"rtne"`. The global NumPy
        RNG is never used.
    round_mode : {"rtne", "sr"}, default "rtne"
        Element rounding mode, passed through to `quantize_e2m1`. Both levels
        of scale are always RTNE regardless -- stochastic rounding applies to
        the elements within a block, not to the scales.

    Returns
    -------
    np.ndarray
        float64 array of the same shape as `x`; every entry is an E2M1 grid
        value times its block's effective scale `s_b * s_global`.
    """
    _check_input(x, block_size)

    scales = nvfp4_block_scales(x, block_size) * nvfp4_global_scale(x)
    # One scale per element: each block's scale repeated block_size times, with
    # the padding of a short final block trimmed back off.
    per_element = np.repeat(scales, block_size, axis=-1)[..., : x.shape[-1]]

    # A block whose scale is zero reconstructs to zero whatever the elements
    # round to, so divide by 1 there rather than turning the block into 0/0.
    divisor = np.where(per_element == 0.0, 1.0, per_element)
    q = quantize_e2m1(x / divisor, mode=round_mode, rng=rng)
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
