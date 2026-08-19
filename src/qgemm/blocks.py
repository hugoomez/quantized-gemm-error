"""Block-wise partitioning and per-block quantization utilities.

Convention: float64 in, float64 out.

Everything here is one parametrized quantizer, `quantize_blocked`, plus thin
presets for the two real hardware formats (`quantize_mxfp4`, `quantize_nvfp4`).
There is deliberately no second implementation behind either preset: the
formats differ only in their parameters, and the point of the study is to vary
those parameters independently.
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

ScaleFormat = Literal["e8m0", "e4m3"]
ElementFormat = Literal["e2m1"]
RoundMode = Literal["rtne", "sr"]

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

# Largest representable value of each supported scale format. Used to size the
# global scale, and (for E4M3) as the clamp that keeps a block scale off the
# NaN code that sits directly above the maximum.
_SCALE_FORMAT_MAX = {"e8m0": E8M0_MAX, "e4m3": E4M3_MAX}
# Largest representable element magnitude, i.e. what a block scale is chosen
# against. Only E2M1 elements are implemented; adding a format means extending
# this table *and* the element quantizer call in `quantize_blocked`.
_ELEMENT_FORMAT_MAX = {"e2m1": E2M1_MAX}


def _validate_formats(scale_format: str, element_format: str) -> None:
    if scale_format not in _SCALE_FORMAT_MAX:
        raise ValueError(
            f"unknown scale_format {scale_format!r}; expected one of {sorted(_SCALE_FORMAT_MAX)}"
        )
    if element_format not in _ELEMENT_FORMAT_MAX:
        raise ValueError(
            f"unsupported element_format {element_format!r}; only "
            f"{sorted(_ELEMENT_FORMAT_MAX)} is implemented"
        )


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


def _quantize_scale(amax: np.ndarray, scale_format: str, elem_max: float) -> np.ndarray:
    """Round each block's ideal scale `amax / elem_max` onto the scale grid.

    The two formats round in opposite directions, and that is a property of
    the formats rather than a free choice here:

    * **E8M0** rounds the exponent **up**, so `amax / s_b <= elem_max` always
      and the element that set the scale never saturates. The exponent is
      derived from `frexp` rather than `log2(amax / elem_max)`: the latter
      rounds twice and can land one exponent low, silently reintroducing the
      saturation the round-up rule exists to prevent. With `amax = m * 2**e`
      and `elem_max = mm * 2**em` (both mantissas in `[0.5, 1)` from `frexp`),
      the smallest `k` with `amax <= elem_max * 2**k` is `e - em` when
      `m <= mm` and one more otherwise -- for E2M1's `elem_max = 6`, exactly
      the documented `e - 3` / `e - 2` rule. Out-of-range exponents are
      clamped to E8M0's `[-127, 127]`, and zero blocks take `s_b = 1`.
    * **E4M3** rounds to **nearest** (`quantize_e4m3`), so `s_b` can land just
      below the ideal scale and let the block maximum saturate -- bounded by
      E4M3's half-ulp, about 6%. The `minimum` is not that bound restated: it
      is the clamp that keeps the argument off E4M3's NaN code, which sits
      directly above 448 because E4M3 does not saturate. It fires whenever the
      ideal scale genuinely exceeds 448 -- the normal state of affairs when
      `use_global_scale` is False -- and also covers the corner where a
      subnormal global scale pushes the argument past 448 numerically.
    """
    if scale_format == "e8m0":
        is_zero = amax == 0.0
        # frexp(0.0) returns exponent 0, which would give a scale of 2**-3
        # rather than the documented 1.0 for an all-zero block, so substitute
        # first and put the 1.0 back afterwards.
        mantissa, exponent = np.frexp(np.where(is_zero, 1.0, amax))
        elem_mantissa, elem_exponent = np.frexp(elem_max)
        exponent = exponent - elem_exponent + np.where(mantissa <= elem_mantissa, 0, 1)
        exponent = np.where(is_zero, 0, exponent)
        return np.ldexp(1.0, np.clip(exponent, _MIN_SCALE_EXPONENT, _MAX_SCALE_EXPONENT))

    return quantize_e4m3(np.minimum(amax / elem_max, _SCALE_FORMAT_MAX[scale_format]))


def _scales(
    x: np.ndarray,
    block_size: int,
    scale_format: str,
    use_global_scale: bool,
    element_format: str,
) -> tuple[np.ndarray, float]:
    """Per-block scales, and the global scale they are expressed relative to."""
    s_global = global_scale(x, scale_format, element_format) if use_global_scale else 1.0
    ideal = _block_amax(x, block_size) / s_global
    scales = _quantize_scale(ideal, scale_format, _ELEMENT_FORMAT_MAX[element_format])
    return scales, s_global


def global_scale(
    x: np.ndarray,
    scale_format: ScaleFormat,
    element_format: ElementFormat = "e2m1",
) -> float:
    """The single per-tensor scale that normalises `x` into the scale format's range.

    ::

        s_global = max |x| / (elem_max * scale_max)

    Divide the tensor by this and its largest block wants a scale of exactly
    `scale_max` -- the top of the scale format -- so every other block, being
    smaller, lands strictly inside. This is what makes an E4M3 block scale
    workable at all: a raw block scale is `amax_b / 6`, a magnitude taken
    straight from the input data, and E4M3 tops out at 448 with **NaN** (not
    saturation) above it.

    This is one number for the whole array, leading axes included -- unlike
    the block scales, which are per row. A row's reconstruction therefore
    depends on the rest of the tensor whenever a global scale is in use.

    Returns
    -------
    float
        `max |x| / (elem_max * scale_max)`, or `1.0` for an all-zero tensor
        and for a tensor small enough that the quotient underflows float64
        (below about `1.3e-320` for E4M3): either way there is nothing to
        scale by.
    """
    _validate_formats(scale_format, element_format)
    _check_input(x, 1)

    amax = float(np.abs(x).max(initial=0.0))
    scale = amax / (_ELEMENT_FORMAT_MAX[element_format] * _SCALE_FORMAT_MAX[scale_format])
    # 0.0 covers both the all-zero tensor and an amax so small the quotient
    # underflows; either way there is nothing to scale by.
    return 1.0 if scale == 0.0 else scale


def block_scales(
    x: np.ndarray,
    block_size: int,
    scale_format: ScaleFormat,
    use_global_scale: bool,
    element_format: ElementFormat = "e2m1",
) -> np.ndarray:
    """Per-block scales for float64 `x`, blocked along the last axis.

    The scale is the interesting object when analysing block-scaling error, so
    it is exposed separately from `quantize_blocked` and takes exactly the same
    parameters. When `use_global_scale` is True these are the *second* level of
    a two-level scaling and are expressed in the global scale's units: the
    effective scale of an element is
    `block_scales(...)[b] * global_scale(x, scale_format)`.

    See `_quantize_scale` for each scale format's rounding rule, and
    `quantize_blocked` for the axis, tail and input policies and for which
    parameter combinations are real formats rather than controls.

    Returns
    -------
    np.ndarray
        float64 array of shape `x.shape[:-1] + (ceil(x.shape[-1] / block_size),)`.
        Under `"e8m0"` every entry is a power of two in `[2**-127, 2**127]` and
        never zero (E8M0 has no zero); an all-zero block gets `1.0`. Under
        `"e4m3"` every entry is an E4M3 value in `[0, 448]`; zero *is*
        representable and occurs both for an all-zero block and for a block
        whose `amax` is more than about `2**19` below the tensor's, so that its
        ideal scale falls under the smallest E4M3 subnormal. Both flush to zero
        in the reconstruction, which is what a scale of zero means.
    """
    _validate_formats(scale_format, element_format)
    _check_input(x, block_size)

    return _scales(x, block_size, scale_format, use_global_scale, element_format)[0]


def quantize_blocked(
    x: np.ndarray,
    block_size: int,
    scale_format: ScaleFormat,
    use_global_scale: bool,
    element_format: ElementFormat = "e2m1",
    round_mode: RoundMode = "rtne",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Quantize float64 `x` block-wise with a parametrized scaling scheme.

    This is the single implementation behind `quantize_mxfp4` and
    `quantize_nvfp4`; both are thin presets of it. The tensor is split into
    contiguous blocks of `block_size` elements along the last axis, each block
    gets one shared scale in `scale_format`, optionally on top of one global
    scale for the whole tensor, and the elements are stored in `element_format`
    relative to that scale. The value returned is the *reconstruction*
    `q_i * s_b * s_global`, not the `(q_i, s_b)` pair -- the error analysis
    downstream works on reconstructions. The scales themselves come from
    `block_scales` and `global_scale`.

    Recipe, per block
    -----------------
    1. `s_global = max |x| / (elem_max * scale_max)` if `use_global_scale`,
       else `1`.
    2. `amax_b = max |x_i|` over the block.
    3. `s_b = round_to(scale_format, (amax_b / s_global) / elem_max)`, rounding
       the exponent **up** for `"e8m0"` and to nearest for `"e4m3"`.
    4. `q_i = quantize_e2m1(x_i / (s_b * s_global), mode=round_mode)`.
    5. Reconstruct `x_hat_i = q_i * s_b * s_global`.

    With `use_global_scale=False` steps 4 and 5 under an E8M0 scale divide and
    multiply by a power of two, so they are exact in float64 except where the
    quotient underflows; no rounding beyond the element grid rounding itself
    is introduced. A global scale is not a power of two and gives that up.

    Which combinations are real formats
    -----------------------------------
    **Only two of these parameter settings are real hardware formats.**

    ============ ============ ================ ================================
    block_size   scale_format use_global_scale meaning
    ============ ============ ================ ================================
    32           ``"e8m0"``   False            **MXFP4** (OCP MX) -- real
    16           ``"e4m3"``   True             **NVFP4** (NVIDIA) -- real
    8, 16, 64    ``"e8m0"``   False            experimental control
    8, 32, 64    ``"e4m3"``   True             experimental control
    ============ ============ ================ ================================

    The controls exist for one reason: MXFP4 and NVFP4 differ in *both* block
    size and scale format at once, so a measured difference between the two
    cannot be attributed to either. Holding one axis fixed while varying the
    other -- block size 16 with an E8M0 scale, block size 32 with an E4M3
    scale, plus the 8 and 64 endpoints on both -- separates the two effects.

    **The control configurations are not proposals.** They correspond to no
    hardware, no specification and no vendor format, they have not been
    designed for anything, and nothing here should be read as introducing a
    new format. They are measurement instruments for a controlled comparison
    between two formats that already exist.

    `use_global_scale` is a free parameter rather than a synonym for the scale
    format, so all four scale-format/global-scale combinations run. The study
    uses only the two pairings in the table, and they are the two that make
    sense: E8M0 spans 255 binades and needs no help reaching any block's
    magnitude, while E4M3 spans about 19 and overflows without one. Turning
    the global scale on under E8M0 is supported but measures a fourth thing
    again -- it multiplies the power-of-two block scale by a non-power-of-two,
    giving up the exactness that is the point of an E8M0 scale. Turning it off
    under E4M3 is the NVFP4 bug with the diagnostic signature described in
    SPEC.md: block scales pinned at 448 and accuracy worse than MXFP4's.

    Blocking axis
    -------------
    Blocks run along the **last axis** of `x`, and only along the last axis;
    leading axes are independent rows. In a GEMM `A @ B` the last axis of `A`
    and of `B.T` is the contraction dimension, which is the axis a shared
    scale factors out of. Blocking along any other axis produces a perfectly
    well-formed array of numbers that measures something else entirely, with
    no error raised, so callers must transpose before calling rather than
    after.

    Tail policy
    -----------
    If `x.shape[-1]` is not a multiple of `block_size`, the final block is
    **short**: it keeps the elements it has and gets its own scale from its
    own `amax`. It is not merged into the previous block, and no padding
    survives into the output. This is numerically identical to zero-padding
    the tail out to a full block and trimming afterwards -- appending zeros
    changes neither `amax_b` nor any element's quantization -- so hardware
    that pads a partial block agrees with this function.

    Parameters
    ----------
    x : np.ndarray
        float64 input, any shape, all entries finite. NaN and infinity are
        rejected: neither has an E2M1 encoding, and either one would poison a
        whole block's `amax` -- and, with a global scale, the entire tensor.
        Not modified.
    block_size : int
        Elements per block along the last axis. The study uses 8, 16, 32 and
        64; 32 is the MX standard and 16 the NVFP4 one.
    scale_format : {"e8m0", "e4m3"}
        Format of the shared per-block scale. `"e8m0"` is a bare power of two
        (the MX scale), `"e4m3"` a 1/4/3-bit float (the NVFP4 scale).
    use_global_scale : bool
        Whether to normalise the tensor by one global scale first, so that the
        block scales are expressed relative to it. Needed for `"e4m3"` in
        practice; unnecessary for `"e8m0"`.
    element_format : {"e2m1"}, default "e2m1"
        Format of the elements within a block. Only E2M1 is implemented -- it
        is what both real formats store, and it is held fixed across the study
        so that the scaling is the only thing that varies.
    round_mode : {"rtne", "sr"}, default "rtne"
        Element rounding mode, passed through to `quantize_e2m1`. The scales
        are always rounded by their own format's rule regardless of this
        setting -- stochastic rounding applies to the elements within a block,
        not to the block or global scale.
    rng : np.random.Generator, optional
        Required for `round_mode="sr"`, ignored for `"rtne"`. The global NumPy
        RNG is never used.

    Returns
    -------
    np.ndarray
        float64 array of the same shape as `x`; every entry is an
        `element_format` grid value times its block's effective scale
        `s_b * s_global`.
    """
    _validate_formats(scale_format, element_format)
    _check_input(x, block_size)

    scales, s_global = _scales(x, block_size, scale_format, use_global_scale, element_format)
    # One scale per element: each block's scale repeated block_size times, with
    # the padding of a short final block trimmed back off.
    per_element = np.repeat(scales * s_global, block_size, axis=-1)[..., : x.shape[-1]]

    # A block whose scale is zero reconstructs to zero whatever the elements
    # round to, so divide by 1 there rather than turning the block into 0/0.
    # (Only reachable under an E4M3 scale; E8M0 has no zero.)
    divisor = np.where(per_element == 0.0, 1.0, per_element)
    q = quantize_e2m1(x / divisor, mode=round_mode, rng=rng)
    return q * per_element


# =============================================================================
# Presets: the two real formats
# =============================================================================


def mxfp4_block_scales(x: np.ndarray, block_size: int = MXFP4_BLOCK_SIZE) -> np.ndarray:
    """Per-block MXFP4 shared scales for float64 `x`, blocked along the last axis.

    Preset for `block_scales(x, block_size, "e8m0", use_global_scale=False)`.
    The scale of a block is the smallest power of two that keeps the block's
    largest magnitude inside the E2M1 range::

        s_b = 2 ** ceil(log2(amax_b / 6))

    so that `amax_b / s_b <= 6` always. See `quantize_mxfp4` for the round-up
    rationale, `_quantize_scale` for why the exponent is taken from `frexp`
    rather than `log2`, and `block_scales` for the return contract.
    """
    return block_scales(x, block_size, scale_format="e8m0", use_global_scale=False)


def quantize_mxfp4(
    x: np.ndarray,
    block_size: int = MXFP4_BLOCK_SIZE,
    rng: np.random.Generator | None = None,
    round_mode: RoundMode = "rtne",
) -> np.ndarray:
    """Quantize float64 `x` to MXFP4 (block-scaled E2M1); float64 in, float64 out.

    MXFP4 is the preset
    `quantize_blocked(x, 32, scale_format="e8m0", use_global_scale=False)`:
    E2M1 elements plus one shared power-of-two scale per block of 32 contiguous
    elements, and no global scale -- E8M0 spans 255 binades, which is more than
    enough to reach any block's magnitude on its own. As with the general
    function, the value returned is the *reconstruction* `q_i * s_b`; the
    scales come from `mxfp4_block_scales`.

    The recipe and the axis, tail and input policies are documented on
    `quantize_blocked`. What is specific to MXFP4 is the direction the scale
    exponent is rounded.

    Why the exponent is rounded up
    ------------------------------
    This is a deliberate design decision, not an implementation detail.
    Rounding the exponent **up** gives `amax_b / s_b <= 6` for every block, so
    the block's largest element -- the one that fixed the scale in the first
    place -- always lands inside the E2M1 range. Rounding to nearest, or down,
    would let `amax_b / s_b` reach almost 8, and everything above 6 saturates
    to exactly 6, clipping the block's largest value by up to 25% of its
    magnitude.

    The cost is real and worth stating: a scale one binade larger halves the
    resolution of every *other* element in the block. So the choice trades a
    bounded, uniform precision loss across the block against an unbounded
    clipping error on its largest element. This implementation takes that
    trade; the OCP MX specification and Microsoft's `microxcaling` reference
    take the other one, using `2 ** (floor(log2(amax_b)) - 2)` and accepting
    the saturation. The two rules agree whenever `amax_b`'s own significand is
    at most 1.5, i.e. for roughly three quarters of a log-uniform spread of
    magnitudes; see SPEC.md and `tests/test_blocks.py` for the comparison,
    which pins the scale exponent as the *only* difference between the two
    implementations.

    Parameters
    ----------
    x : np.ndarray
        float64 input, any shape, all entries finite. Not modified.
    block_size : int, default 32
        Elements per block along the last axis. 32 is the MX standard; other
        values are experimental controls and are not MXFP4 -- see
        `quantize_blocked`.
    rng : np.random.Generator, optional
        Required for `round_mode="sr"`, ignored for `"rtne"`.
    round_mode : {"rtne", "sr"}, default "rtne"
        Element rounding mode. The scale exponent is always rounded up.

    Returns
    -------
    np.ndarray
        float64 array of the same shape as `x`; every entry is an E2M1 grid
        value times its block's scale, and `|x_hat_i| <= 6 * s_b`.
    """
    return quantize_blocked(
        x,
        block_size,
        scale_format="e8m0",
        use_global_scale=False,
        round_mode=round_mode,
        rng=rng,
    )


def nvfp4_global_scale(x: np.ndarray) -> float:
    """The single per-tensor NVFP4 global scale for float64 `x`.

    Preset for `global_scale(x, scale_format="e4m3")`, i.e. `max |x| / (6 * 448)`
    -- or `1.0` where that underflows. See `quantize_nvfp4` for why NVFP4 needs
    it and `global_scale` for the general contract.
    """
    return global_scale(x, scale_format="e4m3")


def nvfp4_block_scales(x: np.ndarray, block_size: int = NVFP4_BLOCK_SIZE) -> np.ndarray:
    """Per-block NVFP4 E4M3 scales for float64 `x`, blocked along the last axis.

    Preset for `block_scales(x, block_size, "e4m3", use_global_scale=True)`.
    These are the *second* level of the two-level scaling and are relative to
    the global scale::

        s_b = quantize_e4m3((amax_b / s_global) / 6)

    so the effective scale of an element is
    `nvfp4_block_scales(x)[b] * nvfp4_global_scale(x)`. See `block_scales` for
    the return contract, including when an E4M3 scale of exactly zero occurs.
    """
    return block_scales(x, block_size, scale_format="e4m3", use_global_scale=True)


def quantize_nvfp4(
    x: np.ndarray,
    block_size: int = NVFP4_BLOCK_SIZE,
    rng: np.random.Generator | None = None,
    round_mode: RoundMode = "rtne",
) -> np.ndarray:
    """Quantize float64 `x` to NVFP4 (two-level block-scaled E2M1); float64 in/out.

    NVFP4 is the preset
    `quantize_blocked(x, 16, scale_format="e4m3", use_global_scale=True)`: the
    same E2M1 elements MXFP4 stores, scaled **twice** -- one E4M3 scale per
    block of 16 elements, and one global scale for the whole tensor. Like
    `quantize_mxfp4` this returns the *reconstruction* `q_i * s_b * s_global`;
    the two levels of scale come from `nvfp4_block_scales` and
    `nvfp4_global_scale`.

    The recipe and the axis, tail and input policies are documented on
    `quantize_blocked`. What is specific to NVFP4 is why there are two levels
    of scale at all.

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

    Dropping this step -- `use_global_scale=False` -- is the natural bug, and
    it has a diagnostic signature: block scales pinned at the top of E4M3
    means the blocks are barely being scaled at all, and NVFP4 measures
    *worse* than MXFP4, the opposite of what the format is for.
    `tests/test_blocks.py` asserts the accuracy ordering directly for that
    reason.

    Contrast with MXFP4
    -------------------
    The scaling is the whole difference, and it cuts both ways:

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

    Parameters
    ----------
    x : np.ndarray
        float64 input, any shape, all entries finite. Not modified.
    block_size : int, default 16
        Elements per block along the last axis. 16 is the NVFP4 standard --
        half of MXFP4's 32, which is the format's other accuracy lever. Other
        values are experimental controls and are not NVFP4 -- see
        `quantize_blocked`.
    rng : np.random.Generator, optional
        Required for `round_mode="sr"`, ignored for `"rtne"`.
    round_mode : {"rtne", "sr"}, default "rtne"
        Element rounding mode. Both levels of scale are always RTNE regardless.

    Returns
    -------
    np.ndarray
        float64 array of the same shape as `x`; every entry is an E2M1 grid
        value times its block's effective scale `s_b * s_global`.
    """
    return quantize_blocked(
        x,
        block_size,
        scale_format="e4m3",
        use_global_scale=True,
        round_mode=round_mode,
        rng=rng,
    )


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
