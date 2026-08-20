"""Pre/post-quantization transforms (scale computation, rotations, etc.).

Convention: float64 in, float64 out.

Implemented so far: the **random Hadamard transform** (RHT), applied per block
along the last axis. See `apply_rht` for the recipe, the invariant it depends
on, and why a global (whole-row) variant is deliberately not offered.
"""

from __future__ import annotations

import numpy as np

# The block sizes this project sweeps. Any power of two is a valid argument --
# the Sylvester construction needs nothing else -- but these four are the ones
# the study measures, matching the block sizes used in `qgemm.blocks`.
RHT_BLOCK_SIZES = (8, 16, 32, 64)


# --- random Hadamard transform ------------------------------------------------


def _validate_block_size(block_size: int) -> None:
    if (
        not isinstance(block_size, (int, np.integer))
        or block_size < 1
        or block_size & (block_size - 1)
    ):
        raise ValueError(
            f"block_size must be a power of two (this project sweeps "
            f"{list(RHT_BLOCK_SIZES)}), got {block_size!r}"
        )


def _check_input(x: np.ndarray, block_size: int) -> None:
    if x.dtype != np.float64:
        raise TypeError(f"expected float64 input, got {x.dtype}")
    if x.shape[-1] % block_size:
        raise ValueError(
            f"last axis of length {x.shape[-1]} is not a multiple of block_size "
            f"{block_size}; the RHT has no partial-block form"
        )


def hadamard_matrix(block_size: int) -> np.ndarray:
    """The normalized Sylvester Hadamard matrix `H` of order `block_size`.

    ::

        H_1 = [1]
        H_2n = (1/sqrt(2)) * [[H_n,  H_n],
                              [H_n, -H_n]]

    `H` is symmetric and orthogonal (`H @ H.T == I`), and every entry has the
    same magnitude `1/sqrt(block_size)` -- which is the whole point: a block
    holding a single spike of magnitude `a` maps to a block whose every entry
    has magnitude `a/sqrt(block_size)`, so the block's `amax` (and with it the
    shared scale a block-quantized format must pick) drops by that factor.

    Built by doubling a `+-1` matrix and dividing once at the end rather than
    by multiplying in `1/sqrt(2)` at each level. The two are mathematically
    identical; the single division keeps the entries exact whenever
    `sqrt(block_size)` is (16 and 64 here), which is what lets the dispersion
    property be asserted with `==` rather than a tolerance.

    Raises
    ------
    ValueError
        If `block_size` is not a power of two.
    """
    _validate_block_size(block_size)

    h = np.ones((1, 1), dtype=np.float64)
    while h.shape[0] < block_size:
        h = np.block([[h, h], [h, -h]])
    return h / np.sqrt(block_size)


def random_signs(block_size: int, rng: np.random.Generator) -> np.ndarray:
    """`block_size` signs drawn uniformly from `{-1, +1}` using `rng`.

    This is the `eps` of the randomization `H' = H @ diag(eps)`. It is public
    because reproducing a transform means reproducing this draw: the caller
    owns the generator state, and two calls that must agree (the two operands
    of a GEMM, or a transform and its inverse) agree exactly when they pass
    generators in the same state. Never uses NumPy's global RNG.
    """
    _validate_block_size(block_size)

    return 1.0 - 2.0 * rng.integers(0, 2, size=block_size).astype(np.float64)


def _randomized_hadamard(block_size: int, rng: np.random.Generator) -> np.ndarray:
    """`H' = H @ diag(eps)` -- scaling `H`'s *columns* by the signs."""
    return hadamard_matrix(block_size) * random_signs(block_size, rng)


def _effective_block_size(x: np.ndarray, block_size: int) -> int:
    """The block size actually used for `x`'s last axis.

    Equal to `block_size` whenever the row holds at least that many elements.
    When the row is *shorter* than the nominal `block_size` -- so the whole
    row is a single block with fewer than `block_size` elements in it --
    this falls back to the row's own length, mirroring
    `qgemm.blocks.quantize_blocked`'s short-block convention: a block that
    doesn't have enough elements to fill the nominal size gets sized to what
    it actually has (see that function's "Tail policy").

    This is deliberately narrow: it does **not** extend to a row that is
    *longer* than `block_size` but not a whole multiple of it (e.g. `n=100`,
    `block_size=32`) -- that case still raises `ValueError` via
    `_check_input`, unchanged. A genuine trailing partial block (some full
    blocks followed by a short remainder) has no single `H'` it could be
    multiplied by without picking an axis to apply that `H'` along a block
    boundary that doesn't exist; only the "one block, and it's short" case
    -- `block_size > n` -- has an unambiguous fallback (the whole row *is*
    that one block), which is the only case this project's grid ever
    produces (every `n` and `block_size` here is a power of two, and
    `block_size` only ever exceeds `n` when `n` itself is the whole row).
    """
    return min(block_size, x.shape[-1])


def _rht(x: np.ndarray, block_size: int, rng: np.random.Generator, inverse: bool) -> np.ndarray:
    _validate_block_size(block_size)
    effective_block_size = _effective_block_size(x, block_size)
    if effective_block_size != block_size:
        # Row shorter than the nominal block_size: falls back to a single
        # block spanning the whole row. That fallback size is not guaranteed
        # to be a power of two just because the nominal block_size was (e.g.
        # block_size=64, n=100 would fall back to 100, which is invalid) --
        # validated explicitly here rather than assumed, even though every
        # (n, block_size) pair this project actually sweeps is a power of
        # two and this branch is therefore never expected to raise in
        # practice.
        _validate_block_size(effective_block_size)
    _check_input(x, effective_block_size)

    h = _randomized_hadamard(effective_block_size, rng)
    # Blocks are rows of the reshaped array, so `H' a` for a column `a` is
    # `a_row @ H'.T`. The inverse is the transpose, `H'.T y -> y_row @ H'`.
    n_blocks = x.shape[-1] // effective_block_size
    blocks = x.reshape(*x.shape[:-1], n_blocks, effective_block_size)
    return (blocks @ (h if inverse else h.T)).reshape(x.shape)


def apply_rht(x: np.ndarray, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """Apply a random Hadamard transform to each block of `x` along the last axis.

    ::

        H' = H @ diag(eps),   eps_i drawn uniformly from {-1, +1}
        x_hat_b = H' @ x_b    for each contiguous block x_b of block_size elements

    `H` is the normalized Sylvester matrix (`hadamard_matrix`) and `H'` is
    orthogonal too, since `diag(eps)` is. The transform is therefore an
    isometry: it changes how a block's energy is *distributed* across its
    entries without changing how much there is. That is what makes it useful
    before block quantization -- a block whose `amax` is set by one outlier
    pays for that outlier in every other element's resolution, and the RHT
    spreads the outlier out. A lone spike of magnitude `a` in a block becomes
    `block_size` entries of magnitude `a/sqrt(block_size)` exactly.

    The randomization matters. A bare `H` has fixed structure and hits
    pathological inputs -- a block already proportional to a row of `H`
    concentrates into a single spike rather than being spread. Random signs
    make that a measure-zero coincidence for any fixed input instead of a
    property of the data.

    Preserving the inner product
    ----------------------------
    **The point of using an orthogonal transform is that it cancels across a
    GEMM, and it only cancels if BOTH operands get the same `H'`:**

    ::

        (H'a).(H'b) = a.(H'.T H' b) = a.b

    This function transforms one array. Nothing here can check that the other
    operand was transformed to match -- the invariant is a property of how the
    two calls are made, and it is enforced at the call site in the GEMM
    pipeline (Step 1.7, `qgemm.gemm`). Two calls agree exactly when they are
    passed generators in the same state, e.g. two `np.random.default_rng(seed)`
    built from one seed, since `H'` depends on `rng` only through
    `random_signs`. Transforming one operand and not the other, or with a
    different draw, computes a different quantity and raises nothing.

    Why per block, and why there is no global variant
    -------------------------------------------------
    Blocks are `block_size` contiguous elements along the **last axis**, the
    same convention as `qgemm.blocks.quantize_blocked`, because that is the
    axis a block-quantized format shares a scale over and (for `A @ B.T`) the
    contraction axis a shared rotation cancels along.

    A global RHT over the whole row of length `n` is a real technique and is
    **deliberately excluded from this project** -- not merely unimplemented.
    Its spread factor is `sqrt(n)` rather than `sqrt(block_size)`, which would
    tie the RHT's effect to the same `n` whose error-growth law the study
    sweeps, confounding the two; keeping it per block leaves `block_size` as
    the only knob that governs the transform. Do not add one "for
    completeness".

    Tail policy
    -----------
    Unlike `quantize_blocked`, which keeps a trailing partial block short, a
    row **longer** than `block_size` must be an exact multiple of it: a
    genuine trailing partial block (full blocks followed by a short
    remainder) has no single `H'` it could be multiplied by, and padding it
    would change the array's shape and break invertibility, so that mismatch
    still raises `ValueError`.

    A row **shorter** than `block_size` is different and is handled: the
    whole row is unambiguously one block, so `apply_rht` falls back to
    transforming it at its own length instead of raising, mirroring
    `quantize_blocked`'s short-block convention (`_effective_block_size`).
    **This has a real, physical consequence for the RHT's spreading power in
    that one corner, not just a shape accommodation**: the transform's spread
    factor is `sqrt(effective_block_size)`, so when the row is shorter than
    the nominal `block_size`, the achieved spread is `sqrt(n)` rather than
    the `sqrt(block_size)` a full-length block would give -- strictly less,
    since `n < block_size` there. This is a physically necessary limitation
    (an outlier cannot be spread across more elements than the row contains),
    not a silent scope change: a lone spike of magnitude `a` in a `block_size
    = 32` configuration ordinarily comes out at `a / sqrt(32)` per entry, but
    at `n = 16` it comes out at `a / sqrt(16)` instead, `sqrt(2)` times
    larger. Only this project's `(n=16, block_size=32, rht=True)` grid cells
    hit this fallback (SPEC.md, "Sweep grid (Step 3.1)"); every other cell
    has `n >= block_size` and is unaffected.

    Parameters
    ----------
    x : np.ndarray
        float64 array. Leading axes are independent rows.
    block_size : int
        A power of two. Normally a divisor of `x.shape[-1]`; if
        `x.shape[-1] < block_size`, the whole row is used as a single block
        instead (see "Tail policy" above), and `x.shape[-1]` itself must then
        be a power of two. The project sweeps `RHT_BLOCK_SIZES`.
    rng : np.random.Generator
        Explicit generator for the sign draw; consumes one `integers` call of
        `effective_block_size` values (`block_size`, or `x.shape[-1]` when
        the row is shorter -- see "Tail policy").

    Returns
    -------
    np.ndarray
        float64 array of the same shape as `x`.

    Raises
    ------
    TypeError
        If `x` is not float64.
    ValueError
        If `block_size` is not a power of two; if `x.shape[-1] > block_size`
        and does not divide evenly by it; or if `x.shape[-1] < block_size`
        and `x.shape[-1]` is itself not a power of two.

    See Also
    --------
    invert_rht : the exact inverse, given a generator in the same state.
    """
    return _rht(x, block_size, rng, inverse=False)


def invert_rht(x: np.ndarray, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """Undo `apply_rht`, per block along the last axis.

    ::

        x_b = H'.T @ x_hat_b

    `H'` is orthogonal, so its inverse is its transpose and the round trip is
    exact up to float64 rounding.

    **`rng` must be in the same state as the one `apply_rht` was given**, since
    both reconstruct `H'` from the same sign draw -- typically a freshly
    constructed `np.random.default_rng(seed)` on either side. A generator in
    any other state produces a different, equally valid orthogonal matrix and
    silently returns something that is not `x`; nothing about the argument
    reveals which draw produced it, so this cannot be checked here.

    Same parameters, contract and errors as `apply_rht`.
    """
    return _rht(x, block_size, rng, inverse=True)


# --- scale computation (TBD) --------------------------------------------------


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
