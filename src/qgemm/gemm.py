"""The quantized GEMM pipeline: transform, quantize, multiply, return.

Convention: float64 in, float64 out.

`qgemm(A, B, config)` computes an approximation of `A @ B` in which the
*operands* have been through a block-scaled low-precision format and the
product is then formed from their reconstructions. Everything the study varies
lives in `GemmConfig`; the function itself has no free parameters.

The four steps
--------------
1. **Optional RHT** (`config.rht`). A random Hadamard transform is applied per
   block along the **contraction dimension** -- the last axis of `A` and the
   *first* axis of `B` -- with the *same* `H'` on both operands. See
   `rht_operands`, which owns the transposes.
2. **Quantize** both operands with `qgemm.blocks.quantize_blocked`, again along
   the contraction dimension, using `config`'s block size, scale format, global
   scale, element format and rounding mode.
3. **Multiply the reconstructions** in the precision named by `config.accum`.
4. **Return** the float64 result. The RHT is *not* undone: an orthogonal
   transform applied to both operands cancels inside the inner product, so
   there is nothing to invert.

Why `accum="exact"` is the primary route
----------------------------------------
There are two independent sources of error in a low-precision GEMM: the
**input quantization** (operands snapped onto a coarse grid) and the
**accumulation** (partial sums rounded in a narrow accumulator). They compose,
and a single number measured with both active cannot be attributed to either.
The main experiment studies the first one, so it multiplies the reconstructed
operands in full float64: whatever error the result carries came from the
quantization of `A` and `B` and from nothing else. `accum="bf16"` is a
secondary ablation that turns the other source on deliberately, to measure how
much it adds -- not the default, and never mixed into a headline result.

The performance constraint
--------------------------
The exact route must do a 512x512x512 GEMM, quantization and RHT included, in
under a second (`tests/test_gemm.py`). This is a hard requirement rather than
a nicety: the sweep in a later phase calls this function once per
(shape, format, block size, rounding mode, seed) cell, thousands of times, and
a per-block Python loop anywhere in the path would push the main experiment
from minutes into hours. Every step here is a whole-array NumPy operation for
that reason. The `bf16` route, being an ablation over a handful of cells, is
exempt -- it loops over the contraction dimension by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import ml_dtypes
import numpy as np

from qgemm.blocks import ElementFormat, RoundMode, ScaleFormat, quantize_blocked
from qgemm.transforms import apply_rht

AccumMode = Literal["exact", "bf16"]

ACCUM_MODES = ("exact", "bf16")

# Seed streams derived from `GemmConfig.seed`, in this fixed order. Independent
# streams rather than one shared generator, so that turning the RHT on does not
# shift the stochastic-rounding draws (and vice versa) -- a config differs from
# another only in the ways it says it does.
_N_STREAMS = 3
_RHT_STREAM, _SR_A_STREAM, _SR_B_STREAM = range(_N_STREAMS)


@dataclass(frozen=True)
class GemmConfig:
    """Everything `qgemm` varies: the quantizer, the transform, the accumulator.

    Frozen and made of plain scalars, so it is hashable, `dataclasses.asdict`-
    able and therefore JSON-serialisable -- which is what the sweep needs to
    hash a config into a result filename (see README's result conventions).

    The defaults are **MXFP4 with exact accumulation and no RHT**: the study's
    baseline cell. NVFP4 is `GemmConfig(block_size=16, scale_format="e4m3",
    use_global_scale=True)`; see `qgemm.blocks.quantize_blocked` for which
    other combinations are real formats and which are experimental controls.

    Attributes
    ----------
    quantize : bool, default True
        Whether step 2 runs at all. `False` leaves the operands in float64 and
        is the control that isolates the RHT's and the accumulator's own
        effects -- with `quantize=False` and `accum="exact"` this pipeline is
        exactly `A @ B`.
    block_size, scale_format, use_global_scale, element_format, round_mode
        Passed straight through to `quantize_blocked`; documented there.
    rht : bool, default False
        Whether step 1 runs.
    rht_block_size : int, optional
        Block size for the transform. Defaults to `block_size`, i.e. the
        transform's blocks line up with the quantizer's, which is the setting
        the study sweeps; it is separable because the two need not agree.
        Must be a power of two dividing the contraction dimension.
    accum : {"exact", "bf16"}, default "exact"
        Accumulation precision for step 3. See the module docstring for why
        `"exact"` is the primary route.
    seed : int, default 0
        Seeds every random draw in the pipeline -- the RHT signs and the
        stochastic rounding of each operand -- through three independent
        streams. An `int` rather than a `numpy.random.Generator` so the config
        stays serialisable; the generators are built from it here and passed
        explicitly, and NumPy's global RNG is never touched.
    """

    quantize: bool = True
    block_size: int = 32
    scale_format: ScaleFormat = "e8m0"
    use_global_scale: bool = False
    element_format: ElementFormat = "e2m1"
    round_mode: RoundMode = "rtne"
    rht: bool = False
    rht_block_size: int | None = None  # resolved to `block_size` in __post_init__
    accum: AccumMode = "exact"
    seed: int = 0

    def __post_init__(self) -> None:
        if self.accum not in ACCUM_MODES:
            raise ValueError(f"unknown accum mode {self.accum!r}; expected one of {ACCUM_MODES}")
        if self.rht_block_size is None:
            object.__setattr__(self, "rht_block_size", self.block_size)

    def _generator(self, stream: int) -> np.random.Generator:
        return np.random.default_rng(np.random.SeedSequence(self.seed).spawn(_N_STREAMS)[stream])

    def rht_generator(self) -> np.random.Generator:
        """The generator `rht_operands` draws this config's shared sign vector from."""
        return self._generator(_RHT_STREAM)

    def rounding_generators(self) -> tuple[np.random.Generator, np.random.Generator]:
        """Independent stochastic-rounding streams for `A` and for `B`.

        Two streams, not one: sharing a draw between the operands would
        correlate their rounding errors, and correlated errors do not cancel in
        the inner product the way independent ones do -- the whole reason
        stochastic rounding is worth measuring.
        """
        return self._generator(_SR_A_STREAM), self._generator(_SR_B_STREAM)


def rht_operands(
    a: np.ndarray,
    b: np.ndarray,
    block_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one shared random Hadamard transform to `A` and `B` along the contraction axis.

    ::

        A_hat = H' A      (blocks along A's last axis)
        B_hat = H' B      (blocks along B's FIRST axis)

    so that `A_hat @ B_hat == A @ B` up to float64 rounding, because every
    inner product in the product is `(H'a).(H'b) = a.(H'.T H' b) = a.b`.

    **The transposes are the whole content of this function.** `apply_rht`
    blocks along the last axis, which for `A` (shape `(M, K)`) is already the
    contraction dimension `K`, but for `B` (shape `(K, N)`) is `N`. `B`
    therefore has to be transposed *before* the call and transposed back
    afterwards. Handing `B` over untransposed rotates the wrong axis: it
    raises nothing, returns an array of the right shape whenever `K == N`, and
    silently computes a different quantity -- the invariant above does not
    hold, so the RHT no longer cancels and the "unquantized" product comes out
    wrong. `tests/test_gemm.py` pins this against a reference that spells the
    transposes out.

    Both operands must be transformed by the **same** `H'`, so the sign draw is
    made once: a shared seed is drawn from `rng`, and two fresh generators are
    built from it, one per `apply_rht` call. (`apply_rht` derives `H'` from its
    generator's state, so generators in the same state give the same `H'`.)

    Parameters
    ----------
    a, b : np.ndarray
        float64, shapes `(M, K)` and `(K, N)`.
    block_size : int
        Power of two dividing `K`. The RHT has no partial-block form, so a `K`
        that is not a whole number of blocks raises `ValueError`.
    rng : np.random.Generator
        Explicit generator; consumes one `integers` draw. Never the global RNG.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        `(A_hat, B_hat)`, float64, same shapes as the inputs.
    """
    shared_seed = int(rng.integers(1 << 63))
    a_hat = apply_rht(a, block_size, np.random.default_rng(shared_seed))
    b_hat = apply_rht(b.T, block_size, np.random.default_rng(shared_seed)).T
    return a_hat, b_hat


def _quantize_operands(
    a: np.ndarray, b: np.ndarray, config: GemmConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Step 2: both operands quantized along the contraction axis.

    Same transpose question as `rht_operands`, for the same reason:
    `quantize_blocked` blocks along the last axis, so `B` is quantized as `B.T`
    and transposed back. Blocking `B` along `N` would share a scale across the
    wrong axis -- again silently, and again measuring something else.
    """
    rng_a, rng_b = config.rounding_generators()
    kwargs = dict(
        block_size=config.block_size,
        scale_format=config.scale_format,
        use_global_scale=config.use_global_scale,
        element_format=config.element_format,
        round_mode=config.round_mode,
    )
    return (
        quantize_blocked(a, rng=rng_a, **kwargs),
        quantize_blocked(b.T, rng=rng_b, **kwargs).T,
    )


def _round_bf16(x: np.ndarray) -> np.ndarray:
    """Round float64 `x` onto the bfloat16 grid (RTNE), returning float64."""
    return x.astype(ml_dtypes.bfloat16).astype(np.float64)


def _accumulate_bf16(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Sequential `A @ B` with the accumulator rounded to bfloat16 after each term.

    One explicit pass over the contraction dimension: term `k` is the outer
    product of `A`'s k-th column and `B`'s k-th row, formed in float64 and
    added to an accumulator that is rounded back to bf16 immediately. That is
    the error model of a bf16 accumulator fed exactly-representable products,
    and it is the ablation described in the module docstring -- deliberately
    *not* the main route, and deliberately not optimized. Speed is irrelevant
    here; the loop is over `K`, not over blocks or elements.
    """
    out = np.zeros((a.shape[0], b.shape[1]), dtype=np.float64)
    for k in range(a.shape[1]):
        out = _round_bf16(out + a[:, k, None] * b[None, k, :])
    return out


def _check_operands(a: np.ndarray, b: np.ndarray) -> None:
    if a.dtype != np.float64 or b.dtype != np.float64:
        raise TypeError(f"expected float64 inputs, got {a.dtype} and {b.dtype}")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"expected two 2-D operands, got shapes {a.shape} and {b.shape}")
    if a.shape[1] != b.shape[0]:
        raise ValueError(
            f"contraction dimension mismatch: A is {a.shape} and B is {b.shape}, "
            f"so {a.shape[1]} != {b.shape[0]}"
        )


def qgemm(a: np.ndarray, b: np.ndarray, config: GemmConfig) -> np.ndarray:
    """Compute `A @ B` with block-quantized operands; float64 in, float64 out.

    The pipeline, in order:

    1. **Transform** (if `config.rht`): one shared random Hadamard transform on
       both operands along the contraction dimension, via `rht_operands`. It
       cancels in the inner product, so it changes nothing here on its own --
       what it changes is what quantization does next, by spreading each
       block's energy so a single outlier no longer sets the block's scale.
    2. **Quantize** both operands with `quantize_blocked`, blocked along the
       contraction dimension (`B` is quantized transposed, then transposed
       back). Skipped entirely when `config.quantize` is False.
    3. **Multiply the reconstructions** -- the dequantized float64 values, not
       the codes -- in `config.accum` precision: `np.matmul` in float64 for
       `"exact"`, an explicit bf16-rounded accumulation loop for `"bf16"`.
    4. **Return** the product. The RHT is not undone; there is nothing to undo.

    `accum="exact"` is the primary route because it isolates the error this
    study is about. Quantizing the inputs and accumulating in low precision are
    two separate error sources; measured together they cannot be told apart,
    and a downstream result could not be attributed to one or the other. Under
    `"exact"` the only approximation in the whole pipeline is step 2, so the
    residual `qgemm(A, B, config) - A @ B` *is* the input-quantization error.
    `"bf16"` exists to measure the second source on its own terms and is an
    ablation, not a baseline.

    Parameters
    ----------
    a, b : np.ndarray
        float64 operands of shape `(M, K)` and `(K, N)`, all entries finite
        (`quantize_blocked` rejects NaN and infinity). Not modified.
    config : GemmConfig
        The quantizer parameters, whether the RHT runs, the accumulation mode
        and the seed. See `GemmConfig`.

    Returns
    -------
    np.ndarray
        float64 array of shape `(M, N)`.

    Raises
    ------
    TypeError
        If either operand is not float64.
    ValueError
        If either operand is not 2-D, if their contraction dimensions disagree,
        or if the RHT is on and `config.rht_block_size` does not divide `K`.

    Notes
    -----
    With `quantize=False` and `accum="exact"` this returns `a @ b` bit for bit:
    the pipeline degenerates to the float64 matmul, which is the control the
    quantized cells are compared against.
    """
    _check_operands(a, b)

    if config.rht:
        a, b = rht_operands(a, b, config.rht_block_size, config.rht_generator())
    if config.quantize:
        a, b = _quantize_operands(a, b, config)

    if config.accum == "bf16":
        return _accumulate_bf16(a, b)
    return np.matmul(a, b)
