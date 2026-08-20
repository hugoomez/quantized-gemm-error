"""Error bounds and the measured quantities they are built from.

Two things live here, and they are not the same kind of object:

* `gamma_n` -- the classical (Higham-style) worst-case bound, a closed form
  taken from the literature.
* `measure_u_eff` and its helpers -- an **empirical** measurement of the
  effective unit roundoff of a block-scaled quantizer, obtained by quantizing
  samples and looking at the per-element relative error directly.

Reference for the classical part: N. J. Higham, "Accuracy and Stability of
Numerical Algorithms", 2nd ed., SIAM, 2002.

Convention: float64 in, float64 out; every stochastic function takes an
explicit `numpy.random.Generator`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from qgemm.blocks import block_scales, global_scale, quantize_blocked
from qgemm.formats import E2M1_MAGNITUDE_GRID
from qgemm.gemm import GemmConfig

# Smallest nonzero magnitude each element format can represent, in units of the
# element's effective scale. Only E2M1 is implemented (`qgemm.blocks` rejects
# anything else), and its grid is `0, 0.5, 1, 1.5, 2, 3, 4, 6`, so the smallest
# nonzero magnitude is `0.5`.
_ELEMENT_FORMAT_MIN_NONZERO = {"e2m1": float(E2M1_MAGNITUDE_GRID[1])}

# Default near-zero policy for `elementwise_relative_error`, in units of the
# element's effective scale. Under round-to-nearest an element whose magnitude
# is at or below half the smallest representable nonzero magnitude quantizes to
# exactly zero, so `0.5 * 0.5 = 0.25` is precisely E2M1's flush-to-zero
# boundary. See `elementwise_relative_error` for why that is the cut.
DEFAULT_ZERO_THRESHOLD = _ELEMENT_FORMAT_MIN_NONZERO["e2m1"] / 2.0

# A sampler takes a shape and an explicit generator and returns float64, which
# is the contract `qgemm.distributions`' samplers already follow.
DistSampler = Callable[[tuple[int, ...], np.random.Generator], np.ndarray]


def gamma_n(n: int, u: float) -> float:
    """Higham's gamma_n(u) = n*u / (1 - n*u).

    Standard bound on the relative error growth from `n` sequential
    floating-point roundoffs at unit roundoff `u`. This is a conservative
    (worst-case, sequential-summation) bound: algorithms with better
    dependency structure (e.g. pairwise/cascade summation, as used by
    `numpy.sum`) satisfy strictly smaller error bounds, so an empirical
    error exceeding `gamma_n` indicates a real problem, not just a loose
    bound.
    """
    if n * u >= 1:
        raise ValueError("n * u must be < 1 for gamma_n to be defined")
    return n * u / (1 - n * u)


def _effective_scales(x: np.ndarray, config: GemmConfig) -> np.ndarray:
    """The scale each element of `x` is actually quantized against.

    `block_scales` returns one entry per block, expressed in the global
    scale's units, so the scale an element sees is
    `block_scale[b] * s_global`. Broadcast back out to one entry per element,
    trimming the repeat for a short trailing block.
    """
    s_global = (
        global_scale(x, config.scale_format, config.element_format)
        if config.use_global_scale
        else 1.0
    )
    scales = block_scales(
        x,
        config.block_size,
        config.scale_format,
        config.use_global_scale,
        config.element_format,
    )
    per_element = np.repeat(scales, config.block_size, axis=-1)[..., : x.shape[-1]]
    return per_element * s_global


def elementwise_relative_error(
    x: np.ndarray,
    config: GemmConfig,
    *,
    rng: np.random.Generator | None = None,
    zero_threshold: float = DEFAULT_ZERO_THRESHOLD,
) -> np.ndarray:
    """Per-element relative quantization error `|x_i - x_hat_i| / |x_i|`, filtered.

    `x` is quantized with `qgemm.blocks.quantize_blocked` using `config`'s
    quantizer parameters (`block_size`, `scale_format`, `use_global_scale`,
    `element_format`, `round_mode`). The rest of `GemmConfig` -- `rht`,
    `accum`, `seed` -- describes the GEMM around the quantizer and is **not**
    read here; `quantize=False` is rejected rather than ignored, since the
    relative error of an unquantized tensor is identically zero and measuring
    it would mean nothing.

    The near-zero policy (a real methodological choice, not a detail)
    ------------------------------------------------------------------
    Relative error is ill-defined as `x_i -> 0`, and for a block-scaled format
    "near zero" only means anything **relative to the element's own block
    scale**: block scales here span many orders of magnitude across a
    heavy-tailed tensor, so a fixed absolute floor would cut different formats
    at different places and make the comparison meaningless.

    The cut is therefore taken at `zero_threshold * s_eff`, where `s_eff` is
    the effective scale (block scale times global scale) the element was
    quantized against. The default, `0.25`, is not a tuning knob: E2M1's
    smallest nonzero magnitude is `0.5 * s_eff`, so under round-to-nearest
    every element at or below `0.25 * s_eff` quantizes to exactly zero and
    carries a relative error of exactly `1.0`. Those elements record the
    format's **dynamic-range floor**, not its **precision**, and they are what
    `u_eff` is being asked about. Keeping them makes every high quantile
    saturate at `1.0` -- measured, not assumed: over 2^20 draws every
    configuration tested returns `p99 = 1.00000` exactly with the cut turned
    off, which is a statement about how much near-zero mass the distribution
    has and none at all about the grid.

    The price is that the excluded fraction is large and varies by cell (about
    10% of Gaussian elements, but nearly two thirds under t-Student(1) at
    block 32), so any use of this function must report how much it dropped.
    `u_eff_samples` returns the number of elements drawn for exactly that
    reason. Passing `zero_threshold=0.0` disables the cut and keeps everything
    except exact zeros, where the ratio has no value at all.

    Parameters
    ----------
    x : np.ndarray
        float64, blocked along the last axis, as `quantize_blocked` blocks it.
    config : GemmConfig
        Quantizer parameters; see above for which fields are read.
    rng : np.random.Generator, optional
        Required when `config.round_mode == "sr"`, unused otherwise. Never the
        global RNG.
    zero_threshold : float, default 0.25
        The near-zero cut, in units of the element's effective scale.

    Returns
    -------
    np.ndarray
        float64, 1-D, one entry per **surviving** element -- shorter than `x`
        whenever the cut fires, so the caller cannot assume alignment with `x`.
    """
    if not config.quantize:
        raise ValueError("config.quantize is False; there is no quantization error to measure")
    if zero_threshold < 0.0:
        raise ValueError(f"zero_threshold must be non-negative, got {zero_threshold}")

    x = np.asarray(x, dtype=np.float64)
    x_hat = quantize_blocked(
        x,
        block_size=config.block_size,
        scale_format=config.scale_format,
        use_global_scale=config.use_global_scale,
        element_format=config.element_format,
        round_mode=config.round_mode,
        rng=rng,
    )
    scales = _effective_scales(x, config)

    magnitude = np.abs(x)
    # `scales > 0` is not redundant: an E4M3 block scale *can* be zero, for a
    # block whose amax falls under the smallest E4M3 subnormal relative to the
    # global scale. Such a block flushes entirely and resolves nothing.
    keep = (magnitude > zero_threshold * scales) & (magnitude > 0.0) & (scales > 0.0)
    return np.abs(x[keep] - x_hat[keep]) / magnitude[keep]


def u_eff_samples(
    config: GemmConfig,
    dist_sampler: DistSampler,
    n_elements: int,
    rng: np.random.Generator,
    *,
    tensor_shape: tuple[int, ...] | None = None,
    zero_threshold: float = DEFAULT_ZERO_THRESHOLD,
) -> tuple[np.ndarray, int]:
    """Draw, quantize, and return the surviving per-element relative errors.

    Returns `(errors, n_drawn)`. `n_drawn` is how many elements were sampled
    *before* the near-zero cut, so `errors.size / n_drawn` is the surviving
    fraction the near-zero policy leaves behind -- a number every caller should
    report alongside its quantiles.

    Why `tensor_shape` exists
    -------------------------
    `use_global_scale=True` (the NVFP4 pairing) computes one scale from the
    **whole tensor's** amax, so the result depends on how much data is handed
    over at once: a single 4M-element draw has a far larger amax than a
    realistic operand, which shrinks every block scale relative to it and
    changes the E4M3 scale quantization. Passing `tensor_shape` draws that many
    independent tensors of that shape instead, so per-tensor quantities are
    computed over a tensor the size the study actually multiplies. The element
    count rounds **up** to a whole number of tensors. With `tensor_shape=None`
    the whole budget is one 1-D tensor, which is exact for the formats that
    have no global scale.
    """
    if n_elements <= 0:
        raise ValueError(f"n_elements must be positive, got {n_elements}")

    if tensor_shape is None:
        shape: tuple[int, ...] = (int(n_elements),)
        n_chunks = 1
    else:
        shape = tuple(int(d) for d in tensor_shape)
        per_chunk = int(np.prod(shape))
        if per_chunk <= 0:
            raise ValueError(f"tensor_shape must have positive extent, got {tensor_shape}")
        n_chunks = -(-int(n_elements) // per_chunk)

    parts: list[np.ndarray] = []
    n_drawn = 0
    for _ in range(n_chunks):
        x = np.asarray(dist_sampler(shape, rng), dtype=np.float64)
        n_drawn += x.size
        parts.append(
            elementwise_relative_error(x, config, rng=rng, zero_threshold=zero_threshold)
        )

    return np.concatenate(parts), n_drawn


def measure_u_eff(
    config: GemmConfig,
    dist_sampler: DistSampler,
    n_elements: int,
    quantiles: Sequence[float] = (0.5, 0.99),
    *,
    rng: np.random.Generator,
    tensor_shape: tuple[int, ...] | None = None,
    zero_threshold: float = DEFAULT_ZERO_THRESHOLD,
) -> dict[float, float]:
    """Quantiles of the per-element relative quantization error of `config`.

    This is the **effective unit roundoff** of a block-scaled quantizer,
    measured directly rather than inferred through a downstream error metric:
    draw `n_elements` values from `dist_sampler`, quantize them under
    `config`, and read the requested quantiles off the resulting relative-error
    distribution.

    The classical `u` is a property of a format alone -- half an ulp of a fixed
    grid. A block-scaled format has no such number: the grid an element lands
    on depends on the largest magnitude in its block, so the achieved relative
    error is a *distribution* whose shape depends on the input, on the block
    size and on the scale format together. `u_eff` is a summary of that
    distribution and is therefore reported at more than one quantile.

    See `elementwise_relative_error` for the near-zero policy, which is a real
    methodological choice and materially affects these numbers, and
    `u_eff_samples` for `tensor_shape` and for recovering the surviving
    fraction the cut leaves behind.

    Parameters
    ----------
    config : GemmConfig
        Quantizer parameters. `quantize=False` is rejected.
    dist_sampler : callable
        `(shape, rng) -> float64 array` -- the contract
        `qgemm.distributions.sample_gaussian` already satisfies. Samplers with
        extra parameters (a t-Student's `nu`) are bound by the caller.
    n_elements : int
        How many elements to draw, before the near-zero cut. Extreme quantiles
        need this large; see `scripts/measure_u_eff.py` for the stability check
        that fixes the value the study uses.
    quantiles : sequence of float, default (0.5, 0.99)
        Each in `[0, 1]`.
    rng : numpy.random.Generator
        Explicit generator, keyword-only. Never the global RNG.

    Returns
    -------
    dict[float, float]
        `{quantile: u_eff}`, in the order requested.

    Raises
    ------
    ValueError
        If `config.quantize` is False, if `n_elements` is not positive, if any
        quantile falls outside `[0, 1]`, or if the near-zero cut leaves nothing
        to take a quantile of.
    """
    for q in quantiles:
        if not 0.0 <= q <= 1.0:
            raise ValueError(f"quantile must lie in [0, 1], got {q}")

    errors, n_drawn = u_eff_samples(
        config,
        dist_sampler,
        n_elements,
        rng,
        tensor_shape=tensor_shape,
        zero_threshold=zero_threshold,
    )
    if errors.size == 0:
        raise ValueError(
            f"the near-zero cut (zero_threshold={zero_threshold}) excluded all {n_drawn} "
            "drawn elements, so there is no relative-error distribution to summarize"
        )

    values = np.quantile(errors, np.asarray(quantiles, dtype=np.float64))
    return {float(q): float(v) for q, v in zip(quantiles, np.atleast_1d(values), strict=True)}
