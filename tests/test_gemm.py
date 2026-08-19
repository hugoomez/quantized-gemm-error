import time

import numpy as np
import pytest

from qgemm.gemm import GemmConfig, qgemm, rht_operands
from qgemm.metrics import rms_error
from qgemm.transforms import apply_rht

# Quantization off: the pipeline degenerates to a plain float64 matmul, which is
# what the identity and RHT-non-interference tests pin down.
EXACT = GemmConfig(quantize=False)
# MXFP4 (the `quantize_blocked` defaults), exact accumulation -- the main route.
MXFP4 = GemmConfig()


def _operands(m: int, k: int, n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Two float64 operands with *different* structure, so a transposed-away bug shows."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((m, k))
    # B gets a per-row ramp, so B and B.T have visibly different block structure and
    # an RHT applied to the wrong axis of B cannot coincidentally agree with the right one.
    b = rng.standard_normal((k, n)) * (1.0 + np.arange(k, dtype=np.float64))[:, None]
    return a, b


# --- exactness of the unquantized route ---------------------------------------


def test_exact_route_without_quantization_reproduces_the_float64_product():
    a, b = _operands(24, 64, 40)

    assert np.array_equal(qgemm(a, b, EXACT), a @ b)


def test_exact_route_returns_float64():
    a, b = _operands(8, 32, 8)

    out = qgemm(a, b, MXFP4)

    assert out.dtype == np.float64
    assert out.shape == (8, 8)


def test_quantization_changes_the_product():
    a, b = _operands(16, 64, 16)

    assert not np.allclose(qgemm(a, b, MXFP4), a @ b)


# --- RHT: non-interference and axis correctness --------------------------------


def test_rht_without_quantization_leaves_the_product_unchanged():
    # The invariant (H'a).(H'b) = a.b holds exactly in exact arithmetic, so with
    # nothing between the transform and the product only float64 rounding separates
    # the two. This is what makes the RHT free to help once quantization is added.
    a, b = _operands(24, 64, 40)
    config = GemmConfig(quantize=False, rht=True, rht_block_size=32)

    out = qgemm(a, b, config)

    assert np.allclose(out, a @ b, rtol=0, atol=1e-9 * np.abs(a @ b).max())


def test_rht_transforms_the_contraction_axis_of_both_operands():
    # The reference spells the transposes out: B's contraction axis is its FIRST,
    # so it is B.T that gets handed to `apply_rht` (which blocks along the last axis)
    # and the result transposed back. Both operands take the same H'.
    a, b = _operands(24, 64, 40)
    block_size, seed = 32, 7

    a_hat, b_hat = rht_operands(a, b, block_size, np.random.default_rng(seed))

    shared = int(np.random.default_rng(seed).integers(1 << 63))
    expected_a = apply_rht(a, block_size, np.random.default_rng(shared))
    expected_b = apply_rht(b.T, block_size, np.random.default_rng(shared)).T
    assert np.array_equal(a_hat, expected_a)
    assert np.array_equal(b_hat, expected_b)


def test_rht_on_the_wrong_axis_of_b_would_be_caught():
    # Guards the test above: with a square B the wrong axis raises nothing and
    # returns a well-formed array, so the reference has to actually disagree with it.
    a, b = _operands(24, 64, 64)
    block_size, seed = 32, 7

    _, b_hat = rht_operands(a, b, block_size, np.random.default_rng(seed))

    shared = int(np.random.default_rng(seed).integers(1 << 63))
    wrong_axis = apply_rht(b, block_size, np.random.default_rng(shared))
    assert not np.allclose(b_hat, wrong_axis)


def test_rht_on_the_wrong_axis_of_b_breaks_the_product():
    # The consequence of the bug, stated at the level of the GEMM: an RHT applied
    # to B's non-contraction axis does not cancel, so even with quantization off
    # the product is simply wrong.
    a, b = _operands(24, 64, 64)
    block_size, shared = 32, 12345

    a_hat = apply_rht(a, block_size, np.random.default_rng(shared))
    b_wrong = apply_rht(b, block_size, np.random.default_rng(shared))

    assert not np.allclose(a_hat @ b_wrong, a @ b)


def test_rht_operands_leave_shapes_unchanged():
    a, b = _operands(24, 64, 40)

    a_hat, b_hat = rht_operands(a, b, 32, np.random.default_rng(0))

    assert a_hat.shape == a.shape
    assert b_hat.shape == b.shape


def test_qgemm_with_rht_matches_the_hand_built_pipeline():
    from qgemm.blocks import quantize_blocked

    a, b = _operands(24, 64, 40)
    config = GemmConfig(rht=True, rht_block_size=32, seed=3)

    a_hat, b_hat = rht_operands(a, b, 32, config.rht_generator())
    expected = quantize_blocked(a_hat, 32, "e8m0", use_global_scale=False) @ quantize_blocked(
        b_hat.T, 32, "e8m0", use_global_scale=False
    ).T

    assert np.array_equal(qgemm(a, b, config), expected)


def test_quantization_blocks_b_along_the_contraction_axis():
    # Same axis question as the RHT one, for step 2: `quantize_blocked` blocks along
    # the last axis, so B must be transposed before it is quantized, not after.
    from qgemm.blocks import quantize_blocked

    a, b = _operands(24, 64, 64)
    right = quantize_blocked(b.T, 32, "e8m0", use_global_scale=False).T
    wrong = quantize_blocked(b, 32, "e8m0", use_global_scale=False)

    out = qgemm(a, b, MXFP4)

    assert np.array_equal(out, quantize_blocked(a, 32, "e8m0", use_global_scale=False) @ right)
    assert not np.allclose(out, quantize_blocked(a, 32, "e8m0", use_global_scale=False) @ wrong)


# --- accumulation modes --------------------------------------------------------


def test_bf16_accumulation_differs_from_exact_accumulation():
    a, b = _operands(32, 256, 32)

    exact = qgemm(a, b, GemmConfig(accum="exact"))
    bf16 = qgemm(a, b, GemmConfig(accum="bf16"))

    assert not np.array_equal(exact, bf16)


def test_bf16_accumulation_adds_error_the_exact_route_does_not_have():
    # The exact route is, by construction, the *exact* product of the quantized
    # operands, so any deviation from it is accumulation error and nothing else.
    # That makes this the sharp form of "bf16 is worse": it holds at any K,
    # unlike the comparison against the true product below.
    a, b = _operands(32, 256, 32)

    exact = qgemm(a, b, GemmConfig(accum="exact"))
    bf16 = qgemm(a, b, GemmConfig(accum="bf16"))

    # ~1.2% of the product's scale in practice; float64 noise would be ~1e-14.
    assert rms_error(bf16, exact) > 1e-3 * np.sqrt(np.mean(exact**2))


def test_bf16_accumulation_is_worse_than_exact_against_the_true_product():
    # Needs a long contraction dimension to be a property rather than a coin
    # flip. The two error sources add in quadrature, and at K=256 the FP4
    # quantization error is ~20x the bf16 accumulation error, so which route
    # lands closer to A @ B is decided by luck. Accumulation error grows with K
    # while quantization error does not, and by K=4096 the ordering is solid
    # (checked over 20 seeds of these operands; the margin runs 2-7%).
    a, b = _operands(16, 4096, 16)
    reference = a @ b

    exact = qgemm(a, b, GemmConfig(accum="exact"))
    bf16 = qgemm(a, b, GemmConfig(accum="bf16"))

    assert rms_error(bf16, reference) > rms_error(exact, reference)


def test_bf16_accumulation_without_quantization_still_rounds():
    a, b = _operands(16, 128, 16)
    config = GemmConfig(quantize=False, accum="bf16")

    out = qgemm(a, b, config)

    reference = a @ b
    assert not np.array_equal(out, reference)
    # Loose, and relative to the product's own scale rather than entry by entry:
    # individual entries are small differences of large partial sums, so their
    # relative error is unbounded. bf16 accumulation should still land within a
    # few percent of the exact product overall.
    assert rms_error(out, reference) < 0.05 * np.sqrt(np.mean(reference**2))


# --- performance ---------------------------------------------------------------


def test_exact_route_512_square_runs_in_under_a_second():
    # Load-bearing, not aspirational: the full sweep runs this route thousands of
    # times, so a slow implementation (e.g. a Python loop over blocks) makes the
    # study's main experiment impractical. Fail, don't log.
    a, b = _operands(512, 512, 512)
    config = GemmConfig(rht=True, rht_block_size=32)
    qgemm(a[:8, :32], b[:32, :8], config)  # warm up, off the clock

    start = time.perf_counter()
    qgemm(a, b, config)
    elapsed = time.perf_counter() - start

    assert elapsed < 1.0, f"512x512 exact route took {elapsed:.3f}s, budget is 1.0s"


# --- configuration and validation ----------------------------------------------


def test_config_defaults_are_mxfp4():
    config = GemmConfig()

    assert (config.block_size, config.scale_format, config.use_global_scale) == (32, "e8m0", False)
    assert config.accum == "exact"
    assert config.rht is False


def test_config_rht_block_size_defaults_to_the_quantization_block_size():
    assert GemmConfig(rht=True, block_size=16).rht_block_size == 16


def test_nvfp4_config_beats_mxfp4_config():
    a, b = _operands(32, 128, 32)
    reference = a @ b
    mxfp4 = GemmConfig(block_size=32, scale_format="e8m0", use_global_scale=False)
    nvfp4 = GemmConfig(block_size=16, scale_format="e4m3", use_global_scale=True)

    assert rms_error(qgemm(a, b, nvfp4), reference) < rms_error(qgemm(a, b, mxfp4), reference)


def test_unknown_accum_mode_is_rejected():
    with pytest.raises(ValueError, match="accum"):
        GemmConfig(accum="fp8")


def test_non_float64_input_is_rejected():
    a, b = _operands(8, 32, 8)

    with pytest.raises(TypeError, match="float64"):
        qgemm(a.astype(np.float32), b, MXFP4)


def test_shape_mismatch_is_rejected():
    a, _ = _operands(8, 32, 8)
    b = np.zeros((16, 8))

    with pytest.raises(ValueError, match="contraction"):
        qgemm(a, b, MXFP4)


def test_stochastic_rounding_uses_the_config_seed():
    a, b = _operands(16, 64, 16)
    sr = GemmConfig(round_mode="sr", seed=11)

    assert np.array_equal(qgemm(a, b, sr), qgemm(a, b, sr))
    assert not np.array_equal(qgemm(a, b, sr), qgemm(a, b, GemmConfig(round_mode="sr", seed=12)))


def test_stochastic_rounding_draws_different_noise_for_the_two_operands():
    # A and B must not share an SR stream: identical noise on both operands would
    # correlate their errors and bias the product.
    a = np.full((8, 32), 0.7)
    sr = GemmConfig(round_mode="sr", seed=5)

    out = qgemm(a, a.T.copy(), sr)

    assert not np.allclose(out, out.T)
