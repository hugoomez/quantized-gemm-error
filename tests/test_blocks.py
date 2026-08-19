import numpy as np
import pytest

from qgemm.blocks import MXFP4_BLOCK_SIZE, MXFP4_ELEM_MAX, mxfp4_block_scales, quantize_mxfp4
from qgemm.formats import E8M0_MAX, E8M0_MIN


def _scale_per_element(scales: np.ndarray, n: int, block_size: int) -> np.ndarray:
    """Broadcast per-block scales back out to one scale per element."""
    return np.repeat(scales, block_size, axis=-1)[..., :n]


# --- the analytic case from the project brief ---------------------------------


def test_block_with_amax_4p5_scales_to_one_and_reconstructs_4():
    # s_ideal = 4.5 / 6 = 0.75, log2(0.75) = -0.415, ceil -> 0, so s_b = 2**0.
    x = np.zeros(32, dtype=np.float64)
    x[0] = 4.5

    scales = mxfp4_block_scales(x, block_size=32)
    assert scales.shape == (1,)
    assert scales[0] == 1.0

    out = quantize_mxfp4(x, block_size=32)
    # 4.5 / 1.0 = 4.5 sits between grid points 4 and 6, nearer 4.
    assert out[0] == 4.0
    assert abs(out[0] - x[0]) == 0.5


# --- the scale rule: round the exponent UP ------------------------------------


def test_scale_rounds_the_exponent_up_so_the_block_max_never_saturates():
    # amax = 7: ceil(log2(7/6)) = 1 -> s_b = 2. A floor/round-down rule would
    # give s_b = 1 and then 7/1 = 7 > 6 would saturate.
    x = np.zeros(32, dtype=np.float64)
    x[0] = 7.0

    assert mxfp4_block_scales(x)[0] == 2.0
    # 7/2 = 3.5 is a tie between 3 (code 101) and 4 (code 110); RTNE takes 4.
    assert quantize_mxfp4(x)[0] == 8.0


def test_scale_rounds_up_only_when_needed():
    # amax exactly on a grid point: 6 / 6 = 1, ceil(log2(1)) = 0, no round-up.
    x = np.zeros(32, dtype=np.float64)
    x[0] = 6.0
    assert mxfp4_block_scales(x)[0] == 1.0
    assert quantize_mxfp4(x)[0] == 6.0


def test_scales_are_always_powers_of_two():
    rng = np.random.default_rng(20260819)
    x = rng.standard_normal((5, 128)) * rng.lognormal(0.0, 6.0, (5, 128))
    scales = mxfp4_block_scales(x, block_size=32)
    np.testing.assert_array_equal(np.log2(scales), np.rint(np.log2(scales)))


def test_no_reconstructed_value_exceeds_six_times_the_block_scale():
    rng = np.random.default_rng(11)
    # Wildly different magnitudes per block, so the scales really do vary.
    x = rng.standard_normal((7, 96)) * rng.lognormal(0.0, 8.0, (7, 96))
    out = quantize_mxfp4(x, block_size=32)
    limit = MXFP4_ELEM_MAX * _scale_per_element(mxfp4_block_scales(x, block_size=32), 96, 32)
    assert np.all(np.abs(out) <= limit)


def test_the_block_maximum_itself_never_saturates():
    rng = np.random.default_rng(12)
    x = rng.standard_normal((40, 32)) * rng.lognormal(0.0, 5.0, (40, 1))
    scales = mxfp4_block_scales(x, block_size=32)
    amax = np.abs(x).max(axis=-1)
    # The defining property of the round-up rule.
    assert np.all(amax / scales[..., 0] <= MXFP4_ELEM_MAX)


# --- degenerate blocks --------------------------------------------------------


def test_all_zero_block_has_unit_scale_and_produces_no_nan_or_inf():
    x = np.zeros((3, 64), dtype=np.float64)
    np.testing.assert_array_equal(mxfp4_block_scales(x, block_size=32), np.ones((3, 2)))
    out = quantize_mxfp4(x, block_size=32)
    assert np.all(np.isfinite(out))
    np.testing.assert_array_equal(out, x)


def test_zero_block_next_to_a_nonzero_block_is_handled_independently():
    x = np.zeros((1, 64), dtype=np.float64)
    x[0, 32] = 4.5
    scales = mxfp4_block_scales(x, block_size=32)
    np.testing.assert_array_equal(scales, np.array([[1.0, 1.0]]))
    assert np.all(np.isfinite(quantize_mxfp4(x, block_size=32)))


def test_signed_zero_is_preserved():
    x = np.zeros(32, dtype=np.float64)
    x[0] = -0.0
    x[1] = 4.5
    out = quantize_mxfp4(x)
    assert np.signbit(out[0])


# --- axis policy: blocks run along the LAST axis ------------------------------


def test_blocks_run_along_the_last_axis_not_the_first():
    # Row 0 is 2**-20 times row 1. Blocked along the last axis each row gets
    # its own scale and both survive; blocked along axis 0 the columns would
    # share row 1's scale and row 0 would flush to zero.
    x = np.empty((2, 32), dtype=np.float64)
    x[0] = 4.5 * 2.0**-20
    x[1] = 4.5

    scales = mxfp4_block_scales(x, block_size=32)
    assert scales.shape == (2, 1)
    assert scales[0, 0] == 2.0**-20
    assert scales[1, 0] == 1.0

    out = quantize_mxfp4(x, block_size=32)
    assert np.all(out[0] == 4.0 * 2.0**-20)
    assert np.all(out[1] == 4.0)


def test_leading_axes_are_untouched_and_shape_is_preserved():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((2, 3, 64))
    assert quantize_mxfp4(x, block_size=32).shape == (2, 3, 64)
    assert mxfp4_block_scales(x, block_size=32).shape == (2, 3, 2)
    # Each (i, j) row is independent: quantizing it alone gives the same answer.
    np.testing.assert_array_equal(
        quantize_mxfp4(x, block_size=32)[1, 2], quantize_mxfp4(x[1, 2], block_size=32)
    )


def test_one_dimensional_input_is_a_single_row():
    rng = np.random.default_rng(4)
    x = rng.standard_normal(64)
    assert quantize_mxfp4(x, block_size=32).shape == (64,)
    assert mxfp4_block_scales(x, block_size=32).shape == (2,)


# --- tail policy: a shorter final block ---------------------------------------


def test_final_partial_block_gets_its_own_scale():
    x = np.zeros(40, dtype=np.float64)
    x[:32] = 4.5
    x[32:] = 4.5 * 2.0**-20

    scales = mxfp4_block_scales(x, block_size=32)
    assert scales.shape == (2,)
    assert scales[0] == 1.0
    assert scales[1] == 2.0**-20

    out = quantize_mxfp4(x, block_size=32)
    assert np.all(out[32:] == 4.0 * 2.0**-20)


def test_short_tail_is_identical_to_zero_padding_the_tail():
    # Documented equivalence: amax is invariant to appended zeros, so a ragged
    # final block and a zero-padded one produce the same scale and the same
    # reconstruction. The ragged form just avoids allocating and trimming.
    rng = np.random.default_rng(5)
    x = rng.standard_normal(70) * 10.0
    padded = np.concatenate([x, np.zeros(32 - 70 % 32)])
    np.testing.assert_array_equal(
        quantize_mxfp4(x, block_size=32), quantize_mxfp4(padded, block_size=32)[:70]
    )


def test_block_shorter_than_block_size_is_allowed():
    x = np.array([4.5, 1.0, -2.0], dtype=np.float64)
    assert mxfp4_block_scales(x, block_size=32).shape == (1,)
    np.testing.assert_array_equal(quantize_mxfp4(x, block_size=32), np.array([4.0, 1.0, -2.0]))


# --- scale range (E8M0) -------------------------------------------------------


def test_scale_is_clamped_to_the_e8m0_representable_range_above():
    x = np.zeros(32, dtype=np.float64)
    x[0] = 1e300  # would want 2**995, which no E8M0 scale can express
    assert mxfp4_block_scales(x)[0] == E8M0_MAX
    # With the largest available scale the block max saturates to 6 * s_b.
    assert quantize_mxfp4(x)[0] == MXFP4_ELEM_MAX * E8M0_MAX


def test_scale_is_clamped_to_the_e8m0_representable_range_below():
    x = np.zeros(32, dtype=np.float64)
    x[0] = 1e-300
    assert mxfp4_block_scales(x)[0] == E8M0_MIN
    assert quantize_mxfp4(x)[0] == 0.0


# --- rounding modes -----------------------------------------------------------


def test_rtne_is_the_default_and_is_deterministic():
    rng = np.random.default_rng(6)
    x = rng.standard_normal((4, 32))
    np.testing.assert_array_equal(quantize_mxfp4(x), quantize_mxfp4(x, round_mode="rtne"))


def test_stochastic_rounding_requires_an_explicit_generator():
    x = np.zeros(32, dtype=np.float64)
    with pytest.raises(ValueError, match="Generator"):
        quantize_mxfp4(x, round_mode="sr")


def test_stochastic_rounding_is_unbiased_within_a_block():
    rng = np.random.default_rng(20260819)
    x = np.empty((20_000, 32), dtype=np.float64)
    x[:, 0] = 4.0  # pins s_b = 1 for every row
    x[:, 1:] = 1.2  # lands 40% of the way from grid point 1.0 to 1.5

    out = quantize_mxfp4(x, block_size=32, rng=rng, round_mode="sr")

    assert np.all(mxfp4_block_scales(x, block_size=32) == 1.0)
    assert set(np.unique(out[:, 1:])) == {1.0, 1.5}
    assert abs(out[:, 1:].mean() - 1.2) < 5e-3


def test_unknown_round_mode_is_rejected():
    with pytest.raises(ValueError, match="mode"):
        quantize_mxfp4(np.zeros(32), round_mode="floor")


# --- input contract -----------------------------------------------------------


def test_non_float64_input_is_rejected():
    with pytest.raises(TypeError, match="float64"):
        quantize_mxfp4(np.zeros(32, dtype=np.float32))
    with pytest.raises(TypeError, match="float64"):
        mxfp4_block_scales(np.zeros(32, dtype=np.float32))


def test_non_finite_input_is_rejected():
    x = np.zeros(32, dtype=np.float64)
    x[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        quantize_mxfp4(x)
    x[0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        quantize_mxfp4(x)


def test_non_positive_block_size_is_rejected():
    with pytest.raises(ValueError, match="block_size"):
        quantize_mxfp4(np.zeros(32), block_size=0)


def test_default_block_size_is_the_mx_standard_32():
    assert MXFP4_BLOCK_SIZE == 32
    rng = np.random.default_rng(8)
    x = rng.standard_normal(64)
    np.testing.assert_array_equal(quantize_mxfp4(x), quantize_mxfp4(x, block_size=32))


# --- golden reference: Microsoft microxcaling ---------------------------------


def _microxcaling_mxfp4(x: np.ndarray, block_size: int) -> np.ndarray:
    """Run microxcaling's MXFP4 quantizer, or skip if it isn't usable here."""
    mx_ops = pytest.importorskip("mx.mx_ops", reason="microxcaling is not installed")
    torch = pytest.importorskip("torch", reason="microxcaling needs torch")
    specs_mod = pytest.importorskip("mx.specs", reason="microxcaling is not installed")

    quantize_mx_op = getattr(mx_ops, "quantize_mx_op", None)
    finalize = getattr(specs_mod, "finalize_mx_specs", None)
    if quantize_mx_op is None or finalize is None:
        pytest.skip("installed microxcaling does not expose the expected entry points")

    specs = finalize(
        {
            "block_size": block_size,
            "scale_bits": 8,
            "shared_exp_method": "max",
            "mx_flush_fp32_subnorms": False,
            "custom_cuda": False,
            "w_elem_format": "fp4_e2m1",
            "a_elem_format": "fp4_e2m1",
        },
        early_exit=False,
    )
    try:
        out = quantize_mx_op(
            torch.tensor(x, dtype=torch.float32), specs, "fp4_e2m1", axes=[-1], round="nearest"
        )
    except (TypeError, KeyError, RuntimeError) as exc:  # pragma: no cover - version drift
        pytest.skip(f"installed microxcaling has an incompatible API: {exc}")
    return np.asarray(out.detach().cpu().numpy(), dtype=np.float64)


def _floor_rule_scale(x: np.ndarray, block_size: int) -> np.ndarray:
    """The OCP-MX / microxcaling scale: 2**(floor(log2(amax)) - emax_elem)."""
    n_blocks = -(-x.shape[-1] // block_size)
    pad = n_blocks * block_size - x.shape[-1]
    padded = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, pad)])
    amax = np.abs(padded.reshape(*x.shape[:-1], n_blocks, block_size)).max(axis=-1)
    with np.errstate(divide="ignore"):
        return np.where(amax == 0.0, 1.0, 2.0 ** (np.floor(np.log2(amax)) - 2.0))


def test_matches_microxcaling_where_the_two_scale_rules_agree():
    # The scale rules coincide whenever amax / 2**floor(log2(amax)) <= 1.5,
    # i.e. whenever amax's own mantissa already fits under 6 = 1.5 * 2**2.
    rng = np.random.default_rng(20260819)
    x = (rng.standard_normal((16, 32)) * rng.lognormal(0.0, 3.0, (16, 1))).astype(np.float64)

    theirs = _microxcaling_mxfp4(x, 32)
    ours = quantize_mxfp4(x, block_size=32)
    agree = (mxfp4_block_scales(x, block_size=32) == _floor_rule_scale(x, 32))[..., 0]

    assert agree.any(), "test tensor exercises none of the agreeing blocks"
    np.testing.assert_array_equal(ours[agree], theirs[agree])


def test_microxcaling_saturates_the_block_max_where_we_round_the_scale_up():
    # The documented divergence. amax = 7 needs a scale of 2 to stay under 6;
    # the floor rule picks 1, so microxcaling clips 7 down to exactly 6.
    x = np.zeros((1, 32), dtype=np.float64)
    x[0, 0] = 7.0

    theirs = _microxcaling_mxfp4(x, 32)
    assert _floor_rule_scale(x, 32)[0, 0] == 1.0
    assert theirs[0, 0] == 6.0

    ours = quantize_mxfp4(x, block_size=32)
    assert mxfp4_block_scales(x, block_size=32)[0, 0] == 2.0
    assert ours[0, 0] == 8.0
