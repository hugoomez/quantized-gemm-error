import numpy as np
import pytest

from qgemm.blocks import (
    MXFP4_BLOCK_SIZE,
    MXFP4_ELEM_MAX,
    NVFP4_BLOCK_SIZE,
    NVFP4_SCALE_MAX,
    block_scales,
    global_scale,
    mxfp4_block_scales,
    nvfp4_block_scales,
    nvfp4_global_scale,
    quantize_blocked,
    quantize_mxfp4,
    quantize_nvfp4,
)
from qgemm.formats import E8M0_MAX, E8M0_MIN, quantize_e2m1, quantize_e4m3


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


def _microxcaling_mxfp4(x: np.ndarray, block_size: int, round: str = "even") -> np.ndarray:
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
        # round="even" is microxcaling's RTNE and the mode that corresponds to
        # quantize_e2m1(mode="rtne"). Its round="nearest" is a *different* rule
        # -- floor(|A| + 0.5), i.e. ties away from zero -- which disagrees with
        # us and with ml_dtypes.float4_e2m1fn at every E2M1 midpoint. Random
        # data never lands exactly on a midpoint, so picking the wrong mode
        # here would leave every test below passing for the wrong reason;
        # test_microxcaling_round_modes_at_the_e2m1_midpoints pins the choice.
        out = quantize_mx_op(
            torch.tensor(x, dtype=torch.float32), specs, "fp4_e2m1", axes=[-1], round=round
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


def test_microxcaling_round_modes_at_the_e2m1_midpoints():
    # Pins which microxcaling round mode is the right reference. "even" is RTNE
    # and agrees with us at all seven E2M1 midpoints; "nearest" is ties-away-
    # from-zero and disagrees at four of them. Random data never lands on a
    # midpoint, so this is the only test that can tell the two modes apart.
    x = np.zeros((1, 32), dtype=np.float64)
    x[0, :7] = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]
    x[0, 7] = 6.0  # pins s_b = 1 under both scale rules

    ours = quantize_mxfp4(x, block_size=32)[0, :7]
    np.testing.assert_array_equal(ours, [0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0])
    np.testing.assert_array_equal(_microxcaling_mxfp4(x, 32, round="even")[0, :7], ours)
    np.testing.assert_array_equal(
        _microxcaling_mxfp4(x, 32, round="nearest")[0, :7],
        [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
    )


def test_the_scale_rule_is_the_only_difference_from_microxcaling():
    # The strong form of the comparison: swap our ceil scale for their floor
    # scale and the two implementations agree bit for bit on every block,
    # including the ones where the scale rules disagree. So the divergence is
    # entirely the scale exponent -- there is no second, hidden difference in
    # the element path, the tie handling or the block reduction.
    rng = np.random.default_rng(20260820)
    # float32-exact input, so microxcaling's float32 arithmetic adds no noise.
    x = (rng.standard_normal((512, 32)) * rng.lognormal(0.0, 4.0, (512, 1))).astype(np.float32)
    x = x.astype(np.float64)

    floor_scale = np.repeat(_floor_rule_scale(x, 32), 32, axis=-1)
    ours_with_their_scale = quantize_e2m1(x / floor_scale, mode="rtne") * floor_scale

    differ = (mxfp4_block_scales(x, block_size=32) != _floor_rule_scale(x, 32))[..., 0]
    assert differ.any(), "test tensor exercises none of the diverging blocks"
    np.testing.assert_array_equal(ours_with_their_scale, _microxcaling_mxfp4(x, 32))


# =============================================================================
# NVFP4: E2M1 elements, E4M3 block scale, one float64 global scale
# =============================================================================


def _nvfp4_scale_per_element(x: np.ndarray, block_size: int) -> np.ndarray:
    """The effective per-element scale `s_b * s_global`, one entry per element."""
    scales = nvfp4_block_scales(x, block_size=block_size) * nvfp4_global_scale(x)
    return _scale_per_element(scales, x.shape[-1], block_size)


# --- the analytic case, in direct contrast with MXFP4 -------------------------


def test_nvfp4_reconstructs_amax_4p5_exactly():
    # s_global = 4.5 / (6 * 448); s_b = quantize_e4m3(448) = 448, so the
    # effective scale is exactly 0.75 -- and 4.5 / 0.75 = 6 is the top E2M1
    # grid point. Nothing is rounded anywhere, so this is exact, not close.
    x = np.zeros(16, dtype=np.float64)
    x[0] = 4.5

    assert nvfp4_global_scale(x) == 4.5 / (6.0 * 448.0)
    assert nvfp4_block_scales(x, block_size=16)[0] == 448.0

    out = quantize_nvfp4(x, block_size=16)
    assert out[0] == 4.5
    assert out[0] - x[0] == 0.0


def test_nvfp4_reconstructs_4p5_exactly_where_mxfp4_rounds_it_to_4():
    # The core claim in miniature, at equal block size so that the only
    # difference is the scale format: a power-of-two scale (E8M0) forces MXFP4
    # onto s_b = 1 and 4.5 falls to the grid point 4; E4M3 expresses 0.75.
    x = np.zeros(16, dtype=np.float64)
    x[0] = 4.5

    assert quantize_mxfp4(x, block_size=16)[0] == 4.0
    assert quantize_nvfp4(x, block_size=16)[0] == 4.5


# --- the global scale ---------------------------------------------------------


def test_global_scale_is_the_tensor_amax_over_six_times_448():
    rng = np.random.default_rng(20260820)
    x = rng.standard_normal((4, 32)) * 17.0
    assert nvfp4_global_scale(x) == np.abs(x).max() / (MXFP4_ELEM_MAX * NVFP4_SCALE_MAX)


def test_the_global_scale_is_per_tensor_so_rows_are_not_independent():
    # Unlike MXFP4, a row's reconstruction depends on the rest of the tensor:
    # the global scale is a single per-tensor number. Documented, not a bug.
    row = np.zeros(16, dtype=np.float64)
    row[0] = 4.5
    together = np.stack([row, row * 100.0])

    assert quantize_nvfp4(row, block_size=16)[0] == 4.5
    embedded = quantize_nvfp4(together, block_size=16)[0, 0]
    assert embedded != 4.5
    assert abs(embedded - 4.5) < 0.05  # still a small effect, not a collapse


# --- the trap: block scales must stay inside E4M3 ------------------------------


def test_no_block_scale_ever_exceeds_the_e4m3_maximum():
    rng = np.random.default_rng(20260820)
    # Magnitudes spanning many binades, and far above 6 * 448 in absolute terms:
    # without the global scale, amax_b / 6 would land outside E4M3 entirely.
    x = rng.standard_normal((9, 64)) * rng.lognormal(0.0, 8.0, (9, 64)) * 1e6
    scales = nvfp4_block_scales(x, block_size=16)

    assert np.all(np.isfinite(scales))
    assert np.all(scales <= NVFP4_SCALE_MAX)
    assert np.all(quantize_e4m3(scales) == scales)  # every scale is an E4M3 value


def test_large_magnitude_tensor_does_not_saturate_its_block_scales():
    # The specific failure mode of dropping the global scale: with block scales
    # taken straight from the data, all four of these blocks pin at the top of
    # E4M3 and stop being scaled relative to one another at all.
    x = np.full((4, 16), 1e12, dtype=np.float64)
    x[1] *= 2.0**-10

    scales = nvfp4_block_scales(x, block_size=16)
    assert np.all(np.isfinite(scales))
    # The block holding the tensor maximum sits exactly at the top of E4M3 --
    # that is what the global scale is chosen to arrange -- and the smaller
    # block sits 2**-10 below it, with room to spare.
    assert scales[0, 0] == NVFP4_SCALE_MAX
    assert scales[1, 0] == NVFP4_SCALE_MAX * 2.0**-10
    assert np.all(np.isfinite(quantize_nvfp4(x, block_size=16)))


def test_nvfp4_is_more_accurate_than_mxfp4_at_the_same_block_size():
    # The trap's symptom, asserted directly: if this ever inverts, suspect a
    # missing or wrong global scale before suspecting the element numerics.
    rng = np.random.default_rng(20260820)
    x = rng.standard_normal((32, 64)) * rng.lognormal(0.0, 2.0, (32, 1)) * 1e6

    err_nv = np.sqrt(np.mean((quantize_nvfp4(x, block_size=16) - x) ** 2))
    err_mx = np.sqrt(np.mean((quantize_mxfp4(x, block_size=16) - x) ** 2))
    assert err_nv < err_mx


# --- degenerate blocks and tensors --------------------------------------------


def test_all_zero_tensor_reconstructs_to_zero_without_nan():
    x = np.zeros((3, 32), dtype=np.float64)
    assert nvfp4_global_scale(x) == 1.0  # no 0/0
    out = quantize_nvfp4(x, block_size=16)
    assert np.all(np.isfinite(out))
    np.testing.assert_array_equal(out, x)


def test_zero_block_next_to_a_nonzero_block_reconstructs_to_zero():
    x = np.zeros((1, 32), dtype=np.float64)
    x[0, 16] = 4.5
    scales = nvfp4_block_scales(x, block_size=16)
    assert scales[0, 0] == 0.0  # E4M3 has a zero, unlike E8M0
    out = quantize_nvfp4(x, block_size=16)
    assert np.all(np.isfinite(out))
    np.testing.assert_array_equal(out[0, :16], np.zeros(16))


def test_block_far_below_the_tensor_max_flushes_to_zero_without_nan():
    # s_b = quantize_e4m3(448 * amax_b / amax) underflows E4M3 once the ratio
    # drops far enough, and the block flushes rather than producing NaN.
    x = np.zeros((2, 16), dtype=np.float64)
    x[0] = 1.0
    x[1] = 1e-9

    out = quantize_nvfp4(x, block_size=16)
    assert np.all(np.isfinite(out))
    assert nvfp4_block_scales(x, block_size=16)[1, 0] == 0.0
    np.testing.assert_array_equal(out[1], np.zeros(16))


def test_extremely_small_tensor_produces_no_nan():
    # The global scale itself underflows float64 here; it falls back to 1.0.
    x = np.full(16, 5e-324, dtype=np.float64)
    assert nvfp4_global_scale(x) == 1.0
    assert np.all(np.isfinite(quantize_nvfp4(x, block_size=16)))


def test_subnormal_global_scale_does_not_overflow_the_block_scale_to_nan():
    # At the very bottom of float64 the global scale is itself subnormal and so
    # carries a large relative rounding error: amax_b / s_global / 6 comes out
    # at 472 here, and E4M3 overflows to NaN rather than saturating.
    x = np.full(16, 1.4e-320, dtype=np.float64)
    scales = nvfp4_block_scales(x, block_size=16)
    assert np.all(np.isfinite(scales))
    assert scales[0] == NVFP4_SCALE_MAX
    assert np.all(np.isfinite(quantize_nvfp4(x, block_size=16)))


def test_nvfp4_signed_zero_is_preserved():
    x = np.zeros(16, dtype=np.float64)
    x[0] = -0.0
    x[1] = 4.5
    assert np.signbit(quantize_nvfp4(x, block_size=16)[0])


# --- axis, shape and tail policy (same conventions as MXFP4) ------------------


def test_nvfp4_blocks_run_along_the_last_axis_not_the_first():
    x = np.empty((2, 16), dtype=np.float64)
    x[0] = 4.5 * 2.0**-8
    x[1] = 4.5

    scales = nvfp4_block_scales(x, block_size=16)
    assert scales.shape == (2, 1)
    # 448 * 2**-8 = 1.75 is an exact E4M3 value, so the small row survives.
    assert scales[0, 0] == 448.0 * 2.0**-8
    assert scales[1, 0] == 448.0

    out = quantize_nvfp4(x, block_size=16)
    assert np.all(out[0] == 4.5 * 2.0**-8)
    assert np.all(out[1] == 4.5)


def test_nvfp4_leading_axes_are_untouched_and_shape_is_preserved():
    rng = np.random.default_rng(21)
    x = rng.standard_normal((2, 3, 32))
    assert quantize_nvfp4(x, block_size=16).shape == (2, 3, 32)
    assert nvfp4_block_scales(x, block_size=16).shape == (2, 3, 2)


def test_nvfp4_one_dimensional_input_is_a_single_row():
    rng = np.random.default_rng(22)
    x = rng.standard_normal(32)
    assert quantize_nvfp4(x, block_size=16).shape == (32,)
    assert nvfp4_block_scales(x, block_size=16).shape == (2,)


def test_nvfp4_final_partial_block_gets_its_own_scale():
    x = np.zeros(20, dtype=np.float64)
    x[:16] = 4.5
    x[16:] = 4.5 * 2.0**-8

    scales = nvfp4_block_scales(x, block_size=16)
    assert scales.shape == (2,)
    assert scales[0] == 448.0
    assert scales[1] == 448.0 * 2.0**-8
    assert np.all(quantize_nvfp4(x, block_size=16)[16:] == 4.5 * 2.0**-8)


def test_nvfp4_short_tail_is_identical_to_zero_padding_the_tail():
    rng = np.random.default_rng(23)
    x = rng.standard_normal(38) * 10.0
    padded = np.concatenate([x, np.zeros(16 - 38 % 16)])
    np.testing.assert_array_equal(
        quantize_nvfp4(x, block_size=16), quantize_nvfp4(padded, block_size=16)[:38]
    )


def test_nvfp4_block_shorter_than_block_size_is_allowed():
    x = np.array([4.5, 1.5, -3.0], dtype=np.float64)
    assert nvfp4_block_scales(x, block_size=16).shape == (1,)
    np.testing.assert_array_equal(quantize_nvfp4(x, block_size=16), x)


def test_nvfp4_default_block_size_is_16():
    assert NVFP4_BLOCK_SIZE == 16
    rng = np.random.default_rng(24)
    x = rng.standard_normal(32)
    np.testing.assert_array_equal(quantize_nvfp4(x), quantize_nvfp4(x, block_size=16))


# --- rounding modes -----------------------------------------------------------


def test_nvfp4_rtne_is_the_default_and_is_deterministic():
    rng = np.random.default_rng(25)
    x = rng.standard_normal((4, 16))
    np.testing.assert_array_equal(quantize_nvfp4(x), quantize_nvfp4(x, round_mode="rtne"))


def test_nvfp4_stochastic_rounding_requires_an_explicit_generator():
    x = np.zeros(16, dtype=np.float64)
    with pytest.raises(ValueError, match="Generator"):
        quantize_nvfp4(x, round_mode="sr")


def test_nvfp4_stochastic_rounding_is_unbiased_within_a_block():
    rng = np.random.default_rng(20260820)
    x = np.empty((20_000, 16), dtype=np.float64)
    x[:, 0] = 6.0  # tensor amax 6 -> s_b * s_global == 1.0 exactly for every block
    x[:, 1:] = 1.2  # 40% of the way from grid point 1.0 to 1.5

    out = quantize_nvfp4(x, block_size=16, rng=rng, round_mode="sr")

    np.testing.assert_array_equal(_nvfp4_scale_per_element(x, 16), np.ones_like(x))
    assert set(np.unique(out[:, 1:])) == {1.0, 1.5}
    assert abs(out[:, 1:].mean() - 1.2) < 5e-3


def test_nvfp4_unknown_round_mode_is_rejected():
    with pytest.raises(ValueError, match="mode"):
        quantize_nvfp4(np.zeros(16), round_mode="floor")


# --- input contract -----------------------------------------------------------


def test_nvfp4_non_float64_input_is_rejected():
    with pytest.raises(TypeError, match="float64"):
        quantize_nvfp4(np.zeros(16, dtype=np.float32))
    with pytest.raises(TypeError, match="float64"):
        nvfp4_block_scales(np.zeros(16, dtype=np.float32))


def test_nvfp4_non_finite_input_is_rejected():
    x = np.zeros(16, dtype=np.float64)
    x[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        quantize_nvfp4(x)
    x[0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        quantize_nvfp4(x)


def test_nvfp4_non_positive_block_size_is_rejected():
    with pytest.raises(ValueError, match="block_size"):
        quantize_nvfp4(np.zeros(16), block_size=0)


# --- golden reference: torchao (sanity check only, see SPEC.md) ---------------


def _torchao_nvfp4(x: np.ndarray, block_size: int) -> np.ndarray:
    """Run torchao's NVFP4 quantizer, or skip if it is not usable here."""
    torch = pytest.importorskip("torch", reason="torchao needs torch")
    nvfp4_mod = pytest.importorskip(
        "torchao.prototype.mx_formats.nvfp4_tensor", reason="torchao is not installed"
    )

    tensor_cls = getattr(nvfp4_mod, "NVFP4Tensor", None)
    if tensor_cls is None or not hasattr(tensor_cls, "to_nvfp4"):
        pytest.skip("installed torchao does not expose NVFP4Tensor.to_nvfp4")

    try:
        quantized = tensor_cls.to_nvfp4(torch.tensor(x, dtype=torch.float32), block_size=block_size)
        out = quantized.to_dtype(torch.float32)
    except (TypeError, KeyError, RuntimeError, AssertionError) as exc:  # pragma: no cover
        pytest.skip(f"installed torchao has an incompatible NVFP4 API: {exc}")
    return np.asarray(out.detach().cpu().numpy(), dtype=np.float64)


def test_matches_torchao_nvfp4_on_a_well_conditioned_tensor():
    # Sanity check, not authoritative: torchao NVFP4 is a prototype and computes
    # the global scale in float32. Restricted to a tensor whose values are
    # float32-exact and whose scales land on E4M3 grid points exactly.
    x = np.zeros((4, 16), dtype=np.float64)
    x[:, 0] = 4.5
    x[:, 1] = 2.25
    x[:, 2] = -1.125

    theirs = _torchao_nvfp4(x, 16)
    ours = quantize_nvfp4(x, block_size=16)
    np.testing.assert_allclose(ours, theirs, rtol=1e-6, atol=0.0)


# =============================================================================
# quantize_blocked: the parametrization both presets are built on
# =============================================================================

# The two axes the study varies. Only (32, "e8m0") and (16, "e4m3") are real
# hardware formats; the other six are experimental controls that exist to
# separate the effect of block size from the effect of scale format.
_BLOCK_SIZES = (8, 16, 32, 64)
_SCALE_FORMATS = ("e8m0", "e4m3")
# Each scale format is paired with the global-scale setting its own range calls
# for: E8M0 spans 255 binades and needs none, E4M3 spans about 19 and would
# otherwise overflow to NaN.
_USES_GLOBAL_SCALE = {"e8m0": False, "e4m3": True}


def _spread_tensor(rows: int = 4, cols: int = 64, seed: int = 20260820) -> np.ndarray:
    """Random float64 with enough per-block dynamic range to separate configs."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((rows, cols)) * rng.lognormal(0.0, 2.0, (rows, cols))


# --- the presets are the general function, not a second implementation --------


def test_mxfp4_is_quantize_blocked_with_the_mx_parameters():
    x = _spread_tensor()
    np.testing.assert_array_equal(
        quantize_mxfp4(x, block_size=32),
        quantize_blocked(x, block_size=32, scale_format="e8m0", use_global_scale=False),
    )


def test_nvfp4_is_quantize_blocked_with_the_nvidia_parameters():
    x = _spread_tensor()
    np.testing.assert_array_equal(
        quantize_nvfp4(x, block_size=16),
        quantize_blocked(x, block_size=16, scale_format="e4m3", use_global_scale=True),
    )


def test_preset_defaults_are_the_real_format_parameters():
    x = _spread_tensor()
    np.testing.assert_array_equal(
        quantize_mxfp4(x),
        quantize_blocked(
            x, block_size=MXFP4_BLOCK_SIZE, scale_format="e8m0", use_global_scale=False
        ),
    )
    np.testing.assert_array_equal(
        quantize_nvfp4(x),
        quantize_blocked(
            x, block_size=NVFP4_BLOCK_SIZE, scale_format="e4m3", use_global_scale=True
        ),
    )


def test_preset_scale_helpers_are_the_general_scale_helpers():
    x = _spread_tensor()
    np.testing.assert_array_equal(
        mxfp4_block_scales(x, block_size=32),
        block_scales(x, block_size=32, scale_format="e8m0", use_global_scale=False),
    )
    np.testing.assert_array_equal(
        nvfp4_block_scales(x, block_size=16),
        block_scales(x, block_size=16, scale_format="e4m3", use_global_scale=True),
    )
    assert nvfp4_global_scale(x) == global_scale(x, scale_format="e4m3")


# --- the full grid: 4 block sizes x 2 scale formats ---------------------------


@pytest.mark.parametrize("block_size", _BLOCK_SIZES)
@pytest.mark.parametrize("scale_format", _SCALE_FORMATS)
def test_every_combination_runs_and_returns_finite_output(block_size, scale_format):
    x = _spread_tensor()
    out = quantize_blocked(
        x,
        block_size=block_size,
        scale_format=scale_format,
        use_global_scale=_USES_GLOBAL_SCALE[scale_format],
    )
    assert out.shape == x.shape
    assert out.dtype == np.float64
    assert np.isfinite(out).all()


def test_every_combination_is_numerically_distinct():
    # A configuration that silently collapses onto another one measures
    # nothing, and would make the block-size/scale-format contrast in the
    # results table a fiction. If this ever fires, find out why -- do not
    # weaken the assertion.
    x = _spread_tensor()
    outs = {
        (block_size, scale_format): quantize_blocked(
            x,
            block_size=block_size,
            scale_format=scale_format,
            use_global_scale=_USES_GLOBAL_SCALE[scale_format],
        )
        for block_size in _BLOCK_SIZES
        for scale_format in _SCALE_FORMATS
    }

    keys = list(outs)
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            assert not np.array_equal(outs[a], outs[b]), f"{a} and {b} produce identical output"


def test_the_global_scale_flag_is_an_independent_axis():
    # use_global_scale is a free parameter, not a synonym for the scale format:
    # turning it on under E8M0 is supported and changes the result, because it
    # makes the effective scale a non-power-of-two, which E8M0 alone never is.
    x = _spread_tensor()
    without = quantize_blocked(x, block_size=32, scale_format="e8m0", use_global_scale=False)
    with_global = quantize_blocked(x, block_size=32, scale_format="e8m0", use_global_scale=True)

    assert np.isfinite(with_global).all()
    assert not np.array_equal(without, with_global)


# --- the E4M3 overflow trap, carried into the general function ----------------


@pytest.mark.parametrize("block_size", _BLOCK_SIZES)
def test_e4m3_block_scales_stay_in_range_at_every_block_size(block_size):
    # The reason the global scale exists: a raw block scale is amax_b / 6, and
    # E4M3 has no saturation -- one code above 448 is NaN. This is the NVFP4
    # trap, and it must not be lost for the control block sizes.
    x = _spread_tensor() * 1e6
    scales = block_scales(x, block_size=block_size, scale_format="e4m3", use_global_scale=True)

    assert np.isfinite(scales).all()
    assert (scales <= NVFP4_SCALE_MAX).all()
    assert np.isfinite(
        quantize_blocked(x, block_size=block_size, scale_format="e4m3", use_global_scale=True)
    ).all()


def test_e4m3_without_a_global_scale_clamps_rather_than_going_nan():
    # The same tensor with the global scale switched off: every block scale
    # wants to be far above 448. Clamping keeps the output finite, but the
    # block maxima saturate badly -- which is exactly why NVFP4 has a global
    # scale, and why this control is not a proposal for a real format.
    x = _spread_tensor() * 1e6
    scales = block_scales(x, block_size=16, scale_format="e4m3", use_global_scale=False)
    out = quantize_blocked(x, block_size=16, scale_format="e4m3", use_global_scale=False)
    with_global = quantize_blocked(x, block_size=16, scale_format="e4m3", use_global_scale=True)

    assert np.isfinite(scales).all()
    assert (scales == NVFP4_SCALE_MAX).any()
    assert np.isfinite(out).all()
    assert np.abs(out - x).max() > np.abs(with_global - x).max()


# --- parameter validation and pass-through ------------------------------------


def test_unknown_scale_format_is_rejected():
    with pytest.raises(ValueError, match="scale_format"):
        quantize_blocked(np.zeros(32), block_size=32, scale_format="e5m2", use_global_scale=False)


def test_unsupported_element_format_is_rejected():
    with pytest.raises(ValueError, match="element_format"):
        quantize_blocked(
            np.zeros(32),
            block_size=32,
            scale_format="e8m0",
            use_global_scale=False,
            element_format="e4m3",
        )


def test_quantize_blocked_enforces_the_shared_input_contract():
    with pytest.raises(TypeError, match="float64"):
        quantize_blocked(
            np.zeros(32, dtype=np.float32),
            block_size=32,
            scale_format="e8m0",
            use_global_scale=False,
        )
    with pytest.raises(ValueError, match="finite"):
        quantize_blocked(
            np.array([np.nan] * 32), block_size=32, scale_format="e8m0", use_global_scale=False
        )
    with pytest.raises(ValueError, match="block_size"):
        quantize_blocked(np.zeros(32), block_size=0, scale_format="e8m0", use_global_scale=False)


@pytest.mark.parametrize("scale_format", _SCALE_FORMATS)
def test_quantize_blocked_round_mode_reaches_the_elements(scale_format):
    x = _spread_tensor()
    kwargs = {
        "block_size": 16,
        "scale_format": scale_format,
        "use_global_scale": _USES_GLOBAL_SCALE[scale_format],
    }

    np.testing.assert_array_equal(
        quantize_blocked(x, **kwargs), quantize_blocked(x, **kwargs, round_mode="rtne")
    )
    with pytest.raises(ValueError, match="rng"):
        quantize_blocked(x, **kwargs, round_mode="sr")

    stochastic = quantize_blocked(x, **kwargs, round_mode="sr", rng=np.random.default_rng(0))
    assert not np.array_equal(stochastic, quantize_blocked(x, **kwargs))
