import numpy as np
import pytest

# float8_e4m3fn is the deep-learning E4M3 (no infinity, max 448). ml_dtypes
# also ships float8_e4m3, which *does* have infinity and tops out at 240 --
# that is a different format and must never be used as the reference here.
from ml_dtypes import float4_e2m1fn, float8_e4m3fn, float8_e5m2, float8_e8m0fnu

from qgemm.formats import (
    E2M1_MAGNITUDE_GRID,
    E2M1_MAX,
    E4M3_MAGNITUDE_GRID,
    E4M3_MAX,
    E5M2_MAGNITUDE_GRID,
    E5M2_MAX,
    E8M0_GRID,
    E8M0_MAX,
    E8M0_MIN,
    int8_quantize,
    quantize_e2m1,
    quantize_e4m3,
    quantize_e5m2,
    quantize_e8m0,
)

# The 15 representable values of E2M1 (both signs; +0 and -0 collapse to one).
SIGNED_GRID = np.unique(
    np.concatenate([E2M1_MAGNITUDE_GRID, -np.asarray(E2M1_MAGNITUDE_GRID)])
)


def _reference_rtne(x: np.ndarray) -> np.ndarray:
    """RTNE ground truth: ml_dtypes' own float4_e2m1fn cast."""
    return np.asarray(x.astype(float4_e2m1fn), dtype=np.float64)


def test_magnitude_grid_is_the_documented_one():
    np.testing.assert_array_equal(
        E2M1_MAGNITUDE_GRID, np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    )
    assert E2M1_MAX == 6.0
    assert SIGNED_GRID.size == 15


def test_rtne_is_idempotent_on_grid_values():
    x = SIGNED_GRID.astype(np.float64)
    once = quantize_e2m1(x, mode="rtne")
    np.testing.assert_array_equal(once, x)
    np.testing.assert_array_equal(quantize_e2m1(once, mode="rtne"), x)


def test_sr_is_idempotent_on_grid_values():
    rng = np.random.default_rng(0)
    x = SIGNED_GRID.astype(np.float64)
    for _ in range(10):
        np.testing.assert_array_equal(quantize_e2m1(x, mode="sr", rng=rng), x)


def test_rtne_dense_sweep_always_lands_on_the_grid():
    x = np.linspace(-8.0, 8.0, 1_000_001, dtype=np.float64)
    q = quantize_e2m1(x, mode="rtne")
    assert np.isin(q, SIGNED_GRID).all()


def test_sr_dense_sweep_always_lands_on_the_grid():
    rng = np.random.default_rng(20260819)
    x = np.linspace(-8.0, 8.0, 1_000_001, dtype=np.float64)
    q = quantize_e2m1(x, mode="sr", rng=rng)
    assert np.isin(q, SIGNED_GRID).all()


def test_rtne_is_bit_exact_against_ml_dtypes_on_a_dense_sweep():
    """Non-negotiable: RTNE must agree with ml_dtypes.float4_e2m1fn exactly."""
    x = np.linspace(-8.0, 8.0, 1_000_001, dtype=np.float64)
    q = quantize_e2m1(x, mode="rtne")
    expected = _reference_rtne(x)
    np.testing.assert_array_equal(q, expected)
    # assert_array_equal treats -0.0 == 0.0, so check zero signs separately.
    np.testing.assert_array_equal(np.signbit(q), np.signbit(expected))


def test_rtne_is_bit_exact_against_ml_dtypes_on_random_values():
    rng = np.random.default_rng(7)
    x = rng.uniform(-10.0, 10.0, size=200_000).astype(np.float64)
    x = np.concatenate([x, rng.standard_normal(200_000) * 3.0])
    q = quantize_e2m1(x, mode="rtne")
    expected = _reference_rtne(x)
    np.testing.assert_array_equal(q, expected)
    np.testing.assert_array_equal(np.signbit(q), np.signbit(expected))


@pytest.mark.parametrize(
    ("midpoint", "expected"),
    [
        (0.25, 0.0),  # between 0 (bits 000, even) and 0.5 (001)
        (0.75, 1.0),  # between 0.5 (001) and 1.0 (010, even)
        (1.25, 1.0),  # between 1.0 (010, even) and 1.5 (011)
        (1.75, 2.0),  # between 1.5 (011) and 2.0 (100, even)
        (2.5, 2.0),  # between 2.0 (100, even) and 3.0 (101)
        (3.5, 4.0),  # between 3.0 (101) and 4.0 (110, even)
        (5.0, 4.0),  # between 4.0 (110, even) and 6.0 (111)
    ],
)
def test_rtne_breaks_ties_toward_the_even_bit_pattern(midpoint, expected):
    x = np.array([midpoint, -midpoint], dtype=np.float64)
    np.testing.assert_array_equal(
        quantize_e2m1(x, mode="rtne"), np.array([expected, -expected])
    )


def test_rtne_ties_are_not_uniform_rounding_up():
    """A grid-index np.round implementation silently gets 2.5 and 5.0 wrong."""
    x = np.array([2.5, 5.0], dtype=np.float64)
    np.testing.assert_array_equal(quantize_e2m1(x, mode="rtne"), np.array([2.0, 4.0]))


def test_subnormal_region_is_rounded_on_its_own_spacing():
    # Spacing below 1.0 is 0.5 (the subnormal step), not a scaled normal step.
    x = np.array([0.24, 0.26, 0.4, 0.6, 0.74, 0.76], dtype=np.float64)
    np.testing.assert_array_equal(
        quantize_e2m1(x, mode="rtne"), np.array([0.0, 0.5, 0.5, 0.5, 0.5, 1.0])
    )


def test_saturation_of_large_magnitudes():
    x = np.array([1000.0, -1000.0, 6.5, -6.5, np.inf, -np.inf], dtype=np.float64)
    np.testing.assert_array_equal(
        quantize_e2m1(x, mode="rtne"), np.array([6.0, -6.0, 6.0, -6.0, 6.0, -6.0])
    )


def test_saturation_is_deterministic_under_stochastic_rounding():
    rng = np.random.default_rng(3)
    x = np.array([1000.0, -1000.0], dtype=np.float64)
    for _ in range(50):
        np.testing.assert_array_equal(
            quantize_e2m1(x, mode="sr", rng=rng), np.array([6.0, -6.0])
        )


def test_signed_zero_is_preserved():
    x = np.array([0.0, -0.0, 0.1, -0.1], dtype=np.float64)
    q = quantize_e2m1(x, mode="rtne")
    np.testing.assert_array_equal(np.signbit(q), np.array([False, True, False, True]))


def test_sr_is_unbiased():
    rng = np.random.default_rng(12345)
    target = 1.3
    n = 100_000
    x = np.full(n, target, dtype=np.float64)
    q = quantize_e2m1(x, mode="sr", rng=rng)

    assert np.isin(q, np.array([1.0, 1.5])).all()
    se = q.std(ddof=1) / np.sqrt(n)
    assert abs(q.mean() - target) < 3.0 * se


def test_sr_only_picks_the_two_bracketing_neighbors():
    rng = np.random.default_rng(99)
    x = np.full(10_000, 2.4, dtype=np.float64)
    q = quantize_e2m1(x, mode="sr", rng=rng)
    assert set(np.unique(q)) == {2.0, 3.0}


def test_sr_is_reproducible_for_a_given_seed():
    x = np.linspace(-6.0, 6.0, 5_000, dtype=np.float64)
    a = quantize_e2m1(x, mode="sr", rng=np.random.default_rng(2026))
    b = quantize_e2m1(x, mode="sr", rng=np.random.default_rng(2026))
    np.testing.assert_array_equal(a, b)


def test_sr_uses_the_passed_generator_not_global_state():
    x = np.full(1_000, 1.3, dtype=np.float64)
    np.random.seed(0)
    a = quantize_e2m1(x, mode="sr", rng=np.random.default_rng(1))
    np.random.seed(999)
    b = quantize_e2m1(x, mode="sr", rng=np.random.default_rng(1))
    np.testing.assert_array_equal(a, b)


def test_sr_requires_a_generator():
    x = np.array([1.3], dtype=np.float64)
    with pytest.raises(ValueError, match="rng"):
        quantize_e2m1(x, mode="sr")


def test_rejects_non_float64_input():
    x = np.array([1.3], dtype=np.float32)
    with pytest.raises(TypeError, match="float64"):
        quantize_e2m1(x, mode="rtne")


def test_rejects_unknown_mode():
    x = np.array([1.3], dtype=np.float64)
    with pytest.raises(ValueError, match="mode"):
        quantize_e2m1(x, mode="rtz")


def test_rejects_nan_input():
    x = np.array([1.0, np.nan], dtype=np.float64)
    with pytest.raises(ValueError, match="NaN"):
        quantize_e2m1(x, mode="rtne")


def test_output_is_float64_and_shape_preserving():
    x = np.linspace(-7.0, 7.0, 24, dtype=np.float64).reshape(2, 3, 4)
    for q in (
        quantize_e2m1(x, mode="rtne"),
        quantize_e2m1(x, mode="sr", rng=np.random.default_rng(5)),
    ):
        assert q.dtype == np.float64
        assert q.shape == x.shape


def test_input_is_not_mutated():
    x = np.linspace(-7.0, 7.0, 101, dtype=np.float64)
    original = x.copy()
    quantize_e2m1(x, mode="rtne")
    quantize_e2m1(x, mode="sr", rng=np.random.default_rng(5))
    np.testing.assert_array_equal(x, original)


# ---------------------------------------------------------------------------
# E4M3 / E5M2 / E8M0
# ---------------------------------------------------------------------------
#
# Reference is ml_dtypes. One wrinkle drives the whole test design: ml_dtypes
# casts float64 -> float32 -> float8, so its rounding double-rounds. Our
# quantizers correctly-round straight from float64 (see SPEC.md), which is the
# same function on every float32-representable input and differs only on
# float64 values lying within half a float32 ulp of a float8 tie. So:
#
#   * float32-exact probes  -> bit-exact agreement is required, strictly;
#   * dense float64 sweeps  -> bit-exact agreement, checked as-is;
#   * the deviation set     -> pinned by explicit tests below, not hidden.


def _cast(x, dtype):
    """Ground truth: ml_dtypes' own cast, returned as float64."""
    with np.errstate(over="ignore", invalid="ignore"):
        return np.asarray(np.asarray(x, dtype=np.float64).astype(dtype), dtype=np.float64)


def _assert_bit_exact(got, expected):
    """Exact match including NaN, and including the sign of zeros and NaNs.

    `assert_array_equal` treats NaN == NaN but also -0.0 == 0.0, so the sign
    bits are compared separately.
    """
    np.testing.assert_array_equal(got, expected)
    np.testing.assert_array_equal(np.signbit(got), np.signbit(expected))


def _codes(dtype, n):
    """The first `n` bit patterns of `dtype`, upcast to float64."""
    return np.asarray(np.arange(n, dtype=np.uint8).view(dtype), dtype=np.float64)


def _float32_exact(values):
    """Keep only the entries that survive a float64 -> float32 -> float64 round trip."""
    v = np.asarray(values, dtype=np.float64)
    v32 = np.asarray(v.astype(np.float32), dtype=np.float64)
    return v[np.isfinite(v32) & (v32 == v)]


def _boundary_probes(grid, signed=True):
    """Every grid point, every midpoint between neighbours, and their float32 neighbours.

    Adjacent float8 magnitudes differ by one float8 ulp, so a midpoint needs
    exactly one more mantissa bit -- these probes are float32-exact by
    construction (anything that is not, e.g. past the float32 range, is
    filtered out). They pin the rounding function down completely: a
    piecewise-constant function is determined by its behaviour at and either
    side of each of its breakpoints.
    """
    g = np.asarray(grid, dtype=np.float64)
    pts = np.concatenate([g, (g[:-1] + g[1:]) / 2.0]).astype(np.float32)
    probes = np.concatenate(
        [pts, np.nextafter(pts, np.float32(-np.inf)), np.nextafter(pts, np.float32(np.inf))]
    )
    probes = _float32_exact(np.asarray(probes, dtype=np.float64))
    if signed:
        probes = np.concatenate([probes, -probes])
    return np.unique(probes)


# The magnitude one step past the maximum: the E4M3 NaN pattern (code 127) and
# the E5M2 infinity pattern (code 124) sit here, so rounding *to* them is what
# produces overflow. Including them makes the probe set cover the max boundary.
E4M3_OVERFLOW_MAGNITUDE = 480.0
E5M2_OVERFLOW_MAGNITUDE = 65536.0


# --- grids ------------------------------------------------------------------


def test_e4m3_grid_is_the_ml_dtypes_code_order():
    # Codes 0..126 are the finite magnitudes; code 127 is the NaN pattern.
    np.testing.assert_array_equal(E4M3_MAGNITUDE_GRID, _codes(float8_e4m3fn, 127))
    assert E4M3_MAGNITUDE_GRID.size == 127
    assert E4M3_MAGNITUDE_GRID[0] == 0.0
    assert E4M3_MAX == 448.0
    assert E4M3_MAGNITUDE_GRID[-1] == E4M3_MAX
    # Subnormals run down to 2**-9, the smallest positive magnitude.
    assert E4M3_MAGNITUDE_GRID[1] == 2.0**-9
    # Smallest normal is 2**-6 (bias 7).
    assert E4M3_MAGNITUDE_GRID[8] == 2.0**-6


def test_e5m2_grid_is_the_ml_dtypes_code_order():
    # Codes 0..123 are the finite magnitudes; code 124 is infinity.
    np.testing.assert_array_equal(E5M2_MAGNITUDE_GRID, _codes(float8_e5m2, 124))
    assert E5M2_MAGNITUDE_GRID.size == 124
    assert E5M2_MAGNITUDE_GRID[0] == 0.0
    assert E5M2_MAX == 57344.0
    assert E5M2_MAGNITUDE_GRID[-1] == E5M2_MAX
    assert E5M2_MAGNITUDE_GRID[1] == 2.0**-16
    # Smallest normal is 2**-14 (bias 15).
    assert E5M2_MAGNITUDE_GRID[4] == 2.0**-14


def test_e8m0_grid_is_the_ml_dtypes_code_order():
    # Codes 0..254 are the powers of two 2**-127 .. 2**127; code 255 is NaN.
    np.testing.assert_array_equal(E8M0_GRID, _codes(float8_e8m0fnu, 255))
    assert E8M0_GRID.size == 255
    assert E8M0_MIN == 2.0**-127
    assert E8M0_MAX == 2.0**127
    assert E8M0_GRID[0] == E8M0_MIN
    assert E8M0_GRID[-1] == E8M0_MAX
    # All powers of two, no zero, strictly increasing.
    np.testing.assert_array_equal(E8M0_GRID, 2.0 ** np.arange(-127.0, 128.0))


# --- bit-exactness against ml_dtypes ---------------------------------------


def _to_float32_exact(x):
    """Snap a sweep onto the float32 grid, where ml_dtypes' intermediate is a no-op."""
    return np.asarray(np.asarray(x, dtype=np.float64).astype(np.float32), dtype=np.float64)


def test_e4m3_bit_exact_at_every_grid_boundary():
    x = _boundary_probes(np.append(E4M3_MAGNITUDE_GRID, E4M3_OVERFLOW_MAGNITUDE))
    _assert_bit_exact(quantize_e4m3(x), _cast(x, float8_e4m3fn))


def test_e5m2_bit_exact_at_every_grid_boundary():
    x = _boundary_probes(np.append(E5M2_MAGNITUDE_GRID, E5M2_OVERFLOW_MAGNITUDE))
    _assert_bit_exact(quantize_e5m2(x), _cast(x, float8_e5m2))


def test_e8m0_bit_exact_at_every_grid_boundary():
    # Below 2**-126 ml_dtypes has a separate float32-subnormal defect, pinned
    # by test_e8m0_ml_dtypes_rounds_up_in_the_float32_subnormal_range.
    x = _boundary_probes(E8M0_GRID)
    x = x[np.abs(x) >= 2.0**-126]
    _assert_bit_exact(quantize_e8m0(x), _cast(x, float8_e8m0fnu))


def test_e4m3_bit_exact_on_a_dense_sweep_across_the_whole_range():
    # Spans zero, the subnormals, the max, and well past it, both signs.
    x = _to_float32_exact(np.linspace(-600.0, 600.0, 1_000_001))
    _assert_bit_exact(quantize_e4m3(x), _cast(x, float8_e4m3fn))


def test_e4m3_bit_exact_on_a_dense_sweep_through_the_subnormals():
    # A sweep across [-600, 600] steps right over the subnormals (all below
    # 2**-6), so they get their own dense sweep.
    x = _to_float32_exact(np.linspace(-(2.0**-5), 2.0**-5, 1_000_001))
    _assert_bit_exact(quantize_e4m3(x), _cast(x, float8_e4m3fn))


def test_e5m2_bit_exact_on_a_dense_sweep_across_the_whole_range():
    x = _to_float32_exact(np.linspace(-70_000.0, 70_000.0, 1_000_001))
    _assert_bit_exact(quantize_e5m2(x), _cast(x, float8_e5m2))


def test_e5m2_bit_exact_on_a_dense_sweep_through_the_subnormals():
    x = _to_float32_exact(np.linspace(-(2.0**-13), 2.0**-13, 1_000_001))
    _assert_bit_exact(quantize_e5m2(x), _cast(x, float8_e5m2))


def test_e8m0_bit_exact_on_a_dense_logarithmic_sweep():
    # E8M0 spans 255 binades, so a linear sweep is useless -- sweep the exponent.
    # The top end reaches past 1.5 * 2**127, so overflow-to-NaN is covered; the
    # bottom stops at 2**-126 (see the float32-subnormal test).
    mag = _to_float32_exact(np.exp2(np.linspace(-126.0, 127.9, 500_001)))
    x = np.concatenate([mag, -mag, np.zeros(1)])
    _assert_bit_exact(quantize_e8m0(x), _cast(x, float8_e8m0fnu))


def test_e8m0_bit_exact_on_a_dense_linear_sweep():
    x = _to_float32_exact(np.linspace(-16.0, 16.0, 1_000_001))
    _assert_bit_exact(quantize_e8m0(x), _cast(x, float8_e8m0fnu))


@pytest.mark.parametrize(
    ("fn", "dtype", "lo", "hi"),
    [
        (quantize_e4m3, float8_e4m3fn, -600.0, 600.0),
        (quantize_e5m2, float8_e5m2, -70_000.0, 70_000.0),
        (quantize_e8m0, float8_e8m0fnu, -16.0, 16.0),
    ],
)
def test_bit_exact_on_random_float32_values(fn, dtype, lo, hi):
    """Random float32-exact input: agreement here must be total, by construction."""
    rng = np.random.default_rng(20260819)
    x = _float32_exact(rng.uniform(lo, hi, 500_000).astype(np.float32).astype(np.float64))
    _assert_bit_exact(fn(x), _cast(x, dtype))


# --- the exact relationship to ml_dtypes ------------------------------------


def _sweeps(lo, hi, tiny):
    rng = np.random.default_rng(4242)
    return [
        np.linspace(lo, hi, 1_000_001, dtype=np.float64),
        np.linspace(-tiny, tiny, 1_000_001, dtype=np.float64),
        rng.uniform(lo, hi, 1_000_000),
        rng.standard_normal(1_000_000) * (hi / 6.0),
        np.array([0.0, -0.0, np.inf, -np.inf, np.nan, -np.nan, 5e-324, -5e-324]),
    ]


@pytest.mark.parametrize(
    ("fn", "dtype", "lo", "hi", "tiny"),
    [
        (quantize_e4m3, float8_e4m3fn, -600.0, 600.0, 2.0**-5),
        (quantize_e5m2, float8_e5m2, -70_000.0, 70_000.0, 2.0**-13),
    ],
)
def test_ml_dtypes_is_exactly_our_quantizer_after_a_float32_round(fn, dtype, lo, hi, tiny):
    """ml_dtypes(x) == ours(float32(x)), for every float64 x -- no exceptions.

    This is the whole story of the deviation, stated as an identity rather
    than as a list of excused mismatches: ml_dtypes casts float64 -> float32
    -> float8, and once that first rounding is applied by hand the two
    quantizers agree bit for bit everywhere, NaNs and signed zeros included.
    So the *only* difference between us and ml_dtypes is the float32
    intermediate, which is exactly what SPEC.md claims.
    """
    for x in _sweeps(lo, hi, tiny):
        x = np.asarray(x, dtype=np.float64)
        with np.errstate(over="ignore"):
            x32 = np.asarray(x.astype(np.float32), dtype=np.float64)
        _assert_bit_exact(_cast(x, dtype), fn(x32))


def test_ml_dtypes_e8m0_is_our_quantizer_after_a_float32_round_above_the_subnormals():
    """Same identity for E8M0, wherever float32 has normal numbers to work with."""
    rng = np.random.default_rng(4243)
    mag = np.exp2(rng.uniform(-126.0, 130.0, 2_000_000))
    x = np.concatenate([mag, -mag, np.array([0.0, -0.0, np.inf, -np.inf, np.nan])])
    x = x[(np.abs(x) >= 2.0**-126) | ~(np.abs(x) > 0.0)]
    with np.errstate(over="ignore"):
        x32 = np.asarray(x.astype(np.float32), dtype=np.float64)
    _assert_bit_exact(_cast(x, float8_e8m0fnu), quantize_e8m0(x32))


# --- zero, subnormals, max, just past the max, negatives --------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.0, 0.0),
        (2.0**-9, 2.0**-9),  # smallest subnormal
        (2.0**-10, 0.0),  # exactly half of it -> ties to even -> 0
        (1.5 * 2.0**-9, 2.0**-8),  # subnormal tie -> even code 2
        (2.5 * 2.0**-9, 2.0**-8),  # subnormal tie -> even code 2
        (2.0**-6, 2.0**-6),  # smallest normal
        (448.0, 448.0),  # max
        (463.9, 448.0),
        (464.0, 448.0),  # tie with the 480 NaN pattern -> even code 126
    ],
)
def test_e4m3_grid_landmarks(value, expected):
    x = np.array([value, -value], dtype=np.float64)
    _assert_bit_exact(quantize_e4m3(x), _cast(x, float8_e4m3fn))
    np.testing.assert_array_equal(quantize_e4m3(x), np.array([expected, -expected]))


def test_e4m3_overflows_to_nan_rather_than_saturating():
    """E4M3 has no infinity; past the max it goes to NaN, sign preserved."""
    x = np.array([465.0, -465.0, 500.0, -500.0, 1e9, np.inf, -np.inf], dtype=np.float64)
    q = quantize_e4m3(x)
    assert np.isnan(q).all()
    np.testing.assert_array_equal(
        np.signbit(q), np.array([False, True, False, True, False, False, True])
    )
    _assert_bit_exact(q, _cast(x, float8_e4m3fn))


def test_e4m3_passes_nan_through():
    x = np.array([np.nan, -np.nan], dtype=np.float64)
    _assert_bit_exact(quantize_e4m3(x), _cast(x, float8_e4m3fn))


def test_e4m3_preserves_signed_zero():
    x = np.array([0.0, -0.0, 1e-30, -1e-30], dtype=np.float64)
    q = quantize_e4m3(x)
    np.testing.assert_array_equal(q, np.zeros(4))
    np.testing.assert_array_equal(np.signbit(q), np.array([False, True, False, True]))


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.0, 0.0),
        (2.0**-16, 2.0**-16),  # smallest subnormal
        (2.0**-17, 0.0),  # half of it -> ties to even -> 0
        (1.5 * 2.0**-16, 2.0**-15),  # subnormal tie -> even code 2
        (2.0**-14, 2.0**-14),  # smallest normal
        (57344.0, 57344.0),  # max
        (61439.0, 57344.0),
    ],
)
def test_e5m2_grid_landmarks(value, expected):
    x = np.array([value, -value], dtype=np.float64)
    _assert_bit_exact(quantize_e5m2(x), _cast(x, float8_e5m2))
    np.testing.assert_array_equal(quantize_e5m2(x), np.array([expected, -expected]))


def test_e5m2_overflows_to_infinity_ieee_style():
    """E5M2 is IEEE-shaped: past the max it goes to +/-inf, and the tie rounds up."""
    x = np.array([61440.0, -61440.0, 70_000.0, np.inf, -np.inf], dtype=np.float64)
    q = quantize_e5m2(x)
    np.testing.assert_array_equal(
        q, np.array([np.inf, -np.inf, np.inf, np.inf, -np.inf])
    )
    _assert_bit_exact(q, _cast(x, float8_e5m2))


def test_e5m2_passes_nan_through():
    x = np.array([np.nan, -np.nan], dtype=np.float64)
    _assert_bit_exact(quantize_e5m2(x), _cast(x, float8_e5m2))


def test_e5m2_preserves_signed_zero():
    x = np.array([0.0, -0.0, 1e-30, -1e-30], dtype=np.float64)
    q = quantize_e5m2(x)
    np.testing.assert_array_equal(q, np.zeros(4))
    np.testing.assert_array_equal(np.signbit(q), np.array([False, True, False, True]))


def test_e8m0_has_no_zero_encoding():
    """E8M0's smallest code is 2**-127; zero is simply not representable."""
    x = np.array([0.0, -0.0], dtype=np.float64)
    q = quantize_e8m0(x)
    assert np.isnan(q).all()
    _assert_bit_exact(q, _cast(x, float8_e8m0fnu))


def test_e8m0_is_unsigned():
    x = np.array([-1.0, -2.0, -0.5, -1e-30, -np.inf], dtype=np.float64)
    q = quantize_e8m0(x)
    assert np.isnan(q).all()
    _assert_bit_exact(q, _cast(x, float8_e8m0fnu))


def test_e8m0_nan_output_is_never_signed():
    """E8M0 has no sign bit, so its single NaN code (0xFF) is unsigned."""
    x = np.array([-1.0, np.nan, -np.nan, 0.0, np.inf, 1e40], dtype=np.float64)
    q = quantize_e8m0(x)
    assert np.isnan(q).all()
    assert not np.signbit(q).any()


def test_e8m0_rounds_ties_up_not_to_even():
    """1.5*2**e is equidistant from 2**e and 2**(e+1); ml_dtypes always takes the upper.

    Ties-to-even on the code would alternate with the parity of e, so this
    fails loudly for a ties-to-even implementation.
    """
    e = np.arange(-126.0, 127.0)
    x = 1.5 * 2.0**e
    np.testing.assert_array_equal(quantize_e8m0(x), 2.0 ** (e + 1))
    _assert_bit_exact(quantize_e8m0(x), _cast(x, float8_e8m0fnu))


def test_e8m0_underflow_clamps_up_to_the_minimum():
    x = np.array([2.0**-128, 2.0**-130, 2.0**-200, 1.5 * 2.0**-128], dtype=np.float64)
    np.testing.assert_array_equal(quantize_e8m0(x), np.full(4, E8M0_MIN))
    # ml_dtypes agrees except at 2**-200, which its float32 step flushes to
    # zero and therefore reports as NaN (see the deviation tests below).
    agree = x >= 2.0**-149
    _assert_bit_exact(quantize_e8m0(x[agree]), _cast(x[agree], float8_e8m0fnu))


def test_e8m0_overflow_goes_to_nan_not_to_the_maximum():
    """Asymmetric with underflow: too small clamps, too large is NaN."""
    below = np.array([1.4999 * 2.0**127, E8M0_MAX], dtype=np.float64)
    np.testing.assert_array_equal(quantize_e8m0(below), np.full(2, E8M0_MAX))

    above = np.array([1.5 * 2.0**127, 2.0**128, 1e40, np.inf], dtype=np.float64)
    assert np.isnan(quantize_e8m0(above)).all()
    _assert_bit_exact(quantize_e8m0(above), _cast(above, float8_e8m0fnu))


# --- documented deviations from ml_dtypes (double rounding) -----------------


def test_e8m0_correctly_rounds_from_float64_where_ml_dtypes_double_rounds():
    """1.4999999999999998 is below the 1.5 tie, so it must round down to 1.0.

    ml_dtypes rounds it to float32 first, which lands exactly on 1.5, and then
    takes the tie upward to 2.0. This deviation is deliberate and is recorded
    in SPEC.md -- correct rounding from float64 is what keeps
    |Q(x) - x| <= half a grid step.
    """
    x = np.array([np.nextafter(1.5, 0.0)], dtype=np.float64)
    assert quantize_e8m0(x)[0] == 1.0
    assert _cast(x, float8_e8m0fnu)[0] == 2.0


def test_e4m3_correctly_rounds_from_float64_where_ml_dtypes_double_rounds():
    """Just above the 464 tie must round up to the 480 NaN pattern, i.e. overflow."""
    x = np.array([np.nextafter(464.0, np.inf)], dtype=np.float64)
    assert np.isnan(quantize_e4m3(x)[0])
    assert _cast(x, float8_e4m3fn)[0] == 448.0


def test_e8m0_clamps_float64_subnormals_where_ml_dtypes_flushes_to_nan():
    """Anything below 2**-149 underflows to zero in float32, and zero is E8M0 NaN.

    Straight from float64 these are simply very small positive numbers, so
    they clamp to 2**-127 like every other underflow.
    """
    x = np.array([5e-324, 2.0**-200, 2.0**-150], dtype=np.float64)
    np.testing.assert_array_equal(quantize_e8m0(x), np.full(3, E8M0_MIN))
    assert np.isnan(_cast(x, float8_e8m0fnu)).all()


def test_e8m0_ml_dtypes_rounds_up_in_the_float32_subnormal_range():
    """A second, independent ml_dtypes defect, unrelated to double rounding.

    float32 subnormals start below 2**-126, and in that range ml_dtypes' E8M0
    cast rounds up a whole binade for anything that is not exactly a power of
    two -- even values a single float32 ulp above 2**-127, which are nowhere
    near the 1.5*2**-127 midpoint. Correct rounding keeps them at 2**-127.

    Note this is not explained by the float32 intermediate: the inputs below
    are float32-exact, so that step is the identity here.
    """
    ulp = 2.0**-149  # the float32 subnormal step
    x = np.array([2.0**-127 + k * ulp for k in (0, 1, 2, 100)], dtype=np.float64)
    np.testing.assert_array_equal(x.astype(np.float32).astype(np.float64), x)

    np.testing.assert_array_equal(quantize_e8m0(x), np.full(4, E8M0_MIN))
    np.testing.assert_array_equal(
        _cast(x, float8_e8m0fnu), np.array([2.0**-127, 2.0**-126, 2.0**-126, 2.0**-126])
    )


# --- format-level properties, independent of ml_dtypes ----------------------


@pytest.mark.parametrize(
    ("fn", "grid"),
    [
        (quantize_e4m3, "e4m3"),
        (quantize_e5m2, "e5m2"),
        (quantize_e8m0, "e8m0"),
    ],
)
def test_quantizing_is_idempotent_on_grid_values(fn, grid):
    if grid == "e8m0":
        x = E8M0_GRID.copy()
    else:
        mag = E4M3_MAGNITUDE_GRID if grid == "e4m3" else E5M2_MAGNITUDE_GRID
        x = np.unique(np.concatenate([mag, -mag]))
    np.testing.assert_array_equal(fn(x), x)
    np.testing.assert_array_equal(fn(fn(x)), x)


@pytest.mark.parametrize(
    ("fn", "grid", "lo", "hi"),
    [
        (quantize_e4m3, "e4m3", -448.0, 448.0),
        (quantize_e5m2, "e5m2", -57_344.0, 57_344.0),
    ],
)
def test_result_is_always_a_nearest_grid_point(fn, grid, lo, hi):
    """Correct rounding, checked against the grid directly rather than ml_dtypes."""
    mag = E4M3_MAGNITUDE_GRID if grid == "e4m3" else E5M2_MAGNITUDE_GRID
    signed = np.unique(np.concatenate([mag, -mag]))
    rng = np.random.default_rng(11)
    x = rng.uniform(lo, hi, 20_000)
    q = fn(x)
    assert np.isin(q, signed).all()
    # No grid point is strictly closer to x than the one we returned.
    assert (np.abs(q - x) <= np.abs(signed[None, :] - x[:, None]).min(axis=1)).all()


def test_e8m0_result_is_always_a_nearest_power_of_two():
    rng = np.random.default_rng(12)
    x = np.exp2(rng.uniform(-120.0, 120.0, 20_000))
    q = quantize_e8m0(x)
    assert np.isin(q, E8M0_GRID).all()
    assert (np.abs(q - x) <= np.abs(E8M0_GRID[None, :] - x[:, None]).min(axis=1)).all()


# --- contract ---------------------------------------------------------------


@pytest.mark.parametrize("fn", [quantize_e4m3, quantize_e5m2, quantize_e8m0])
def test_float8_rejects_non_float64_input(fn):
    with pytest.raises(TypeError, match="float64"):
        fn(np.array([1.3], dtype=np.float32))


@pytest.mark.parametrize("fn", [quantize_e4m3, quantize_e5m2, quantize_e8m0])
def test_float8_output_is_float64_and_shape_preserving(fn):
    x = np.linspace(0.1, 7.0, 24, dtype=np.float64).reshape(2, 3, 4)
    q = fn(x)
    assert q.dtype == np.float64
    assert q.shape == x.shape


@pytest.mark.parametrize("fn", [quantize_e4m3, quantize_e5m2, quantize_e8m0])
def test_float8_input_is_not_mutated(fn):
    x = np.linspace(-7.0, 7.0, 101, dtype=np.float64)
    original = x.copy()
    fn(x)
    np.testing.assert_array_equal(x, original)


# ---------------------------------------------------------------------------
# INT8 (symmetric, per-tensor)
# ---------------------------------------------------------------------------
#
# No ml_dtypes reference here: int8 quantization is not a number format in the
# sense the float8s are -- the grid depends on the *tensor*, through the single
# amax-derived scale -- so the tests below pin the recipe itself (scale, code
# range, symmetry, saturation) rather than a bit pattern.


def test_int8_round_trip_on_known_values():
    # amax = 127 makes the scale exactly 1.0, so the expected codes are the
    # rounded inputs and no floating-point scaling obscures the comparison.
    x = np.array([-127.0, -3.7, -0.4, 0.0, 42.2, 126.5, 127.0])
    expected = np.array([-127.0, -4.0, -0.0, 0.0, 42.0, 126.0, 127.0])
    np.testing.assert_array_equal(int8_quantize(x), expected)


def test_int8_scale_is_amax_over_127():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(1000) * 3.0
    q = int8_quantize(x)
    scale = np.max(np.abs(x)) / 127.0
    codes = q / scale
    np.testing.assert_allclose(codes, np.round(codes), atol=1e-9)
    assert np.abs(codes).max() == 127.0


def test_int8_is_symmetric_about_zero():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((7, 11)) * 5.0
    np.testing.assert_array_equal(int8_quantize(-x), -int8_quantize(x))


def test_int8_reproduces_the_extreme_element_exactly():
    # The element attaining amax lands on code +-127 by construction, so it is
    # the one input value the round trip is exact on.
    x = np.array([-100.0, 0.3, 50.0, 7.0])
    np.testing.assert_array_equal(int8_quantize(x)[0], -100.0)
    np.testing.assert_array_equal(int8_quantize(-x)[0], 100.0)


def test_int8_codes_never_leave_the_symmetric_range():
    rng = np.random.default_rng(2)
    # Heavy tails: one huge outlier sets amax, everything else is tiny.
    x = np.concatenate([rng.standard_normal(5000), np.array([1e6, -1e6])])
    q = int8_quantize(x)
    codes = q / (np.max(np.abs(x)) / 127.0)
    assert codes.min() >= -127.0
    assert codes.max() <= 127.0
    assert np.abs(q).max() <= np.abs(x).max()


def test_int8_saturates_rather_than_wrapping_at_the_top_code():
    # A value a hair above amax cannot occur from the same tensor, but the
    # clip is what guarantees round-off at the top code cannot produce 128.
    x = np.array([np.nextafter(1.0, 2.0), -1.0, 0.5])
    q = int8_quantize(x)
    codes = q / (np.max(np.abs(x)) / 127.0)
    assert np.abs(codes).max() == 127.0


def test_int8_all_zero_tensor_returns_zeros():
    x = np.zeros((3, 4))
    q = int8_quantize(x)
    np.testing.assert_array_equal(q, np.zeros((3, 4)))


def test_int8_rejects_non_float64_input():
    with pytest.raises(TypeError, match="float64"):
        int8_quantize(np.array([1.3], dtype=np.float32))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_int8_rejects_non_finite_input(bad):
    with pytest.raises(ValueError, match="finite"):
        int8_quantize(np.array([1.0, bad, -2.0]))


def test_int8_output_is_float64_and_shape_preserving():
    x = np.linspace(-7.0, 7.0, 24, dtype=np.float64).reshape(2, 3, 4)
    q = int8_quantize(x)
    assert q.dtype == np.float64
    assert q.shape == x.shape


def test_int8_input_is_not_mutated():
    x = np.linspace(-7.0, 7.0, 101, dtype=np.float64)
    original = x.copy()
    int8_quantize(x)
    np.testing.assert_array_equal(x, original)
