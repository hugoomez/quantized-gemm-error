import numpy as np
import pytest
from ml_dtypes import float4_e2m1fn

from qgemm.formats import E2M1_MAGNITUDE_GRID, E2M1_MAX, quantize_e2m1

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
