import numpy as np
import pytest

from qgemm.distributions import sample_gaussian, sample_uniform
from qgemm.stats import median_absolute_deviation


def test_sample_gaussian_reproducible_with_explicit_generator():
    rng_a = np.random.default_rng(0)
    rng_b = np.random.default_rng(0)
    a = sample_gaussian((100,), rng_a)
    b = sample_gaussian((100,), rng_b)
    np.testing.assert_array_equal(a, b)
    assert a.dtype == np.float64


def test_sample_uniform_bounds():
    rng = np.random.default_rng(0)
    x = sample_uniform((1000,), rng, low=-2.0, high=3.0)
    assert x.min() >= -2.0
    assert x.max() < 3.0
    assert x.dtype == np.float64


# --- normalize="mad" opt-in, additive to the raw generators above -------------


def test_sample_gaussian_default_is_unnormalized_and_matches_raw_behavior():
    # Regression check: existing diagnostic scripts call sample_gaussian without
    # `normalize`, and that path must stay byte-identical to the raw draw.
    rng_a = np.random.default_rng(1)
    rng_b = np.random.default_rng(1)
    raw = (rng_a.standard_normal((500,)) * 2.0).astype(np.float64)
    out = sample_gaussian((500,), rng_b, scale=2.0)
    np.testing.assert_array_equal(out, raw)


def test_sample_uniform_default_is_unnormalized_and_matches_raw_behavior():
    rng_a = np.random.default_rng(1)
    rng_b = np.random.default_rng(1)
    raw = rng_a.uniform(-2.0, 3.0, size=(500,)).astype(np.float64)
    out = sample_uniform((500,), rng_b, low=-2.0, high=3.0)
    np.testing.assert_array_equal(out, raw)


def test_sample_gaussian_normalize_mad_sets_mad_to_one():
    rng = np.random.default_rng(2)
    x = sample_gaussian((10_000,), rng, scale=7.0, normalize="mad")
    assert median_absolute_deviation(x) == pytest.approx(1.0, abs=1e-9)


def test_sample_uniform_normalize_mad_sets_mad_to_one():
    rng = np.random.default_rng(3)
    x = sample_uniform((10_000,), rng, low=-5.0, high=5.0, normalize="mad")
    assert median_absolute_deviation(x) == pytest.approx(1.0, abs=1e-9)


def test_sample_gaussian_unknown_normalize_value_is_rejected():
    rng = np.random.default_rng(4)
    with pytest.raises(ValueError, match="normalize"):
        sample_gaussian((10,), rng, normalize="zscore")


def test_sample_uniform_unknown_normalize_value_is_rejected():
    rng = np.random.default_rng(4)
    with pytest.raises(ValueError, match="normalize"):
        sample_uniform((10,), rng, normalize="zscore")
