import numpy as np

from qgemm.distributions import sample_gaussian, sample_uniform


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
