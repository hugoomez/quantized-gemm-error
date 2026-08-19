import numpy as np

from qgemm.metrics import abs_error, max_abs_error, mean_abs_error, relative_error, rms_error


def test_metrics_zero_for_identical_arrays():
    x = np.array([1.0, -2.0, 3.5], dtype=np.float64)
    assert max_abs_error(x, x) == 0.0
    assert mean_abs_error(x, x) == 0.0
    assert rms_error(x, x) == 0.0
    np.testing.assert_array_equal(abs_error(x, x), np.zeros_like(x))
    np.testing.assert_array_equal(relative_error(x, x), np.zeros_like(x))


def test_metrics_known_values():
    approx = np.array([1.0, 2.0], dtype=np.float64)
    reference = np.array([1.5, 2.0], dtype=np.float64)
    assert max_abs_error(approx, reference) == 0.5
    assert mean_abs_error(approx, reference) == 0.25
