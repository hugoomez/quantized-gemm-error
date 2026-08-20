import numpy as np

from qgemm.metrics import (
    abs_error,
    backward_error,
    max_abs_error,
    mean_abs_error,
    relative_error,
    rms_error,
)


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


def test_backward_error_scalar_hand_check():
    # A @ B = [[6.0]], Chat = [[7.0]] -> num = 1, den = |2|*|3| = 6, BE = 1/6.
    a = np.array([[2.0]], dtype=np.float64)
    b = np.array([[3.0]], dtype=np.float64)
    chat = np.array([[7.0]], dtype=np.float64)
    be = backward_error(a, b, chat)
    np.testing.assert_allclose(be, np.array([[1.0 / 6.0]]))


def test_backward_error_zero_for_exact_reconstruction():
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64)
    chat = a @ b
    be = backward_error(a, b, chat)
    np.testing.assert_array_equal(be, np.zeros_like(be))


def test_backward_error_known_2x2_value():
    # A @ B = [[19, 22], [43, 50]]; |A| @ |B| is the same here since all
    # entries are positive. Chat perturbs only the (0, 0) entry by 1.
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64)
    chat = np.array([[20.0, 22.0], [43.0, 50.0]], dtype=np.float64)
    be = backward_error(a, b, chat)
    expected = np.array([[1.0 / 19.0, 0.0], [0.0, 0.0]])
    np.testing.assert_allclose(be, expected)


def test_backward_error_denominator_uses_abs_not_exact_product():
    # A @ B = [[0.0]] (cancellation), but |A| @ |B| = [[2.0]] -- distinct
    # from a naive relative-error denominator built from the exact product.
    a = np.array([[1.0, -1.0]], dtype=np.float64)
    b = np.array([[1.0], [1.0]], dtype=np.float64)
    chat = np.array([[0.5]], dtype=np.float64)
    be = backward_error(a, b, chat)
    np.testing.assert_allclose(be, np.array([[0.25]]))


def test_backward_error_shape_and_dtype():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((4, 5)).astype(np.float64)
    b = rng.standard_normal((5, 3)).astype(np.float64)
    chat = (a @ b) + 0.01
    be = backward_error(a, b, chat)
    assert be.shape == (4, 3)
    assert be.dtype == np.float64
