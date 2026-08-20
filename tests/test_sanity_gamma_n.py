"""Unit tests for the fp16 rounding helpers in scripts/sanity_gamma_n.py.

These helpers are deliberately local to the script (fp16 is a control/
reference format for validating the measurement pipeline, not one of the
quantized formats this project studies -- see the module docstring), so
they are loaded directly from the script file rather than from `qgemm`,
mirroring how `scripts/check_metric_stability.py` keeps its own
diagnostic-only `sample_t` local and untested via the package API.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "sanity_gamma_n.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("sanity_gamma_n", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sanity_gamma_n = _load_module()


def test_fp16_unit_roundoff_matches_10_bit_mantissa():
    assert np.finfo(np.float16).nmant == 10
    assert sanity_gamma_n.U_FP16 == pytest.approx(2.0**-11)


def test_round_stochastic_fp16_is_deterministic_on_exact_grid_points():
    rng = np.random.default_rng(0)
    x = np.array([1.0, 0.5, -2.0, 0.0], dtype=np.float64)  # all exactly fp16-representable
    for _ in range(5):
        result = sanity_gamma_n.round_stochastic_fp16(x, rng)
        np.testing.assert_array_equal(result, x)


def test_round_stochastic_fp16_hits_expected_neighbors_with_expected_probability():
    # fp16 spacing at [1, 2) is 2**-10; pick x a quarter-step above 1.0, so
    # P(round up to the next grid point) should be 0.25.
    lo = 1.0
    hi = float(np.nextafter(np.float16(1.0), np.float16(2.0)))
    step = hi - lo
    x = np.full(200_000, lo + 0.25 * step, dtype=np.float64)
    rng = np.random.default_rng(1)
    result = sanity_gamma_n.round_stochastic_fp16(x, rng)
    assert set(np.unique(result).tolist()).issubset({lo, hi})
    frac_hi = float(np.mean(result == hi))
    assert frac_hi == pytest.approx(0.25, abs=0.01)


def test_round_stochastic_fp16_is_unbiased():
    rng = np.random.default_rng(2)
    x = rng.uniform(-4.0, 4.0, size=2000).astype(np.float64)
    # Average many independent SR draws of the *same* x to estimate E[Q(x)].
    draws = np.stack([sanity_gamma_n.round_stochastic_fp16(x, rng) for _ in range(300)])
    np.testing.assert_allclose(draws.mean(axis=0), x, atol=5e-4)


def test_round_stochastic_fp16_output_dtype_and_shape():
    rng = np.random.default_rng(4)
    x = rng.standard_normal((3, 5)).astype(np.float64)
    result = sanity_gamma_n.round_stochastic_fp16(x, rng)
    assert result.shape == x.shape
    assert result.dtype == np.float64


def test_fp16_dot_rtne_matches_manual_sequential_rounding():
    a = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
    b = np.array([[1.0, 1.0, 1.0]], dtype=np.float64)
    result = sanity_gamma_n.fp16_dot_rtne(a, b)
    acc = np.float16(0.0)
    for ai, bi in zip(a[0], b[0], strict=True):
        acc = acc + np.float16(ai * bi)
    np.testing.assert_allclose(result, np.array([float(acc)]))


def test_fp16_dot_sr_matches_grid_when_products_are_exact():
    # Products and partial sums all land exactly on the fp16 grid, so SR
    # must agree with the (unrounded) float64 result exactly, regardless of
    # the RNG draws.
    a = np.array([[1.0, 2.0, 4.0]], dtype=np.float64)
    b = np.array([[1.0, 1.0, 1.0]], dtype=np.float64)
    rng = np.random.default_rng(3)
    result = sanity_gamma_n.fp16_dot_sr(a, b, rng)
    np.testing.assert_allclose(result, np.array([7.0]))


def test_fp16_dot_rtne_and_sr_batch_over_trials_independently():
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    b = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=np.float64)
    rtne = sanity_gamma_n.fp16_dot_rtne(a, b)
    assert rtne.shape == (2,)
    rng = np.random.default_rng(5)
    sr = sanity_gamma_n.fp16_dot_sr(a, b, rng)
    assert sr.shape == (2,)


def test_fit_loglog_slope_recovers_known_slope():
    n_values = np.array([16, 64, 256, 1024], dtype=np.float64)
    true_slope = 0.5
    y = 3.0 * n_values**true_slope
    slope, _intercept = sanity_gamma_n.fit_loglog_slope(n_values, y)
    assert slope == pytest.approx(true_slope, abs=1e-9)
