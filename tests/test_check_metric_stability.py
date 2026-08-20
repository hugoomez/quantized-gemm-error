"""Unit tests for sample_t's `normalize="mad"` opt-in in
scripts/check_metric_stability.py.

`sample_t` is deliberately local to that script (PREREGISTRATION.md R15 --
the nu-axis sampler is study infrastructure that must be implemented and
tested on its own terms, not smuggled in through a diagnostic), so it is
loaded directly from the script file rather than from `qgemm`, mirroring
tests/test_sanity_gamma_n.py.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from qgemm.stats import median_absolute_deviation

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "check_metric_stability.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("check_metric_stability", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check_metric_stability = _load_module()

# The project's nu grid (configs/sweep_main.yaml), minus the "gaussian" end --
# sample_t only takes a finite nu; the gaussian end is sampled separately by
# qgemm.distributions.sample_gaussian elsewhere in the project.
NU_GRID = (1.0, 2.0, 3.0, 5.0, 8.0, 15.0, 30.0)


def test_sample_t_default_is_unnormalized_and_matches_raw_standard_t():
    # Regression check: existing callers (measure_u_eff.py, probe_n_scaling.py,
    # sanity_reproduce_2408.py, run_sweep.py) call sample_t without
    # `normalize`, and that path must stay byte-identical to the raw draw.
    rng_a = np.random.default_rng(10)
    rng_b = np.random.default_rng(10)
    raw = np.asarray(rng_a.standard_t(3.0, size=(500,)), dtype=np.float64)
    out = check_metric_stability.sample_t((500,), 3.0, rng_b)
    np.testing.assert_array_equal(out, raw)


@pytest.mark.parametrize("nu", NU_GRID)
def test_sample_t_normalize_mad_sets_mad_to_one_across_the_nu_grid(nu):
    rng = np.random.default_rng(11)
    x = check_metric_stability.sample_t((20_000,), nu, rng, normalize="mad")
    assert median_absolute_deviation(x) == pytest.approx(1.0, abs=1e-9)


def test_sample_t_unknown_normalize_value_is_rejected():
    rng = np.random.default_rng(12)
    with pytest.raises(ValueError, match="normalize"):
        check_metric_stability.sample_t((10,), 3.0, rng, normalize="zscore")
