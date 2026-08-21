"""Tests for the Step 3.3 real execution harness in scripts/run_sweep.py.

Loaded by file path (mirroring tests/test_check_metric_stability.py), since
scripts/ is not an installed package.

Covers, in order:
  - cell identity / seed derivation (pure functions, no I/O)
  - single-trial and per-cell execution (normalize="mad", u_eff reuse)
  - the checkpoint file format (atomic write, skip-if-exists)
  - --cell-filter parsing
  - the two acceptance-critical properties: kill/resume produces an EXACTLY
    identical combined result, and no RNG state is shared/reused across
    parallel workers.
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_sweep.py"

# Imported as a real, sys.path-resolvable module (not via
# importlib.util.spec_from_file_location, unlike this project's other
# script-loading tests) because a handful of tests below send `run_sweep`
# functions into real multiprocessing worker processes: `multiprocessing`
# pickles those by `(module_name, qualname)` and the spawned child re-imports
# the module by name, which only works if "run_sweep" is actually importable
# from sys.path in that child.
if str(SCRIPT_PATH.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_PATH.parent))
run_sweep = importlib.import_module("run_sweep")


# ---------------------------------------------------------------------------
# A tiny grid, sized for test speed: 2 nu x 2 n x 1 block_size x 1
# scale_format x 1 round_mode x 1 rht = 4 cells, single-digit trial counts.
# matrix_shape must match run_sweep.MATRIX_M / MATRIX_K (load_grid_config
# asserts this), so it is kept at m=64, k=64 like the real grid.
def _tiny_grid_dict(heavy_trials: int = 6, light_trials: int = 6) -> dict:
    return {
        "matrix_shape": {"m": run_sweep.MATRIX_M, "k": run_sweep.MATRIX_K},
        "accum": "exact",
        "element_format": "e2m1",
        "use_global_scale_by_scale_format": {"e8m0": False, "e4m3": True},
        "main_grid": {
            "nu": [1, "gaussian"],
            "n": [16, 64],
            "block_size": [16],
            "scale_format": ["e8m0"],
            "round_mode": ["rtne"],
            "rht": [False],
        },
        "trial_budget": {
            "heavy_tail_nu": [1],
            "heavy_tail_trials": heavy_trials,
            "light_tail_nu": [5, 8, 15, 30, "gaussian"],
            "light_tail_trials": light_trials,
        },
        "reference_configs": {
            "formats": ["fp8_e4m3_per_tensor"],
            "nu": [1, "gaussian"],
            "n": [16],
        },
    }


def _write_tiny_grid(tmp_path: Path, **kwargs) -> Path:
    path = tmp_path / "tiny_grid.yaml"
    path.write_text(yaml.safe_dump(_tiny_grid_dict(**kwargs)), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# cell identity / seeding
# ---------------------------------------------------------------------------


def _sample_cell_key():
    return run_sweep.cell_key_from_cell(
        {
            "nu": 1,
            "n": 16,
            "block_size": 16,
            "scale_format": "e8m0",
            "use_global_scale": False,
            "round_mode": "rtne",
            "rht": False,
            "trials": 5000,
        }
    )


def test_cell_key_excludes_trial_count():
    key = _sample_cell_key()
    assert "trials" not in key


def test_cell_key_records_normalize_mad():
    # Step 3.2's decision must be visible in the identity of every cell this
    # harness produces, not just active silently.
    assert _sample_cell_key()["normalize"] == "mad"


def test_cell_id_is_deterministic():
    key = _sample_cell_key()
    assert run_sweep.cell_id(key) == run_sweep.cell_id(key)


def test_cell_id_differs_for_different_cells():
    a = run_sweep.cell_id(_sample_cell_key())
    b = run_sweep.cell_id({**_sample_cell_key(), "n": 64})
    assert a != b


def test_derive_seed_is_deterministic():
    key = _sample_cell_key()
    assert run_sweep.derive_seed(key, 0, "data") == run_sweep.derive_seed(key, 0, "data")


def test_derive_seed_varies_by_trial_index():
    key = _sample_cell_key()
    seeds = {run_sweep.derive_seed(key, t, "data") for t in range(20)}
    assert len(seeds) == 20


def test_derive_seed_varies_by_stream():
    key = _sample_cell_key()
    assert run_sweep.derive_seed(key, 0, "data") != run_sweep.derive_seed(key, 0, "gemm")


def test_derive_seed_varies_by_cell():
    key_a = _sample_cell_key()
    key_b = {**key_a, "n": 64}
    assert run_sweep.derive_seed(key_a, 0, "data") != run_sweep.derive_seed(key_b, 0, "data")


def test_derive_cell_seed_independent_of_trial_seeds():
    key = _sample_cell_key()
    cell_level = run_sweep.derive_cell_seed(key, "u_eff")
    trial_level = {run_sweep.derive_seed(key, t, "data") for t in range(10)}
    assert cell_level not in trial_level


# ---------------------------------------------------------------------------
# sampling / trial execution
# ---------------------------------------------------------------------------


def test_sampler_for_gaussian_applies_mad_normalization():
    from qgemm.stats import median_absolute_deviation

    sampler = run_sweep._sampler_for("gaussian")
    rng = np.random.default_rng(0)
    x = sampler((5000,), rng)
    assert median_absolute_deviation(x) == pytest.approx(1.0, abs=1e-6)


def test_sampler_for_t_applies_mad_normalization():
    from qgemm.stats import median_absolute_deviation

    sampler = run_sweep._sampler_for(3)
    rng = np.random.default_rng(0)
    x = sampler((20_000,), rng)
    assert median_absolute_deviation(x) == pytest.approx(1.0, abs=1e-6)


def test_run_trial_is_reproducible_given_same_cell_and_trial():
    key = _sample_cell_key()
    a = run_sweep.run_trial(key, 0, run_sweep.MATRIX_M, run_sweep.MATRIX_K)
    b = run_sweep.run_trial(key, 0, run_sweep.MATRIX_M, run_sweep.MATRIX_K)
    assert a == b


def test_run_trial_differs_across_trial_index():
    key = _sample_cell_key()
    a = run_sweep.run_trial(key, 0, run_sweep.MATRIX_M, run_sweep.MATRIX_K)
    b = run_sweep.run_trial(key, 1, run_sweep.MATRIX_M, run_sweep.MATRIX_K)
    assert a["seed_data"] != b["seed_data"]
    assert a["be_median"] != b["be_median"]


def test_run_trial_reports_median_and_p99():
    key = _sample_cell_key()
    row = run_sweep.run_trial(key, 0, run_sweep.MATRIX_M, run_sweep.MATRIX_K)
    assert row["be_median"] <= row["be_p99"]
    assert row["n_be_values"] == run_sweep.MATRIX_M * run_sweep.MATRIX_K


def test_compute_cell_u_eff_matches_direct_measure_u_eff_call():
    # Reuse check: compute_cell_u_eff must be a thin wrapper around
    # qgemm.bounds.measure_u_eff with this cell's exact config, not a
    # reimplementation -- so calling measure_u_eff by hand with the same
    # derived seed/sampler/tensor_shape must reproduce it exactly.
    from qgemm.bounds import measure_u_eff
    from qgemm.gemm import GemmConfig

    key = _sample_cell_key()
    n_elements = 5000
    result = run_sweep.compute_cell_u_eff(key, run_sweep.MATRIX_M, n_elements)

    config = GemmConfig(
        block_size=key["block_size"],
        scale_format=key["scale_format"],
        use_global_scale=key["use_global_scale"],
        element_format=key["element_format"],
        round_mode=key["round_mode"],
        rht=key["rht"],
        accum=key["accum"],
        seed=0,
    )
    sampler = run_sweep._sampler_for(key["nu"])
    seed = run_sweep.derive_cell_seed(key, "u_eff")
    rng = np.random.default_rng(seed)
    expected = measure_u_eff(
        config,
        sampler,
        n_elements,
        quantiles=(0.5, 0.99),
        rng=rng,
        tensor_shape=(run_sweep.MATRIX_M, key["n"]),
    )
    assert result["u_eff_p50"] == expected[0.5]
    assert result["u_eff_p99"] == expected[0.99]


# ---------------------------------------------------------------------------
# checkpoint file format
# ---------------------------------------------------------------------------


def test_atomic_write_parquet_leaves_no_tmp_file(tmp_path):
    df = pd.DataFrame({"x": [1, 2, 3]})
    out_path = tmp_path / "cell_abc123.parquet"
    run_sweep._atomic_write_parquet(df, out_path)
    assert out_path.exists()
    assert list(tmp_path.glob("*.tmp")) == []
    pd.testing.assert_frame_equal(pd.read_parquet(out_path), df)


def test_process_cell_writes_checkpoint(tmp_path):
    key = _sample_cell_key()
    out_path = tmp_path / f"cell_{run_sweep.cell_id(key)}.parquet"
    job = {
        "cell_key": key,
        "cell_id": run_sweep.cell_id(key),
        "trials": 3,
        "m": run_sweep.MATRIX_M,
        "k": run_sweep.MATRIX_K,
        "out_path": str(out_path),
        "u_eff_n_elements": 2000,
    }
    result = run_sweep.process_cell(job)
    assert result["status"] == "ok"
    assert out_path.exists()
    df = pd.read_parquet(out_path)
    assert len(df) == 3
    assert set(df["trial_index"]) == {0, 1, 2}
    assert (df["u_eff_p50"] == df["u_eff_p50"].iloc[0]).all()  # repeated per-cell column


def test_process_cell_skips_existing_checkpoint(tmp_path):
    key = _sample_cell_key()
    out_path = tmp_path / f"cell_{run_sweep.cell_id(key)}.parquet"
    job = {
        "cell_key": key,
        "cell_id": run_sweep.cell_id(key),
        "trials": 3,
        "m": run_sweep.MATRIX_M,
        "k": run_sweep.MATRIX_K,
        "out_path": str(out_path),
        "u_eff_n_elements": 2000,
    }
    first = run_sweep.process_cell(job)
    assert first["status"] == "ok"
    mtime_before = out_path.stat().st_mtime_ns
    second = run_sweep.process_cell(job)
    assert second["status"] == "skipped"
    assert out_path.stat().st_mtime_ns == mtime_before


def test_process_cell_catches_errors_without_raising(tmp_path):
    key = {**_sample_cell_key(), "round_mode": "bogus"}
    out_path = tmp_path / f"cell_{run_sweep.cell_id(key)}.parquet"
    job = {
        "cell_key": key,
        "cell_id": run_sweep.cell_id(key),
        "trials": 3,
        "m": run_sweep.MATRIX_M,
        "k": run_sweep.MATRIX_K,
        "out_path": str(out_path),
        "u_eff_n_elements": 2000,
    }
    result = run_sweep.process_cell(job)
    assert result["status"] == "error"
    assert "bogus" in result["error"] or "round_mode" in json.dumps(result["cell_key"])
    assert not out_path.exists()


# ---------------------------------------------------------------------------
# --cell-filter parsing
# ---------------------------------------------------------------------------


def test_parse_cell_filter_none_matches_everything():
    predicate = run_sweep.parse_cell_filter(None)
    assert predicate({"n": 16})


def test_parse_cell_filter_single_field():
    predicate = run_sweep.parse_cell_filter("n=16")
    assert predicate({"n": 16})
    assert not predicate({"n": 64})


def test_parse_cell_filter_or_within_field():
    predicate = run_sweep.parse_cell_filter("n=16|64")
    assert predicate({"n": 16})
    assert predicate({"n": 64})
    assert not predicate({"n": 256})


def test_parse_cell_filter_and_across_fields():
    predicate = run_sweep.parse_cell_filter("n=16,nu=1")
    assert predicate({"n": 16, "nu": 1})
    assert not predicate({"n": 16, "nu": 2})


def test_parse_cell_filter_unknown_field_raises():
    predicate = run_sweep.parse_cell_filter("bogus=1")
    with pytest.raises(ValueError, match="bogus"):
        predicate({"n": 16})


# ---------------------------------------------------------------------------
# full execute_sweep: manifest, combined output
# ---------------------------------------------------------------------------


def test_execute_sweep_writes_manifest_and_combined_parquet(tmp_path, monkeypatch):
    monkeypatch.setattr(run_sweep, "RESULTS_DIR", tmp_path)
    grid_path = _write_tiny_grid(tmp_path, heavy_trials=3, light_trials=3)
    grid_cfg = run_sweep.load_grid_config(grid_path)
    digest = run_sweep.config_digest(grid_cfg)

    manifest = run_sweep.execute_sweep(
        grid_cfg,
        digest,
        resume=False,
        cell_predicate=run_sweep.parse_cell_filter(None),
        jobs=1,
        u_eff_n_elements=1000,
    )
    assert manifest["total_cells_requested"] == 4
    assert manifest["cells_ok"] == 4
    assert manifest["cells_error"] == 0
    assert "git_commit" in manifest
    assert "start_time" in manifest and "end_time" in manifest
    assert manifest["wall_clock_seconds"] >= 0

    combined_path = tmp_path / f"sweep_{digest}.parquet"
    assert combined_path.exists()
    df = pd.read_parquet(combined_path)
    assert len(df) == 4 * 3  # 4 cells x 3 trials
    assert not df.duplicated(subset=["cell_id", "trial_index"]).any()


# ---------------------------------------------------------------------------
# ACCEPTANCE CRITERION 1: kill/resume produces an EXACTLY identical result
# (deterministic simulation: partial completion built directly, then resumed)
# ---------------------------------------------------------------------------


def test_resume_after_partial_completion_matches_uninterrupted_run(tmp_path, monkeypatch):
    grid_path = _write_tiny_grid(tmp_path, heavy_trials=4, light_trials=4)
    grid_cfg = run_sweep.load_grid_config(grid_path)
    digest = run_sweep.config_digest(grid_cfg)
    predicate = run_sweep.parse_cell_filter(None)

    # Reference: one uninterrupted run.
    reference_dir = tmp_path / "reference"
    monkeypatch.setattr(run_sweep, "RESULTS_DIR", reference_dir)
    run_sweep.execute_sweep(
        grid_cfg, digest, resume=False, cell_predicate=predicate, jobs=1, u_eff_n_elements=1000
    )
    reference_df = run_sweep.combine_cell_results(
        run_sweep.sweep_output_dir(digest) / "cells"
    )

    # Interrupted: run only the first 2 of 4 cells directly (simulating a kill
    # after those completed and before the rest started), then resume with
    # the full grid.
    interrupted_dir = tmp_path / "interrupted"
    monkeypatch.setattr(run_sweep, "RESULTS_DIR", interrupted_dir)
    all_cells = [c for c in run_sweep.iter_main_cells(grid_cfg) if predicate(c)]
    assert len(all_cells) == 4
    # iter_main_cells cycles n faster than nu (nu slowest), so nu==nu of the
    # first cell selects exactly its first two cells for this tiny grid
    # (n in {16, 64} at that one nu) -- asserted below rather than assumed.
    partial_predicate = run_sweep.parse_cell_filter(f"nu={all_cells[0]['nu']}")
    first_two = [c for c in all_cells if partial_predicate(c)]
    assert {run_sweep.cell_id(run_sweep.cell_key_from_cell(c)) for c in first_two} == {
        run_sweep.cell_id(run_sweep.cell_key_from_cell(c)) for c in all_cells[:2]
    }

    run_sweep.execute_sweep(
        grid_cfg,
        digest,
        resume=False,
        cell_predicate=partial_predicate,
        jobs=1,
        u_eff_n_elements=1000,
    )
    cells_dir = run_sweep.sweep_output_dir(digest) / "cells"
    partial_files = sorted(cells_dir.glob("cell_*.parquet"))
    assert 0 < len(partial_files) < 4

    # Resume with the FULL grid: only the missing cells should run.
    manifest = run_sweep.execute_sweep(
        grid_cfg, digest, resume=True, cell_predicate=predicate, jobs=1, u_eff_n_elements=1000
    )
    assert manifest["cells_skipped_resume"] == len(partial_files)
    assert manifest["cells_run_this_invocation"] == 4 - len(partial_files)

    resumed_df = run_sweep.combine_cell_results(cells_dir)
    assert len(resumed_df) == len(reference_df)
    assert not resumed_df.duplicated(subset=["cell_id", "trial_index"]).any()
    pd.testing.assert_frame_equal(
        resumed_df.reset_index(drop=True), reference_df.reset_index(drop=True)
    )


def test_a_stray_tmp_file_is_not_mistaken_for_a_completed_checkpoint(tmp_path):
    key = _sample_cell_key()
    cid = run_sweep.cell_id(key)
    out_path = tmp_path / f"cell_{cid}.parquet"
    tmp_leftover = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp_leftover.write_bytes(b"not a valid parquet file, simulating a torn write")

    job = {
        "cell_key": key,
        "cell_id": cid,
        "trials": 2,
        "m": run_sweep.MATRIX_M,
        "k": run_sweep.MATRIX_K,
        "out_path": str(out_path),
        "u_eff_n_elements": 1000,
    }
    result = run_sweep.process_cell(job)
    assert result["status"] == "ok"
    assert out_path.exists()
    df = pd.read_parquet(out_path)
    assert len(df) == 2
    # The stray .tmp is overwritten (same name _atomic_write_parquet uses),
    # so nothing is left behind that could confuse a later --resume.
    assert list(tmp_path.glob("*.tmp")) == []


# ---------------------------------------------------------------------------
# ACCEPTANCE CRITERION 1b: a real, literal process kill mid-sweep, resumed
# via the actual CLI -- not merely simulated in-process.
# ---------------------------------------------------------------------------


def _run_cli(
    args: list[str], results_dir: Path, timeout: float | None = None
) -> subprocess.CompletedProcess:
    env = {**os.environ, "QGEMM_RESULTS_DIR": str(results_dir)}
    return subprocess.run(
        [sys.executable, str(SCRIPT_PATH), *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


@pytest.mark.slow
def test_kill_process_partway_through_then_resume_matches_uninterrupted_run(tmp_path):
    grid_path = _write_tiny_grid(tmp_path, heavy_trials=60, light_trials=60)

    reference_dir = tmp_path / "results"
    reference_dir.mkdir()
    proc = _run_cli(
        ["--config", str(grid_path), "--jobs", "1", "--u-eff-n-elements", "150000"],
        results_dir=reference_dir,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr
    grid_cfg = run_sweep.load_grid_config(grid_path)
    digest = run_sweep.config_digest(grid_cfg)
    reference_cells_dir = reference_dir / f"sweep_{digest}" / "cells"
    reference_files = sorted(reference_cells_dir.glob("cell_*.parquet"))
    assert len(reference_files) == 4
    reference_df = pd.concat(
        [pd.read_parquet(p) for p in reference_files], ignore_index=True
    ).sort_values(["cell_id", "trial_index"]).reset_index(drop=True)

    # Fresh run, killed partway through.
    kill_dir = tmp_path / "results_kill"
    kill_dir.mkdir()
    kill_cells_dir = kill_dir / f"sweep_{digest}" / "cells"
    kill_env = {**os.environ, "QGEMM_RESULTS_DIR": str(kill_dir)}
    kill_proc = subprocess.Popen(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--config",
            str(grid_path),
            "--jobs",
            "1",
            "--u-eff-n-elements",
            "150000",
        ],
        env=kill_env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.time() + 60
        completed = 0
        while time.time() < deadline:
            if kill_cells_dir.exists():
                completed = len(list(kill_cells_dir.glob("cell_*.parquet")))
                if 0 < completed < 4:
                    break
            if kill_proc.poll() is not None:
                break
            time.sleep(0.02)
        kill_proc.kill()
        kill_proc.wait(timeout=30)
    finally:
        if kill_proc.poll() is None:
            kill_proc.kill()
            kill_proc.wait(timeout=30)

    if not (0 < completed < 4):
        pytest.skip(
            f"could not reliably catch the process mid-sweep (completed={completed} of 4); "
            "timing-dependent, not a correctness failure -- rerun."
        )

    resume_proc = _run_cli(
        ["--config", str(grid_path), "--resume", "--jobs", "1", "--u-eff-n-elements", "150000"],
        results_dir=kill_dir,
        timeout=180,
    )
    assert resume_proc.returncode == 0, resume_proc.stderr

    resumed_files = sorted(kill_cells_dir.glob("cell_*.parquet"))
    assert len(resumed_files) == 4
    resumed_df = pd.concat(
        [pd.read_parquet(p) for p in resumed_files], ignore_index=True
    ).sort_values(["cell_id", "trial_index"]).reset_index(drop=True)

    assert len(resumed_df) == len(reference_df)
    assert not resumed_df.duplicated(subset=["cell_id", "trial_index"]).any()
    pd.testing.assert_frame_equal(resumed_df, reference_df)


# ---------------------------------------------------------------------------
# ACCEPTANCE CRITERION 2: no shared/reused RNG state across parallel workers.
# ---------------------------------------------------------------------------


def test_different_trials_across_real_worker_processes_are_not_identical():
    """Trap (1): if a bug made workers share/reuse global RNG state, two
    different trials of the same cell, executed in two different real OS
    worker processes, could come out identical. They must not."""
    import multiprocessing as mp

    key = _sample_cell_key()
    jobs = [
        (key, 0, run_sweep.MATRIX_M, run_sweep.MATRIX_K),
        (key, 1, run_sweep.MATRIX_M, run_sweep.MATRIX_K),
    ]
    with mp.Pool(processes=2) as pool:
        results = pool.starmap(run_sweep.run_trial, jobs)

    assert results[0]["seed_data"] != results[1]["seed_data"]
    assert results[0]["seed_gemm"] != results[1]["seed_gemm"]
    assert results[0]["be_median"] != results[1]["be_median"]


def test_same_trial_across_real_worker_processes_is_reproducible():
    """The flip side: which worker runs a given (cell, trial) must not
    matter -- two independent processes computing the *same* trial must
    agree exactly, or the sweep would not be reproducible in isolation."""
    import multiprocessing as mp

    key = _sample_cell_key()
    jobs = [(key, 0, run_sweep.MATRIX_M, run_sweep.MATRIX_K)] * 2
    with mp.Pool(processes=2) as pool:
        results = pool.starmap(run_sweep.run_trial, jobs)

    assert results[0] == results[1]
