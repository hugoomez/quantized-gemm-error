"""Entry point for `make run-sweep`, plus `--dry-run` timing for the frozen
Step 3.1 confirmatory grid (configs/sweep_main.yaml).

The default (no-flag) path sweeps a grid of matmul sizes/dtypes read from a
JSON config (default: configs/default.json) -- unchanged, and unrelated to
the Step 3.1 grid. Follows the project's fixed result conventions: output is
Parquet, named sweep_{sha256(config)[:12]}.parquet, with the full config
saved alongside as sweep_{...}.config.json.

`--dry-run` is a separate, independent path: it loads configs/sweep_main.yaml
(the frozen factorial grid, see SPEC.md "Sweep grid (Step 3.1)"), times a
small representative sample of actual qgemm + backward_error calls in a
single process to build a per-cell cost model, and *separately* runs a real
`multiprocessing.Pool` over a sample of actual grid cells to measure real
parallel throughput -- not a naive serial-time-divided-by-core-count ideal.
Both feed an extrapolate to total wall-clock time for the full grid at its
adaptive trial budget. It does not run the sweep and writes no results. Full
sweep execution (the loop that actually runs all 640 + reference cells and
writes Parquet) is a later step and is intentionally not implemented here.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
import subprocess
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from qgemm.bounds import measure_u_eff
from qgemm.distributions import sample_gaussian
from qgemm.formats import E4M3_MAX, E5M2_MAX, int8_quantize, quantize_e4m3, quantize_e5m2
from qgemm.gemm import GemmConfig, qgemm
from qgemm.metrics import backward_error

REPO_ROOT = Path(__file__).resolve().parent.parent
# Overridable via QGEMM_RESULTS_DIR (not just `cwd`, which the results path
# does not otherwise depend on) so tests can point a real `python
# scripts/run_sweep.py` subprocess at an isolated, throwaway directory
# instead of this repo's own results/ -- needed by the kill/resume
# acceptance test in tests/test_run_sweep.py, which runs the actual CLI.
RESULTS_DIR = Path(os.environ.get("QGEMM_RESULTS_DIR", REPO_ROOT / "results"))
CONFIGS_DIR = REPO_ROOT / "configs"
DEFAULT_GRID_CONFIG = CONFIGS_DIR / "sweep_main.yaml"

# Matrix shape used for dry-run timing: kept in sync with
# configs/sweep_main.yaml's matrix_shape (m=64, k=64; n varies). Not read
# from the YAML because the dry-run must work even before the grid config's
# schema is otherwise touched; the two are asserted equal at load time
# instead (see `load_grid_config`).
MATRIX_M = 64
MATRIX_K = 64

SECONDS_PER_HOUR = 3600.0
RUNTIME_GATE_HOURS = 8.0


def config_digest(config: dict) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def save_result(df: pd.DataFrame, config: dict) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    digest = config_digest(config)
    out_path = RESULTS_DIR / f"sweep_{digest}.parquet"
    df.to_parquet(out_path, engine="pyarrow")
    (RESULTS_DIR / f"sweep_{digest}.config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
    )
    return out_path


def run_sweep(config: dict, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for n in tqdm(config["sizes"], desc="sweep"):
        for dtype_name in config["dtypes"]:
            dtype = np.dtype(dtype_name)
            a = rng.standard_normal((n, n)).astype(dtype)
            b = rng.standard_normal((n, n)).astype(dtype)
            c = (a @ b).astype(np.float64)
            rows.append(
                {
                    "n": n,
                    "dtype": dtype_name,
                    "mean_abs": float(np.mean(np.abs(c))),
                    "max_abs": float(np.max(np.abs(c))),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 3.3: the real execution harness for the frozen 640-cell main grid
# (configs/sweep_main.yaml). Two failure modes this is built to close, both
# named explicitly in this project's own planning notes rather than left to
# "be careful": (1) a worker touching NumPy's global RNG state instead of an
# explicitly-derived Generator, which would make every worker draw the same
# "random" numbers; (2) buffering results until the very end, which loses a
# multi-hour run's compute the moment the process dies. See SPEC.md, "Sweep
# execution harness (Step 3.3)" for the storage schema and seeding scheme.

# The t-Student sampler is reused, not reimplemented: it lives in
# scripts/check_metric_stability.py deliberately (PREREGISTRATION.md R15 --
# the nu-axis sampler is study infrastructure to be implemented and tested on
# its own terms before it enters qgemm.distributions), and
# scripts/measure_u_eff.py already loads it exactly this way.
_CMS_PATH = REPO_ROOT / "scripts" / "check_metric_stability.py"
_cms_spec = importlib.util.spec_from_file_location("check_metric_stability", _CMS_PATH)
_cms = importlib.util.module_from_spec(_cms_spec)
_cms_spec.loader.exec_module(_cms)
sample_t = _cms.sample_t

# Step 3.2's decision: every sample generated for the production sweep is
# normalized so its MAD is 1 (SPEC.md, "Cross-distribution normalization").
# This is what actually turns that decision on for real data generation, as
# opposed to the diagnostics, which intentionally left it off.
NORMALIZE = "mad"

# Elements drawn per cell for the per-cell measure_u_eff call (requirement 4).
# This is a per-*configuration* quantity, computed once per cell regardless of
# trial count, so it is sized independently of the trial budget -- a few
# hundred thousand draws is enough for stable p50/p99 (see SPEC.md's u_eff
# sample-size discussion) without materially adding to a cell's wall-clock
# cost next to its 1000-5000 real trials.
DEFAULT_U_EFF_N_ELEMENTS = 200_000


def cell_key_from_cell(cell: dict[str, Any]) -> dict[str, Any]:
    """The canonical, hashable identity of a main-grid cell.

    Everything that affects what data a cell's trials generate and how they
    are quantized -- and nothing else. `trials` (the trial *budget*) is
    deliberately excluded: it is a property of how much a cell is sampled,
    not of which cell it is, so two runs of the same cell at different trial
    counts are still recognized as the same cell for checkpoint/resume
    purposes. `element_format`, `accum` and `normalize` are constants across
    this grid (SPEC.md's frozen Step 3.1 grid, Step 3.2's decision) but are
    included explicitly rather than left implicit, so the cell's identity
    fully determines its execution.
    """
    return {
        # Canonicalized to a string: `nu` is an int for a t-Student cell and
        # the literal string "gaussian" for the Gaussian limit, and a single
        # trial-level output table needs one consistent dtype for this column
        # (a mixed int/str object column is not a representable Arrow/Parquet
        # column). `_sampler_for` and `float(nu)` both accept the string form
        # of a numeric nu unchanged.
        "nu": str(cell["nu"]),
        "n": cell["n"],
        "block_size": cell["block_size"],
        "scale_format": cell["scale_format"],
        "use_global_scale": cell["use_global_scale"],
        "round_mode": cell["round_mode"],
        "rht": cell["rht"],
        "element_format": "e2m1",
        "accum": "exact",
        "normalize": NORMALIZE,
    }


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def cell_id(cell_key: dict[str, Any]) -> str:
    """Stable short hash identifying a cell -- the checkpoint filename key."""
    return hashlib.sha256(_canonical_json(cell_key).encode("utf-8")).hexdigest()[:12]


def _seed_from_payload(payload: str) -> int:
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def derive_seed(cell_key: dict[str, Any], trial_index: int, stream: str) -> int:
    """Deterministic seed from `(cell_key, trial_index, stream)` alone.

    No shared or global RNG state anywhere in this module: every
    `numpy.random.Generator` built for a trial traces back to a seed produced
    by this function, whose only inputs are plain, picklable, hashable
    values -- never process identity, worker id, or call order, none of which
    a worker process could rely on being consistent with another. Two workers
    computing the same `(cell_key, trial_index, stream)` therefore always
    agree exactly (a single cell is reproducible in isolation), and two
    different trials, cells, or streams never collide by construction (mixed
    through sha256 of a labeled string, not through a shared counter).
    `stream` separates independent uses within one trial (data generation vs.
    the seed handed to `GemmConfig`) the same way `GemmConfig`'s own three
    internal streams (RHT signs, SR for A, SR for B) are kept independent.
    """
    payload = f"{_canonical_json(cell_key)}|trial={trial_index}|stream={stream}"
    return _seed_from_payload(payload)


def derive_cell_seed(cell_key: dict[str, Any], stream: str) -> int:
    """Like `derive_seed`, for a per-cell (not per-trial) quantity such as u_eff."""
    payload = f"{_canonical_json(cell_key)}|cell-level|stream={stream}"
    return _seed_from_payload(payload)


def _sampler_for(nu: Any):
    """`(shape, rng) -> float64`, MAD-normalized: t-Student(nu), or Gaussian at nu='gaussian'."""
    if nu == "gaussian":
        return lambda shape, rng: sample_gaussian(shape, rng, normalize=NORMALIZE)
    nu_f = float(nu)
    return lambda shape, rng: sample_t(shape, nu_f, rng, normalize=NORMALIZE)


def run_trial(cell_key: dict[str, Any], trial_index: int, m: int, k: int) -> dict[str, Any]:
    """One trial of one main-grid cell: sample, qgemm, backward_error, summarize.

    Every RNG used here is explicitly constructed from `derive_seed` -- never
    NumPy's global state -- so the result depends only on
    `(cell_key, trial_index, m, k)`: reproducible in isolation, regardless of
    which process or worker calls it, and independent of every other trial.

    Returns a dict with the per-trial BE summary (median and p99 at minimum,
    per the sweep's storage requirement -- see SPEC.md) plus the two derived
    seeds, for a trial-granularity, tidy/long-format row. Factor columns
    (nu, n, block_size, ...) and the per-cell u_eff columns are added by the
    caller, not here, since this function has no cell-level context to add
    them from beyond what it was explicitly given.
    """
    seed_data = derive_seed(cell_key, trial_index, "data")
    seed_gemm = derive_seed(cell_key, trial_index, "gemm")
    rng = np.random.default_rng(seed_data)
    sampler = _sampler_for(cell_key["nu"])
    n = cell_key["n"]
    a = sampler((m, n), rng)
    b = sampler((n, k), rng)
    config = GemmConfig(
        block_size=cell_key["block_size"],
        scale_format=cell_key["scale_format"],
        use_global_scale=cell_key["use_global_scale"],
        element_format=cell_key["element_format"],
        round_mode=cell_key["round_mode"],
        rht=cell_key["rht"],
        accum=cell_key["accum"],
        seed=seed_gemm,
    )
    approx = qgemm(a, b, config)
    be = backward_error(a, b, approx)
    finite = be[np.isfinite(be)]
    if finite.size == 0:
        raise ValueError(f"cell {cell_key}, trial {trial_index}: no finite BE values")
    return {
        "trial_index": trial_index,
        "seed_data": seed_data,
        "seed_gemm": seed_gemm,
        "be_median": float(np.median(finite)),
        "be_p99": float(np.quantile(finite, 0.99)),
        "be_mean": float(np.mean(finite)),
        "be_max": float(np.max(finite)),
        "n_be_values": int(be.size),
        "n_be_finite": int(finite.size),
    }


def compute_cell_u_eff(cell_key: dict[str, Any], m: int, n_elements: int) -> dict[str, float]:
    """Per-cell (not per-trial) effective unit roundoff, via `qgemm.bounds.measure_u_eff`.

    A thin binding, not a reimplementation: `measure_u_eff` (from the 🧠1
    groundwork) already does the sampling, quantizing, and quantile work.
    This just builds the `GemmConfig` and sampler this cell's factors imply,
    and hands them over with a seed derived the same way every other RNG in
    this module is derived -- `derive_cell_seed`, not a trial seed, since this
    is one measurement per cell regardless of trial count.

    `config.seed` is irrelevant here (`measure_u_eff` never reads it -- the
    stochastic-rounding draw it may need comes from the explicit `rng` kwarg,
    not from `GemmConfig`'s internal streams), so it is left at its default.
    """
    config = GemmConfig(
        block_size=cell_key["block_size"],
        scale_format=cell_key["scale_format"],
        use_global_scale=cell_key["use_global_scale"],
        element_format=cell_key["element_format"],
        round_mode=cell_key["round_mode"],
        rht=cell_key["rht"],
        accum=cell_key["accum"],
    )
    sampler = _sampler_for(cell_key["nu"])
    seed = derive_cell_seed(cell_key, "u_eff")
    rng = np.random.default_rng(seed)
    result = measure_u_eff(
        config,
        sampler,
        n_elements,
        quantiles=(0.5, 0.99),
        rng=rng,
        tensor_shape=(m, cell_key["n"]),
    )
    return {"u_eff_p50": result[0.5], "u_eff_p99": result[0.99]}


def _atomic_write_parquet(df: pd.DataFrame, out_path: Path) -> None:
    """Write `df` to `out_path` atomically: write to a temp file, then rename.

    This is what makes per-cell checkpointing safe against a process dying
    mid-write (trap 2's other half, beyond just checkpointing per-cell rather
    than only at the very end): `os.replace` is an atomic rename on both
    POSIX and Windows when source and destination share a filesystem, so a
    killed process leaves either a complete `out_path` or none at all --
    never a truncated one that `--resume` could mistake for a finished cell.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    df.to_parquet(tmp_path, engine="pyarrow")
    os.replace(tmp_path, out_path)


def process_cell(job: dict[str, Any]) -> dict[str, Any]:
    """Worker: run every trial of one cell, compute its u_eff, checkpoint, report.

    Module-level, not a closure: `multiprocessing`'s `spawn` start method (the
    default on Windows, and this project has no reason to override it) pickles
    the target callable, and a closure over local state cannot be pickled --
    the same constraint noted on `_run_cell_workload` above.

    Per-cell errors are caught here, not left to propagate: a single bad cell
    raising out of a `Pool` worker would otherwise abort the whole sweep
    (requirement 6), so every exception is turned into an `"error"` status
    dict carrying the failing cell's own config instead.
    """
    cell_key = job["cell_key"]
    cid = job["cell_id"]
    out_path = Path(job["out_path"])

    if out_path.exists():
        return {"cell_id": cid, "status": "skipped"}

    try:
        trials = job["trials"]
        m, k = job["m"], job["k"]
        rows = [run_trial(cell_key, t, m, k) for t in range(trials)]
        u_eff = compute_cell_u_eff(cell_key, m, job["u_eff_n_elements"])

        df = pd.DataFrame(rows)
        for key, value in cell_key.items():
            df[key] = value
        for key, value in u_eff.items():
            df[key] = value
        df["cell_id"] = cid

        _atomic_write_parquet(df, out_path)
        return {"cell_id": cid, "status": "ok", "n_trials": trials}
    except Exception as exc:  # noqa: BLE001 -- must not abort the sweep; see docstring
        return {
            "cell_id": cid,
            "status": "error",
            "cell_key": cell_key,
            "error": f"{type(exc).__name__}: {exc}",
        }


def parse_cell_filter(spec: str | None) -> Callable[[dict[str, Any]], bool]:
    """Parse `--cell-filter`: `field=value[|value2...][,field2=value3...]`.

    Comma-separated clauses are ANDed together; `|`-separated values within
    one clause are ORed. E.g. `"n=16|64,nu=1"` selects cells with
    `n in {16, 64}` and `nu == 1`. `field` is matched against `str(cell[field])`
    so both numeric and string factor values (like `nu`'s `"gaussian"`) work
    without special-casing.
    """
    if not spec:
        return lambda cell: True

    constraints: dict[str, set[str]] = {}
    for clause in spec.split(","):
        field, sep, values = clause.partition("=")
        if not sep or not field.strip() or not values.strip():
            raise ValueError(f"invalid --cell-filter clause {clause!r}; expected field=value")
        constraints[field.strip()] = {v.strip() for v in values.split("|")}

    def predicate(cell: dict[str, Any]) -> bool:
        for field, allowed in constraints.items():
            if field not in cell:
                raise ValueError(
                    f"--cell-filter references unknown field {field!r}; "
                    f"known fields: {sorted(cell)}"
                )
            if str(cell[field]) not in allowed:
                return False
        return True

    return predicate


def sweep_output_dir(grid_digest: str) -> Path:
    return RESULTS_DIR / f"sweep_{grid_digest}"


def combine_cell_results(cells_dir: Path) -> pd.DataFrame:
    """Concatenate every completed cell's checkpoint into one tidy trial-level table.

    Sorted by `(cell_id, trial_index)` so the combined result does not depend
    on filesystem iteration order or on which cells happened to run in which
    invocation -- required for the kill/resume acceptance test's "exactly
    identical, not just similar" bar.
    """
    paths = sorted(cells_dir.glob("cell_*.parquet")) if cells_dir.exists() else []
    if not paths:
        return pd.DataFrame()
    combined = pd.concat((pd.read_parquet(p, engine="pyarrow") for p in paths), ignore_index=True)
    return combined.sort_values(["cell_id", "trial_index"]).reset_index(drop=True)


def _git_commit_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return result.stdout.strip()
    except Exception:  # noqa: BLE001 -- manifest recording must not hard-fail the sweep
        return None


def _log_error(errors_path: Path, result: dict[str, Any]) -> None:
    errors_path.parent.mkdir(parents=True, exist_ok=True)
    with errors_path.open("a", encoding="utf-8") as fh:
        fh.write(_canonical_json({"time": datetime.now(UTC).isoformat(), **result}) + "\n")


def execute_sweep(
    grid_cfg: dict[str, Any],
    grid_digest: str,
    *,
    resume: bool,
    cell_predicate: Callable[[dict[str, Any]], bool],
    jobs: int,
    u_eff_n_elements: int,
) -> dict[str, Any]:
    """Run (or resume) the real main-grid sweep and return its manifest.

    Per-cell checkpointing (requirement 1): each cell's trials and u_eff are
    computed and written to `cells/cell_{cell_id}.parquet` as soon as that
    cell finishes, atomically. `resume=True` skips any cell whose checkpoint
    already exists rather than recomputing it; `resume=False` clears any
    stale checkpoint for a cell it is about to (re)run first, so a fresh
    (non-resumed) invocation never silently mixes in leftover results from an
    earlier, differently-configured run of the same grid digest.

    Parallelism is across cells (requirement 3), via `multiprocessing.Pool`
    when `jobs > 1`. At `jobs <= 1` cells run sequentially in this process
    with no `Pool` at all -- both a reasonable default for a small subset and
    what keeps a single OS process (and hence a single kill target) doing all
    the work, which is what the kill/resume test relies on.
    """
    out_dir = sweep_output_dir(grid_digest)
    cells_dir = out_dir / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)
    errors_path = out_dir / "errors.log"

    all_cells = [c for c in iter_main_cells(grid_cfg) if cell_predicate(c)]
    jobs_list: list[dict[str, Any]] = []
    for cell in all_cells:
        key = cell_key_from_cell(cell)
        cid = cell_id(key)
        out_path = cells_dir / f"cell_{cid}.parquet"
        if resume and out_path.exists():
            continue
        jobs_list.append(
            {
                "cell_key": key,
                "cell_id": cid,
                "trials": cell["trials"],
                "m": MATRIX_M,
                "k": MATRIX_K,
                "out_path": str(out_path),
                "u_eff_n_elements": u_eff_n_elements,
            }
        )
    if not resume:
        for job in jobs_list:
            Path(job["out_path"]).unlink(missing_ok=True)

    started = datetime.now(UTC)
    started_perf = time.perf_counter()
    results: list[dict[str, Any]] = []
    if jobs_list:
        if jobs <= 1:
            iterator = (process_cell(job) for job in jobs_list)
            for result in tqdm(iterator, total=len(jobs_list), desc="cells"):
                results.append(result)
                if result["status"] == "error":
                    _log_error(errors_path, result)
        else:
            with mp.Pool(processes=jobs) as pool:
                for result in tqdm(
                    pool.imap_unordered(process_cell, jobs_list),
                    total=len(jobs_list),
                    desc="cells",
                ):
                    results.append(result)
                    if result["status"] == "error":
                        _log_error(errors_path, result)
    ended = datetime.now(UTC)
    wall_seconds = time.perf_counter() - started_perf

    combined = combine_cell_results(cells_dir)
    if not combined.empty:
        combined_path = RESULTS_DIR / f"sweep_{grid_digest}.parquet"
        _atomic_write_parquet(combined, combined_path)
        (RESULTS_DIR / f"sweep_{grid_digest}.config.json").write_text(
            json.dumps(grid_cfg, indent=2, sort_keys=True), encoding="utf-8"
        )

    environment_md_path = REPO_ROOT / "ENVIRONMENT.md"
    manifest = {
        "grid_config_hash": grid_digest,
        "git_commit": _git_commit_hash(),
        "start_time": started.isoformat(),
        "end_time": ended.isoformat(),
        "wall_clock_seconds": wall_seconds,
        "total_cells_requested": len(all_cells),
        "cells_run_this_invocation": len(jobs_list),
        "cells_skipped_resume": len(all_cells) - len(jobs_list),
        "cells_ok": sum(1 for r in results if r["status"] == "ok"),
        "cells_error": sum(1 for r in results if r["status"] == "error"),
        "resume": resume,
        "jobs": jobs,
        "u_eff_n_elements": u_eff_n_elements,
        "environment_md": environment_md_path.read_text(encoding="utf-8")
        if environment_md_path.exists()
        else None,
    }
    manifest_path = out_dir / f"manifest_{started.strftime('%Y%m%dT%H%M%S%f')}Z.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


# ---------------------------------------------------------------------------
# --dry-run: timing estimate for the frozen Step 3.1 grid (sweep_main.yaml).
# Does not run the sweep; writes nothing to results/.

def load_grid_config(path: Path) -> dict[str, Any]:
    grid_cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    shape = grid_cfg["matrix_shape"]
    if shape["m"] != MATRIX_M or shape["k"] != MATRIX_K:
        raise ValueError(
            f"{path} matrix_shape {shape} does not match this script's "
            f"MATRIX_M={MATRIX_M}, MATRIX_K={MATRIX_K}"
        )
    return grid_cfg


def trial_count(grid_cfg: dict[str, Any], nu: Any) -> int:
    tb = grid_cfg["trial_budget"]
    if nu in tb["heavy_tail_nu"]:
        return int(tb["heavy_tail_trials"])
    if nu in tb["light_tail_nu"]:
        return int(tb["light_tail_trials"])
    raise ValueError(f"nu {nu!r} is in neither trial_budget nu list")


def iter_main_cells(grid_cfg: dict[str, Any]):
    """Yield one dict per cell of the 1280-cell main factorial grid.

    `use_global_scale` is derived from `scale_format` here, per
    `use_global_scale_by_scale_format` -- not an independent loop.
    """
    g = grid_cfg["main_grid"]
    use_global = grid_cfg["use_global_scale_by_scale_format"]
    for nu in g["nu"]:
        trials = trial_count(grid_cfg, nu)
        for n in g["n"]:
            for block_size in g["block_size"]:
                for scale_format in g["scale_format"]:
                    for round_mode in g["round_mode"]:
                        for rht in g["rht"]:
                            yield {
                                "nu": nu,
                                "n": n,
                                "block_size": block_size,
                                "scale_format": scale_format,
                                "use_global_scale": use_global[scale_format],
                                "round_mode": round_mode,
                                "rht": rht,
                                "trials": trials,
                            }


def iter_reference_cells(grid_cfg: dict[str, Any]):
    """Yield one dict per cell of the reference (per-tensor) configs."""
    r = grid_cfg["reference_configs"]
    for fmt in r["formats"]:
        for nu in r["nu"]:
            trials = trial_count(grid_cfg, nu)
            for n in r["n"]:
                yield {"format": fmt, "nu": nu, "n": n, "trials": trials}


def _time_calls(fn, reps: int) -> float:
    """Average wall-clock seconds per call to `fn`, over `reps` reps.

    One untimed warm-up call first, so first-call allocation overhead (array
    buffers, etc.) doesn't skew the estimate.
    """
    fn()
    started = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - started) / reps


def _main_trial(
    n: int, block_size: int, scale_format: str, use_global_scale: bool, round_mode: str, rht: bool
):
    """One real (sample -> qgemm -> backward_error) call at these cell parameters.

    The distribution the operands are drawn from does not affect run time --
    `qgemm`'s cost depends on array shape and quantizer parameters, not on
    the values inside the array (the only value-dependent branches in
    `quantize_blocked` are degenerate-block cases like `amax == 0`, which a
    continuous random draw essentially never hits) -- so `sample_gaussian` is
    used for every timing point regardless of which `nu` it stands in for.
    """
    rng = np.random.default_rng(0)
    a = sample_gaussian((MATRIX_M, n), rng)
    b = sample_gaussian((n, MATRIX_K), rng)
    config = GemmConfig(
        block_size=block_size,
        scale_format=scale_format,
        use_global_scale=use_global_scale,
        round_mode=round_mode,
        rht=rht,
        accum="exact",
        seed=0,
    )

    def call() -> np.ndarray:
        approx = qgemm(a, b, config)
        return backward_error(a, b, approx)

    return call


def fp8_e4m3_per_tensor(x: np.ndarray) -> np.ndarray:
    """Per-tensor amax/448 FP8-E4M3, matching scripts/sanity_reproduce_2408.py."""
    amax = float(np.max(np.abs(x)))
    if amax == 0.0:
        return x.copy()
    scale = amax / E4M3_MAX
    return quantize_e4m3(x / scale) * scale


def fp8_e5m2_per_tensor(x: np.ndarray) -> np.ndarray:
    """Per-tensor amax/57344 FP8-E5M2, mirroring `fp8_e4m3_per_tensor`."""
    amax = float(np.max(np.abs(x)))
    if amax == 0.0:
        return x.copy()
    scale = amax / E5M2_MAX
    return quantize_e5m2(x / scale) * scale


REFERENCE_QUANTIZERS = {
    "fp8_e4m3_per_tensor": fp8_e4m3_per_tensor,
    "fp8_e5m2_per_tensor": fp8_e5m2_per_tensor,
    "int8_per_tensor": int8_quantize,
}


def _reference_trial(fmt: str, n: int):
    """One real (sample -> quantize -> matmul -> backward_error) call for a
    reference per-tensor format."""
    rng = np.random.default_rng(0)
    a = sample_gaussian((MATRIX_M, n), rng)
    b = sample_gaussian((n, MATRIX_K), rng)
    quantize = REFERENCE_QUANTIZERS[fmt]

    def call() -> np.ndarray:
        approx = quantize(a) @ quantize(b)
        return backward_error(a, b, approx)

    return call


# Points, in (n, block_size), where the RHT / stochastic-rounding / E4M3
# overhead factors are measured. Averaged over more than one point for
# robustness, rather than measured at every (n, block_size) combination --
# the base grid below already covers how cost scales with n and block_size
# on its own; these three factors are treated as roughly multiplicative
# on top of that base cost. block_size=64 was dropped from the grid (Step
# 3.1 cut, see sweep_main.yaml); these points use the two block sizes that
# remain, 16 and 32.
FACTOR_SAMPLE_POINTS = ((256, 16), (4096, 32))


def format_hours(seconds: float) -> str:
    return f"{seconds:,.1f} s ({seconds / SECONDS_PER_HOUR:,.3f} h)"


def _run_cell_workload(args: tuple) -> float:
    """Worker: run `n_trials` real trials of one grid cell; return elapsed seconds.

    Executed in its own OS process by `multiprocessing.Pool` -- this
    project's convention for parallelizing across cells: one process per
    cell, each with its own RNG state. Every trial gets a distinct seed
    derived from `(cell_index, trial)` via `SeedSequence`, so no RNG state is
    shared across trials or across cells, matching the repo's explicit-
    Generator rule (`qgemm.gemm.GemmConfig`: never NumPy's global RNG).

    Must be a module-level function, not a closure: `multiprocessing` on
    Windows uses the `spawn` start method, which pickles the target
    callable, and a closure over local state cannot be pickled.
    """
    cell_index, n, block_size, scale_format, use_global_scale, round_mode, rht, n_trials = args
    started = time.perf_counter()
    for trial in range(n_trials):
        seed = int(np.random.SeedSequence([cell_index, trial]).generate_state(1)[0])
        rng = np.random.default_rng(seed)
        a = sample_gaussian((MATRIX_M, n), rng)
        b = sample_gaussian((n, MATRIX_K), rng)
        config = GemmConfig(
            block_size=block_size,
            scale_format=scale_format,
            use_global_scale=use_global_scale,
            round_mode=round_mode,
            rht=rht,
            accum="exact",
            seed=seed,
        )
        approx = qgemm(a, b, config)
        backward_error(a, b, approx)
    return time.perf_counter() - started


def cell_is_executable(cell: dict[str, Any]) -> bool:
    """False for a cell `qgemm` would raise on outright.

    Historical note: before the 2026-08-20 `apply_rht` fix
    (`qgemm.transforms._effective_block_size`), this returned False for any
    `rht=True` cell with `n < block_size` -- discovered empirically while
    building this script's real-multiprocessing sample, when `qgemm()`
    raised on a sampled cell instead of just being timed. `apply_rht` now
    falls back to a single block sized at `n` in that case (see its "Tail
    policy"), so `n < block_size` no longer raises. The only case that
    still does is a genuine *trailing* partial block -- `n > block_size`
    and not a whole multiple of it -- which this grid never produces, since
    every `n` and `block_size` it sweeps is a power of two and `n >=
    block_size` between two powers of two is always an exact multiple. This
    function is expected to always return True for `sweep_main.yaml`'s grid
    now; it is kept as a real check rather than hardcoded to True so it
    stays correct if the grid ever grows an (n, block_size) pair outside
    that pattern.
    """
    if not cell["rht"]:
        return True
    n, block_size = cell["n"], cell["block_size"]
    if n < block_size:
        return True  # apply_rht falls back to one block sized at n
    return n % block_size == 0


def _select_sample_cells(grid_cfg: dict[str, Any], sample_size: int) -> list[dict[str, Any]]:
    """`sample_size` real, executable cells, evenly spaced through the grid's
    iteration order.

    Even spacing (not random sampling) over `iter_main_cells`'s output --
    which cycles rht fastest, then round_mode, scale_format, block_size, n,
    nu slowest -- gives deterministic, reproducible coverage across every
    factor, including nu, rather than risking a sample skewed toward one
    corner of the grid. Non-executable cells (`cell_is_executable` is False)
    are excluded before sampling, since running one would crash the
    measurement rather than time it; see `cell_is_executable`.
    """
    cells = [c for c in iter_main_cells(grid_cfg) if cell_is_executable(c)]
    idx = sorted({int(i) for i in np.linspace(0, len(cells) - 1, num=sample_size).round()})
    return [cells[i] for i in idx]


def measure_real_parallel(
    grid_cfg: dict[str, Any],
    base_times: dict[tuple[int, int], float],
    rht_factor: float,
    sr_factor: float,
    e4m3_factor: float,
    n_cores: int,
    sample_size: int,
    trials_per_cell: int,
) -> dict[str, float]:
    """Actually run a sample of real grid cells in parallel and time the wall-clock.

    Every sampled cell runs the *same* fixed `trials_per_cell` trials (not
    its real 1000/5000-trial budget) -- this measures real per-trial
    parallel throughput (process-pool startup, IPC, memory-bandwidth
    contention across `n_cores` simultaneous NumPy workers) without the
    wall-clock cost of running any single cell's full budget. It does NOT
    capture load imbalance from cells' unequal real trial counts (a 5000-
    trial cell scheduled alongside 1000-trial cells is a straggler in a real
    run); that is a real, separate source of overhead this measurement does
    not include, noted here rather than silently absorbed into the estimate.

    Returns measured wall-clock time for the sample alongside the
    already-measured single-process cost model's *prediction* for that same
    sample workload under naive ideal scaling, so the two can be compared
    directly -- this is the "measured vs. ideal-scaling prediction" the
    dry-run report shows.
    """
    sample = _select_sample_cells(grid_cfg, sample_size)

    def predicted_cost(cell: dict[str, Any]) -> float:
        c = base_times[(cell["n"], cell["block_size"])]
        if cell["scale_format"] == "e4m3":
            c *= e4m3_factor
        if cell["round_mode"] == "sr":
            c *= sr_factor
        if cell["rht"]:
            c *= rht_factor
        return c

    ideal_serial_sample_seconds = sum(predicted_cost(c) for c in sample) * trials_per_cell
    ideal_parallel_sample_seconds = ideal_serial_sample_seconds / n_cores

    jobs = [
        (
            i,
            cell["n"],
            cell["block_size"],
            cell["scale_format"],
            cell["use_global_scale"],
            cell["round_mode"],
            cell["rht"],
            trials_per_cell,
        )
        for i, cell in enumerate(sample)
    ]

    started = time.perf_counter()
    with mp.Pool(processes=n_cores) as pool:
        pool.map(_run_cell_workload, jobs)
    measured_wall_seconds = time.perf_counter() - started

    sample_trials = len(sample) * trials_per_cell
    return {
        "sample_size": len(sample),
        "trials_per_cell": trials_per_cell,
        "sample_trials": sample_trials,
        "ideal_serial_sample_seconds": ideal_serial_sample_seconds,
        "ideal_parallel_sample_seconds": ideal_parallel_sample_seconds,
        "measured_wall_seconds": measured_wall_seconds,
        # efficiency = 1.0 means the real Pool matched the naive ideal-scaling
        # prediction exactly; < 1.0 means real overhead made it slower.
        "efficiency": ideal_parallel_sample_seconds / measured_wall_seconds,
        "measured_trials_per_second": sample_trials / measured_wall_seconds,
    }


def dry_run(
    grid_config_path: Path,
    reps: int,
    parallel_sample_cells: int,
    parallel_sample_trials: int,
) -> None:
    grid_cfg = load_grid_config(grid_config_path)
    main_grid = grid_cfg["main_grid"]
    ref_cfg = grid_cfg["reference_configs"]
    all_n = list(main_grid["n"])
    all_block = list(main_grid["block_size"])

    print(
        f"Loaded {grid_config_path.name}. Timing base grid: "
        f"{len(all_n)} n values x {len(all_block)} block sizes = "
        f"{len(all_n) * len(all_block)} points, {reps} reps each "
        "(rht=False, round_mode=rtne, scale_format=e8m0)..."
    )
    base_times: dict[tuple[int, int], float] = {}
    for n in all_n:
        for block_size in all_block:
            call = _main_trial(n, block_size, "e8m0", False, "rtne", False)
            base_times[(n, block_size)] = _time_calls(call, reps)

    print(
        f"Timing rht / stochastic-rounding / e4m3-global-scale overhead at "
        f"{len(FACTOR_SAMPLE_POINTS)} representative points..."
    )
    rht_ratios, sr_ratios, e4m3_ratios = [], [], []
    for n, block_size in FACTOR_SAMPLE_POINTS:
        base = base_times[(n, block_size)]
        t_rht = _time_calls(_main_trial(n, block_size, "e8m0", False, "rtne", True), reps)
        t_sr = _time_calls(_main_trial(n, block_size, "e8m0", False, "sr", False), reps)
        t_e4m3 = _time_calls(_main_trial(n, block_size, "e4m3", True, "rtne", False), reps)
        rht_ratios.append(t_rht / base)
        sr_ratios.append(t_sr / base)
        e4m3_ratios.append(t_e4m3 / base)
    rht_factor = float(np.mean(rht_ratios))
    sr_factor = float(np.mean(sr_ratios))
    e4m3_factor = float(np.mean(e4m3_ratios))
    print(
        f"  overhead factors (multiplicative on the base n/block_size cost): "
        f"rht={rht_factor:.3f}x  sr={sr_factor:.3f}x  e4m3+global={e4m3_factor:.3f}x"
    )

    def cost(n: int, block_size: int, scale_format: str, round_mode: str, rht: bool) -> float:
        c = base_times[(n, block_size)]
        if scale_format == "e4m3":
            c *= e4m3_factor
        if round_mode == "sr":
            c *= sr_factor
        if rht:
            c *= rht_factor
        return c

    main_seconds = 0.0
    main_trials = 0
    main_cell_count = 0
    non_executable_cells = 0
    non_executable_trials = 0
    for cell in iter_main_cells(grid_cfg):
        per_trial = cost(
            cell["n"], cell["block_size"], cell["scale_format"], cell["round_mode"], cell["rht"]
        )
        main_seconds += per_trial * cell["trials"]
        main_trials += cell["trials"]
        main_cell_count += 1
        if not cell_is_executable(cell):
            non_executable_cells += 1
            non_executable_trials += cell["trials"]

    if non_executable_cells:
        print(
            f"\n*** FINDING (not a decision): {non_executable_cells} of {main_cell_count} "
            f"cells ({non_executable_trials:,} trials) are currently NOT EXECUTABLE -- "
            "qgemm() raises ValueError on them. Excluded from the real-multiprocessing "
            "sample below (they would crash it); their time is still counted in the "
            "cost-model totals below since the cost model is arithmetic, not an actual "
            "qgemm() call. Whether/how to fix this is not decided here -- see SPEC.md.\n"
        )
    else:
        print(
            f"All {main_cell_count} main-grid cells are executable "
            "(the pre-existing rht=True/n<block_size crash -- see SPEC.md's 'Sweep grid "
            "(Step 3.1)' -- was fixed in qgemm.transforms; nothing excluded from the "
            "real-multiprocessing sample below)."
        )

    print(
        f"Timing reference-config points: {len(ref_cfg['formats'])} formats x "
        f"{len(ref_cfg['n'])} n = {len(ref_cfg['formats']) * len(ref_cfg['n'])} points, "
        f"{reps} reps each..."
    )
    ref_times: dict[tuple[str, int], float] = {}
    for fmt in ref_cfg["formats"]:
        for n in ref_cfg["n"]:
            ref_times[(fmt, n)] = _time_calls(_reference_trial(fmt, n), reps)

    ref_seconds = 0.0
    ref_trials = 0
    ref_cell_count = 0
    for cell in iter_reference_cells(grid_cfg):
        ref_seconds += ref_times[(cell["format"], cell["n"])] * cell["trials"]
        ref_trials += cell["trials"]
        ref_cell_count += 1

    total_cells = main_cell_count + ref_cell_count
    total_trials = main_trials + ref_trials
    total_serial_seconds = main_seconds + ref_seconds
    n_cores = os.cpu_count() or 1
    ideal_parallel_seconds = total_serial_seconds / n_cores

    print(
        f"\nRunning REAL multiprocessing.Pool(processes={n_cores}) over "
        f"{parallel_sample_cells} sampled real cells x {parallel_sample_trials} "
        f"trials each ({parallel_sample_cells * parallel_sample_trials:,} trials) "
        "to measure actual parallel throughput..."
    )
    measured = measure_real_parallel(
        grid_cfg,
        base_times,
        rht_factor,
        sr_factor,
        e4m3_factor,
        n_cores,
        parallel_sample_cells,
        parallel_sample_trials,
    )
    efficiency = measured["efficiency"]
    # The measured/ideal ratio on the sample, applied as a correction to the
    # full-grid ideal-scaling estimate. Chosen over extrapolating absolute
    # throughput from the sample alone because the per-cell cost model above
    # was built from a much larger, more representative timing sweep (the
    # full n x block_size grid) than 20-40 sampled cells could give on their
    # own; the sample's job is to measure *overhead relative to that model*,
    # not to re-derive the cost model from scratch.
    measured_parallel_seconds = ideal_parallel_seconds / efficiency
    # Cross-check: extrapolating directly from the sample's own measured
    # trials/second, ignoring the cost model entirely.
    cross_check_parallel_seconds = total_trials / measured["measured_trials_per_second"]

    print("\n" + "=" * 72)
    print("DRY-RUN TIMING ESTIMATE -- configs/sweep_main.yaml (Step 3.1)")
    print("=" * 72)
    print(
        f"Main factorial grid: {main_cell_count} cells "
        f"(expected 640), {main_trials:,} trials total."
    )
    print(f"Reference configs:   {ref_cell_count} cells, {ref_trials:,} trials total.")
    print(f"Grid total:          {total_cells} cells, {total_trials:,} trials.")
    print()
    print(f"Main-grid compute (single-process cost model): {format_hours(main_seconds)}")
    print(f"Reference-config compute (single-process cost model): {format_hours(ref_seconds)}")
    print(
        f"TOTAL, serial (1 core, single-process cost model): {format_hours(total_serial_seconds)}"
    )
    print(
        f"TOTAL, parallel, IDEAL SCALING ({n_cores} cores, naive "
        f"serial-time / n_cores): {format_hours(ideal_parallel_seconds)}  "
        "[reference figure only -- see measured figure below]"
    )
    print()
    print(
        f"REAL multiprocessing sample: {measured['sample_size']} cells x "
        f"{measured['trials_per_cell']} trials = {measured['sample_trials']:,} trials, "
        f"Pool(processes={n_cores})."
    )
    print(
        "  ideal-scaling prediction for this sample: "
        f"{format_hours(measured['ideal_parallel_sample_seconds'])}"
    )
    print(
        "  ACTUALLY MEASURED wall-clock for this sample: "
        f"{format_hours(measured['measured_wall_seconds'])}"
    )
    print(
        f"  measured parallel efficiency vs. ideal scaling: {efficiency:.3f} "
        f"({efficiency * 100:.1f}% -- 100% would mean the real Pool matched the "
        "naive serial-time/n_cores prediction exactly; below 100% is real overhead: "
        "process startup, IPC, memory-bandwidth contention across simultaneous "
        f"NumPy workers. NOT measured: load imbalance from cells' real, unequal "
        "1000-vs-5000-trial budgets, since every sampled cell here ran the same "
        f"{measured['trials_per_cell']} trials -- so this efficiency figure is "
        "likely still a mild overestimate of real production throughput.)"
    )
    print()
    print(
        f"TOTAL, parallel, MEASURED (ideal-scaling estimate corrected by measured "
        f"efficiency): {format_hours(measured_parallel_seconds)}"
    )
    print(
        f"  cross-check (extrapolated directly from the sample's own measured "
        f"trials/sec, ignoring the cost model): {format_hours(cross_check_parallel_seconds)}"
    )
    print()
    gate = RUNTIME_GATE_HOURS * SECONDS_PER_HOUR
    for label, seconds in (
        ("serial (single-process cost model)", total_serial_seconds),
        ("parallel, ideal scaling (reference only)", ideal_parallel_seconds),
        ("parallel, MEASURED (headline figure)", measured_parallel_seconds),
    ):
        verdict = "EXCEEDS" if seconds > gate else "within"
        print(
            f"8h gate check ({label}): {seconds / SECONDS_PER_HOUR:.3f} h "
            f"{verdict} the {RUNTIME_GATE_HOURS:.0f}h budget."
        )
    print(
        "\nThis script does not decide whether to cut the grid. If the MEASURED "
        "parallel estimate above exceeds the 8h budget, that is a human call, made "
        "before anything is launched -- not made here."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIGS_DIR / "default.json",
        help=(
            "Path to a sweep config. A .json file runs the legacy toy sweep "
            "(unrelated to the Step 3.3 grid harness). A .yaml/.yml file "
            "(e.g. configs/sweep_main.yaml) runs the real, checkpointed "
            "main-grid execution harness -- see --resume, --cell-filter, --jobs."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Estimate total wall-clock time for the frozen Step 3.1 grid "
            "(--grid-config) without running it. Writes no results."
        ),
    )
    parser.add_argument(
        "--grid-config",
        type=Path,
        default=DEFAULT_GRID_CONFIG,
        help="Path to the frozen factorial sweep grid YAML (for --dry-run).",
    )
    parser.add_argument(
        "--dry-run-reps",
        type=int,
        default=20,
        help="Repetitions per single-process timing sample point in --dry-run.",
    )
    parser.add_argument(
        "--dry-run-parallel-sample-cells",
        type=int,
        default=32,
        help="Real grid cells sampled for the --dry-run real-multiprocessing measurement.",
    )
    parser.add_argument(
        "--dry-run-parallel-sample-trials",
        type=int,
        default=150,
        help="Trials run per sampled cell in the --dry-run real-multiprocessing measurement.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip any main-grid cell whose checkpoint parquet already exists.",
    )
    parser.add_argument(
        "--cell-filter",
        type=str,
        default=None,
        help=(
            "Restrict to matching main-grid cells: 'field=value[|value2],...' "
            "(comma = AND, | = OR within a field). E.g. 'n=16|64,nu=1'."
        ),
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=os.cpu_count() or 1,
        help="Parallel worker processes for the real grid harness (<=1 runs sequentially).",
    )
    parser.add_argument(
        "--u-eff-n-elements",
        type=int,
        default=DEFAULT_U_EFF_N_ELEMENTS,
        help="Elements drawn per cell for the per-cell measure_u_eff call.",
    )
    args = parser.parse_args()

    if args.dry_run:
        dry_run(
            args.grid_config,
            args.dry_run_reps,
            args.dry_run_parallel_sample_cells,
            args.dry_run_parallel_sample_trials,
        )
        return

    if args.config.suffix.lower() in (".yaml", ".yml"):
        grid_cfg = load_grid_config(args.config)
        grid_digest = config_digest(grid_cfg)
        predicate = parse_cell_filter(args.cell_filter)
        manifest = execute_sweep(
            grid_cfg,
            grid_digest,
            resume=args.resume,
            cell_predicate=predicate,
            jobs=args.jobs,
            u_eff_n_elements=args.u_eff_n_elements,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return

    config = json.loads(args.config.read_text(encoding="utf-8"))
    rng = np.random.default_rng(config["seed"])
    df = run_sweep(config, rng)
    out_path = save_result(df, config)
    print(f"Wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
