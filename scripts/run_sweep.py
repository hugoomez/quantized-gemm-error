"""Entry point for `make run-sweep`, plus `--dry-run` timing for the frozen
Step 3.1 confirmatory grid (configs/sweep_main.yaml).

The default (no-flag) path sweeps a grid of matmul sizes/dtypes read from a
JSON config (default: configs/default.json) -- unchanged, and unrelated to
the Step 3.1 grid. Follows the project's fixed result conventions: output is
Parquet, named sweep_{sha256(config)[:12]}.parquet, with the full config
saved alongside as sweep_{...}.config.json.

`--dry-run` is a separate, independent path: it loads configs/sweep_main.yaml
(the frozen factorial grid, see SPEC.md "Sweep grid (Step 3.1)"), times a
small representative sample of actual qgemm + backward_error calls, and
extrapolates total wall-clock time for the full grid at its adaptive trial
budget -- serial and assuming multiprocessing across every detected CPU core.
It does not run the sweep and writes no results. Full sweep execution (the
loop that actually runs all 1280 + reference cells and writes Parquet) is a
later step and is intentionally not implemented here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from qgemm.distributions import sample_gaussian
from qgemm.formats import E4M3_MAX, E5M2_MAX, int8_quantize, quantize_e4m3, quantize_e5m2
from qgemm.gemm import GemmConfig, qgemm
from qgemm.metrics import backward_error

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
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
# on top of that base cost.
FACTOR_SAMPLE_POINTS = ((256, 16), (4096, 64))


def format_hours(seconds: float) -> str:
    return f"{seconds:,.1f} s ({seconds / SECONDS_PER_HOUR:,.3f} h)"


def dry_run(grid_config_path: Path, reps: int) -> None:
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
    for cell in iter_main_cells(grid_cfg):
        per_trial = cost(
            cell["n"], cell["block_size"], cell["scale_format"], cell["round_mode"], cell["rht"]
        )
        main_seconds += per_trial * cell["trials"]
        main_trials += cell["trials"]
        main_cell_count += 1

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
    total_parallel_seconds = total_serial_seconds / n_cores

    print("\n" + "=" * 72)
    print("DRY-RUN TIMING ESTIMATE -- configs/sweep_main.yaml (Step 3.1)")
    print("=" * 72)
    print(
        f"Main factorial grid: {main_cell_count} cells "
        f"(expected 1280), {main_trials:,} trials total."
    )
    print(f"Reference configs:   {ref_cell_count} cells, {ref_trials:,} trials total.")
    print(f"Grid total:          {total_cells} cells, {total_trials:,} trials.")
    print()
    print(f"Main-grid compute:      {format_hours(main_seconds)}")
    print(f"Reference-config compute: {format_hours(ref_seconds)}")
    print(f"TOTAL, serial (1 core):     {format_hours(total_serial_seconds)}")
    print(
        f"TOTAL, parallel ({n_cores} cores detected): "
        f"{format_hours(total_parallel_seconds)}"
    )
    print(
        "(Parallel figure assumes perfect embarrassingly-parallel scaling across "
        "cells -- no process-pool or IPC overhead is modeled. Actual parallel "
        "execution is not implemented by this script; only the estimate is.)"
    )
    print()
    gate = RUNTIME_GATE_HOURS * SECONDS_PER_HOUR
    for label, seconds in (
        ("serial", total_serial_seconds),
        ("parallel", total_parallel_seconds),
    ):
        verdict = "EXCEEDS" if seconds > gate else "within"
        print(
            f"8h gate check ({label}): {seconds / SECONDS_PER_HOUR:.3f} h "
            f"{verdict} the {RUNTIME_GATE_HOURS:.0f}h budget."
        )
    print(
        "\nThis script does not decide whether to cut the grid. If either estimate "
        "above exceeds the 8h budget, that is a human call, made before anything is "
        "launched -- not made here."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIGS_DIR / "default.json",
        help="Path to a sweep config JSON file (unrelated to --dry-run).",
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
        help="Repetitions per timing sample point in --dry-run.",
    )
    args = parser.parse_args()

    if args.dry_run:
        dry_run(args.grid_config, args.dry_run_reps)
        return

    config = json.loads(args.config.read_text(encoding="utf-8"))
    rng = np.random.default_rng(config["seed"])
    df = run_sweep(config, rng)
    out_path = save_result(df, config)
    print(f"Wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
