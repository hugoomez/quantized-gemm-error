"""Step 4.1: fit the empirical n-scaling exponent of median(BE) for every
configuration in the frozen 640-cell production sweep, and compare it
against the theoretical -0.5 exponent derived in SPEC.md ("Theoretical bound
definition (\U0001f9e01) -- RESOLVED").

This is REAL CONFIRMATORY ANALYSIS on the production sweep
(`results/sweep_e945b87a2395`), not a diagnostic probe -- its output is kept
under `results/analysis/`, separate from every earlier diagnostic script's
`results/diagnostics/` location.

Why -0.5, and why this is the corrected regime
------------------------------------------------
Under this project's primary route (`accum="exact"`), there is no repeated
rounding during accumulation, so the classical growing bounds (gamma_n,
sqrt(n)*u, slope +0.5/+1.0) do not apply. SPEC.md's resolved derivation gives

    cota(n, format, block, nu) = c * u_eff(format, block, nu) / sqrt(n)

with `c` FIXED at 1 for all confirmatory analysis (PREREGISTRATION.md sec
3.2's anti-circularity constraint) -- so the theoretically expected regime is
median(BE) DECAYING like n^-0.5, not growing. This was already confirmed at
nu=30 in the Step 2.3 n-scaling probe (slopes ~= -0.4978 / -0.4966).

What this script does, per one of the 128 configurations (nu x block_size x
scale_format x round_mode x rht -- the full factorial, 8x2x2x2x2)
------------------------------------------------------------------------
1. Aggregate median(BE) per n (5 points: n in {16, 64, 256, 1024, 4096}) from
   the sweep's trial-level `be_median` column -- the median, over trials, of
   each trial's own median BE over its 64x64 output (matching the convention
   `scripts/probe_n_scaling.py` already established).
2. Fit log10(median BE) vs log10(n) by ordinary least squares -- the point
   estimate of the empirical slope p_hat.
3. Bootstrap the slope's CI by resampling TRIALS independently within each n
   (not the 5 aggregate points), recomputing the 5 per-n medians, and
   refitting the slope, >=2000 times, via
   `qgemm.stats.bootstrap_loglog_slope_ci` -- a generalization of
   `qgemm.stats.bootstrap_ci` for a statistic (a fitted slope) that spans
   several independent, unequally-sized trial samples at once, which
   `bootstrap_ci`'s single-array API cannot express on its own (see that
   function's docstring). Each configuration gets its own deterministically
   seeded `numpy.random.Generator` -- no global RNG state.
4. Fit and report the multiplicative constant c_hat from
   `error = c_hat * n^p_hat * u_eff`, using p_hat from step 2 and this
   CONFIGURATION'S OWN u_eff (the `u_eff_p50` column already measured and
   stored per cell by the sweep harness, `scripts/run_sweep.py`'s
   `compute_cell_u_eff` -- not recomputed here). c_hat IS DESCRIPTIVE /
   EXPLORATORY ONLY (PREREGISTRATION.md sec 3.2): it must never be read as a
   candidate value for the confirmatory bound's c, which stays fixed at 1
   everywhere else in this project. See `fit_c_hat` below, where this
   constraint is restated at the point c_hat is computed.

Interpretation buckets (reported per configuration, not just the raw slope)
-----------------------------------------------------------------------------
* consistent_with_-0.5: the slope's 95% bootstrap CI contains -0.5 --
  consistent with the u_eff/sqrt(n) cancellation regime holding.
* weakening (slope CI entirely above -0.5, i.e. less negative / closer to 0
  or positive): error decays reliably SLOWER than the cancellation regime
  predicts. Configurations landing here are candidates for where nu* (Step
  4.2, not run by this script) will be found.
* faster_than_predicted (slope CI entirely below -0.5): error decays faster
  than predicted. Not a bound concern, but reported rather than silently
  folded into "consistent".
Any configuration whose slope CI's lower bound exceeds 0 (error reliably
GROWING with n) is additionally flagged POSITIVE_SLOPE -- a striking result
given exact accumulation, per the task that produced this script.

Usage
-----
    python scripts/fit_n_scaling.py [--sweep-dir results/sweep_e945b87a2395]
        [--n-resamples 2000] [--seed 0]

Output
------
`results/analysis/n_scaling_fits/n_scaling_fits_{digest}_table.parquet` (128
rows, one per configuration) and `..._table.csv` (the same, human-readable),
plus `..._detail.parquet` (640 rows, one per configuration x n, for
auditability), `...config.json`, and `..._statement.txt`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from qgemm.stats import bootstrap_loglog_slope_ci

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SWEEP_DIR = REPO_ROOT / "results" / "sweep_e945b87a2395"
OUT_DIR = REPO_ROOT / "results" / "analysis" / "n_scaling_fits"

# The five swept configuration factors (excluding `n`, which is the
# regression axis) -- 8 x 2 x 2 x 2 x 2 = 128 configurations, per
# configs/sweep_main.yaml's frozen grid.
CONFIG_FACTORS = ["nu", "block_size", "scale_format", "round_mode", "rht"]

# Fixed iteration order for deterministic per-configuration seeding (see
# `config_seed` below) -- independent of dict/groupby iteration order, which
# pandas does not guarantee is stable across versions.
NU_ORDER = ["1", "2", "3", "5", "8", "15", "30", "gaussian"]
BLOCK_SIZE_ORDER = [16, 32]
SCALE_FORMAT_ORDER = ["e8m0", "e4m3"]
ROUND_MODE_ORDER = ["rtne", "sr"]

# The theoretical exponent under exact accumulation (SPEC.md, "Theoretical
# bound definition"). NOT fitted -- fixed a priori, same status as c=1.
THEORETICAL_SLOPE = -0.5

# How large |slope - (-0.5)| must be to call a deviation "material" in the
# statement's narrative, as opposed to merely CI-significant. Distinct from
# `bucket`, which uses CI containment alone (the task's specified break
# criterion) and is NOT gated by this threshold. At this sweep's trial
# counts (1000-5000/cell) bootstrap CIs are narrow enough that even a
# slope of -0.496 can statistically exclude -0.5, which is a true but easy
# to over-read statement on its own -- this is the same distinction
# `scripts/probe_n_scaling.py`'s `SLOPE_MATCH_TOL` (also 0.02) draws
# between CI containment and practical closeness to a reference slope.
MATERIAL_GAP_TOL = 0.02

DEFAULT_N_RESAMPLES = 2000
DEFAULT_SEED = 0
CI = 0.95


def config_seed(
    base_seed: int, nu: str, block_size: int, scale_format: str, round_mode: str, rht: bool
) -> list[int]:
    """Deterministic entropy for this configuration's `numpy.random.Generator`.

    Plain small integers only (never Python's randomized `hash()` on a
    string), matching this project's established convention
    (`scripts/probe_n_scaling.py`'s `[base_seed, block_size, int(nu), n]`).
    Each factor is encoded via its position in a fixed order declared above,
    so the seed is stable across runs regardless of iteration order.
    """
    return [
        base_seed,
        NU_ORDER.index(nu),
        BLOCK_SIZE_ORDER.index(block_size),
        SCALE_FORMAT_ORDER.index(scale_format),
        ROUND_MODE_ORDER.index(round_mode),
        int(rht),
    ]


def fit_c_hat(
    n_values: np.ndarray, median_be: np.ndarray, u_eff: np.ndarray, p_hat: float
) -> float:
    """c_hat from `median_be = c_hat * n^p_hat * u_eff`, fixing `p_hat`.

    DESCRIPTIVE / EXPLORATORY ONLY (PREREGISTRATION.md sec 3.2). This is a
    least-squares fit of the log-space intercept alone (slope pinned at the
    already-fitted p_hat): `log10(c_hat)` is the mean, over the 5 n's, of
    `log10(median_be) - log10(u_eff) - p_hat * log10(n)`. c_hat must NEVER be
    substituted for the confirmatory bound's constant, which is fixed at
    c=1 everywhere else in this project (SPEC.md, "Theoretical bound
    definition (\U0001f9e01) -- RESOLVED"); it is reported here purely as a
    descriptive summary of this configuration's data, per Step 4.1's task.
    """
    log_residual = np.log10(median_be) - np.log10(u_eff) - p_hat * np.log10(n_values)
    return float(10.0 ** np.mean(log_residual))


def classify(slope_ci_low: float, slope_ci_high: float) -> str:
    if slope_ci_low <= THEORETICAL_SLOPE <= slope_ci_high:
        return "consistent_with_-0.5"
    if slope_ci_low > THEORETICAL_SLOPE:
        return "weakening (slower decay than -0.5)"
    return "faster_than_predicted (CI entirely below -0.5)"


def build_statement(table: pd.DataFrame, n_resamples: int, runtime_s: float) -> str:
    lines: list[str] = []
    lines.append(
        f"Step 4.1 n-scaling fits: {len(table)} configurations "
        f"(8 nu x 2 block_size x 2 scale_format x 2 round_mode x 2 rht), "
        f"{n_resamples} bootstrap resamples per configuration. "
        f"Runtime {runtime_s / 60.0:.1f} min."
    )
    lines.append("")
    n_consistent = int((table["bucket"] == "consistent_with_-0.5").sum())
    n_weakening = int(table["bucket"].str.startswith("weakening").sum())
    n_faster = int(table["bucket"].str.startswith("faster_than_predicted").sum())
    n_positive = int(table["positive_slope_flag"].sum())
    lines.append(
        f"consistent with -0.5 (CI contains it): {n_consistent}/{len(table)}\n"
        f"weakening (CI entirely above -0.5, decays slower than predicted): "
        f"{n_weakening}/{len(table)}\n"
        f"faster than predicted (CI entirely below -0.5): {n_faster}/{len(table)}\n"
        f"POSITIVE SLOPE (CI lower bound > 0 -- error growing with n): {n_positive}/{len(table)}"
    )
    lines.append("")
    weakening = table[table["bucket"].str.startswith("weakening")]
    is_material = weakening["gap_from_theoretical"].abs() > MATERIAL_GAP_TOL
    n_material = int(is_material.sum())
    material_nus = sorted(weakening.loc[is_material, "nu"].unique(), key=NU_ORDER.index)
    nonmaterial_nus = sorted(weakening.loc[~is_material, "nu"].unique(), key=NU_ORDER.index)
    lines.append(
        f"Of the {n_weakening} 'weakening' configurations, {n_material} deviate from "
        f"-0.5 by more than {MATERIAL_GAP_TOL} in slope magnitude (MATERIALLY weakening, "
        f"not just CI-significant) and {n_weakening - n_material} do not -- distinguishing "
        "statistical significance (the bucket, driven by this sweep's large trial counts "
        "narrowing every CI) from practical magnitude (the gap). "
        f"nu values with at least one materially-weakening configuration: {material_nus}. "
        f"nu values where every configuration's gap is <= {MATERIAL_GAP_TOL}: {nonmaterial_nus}."
    )
    lines.append("")
    if n_positive:
        lines.append("Positive-slope configurations (flagged explicitly, not just counted):")
        for _, row in table[table["positive_slope_flag"]].iterrows():
            lines.append(
                f"  nu={row['nu']} block_size={row['block_size']} "
                f"scale_format={row['scale_format']} round_mode={row['round_mode']} "
                f"rht={row['rht']}: slope={row['slope']:+.4f} "
                f"[{row['slope_ci_low']:+.4f}, {row['slope_ci_high']:+.4f}]"
            )
        lines.append("")
    if n_weakening:
        lines.append("Weakening configurations (candidates for nu* localization, Step 4.2):")
        for _, row in table[table["bucket"].str.startswith("weakening")].sort_values(
            "slope", ascending=False
        ).iterrows():
            material_tag = (
                "MATERIAL" if abs(row["gap_from_theoretical"]) > MATERIAL_GAP_TOL else "minor"
            )
            lines.append(
                f"  nu={row['nu']} block_size={row['block_size']} "
                f"scale_format={row['scale_format']} round_mode={row['round_mode']} "
                f"rht={row['rht']}: slope={row['slope']:+.4f} "
                f"[{row['slope_ci_low']:+.4f}, {row['slope_ci_high']:+.4f}]  "
                f"gap={row['gap_from_theoretical']:+.4f} ({material_tag})"
            )
        lines.append("")
    lines.append(
        "c_hat values in the table are DESCRIPTIVE/EXPLORATORY ONLY "
        "(PREREGISTRATION.md sec 3.2) -- never a candidate value for the "
        "confirmatory bound's constant, which stays fixed at c=1 everywhere "
        "else in this project (SPEC.md, 'Theoretical bound definition "
        "(\U0001f9e01) -- RESOLVED')."
    )
    lines.append("")
    lines.append(
        "This script fits and reports; it does not perform nu* localization "
        "(Step 4.2, a separate step) and does not decide whether the bound is "
        "broken anywhere."
    )
    return "\n".join(lines)


def main() -> None:
    # This project's SPEC.md section headers use a literal U+1F9E0 (brain
    # emoji) marker; some Windows consoles default stdout to a legacy
    # codepage (cp1252) that cannot encode it. Reconfigure rather than avoid
    # the character, so file output (already UTF-8 via `Path.write_text`)
    # and console output agree on content.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sweep-dir", type=Path, default=DEFAULT_SWEEP_DIR)
    parser.add_argument("--n-resamples", type=int, default=DEFAULT_N_RESAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    combined_path = args.sweep_dir.parent / f"{args.sweep_dir.name}.parquet"
    print(f"Loading {combined_path} ...")
    df = pd.read_parquet(combined_path, engine="pyarrow")
    print(f"Loaded {len(df):,} trial rows.")

    grouped = df.groupby(CONFIG_FACTORS, sort=False)
    n_configs = grouped.ngroups
    if n_configs != 128:
        raise ValueError(f"expected 128 configurations, found {n_configs}")

    started = time.perf_counter()
    summary_rows: list[dict] = []
    detail_rows: list[dict] = []

    for i, (config_key, cell_df) in enumerate(grouped):
        nu, block_size, scale_format, round_mode, rht = config_key
        by_n = cell_df.groupby("n", sort=True)
        n_values = np.array(sorted(by_n.groups), dtype=np.float64)
        if n_values.size != 5:
            raise ValueError(f"configuration {config_key} has {n_values.size} n values, expected 5")

        be_by_n: dict[float, np.ndarray] = {}
        median_be_by_n: dict[float, float] = {}
        u_eff_by_n: dict[float, float] = {}
        for n, cell in by_n:
            be_values = cell["be_median"].to_numpy(dtype=np.float64)
            be_by_n[float(n)] = be_values
            median_be_by_n[float(n)] = float(np.median(be_values))
            u_eff_unique = cell["u_eff_p50"].unique()
            if u_eff_unique.size != 1:
                raise ValueError(
                    f"configuration {config_key}, n={n}: u_eff_p50 is not constant within "
                    f"the cell ({u_eff_unique})"
                )
            u_eff_by_n[float(n)] = float(u_eff_unique[0])
            detail_rows.append(
                {
                    "nu": nu,
                    "block_size": block_size,
                    "scale_format": scale_format,
                    "round_mode": round_mode,
                    "rht": rht,
                    "n": int(n),
                    "n_trials": int(be_values.size),
                    "median_be": median_be_by_n[float(n)],
                    "u_eff_p50": u_eff_by_n[float(n)],
                }
            )

        rng = np.random.default_rng(
            config_seed(args.seed, nu, block_size, scale_format, round_mode, bool(rht))
        )
        slope, intercept, ci_low, ci_high = bootstrap_loglog_slope_ci(
            be_by_n, rng, n_resamples=args.n_resamples, ci=CI
        )

        median_be_arr = np.array([median_be_by_n[n] for n in sorted(median_be_by_n)])
        u_eff_arr = np.array([u_eff_by_n[n] for n in sorted(u_eff_by_n)])
        c_hat = fit_c_hat(n_values, median_be_arr, u_eff_arr, slope)

        bucket = classify(ci_low, ci_high)
        positive_slope_flag = ci_low > 0.0

        summary_rows.append(
            {
                "nu": nu,
                "block_size": block_size,
                "scale_format": scale_format,
                "round_mode": round_mode,
                "rht": rht,
                "slope": slope,
                "slope_ci_low": ci_low,
                "slope_ci_high": ci_high,
                # Point-estimate distance from the theoretical -0.5, kept
                # separate from `bucket`: at this sweep's trial counts (1000-
                # 5000/cell) bootstrap CIs are narrow enough that many
                # configurations are *statistically* distinguishable from
                # -0.5 while being, in slope magnitude, nearly identical to
                # it -- the same distinction `scripts/probe_n_scaling.py`
                # draws between "the CI excludes it" and "resembles the
                # pattern" (see its `describe_slope`). `bucket` alone would
                # not let a reader tell "off by 0.003" from "off by 0.4".
                "gap_from_theoretical": slope - THEORETICAL_SLOPE,
                "intercept": intercept,
                "c_hat": c_hat,
                "bucket": bucket,
                "positive_slope_flag": positive_slope_flag,
                "n_resamples": args.n_resamples,
            }
        )
        if (i + 1) % 16 == 0 or (i + 1) == n_configs:
            print(f"  fitted {i + 1}/{n_configs} configurations "
                  f"({time.perf_counter() - started:.1f}s elapsed)")

    runtime_s = time.perf_counter() - started
    table = pd.DataFrame(summary_rows)
    detail = pd.DataFrame(detail_rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config = {
        "script": "fit_n_scaling.py",
        "version": 1,
        "sweep_dir": str(args.sweep_dir),
        "n_resamples": args.n_resamples,
        "seed": args.seed,
        "ci": CI,
        "theoretical_slope": THEORETICAL_SLOPE,
    }
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
    stem = f"n_scaling_fits_{digest}"

    table.to_parquet(OUT_DIR / f"{stem}_table.parquet", engine="pyarrow")
    table.to_csv(OUT_DIR / f"{stem}_table.csv", index=False)
    detail.to_parquet(OUT_DIR / f"{stem}_detail.parquet", engine="pyarrow")
    (OUT_DIR / f"{stem}.config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
    )

    statement = build_statement(table, args.n_resamples, runtime_s)
    print("\n" + statement + "\n")
    (OUT_DIR / f"{stem}_statement.txt").write_text(statement + "\n", encoding="utf-8")

    print(f"Wrote table, detail, config and statement to {OUT_DIR} (stem={stem})")


if __name__ == "__main__":
    main()
