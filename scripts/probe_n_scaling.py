"""Diagnostic probe: how does `median(BE)` scale with the contraction dimension `n`
under **exact** accumulation, and does that scaling depend on the block size?

Run once, as an input to an open conceptual question (SPEC.md: the definition of
the theoretical error bound for block-scaled formats). Nothing here is a
decision, and nothing here touches `qgemm.bounds` -- `bounds.py` is still
unimplemented and this script deliberately does not anticipate what goes in it.

Why the question is open at all
-------------------------------
The classical bounds (`gamma_n = n*u / (1 - n*u)`, and its probabilistic
`sqrt(n)*u` relative) get their `n`-dependence from **repeated rounding during
accumulation**: `n` sequential partial sums, each rounded. This project's
primary route accumulates in exact float64 (`accum="exact"`), so that mechanism
is simply absent -- the residual `qgemm(A, B, cfg) - A @ B` is input-
quantization error and nothing else (see `qgemm.gemm`'s module docstring).

So if `BE` carries any `n`-dependence here, it comes from somewhere else. Two
candidate mechanisms, which this probe is built to tell apart:

* **Cancellation.** `BE`'s numerator is `|sum_k eps_k|` over `n` per-element
  quantization errors, while its denominator `sum_k |a_k||b_k|` grows like `n`.
  Signed errors partially cancel in the numerator but not in the denominator,
  so the ratio shrinks with `n`. Crucially, elements sharing a block share a
  *scale*, so their errors are not independent: the effective number of
  roughly-independent summands is plausibly the number of **blocks**,
  `n / block_size`, not `n`. Under that reading `median(BE) ~ sqrt(block_size /
  n)` -- a log-log slope of **-0.5 at both block sizes**, with block 32 sitting
  a factor `sqrt(2)` *above* block 16 in level.
* **Extreme-value dominance.** Under heavy tails the numerator may be owned by
  a single worst block rather than by an average over many, in which case
  growing `n` buys progressively less cancellation and the slope flattens
  toward 0 -- or the slope itself differs between block sizes, which the simple
  `n / block_size` rescaling above cannot produce.

Both are hypotheses. This script measures the slope; it does not adjudicate.

What is held fixed, and why this is not MXFP4 or NVFP4
------------------------------------------------------
`scale_format="e8m0"` and `use_global_scale=False` are held fixed across
**every** cell, at both block sizes, so that `block_size` is the only quantizer
parameter that moves. That is the point: MXFP4 and NVFP4 differ in block size
*and* scale format at once, so a comparison between the two presets cannot
attribute anything to either. Holding the scale format fixed isolates the block
mechanism -- and it means the `block_size=16` arm here is
**`(16, e8m0, no global scale)`, an experimental control that corresponds to no
hardware format, no specification and no vendor** (SPEC.md, "Why the controls
exist"). This probe reproduces neither preset and must not be read as doing so.
`element_format="e2m1"` and `round_mode="rtne"` are likewise fixed throughout.

Grid and shape
--------------
`block_size in {16, 32}` x `nu in {1, 30}` (heavy-tailed vs. near-Gaussian, so
the answer is allowed to differ by tail weight) x `n in {64 ... 4096}`, every
`n` a multiple of both block sizes so neither arm ever gets a short tail block.

The shape is `(m=64, n) @ (n, k=64)`: small **fixed** output dimensions with
only the contraction length varying, following the convention
`scripts/check_metric_stability.py` already uses and the main sweep plans to
use. The statistical question is entirely about the contraction length -- `n`
is what the numerator's cancellation happens over, what the denominator sums
over, and what the operands are blocked along -- so paying for a large output
would buy nothing this probe asks about.

The reported statistic
----------------------
`median(BE)` per cell is the **median over trials of each trial's median** over
its 64x64 = 4096 output elements. The resampling unit is the independent trial
(PREREGISTRATION.md sec 3.2): elements within a trial are not independent -- a
row shares `A`'s row, a block shares a scale -- so the trial is the only level
at which a bootstrap here is honest. The pooled median over all
`trials x 4096` values is recorded alongside in the summary table, so the two
can be compared rather than assumed equal.

Scope and stop condition (do not skip)
--------------------------------------
This is a **cheap probe**, not a sweep cell: the trial count is well below what
a confirmatory measurement would use, and the actual count and total runtime
are reported with the results. It covers two block sizes, two `nu`, one scale
format, one shape family, RTNE only, no RHT, exact accumulation only. It says
nothing about E4M3 scales, global scales, stochastic rounding, the RHT, bf16
accumulation, or block sizes 8 and 64.

The script measures and reports. What the theoretical bound's functional form
should be -- whether it should carry an `n`-dependence at all, and on what
argument -- is a human decision made after review, not made here and not
implied by anything below.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

from qgemm.gemm import GemmConfig, qgemm  # noqa: E402
from qgemm.metrics import backward_error  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "results" / "diagnostics" / "n_scaling"

# The grid. Both block sizes divide every n, so no cell ever gets a short
# trailing block -- the tail policy is held out of this comparison on purpose.
GRID_BLOCK_SIZE = (16, 32)
GRID_NU = (1.0, 30.0)
GRID_N = (64, 128, 256, 512, 1024, 2048, 4096)

# Quantizer parameters held fixed across every cell. `scale_format` and
# `use_global_scale` are fixed so that `block_size` is the only thing moving;
# see the module docstring on why that makes neither arm a real format.
FIXED_SCALE_FORMAT = "e8m0"
FIXED_USE_GLOBAL_SCALE = False
FIXED_ELEMENT_FORMAT = "e2m1"
FIXED_ROUND_MODE = "rtne"
FIXED_ACCUM = "exact"

DEFAULT_TRIALS = 500  # a probe, not a sweep cell; reported with the results
DEFAULT_SEED = 0
OUT_ROWS = 64  # m
OUT_COLS = 64  # k; the output is 64x64 = 4096 BE values per trial

N_BOOTSTRAP = 10_000  # PREREGISTRATION.md sec 3.2's bootstrap convention
CI_PERCENTILES = (2.5, 97.5)  # 95% percentile-method CI, same convention

# The n at which the level (not slope) comparison between block sizes is read
# off. Chosen before the run, in the middle of the swept log range.
LEVEL_COMPARISON_N = 1024

# The three reference patterns the observed slope is described against. Signs
# are explicit: cancellation makes BE *decrease* with n, so the cancellation
# reference is -0.5, not +0.5.
REFERENCE_SLOPES = (
    (0.0, "flat (~0): no n-dependence at all"),
    (-0.5, "~-0.5: cancellation among ~n/block_size roughly-independent blocks"),
    (-1.0, "~-1: full averaging over ~n independent summands"),
)

# How close to a reference slope counts as "resembles it" in the plain-language
# verdict. This is a **reporting threshold, not a hypothesis test**: with 500
# trials the bootstrap CIs are narrow enough (half-width ~0.001) that a slope
# of -0.4978 is formally distinguishable from -0.5 while being, for any purpose
# this probe serves, the same pattern. Saying "the CI excludes -0.5" and
# stopping there would be true and useless. 0.02 is one twenty-fifth of the gap
# between adjacent references, chosen before the run.
SLOPE_MATCH_TOL = 0.02


def sample_t(shape: tuple[int, ...], nu: float, rng: np.random.Generator) -> np.ndarray:
    """Standard t-Student(nu) sample, float64.

    Deliberately local to this script rather than added to
    `qgemm.distributions`, following `scripts/check_metric_stability.py`: the
    nu-axis sampler is study infrastructure PREREGISTRATION.md R15 wants
    implemented and tested on its own terms, not smuggled in through a
    diagnostic. No rescaling is applied -- `BE` is a ratio, invariant to a
    common rescaling of the operands up to the quantizer's own grid, and for
    nu <= 2 the variance a rescaling would normalize does not exist anyway.
    """
    return np.asarray(rng.standard_t(nu, size=shape), dtype=np.float64)


def config_digest(config: dict) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def cell_config(block_size: int) -> GemmConfig:
    """The `GemmConfig` for one arm. Everything but `block_size` is pinned."""
    return GemmConfig(
        quantize=True,
        block_size=block_size,
        scale_format=FIXED_SCALE_FORMAT,
        use_global_scale=FIXED_USE_GLOBAL_SCALE,
        element_format=FIXED_ELEMENT_FORMAT,
        round_mode=FIXED_ROUND_MODE,
        rht=False,
        accum=FIXED_ACCUM,
    )


def run_cell(
    block_size: int, nu: float, n: int, trials: int, base_seed: int
) -> tuple[np.ndarray, float]:
    """Run one `(block_size, nu, n)` cell.

    Returns `(per_trial_median_be, pooled_median_be)`: the per-trial medians
    (shape `(trials,)`, the bootstrap's resampling unit) and the median over
    every one of the `trials x 4096` `BE` values pooled, kept for comparison.
    """
    rng = np.random.default_rng([base_seed, block_size, int(nu), n])
    base = cell_config(block_size)
    per_trial = np.empty(trials, dtype=np.float64)
    pooled = np.empty((trials, OUT_ROWS * OUT_COLS), dtype=np.float64)

    for t in tqdm(range(trials), desc=f"block={block_size} nu={nu:g} n={n}", leave=False):
        a = sample_t((OUT_ROWS, n), nu, rng)
        b = sample_t((n, OUT_COLS), nu, rng)
        # `seed` varies per trial for form's sake; under RTNE with no RHT the
        # pipeline makes no random draw at all, so it changes no number here.
        approx = qgemm(a, b, replace(base, seed=t))
        be = backward_error(a, b, approx).ravel()
        pooled[t] = be
        per_trial[t] = float(np.median(be))

    return per_trial, float(np.median(pooled))


def fit_loglog_slope(n_values: np.ndarray, y_values: np.ndarray) -> tuple[float, float]:
    """Closed-form least-squares slope/intercept of `log10(y)` vs `log10(n)`."""
    x = np.log10(np.asarray(n_values, dtype=np.float64))
    y = np.log10(np.asarray(y_values, dtype=np.float64))
    x_centered = x - x.mean()
    slope = float(np.sum(x_centered * (y - y.mean())) / np.sum(x_centered**2))
    intercept = float(y.mean() - slope * x.mean())
    return slope, intercept


def bootstrap_medians(per_trial: np.ndarray, rng: np.random.Generator, n_boot: int) -> np.ndarray:
    """`n_boot` bootstrap replicates of `median(per_trial)`, resampling trials."""
    idx = rng.integers(0, per_trial.size, size=(n_boot, per_trial.size))
    return np.median(per_trial[idx], axis=1)


def bootstrap_slopes(
    per_trial_by_n: dict[int, np.ndarray], rng: np.random.Generator, n_boot: int
) -> np.ndarray:
    """`n_boot` bootstrap replicates of the log-log slope of `median(BE)` vs `n`.

    Trials are resampled independently at each `n`, which is what they are:
    every `(block_size, nu, n)` cell draws its own operands from its own seed
    stream, so no trial is shared across `n`. The replicates are returned
    rather than reduced to a CI so that derived quantities -- notably the
    *difference* of two arms' slopes -- can be formed on the same replicates.
    """
    n_values = np.array(sorted(per_trial_by_n), dtype=np.float64)
    medians = np.empty((len(n_values), n_boot), dtype=np.float64)
    for row, n in enumerate(sorted(per_trial_by_n)):
        medians[row] = bootstrap_medians(per_trial_by_n[n], rng, n_boot)

    x = np.log10(n_values)
    x_centered = x - x.mean()
    y = np.log10(medians)  # (n_values, n_boot)
    y_centered = y - y.mean(axis=0, keepdims=True)
    return (x_centered[:, None] * y_centered).sum(axis=0) / np.sum(x_centered**2)


def percentile_ci(replicates: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(replicates, CI_PERCENTILES)
    return float(lo), float(hi)


def bootstrap_ratio_ci(
    numerator_trials: np.ndarray,
    denominator_trials: np.ndarray,
    rng: np.random.Generator,
    n_boot: int,
) -> tuple[float, float]:
    """Percentile bootstrap CI of `median(numerator) / median(denominator)`.

    The two samples come from independent cells, so they are resampled
    independently.
    """
    num = bootstrap_medians(numerator_trials, rng, n_boot)
    den = bootstrap_medians(denominator_trials, rng, n_boot)
    lo, hi = np.percentile(num / den, CI_PERCENTILES)
    return float(lo), float(hi)


def describe_slope(slope: float, ci_low: float, ci_high: float) -> str:
    """Describe the fitted slope against the three reference patterns.

    Two separate facts are reported and never conflated: how *far* the slope is
    from the nearest reference (a magnitude, judged against `SLOPE_MATCH_TOL`),
    and whether the bootstrap CI *contains* that reference (a precision
    statement). At this trial count the second is much the stricter of the two,
    so reporting only it would turn every arm into "excludes everything".
    """
    ref, label = min(REFERENCE_SLOPES, key=lambda r: abs(slope - r[0]))
    gap = slope - ref
    contained = ci_low <= ref <= ci_high
    precision = (
        "the CI contains it"
        if contained
        else f"the CI excludes it, by {min(abs(ci_low - ref), abs(ci_high - ref)):.4f}"
    )
    if abs(gap) <= SLOPE_MATCH_TOL:
        return (
            f"resembles the {label} pattern -- offset {gap:+.4f}, inside the "
            f"{SLOPE_MATCH_TOL} reporting threshold ({precision})"
        )
    distances = ", ".join(f"{r:+.1f}: {abs(slope - r):.3f}" for r, _ in REFERENCE_SLOPES)
    return (
        f"resembles none of the three -- nearest is {ref:+.1f} ({label.split(':')[0]}) "
        f"at offset {gap:+.4f}, outside the {SLOPE_MATCH_TOL} reporting threshold. "
        f"Distances to each reference: {distances}"
    )


def make_figure(summary: pd.DataFrame, fits: pd.DataFrame) -> plt.Figure:
    """log-log median(BE) vs n, one panel per nu, both block sizes overlaid."""
    fig, axes = plt.subplots(1, len(GRID_NU), figsize=(12.5, 5.8))
    colors = {16: "C0", 32: "C3"}
    markers = {16: "o", 32: "s"}

    for ax, nu in zip(axes, GRID_NU, strict=True):
        for block_size in GRID_BLOCK_SIZE:
            sel = summary[
                (summary["nu"] == nu) & (summary["block_size"] == block_size)
            ].sort_values("n")
            ax.errorbar(
                sel["n"],
                sel["median_be"],
                yerr=[
                    sel["median_be"] - sel["median_be_ci_low"],
                    sel["median_be_ci_high"] - sel["median_be"],
                ],
                marker=markers[block_size],
                ms=5,
                lw=1.6,
                capsize=3,
                color=colors[block_size],
                label=f"block_size={block_size}: median(BE)",
            )
            fit = fits[(fits["nu"] == nu) & (fits["block_size"] == block_size)].iloc[0]
            n_line = np.array([sel["n"].min(), sel["n"].max()], dtype=np.float64)
            fitted = 10.0 ** (fit["intercept"] + fit["slope"] * np.log10(n_line))
            ax.plot(
                n_line,
                fitted,
                color=colors[block_size],
                lw=2.6,
                alpha=0.4,
                label=(
                    f"  fit: slope={fit['slope']:+.3f} "
                    f"[{fit['ci_low']:+.3f}, {fit['ci_high']:+.3f}]"
                ),
            )

        # Reference slopes, anchored to the block-16 curve at the smallest n so
        # only the *shape* of each guide is being compared, not its height.
        anchor = summary[
            (summary["nu"] == nu)
            & (summary["block_size"] == GRID_BLOCK_SIZE[0])
            & (summary["n"] == min(GRID_N))
        ]["median_be"].iloc[0]
        n_ref = np.array([min(GRID_N), max(GRID_N)], dtype=np.float64)
        for (ref, _label), style in zip(REFERENCE_SLOPES, ("--", "-.", ":"), strict=True):
            ax.plot(
                n_ref,
                anchor * (n_ref / min(GRID_N)) ** ref,
                color="gray",
                ls=style,
                lw=1.2,
                label=f"reference slope {ref:+.1f}",
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("n (contraction dimension)")
        ax.set_ylabel("median(BE)")
        ax.set_title(f"nu = {nu:g}" + ("  (heavy-tailed)" if nu <= 2 else "  (near-Gaussian)"))
        ax.legend(fontsize=7.5, loc="best")
        ax.grid(True, which="both", alpha=0.15)

    fig.suptitle(
        "median(BE) vs contraction dimension under exact accumulation\n"
        f"E2M1 elements, {FIXED_SCALE_FORMAT.upper()} block scale, no global scale, RTNE "
        "-- a controlled block_size comparison, NOT MXFP4 or NVFP4",
        fontsize=10,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    return fig


def build_statement(
    fits: pd.DataFrame,
    levels: pd.DataFrame,
    slope_diffs: pd.DataFrame,
    trials: int,
    runtime_s: float,
) -> str:
    lines: list[str] = []
    lines.append(
        f"n-scaling probe under exact accumulation: {trials} trials per cell, shape "
        f"({OUT_ROWS} x n) @ (n x {OUT_COLS}), {trials * OUT_ROWS * OUT_COLS} BE values "
        f"per cell, {len(GRID_BLOCK_SIZE) * len(GRID_NU) * len(GRID_N)} cells. "
        f"Total runtime {runtime_s / 60.0:.1f} min."
    )
    lines.append("")
    lines.append(
        "CONFIGURATION HELD FIXED ACROSS EVERY CELL: scale_format="
        f"{FIXED_SCALE_FORMAT!r}, use_global_scale={FIXED_USE_GLOBAL_SCALE}, "
        f"element_format={FIXED_ELEMENT_FORMAT!r}, round_mode={FIXED_ROUND_MODE!r}, "
        f"accum={FIXED_ACCUM!r}, no RHT. The scale format is deliberately held constant "
        "so that block_size is the only quantizer parameter that moves, which isolates "
        "the block mechanism. THIS REPRODUCES NEITHER MXFP4 NOR NVFP4: MXFP4 is "
        "(block 32, E8M0, no global scale) and NVFP4 is (block 16, E4M3, global scale), "
        "so the block_size=16 arm here is an experimental control corresponding to no "
        "hardware format, no specification and no vendor."
    )
    lines.append("")
    lines.append("FITTED LOG-LOG SLOPES of median(BE) vs n:")
    lines.append("")
    for _, row in fits.iterrows():
        half = (row["ci_high"] - row["ci_low"]) / 2.0
        lines.append(
            f"  block_size={int(row['block_size']):2d}  nu={row['nu']:g}  "
            f"slope = {row['slope']:+.4f} +/- {half:.4f}  "
            f"(95% CI [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}])"
        )
        lines.append(f"      {describe_slope(row['slope'], row['ci_low'], row['ci_high'])}")
    lines.append("")
    lines.append(f"LEVEL COMPARISON at n = {LEVEL_COMPARISON_N} (height of the curve, not slope):")
    lines.append("")
    for _, row in levels.iterrows():
        lines.append(
            f"  nu={row['nu']:g}:  block16 = {row['median_be_16']:.6g}   "
            f"block32 = {row['median_be_32']:.6g}   "
            f"ratio 32/16 = {row['ratio']:.4f} "
            f"(95% CI [{row['ratio_ci_low']:.4f}, {row['ratio_ci_high']:.4f}])"
        )
        separated = row["ratio_ci_low"] > 1.0 or row["ratio_ci_high"] < 1.0
        contains_sqrt2 = row["ratio_ci_low"] <= np.sqrt(2.0) <= row["ratio_ci_high"]
        lines.append(
            "      block 32 sits "
            + (
                "at a measurably greater height than block 16"
                if separated and row["ratio"] > 1.0
                else "at a measurably lower height than block 16"
                if separated
                else "at a height not distinguishable from block 16 (the ratio's CI contains 1.0)"
            )
            + f". Against sqrt(32/16) = {np.sqrt(2.0):.4f}, the level ratio the "
            "block-cancellation reading would predict: "
            + (
                "the CI contains it."
                if contains_sqrt2
                else f"the CI excludes it (observed {row['ratio']:.4f})."
            )
        )
    lines.append("")
    lines.append("WHICH REFERENCE PATTERN THE DATA RESEMBLES, stated separately per nu:")
    lines.append("")
    for nu in GRID_NU:
        sel = fits[fits["nu"] == nu].sort_values("block_size")
        s16 = sel[sel["block_size"] == 16].iloc[0]
        s32 = sel[sel["block_size"] == 32].iloc[0]
        diff = slope_diffs[slope_diffs["nu"] == nu].iloc[0]
        lines.append(
            f"  nu = {nu:g}: slopes are {s16['slope']:+.4f} (block 16) and "
            f"{s32['slope']:+.4f} (block 32)."
        )
        for _, row in sel.iterrows():
            lines.append(
                f"      block {int(row['block_size']):2d}: "
                f"{describe_slope(row['slope'], row['ci_low'], row['ci_high'])}"
            )
        material = abs(diff["slope_diff"]) > SLOPE_MATCH_TOL
        lines.append(
            f"      slope(block 32) - slope(block 16) = {diff['slope_diff']:+.4f} "
            f"(95% CI [{diff['ci_low']:+.4f}, {diff['ci_high']:+.4f}]) -- "
            + (
                "a difference larger than the "
                if material
                else "smaller in magnitude than the "
            )
            + f"{SLOPE_MATCH_TOL} reporting threshold, so the two block sizes "
            + ("do NOT share a slope here." if material else "share a slope to within it.")
        )
    lines.append("")
    lines.append(
        "This is a probe, not a decision. It reports the measured scaling and nothing "
        "more; what functional form the theoretical bound should take -- including "
        "whether it should carry an n-dependence at all, and on what argument -- is a "
        "human call made on review. qgemm.bounds is untouched by this script."
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    args = parser.parse_args()

    config = {
        "script": "probe_n_scaling.py",
        "version": 1,
        "grid_block_size": list(GRID_BLOCK_SIZE),
        "grid_nu": list(GRID_NU),
        "grid_n": list(GRID_N),
        "out_rows": OUT_ROWS,
        "out_cols": OUT_COLS,
        "trials": args.trials,
        "seed": args.seed,
        "scale_format": FIXED_SCALE_FORMAT,
        "use_global_scale": FIXED_USE_GLOBAL_SCALE,
        "element_format": FIXED_ELEMENT_FORMAT,
        "round_mode": FIXED_ROUND_MODE,
        "accum": FIXED_ACCUM,
        "rht": False,
        "distribution": "t-student",
        "n_bootstrap": args.n_bootstrap,
        "ci_percentiles": list(CI_PERCENTILES),
        "level_comparison_n": LEVEL_COMPARISON_N,
        "slope_match_tol": SLOPE_MATCH_TOL,
    }
    digest = config_digest(config)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"probe_n_scaling_{digest}"

    started = time.perf_counter()
    summary_rows: list[dict] = []
    trial_frames: list[pd.DataFrame] = []
    per_trial_all: dict[tuple[int, float], dict[int, np.ndarray]] = {}

    for block_size in GRID_BLOCK_SIZE:
        for nu in GRID_NU:
            by_n: dict[int, np.ndarray] = {}
            for n in GRID_N:
                per_trial, pooled_median = run_cell(block_size, nu, n, args.trials, args.seed)
                by_n[n] = per_trial

                ci_rng = np.random.default_rng([args.seed, 0xC1, block_size, int(nu), n])
                reps = bootstrap_medians(per_trial, ci_rng, args.n_bootstrap)
                ci_low, ci_high = np.percentile(reps, CI_PERCENTILES)

                summary_rows.append(
                    {
                        "block_size": block_size,
                        "nu": nu,
                        "n": n,
                        "trials": args.trials,
                        "median_be": float(np.median(per_trial)),
                        "median_be_ci_low": float(ci_low),
                        "median_be_ci_high": float(ci_high),
                        "pooled_median_be": pooled_median,
                        "mean_of_trial_medians": float(np.mean(per_trial)),
                    }
                )
                trial_frames.append(
                    pd.DataFrame(
                        {
                            "block_size": block_size,
                            "nu": nu,
                            "n": n,
                            "trial": np.arange(args.trials),
                            "trial_median_be": per_trial,
                        }
                    )
                )
            per_trial_all[(block_size, nu)] = by_n

    runtime_s = time.perf_counter() - started
    summary = pd.DataFrame(summary_rows)

    fit_rows = []
    slope_replicates: dict[tuple[int, float], np.ndarray] = {}
    for (block_size, nu), by_n in per_trial_all.items():
        sel = summary[(summary["block_size"] == block_size) & (summary["nu"] == nu)].sort_values(
            "n"
        )
        slope, intercept = fit_loglog_slope(
            sel["n"].to_numpy(dtype=np.float64), sel["median_be"].to_numpy(dtype=np.float64)
        )
        boot_rng = np.random.default_rng([args.seed, 0xB007, block_size, int(nu)])
        replicates = bootstrap_slopes(by_n, boot_rng, args.n_bootstrap)
        slope_replicates[(block_size, nu)] = replicates
        ci_low, ci_high = percentile_ci(replicates)
        fit_rows.append(
            {
                "block_size": block_size,
                "nu": nu,
                "slope": slope,
                "intercept": intercept,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
    fits = pd.DataFrame(fit_rows).sort_values(["nu", "block_size"]).reset_index(drop=True)

    # Does the *slope itself* differ between block sizes? That is the third
    # reference pattern, and the simple n/block_size rescaling cannot produce
    # it: rescaling the argument shifts a power law's level, never its exponent.
    # The two arms are independent cells, so their replicates pair arbitrarily.
    diff_rows = []
    for nu in GRID_NU:
        d = slope_replicates[(32, nu)] - slope_replicates[(16, nu)]
        lo, hi = percentile_ci(d)
        s16 = fits[(fits["nu"] == nu) & (fits["block_size"] == 16)]["slope"].iloc[0]
        s32 = fits[(fits["nu"] == nu) & (fits["block_size"] == 32)]["slope"].iloc[0]
        diff_rows.append(
            {"nu": nu, "slope_diff": float(s32 - s16), "ci_low": lo, "ci_high": hi}
        )
    slope_diffs = pd.DataFrame(diff_rows)

    level_rows = []
    for nu in GRID_NU:
        t16 = per_trial_all[(16, nu)][LEVEL_COMPARISON_N]
        t32 = per_trial_all[(32, nu)][LEVEL_COMPARISON_N]
        ratio_rng = np.random.default_rng([args.seed, 0x1E7E, int(nu)])
        lo, hi = bootstrap_ratio_ci(t32, t16, ratio_rng, args.n_bootstrap)
        level_rows.append(
            {
                "nu": nu,
                "n": LEVEL_COMPARISON_N,
                "median_be_16": float(np.median(t16)),
                "median_be_32": float(np.median(t32)),
                "ratio": float(np.median(t32) / np.median(t16)),
                "ratio_ci_low": lo,
                "ratio_ci_high": hi,
            }
        )
    levels = pd.DataFrame(level_rows)

    summary.to_parquet(OUT_DIR / f"{stem}_summary.parquet", engine="pyarrow")
    fits.to_parquet(OUT_DIR / f"{stem}_slopes.parquet", engine="pyarrow")
    slope_diffs.to_parquet(OUT_DIR / f"{stem}_slope_diffs.parquet", engine="pyarrow")
    levels.to_parquet(OUT_DIR / f"{stem}_levels.parquet", engine="pyarrow")
    pd.concat(trial_frames, ignore_index=True).to_parquet(
        OUT_DIR / f"{stem}_trials.parquet", engine="pyarrow"
    )
    (OUT_DIR / f"{stem}.config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
    )

    fig = make_figure(summary, fits)
    fig_path = OUT_DIR / f"{stem}_n_scaling.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    statement = build_statement(fits, levels, slope_diffs, args.trials, runtime_s)
    print("\n" + statement + "\n")
    (OUT_DIR / f"{stem}_statement.txt").write_text(statement + "\n", encoding="utf-8")

    print("SUMMARY (median(BE) per cell):")
    print(summary.to_string(index=False))
    print("\nSLOPES:")
    print(fits.to_string(index=False))
    print("\nSLOPE DIFFERENCES (block 32 - block 16):")
    print(slope_diffs.to_string(index=False))
    print("\nLEVELS:")
    print(levels.to_string(index=False))
    print(
        f"\nWrote summary, slopes, levels, per-trial medians, config, figure "
        f"({fig_path.name}) and statement to {OUT_DIR}"
    )
    print(
        "\nSTOP: this script only measures and reports. What the theoretical bound's "
        "functional form should be is a human decision made on review; qgemm.bounds is "
        "not touched here."
    )


if __name__ == "__main__":
    main()
