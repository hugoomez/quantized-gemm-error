"""Diagnostic: is the backward-error metric stable enough to build the study on?

Run once, before any large-scale sweep, to decide **how the primary metric is
reported** -- raw `BE` or `log(BE)`. Nothing here changes `qgemm.metrics`; this
script only measures, and its recommendation is written to SPEC.md as a
proposal for human review (PREREGISTRATION.md section 7 item 3 stays open
until then).

The metric under test
---------------------
For a GEMM `C = A @ B` approximated by `Chat = qgemm(A, B, config)`::

    BE_ij = |(A @ B)_ij - Chat_ij| / (|A| @ |B|)_ij

Numerator and denominator are built from the *same* `A`, `B`, so they are
correlated, and under heavy-tailed operands both are outlier-dominated. The
ratio of two heavy-tailed correlated quantities need not resemble either one:
if the denominator has mass near zero, `BE` can carry a heavier tail than the
numerator does on its own. That is a claim to measure, not to assume, so the
denominator is examined **as its own distribution** (step 3 below) rather than
argued away with a concentration argument.

What is measured, per cell `(nu, n)`
------------------------------------
1. `BE` over `--trials` independent trials, operands drawn from t-Student(nu).
2. Histogram of `BE` on a log axis (`fig ..._be_hist.png`).
3. The **denominator** `(|A| @ |B|)_ij` on its own, scale-normalized by its
   own median, with the near-zero mass measured directly: left-tail quantiles
   and `P(D < 10^-k * median D)`. Plus a tail *attribution* -- for the largest
   `BE` values, where do their numerator and denominator sit in their own
   marginals? That is what separates "the ratio is heavy because the numerator
   is" from "the ratio is heavy because the denominator nearly vanished".
4. Tail index by Hill estimator over a grid of tail fractions `k`, for `BE`,
   for the numerator, and for `1/D` (so the ratio's tail can be compared
   against each input's own tail). Moment existence is read off `alpha`: the
   mean needs `alpha > 1`, the variance `alpha > 2`. Cross-checked with the
   max-to-sum ratio `max X_i^p / sum X_i^p`, which decays to 0 iff the p-th
   moment is finite -- an estimator-free check that does not depend on
   choosing `k`.
5. The same diagnostics for `log10(BE)`, the candidate fallback, plus a direct
   comparison of how much three estimators (mean, median, geometric mean) move
   between disjoint batches of trials.

Grid and configuration
----------------------
`nu in {1, 2, 3, 30}` x `n in {16, 256, 4096}`, where `n` is the **contraction
dimension** (PREREGISTRATION.md notation). `n = 16` is deliberately included:
it is the highest-risk corner, with the fewest terms in the denominator's sum
and the heaviest tails at once.

Quantization is **active**, in the **MXFP4 preset** -- `GemmConfig()`'s
defaults: `block_size=32`, `scale_format="e8m0"`, `use_global_scale=False`,
`element_format="e2m1"`, `round_mode="rtne"`, `rht=False`, `accum="exact"`.
`accum="exact"` matters: it makes the residual `Chat - A @ B` input-
quantization error and nothing else. Note that at `n = 16` a block-32
quantizer has one *short* block covering the whole contraction dimension
(SPEC.md's tail policy), so that cell's block structure differs from the
others by construction, not by accident.

The output matrix is kept small (`--out-rows`, default 32 -> 32x32 = 1024 `BE`
values per trial) while `n` sweeps up to 4096, because the statistical
question is entirely about the contraction length: `n` is what the
denominator's sum is over, and what the operands are blocked along. This buys
~10^6 `BE` samples per cell at every `n` for a few seconds of compute.
Elements within a trial are *not* independent (a row shares `A`'s row, a block
shares a scale), so every estimator-stability figure resamples **trials**, per
PREREGISTRATION.md section 3.2's resampling unit.

Outputs (all under `results/diagnostics/metric_stability/`, one run, no manual
steps) are named by the sha256 of the run config, following the repo's results
convention, with the config saved alongside.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

from qgemm.gemm import GemmConfig, qgemm  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "results" / "diagnostics" / "metric_stability"

# The grid. nu is the t-Student degrees of freedom (smaller = heavier tails);
# n is the contraction dimension. n = 16 is the high-risk corner and is not
# optional -- it is the reason this diagnostic exists.
GRID_NU = (1.0, 2.0, 3.0, 30.0)
GRID_N = (16, 256, 4096)

DEFAULT_TRIALS = 1000
DEFAULT_OUT_ROWS = 32  # M = N; the output is small on purpose, see module docstring.

# Tail fractions for the Hill estimator. Reported as a curve rather than a
# single number: a tail index that swings with k is itself the finding.
HILL_K_FRACS = (0.005, 0.01, 0.02, 0.05, 0.10)
HILL_PRIMARY_K_FRAC = 0.05

N_BATCHES = 10  # disjoint batches of trials, for estimator dispersion
TOP_FRAC = 0.001  # "the tail of BE" for the attribution diagnostic

# Raw (numerator, denominator, BE) triples kept per cell so the analysis can be
# redone without re-running: a uniform subsample for the bulk, plus the largest
# BE values outright, because a uniform subsample of 10^6 samples does not
# preserve the tail that the whole question is about.
SUBSAMPLE_UNIFORM = 4_000
SUBSAMPLE_TOP = 4_000

LOG_BE_EDGES = np.linspace(-10.0, 1.0, 111)  # log10(BE) histogram bins
LOG_DEN_EDGES = np.linspace(-4.0, 8.0, 121)  # log10(D / median D) histogram bins


def sample_t(shape: tuple[int, ...], nu: float, rng: np.random.Generator) -> np.ndarray:
    """Standard t-Student(nu) sample, float64.

    Deliberately local to this script rather than added to
    `qgemm.distributions`: this is a diagnostic, and the nu-axis sampler is a
    piece of study infrastructure that PREREGISTRATION.md R15 wants
    implemented and tested on its own terms, not smuggled in through a
    diagnostic. No rescaling is applied -- `BE` is a ratio that is invariant to
    a common rescaling of the operands up to the quantizer's own grid, and for
    nu <= 2 the variance that a rescaling would normalize does not exist
    anyway.
    """
    return np.asarray(rng.standard_t(nu, size=shape), dtype=np.float64)


def config_digest(config: dict) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def hill_alpha(x: np.ndarray, k: int) -> float:
    """Hill estimate of the tail index alpha of `x` from its `k` largest values.

    `alpha = [ (1/k) sum_{i<=k} log X_(i) - log X_(k+1) ]^-1` on the order
    statistics sorted descending. The p-th moment of a tail `P(X > x) ~ x^-a`
    is finite iff `a > p`, so this is what decides whether a mean (p = 1) or a
    variance (p = 2) exists. Standard error is `alpha / sqrt(k)` asymptotically.
    """
    x = np.sort(x[np.isfinite(x) & (x > 0.0)])[::-1]
    if k < 2 or k >= x.size:
        return float("nan")
    logs = np.log(x[: k + 1])
    denom = float(logs[:k].mean() - logs[k])
    return float("inf") if denom <= 0.0 else 1.0 / denom


def max_to_sum(x: np.ndarray, p: float, n_points: int = 120) -> tuple[np.ndarray, np.ndarray]:
    """Trajectory of `R_p(m) = max_{i<=m} |x_i|^p / sum_{i<=m} |x_i|^p` over growing `m`.

    `R_p(m) -> 0` almost surely iff `E|X|^p < inf`, so this answers the
    moment-existence question without choosing a tail fraction: a ratio that
    plateaus -- one observation still owning a fixed share of the whole sum --
    is a divergent moment, visible directly.
    """
    v = np.abs(x[np.isfinite(x)]) ** p
    cumsum = np.cumsum(v)
    cummax = np.maximum.accumulate(v)
    m = np.unique(np.geomspace(10, v.size, n_points).astype(np.int64))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = cummax[m - 1] / cumsum[m - 1]
    return m, ratio


def run_cell(
    nu: float, n: int, trials: int, out_rows: int, base_seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return `(numerator, denominator, BE)`, each of shape `(trials, out_rows**2)`.

    One independent draw of `(A, B)` per trial, MXFP4 quantization active,
    exact accumulation. Rows are trials so that every downstream statistic can
    resample at the trial level.
    """
    rng = np.random.default_rng([base_seed, int(nu * 100), n])
    num = np.empty((trials, out_rows * out_rows), dtype=np.float64)
    den = np.empty_like(num)
    for t in tqdm(range(trials), desc=f"nu={nu:g} n={n}", leave=False):
        a = sample_t((out_rows, n), nu, rng)
        b = sample_t((n, out_rows), nu, rng)
        approx = qgemm(a, b, GemmConfig(seed=t))  # MXFP4 preset; see module docstring
        num[t] = np.abs(approx - a @ b).ravel()
        den[t] = (np.abs(a) @ np.abs(b)).ravel()
    with np.errstate(divide="ignore", invalid="ignore"):
        be = num / den
    return num, den, be


def summarize_cell(nu: float, n: int, num: np.ndarray, den: np.ndarray, be: np.ndarray) -> dict:
    """Every scalar this diagnostic reports for one `(nu, n)` cell."""
    flat_num, flat_den, flat_be = num.ravel(), den.ravel(), be.ravel()
    finite = np.isfinite(flat_be)
    positive = finite & (flat_be > 0.0)
    log_be = np.log10(flat_be[positive])

    # --- step 3: the denominator on its own terms, scale-free ---
    den_med = float(np.median(flat_den))
    den_rel = flat_den / den_med
    row = {
        "nu": nu,
        "n": n,
        "n_samples": int(flat_be.size),
        "n_trials": int(be.shape[0]),
        "be_zero_count": int(np.sum(finite & (flat_be == 0.0))),
        "be_nonfinite_count": int(np.sum(~finite)),
        "den_zero_count": int(np.sum(flat_den == 0.0)),
        "den_median": den_med,
        "den_rel_min": float(den_rel.min()),
        "den_rel_q1e5": float(np.quantile(den_rel, 1e-5)),
        "den_rel_q1e4": float(np.quantile(den_rel, 1e-4)),
        "den_rel_q1e3": float(np.quantile(den_rel, 1e-3)),
        "den_rel_p1": float(np.quantile(den_rel, 0.01)),
        "den_rel_p10": float(np.quantile(den_rel, 0.10)),
        "den_rel_p90": float(np.quantile(den_rel, 0.90)),
        "den_rel_p99": float(np.quantile(den_rel, 0.99)),
        "den_rel_max": float(den_rel.max()),
        "p_den_below_1e-1_med": float(np.mean(den_rel < 1e-1)),
        "p_den_below_1e-2_med": float(np.mean(den_rel < 1e-2)),
        "p_den_below_1e-3_med": float(np.mean(den_rel < 1e-3)),
        "den_iqr_ratio": float(np.quantile(den_rel, 0.75) / np.quantile(den_rel, 0.25)),
    }

    # --- BE marginal ---
    for q in (0.5, 0.9, 0.99, 0.999, 0.9999):
        row["be_q" + f"{q:g}".replace(".", "")] = float(np.quantile(flat_be[finite], q))
    row["be_max"] = float(flat_be[finite].max())
    row["be_mean"] = float(flat_be[finite].mean())
    row["be_std"] = float(flat_be[finite].std())
    row["be_median"] = float(np.median(flat_be[finite]))
    row["be_geomean"] = float(10.0 ** log_be.mean())
    # How far the mean is dragged past the median -- a crude heavy-tail tell.
    row["be_mean_over_median"] = row["be_mean"] / row["be_median"]
    row["be_max_over_median"] = row["be_max"] / row["be_median"]

    # --- step 3 (attribution): where do the largest BE values come from? ---
    k_top = max(1, int(TOP_FRAC * flat_be.size))
    top = np.argpartition(flat_be, -k_top)[-k_top:]
    num_rank = flat_num.argsort().argsort() / (flat_num.size - 1)
    den_rank = flat_den.argsort().argsort() / (flat_den.size - 1)
    row["top_be_num_rank_median"] = float(np.median(num_rank[top]))
    row["top_be_den_rank_median"] = float(np.median(den_rank[top]))
    row["top_be_frac_den_below_p1"] = float(np.mean(den_rank[top] < 0.01))
    row["top_be_frac_num_above_p99"] = float(np.mean(num_rank[top] > 0.99))
    # Counterfactual: the same numerators divided by a *fixed* denominator. If
    # the BE tail were numerator-driven, this would have a comparable spread.
    be_fixed_den = flat_num / den_med
    row["fixed_den_max_over_median"] = float(
        be_fixed_den.max() / np.median(be_fixed_den[be_fixed_den > 0])
    )
    both_pos = (flat_num > 0.0) & (flat_den > 0.0)
    row["corr_log_num_log_den"] = float(
        np.corrcoef(np.log(flat_num[both_pos]), np.log(flat_den[both_pos]))[0, 1]
    )

    # --- step 4: tail index and moment existence ---
    k = int(HILL_PRIMARY_K_FRAC * flat_be.size)
    alpha = hill_alpha(flat_be[finite], k)
    se = alpha / np.sqrt(k) if np.isfinite(alpha) else float("nan")
    row["hill_k"] = k
    row["hill_alpha_be"] = alpha
    row["hill_alpha_be_ci_low"] = alpha - 1.96 * se
    row["hill_alpha_be_ci_high"] = alpha + 1.96 * se
    row["hill_alpha_num"] = hill_alpha(flat_num, k)
    row["hill_alpha_inv_den"] = hill_alpha(1.0 / flat_den[flat_den > 0], k)
    row["mean_exists"] = bool(row["hill_alpha_be_ci_low"] > 1.0)
    row["variance_exists"] = bool(row["hill_alpha_be_ci_low"] > 2.0)

    # --- step 5: log10(BE) as the fallback ---
    centered = np.abs(log_be - np.median(log_be))
    row["log_be_mean"] = float(log_be.mean())
    row["log_be_std"] = float(log_be.std())
    row["log_be_median"] = float(np.median(log_be))
    row["log_be_skew"] = float(np.mean(((log_be - log_be.mean()) / log_be.std()) ** 3))
    row["log_be_kurtosis"] = float(np.mean(((log_be - log_be.mean()) / log_be.std()) ** 4))
    row["log_be_q1e3"] = float(np.quantile(log_be, 1e-3))
    row["log_be_q999e3"] = float(np.quantile(log_be, 1 - 1e-3))
    row["hill_alpha_log_be_centered"] = hill_alpha(centered, k)

    # --- estimator dispersion across disjoint batches of trials ---
    batches = np.array_split(np.arange(be.shape[0]), N_BATCHES)
    estimators = (
        ("mean", lambda v: float(np.mean(v))),
        ("median", lambda v: float(np.median(v))),
        ("geomean", lambda v: float(10.0 ** np.mean(np.log10(v[v > 0.0])))),
    )
    for name, fn in estimators:
        vals = np.array([fn(be[idx].ravel()[np.isfinite(be[idx].ravel())]) for idx in batches])
        row[f"batch_{name}_spread"] = float(vals.max() / vals.min())
        row[f"batch_{name}_cv"] = float(vals.std(ddof=1) / vals.mean())
    return row


def cell_frames(
    nu: float, n: int, num: np.ndarray, den: np.ndarray, be: np.ndarray, rng: np.random.Generator
) -> dict[str, pd.DataFrame]:
    """The per-cell tabular outputs: histograms, Hill curves, max-to-sum, subsample."""
    flat_num, flat_den, flat_be = num.ravel(), den.ravel(), be.ravel()
    positive = np.isfinite(flat_be) & (flat_be > 0.0)
    log_be = np.log10(flat_be[positive])
    den_rel = flat_den / float(np.median(flat_den))
    log_be_centered = np.abs(log_be - np.median(log_be))

    hist_frames = []
    for name, values, edges in (
        ("log10_be", log_be, LOG_BE_EDGES),
        ("log10_den_over_median", np.log10(den_rel[den_rel > 0.0]), LOG_DEN_EDGES),
    ):
        counts, _ = np.histogram(values, bins=edges)
        hist_frames.append(
            pd.DataFrame(
                {
                    "nu": nu,
                    "n": n,
                    "quantity": name,
                    "bin_left": edges[:-1],
                    "bin_right": edges[1:],
                    "count": counts,
                    "n_below": int(np.sum(values < edges[0])),
                    "n_above": int(np.sum(values > edges[-1])),
                    "n_total": int(values.size),
                }
            )
        )

    hill_rows = []
    for name, values in (
        ("be", flat_be[np.isfinite(flat_be)]),
        ("numerator", flat_num),
        ("inv_denominator", 1.0 / flat_den[flat_den > 0.0]),
        ("abs_log10_be_centered", log_be_centered),
    ):
        for frac in HILL_K_FRACS:
            k = int(frac * values.size)
            alpha = hill_alpha(values, k)
            hill_rows.append(
                {
                    "nu": nu,
                    "n": n,
                    "quantity": name,
                    "k_frac": frac,
                    "k": k,
                    "alpha": alpha,
                    "alpha_se": alpha / np.sqrt(k) if np.isfinite(alpha) else float("nan"),
                }
            )

    # Shuffled before accumulating, so the trajectory is over exchangeable
    # draws rather than over whole trials in sequence. The numerator is carried
    # along as the contrast: if the ratio inherited its numerator's tail, the
    # two would plateau together.
    ms_frames = []
    for name, values in (
        ("be", flat_be),
        ("numerator", flat_num),
        ("abs_log10_be_centered", log_be_centered),
    ):
        shuffled = values[rng.permutation(values.size)]
        for p in (1.0, 2.0):
            m, ratio = max_to_sum(shuffled, p)
            ms_frames.append(
                pd.DataFrame({"nu": nu, "n": n, "quantity": name, "p": p, "m": m, "ratio": ratio})
            )

    uniform = rng.choice(flat_be.size, size=min(SUBSAMPLE_UNIFORM, flat_be.size), replace=False)
    top = np.argpartition(flat_be, -SUBSAMPLE_TOP)[-SUBSAMPLE_TOP:]
    take = np.concatenate([uniform, top])
    subsample = pd.DataFrame(
        {
            "nu": nu,
            "n": n,
            "sampling": ["uniform"] * uniform.size + ["top_be"] * top.size,
            "numerator": flat_num[take],
            "denominator": flat_den[take],
            "be": flat_be[take],
        }
    )
    return {
        "hist": pd.concat(hist_frames, ignore_index=True),
        "hill": pd.DataFrame(hill_rows),
        "maxsum": pd.concat(ms_frames, ignore_index=True),
        "subsample": subsample,
    }


def _panel_grid(title: str) -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(
        len(GRID_N), len(GRID_NU), figsize=(4.0 * len(GRID_NU), 3.0 * len(GRID_N)), squeeze=False
    )
    fig.suptitle(title)
    return fig, axes


def figure_histograms(hist: pd.DataFrame, quantity: str, xlabel: str, title: str) -> plt.Figure:
    """Grid of histograms, rows = n, cols = nu, plotted from the saved bin counts."""
    fig, axes = _panel_grid(title)
    for i, n in enumerate(GRID_N):
        for j, nu in enumerate(GRID_NU):
            ax = axes[i][j]
            sel = hist[(hist["quantity"] == quantity) & (hist["n"] == n) & (hist["nu"] == nu)]
            centers = 0.5 * (sel["bin_left"].to_numpy() + sel["bin_right"].to_numpy())
            width = sel["bin_right"].to_numpy() - sel["bin_left"].to_numpy()
            total = float(sel["n_total"].iloc[0])
            ax.bar(centers, sel["count"].to_numpy() / total, width=width, color="C0")
            ax.set_title(f"nu={nu:g}, n={n}", fontsize=9)
            ax.set_yscale("log")
            if i == len(GRID_N) - 1:
                ax.set_xlabel(xlabel, fontsize=8)
            if j == 0:
                ax.set_ylabel("fraction of samples", fontsize=8)
            ax.tick_params(labelsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def figure_denominator(hist: pd.DataFrame, summary: pd.DataFrame) -> plt.Figure:
    """The denominator's own distribution, with its near-zero mass marked."""
    fig = figure_histograms(
        hist,
        "log10_den_over_median",
        "log10( (|A|@|B|)_ij / median )",
        "Denominator distribution, normalized by its own median "
        "(dashed: 1e-1, 1e-2, 1e-3 x median)",
    )
    axes = np.array(fig.axes[: len(GRID_N) * len(GRID_NU)]).reshape(len(GRID_N), len(GRID_NU))
    for i, n in enumerate(GRID_N):
        for j, nu in enumerate(GRID_NU):
            ax = axes[i][j]
            row = summary[(summary["n"] == n) & (summary["nu"] == nu)].iloc[0]
            for exponent in (-1, -2, -3):
                ax.axvline(exponent, color="crimson", ls="--", lw=0.8)
            ax.text(
                0.02,
                0.96,
                f"P(D<0.1 med) = {row['p_den_below_1e-1_med']:.2e}\n"
                f"P(D<0.01 med) = {row['p_den_below_1e-2_med']:.2e}\n"
                f"min/med = {row['den_rel_min']:.2e}",
                transform=ax.transAxes,
                fontsize=6.5,
                va="top",
            )
    return fig


def figure_hill(hill: pd.DataFrame) -> plt.Figure:
    """Hill curves for BE and, for contrast, for the numerator and 1/denominator."""
    fig, axes = _panel_grid(
        "Hill tail index vs tail fraction k (shaded: +/-1.96 SE; y clipped at 6). "
        "alpha>1 => mean exists, alpha>2 => variance exists"
    )
    styles = {"be": "C0", "numerator": "C1", "inv_denominator": "C2"}
    for i, n in enumerate(GRID_N):
        for j, nu in enumerate(GRID_NU):
            ax = axes[i][j]
            for quantity, color in styles.items():
                sel = hill[
                    (hill["quantity"] == quantity) & (hill["n"] == n) & (hill["nu"] == nu)
                ].sort_values("k_frac")
                k_frac = sel["k_frac"].to_numpy()
                alpha = sel["alpha"].to_numpy()
                se = sel["alpha_se"].to_numpy()
                ax.plot(k_frac, alpha, marker="o", ms=3, color=color, label=quantity)
                lo, hi = alpha - 1.96 * se, alpha + 1.96 * se
                ax.fill_between(k_frac, lo, hi, color=color, alpha=0.2)
            ax.axhline(1.0, color="k", ls=":", lw=0.8)
            ax.axhline(2.0, color="k", ls="--", lw=0.8)
            ax.set_xscale("log")
            ax.set_ylim(0, 6)
            ax.set_title(f"nu={nu:g}, n={n}", fontsize=9)
            if i == len(GRID_N) - 1:
                ax.set_xlabel("k / sample size", fontsize=8)
            if j == 0:
                ax.set_ylabel("alpha", fontsize=8)
            ax.tick_params(labelsize=7)
            if i == 0 and j == 0:
                ax.legend(fontsize=6)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def figure_maxsum(maxsum: pd.DataFrame) -> plt.Figure:
    """Max-to-sum ratios: an estimator-free read on which moments exist."""
    fig, axes = plt.subplots(
        len(GRID_N), 2, figsize=(9.0, 3.0 * len(GRID_N)), squeeze=False, sharex=True
    )
    fig.suptitle(
        "Max-to-sum ratio R_p(m) for BE (solid), its numerator alone (dotted)\n"
        "and |log10 BE - median| (dashed). R_p -> 0 iff the p-th moment is finite;\n"
        "a plateau is a divergent moment."
    )
    for i, n in enumerate(GRID_N):
        for jp, p in enumerate((1.0, 2.0)):
            ax = axes[i][jp]
            for jn, nu in enumerate(GRID_NU):
                for quantity, ls in (
                    ("be", "-"),
                    ("numerator", ":"),
                    ("abs_log10_be_centered", "--"),
                ):
                    sel = maxsum[
                        (maxsum["quantity"] == quantity)
                        & (maxsum["p"] == p)
                        & (maxsum["n"] == n)
                        & (maxsum["nu"] == nu)
                    ]
                    ax.plot(
                        sel["m"],
                        sel["ratio"],
                        ls=ls,
                        color=f"C{jn}",
                        lw=1.2,
                        label=f"nu={nu:g}" if quantity == "be" else None,
                    )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(f"n={n}, p={p:g}", fontsize=9)
            ax.set_xlabel("sample size m", fontsize=8)
            if jp == 0:
                ax.set_ylabel("R_p(m)", fontsize=8)
            ax.tick_params(labelsize=7)
            if i == 0 and jp == 0:
                ax.legend(fontsize=6)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def figure_estimator_stability(summary: pd.DataFrame) -> plt.Figure:
    """How far each candidate estimator moves between disjoint batches of trials."""
    fig, axes = plt.subplots(1, len(GRID_N), figsize=(4.2 * len(GRID_N), 3.6), squeeze=False)
    fig.suptitle(
        f"Estimator dispersion across {N_BATCHES} disjoint batches of trials "
        "(max/min over batches; 1.0 = perfectly reproducible)"
    )
    for i, n in enumerate(GRID_N):
        ax = axes[0][i]
        sel = summary[summary["n"] == n].sort_values("nu")
        for name, color in (("mean", "C3"), ("median", "C0"), ("geomean", "C2")):
            ax.plot(
                sel["nu"], sel[f"batch_{name}_spread"], marker="o", ms=4, color=color, label=name
            )
        ax.axhline(1.0, color="k", ls=":", lw=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(list(GRID_NU))
        ax.set_xticklabels([f"{nu:g}" for nu in GRID_NU])
        ax.set_title(f"n={n}", fontsize=9)
        ax.set_xlabel("nu", fontsize=8)
        if i == 0:
            ax.set_ylabel("batch max / batch min", fontsize=8)
            ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


def print_report(summary: pd.DataFrame) -> None:
    """A terminal digest of the numbers the SPEC recommendation has to rest on."""
    pd.set_option("display.width", 200)
    print("\n--- step 2: the BE marginal (note be_max: BE is bounded, not merely light-tailed) ---")
    print(
        summary[
            [
                "nu",
                "n",
                "n_samples",
                "be_median",
                "be_geomean",
                "be_mean",
                "be_std",
                "be_q0999",
                "be_max",
                "be_zero_count",
                "be_nonfinite_count",
            ]
        ].to_string(index=False)
    )
    print("\n--- step 3: denominator near-zero mass (normalized by its own median) ---")
    print(
        summary[
            [
                "nu",
                "n",
                "den_rel_min",
                "den_rel_q1e5",
                "den_rel_q1e3",
                "den_rel_p1",
                "p_den_below_1e-1_med",
                "p_den_below_1e-2_med",
                "den_iqr_ratio",
            ]
        ].to_string(index=False)
    )
    print("\n--- step 3: attribution of the BE tail (top 0.1% of BE) ---")
    print(
        summary[
            [
                "nu",
                "n",
                "top_be_num_rank_median",
                "top_be_den_rank_median",
                "top_be_frac_den_below_p1",
                "top_be_frac_num_above_p99",
                "be_max_over_median",
                "fixed_den_max_over_median",
            ]
        ].to_string(index=False)
    )
    print("\n--- step 4: Hill tail index and moment existence (k = 5% of sample) ---")
    print(
        summary[
            [
                "nu",
                "n",
                "hill_alpha_be",
                "hill_alpha_be_ci_low",
                "hill_alpha_be_ci_high",
                "hill_alpha_num",
                "hill_alpha_inv_den",
                "mean_exists",
                "variance_exists",
            ]
        ].to_string(index=False)
    )
    print("\n--- step 5: log10(BE), and estimator dispersion across batches ---")
    print(
        summary[
            [
                "nu",
                "n",
                "log_be_median",
                "log_be_std",
                "log_be_skew",
                "log_be_kurtosis",
                "hill_alpha_log_be_centered",
                "batch_mean_spread",
                "batch_median_spread",
                "batch_geomean_spread",
            ]
        ].to_string(index=False)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS, help="Independent trials.")
    parser.add_argument(
        "--out-rows", type=int, default=DEFAULT_OUT_ROWS, help="M = N of the output matrix."
    )
    parser.add_argument("--seed", type=int, default=0, help="Base seed for operand draws.")
    args = parser.parse_args()

    config = {
        "script": "check_metric_stability.py",
        "version": 1,
        "grid_nu": list(GRID_NU),
        "grid_n": list(GRID_N),
        "trials": args.trials,
        "out_rows": args.out_rows,
        "seed": args.seed,
        "distribution": "t-student",
        "gemm_config": {
            "preset": "mxfp4",
            "block_size": 32,
            "scale_format": "e8m0",
            "use_global_scale": False,
            "element_format": "e2m1",
            "round_mode": "rtne",
            "rht": False,
            "accum": "exact",
        },
        "hill_k_fracs": list(HILL_K_FRACS),
        "hill_primary_k_frac": HILL_PRIMARY_K_FRAC,
        "n_batches": N_BATCHES,
        "subsample_uniform": SUBSAMPLE_UNIFORM,
        "subsample_top": SUBSAMPLE_TOP,
        "top_frac": TOP_FRAC,
    }
    digest = config_digest(config)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    frames: dict[str, list[pd.DataFrame]] = {"hist": [], "hill": [], "maxsum": [], "subsample": []}
    for n in GRID_N:
        for nu in GRID_NU:
            num, den, be = run_cell(nu, n, args.trials, args.out_rows, args.seed)
            summary_rows.append(summarize_cell(nu, n, num, den, be))
            # A separate stream for the analysis-side randomness (shuffles,
            # subsampling), so it cannot perturb the operand draws.
            rng = np.random.default_rng([args.seed, int(nu * 100), n, 0xA11A])
            for key, frame in cell_frames(nu, n, num, den, be, rng).items():
                frames[key].append(frame)
            print(f"done: nu={nu:g} n={n}")

    summary = pd.DataFrame(summary_rows)
    tables = {key: pd.concat(value, ignore_index=True) for key, value in frames.items()}
    tables["summary"] = summary

    stem = f"metric_stability_{digest}"
    for name, table in tables.items():
        table.to_parquet(OUT_DIR / f"{stem}_{name}.parquet", engine="pyarrow")
    (OUT_DIR / f"{stem}.config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
    )

    figures = {
        "be_hist": figure_histograms(
            tables["hist"],
            "log10_be",
            "log10( BE )",
            "Backward error BE = |A@B - qgemm(A,B)| / (|A|@|B|), MXFP4, log scale",
        ),
        "denominator": figure_denominator(tables["hist"], summary),
        "hill": figure_hill(tables["hill"]),
        "maxsum": figure_maxsum(tables["maxsum"]),
        "estimator_stability": figure_estimator_stability(summary),
    }
    for name, fig in figures.items():
        fig.savefig(OUT_DIR / f"{stem}_{name}.png", dpi=150)
        plt.close(fig)

    print_report(summary)
    print(f"\nWrote {len(tables)} tables and {len(figures)} figures to {OUT_DIR} (stem {stem})")


if __name__ == "__main__":
    main()
