"""Sanity check: is the measurement pipeline itself trustworthy?

This is a **control experiment**, deliberately kept outside the quantized
formats this project studies (no MXFP4/NVFP4/E2M1/block scaling anywhere in
this file). It rounds standard IEEE binary16 (fp16) inner products of
Gaussian operands and checks two things before `backward_error` and the
bootstrap/slope-fitting machinery get used on anything exotic:

1. The **deterministic worst-case bound** `gamma_n = n*u / (1 - n*u)`
   (Higham, "Accuracy and Stability of Numerical Algorithms", 2nd ed., eq.
   3.1) must never be exceeded, over every single trial, for both
   round-to-nearest-even (RTNE) and stochastic rounding (SR). Zero tolerance:
   one violation is a bug in `qgemm.metrics.backward_error`, the trial
   generator, or the fp16 rounding here -- not noise.
2. The empirical **growth rate** of `median(BE)` with `n` is measured via a
   log-log slope fit (with a bootstrap CI), for comparison against the two
   textbook regimes: slope 1.0 (the `gamma_n` worst case) and slope 0.5 (the
   probabilistic `sqrt(n)` regime random data is expected to land in).

The fp16 rounding helpers below (`round_stochastic_fp16`, `fp16_dot_rtne`,
`fp16_dot_sr`) are deliberately local to this script rather than added to
`qgemm.rounding`: fp16 is a reference/control format for validating this
measurement pipeline, not one of the quantized formats under study, the same
reasoning `scripts/check_metric_stability.py` gives for keeping its
t-Student sampler local rather than adding it to `qgemm.distributions`.

Stop condition (do not skip): this script produces a figure and a plain-text
slope statement and stops there. Whether the result supports moving on to
Phase 2 or revisiting Phase 1 is a human call, not made here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

from qgemm.bounds import gamma_n  # noqa: E402
from qgemm.distributions import sample_gaussian  # noqa: E402
from qgemm.metrics import backward_error  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "results" / "diagnostics" / "sanity_gamma_n"

# fp16's unit roundoff, derived rather than hardcoded: u = eps/2, the
# half-ulp bound on round-to-nearest relative error. Cross-checked against
# the 10-bit mantissa fp16 is defined to have.
U_FP16 = float(np.finfo(np.float16).eps) / 2.0
assert np.finfo(np.float16).nmant == 10, "fp16 is expected to have a 10-bit mantissa"
assert U_FP16 == 2.0**-11, f"expected u = 2**-11 for fp16, got {U_FP16!r}"

N_VALUES = (16, 64, 256, 1024, 4096, 16384)
DEFAULT_TRIALS = 2000  # >= 1000 required; this leaves headroom for a stable median/p99
DEFAULT_SEED = 0
N_BOOTSTRAP = 10_000  # matches PREREGISTRATION.md sec 3.2's bootstrap convention
CI_PERCENTILES = (2.5, 97.5)  # 95% percentile-method CI, same convention


def round_stochastic_fp16(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Stochastically round float64 `x` onto the fp16 grid. Returns float64.

    Follows this project's existing SR convention (`quantize_e2m1`, SPEC.md
    "Stochastic rounding"): round to one of the two bracketing fp16 neighbors
    `lo <= x <= hi`, taking `hi` with probability `(x - lo) / (hi - lo)`, so
    `E[Q(x)] = x`. Values already exactly representable in fp16 are returned
    deterministically (matches SPEC.md: "grid values ... are returned
    deterministically"). An explicit `numpy.random.Generator` is required.
    """
    x = np.asarray(x, dtype=np.float64)
    nearest = x.astype(np.float16)
    nearest64 = nearest.astype(np.float64)
    exact = nearest64 == x

    # `nearest` under-shoots x when the RTNE cast rounded down; use that to
    # pick which side of x each neighbor sits on, then step to the other
    # neighbor with nextafter on the fp16 grid itself.
    nearest_is_lo = nearest64 <= x
    with np.errstate(over="ignore"):
        lo16 = np.where(nearest_is_lo, nearest, np.nextafter(nearest, np.float16(-np.inf)))
        hi16 = np.where(nearest_is_lo, np.nextafter(nearest, np.float16(np.inf)), nearest)
    lo = lo16.astype(np.float64)
    hi = hi16.astype(np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        p_hi = np.where(exact, 0.0, (x - lo) / (hi - lo))
    p_hi = np.nan_to_num(p_hi, nan=0.0, posinf=0.0, neginf=0.0)

    draws = rng.random(x.shape)
    take_hi = draws < p_hi
    result = np.where(exact, nearest64, np.where(take_hi, hi, lo))
    return result.astype(np.float64)


def fp16_dot_rtne(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """RTNE fp16 inner product of `a` and `b`, shape `(trials, n)` -> `(trials,)`.

    Every intermediate step is rounded to fp16 -- both the elementwise
    product and each partial sum -- by doing the whole accumulation in
    `float16` dtype arrays. numpy's float16 arithmetic performs the
    operation and rounds once (RTNE), the same "round to nearest,
    correctly" contract the rest of this project relies on for its own
    quantizers, so this is exact and needs no validation against pychop.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    trials, n = a.shape
    acc = np.zeros(trials, dtype=np.float16)
    for i in range(n):
        term = (a[:, i] * b[:, i]).astype(np.float16)
        acc = acc + term
    return acc.astype(np.float64)


def fp16_dot_sr(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """SR fp16 inner product of `a` and `b`, shape `(trials, n)` -> `(trials,)`.

    Same step structure as `fp16_dot_rtne` (product rounded, then each
    partial sum rounded) but every rounding is stochastic via
    `round_stochastic_fp16`. The accumulator is kept in a float64 container
    since SR's output already lies exactly on the fp16 grid -- no precision
    is added by not narrowing the dtype between steps.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    trials, n = a.shape
    acc = np.zeros(trials, dtype=np.float64)
    for i in range(n):
        term = round_stochastic_fp16(a[:, i] * b[:, i], rng)
        acc = round_stochastic_fp16(acc + term, rng)
    return acc


def fit_loglog_slope(n_values: np.ndarray, y_values: np.ndarray) -> tuple[float, float]:
    """Closed-form least-squares slope/intercept of `log10(y)` vs `log10(n)`."""
    x = np.log10(np.asarray(n_values, dtype=np.float64))
    y = np.log10(np.asarray(y_values, dtype=np.float64))
    x_centered = x - x.mean()
    slope = float(np.sum(x_centered * (y - y.mean())) / np.sum(x_centered**2))
    intercept = float(y.mean() - slope * x.mean())
    return slope, intercept


def bootstrap_slope_ci(
    be_by_n: dict[int, np.ndarray], rng: np.random.Generator, n_boot: int = N_BOOTSTRAP
) -> tuple[float, float]:
    """Percentile bootstrap CI of the log-log slope of median(BE) vs n.

    Resampling unit is the independent trial, per PREREGISTRATION.md sec
    3.2 (never individual matrix/vector elements -- here there is only one
    scalar BE per trial, so this is the only sensible unit anyway).
    """
    n_values = np.array(sorted(be_by_n), dtype=np.float64)
    medians_by_replicate = np.empty((len(n_values), n_boot), dtype=np.float64)
    for row, n in enumerate(sorted(be_by_n)):
        be = be_by_n[n]
        idx = rng.integers(0, be.size, size=(n_boot, be.size))
        medians_by_replicate[row] = np.median(be[idx], axis=1)

    x = np.log10(n_values)
    x_centered = x - x.mean()
    y = np.log10(medians_by_replicate)  # (n_values, n_boot)
    y_centered = y - y.mean(axis=0, keepdims=True)
    slopes = (x_centered[:, None] * y_centered).sum(axis=0) / np.sum(x_centered**2)
    lo, hi = np.percentile(slopes, CI_PERCENTILES)
    return float(lo), float(hi)


def run_mode(
    mode: str, trials: int, seed: int, rng_analysis: np.random.Generator
) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    """Run the full n-sweep for one rounding mode. Returns (summary, be_by_n)."""
    rows = []
    be_by_n: dict[int, np.ndarray] = {}
    for n in tqdm(N_VALUES, desc=f"mode={mode}"):
        rng = np.random.default_rng([seed, n, 0 if mode == "rtne" else 1])
        a = sample_gaussian((trials, n), rng)
        b = sample_gaussian((trials, n), rng)
        chat = fp16_dot_rtne(a, b) if mode == "rtne" else fp16_dot_sr(a, b, rng)

        # Treat each trial as its own (1, n) @ (n, 1) GEMM so the exact same
        # `backward_error` this project uses elsewhere computes BE here --
        # this check exists specifically to validate that function, so it
        # must not be reimplemented inline.
        a3 = a[:, None, :]
        b3 = b[:, :, None]
        chat3 = chat[:, None, None]
        be = backward_error(a3, b3, chat3).reshape(trials)
        be_by_n[n] = be

        try:
            bound = gamma_n(n, U_FP16)
        except ValueError:
            bound = None  # n * u >= 1: gamma_n's closed form is out of its domain

        violations = int(np.sum(be > bound)) if bound is not None else None
        rows.append(
            {
                "mode": mode,
                "n": n,
                "trials": trials,
                "median_be": float(np.median(be)),
                "p99_be": float(np.quantile(be, 0.99)),
                "max_be": float(np.max(be)),
                "gamma_n_bound": bound,
                "bound_defined": bound is not None,
                "violations": violations,
            }
        )

        if bound is not None and violations:
            raise SystemExit(
                f"HARD FAILURE: {violations}/{trials} trials at mode={mode}, n={n} exceeded "
                f"the deterministic worst-case bound gamma_n({n}, u)={bound:.6g}. This must "
                "hold with zero exceptions -- treat as a measurement-pipeline bug, not noise."
            )

    return pd.DataFrame(rows), be_by_n


def config_digest(config: dict) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def make_figure(summary: pd.DataFrame, fits: dict[str, dict]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    colors = {"rtne": "C0", "sr": "C1"}
    labels = {"rtne": "RTNE", "sr": "SR"}

    for mode in ("rtne", "sr"):
        sel = summary[summary["mode"] == mode].sort_values("n")
        ax.plot(
            sel["n"],
            sel["median_be"],
            marker="o",
            color=colors[mode],
            label=f"{labels[mode]}: median(BE)",
        )
        ax.plot(
            sel["n"],
            sel["p99_be"],
            marker="s",
            ms=4,
            ls=":",
            color=colors[mode],
            alpha=0.6,
            label=f"{labels[mode]}: p99(BE)",
        )
        fit = fits[mode]
        n_line = np.array([sel["n"].min(), sel["n"].max()], dtype=np.float64)
        fitted = 10.0 ** (fit["intercept"] + fit["slope"] * np.log10(n_line))
        ax.plot(
            n_line,
            fitted,
            color=colors[mode],
            lw=2.0,
            alpha=0.9,
            label=f"{labels[mode]} fit: slope={fit['slope']:.3f} "
            f"[{fit['ci_low']:.3f}, {fit['ci_high']:.3f}]",
        )

    # Reference lines, anchored to the RTNE curve's value at the smallest n
    # so slope is visually comparable regardless of absolute scale.
    anchor_n = summary["n"].min()
    anchor_y = summary.loc[
        (summary["mode"] == "rtne") & (summary["n"] == anchor_n), "median_be"
    ].iloc[0]
    n_ref = np.array([summary["n"].min(), summary["n"].max()], dtype=np.float64)
    reference_slopes = (
        (1.0, "slope 1.0 (gamma_n regime)", "--"),
        (0.5, "slope 0.5 (sqrt(n) regime)", "-."),
    )
    for slope, label, style in reference_slopes:
        y_ref = anchor_y * (n_ref / anchor_n) ** slope
        ax.plot(n_ref, y_ref, color="gray", ls=style, lw=1.2, label=label)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n (inner-product length)")
    ax.set_ylabel("backward error (BE)")
    ax.set_title("fp16 inner-product BE vs n: RTNE/SR against gamma_n / sqrt(n) reference slopes")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    return fig


def main() -> None:
    trials = DEFAULT_TRIALS
    seed = DEFAULT_SEED

    config = {
        "script": "sanity_gamma_n.py",
        "version": 1,
        "n_values": list(N_VALUES),
        "trials": trials,
        "seed": seed,
        "distribution": "gaussian",
        "u_fp16": U_FP16,
        "n_bootstrap": N_BOOTSTRAP,
        "ci_percentiles": list(CI_PERCENTILES),
    }
    digest = config_digest(config)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"sanity_gamma_n_{digest}"

    summaries = []
    fits: dict[str, dict] = {}
    be_by_mode: dict[str, dict[int, np.ndarray]] = {}
    rng_analysis = np.random.default_rng([seed, 0xA11A])

    for mode in ("rtne", "sr"):
        summary, be_by_n = run_mode(mode, trials, seed, rng_analysis)
        summaries.append(summary)
        be_by_mode[mode] = be_by_n

        n_arr = summary.sort_values("n")["n"].to_numpy(dtype=np.float64)
        median_arr = summary.sort_values("n")["median_be"].to_numpy(dtype=np.float64)
        slope, intercept = fit_loglog_slope(n_arr, median_arr)
        boot_rng = np.random.default_rng([seed, 0xB007, 0 if mode == "rtne" else 1])
        ci_low, ci_high = bootstrap_slope_ci(be_by_n, boot_rng)
        fits[mode] = {
            "slope": slope,
            "intercept": intercept,
            "ci_low": ci_low,
            "ci_high": ci_high,
        }

    summary_all = pd.concat(summaries, ignore_index=True)
    summary_all.to_parquet(OUT_DIR / f"{stem}_summary.parquet", engine="pyarrow")
    (OUT_DIR / f"{stem}.config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
    )

    fig = make_figure(summary_all, fits)
    fig_path = OUT_DIR / f"{stem}_slope.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    lines = []
    for mode in ("rtne", "sr"):
        fit = fits[mode]
        half_width = (fit["ci_high"] - fit["ci_low"]) / 2.0
        label = "RTNE" if mode == "rtne" else "SR"
        consistency = "consistent" if abs(fit["slope"] - 0.5) <= half_width else "inconsistent"
        lines.append(
            f"{label}: empirical log-log slope of median(BE) vs n is "
            f"{fit['slope']:.4f} +/- {half_width:.4f} "
            f"(95% CI [{fit['ci_low']:.4f}, {fit['ci_high']:.4f}]), "
            f"{consistency} with sqrt(n) (slope 0.5)."
        )

    undefined_ns = sorted(
        summary_all.loc[~summary_all["bound_defined"], "n"].unique().tolist()
    )
    if undefined_ns:
        lines.append(
            f"gamma_n bound undefined (n * u_fp16 >= 1) at n = {undefined_ns}; "
            "zero-violation check performed only at the remaining n values, "
            "where it passed with zero exceptions across all trials."
        )
    else:
        lines.append("gamma_n bound defined at every tested n; zero-violation check passed.")

    statement = "\n".join(lines)
    print("\n" + statement + "\n")
    (OUT_DIR / f"{stem}_statement.txt").write_text(statement + "\n", encoding="utf-8")

    print(summary_all.to_string(index=False))
    print(f"\nWrote summary, config, figure ({fig_path.name}) and statement to {OUT_DIR}")
    print(
        "\nSTOP: this script only measures and reports. Whether the result supports "
        "moving on to Phase 2 or revisiting Phase 1 is a human decision, not made here."
    )


if __name__ == "__main__":
    main()
