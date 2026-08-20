"""Sanity check: reproduce the qualitative finding of arXiv:2408.02897.

Rasquinha & Tabak (Google), "A Metric Driven Approach to Mixed Precision
Training" -- the methodologically closest prior work to this project. Where
`scripts/sanity_gamma_n.py` validates the measurement pipeline against
*theory* (Higham's deterministic worst-case bound), this validates it against
an *independent published empirical result*. Two different kinds of external
check; neither substitutes for the other.

What the paper does (the facts this script targets)
---------------------------------------------------
* Backward error is defined at the inner-product level as
  `BE = |L.R - Q(L,R)| / (|L|.|R|)`. That is *exactly*
  `qgemm.metrics.backward_error`, which is used here unmodified -- the
  denominator is the product of absolute values, not `|L.R|`.
* Base case: 512x512 matrices.
* Inputs are t-Student with a varying "normality parameter" (degrees of
  freedom); smaller = heavier tails.
* Finding: **FP8 stays much more bounded than INT8 as the tails get heavier.**

That last item is the target. It is a *shape*, not a number: we have neither
the paper's code nor its seeds, so no numeric value is claimed to be matched
and there is no pass/fail tolerance in this script. An earlier placeholder
version of this file was built around a single `REFERENCE_VALUE` and a
`REFERENCE_TOLERANCE`; that contract was dropped deliberately, because a
tolerance gate against a number we cannot source would be fake precision. The
script measures, plots, and states the direction it found. Whether that
direction matches the paper is written down in SPEC.md, including if it does
not.

The two quantization recipes being compared
-------------------------------------------
Both are **per-tensor** -- a single scale for the whole operand matrix, not
the block scaling that the main study uses for MXFP4/NVFP4. Per-tensor is the
classical setting, and it is the setting in which the INT8-vs-FP8 dynamic
range question is actually interesting.

* **INT8**: `qgemm.formats.int8_quantize`. `scale = amax/127`, symmetric,
  codes clipped to `[-127, 127]`.
* **FP8-E4M3**: `scale = amax/448`, then `quantize_e4m3(x/scale) * scale`.

The FP8 scale is a **documented methodological assumption**: the paper's exact
FP8 recipe is not available to this project, so the choice was made here and
is recorded in SPEC.md rather than left as a hidden default. The alternative
-- applying `quantize_e4m3` directly, with no scale, relying on its native
dynamic range -- is not merely a different choice but an untenable one on this
grid: E4M3 has no infinity and overflows to **NaN**, and at `nu = 1` about
0.14% of a 512x512 t-Student sample exceeds 448, which turns the entire
quantized product into NaN. The amax scale makes the comparison both
well-defined and fair, since it is the same recipe INT8 gets.

Accumulation is exact (float64), matching `GemmConfig(accum="exact")`, the
study's primary route: the error measured here comes from operand
quantization and from nothing else.

Stop condition (do not skip): this script measures, writes a figure and a
summary table, and stops. It does not decide what the result implies for the
next phase.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

from qgemm.distributions import sample_gaussian  # noqa: E402
from qgemm.formats import E4M3_MAX, int8_quantize, quantize_e4m3  # noqa: E402
from qgemm.metrics import backward_error  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "results" / "diagnostics" / "sanity_reproduce_2408"

PAPER_ID = "2408.02897"

# The t-Student sampler lives in scripts/check_metric_stability.py, which keeps
# it local on purpose (see its docstring, and PREREGISTRATION.md R15: the nu
# sampler is study infrastructure that must be implemented and tested on its
# own terms, not smuggled into `qgemm.distributions` through a diagnostic). It
# is loaded from the script file rather than reimplemented -- the same
# file-loading pattern tests/test_sanity_gamma_n.py already uses on scripts.
_CMS_PATH = REPO_ROOT / "scripts" / "check_metric_stability.py"
_spec = importlib.util.spec_from_file_location("check_metric_stability", _CMS_PATH)
_cms = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cms)
sample_t = _cms.sample_t

# The nu axis. `None` is the Gaussian limit (nu = infinity), sampled with
# `qgemm.distributions.sample_gaussian` rather than by passing a huge nu.
NU_GRID: tuple[tuple[str, float | None], ...] = (
    ("1", 1.0),
    ("2", 2.0),
    ("3", 3.0),
    ("5", 5.0),
    ("8", 8.0),
    ("15", 15.0),
    ("30", 30.0),
    ("gaussian", None),
)

MATRIX_SIZE = 512  # the paper's base case: 512x512, so the contraction dim is 512 too
DEFAULT_TRIALS = 50  # 50 x 512^2 = 13.1M inner products per (nu, format) cell
DEFAULT_SEED = 0

FORMATS = ("int8", "fp8_e4m3")
FORMAT_LABELS = {
    "int8": "INT8 (per-tensor, amax/127)",
    "fp8_e4m3": "FP8-E4M3 (per-tensor, amax/448)",
}

# Pooled quantiles reported per cell. p50 is the headline; p25/p75 are the
# spread indicator drawn as a band in the figure; p99 shows the tail.
QUANTILES = (0.25, 0.50, 0.75, 0.99)


def fp8_e4m3_per_tensor(x: np.ndarray) -> np.ndarray:
    """Per-tensor-scaled FP8-E4M3 quantization; float64 in, float64 out.

    `scale = amax/448` then quantize on the E4M3 grid, deliberately mirroring
    `int8_quantize`'s `amax/127` so the two formats differ only in the *grid*
    they land on and not in how they are scaled. See the module docstring for
    why this recipe was chosen over unscaled `quantize_e4m3`.

    With this scale the largest-magnitude element maps to exactly `+-448`, so
    E4M3 overflow (which would produce NaN, not saturation) cannot occur. The
    cost is at the other end: elements more than a factor `448 * 2**9` below
    amax fall under the smallest E4M3 subnormal and flush to zero. That
    trade -- top of the range guaranteed, bottom of the range at risk -- is
    the same trade INT8 makes, on a grid with far more dynamic range.
    """
    if x.dtype != np.float64:
        raise TypeError(f"expected float64 input, got {x.dtype}")
    amax = np.max(np.abs(x))
    if amax == 0.0:
        return np.zeros_like(x)
    scale = amax / E4M3_MAX
    return quantize_e4m3(x / scale) * scale


QUANTIZERS = {"int8": int8_quantize, "fp8_e4m3": fp8_e4m3_per_tensor}


def sample_operand(
    shape: tuple[int, ...], nu: float | None, rng: np.random.Generator
) -> np.ndarray:
    """One operand: t-Student(nu), or standard Gaussian when `nu is None`."""
    if nu is None:
        return sample_gaussian(shape, rng)
    return sample_t(shape, nu, rng)


def run_cell(
    nu_index: int, nu: float | None, fmt: str, trials: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """All BE values for one `(nu, format)` cell, plus the per-trial medians.

    The RNG stream is keyed on `(seed, nu_index, trial)` and **not** on the
    format, so INT8 and FP8 see bit-identical operand matrices at every trial.
    The comparison between the two series is therefore paired: any difference
    between them is the quantizer, not the draw.
    """
    quantize = QUANTIZERS[fmt]
    pooled = []
    trial_medians = np.empty(trials, dtype=np.float64)

    for trial in tqdm(range(trials), desc=f"nu={NU_GRID[nu_index][0]} {fmt}", leave=False):
        rng = np.random.default_rng([seed, nu_index, trial])
        a = sample_operand((MATRIX_SIZE, MATRIX_SIZE), nu, rng)
        b = sample_operand((MATRIX_SIZE, MATRIX_SIZE), nu, rng)

        # Exact (float64) accumulation of the reconstructed operands, so the
        # only error present is the operand quantization.
        chat = quantize(a) @ quantize(b)
        be = backward_error(a, b, chat).ravel()
        pooled.append(be)
        trial_medians[trial] = float(np.median(be))

    return np.concatenate(pooled), trial_medians


def summarize_cell(
    nu_label: str, nu: float | None, fmt: str, be: np.ndarray, trial_medians: np.ndarray
) -> dict:
    q = np.quantile(be, QUANTILES)
    return {
        "nu_label": nu_label,
        "nu": np.inf if nu is None else nu,
        "format": fmt,
        "n_inner_products": int(be.size),
        "trials": int(trial_medians.size),
        "median_be": float(q[1]),
        "p25_be": float(q[0]),
        "p75_be": float(q[2]),
        "p99_be": float(q[3]),
        "max_be": float(np.max(be)),
        # Stability of the headline number across independent trials: if these
        # two bracket the pooled median tightly, the median is not seed noise.
        "trial_median_min": float(np.min(trial_medians)),
        "trial_median_max": float(np.max(trial_medians)),
        "n_nonfinite": int(np.sum(~np.isfinite(be))),
    }


def config_digest(config: dict) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def make_figure(summary: pd.DataFrame) -> plt.Figure:
    """median(BE) vs nu, one series per format, log error axis.

    nu is drawn on a categorical axis (evenly spaced grid positions) rather
    than a numeric one: the grid is irregular and its last entry is the
    Gaussian limit, which has no finite position on a nu axis.
    """
    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    labels = [label for label, _ in NU_GRID]
    positions = np.arange(len(labels), dtype=np.float64)
    colors = {"int8": "C3", "fp8_e4m3": "C0"}
    markers = {"int8": "s", "fp8_e4m3": "o"}

    for fmt in FORMATS:
        sel = summary[summary["format"] == fmt].set_index("nu_label").loc[labels]
        ax.plot(
            positions,
            sel["median_be"],
            marker=markers[fmt],
            color=colors[fmt],
            lw=2.0,
            label=f"{FORMAT_LABELS[fmt]}: median",
        )
        ax.fill_between(
            positions,
            sel["p25_be"],
            sel["p75_be"],
            color=colors[fmt],
            alpha=0.18,
            lw=0,
            label=f"{FORMAT_LABELS[fmt]}: p25-p75",
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_yscale("log")
    ax.set_xlabel("t-Student normality parameter nu  (left = heavier tails)")
    ax.set_ylabel("backward error  |L.R - Q(L,R)| / (|L|.|R|)")
    ax.set_title(
        f"arXiv:{PAPER_ID} reproduction: per-tensor INT8 vs FP8-E4M3\n"
        f"{MATRIX_SIZE}x{MATRIX_SIZE} operands, exact accumulation"
    )
    ax.grid(True, which="both", axis="y", alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    return fig


def make_ratio_figure(summary: pd.DataFrame) -> plt.Figure:
    """The same data as one number per nu: how many times worse INT8 is.

    This is the paper's claim reduced to a single curve, so that "FP8 stays
    more bounded as the tails get heavier" is a *slope* to read off rather
    than a gap to eyeball between two log-scaled series.
    """
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    labels = [label for label, _ in NU_GRID]
    positions = np.arange(len(labels), dtype=np.float64)
    indexed = summary.set_index(["format", "nu_label"])
    ratio = np.array(
        [
            indexed.loc[("int8", label), "median_be"]
            / indexed.loc[("fp8_e4m3", label), "median_be"]
            for label in labels
        ]
    )
    ax.plot(positions, ratio, marker="D", color="C2", lw=2.0)
    ax.axhline(1.0, color="gray", ls="--", lw=1.2, label="parity (no difference)")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_yscale("log")
    ax.set_xlabel("t-Student normality parameter nu  (left = heavier tails)")
    ax.set_ylabel("median BE(INT8) / median BE(FP8)")
    ax.set_title("How much worse per-tensor INT8 is than per-tensor FP8-E4M3, per nu")
    ax.grid(True, which="both", axis="y", alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    config = {
        "script": "sanity_reproduce_2408.py",
        "version": 1,
        "paper_id": PAPER_ID,
        "nu_grid": [label for label, _ in NU_GRID],
        "matrix_size": MATRIX_SIZE,
        "trials": args.trials,
        "seed": args.seed,
        "formats": list(FORMATS),
        "int8_recipe": "per-tensor symmetric, scale = amax/127, codes clipped to [-127, 127]",
        "fp8_recipe": "per-tensor, scale = amax/448, quantize_e4m3(x/scale) * scale",
        "accumulation": "exact (float64)",
        "distribution": "t-student (gaussian at the nu = infinity end)",
        "quantiles": list(QUANTILES),
    }
    digest = config_digest(config)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"sanity_reproduce_2408_{digest}"

    started = time.perf_counter()
    rows = []
    for nu_index, (nu_label, nu) in enumerate(tqdm(NU_GRID, desc="nu grid")):
        for fmt in FORMATS:
            be, trial_medians = run_cell(nu_index, nu, fmt, args.trials, args.seed)
            rows.append(summarize_cell(nu_label, nu, fmt, be, trial_medians))
            del be
    elapsed = time.perf_counter() - started

    summary = pd.DataFrame(rows)
    summary["runtime_seconds_total"] = elapsed
    summary.to_parquet(OUT_DIR / f"{stem}_summary.parquet", engine="pyarrow")
    config["runtime_seconds_total"] = round(elapsed, 1)
    (OUT_DIR / f"{stem}.config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
    )

    fig = make_figure(summary)
    fig.savefig(OUT_DIR / f"{stem}_be_vs_nu.png", dpi=150)
    plt.close(fig)
    fig = make_ratio_figure(summary)
    fig.savefig(OUT_DIR / f"{stem}_ratio.png", dpi=150)
    plt.close(fig)

    indexed = summary.set_index(["format", "nu_label"])
    lines = [
        f"Reproduction target: arXiv:{PAPER_ID}, qualitative only "
        "(no shared code or seeds; no numeric value is claimed to be matched).",
        f"{args.trials} trials per cell, {MATRIX_SIZE}x{MATRIX_SIZE} operands, "
        f"seed {args.seed}, {elapsed:.1f}s total.",
        "",
        "median BE, INT8 / FP8-E4M3 / ratio, by nu:",
    ]
    for label, _ in NU_GRID:
        i8 = indexed.loc[("int8", label), "median_be"]
        f8 = indexed.loc[("fp8_e4m3", label), "median_be"]
        lines.append(f"  nu={label:<9} INT8 {i8:.4g}   FP8 {f8:.4g}   ratio {i8 / f8:.3g}x")

    ratio_heavy = (
        indexed.loc[("int8", "1"), "median_be"] / indexed.loc[("fp8_e4m3", "1"), "median_be"]
    )
    ratio_light = (
        indexed.loc[("int8", "gaussian"), "median_be"]
        / indexed.loc[("fp8_e4m3", "gaussian"), "median_be"]
    )
    lines += [
        "",
        f"INT8/FP8 median ratio at the heaviest tail (nu=1): {ratio_heavy:.3g}x",
        f"INT8/FP8 median ratio at the Gaussian end:         {ratio_light:.3g}x",
        f"Change in that ratio across the grid:              {ratio_heavy / ratio_light:.3g}x",
        "",
        "The paper's claim reproduces qualitatively if and only if the ratio grows as nu "
        "decreases. It does NOT reproduce if the ratio is flat (both formats degrade "
        "alike) or falls below 1 (the effect is inverted). Read the number above, not "
        "the expectation.",
    ]
    statement = "\n".join(lines)
    (OUT_DIR / f"{stem}_statement.txt").write_text(statement + "\n", encoding="utf-8")

    print("\n" + statement + "\n")
    print(summary.drop(columns=["runtime_seconds_total"]).to_string(index=False))
    print(f"\nWrote summary, config, two figures and statement to {OUT_DIR} (stem {stem})")
    print(
        "\nSTOP: this script only measures and reports. What the result implies for the "
        "next phase is a human decision, not made here."
    )


if __name__ == "__main__":
    main()
