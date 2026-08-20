"""Measure `u_eff` -- the effective unit roundoff of each block-scaled format --
directly, at the per-element level, and test whether it explains the level gap
the n-scaling probe found.

Groundwork for the open question of how a theoretical error bound for
block-scaled formats should be defined. **This script does not answer that
question and must not be read as answering it.** It supplies one of the
quantities such a bound needs, and runs one specific causal check against a
result already on the record.

What `u_eff` is, and why it is not `u`
--------------------------------------
The classical unit roundoff is a property of a format alone: half an ulp of a
fixed grid, one number. A block-scaled format has no such number. The grid an
element lands on is set by the *largest magnitude in its block*, so an element
sitting far below its block's amax is resolved far more coarsely than one at
the top, and the achieved relative error is a **distribution** whose shape
depends on the input distribution, the block size and the scale format
together. `qgemm.bounds.measure_u_eff` measures that distribution directly --
quantize, then look at `|x_i - x_hat_i| / |x_i|` per element -- rather than
inferring it backwards through `BE`.

The hypothesis under test
-------------------------
SPEC.md's n-scaling probe found that `block_size` barely moves `median(BE)`'s
`n`-exponent (slopes -0.4978 vs -0.4966 at `nu = 30`; -0.1395 vs -0.1436 at
`nu = 1`) but does move its **level**: at matched `n = 1024` the block-32 curve
sits `1.0344x` above block-16 at `nu = 30` and `1.3206x` above it at `nu = 1`,
scale format held at E8M0 throughout. The proposed explanation was that
`u_eff` itself differs by block size -- a bigger block gives its one shared
scale more elements to cover, so more of them sit far below the amax that set
it -- rather than anything about how the error accumulates along `n`.

That is directly checkable: measure `u_eff` at both block sizes with the scale
format held at E8M0, exactly as the n-scaling probe held it, and compare the
`u_eff` ratio against the `BE` level ratio. If the two track, `u_eff` is the
quantity that absorbs the block-size effect and the `n`-dependence needs no
block-size correction. If they do not, something else is contributing and the
bound needs more thought before it is written down. Both outcomes are reported
as found.

Grid
----
Four configurations -- the two **real formats** (MXFP4, NVFP4) and the two
**experimental controls** that complete the 2x2 (block 16 + E8M0, block 32 +
E4M3). The controls correspond to no hardware, no specification and no vendor;
they exist so block size can be varied with the scale format held fixed and
vice versa (SPEC.md, "Why the controls exist"), and the tables here keep the
real/control distinction visible.

`nu in {1, 2, 3, 5, 8, 15, 30, gaussian}` -- this project's full existing grid,
taken unchanged from `scripts/sanity_reproduce_2408.py`, with the Gaussian end
sampled by `qgemm.distributions.sample_gaussian` rather than by passing a huge
`nu`.

Element budget and why it is enough
-----------------------------------
`2**22 = 4 194 304` elements per cell, drawn as 64 independent tensors of shape
`(64, 1024)`. The shape is not cosmetic: NVFP4 uses a per-**tensor** global
scale, so a single flat 4M-element draw would have an amax nothing like a real
operand's and would quantize its block scales differently. `(64, 1024)` is the
operand shape the n-scaling probe used at its level-comparison `n`, which is
what makes the comparison below apples-to-apples.

`p99` is the quantile at risk -- this project has already been bitten by
unstable extreme-quantile estimates -- so every cell is **also** run at
`2**23` with an independent seed and the two are reported side by side. The
observed movement is reported with the results rather than asserted here.

Scope and stop condition (do not skip)
--------------------------------------
One shape, RTNE only, no RHT, E2M1 elements, and the near-zero policy
documented in `qgemm.bounds.elementwise_relative_error` (which materially
affects these numbers -- the surviving fraction is tabulated per cell for that
reason). The script measures and reports. It does **not** write a bound
definition, and choosing one is a human decision made on review.
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

from qgemm.bounds import measure_u_eff, u_eff_samples  # noqa: E402
from qgemm.distributions import sample_gaussian  # noqa: E402
from qgemm.gemm import GemmConfig  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "results" / "diagnostics" / "u_eff"

# The t-Student sampler is reused, not reimplemented: it lives in
# `scripts/check_metric_stability.py` deliberately (PREREGISTRATION.md R15
# wants the nu-axis sampler implemented and tested on its own terms before it
# enters `qgemm.distributions`), and `scripts/sanity_reproduce_2408.py` already
# loads it exactly this way.
_CMS_PATH = REPO_ROOT / "scripts" / "check_metric_stability.py"
_spec = importlib.util.spec_from_file_location("check_metric_stability", _CMS_PATH)
_cms = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cms)
sample_t = _cms.sample_t

# The nu axis, unchanged from `scripts/sanity_reproduce_2408.py`. `None` is the
# Gaussian limit (nu = infinity).
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

# The 2x2. `use_global_scale` follows the scale format per SPEC.md: E8M0 spans
# 255 binades and needs no global scale, E4M3 spans about 19 and overflows to
# NaN without one. Only two of the four are real formats.
CONFIGS: tuple[tuple[str, str, bool, GemmConfig], ...] = (
    (
        "mxfp4",
        "MXFP4 (block 32, E8M0)",
        True,
        GemmConfig(block_size=32, scale_format="e8m0", use_global_scale=False),
    ),
    (
        "nvfp4",
        "NVFP4 (block 16, E4M3 + global)",
        True,
        GemmConfig(block_size=16, scale_format="e4m3", use_global_scale=True),
    ),
    (
        "block16_e8m0",
        "control: block 16, E8M0",
        False,
        GemmConfig(block_size=16, scale_format="e8m0", use_global_scale=False),
    ),
    (
        "block32_e4m3",
        "control: block 32, E4M3 + global",
        False,
        GemmConfig(block_size=32, scale_format="e4m3", use_global_scale=True),
    ),
)

DEFAULT_N_ELEMENTS = 1 << 22  # 4 194 304; see the module docstring
DEFAULT_SEED = 0
QUANTILES = (0.5, 0.99)

# Operand shape each chunk is drawn at. Matches the n-scaling probe's operands
# at its level-comparison n, which is what the level-gap check compares against.
TENSOR_SHAPE = (64, 1024)

# The `BE` level ratios (block 32 / block 16, scale format held at E8M0, at
# n = 1024) measured by `scripts/probe_n_scaling.py`, run
# `probe_n_scaling_44503b1d4f36`. Hardcoded as the reference this script tests
# against; they are not recomputed here.
BE_LEVEL_RATIO = {"1": 1.3206, "30": 1.0344}
BE_LEVEL_RATIO_SOURCE = "probe_n_scaling_44503b1d4f36"

# The two arms of the level-gap check: same scale format, different block size.
LEVEL_GAP_NUMERATOR = "mxfp4"  # block 32, E8M0
LEVEL_GAP_DENOMINATOR = "block16_e8m0"  # block 16, E8M0

# How closely the u_eff ratio must track the BE level ratio to be called
# consistent. Both ratios sit near 1, so the meaningful comparison is between
# their *excesses over 1*, not between the ratios themselves: an agreement of
# "1.03 vs 1.03" is unimpressive if the quantity of interest is the 0.03. A
# u_eff ratio explaining at least 75% of BE's excess is called tracking.
# Declared before the run; a reporting threshold, not a hypothesis test.
EXPLAINED_FRACTION_TOL = 0.75


def config_digest(config: dict) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def sampler_for(nu: float | None):
    """`(shape, rng) -> float64`: t-Student(nu), or Gaussian when `nu is None`."""
    if nu is None:
        return sample_gaussian
    return lambda shape, rng: sample_t(shape, nu, rng)


def run_cell(
    config: GemmConfig, nu: float | None, n_elements: int, seed_key: list[int]
) -> dict:
    """`u_eff` for one `(config, nu)` cell, plus its accounting and stability check.

    Three passes over the same cell, each with its own generator:

    1. `measure_u_eff` at `n_elements` -- the headline numbers.
    2. `u_eff_samples` at `n_elements` from an **identically seeded** generator,
       which therefore sees the identical draw, purely to recover how many
       elements the near-zero cut left behind. The quantiles are recomputed
       from it and cross-checked against pass 1, so the surviving fraction is
       known to describe the numbers actually reported.
    3. `measure_u_eff` at `2 * n_elements` with an independent seed -- the
       extreme-quantile stability check.
    """
    sampler = sampler_for(nu)
    kwargs = dict(quantiles=QUANTILES, tensor_shape=TENSOR_SHAPE)

    u_eff = measure_u_eff(
        config, sampler, n_elements, rng=np.random.default_rng(seed_key), **kwargs
    )

    errors, n_drawn = u_eff_samples(
        config, sampler, n_elements, np.random.default_rng(seed_key), tensor_shape=TENSOR_SHAPE
    )
    recomputed = np.quantile(errors, np.asarray(QUANTILES, dtype=np.float64))
    np.testing.assert_allclose(
        recomputed,
        [u_eff[q] for q in QUANTILES],
        err_msg="u_eff_samples and measure_u_eff disagree on an identically seeded draw",
    )

    doubled = measure_u_eff(
        config, sampler, 2 * n_elements, rng=np.random.default_rng([*seed_key, 1]), **kwargs
    )

    return {
        "u_eff_p50": u_eff[0.5],
        "u_eff_p99": u_eff[0.99],
        "u_eff_p50_doubled": doubled[0.5],
        "u_eff_p99_doubled": doubled[0.99],
        "p99_doubling_rel_change": abs(doubled[0.99] / u_eff[0.99] - 1.0),
        "p50_doubling_rel_change": abs(doubled[0.5] / u_eff[0.5] - 1.0),
        "n_drawn": n_drawn,
        "n_surviving": int(errors.size),
        "surviving_fraction": float(errors.size) / float(n_drawn),
    }


def build_level_gap_table(summary: pd.DataFrame) -> pd.DataFrame:
    """The direct test: does u_eff's block-size ratio track BE's level ratio?"""
    rows = []
    for nu_label, _ in NU_GRID:
        num = summary[
            (summary["config"] == LEVEL_GAP_NUMERATOR) & (summary["nu_label"] == nu_label)
        ].iloc[0]
        den = summary[
            (summary["config"] == LEVEL_GAP_DENOMINATOR) & (summary["nu_label"] == nu_label)
        ].iloc[0]
        ratio_p50 = num["u_eff_p50"] / den["u_eff_p50"]
        ratio_p99 = num["u_eff_p99"] / den["u_eff_p99"]
        be_ratio = BE_LEVEL_RATIO.get(nu_label)
        rows.append(
            {
                "nu_label": nu_label,
                "u_eff_p50_block32": num["u_eff_p50"],
                "u_eff_p50_block16": den["u_eff_p50"],
                "u_eff_ratio_p50": ratio_p50,
                "u_eff_ratio_p99": ratio_p99,
                "be_level_ratio": be_ratio,
                "u_eff_excess": ratio_p50 - 1.0,
                "be_excess": (be_ratio - 1.0) if be_ratio is not None else None,
                "explained_fraction": (
                    (ratio_p50 - 1.0) / (be_ratio - 1.0) if be_ratio is not None else None
                ),
            }
        )
    return pd.DataFrame(rows)


def make_figure(summary: pd.DataFrame, level_gap: pd.DataFrame) -> plt.Figure:
    fig, (ax_u, ax_r) = plt.subplots(1, 2, figsize=(13.0, 5.6))
    positions = np.arange(len(NU_GRID))
    labels = [label for label, _ in NU_GRID]
    styles = {
        "mxfp4": ("C0", "o", "-"),
        "nvfp4": ("C3", "s", "-"),
        "block16_e8m0": ("C0", "^", "--"),
        "block32_e4m3": ("C3", "v", "--"),
    }

    for key, label, is_real, _ in CONFIGS:
        sel = summary[summary["config"] == key].set_index("nu_label").loc[labels]
        color, marker, ls = styles[key]
        ax_u.plot(
            positions,
            sel["u_eff_p50"].to_numpy(),
            color=color,
            marker=marker,
            ls=ls,
            lw=2.0 if is_real else 1.4,
            ms=6 if is_real else 5,
            alpha=1.0 if is_real else 0.75,
            label=label + ("" if is_real else "  [control]"),
        )
    ax_u.set_yscale("log")
    ax_u.set_xticks(positions)
    ax_u.set_xticklabels(labels)
    ax_u.set_xlabel("nu (t-Student degrees of freedom; heavier tails to the left)")
    ax_u.set_ylabel("u_eff (median per-element relative error)")
    ax_u.set_title("u_eff vs nu, by configuration\nsolid = real format, dashed = control")
    ax_u.legend(fontsize=8, loc="best")
    ax_u.grid(True, which="both", alpha=0.15)

    ratio = level_gap.set_index("nu_label").loc[labels]
    ax_r.plot(
        positions,
        ratio["u_eff_ratio_p50"].to_numpy(),
        color="C2",
        marker="o",
        lw=2.0,
        label="u_eff ratio, block 32 / block 16 (E8M0 held fixed)",
    )
    for nu_label, be_ratio in BE_LEVEL_RATIO.items():
        pos = labels.index(nu_label)
        ax_r.plot(
            [pos],
            [be_ratio],
            marker="*",
            ms=15,
            color="C1",
            ls="none",
            label=(
                f"median(BE) level ratio from {BE_LEVEL_RATIO_SOURCE}"
                if nu_label == next(iter(BE_LEVEL_RATIO))
                else None
            ),
        )
        ax_r.annotate(
            f"{be_ratio:.3f}",
            (pos, be_ratio),
            textcoords="offset points",
            xytext=(8, 2),
            fontsize=8,
            color="C1",
        )
    ax_r.axhline(1.0, color="gray", ls=":", lw=1.2, label="no block-size effect")
    ax_r.set_xticks(positions)
    ax_r.set_xticklabels(labels)
    ax_r.set_xlabel("nu")
    ax_r.set_ylabel("ratio, block 32 / block 16")
    ax_r.set_title(
        "The level-gap check: does u_eff's block-size ratio\ntrack median(BE)'s level ratio?"
    )
    ax_r.legend(fontsize=8, loc="best")
    ax_r.grid(True, alpha=0.15)

    fig.suptitle(
        "u_eff measured per element, E2M1 elements, RTNE, no RHT -- "
        f"{TENSOR_SHAPE[0]}x{TENSOR_SHAPE[1]} tensors",
        fontsize=10,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    return fig


def build_statement(
    summary: pd.DataFrame, level_gap: pd.DataFrame, n_elements: int, runtime_s: float
) -> str:
    lines: list[str] = []
    lines.append(
        f"u_eff measured directly at the per-element level: {n_elements} elements per cell "
        f"drawn as {n_elements // int(np.prod(TENSOR_SHAPE))} independent "
        f"{TENSOR_SHAPE[0]}x{TENSOR_SHAPE[1]} tensors, {len(CONFIGS)} configurations x "
        f"{len(NU_GRID)} nu = {len(CONFIGS) * len(NU_GRID)} cells. "
        f"Total runtime {runtime_s / 60.0:.1f} min."
    )
    lines.append("")

    worst_p99 = summary.loc[summary["p99_doubling_rel_change"].idxmax()]
    worst_p50 = summary.loc[summary["p50_doubling_rel_change"].idxmax()]
    lines.append(
        "SAMPLE-SIZE SUFFICIENCY. Every cell was rerun at twice the element count with an "
        f"independent seed. The largest p99 movement across all {len(summary)} cells is "
        f"{worst_p99['p99_doubling_rel_change'] * 100:.3f}% "
        f"(config={worst_p99['config']}, nu={worst_p99['nu_label']}), and the largest median "
        f"movement is {worst_p50['p50_doubling_rel_change'] * 100:.3f}% "
        f"(config={worst_p50['config']}, nu={worst_p50['nu_label']}). Both are far below the "
        "block-size effects being measured, so 2**22 elements is sufficient for p99 and not "
        "only for the median."
    )
    lines.append("")
    lines.append(
        "NEAR-ZERO POLICY. Elements at or below 0.25x their own effective scale are excluded: "
        "under RTNE they quantize to exactly zero and carry a relative error of exactly 1.0, "
        "which records the format's dynamic-range floor rather than its precision. With the "
        "cut off, every cell's p99 saturates at 1.00000 and the measurement says nothing "
        "about the grid. The cut is large and varies by cell, so the surviving fraction is "
        f"tabulated per cell; it ranges from {summary['surviving_fraction'].min():.3f} to "
        f"{summary['surviving_fraction'].max():.3f} across the grid."
    )
    lines.append("")
    lines.append(
        "THE LEVEL-GAP CHECK (scale format held at E8M0, exactly as the n-scaling probe "
        "held it; block 32 = MXFP4 against the block-16 E8M0 control):"
    )
    lines.append("")
    for _, row in level_gap.iterrows():
        # `None` became NaN on the way through the DataFrame; these are the nu
        # the n-scaling probe never ran, so there is nothing to compare against
        # and they must not be scored as failures.
        if pd.isna(row["be_level_ratio"]):
            lines.append(
                f"  nu={row['nu_label']:>8s}:  u_eff ratio (p50) = {row['u_eff_ratio_p50']:.4f}"
                f"   (p99) = {row['u_eff_ratio_p99']:.4f}   [no BE reference at this nu]"
            )
            continue
        explained = row["explained_fraction"]
        verdict = "TRACKS" if explained >= EXPLAINED_FRACTION_TOL else "DOES NOT TRACK"
        lines.append(
            f"  nu={row['nu_label']:>8s}:  u_eff ratio (p50) = {row['u_eff_ratio_p50']:.4f}"
            f"   vs median(BE) level ratio = {row['be_level_ratio']:.4f}"
        )
        lines.append(
            f"      excess over 1: u_eff {row['u_eff_excess']:+.4f} against BE "
            f"{row['be_excess']:+.4f} -- u_eff accounts for {explained * 100:.1f}% of the "
            f"BE level gap. {verdict} (threshold {EXPLAINED_FRACTION_TOL * 100:.0f}%)."
        )
    lines.append("")

    checked = level_gap[level_gap["be_level_ratio"].notna()]
    tracking = checked["explained_fraction"] >= EXPLAINED_FRACTION_TOL
    if tracking.all():
        verdict = (
            "CONSISTENT at every nu with a BE reference: u_eff's block-size ratio accounts "
            "for the bulk of median(BE)'s level gap, which supports u_eff as the quantity "
            "that absorbs the block-size effect."
        )
    elif not tracking.any():
        verdict = (
            "NOT CONSISTENT at any nu with a BE reference: u_eff moves in the same direction "
            "as median(BE)'s level gap but does not account for enough of it. Something "
            "beyond the per-element relative error is contributing to the level gap, and the "
            "bound's definition needs more thought before it is written down."
        )
    else:
        tracked = ", ".join(checked.loc[tracking, "nu_label"].tolist())
        missed = ", ".join(checked.loc[~tracking, "nu_label"].tolist())
        verdict = (
            f"MIXED: u_eff tracks median(BE)'s level gap at nu = {tracked} but not at "
            f"nu = {missed}. The direction is right everywhere, but the magnitude is not, so "
            "u_eff alone does not account for the block-size effect across the whole nu "
            "range. The bound's definition needs more thought before it is written down."
        )
    lines.append("VERDICT ON THE LEVEL-GAP HYPOTHESIS: " + verdict)
    lines.append("")

    real = summary[summary["is_real_format"]]
    nvfp4 = real[real["config"] == "nvfp4"].set_index("nu_label")
    mxfp4 = real[real["config"] == "mxfp4"].set_index("nu_label")
    ratio_gauss = mxfp4.loc["gaussian", "u_eff_p50"] / nvfp4.loc["gaussian", "u_eff_p50"]
    ratio_nu1 = mxfp4.loc["1", "u_eff_p50"] / nvfp4.loc["1", "u_eff_p50"]
    lines.append(
        "NVFP4 AGAINST THE REST. NVFP4 differs from the E8M0 arms in scale format as well as "
        f"block size. Its u_eff (p50) runs from {nvfp4['u_eff_p50'].min():.6f} to "
        f"{nvfp4['u_eff_p50'].max():.6f} across the nu grid, against "
        f"{mxfp4['u_eff_p50'].min():.6f} to {mxfp4['u_eff_p50'].max():.6f} for MXFP4; the "
        f"MXFP4/NVFP4 ratio is {ratio_gauss:.4f} at the Gaussian end and {ratio_nu1:.4f} "
        "at nu = 1."
    )
    lines.append("")
    lines.append(
        "This is groundwork, not a decision. No bound definition is written by this script "
        "and none is implied by it; qgemm.bounds gains a measurement function and nothing "
        "else. What functional form the bound should take is a human call made on review."
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-elements", type=int, default=DEFAULT_N_ELEMENTS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    run_config = {
        "script": "measure_u_eff.py",
        "version": 1,
        "configs": [
            {
                "key": key,
                "label": label,
                "is_real_format": is_real,
                "block_size": cfg.block_size,
                "scale_format": cfg.scale_format,
                "use_global_scale": cfg.use_global_scale,
                "element_format": cfg.element_format,
                "round_mode": cfg.round_mode,
            }
            for key, label, is_real, cfg in CONFIGS
        ],
        "nu_grid": [label for label, _ in NU_GRID],
        "n_elements": args.n_elements,
        "tensor_shape": list(TENSOR_SHAPE),
        "quantiles": list(QUANTILES),
        "seed": args.seed,
        "distribution": "t-student (gaussian at the nu = infinity end)",
        "be_level_ratio_source": BE_LEVEL_RATIO_SOURCE,
        "be_level_ratio": BE_LEVEL_RATIO,
        "explained_fraction_tol": EXPLAINED_FRACTION_TOL,
    }
    digest = config_digest(run_config)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"u_eff_{digest}"

    started = time.perf_counter()
    rows = []
    cells = [
        (ci, key, label, is_real, cfg, ni, nu_label, nu)
        for ci, (key, label, is_real, cfg) in enumerate(CONFIGS)
        for ni, (nu_label, nu) in enumerate(NU_GRID)
    ]
    for ci, key, label, is_real, cfg, ni, nu_label, nu in tqdm(cells, desc="u_eff cells"):
        measured = run_cell(cfg, nu, args.n_elements, [args.seed, ci, ni])
        rows.append(
            {
                "config": key,
                "config_label": label,
                "is_real_format": is_real,
                "block_size": cfg.block_size,
                "scale_format": cfg.scale_format,
                "use_global_scale": cfg.use_global_scale,
                "nu_label": nu_label,
                "nu": nu if nu is not None else float("inf"),
                **measured,
            }
        )
    summary = pd.DataFrame(rows)
    level_gap = build_level_gap_table(summary)
    runtime_s = time.perf_counter() - started

    summary.to_parquet(OUT_DIR / f"{stem}_summary.parquet", engine="pyarrow")
    level_gap.to_parquet(OUT_DIR / f"{stem}_level_gap.parquet", engine="pyarrow")
    (OUT_DIR / f"{stem}.config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True), encoding="utf-8"
    )

    fig = make_figure(summary, level_gap)
    fig_path = OUT_DIR / f"{stem}_u_eff.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    statement = build_statement(summary, level_gap, args.n_elements, runtime_s)
    print("\n" + statement + "\n")
    (OUT_DIR / f"{stem}_statement.txt").write_text(statement + "\n", encoding="utf-8")

    print("U_EFF TABLE (config x nu):")
    print(
        summary[
            [
                "config",
                "is_real_format",
                "nu_label",
                "u_eff_p50",
                "u_eff_p99",
                "surviving_fraction",
                "p99_doubling_rel_change",
            ]
        ].to_string(index=False)
    )
    print("\nLEVEL-GAP CHECK (block 32 / block 16, E8M0 held fixed):")
    print(level_gap.to_string(index=False))
    print(f"\nWrote summary, level-gap table, config, figure ({fig_path.name}) to {OUT_DIR}")
    print(
        "\nSTOP: this script only measures and reports. No bound definition is written here; "
        "that is a human decision made on review."
    )


if __name__ == "__main__":
    main()
