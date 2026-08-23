"""Entry point for `make figures`.

Regenerates the paper's three figures as vector PDF (plus PNG previews)
directly from the already-committed analysis tables under
results/analysis/{nu_star,n_scaling_fits,causal_decomposition}/. This
script does not resample, refit, or recompute any statistic -- every
number plotted here was already produced by scripts/localize_nu_star.py,
scripts/fit_n_scaling.py or scripts/causal_decomposition.py and is read
straight off their committed Parquet output. See SPEC.md Steps 4.1-4.3
for how those numbers were derived.

Deterministic: given the same input tables, running this script twice
produces visually identical PDFs. The PDF `CreationDate` metadata field
is explicitly suppressed (see `_save`) so the two runs' PDF bytes match
too, modulo matplotlib's own version string embedded in PDF/PNG metadata,
which changes only when the matplotlib version changes.

All three figures below deviate from the project's original figure
roadmap, because the confirmatory analysis (Step 4.2) found a different
result than anticipated: the bound `cota(n)` breaks at every tested nu,
so nu* is undefined everywhere and the figures instead report what Step
4.2 actually found (a uniformly-broken bound of varying margin) and what
Step 4.3's exploratory follow-up found in its place (scale_format, not
block_size, is the larger lever on backward error). See SPEC.md Step 4.2
("Headline finding: GRID MIS-SPECIFICATION") for the full story.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = REPO_ROOT / "results" / "analysis"
FIGURES_DIR = REPO_ROOT / "paper" / "figures"

# Okabe-Ito colorblind-safe palette (also distinguishable in grayscale via
# the linestyle/marker redundancy each figure pairs with it).
COLOR_E8M0 = "#0072B2"  # blue
COLOR_E8M0_LIGHT = "#56B4E9"  # sky blue
COLOR_E4M3 = "#D55E00"  # vermillion
COLOR_E4M3_LIGHT = "#E69F00"  # orange

NU_ORDER = ["1", "2", "3", "5", "8", "15", "30", "gaussian"]
NU_TICK_LABELS = ["1", "2", "3", "5", "8", "15", "30", "∞\n(Gaussian)"]

# PDF metadata to suppress for run-to-run byte-determinism (see module
# docstring). CreationDate is the only field matplotlib's PDF backend sets
# from wall-clock time by default.
_PDF_METADATA = {"CreationDate": None}


def _find_one(pattern: str) -> Path:
    """Locate exactly one committed analysis file under results/analysis/.

    Globs rather than hardcoding the sha256-derived filename prefixes, so
    this script keeps working if an analysis script is re-run and its
    output digest changes -- it always picks up whatever is currently
    committed, not a frozen filename.
    """
    matches = sorted(ANALYSIS_DIR.glob(pattern))
    if not matches:
        raise SystemExit(
            f"No files match results/analysis/{pattern} -- run the corresponding "
            "analysis script (localize_nu_star.py / fit_n_scaling.py / "
            "causal_decomposition.py) first."
        )
    if len(matches) > 1:
        raise SystemExit(
            f"Ambiguous: multiple files match results/analysis/{pattern}: {matches}. "
            "Pass the specific path explicitly via the corresponding --*-path flag."
        )
    return matches[0]


def _nu_to_x(nu: str) -> float:
    """Map nu onto -1/sqrt(nu), so nu=1 (heaviest tail) sits at -1, nu
    grows toward 0 monotonically, and Gaussian (the nu -> infinity limit,
    not a number) sits exactly at 0 rather than needing an arbitrary
    finite stand-in. This still spaces the heavy-tail region (nu=1,2,3),
    where Fig 1's ratio varies most, out more widely than a linear nu axis
    would -- but a plain -1/nu is too aggressive at this grid's spacing
    (1,2,3,5,8,15,30,gaussian): it crushes nu=8,15,30 into the final 12%
    of the axis, close enough that their tick labels overlap. -1/sqrt(nu)
    keeps the heavy-tail emphasis while leaving all 8 ticks legibly
    spaced, verified by rendering at this figure's actual print width.
    """
    return 0.0 if nu == "gaussian" else -1.0 / float(nu) ** 0.5


def _set_print_rcparams() -> None:
    """Font sizes chosen and verified by rendering at the figsize each
    make_figure_* call below actually uses (~3.3in single-column, ~6.8in
    for Figure 3's two-panel width) and reading the output back -- not
    assumed from matplotlib defaults, which are tuned for a much larger
    canvas and become illegible at print width.
    """
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.3,
            "figure.titlesize": 8.5,
            "pdf.fonttype": 42,  # embed real (searchable) text, not paths
            "ps.fonttype": 42,
            "axes.grid": False,
        }
    )


def _save(fig: plt.Figure, out_dir: Path, stem: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{stem}.pdf"
    png_path = out_dir / f"{stem}.png"
    fig.savefig(pdf_path, metadata=_PDF_METADATA)
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    print(f"Wrote {pdf_path}")
    print(f"Wrote {png_path}")
    return pdf_path


def make_figure_1(nu_star_detail_path: Path, nu_star_config_path: Path, out_dir: Path) -> Path:
    """Figure 1 -- the paper's central figure.

    5-second read this figure must support: "No curve ever drops to or
    below y=1 -- the bound is broken everywhere. But the MARGIN above 1
    varies by over an order of magnitude across nu, and scale_format's
    curves separate visibly from block_size's at every nu, showing scale
    format drives the gap more than block size."
    """
    config = json.loads(nu_star_config_path.read_text())
    n_primary = config["n_primary"]
    round_mode = config["canonical_round_mode"]
    rht = config["canonical_rht"]

    detail = pd.read_parquet(nu_star_detail_path)
    canon = detail[
        (detail["n"] == n_primary) & (detail["round_mode"] == round_mode) & (detail["rht"] == rht)
    ]

    curves = [
        (16, "e8m0", COLOR_E8M0, "-", "o"),
        (32, "e8m0", COLOR_E8M0_LIGHT, "--", "s"),
        (16, "e4m3", COLOR_E4M3, "-", "^"),
        (32, "e4m3", COLOR_E4M3_LIGHT, "--", "D"),
    ]

    _set_print_rcparams()
    fig, ax = plt.subplots(figsize=(3.3, 2.75))

    x = np.array([_nu_to_x(nu) for nu in NU_ORDER])  # already ascending: nu=1 -> -1, gaussian -> 0
    for block_size, scale_format, color, ls, marker in curves:
        cell = (
            canon[(canon["block_size"] == block_size) & (canon["scale_format"] == scale_format)]
            .set_index("nu")
            .reindex(NU_ORDER)
        )
        y = cell["ratio_median"].to_numpy()
        lo = cell["ratio_ci_low"].to_numpy()
        hi = cell["ratio_ci_high"].to_numpy()
        label = f"block={block_size}, {scale_format}"
        ax.plot(
            x, y, color=color, linestyle=ls, marker=marker,
            markersize=3.2, linewidth=1.3, label=label,
        )
        ax.fill_between(x, lo, hi, color=color, alpha=0.22, linewidth=0)

    ax.axhline(
        1.0,
        color="black",
        linestyle=":",
        linewidth=1.6,
        zorder=5,
        label="bound threshold (ratio = 1)",
    )

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(NU_TICK_LABELS)
    ax.set_xlabel("ν (heavier tail ←  ·  → lighter tail; axis: -1/√ν)")
    ax.set_ylabel(f"median(BE) / cota(n) at n={n_primary} (log scale)")
    ax.set_title("Backward error vs. theoretical bound\n(canonical sub-config)")
    ax.legend(loc="upper right", framealpha=0.9, borderpad=0.4, labelspacing=0.3)
    fig.tight_layout()

    return _save(fig, out_dir, "figure1_nu_star")


def make_figure_2(n_scaling_detail_path: Path, nu_star_config_path: Path, out_dir: Path) -> Path:
    """Figure 2 -- the n-scaling law.

    Representative subset (2-4 series, not all 128 configurations): the
    two real, named hardware presets -- MXFP4 (block=32, e8m0) and NVFP4
    (block=16, e4m3) -- each shown at nu=1 (heaviest tail, where Step 4.1
    found the largest deviation from the -0.5 exponent, gap up to 0.49)
    and nu=gaussian (lightest tail, where Step 4.1 found the exponent
    matches -0.5 to within 0.02). Four series therefore illustrate both
    axes of Step 4.1's finding -- format and tail weight -- without the
    128-configuration clutter a literal reading of the figure would need.
    """
    config = json.loads(nu_star_config_path.read_text())
    round_mode = config["canonical_round_mode"]
    rht = config["canonical_rht"]

    detail = pd.read_parquet(n_scaling_detail_path)
    canon = detail[(detail["round_mode"] == round_mode) & (detail["rht"] == rht)]

    series = [
        (32, "e8m0", "1", COLOR_E8M0, "-", "o", "MXFP4 (32, e8m0), ν=1"),
        (32, "e8m0", "gaussian", COLOR_E8M0_LIGHT, "--", "o", "MXFP4, ν=Gaussian"),
        (16, "e4m3", "1", COLOR_E4M3, "-", "^", "NVFP4 (16, e4m3), ν=1"),
        (16, "e4m3", "gaussian", COLOR_E4M3_LIGHT, "--", "^", "NVFP4, ν=Gaussian"),
    ]

    _set_print_rcparams()
    fig, ax = plt.subplots(figsize=(3.3, 2.95))

    anchor_x0, anchor_y0 = [], []
    for block_size, scale_format, nu, color, ls, marker, label in series:
        cell = canon[
            (canon["block_size"] == block_size)
            & (canon["scale_format"] == scale_format)
            & (canon["nu"] == nu)
        ].sort_values("n")
        ax.plot(
            cell["n"],
            cell["median_be"],
            color=color,
            linestyle=ls,
            marker=marker,
            markersize=3.6,
            linewidth=1.3,
            label=label,
        )
        anchor_x0.append(cell["n"].iloc[0])
        anchor_y0.append(cell["median_be"].iloc[0])

    # Reference lines are not fits to any one series -- they are anchored at
    # the geometric mean, over the four displayed series, of the value at
    # the smallest measured n, so both lines start from one shared,
    # data-grounded point and their divergence from there is what carries
    # the pedagogical contrast (derived bound decays, classical bound would
    # have grown).
    n0 = float(np.mean(anchor_x0))
    y0 = float(np.exp(np.mean(np.log(anchor_y0))))
    n_range = np.array(sorted(canon["n"].unique()), dtype=float)
    ax.plot(
        n_range,
        y0 * (n_range / n0) ** -0.5,
        color="black",
        linestyle=":",
        linewidth=1.6,
        label="slope -0.5 (u_eff/√n, this project's derived bound)",
    )
    ax.plot(
        n_range,
        y0 * (n_range / n0) ** 1.0,
        color="dimgray",
        linestyle="-.",
        linewidth=1.6,
        label=(
            "slope +1 (classical γ_n bound -- does not apply\n"
            "under exact accumulation, shown for contrast)"
        ),
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("contraction dimension n")
    ax.set_ylabel("median(BE) (log scale)")
    ax.set_title("Backward-error n-scaling, MXFP4 vs. NVFP4")
    ax.legend(loc="upper left", framealpha=0.9, borderpad=0.4, labelspacing=0.3)
    fig.tight_layout()

    return _save(fig, out_dir, "figure2_n_scaling")


def make_figure_3(
    n_scaling_detail_path: Path,
    causal_table_path: Path,
    nu_star_config_path: Path,
    out_dir: Path,
) -> Path:
    """Figure 3 -- the causal heatmap.

    Cell values (median(BE) at n_primary, canonical sub-config) are read
    from n_scaling_fits' own per-cell detail table, not from
    causal_decomposition's table: Step 4.3's committed table stores only
    the pooled effect_block/effect_scale/interaction differences (SPEC.md,
    Step 4.3 "Method"), not the per-(block_size, scale_format) medians
    those differences are computed from, so the per-cell values this
    heatmap needs live in n_scaling_fits_*_detail.parquet instead -- a
    direct read of an already-committed column either way, not a
    recomputation. causal_decomposition's own table is still read here,
    for its dominant_factor column, to caption the panel-vs-row contrast
    with Step 4.3's own already-computed verdict rather than a new one.

    5-second read: "the two panels look visibly different from each
    other (scale format effect), while within each panel the two rows
    (block sizes) look more similar to each other than the panels do to
    each other -- visually showing scale format dominates block size."
    """
    config = json.loads(nu_star_config_path.read_text())
    n_primary = config["n_primary"]
    round_mode = config["canonical_round_mode"]
    rht = config["canonical_rht"]

    detail = pd.read_parquet(n_scaling_detail_path)
    canon = detail[
        (detail["n"] == n_primary) & (detail["round_mode"] == round_mode) & (detail["rht"] == rht)
    ]

    causal = pd.read_parquet(causal_table_path)
    causal_canon = causal[
        (causal["n"] == n_primary) & (causal["round_mode"] == round_mode) & (causal["rht"] == rht)
    ]
    n_scale_dominant = int((causal_canon["dominant_factor"] == "scale").sum())
    n_total = len(causal_canon)

    block_sizes = [16, 32]
    scale_formats = ["e8m0", "e4m3"]

    value_grids: dict[str, np.ndarray] = {}
    for scale_format in scale_formats:
        mat = np.full((len(block_sizes), len(NU_ORDER)), np.nan)
        for i, block_size in enumerate(block_sizes):
            for j, nu in enumerate(NU_ORDER):
                row = canon[
                    (canon["block_size"] == block_size)
                    & (canon["scale_format"] == scale_format)
                    & (canon["nu"] == nu)
                ]
                mat[i, j] = row["median_be"].iloc[0]
        value_grids[scale_format] = mat

    log_grids = {sf: np.log10(mat) for sf, mat in value_grids.items()}
    vmin = min(g.min() for g in log_grids.values())
    vmax = max(g.max() for g in log_grids.values())

    _set_print_rcparams()
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.0), sharey=True)

    # Cells are 2 rows x 8 columns -- much taller than wide -- so cell
    # annotations are rotated 90 deg to use the generous vertical room
    # instead of the ~0.4in-per-column horizontal room 8 side-by-side
    # numbers would need (which overlapped when tested unrotated).
    # pcolormesh (vector rectangles), not imshow (a raster image XObject),
    # so the "vector PDF" requirement holds for the heatmap cells too --
    # cheap here since the grid is only 2x8.
    im = None
    for ax, scale_format in zip(axes, scale_formats):
        im = ax.pcolormesh(log_grids[scale_format], cmap="viridis", vmin=vmin, vmax=vmax)
        ax.invert_yaxis()
        ax.set_xticks(np.arange(len(NU_ORDER)) + 0.5)
        ax.set_xticklabels(NU_TICK_LABELS, fontsize=6)
        ax.set_yticks(np.arange(len(block_sizes)) + 0.5)
        ax.set_yticklabels([str(b) for b in block_sizes])
        ax.set_xlabel("ν")
        ax.set_title(f"scale_format = {scale_format}")
        for i in range(len(block_sizes)):
            for j in range(len(NU_ORDER)):
                frac = (log_grids[scale_format][i, j] - vmin) / (vmax - vmin)
                textcolor = "white" if frac < 0.6 else "black"
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    f"{value_grids[scale_format][i, j]:.2g}",
                    ha="center",
                    va="center",
                    rotation=90,
                    fontsize=6,
                    color=textcolor,
                )
    axes[0].set_ylabel("block_size")

    fig.subplots_adjust(left=0.09, right=0.87, top=0.80, bottom=0.22, wspace=0.08)
    cbar_ax = fig.add_axes((0.89, 0.22, 0.02, 0.58))
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("log10(median BE)", fontsize=6.5)
    cbar.ax.tick_params(labelsize=6)

    fig.suptitle(
        f"median(BE) by ν and block_size, n={n_primary}, canonical sub-config\n"
        f"causal decomposition (Step 4.3, EXPLORATORY): scale format dominates block size "
        f"in {n_scale_dominant}/{n_total} canonical cells at this n",
        fontsize=6.8,
        y=0.985,
    )

    return _save(fig, out_dir, "figure3_causal_heatmap")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--nu-star-detail", type=Path, default=None)
    parser.add_argument("--nu-star-config", type=Path, default=None)
    parser.add_argument("--n-scaling-detail", type=Path, default=None)
    parser.add_argument("--causal-table", type=Path, default=None)
    args = parser.parse_args()

    nu_star_detail = args.nu_star_detail or _find_one("nu_star/*_detail.parquet")
    nu_star_config = args.nu_star_config or _find_one("nu_star/*.config.json")
    n_scaling_detail = args.n_scaling_detail or _find_one("n_scaling_fits/*_detail.parquet")
    causal_table = args.causal_table or _find_one("causal_decomposition/*_table.parquet")

    make_figure_1(nu_star_detail, nu_star_config, args.out_dir)
    make_figure_2(n_scaling_detail, nu_star_config, args.out_dir)
    make_figure_3(n_scaling_detail, causal_table, nu_star_config, args.out_dir)


if __name__ == "__main__":
    main()
