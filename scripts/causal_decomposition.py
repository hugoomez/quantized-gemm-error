"""Step 4.3 (EXPLORATORY, post-4.2 pivot): decompose observed median(BE) by
block_size and scale_format, directly -- no cota(n), no c, no ratio to 1.0.

PREREGISTRATION.md sec 1/sec 4's H1 and P1-P3 are claims about nu* (a
crossing in the bound-broken/bound-holds pattern). Step 4.2
(`results/analysis/nu_star/`) found the bound broken at every tested nu, in
every one of the 16 families, at every tested n -- there is no nu* anywhere
on this grid, so P1/P2/P3 are structurally undefined, not merely "no effect
found." PREREGISTRATION.md sec 8.4 (a dated, disclosed post-data amendment)
records the pivot this script implements: apply the originally-planned
causal decomposition (block_size x scale_format, from the Step 1.5 2x2)
directly to measured `median(BE)`, in log space, instead of to nu*.

**This is EXPLORATORY, not a resolution of H1 as originally formulated.** It
answers a related but WEAKER question -- which factor moves backward error
more, and by how much, at a given (nu, n) -- using no theoretical bound at
all. It cannot confirm, reject, or null-reframe H1 in PREREGISTRATION.md
sec 1/2/5's sense, since those are specifically about where a bound breaks.
Output is kept under `results/analysis/causal_decomposition/`, labeled
EXPLORATORY throughout, separate from Step 4.2's confirmatory output.

Effect definitions (log space; median of log(BE), NOT log of median(BE) --
stated once here, applied everywhere below)
-------------------------------------------------------------------------
For each (nu, n, round_mode, rht), pooling trials across the factor being
held out (never collapsing across nu or n themselves, per the task this
script implements -- Step 4.2's failure came partly from trusting a single
n):

    effect_block(nu, n) = median(log BE | block=16, pooled over scale_format)
                         - median(log BE | block=32, pooled over scale_format)
    effect_scale(nu, n) = median(log BE | scale=e4m3, pooled over block_size)
                         - median(log BE | scale=e8m0, pooled over block_size)
    interaction(nu, n)  = [median(log BE|16,e4m3) - median(log BE|32,e4m3)]
                         - [median(log BE|16,e8m0) - median(log BE|32,e8m0)]

`interaction` is symmetric: it equals the analogous scale-effect difference
between block sizes too (same algebraic expression reordered), so it is
computed once, not twice.

Each is a percentile bootstrap over TRIALS, resampling each of the four
underlying (block_size, scale_format) cells independently within a single
vectorized pass (so `interaction`'s replicates are drawn from the same
per-replicate resample of all four cells as `effect_block`/`effect_scale`,
not recombined from separately-bootstrapped intermediates) -- explicit
`numpy.random.Generator`, deterministic per (nu, n, round_mode, rht).

`n_resamples=2000` here, not PREREGISTRATION.md sec 3.2's confirmatory
B=10000: this step is explicitly exploratory (sec 8.4), not bound by the
confirmatory bootstrap spec, and disclosed rather than silently substituted
(same disclosure convention as Step 3.4's coverage simulation, which used
1000 for the same reason).

Usage
-----
    python scripts/causal_decomposition.py [--sweep-dir results/sweep_e945b87a2395]
        [--n-resamples 2000] [--seed 0]

Output
------
`results/analysis/causal_decomposition/causal_decomposition_{digest}_EXPLORATORY_table.parquet`
(160 rows: 8 nu x 5 n x 2 round_mode x 2 rht) and `..._table.csv`, the
headline figure (canonical sub-config, effect_block/effect_scale vs nu, n as
an overlay), `..._statement.txt`, `...config.json`.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SWEEP_DIR = REPO_ROOT / "results" / "sweep_e945b87a2395"
OUT_DIR = REPO_ROOT / "results" / "analysis" / "causal_decomposition"

NU_ORDER = ["1", "2", "3", "5", "8", "15", "30", "gaussian"]
NU_DISPLAY = ["1", "2", "3", "5", "8", "15", "30", "∞"]
ROUND_MODE_ORDER = ["rtne", "sr"]
RHT_ORDER = [False, True]
BLOCK_SIZE_ORDER = [16, 32]
SCALE_FORMAT_ORDER = ["e8m0", "e4m3"]

CANONICAL_ROUND_MODE = "rtne"
CANONICAL_RHT = False

DEFAULT_N_RESAMPLES = 2000  # exploratory -- see module docstring
DEFAULT_SEED = 0
CI = 0.95


def cell_seed(base_seed: int, nu: str, n: int, round_mode: str, rht: bool) -> list[int]:
    """Deterministic entropy for one (nu, n, round_mode, rht)'s Generator.

    Plain small integers only, matching this project's established
    convention (`scripts/fit_n_scaling.py`'s `config_seed`,
    `scripts/localize_nu_star.py`'s `nu_seed`).
    """
    return [
        base_seed,
        NU_ORDER.index(nu),
        int(n),
        ROUND_MODE_ORDER.index(round_mode),
        int(rht),
    ]


def bootstrap_decomposition(
    log_be_16_e8: np.ndarray,
    log_be_16_e4: np.ndarray,
    log_be_32_e8: np.ndarray,
    log_be_32_e4: np.ndarray,
    rng: np.random.Generator,
    n_resamples: int,
    ci: float = CI,
) -> dict:
    """Point estimate + bootstrap CI for effect_block, effect_scale, interaction.

    All four inputs are already log-transformed trial-level `be_median`
    arrays for one (nu, n, round_mode, rht)'s four (block_size,
    scale_format) cells. A single vectorized bootstrap pass resamples all
    four cells independently, `n_resamples` times, so the three derived
    statistics' replicates share the same underlying draws -- required for
    `interaction`'s replicates to be a real function of the same resample as
    `effect_block`/`effect_scale`, not an artifact of combining separately-
    bootstrapped pieces.
    """

    def stats_from(a16e8: np.ndarray, a16e4: np.ndarray, a32e8: np.ndarray, a32e4: np.ndarray):
        pooled_16 = np.concatenate([a16e8, a16e4], axis=-1)
        pooled_32 = np.concatenate([a32e8, a32e4], axis=-1)
        pooled_e8 = np.concatenate([a16e8, a32e8], axis=-1)
        pooled_e4 = np.concatenate([a16e4, a32e4], axis=-1)
        effect_block = np.median(pooled_16, axis=-1) - np.median(pooled_32, axis=-1)
        effect_scale = np.median(pooled_e4, axis=-1) - np.median(pooled_e8, axis=-1)
        block_effect_at_e8 = np.median(a16e8, axis=-1) - np.median(a32e8, axis=-1)
        block_effect_at_e4 = np.median(a16e4, axis=-1) - np.median(a32e4, axis=-1)
        interaction = block_effect_at_e4 - block_effect_at_e8
        return effect_block, effect_scale, interaction

    point_block, point_scale, point_interaction = stats_from(
        log_be_16_e8, log_be_16_e4, log_be_32_e8, log_be_32_e4
    )

    def resample(a: np.ndarray) -> np.ndarray:
        idx = rng.integers(0, a.size, size=(n_resamples, a.size), dtype=np.int32)
        return a[idx]

    r16e8 = resample(log_be_16_e8)
    r16e4 = resample(log_be_16_e4)
    r32e8 = resample(log_be_32_e8)
    r32e4 = resample(log_be_32_e4)

    block_reps, scale_reps, interaction_reps = stats_from(r16e8, r16e4, r32e8, r32e4)

    alpha = (1.0 - ci) / 2.0

    def pct(reps: np.ndarray) -> tuple[float, float]:
        lo, hi = np.percentile(reps, [100.0 * alpha, 100.0 * (1.0 - alpha)])
        return float(lo), float(hi)

    eb_lo, eb_hi = pct(block_reps)
    es_lo, es_hi = pct(scale_reps)
    it_lo, it_hi = pct(interaction_reps)

    return {
        "effect_block": float(point_block),
        "effect_block_ci_low": eb_lo,
        "effect_block_ci_high": eb_hi,
        "effect_scale": float(point_scale),
        "effect_scale_ci_low": es_lo,
        "effect_scale_ci_high": es_hi,
        "interaction": float(point_interaction),
        "interaction_ci_low": it_lo,
        "interaction_ci_high": it_hi,
    }


def compute_table(df_all: pd.DataFrame, seed: int, n_resamples: int) -> pd.DataFrame:
    n_values = sorted(int(n) for n in df_all["n"].unique())
    rows: list[dict] = []
    combos = list(itertools.product(NU_ORDER, n_values, ROUND_MODE_ORDER, RHT_ORDER))
    for nu, n_value, round_mode, rht in combos:
        cell = df_all[
            (df_all["nu"] == nu)
            & (df_all["n"] == n_value)
            & (df_all["round_mode"] == round_mode)
            & (df_all["rht"] == rht)
        ]
        subcells = {}
        for block_size, scale_format in itertools.product(BLOCK_SIZE_ORDER, SCALE_FORMAT_ORDER):
            sub = cell[(cell["block_size"] == block_size) & (cell["scale_format"] == scale_format)]
            if sub.empty:
                raise ValueError(
                    f"missing cell: nu={nu} n={n_value} round_mode={round_mode} rht={rht} "
                    f"block_size={block_size} scale_format={scale_format}"
                )
            subcells[(block_size, scale_format)] = np.log(
                sub["be_median"].to_numpy(dtype=np.float64)
            )

        rng = np.random.default_rng(cell_seed(seed, nu, n_value, round_mode, rht))
        result = bootstrap_decomposition(
            subcells[(16, "e8m0")],
            subcells[(16, "e4m3")],
            subcells[(32, "e8m0")],
            subcells[(32, "e4m3")],
            rng,
            n_resamples,
        )
        block_sig = result["effect_block_ci_low"] > 0 or result["effect_block_ci_high"] < 0
        scale_sig = result["effect_scale_ci_low"] > 0 or result["effect_scale_ci_high"] < 0
        interaction_sig = (
            result["interaction_ci_low"] > 0 or result["interaction_ci_high"] < 0
        )
        dominant = (
            "block"
            if abs(result["effect_block"]) > abs(result["effect_scale"])
            else "scale"
            if abs(result["effect_scale"]) > abs(result["effect_block"])
            else "neither"
        )
        rows.append(
            {
                "nu": nu,
                "n": n_value,
                "round_mode": round_mode,
                "rht": rht,
                "n_trials_per_cell": int(subcells[(16, "e8m0")].size),
                **result,
                "effect_block_significant": bool(block_sig),
                "effect_scale_significant": bool(scale_sig),
                "interaction_significant": bool(interaction_sig),
                "dominant_factor": dominant,
            }
        )
    return pd.DataFrame(rows)


def make_figure(table: pd.DataFrame) -> plt.Figure:
    """effect_block and effect_scale vs nu, canonical sub-config, n as an overlay."""
    sel = table[(table["round_mode"] == CANONICAL_ROUND_MODE) & (~table["rht"])]
    n_values = sorted(sel["n"].unique())
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(n_values)))
    x = np.arange(len(NU_ORDER))

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.2))
    for ax, metric, title in zip(
        axes,
        ["effect_block", "effect_scale"],
        [
            "effect_block = median(log BE | 16) − median(log BE | 32)",
            "effect_scale = median(log BE | e4m3) − median(log BE | e8m0)",
        ],
        strict=True,
    ):
        for n_value, color in zip(n_values, colors, strict=True):
            row = sel[sel["n"] == n_value].set_index("nu").loc[NU_ORDER]
            point = row[metric].to_numpy()
            lo = row[f"{metric}_ci_low"].to_numpy()
            hi = row[f"{metric}_ci_high"].to_numpy()
            ax.errorbar(
                x, point, yerr=[point - lo, hi - point],
                marker="o", ms=5, lw=1.6, capsize=3, color=color, label=f"n={n_value}",
            )
        ax.axhline(0.0, color="gray", ls="--", lw=1.2, zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels(NU_DISPLAY)
        ax.set_xlabel("ν (heavier tail →)")
        ax.set_ylabel(metric)
        ax.set_title(title, fontsize=10.5)
        ax.grid(True, alpha=0.15)
        ax.legend(fontsize=8.5, loc="best", ncol=2)

    fig.suptitle(
        "Step 4.3 (EXPLORATORY): block_size / scale_format effects on log(BE)\n"
        "round_mode=rtne, rht=False (canonical) -- error bars are 95% bootstrap CIs, "
        "dashed line = no effect",
        fontsize=12.5,
    )
    fig.subplots_adjust(left=0.06, right=0.98, top=0.82, bottom=0.11, wspace=0.22)
    return fig


def build_statement(table: pd.DataFrame, n_resamples: int, runtime_s: float) -> str:
    lines: list[str] = []
    lines.append(
        "Step 4.3 (EXPLORATORY, post-4.2 pivot): block_size/scale_format decomposition of "
        f"observed median(BE), log space. {len(table)} (nu, n, round_mode, rht) cells "
        f"(8 nu x {table['n'].nunique()} n x 2 round_mode x 2 rht), {n_resamples} bootstrap "
        f"resamples per cell. Runtime {runtime_s / 60.0:.1f} min. NOT a resolution of H1 as "
        "originally formulated (PREREGISTRATION.md sec 1, sec 8.4) -- no cota(n), no c, no "
        "ratio to 1.0 anywhere in this computation."
    )
    lines.append("")
    canonical = table[(table["round_mode"] == CANONICAL_ROUND_MODE) & (~table["rht"])]
    lines.append("=" * 72)
    lines.append("CANONICAL (round_mode=rtne, rht=False): dominant factor per (nu, n)")
    lines.append("=" * 72)
    for n_value in sorted(canonical["n"].unique()):
        sub = canonical[canonical["n"] == n_value].set_index("nu").loc[NU_ORDER]
        lines.append(f"  n={n_value}:")
        for nu, row in sub.iterrows():
            bsig = "*" if row["effect_block_significant"] else " "
            ssig = "*" if row["effect_scale_significant"] else " "
            isig = "*" if row["interaction_significant"] else " "
            lines.append(
                f"    nu={nu:>8}: effect_block={row['effect_block']:+.4f}{bsig} "
                f"[{row['effect_block_ci_low']:+.4f},{row['effect_block_ci_high']:+.4f}]  "
                f"effect_scale={row['effect_scale']:+.4f}{ssig} "
                f"[{row['effect_scale_ci_low']:+.4f},{row['effect_scale_ci_high']:+.4f}]  "
                f"interaction={row['interaction']:+.4f}{isig}  dominant={row['dominant_factor']}"
            )
    lines.append("  (* = CI excludes 0, i.e. statistically distinguishable from no effect)")
    lines.append("")

    dominant_counts = canonical["dominant_factor"].value_counts()
    lines.append(
        f"Dominant factor across all {len(canonical)} canonical (nu, n) cells: "
        f"{dominant_counts.to_dict()}"
    )
    block_dominant_nus = sorted(
        canonical[canonical["dominant_factor"] == "block"]["nu"].unique(), key=NU_ORDER.index
    )
    scale_dominant_nus = sorted(
        canonical[canonical["dominant_factor"] == "scale"]["nu"].unique(), key=NU_ORDER.index
    )
    lines.append(f"  nu values where block_size dominates at >=1 n: {block_dominant_nus}")
    lines.append(f"  nu values where scale_format dominates at >=1 n: {scale_dominant_nus}")
    lines.append("")

    # Does the ranking (block vs scale dominant) hold across all 5 n at fixed nu?
    lines.append("RANKING CONSISTENCY ACROSS n (fixed nu, canonical sub-config):")
    flips = []
    for nu in NU_ORDER:
        sub = canonical[canonical["nu"] == nu]
        doms = set(sub["dominant_factor"].unique())
        consistent = len(doms) == 1
        if not consistent:
            flips.append(nu)
        n_list = ", ".join(sorted(sub["n"].astype(str)))
        tag = "[CONSISTENT]" if consistent else "[DEPENDS ON n]"
        lines.append(
            f"  nu={nu:>8}: dominant factor across n in {{{n_list}}} = {sorted(doms)}  {tag}"
        )
    lines.append("")
    if flips:
        lines.append(
            f"The ranking DEPENDS ON WHICH n YOU'D HAVE PICKED at nu in {flips} -- reported "
            "honestly rather than papered over, since this directly bears on the same "
            "n-choice fragility that produced Step 4.2's grid mis-specification."
        )
    else:
        lines.append(
            "The ranking is CONSISTENT across every tested n at every nu -- not sensitive to "
            "which n would have been picked as primary."
        )
    lines.append("")

    # nu-dependence of magnitude
    at_n4096 = canonical[canonical["n"] == canonical["n"].max()].set_index("nu").loc[NU_ORDER]
    block_abs = at_n4096["effect_block"].abs()
    scale_abs = at_n4096["effect_scale"].abs()
    lines.append(
        f"MAGNITUDE vs nu (n={canonical['n'].max()}, canonical): "
        f"|effect_block| ranges {block_abs.min():.4f} (nu={block_abs.idxmin()}) to "
        f"{block_abs.max():.4f} (nu={block_abs.idxmax()}); "
        f"|effect_scale| ranges {scale_abs.min():.4f} (nu={scale_abs.idxmin()}) to "
        f"{scale_abs.max():.4f} (nu={scale_abs.idxmax()})."
    )
    block_grows_with_tail_weight = (
        at_n4096["effect_block"].abs().loc["1"] > at_n4096["effect_block"].abs().loc["gaussian"]
    )
    lines.append(
        "block_size's effect on log(BE) "
        + ("GROWS" if block_grows_with_tail_weight else "does NOT grow")
        + " as the tail gets heavier (nu=1 vs nu=gaussian, n="
        f"{canonical['n'].max()}) -- cross-checked against the u_eff level-gap finding "
        "(SPEC.md, 'u_eff measurement', Result 2: median(BE) level ratio 32/16 at matched "
        "n=1024 grows from 1.0344 at nu=30 to 1.3206 at nu=1), "
        + ("CONSISTENT with it." if block_grows_with_tail_weight else "NOT consistent with it.")
    )
    lines.append("")

    lines.append("=" * 72)
    lines.append("ROBUSTNESS across the other 3 (round_mode, rht) combos")
    lines.append("=" * 72)
    combo_dominants = {}
    for round_mode, rht in itertools.product(ROUND_MODE_ORDER, RHT_ORDER):
        sub = table[(table["round_mode"] == round_mode) & (table["rht"] == rht)]
        counts = sub["dominant_factor"].value_counts().to_dict()
        combo_dominants[(round_mode, rht)] = counts
        overall = max(counts, key=counts.get)
        lines.append(
            f"  round_mode={round_mode:<4} rht={rht!s:<5}: dominant-factor counts = {counts}  "
            f"(overall majority: {overall})"
        )
    majorities = {k: max(v, key=v.get) for k, v in combo_dominants.items()}
    stable = len(set(majorities.values())) == 1
    lines.append("")
    lines.append(
        f"Overall majority dominant factor per combo: {majorities}. "
        f"Stable across all 4 round_mode x rht combinations: {stable}."
    )
    lines.append("")

    lines.append(
        "This relates to, but does not replace, Step 4.2's grid mis-specification finding: "
        "Step 4.2 showed the c=1 theoretical bound is uncalibrated everywhere on this grid, "
        "so no nu* exists to compare between block sizes or scale formats; this step answers "
        "a narrower, descriptive question -- which factor moves raw backward error more, and "
        "under what conditions -- using no bound at all. Neither step's conclusion transfers "
        "to the other's question."
    )
    return "\n".join(lines)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sweep-dir", type=Path, default=DEFAULT_SWEEP_DIR)
    parser.add_argument("--n-resamples", type=int, default=DEFAULT_N_RESAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    combined_path = args.sweep_dir.parent / f"{args.sweep_dir.name}.parquet"
    print(f"Loading {combined_path} ...")
    df_all = pd.read_parquet(combined_path, engine="pyarrow")
    print(f"Loaded {len(df_all):,} trial rows.")

    started = time.perf_counter()
    table = compute_table(df_all, args.seed, args.n_resamples)
    runtime_s = time.perf_counter() - started
    print(f"Computed {len(table)} (nu, n, round_mode, rht) cells in {runtime_s:.1f}s.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config = {
        "script": "causal_decomposition.py",
        "version": 1,
        "sweep_dir": str(args.sweep_dir),
        "n_resamples": args.n_resamples,
        "seed": args.seed,
        "ci": CI,
        "canonical_round_mode": CANONICAL_ROUND_MODE,
        "canonical_rht": CANONICAL_RHT,
        "status": "EXPLORATORY -- see PREREGISTRATION.md sec 8.4",
    }
    canonical_json = json.dumps(config, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()[:12]
    stem = f"causal_decomposition_{digest}_EXPLORATORY"

    table.to_parquet(OUT_DIR / f"{stem}_table.parquet", engine="pyarrow")
    table.to_csv(OUT_DIR / f"{stem}_table.csv", index=False)
    (OUT_DIR / f"{stem}.config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
    )

    statement = build_statement(table, args.n_resamples, runtime_s)
    print("\n" + statement + "\n")
    (OUT_DIR / f"{stem}_statement.txt").write_text(statement + "\n", encoding="utf-8")

    fig = make_figure(table)
    fig_path = OUT_DIR / f"{stem}_effects.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    print(f"Wrote table, config, statement and figure ({fig_path.name}) to {OUT_DIR} "
          f"(stem={stem})")


if __name__ == "__main__":
    main()
