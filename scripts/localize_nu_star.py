"""Step 4.2 (this project's PRIMARY RESULT): locate nu* -- the critical
tail-weight where the theoretical bound cota(n) = u_eff/sqrt(n) (c fixed at
1, SPEC.md "Theoretical bound definition (\U0001f9e01) -- RESOLVED") breaks --
and use it to answer H1 (PREREGISTRATION.md sec 1).

This is REAL CONFIRMATORY ANALYSIS on the production sweep
(`results/sweep_e945b87a2395`), evaluated at PREREGISTRATION.md sec 4.0's
`n_primary`, whose value (4096) was declared in sec 8.3 -- a dated amendment,
made after the sweep already existed, disclosed as post-data rather than
claimed clean (see that amendment for the full disclosure). Output goes
under `results/analysis/nu_star/`, separate from every diagnostic script's
`results/diagnostics/` location.

The break criterion (PREREGISTRATION.md sec 3, applied exactly as written)
------------------------------------------------------------------------
The bound is broken at (family, nu) if the lower bound of the 95% bootstrap
CI of the median of the ratio `empirical_BE / cota(n_primary)` exceeds 1.0.
Median, not mean. CI lower bound, not point estimate (conservative). The
bootstrap resamples TRIALS (the independent seed), never elements within a
trial (sec 3.2) -- via `qgemm.stats.bootstrap_ci`, B=10000 (sec 3.2's
default), an explicit `numpy.random.Generator` seeded deterministically per
(family, nu) -- never global RNG state.

nu* localization (PREREGISTRATION.md sec 1.1, applied exactly as written)
------------------------------------------------------------------------
For each of the 16 "families" (block_size x scale_format x round_mode x rht
-- every swept factor except nu, which is the axis being scanned), let
B(family) be the set of tested nu at which the bound breaks. The fragility
rank r(family) is 0 if B is empty, or k where nu_k = max(B) otherwise.
nu*(family) is read off as the open interval (nu_r, nu_{r+1}), with the
conventions nu_0 := 0 and nu_9 := infinity (nu=8 in the grid means "breaks at
every tested nu"; the Gaussian limit, "nu=infinity" in this project's
notation, sits at grid position 8 of 8). r=0 ("censored below": the bound
never breaks in range) and r=8 ("censored above": the bound breaks even at
the lightest-tailed nu tested) are NOT dropped or interpolated past -- they
are reported as informative null results for that family, exactly as
PREREGISTRATION.md sec 1.1 and this step's own task specify.

Monotonicity is checked, not assumed (sec 1.1): B(family) must be a down-set
in nu-order (breaks at nu_k implies breaks at every heavier-tailed nu_j <
nu_k). A family that violates this is flagged non-monotone; r(family) is
still computed by the rule above (the definition never becomes ambiguous),
but that family's contribution to the P1/P2 H1 comparisons is reported as
inconclusive-exploratory rather than confirmatory, per sec 1.1.

H1 (PREREGISTRATION.md sec 1, sec 2.1, sec 4.1)
------------------------------------------------
H1: nu*(block=32) > nu*(block=16) at both scale formats (block-32 is more
fragile). P1 (nu* vs block size, scale format held constant) is the SOLE
confirmatory test of H1 (sec 4.1) -- evaluated on the canonical
sub-configuration (round_mode="rtne", rht=False, matching how MXFP4/NVFP4
are actually used) via the conjunctive rule in sec 2.1. P2 (nu* vs scale
format, block size held constant) is a design-validity comparison, not a
test of H1 (sec 4.1) -- reported regardless of what it shows. Neither round
mode nor RHT is part of the primary 2x2 design (sec 4: "Everything else --
... rounding mode ... RHT on/off ... is exploratory"); the other three
(round_mode, rht) combinations are reported as a robustness check on the
canonical P1/P2 verdict, never as additional confirmatory tests.

Usage
-----
    python scripts/localize_nu_star.py [--sweep-dir results/sweep_e945b87a2395]
        [--n-primary 4096] [--n-resamples 10000] [--seed 0]

Output
------
`results/analysis/nu_star/nu_star_{digest}_families.parquet` (16 rows, one
per family) and `..._families.csv`, `..._detail.parquet` (128 rows, one per
family x nu, with the ratio point/CI/broken status), `..._h1.txt` (the H1
finding, canonical + robustness), `..._nu_star.png` (the headline figure),
`...config.json`, and `..._statement.txt`.
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

from qgemm.stats import bootstrap_ci  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SWEEP_DIR = REPO_ROOT / "results" / "sweep_e945b87a2395"
OUT_DIR = REPO_ROOT / "results" / "analysis" / "nu_star"

# nu1 < ... < nu8, per PREREGISTRATION.md's notation ("nu = infinity is the
# Gaussian case"): "gaussian" occupies grid position 8, the lightest tail.
NU_ORDER = ["1", "2", "3", "5", "8", "15", "30", "gaussian"]
NU_DISPLAY = ["1", "2", "3", "5", "8", "15", "30", "∞"]  # for figure labels
# Interval boundary labels: index 0 is PREREGISTRATION.md sec 1.1's nu_0:=0
# sentinel, indices 1-8 are the tested grid, index 9 is nu_9:=infinity.
GRID_BOUNDARY_DISPLAY = ["0", *NU_DISPLAY[:-1], "∞", "∞"]

BLOCK_SIZE_ORDER = [16, 32]
SCALE_FORMAT_ORDER = ["e8m0", "e4m3"]
ROUND_MODE_ORDER = ["rtne", "sr"]
RHT_ORDER = [False, True]

FAMILY_FACTORS = ["block_size", "scale_format", "round_mode", "rht"]

DEFAULT_N_PRIMARY = 4096
DEFAULT_N_RESAMPLES = 10_000  # PREREGISTRATION.md sec 3.2's B=10000
DEFAULT_SEED = 0
CI = 0.95

CANONICAL_ROUND_MODE = "rtne"
CANONICAL_RHT = False


def nu_seed(
    base_seed: int, block_size: int, scale_format: str, round_mode: str, rht: bool, nu: str
) -> list[int]:
    """Deterministic entropy for one (family, nu)'s `numpy.random.Generator`.

    Plain small integers only, matching this project's established
    convention (`scripts/fit_n_scaling.py`'s `config_seed`) -- never Python's
    randomized `hash()` on a string.
    """
    return [
        base_seed,
        BLOCK_SIZE_ORDER.index(block_size),
        SCALE_FORMAT_ORDER.index(scale_format),
        ROUND_MODE_ORDER.index(round_mode),
        int(rht),
        NU_ORDER.index(nu),
    ]


def is_down_set(broken: list[bool]) -> bool:
    """True iff `broken` (indexed nu1..nu8) is a down-set: breaks at nu_k
    implies breaks at every nu_j < nu_k (PREREGISTRATION.md sec 1.1)."""
    for i in range(len(broken)):
        for j in range(i + 1, len(broken)):
            if not broken[i] and broken[j]:
                return False
    return True


def fragility_rank(broken: list[bool]) -> int:
    """r(family) per PREREGISTRATION.md sec 1.1: 0 if never broken, else the
    (1-indexed) position of the largest-nu break."""
    broken_positions = [k + 1 for k, b in enumerate(broken) if b]
    return max(broken_positions) if broken_positions else 0


def nu_star_interval(r: int) -> tuple[str, str]:
    return GRID_BOUNDARY_DISPLAY[r], GRID_BOUNDARY_DISPLAY[r + 1]


def null_result_flag(r: int) -> str | None:
    if r == 0:
        return "censored_below (bound never breaks in the tested range)"
    if r == len(NU_ORDER):
        return "censored_above (bound breaks at every tested nu)"
    return None


def compute_detail_for_n(
    df_all: pd.DataFrame, n_value: int, seed: int, n_resamples: int
) -> pd.DataFrame:
    """Ratio bootstrap for all 16 families x 8 nu, at one contraction dimension `n_value`."""
    df = df_all[df_all["n"] == n_value]
    families = list(
        itertools.product(BLOCK_SIZE_ORDER, SCALE_FORMAT_ORDER, ROUND_MODE_ORDER, RHT_ORDER)
    )
    rows: list[dict] = []
    for block_size, scale_format, round_mode, rht in families:
        for nu in NU_ORDER:
            cell = df[
                (df["block_size"] == block_size)
                & (df["scale_format"] == scale_format)
                & (df["round_mode"] == round_mode)
                & (df["rht"] == rht)
                & (df["nu"] == nu)
            ]
            if cell.empty:
                raise ValueError(
                    f"missing cell: n={n_value} {block_size, scale_format, round_mode, rht, nu}"
                )

            u_eff_unique = cell["u_eff_p50"].unique()
            if u_eff_unique.size != 1:
                raise ValueError(
                    f"u_eff_p50 not constant within cell: n={n_value} "
                    f"{block_size, scale_format, round_mode, rht, nu}: {u_eff_unique}"
                )
            u_eff = float(u_eff_unique[0])
            cota = u_eff / np.sqrt(n_value)

            be_values = cell["be_median"].to_numpy(dtype=np.float64)
            ratio_values = be_values / cota

            rng = np.random.default_rng(
                [*nu_seed(seed, block_size, scale_format, round_mode, rht, nu), int(n_value)]
            )
            point, ci_low, ci_high = bootstrap_ci(
                ratio_values, np.median, rng, n_resamples=n_resamples, ci=CI
            )
            broken = ci_low > 1.0

            rows.append(
                {
                    "n": int(n_value),
                    "block_size": block_size,
                    "scale_format": scale_format,
                    "round_mode": round_mode,
                    "rht": rht,
                    "nu": nu,
                    "n_trials": int(be_values.size),
                    "u_eff_p50": u_eff,
                    "cota": cota,
                    "ratio_median": point,
                    "ratio_ci_low": ci_low,
                    "ratio_ci_high": ci_high,
                    "broken": bool(broken),
                }
            )
    return pd.DataFrame(rows)


def compute_families_table(detail_for_n: pd.DataFrame) -> pd.DataFrame:
    """Fragility rank r(family), nu* interval, monotonicity -- one row per family."""
    families = list(
        itertools.product(BLOCK_SIZE_ORDER, SCALE_FORMAT_ORDER, ROUND_MODE_ORDER, RHT_ORDER)
    )
    n_value = int(detail_for_n["n"].iloc[0])
    rows: list[dict] = []
    for block_size, scale_format, round_mode, rht in families:
        sel = detail_for_n[
            (detail_for_n["block_size"] == block_size)
            & (detail_for_n["scale_format"] == scale_format)
            & (detail_for_n["round_mode"] == round_mode)
            & (detail_for_n["rht"] == rht)
        ].set_index("nu")
        broken = [bool(sel.loc[nu, "broken"]) for nu in NU_ORDER]
        r = fragility_rank(broken)
        low, high = nu_star_interval(r)
        rows.append(
            {
                "n": n_value,
                "block_size": block_size,
                "scale_format": scale_format,
                "round_mode": round_mode,
                "rht": rht,
                "r": r,
                "nu_star_low": low,
                "nu_star_high": high,
                "nu_star": f"({low}, {high})",
                "null_result": null_result_flag(r),
                "monotone": is_down_set(broken),
                "broken_at_nu": ",".join(nu for nu, b in zip(NU_ORDER, broken, strict=True) if b)
                or "(none)",
            }
        )
    return pd.DataFrame(rows)


def bigger_effect_label(p1_mag: float, p2_mag: float) -> str:
    """Which factor has the larger mean |effect| on nu*, as a canonical short
    label -- used consistently everywhere this comparison is reported, so an
    'are all 4 combos consistent' check compares like with like."""
    if p1_mag > p2_mag:
        return "block_size (P1)"
    if p2_mag > p1_mag:
        return "scale_format (P2)"
    return "NEITHER"


def conjunctive_verdict(r32_a: int, r16_a: int, r32_b: int, r16_b: int) -> str:
    """PREREGISTRATION.md sec 2.1's conjunctive H1 rule across both scale formats."""
    d_a, d_b = r32_a - r16_a, r32_b - r16_b
    if d_a > 0 and d_b > 0:
        return "H1 SUPPORTED (block-32 strictly more fragile at both scale formats)"
    if d_a == 0 and d_b == 0:
        return "H0 (no detectable block-size effect at either scale format)"
    if d_a < 0 and d_b < 0:
        return "H1 REJECTED -- directional reversal (block-16 is the more fragile configuration)"
    return (
        "INTERACTION-DOMINATED / INCONCLUSIVE (sign differs across scale formats, "
        "or one contrast is 0)"
    )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sweep-dir", type=Path, default=DEFAULT_SWEEP_DIR)
    parser.add_argument("--n-primary", type=int, default=DEFAULT_N_PRIMARY)
    parser.add_argument("--n-resamples", type=int, default=DEFAULT_N_RESAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    combined_path = args.sweep_dir.parent / f"{args.sweep_dir.name}.parquet"
    print(f"Loading {combined_path} ...")
    df_all = pd.read_parquet(combined_path, engine="pyarrow")
    n_grid = sorted(int(n) for n in df_all["n"].unique())
    if args.n_primary not in n_grid:
        raise ValueError(f"n_primary={args.n_primary} not in sweep's n grid {n_grid}")
    print(f"n grid: {n_grid}. n_primary={args.n_primary} (the only n that can move the "
          f"H1 verdict, per PREREGISTRATION.md sec 4.0); the rest are sensitivity checks.")

    started = time.perf_counter()
    detail = compute_detail_for_n(df_all, args.n_primary, args.seed, args.n_resamples)
    families_table = compute_families_table(detail)
    print(f"n_primary={args.n_primary}: computed {len(detail)} (family, nu) ratios in "
          f"{time.perf_counter() - started:.1f}s.")

    # PREREGISTRATION.md sec 4.0: every non-primary n is a disclosed
    # robustness/sensitivity check, computed the same way, but never allowed
    # to move the H1 verdict above.
    sensitivity_frames = [families_table]
    for n_value in n_grid:
        if n_value == args.n_primary:
            continue
        sens_started = time.perf_counter()
        sens_detail = compute_detail_for_n(df_all, n_value, args.seed, args.n_resamples)
        sensitivity_frames.append(compute_families_table(sens_detail))
        print(f"  sensitivity n={n_value}: computed in "
              f"{time.perf_counter() - sens_started:.1f}s.")
    sensitivity_table = pd.concat(sensitivity_frames, ignore_index=True)

    # ------------------------------------------------------------------
    # H1 analysis: P1 (block-size effect, the sole confirmatory test of H1)
    # and P2 (scale-format effect, design-validity only) -- canonical
    # sub-configuration first, then the other three (round_mode, rht) combos
    # as a robustness check. PREREGISTRATION.md sec 4.1.
    # ------------------------------------------------------------------
    def r_of(block_size: int, scale_format: str, round_mode: str, rht: bool) -> int:
        row = families_table[
            (families_table["block_size"] == block_size)
            & (families_table["scale_format"] == scale_format)
            & (families_table["round_mode"] == round_mode)
            & (families_table["rht"] == rht)
        ]
        return int(row["r"].iloc[0])

    def h1_analysis(round_mode: str, rht: bool) -> dict:
        r32_e8 = r_of(32, "e8m0", round_mode, rht)
        r16_e8 = r_of(16, "e8m0", round_mode, rht)
        r32_e4 = r_of(32, "e4m3", round_mode, rht)
        r16_e4 = r_of(16, "e4m3", round_mode, rht)
        r_e4_16 = r_of(16, "e4m3", round_mode, rht)
        r_e8_16 = r_of(16, "e8m0", round_mode, rht)
        r_e4_32 = r_of(32, "e4m3", round_mode, rht)
        r_e8_32 = r_of(32, "e8m0", round_mode, rht)

        p1a = r32_e8 - r16_e8  # block effect at e8m0
        p1b = r32_e4 - r16_e4  # block effect at e4m3
        p2a = r_e4_16 - r_e8_16  # scale effect at block16
        p2b = r_e4_32 - r_e8_32  # scale effect at block32

        verdict = conjunctive_verdict(r32_e8, r16_e8, r32_e4, r16_e4)
        any_non_monotone = not all(
            families_table[
                (families_table["block_size"].isin([16, 32]))
                & (families_table["scale_format"].isin(["e8m0", "e4m3"]))
                & (families_table["round_mode"] == round_mode)
                & (families_table["rht"] == rht)
            ]["monotone"]
        )
        return {
            "round_mode": round_mode,
            "rht": rht,
            "r_block32_e8m0": r32_e8,
            "r_block16_e8m0": r16_e8,
            "r_block32_e4m3": r32_e4,
            "r_block16_e4m3": r16_e4,
            "P1a_block_effect_e8m0": p1a,
            "P1b_block_effect_e4m3": p1b,
            "P2a_scale_effect_block16": p2a,
            "P2b_scale_effect_block32": p2b,
            "verdict": verdict,
            "any_non_monotone": any_non_monotone,
        }

    canonical = h1_analysis(CANONICAL_ROUND_MODE, CANONICAL_RHT)
    robustness = [
        h1_analysis(rm, rht)
        for rm, rht in itertools.product(ROUND_MODE_ORDER, RHT_ORDER)
        if not (rm == CANONICAL_ROUND_MODE and rht == CANONICAL_RHT)
    ]

    # PREREGISTRATION.md sec 5.1: "All cells censored above (bound breaks
    # even at the lightest-tailed nu tested, r=8 everywhere): H1 untestable
    # on this grid ... labelled a grid mis-specification, not an H1
    # confirmation." This supersedes the mechanical P1/P2 verdict above
    # (which would otherwise read the all-r=8 pattern as a literal "H0" --
    # technically true by sec 2.1's table, but not the informative reading
    # sec 5.1 requires) whenever it applies at n_primary specifically.
    grid_mis_specified_above = bool((families_table["r"] == len(NU_ORDER)).all())
    grid_mis_specified_below = bool((families_table["r"] == 0).all())

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config = {
        "script": "localize_nu_star.py",
        "version": 1,
        "sweep_dir": str(args.sweep_dir),
        "n_primary": args.n_primary,
        "n_resamples": args.n_resamples,
        "seed": args.seed,
        "ci": CI,
        "canonical_round_mode": CANONICAL_ROUND_MODE,
        "canonical_rht": CANONICAL_RHT,
    }
    canonical_json = json.dumps(config, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()[:12]
    stem = f"nu_star_{digest}"

    families_table.to_parquet(OUT_DIR / f"{stem}_families.parquet", engine="pyarrow")
    families_table.to_csv(OUT_DIR / f"{stem}_families.csv", index=False)
    detail.to_parquet(OUT_DIR / f"{stem}_detail.parquet", engine="pyarrow")
    sensitivity_table.to_parquet(OUT_DIR / f"{stem}_sensitivity.parquet", engine="pyarrow")
    sensitivity_table.to_csv(OUT_DIR / f"{stem}_sensitivity.csv", index=False)
    (OUT_DIR / f"{stem}.config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
    )

    statement = build_statement(
        families_table,
        canonical,
        robustness,
        args.n_primary,
        sensitivity_table,
        grid_mis_specified_above,
        grid_mis_specified_below,
    )
    print("\n" + statement + "\n")
    (OUT_DIR / f"{stem}_statement.txt").write_text(statement + "\n", encoding="utf-8")

    fig = make_figure(detail, args.n_primary)
    fig_path = OUT_DIR / f"{stem}_nu_star.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    print(f"Wrote families table, detail, config, statement and figure "
          f"({fig_path.name}) to {OUT_DIR} (stem={stem})")


def build_statement(
    families_table: pd.DataFrame,
    canonical: dict,
    robustness: list[dict],
    n_primary: int,
    sensitivity_table: pd.DataFrame,
    grid_mis_specified_above: bool,
    grid_mis_specified_below: bool,
) -> str:
    lines: list[str] = []
    lines.append(
        f"Step 4.2 nu* localization (PRIMARY RESULT), n_primary={n_primary}. "
        f"16 families (block_size x scale_format x round_mode x rht) x 8 nu."
    )
    lines.append("")
    n_censored_below = int((families_table["r"] == 0).sum())
    n_censored_above = int((families_table["r"] == len(NU_ORDER)).sum())
    n_non_monotone = int((~families_table["monotone"]).sum())
    lines.append(
        f"Censored below (bound never breaks, r=0): {n_censored_below}/16\n"
        f"Censored above (bound breaks at every tested nu, r=8): {n_censored_above}/16\n"
        f"Non-monotone (flagged inconclusive-exploratory per sec 1.1): {n_non_monotone}/16"
    )
    lines.append("")
    lines.append("Per-family fragility rank and nu* interval:")
    for _, row in families_table.sort_values(
        ["round_mode", "rht", "scale_format", "block_size"]
    ).iterrows():
        flag = f"  [{row['null_result']}]" if row["null_result"] else ""
        mono = "" if row["monotone"] else "  [NON-MONOTONE -- inconclusive-exploratory]"
        lines.append(
            f"  block={row['block_size']:>2} scale={row['scale_format']} "
            f"round={row['round_mode']} rht={row['rht']!s:>5}: r={row['r']} "
            f"nu* = ({row['nu_star_low']}, {row['nu_star_high']})  "
            f"broken at nu in {{{row['broken_at_nu']}}}{flag}{mono}"
        )
    lines.append("")
    if grid_mis_specified_above or grid_mis_specified_below:
        lines.append("!" * 72)
        if grid_mis_specified_above:
            lines.append(
                "GRID MIS-SPECIFICATION (PREREGISTRATION.md sec 5.1): the bound breaks at "
                "EVERY tested nu, in EVERY one of the 16 families, at n_primary. This means "
                "H1 IS UNTESTABLE ON THIS GRID -- not H0, and not an H1 confirmation. The "
                "P1/P2 rank comparisons below are still computed (every r=8, so they "
                "mechanically read as 'H0: no detectable difference'), but sec 5.1 explicitly "
                "supersedes that reading here: there is no headroom left in the tested nu "
                "range to observe *where* fragility differs between block sizes or scale "
                "formats, because the bound has already failed even in the near-Gaussian "
                "limit. Reported per sec 5.1's own words: 'a strong result in its own right' "
                "-- the c=1 prefactor is not calibrated at n_primary -- 'labelled a grid "
                "mis-specification, not an H1 confirmation.'"
            )
        if grid_mis_specified_below:
            lines.append(
                "GRID MIS-SPECIFICATION (r=0 everywhere): the bound never breaks at any "
                "tested nu, in any family, at n_primary -- H1 is equally untestable in this "
                "direction (no fragile-enough regime was sampled)."
            )
        lines.append("!" * 72)
        lines.append("")
    lines.append("=" * 72)
    lines.append("H1 ANALYSIS -- canonical sub-configuration (round_mode=rtne, rht=False)")
    lines.append("=" * 72)
    if grid_mis_specified_above or grid_mis_specified_below:
        lines.append(
            "(The verdict computed below is the MECHANICAL P1/P2 rank comparison per sec "
            "2.1's literal table. Per the grid mis-specification banner above, it is NOT "
            "the informative reading of this data -- read that banner first.)"
        )
    lines.append(
        f"r(block=32, e8m0) = {canonical['r_block32_e8m0']}   "
        f"r(block=16, e8m0) = {canonical['r_block16_e8m0']}   "
        f"P1a (block effect @ e8m0) = {canonical['P1a_block_effect_e8m0']:+d}"
    )
    lines.append(
        f"r(block=32, e4m3) = {canonical['r_block32_e4m3']}   "
        f"r(block=16, e4m3) = {canonical['r_block16_e4m3']}   "
        f"P1b (block effect @ e4m3) = {canonical['P1b_block_effect_e4m3']:+d}"
    )
    lines.append(f"P1 VERDICT (sole confirmatory test of H1): {canonical['verdict']}")
    lines.append("")
    lines.append(
        f"P2a (scale effect @ block16, e4m3 - e8m0) = {canonical['P2a_scale_effect_block16']:+d}   "
        f"P2b (scale effect @ block32, e4m3 - e8m0) = {canonical['P2b_scale_effect_block32']:+d}   "
        "(design-validity only, NOT a test of H1 -- sec 4.1)"
    )
    p1_mag = (abs(canonical["P1a_block_effect_e8m0"]) + abs(canonical["P1b_block_effect_e4m3"])) / 2
    p2_mag = (
        abs(canonical["P2a_scale_effect_block16"]) + abs(canonical["P2b_scale_effect_block32"])
    ) / 2
    bigger = bigger_effect_label(p1_mag, p2_mag)
    bigger_suffix = " (P1 and P2 have equal magnitude)" if bigger == "NEITHER" else ""
    lines.append(
        f"Mean |effect| in fragility-rank steps: P1 (block size) = {p1_mag:.2f}, "
        f"P2 (scale format) = {p2_mag:.2f}. LARGER EFFECT ON nu*: {bigger}{bigger_suffix}."
    )
    p1_consistent = canonical["P1a_block_effect_e8m0"] == canonical["P1b_block_effect_e4m3"]
    p2_consistent = canonical["P2a_scale_effect_block16"] == canonical["P2b_scale_effect_block32"]
    lines.append(
        "INTERACTION: block size's effect on nu* "
        + ("does NOT" if p1_consistent else "DOES")
        + f" depend on scale format (P1a={canonical['P1a_block_effect_e8m0']:+d} vs "
        f"P1b={canonical['P1b_block_effect_e4m3']:+d}); scale format's effect on nu* "
        + ("does NOT" if p2_consistent else "DOES")
        + f" depend on block size (P2a={canonical['P2a_scale_effect_block16']:+d} vs "
        f"P2b={canonical['P2b_scale_effect_block32']:+d})."
    )
    if canonical["any_non_monotone"]:
        lines.append(
            "WARNING: at least one of the four canonical-cell families is flagged "
            "non-monotone -- per sec 1.1, this P1 result is INCONCLUSIVE-EXPLORATORY, "
            "not confirmatory evidence for or against H1, notwithstanding the verdict above."
        )
    lines.append("")
    lines.append("=" * 72)
    lines.append("ROBUSTNESS -- same P1/P2 comparisons at the other 3 (round_mode, rht) combos")
    lines.append("(exploratory: neither round_mode nor rht is part of the primary 2x2 design)")
    lines.append("=" * 72)
    for r in robustness:
        p1_mag_r = (abs(r["P1a_block_effect_e8m0"]) + abs(r["P1b_block_effect_e4m3"])) / 2
        p2_mag_r = (abs(r["P2a_scale_effect_block16"]) + abs(r["P2b_scale_effect_block32"])) / 2
        bigger_r = bigger_effect_label(p1_mag_r, p2_mag_r)
        mono_note = "  [NON-MONOTONE present]" if r["any_non_monotone"] else ""
        lines.append(
            f"  round_mode={r['round_mode']:<4} rht={r['rht']!s:<5}: "
            f"P1a={r['P1a_block_effect_e8m0']:+d} P1b={r['P1b_block_effect_e4m3']:+d} "
            f"P2a={r['P2a_scale_effect_block16']:+d} P2b={r['P2b_scale_effect_block32']:+d}  "
            f"verdict={r['verdict']}  larger effect: {bigger_r}{mono_note}"
        )
    all_bigger = [bigger] + [
        bigger_effect_label(
            (abs(r["P1a_block_effect_e8m0"]) + abs(r["P1b_block_effect_e4m3"])) / 2,
            (abs(r["P2a_scale_effect_block16"]) + abs(r["P2b_scale_effect_block32"])) / 2,
        )
        for r in robustness
    ]
    lines.append("")
    lines.append(
        f"Conclusion ('{bigger}' has the larger effect on nu*) holds across all 4 "
        f"round_mode x rht combinations: {all(b == bigger for b in all_bigger)}. "
        f"Per-combination verdicts: {all_bigger}."
    )
    lines.append("")
    lines.append("=" * 72)
    lines.append(
        "SENSITIVITY (PREREGISTRATION.md sec 4.0): r(family) at every n in the grid, "
        "disclosed but never substituted for n_primary in the verdict above."
    )
    lines.append("=" * 72)
    n_values = sorted(sensitivity_table["n"].unique())
    for n_value in n_values:
        sub = sensitivity_table[sensitivity_table["n"] == n_value]
        n_above = int((sub["r"] == len(NU_ORDER)).sum())
        n_below = int((sub["r"] == 0).sum())
        primary_tag = "  <-- n_primary" if n_value == n_primary else ""
        lines.append(
            f"  n={n_value:>5}: censored-above (r=8) {n_above:2d}/16, "
            f"censored-below (r=0) {n_below:2d}/16, "
            f"mean r = {sub['r'].mean():.2f}{primary_tag}"
        )
    lines.append("")
    if grid_mis_specified_above:
        lines.append(
            "Whether n_primary's all-broken pattern is n=4096-specific or holds at every "
            "tested n is read directly off the table above: if smaller n show fewer "
            "censored-above families, the gap between empirical error and cota(n) is "
            "growing with n (consistent with Step 4.1's finding that the empirical decay "
            "slope is uniformly less steep than the -0.5 the c=1 prefactor assumes, so the "
            "unresolved gap compounds as n grows); if every n is equally saturated, the "
            "miscalibration is present at every tested scale, not an n=4096 artifact."
        )
    lines.append("")
    lines.append(
        "This script fits and reports Step 4.2 only; it does not perform the causal "
        "decomposition (Step 4.3, a separate step not run here)."
    )
    return "\n".join(lines)


def make_figure(detail: pd.DataFrame, n_primary: int) -> plt.Figure:
    """The headline figure: ratio (with CI band) vs nu, canonical sub-config only,
    one panel per (block_size, scale_format), nu*=1.0 threshold marked."""
    sel = detail[(detail["round_mode"] == CANONICAL_ROUND_MODE) & (~detail["rht"])]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.5), sharex=True)
    x = np.arange(len(NU_ORDER))
    panels = [(16, "e8m0"), (32, "e8m0"), (16, "e4m3"), (32, "e4m3")]

    for ax, (block_size, scale_format) in zip(axes.ravel(), panels, strict=True):
        row = sel[
            (sel["block_size"] == block_size) & (sel["scale_format"] == scale_format)
        ].set_index("nu").loc[NU_ORDER]
        point = row["ratio_median"].to_numpy()
        lo = row["ratio_ci_low"].to_numpy()
        hi = row["ratio_ci_high"].to_numpy()
        broken = row["broken"].to_numpy()

        ax.axhline(1.0, color="gray", ls="--", lw=1.3, zorder=1, label="threshold (ratio = 1.0)")
        ax.errorbar(
            x, point, yerr=[point - lo, hi - point],
            fmt="none", ecolor="0.35", elinewidth=1.4, capsize=4, zorder=2,
        )
        colors = np.where(broken, "#c0392b", "#2166ac")
        ax.scatter(x, point, c=colors, s=55, zorder=3, edgecolors="white", linewidths=0.8)

        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(NU_DISPLAY)
        ax.set_title(f"block_size={block_size}, scale_format={scale_format}", fontsize=11)
        ax.grid(True, which="both", alpha=0.15)
        ax.set_xlabel("ν (heavier tail →)")
        ax.set_ylabel("median(BE) / cota(n_primary)")

    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D([0], [0], color="gray", ls="--", lw=1.3, label="threshold (ratio = 1.0)"),
        Line2D(
            [0], [0], marker="o", color="w", markerfacecolor="#2166ac", markersize=9,
            label="bound holds (CI lower bound ≤ 1.0)",
        ),
        Line2D(
            [0], [0], marker="o", color="w", markerfacecolor="#c0392b", markersize=9,
            label="bound BROKEN (CI lower bound > 1.0)",
        ),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, 0.015), frameon=False)
    fig.suptitle(
        f"Backward error vs. theoretical bound, n_primary={n_primary}\n"
        "round_mode=rtne, rht=False (canonical) -- error bars are 95% bootstrap CIs",
        fontsize=13, y=0.975,
    )
    fig.subplots_adjust(left=0.08, right=0.97, top=0.89, bottom=0.11, hspace=0.30, wspace=0.24)
    return fig


if __name__ == "__main__":
    main()
