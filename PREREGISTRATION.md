# Preregistration — Numerical Error Analysis of Quantized GEMM

Recorded before running the confirmatory sweeps referenced in the paper.
Any deviation from this document made after data collection starts must be
logged under "Amendments and deviations" below, as a dated and justified
entry — never silently edited in above.

- **Version:** 1 (git tag `preregistration-v1`)
- **Written:** 2026-08-19
- **Deadline:** see README.md (August 29, 2026, 23:59 AoE)
- **Status of data collection:** not started as of 2026-08-19.

### Editorial conventions used in this document

- Unmarked prose in §1–§5 is the author's own commitment.
- Blocks marked **[R-added]** were inserted during the adversarial review of
  2026-08-19 to close a gap that made an author commitment ambiguous or
  undefined. They add decision procedure, not scientific claims. Each is
  cross-referenced to a numbered finding in §9.
- Blocks marked **⚑ INFERRED** encode intent that the reviewer inferred from
  surrounding context rather than from an explicit statement. The author must
  confirm or overwrite each of these before the first confirmatory run.
- Items marked **⚠ OPEN** are commitments that are *structurally* fixed here
  but whose *value* is deliberately not yet chosen. They must be closed by a
  dated amendment before the first confirmatory run. See §7.

### Notation **[R-added, see R14]**

- `ν` — degrees of freedom of the t-Student distribution from which GEMM
  operands are drawn. **Smaller ν = heavier tails.** ν = ∞ is the Gaussian case.
- `n` — the contraction (inner) dimension of the GEMM.
- *block* — the microscaling block size, 16 or 32 elements.
- *scale* — the dtype of the per-block scale factor, E4M3 or E8M0.
- *cell* — one of the four (block, scale) combinations of the 2×2 factorial.

## 1. Primary Hypothesis (H1)

There exists a critical value ν* such that, for ν < ν*, the ratio
empirical_error / theoretical_bound exceeds 1, and ν*(block=32) > ν*(block=16).
That is: the block-32 configuration breaks under lighter-tailed distributions
than block-16 (block-32 is the more fragile configuration).

Because only 8 discrete ν values are tested, ν* is reported as an interval
between the largest ν where the bound **breaks** and the smallest ν where it
**holds**, not as a point estimate.
<!-- [R-added, R1] The original draft read "between the largest ν where the
bound holds and the smallest ν where it breaks", which brackets the wrong
side of the crossing given H1's own direction (breakage occurs *below* ν*).
Corrected to match H1. This is an internal-consistency fix, not a change of
commitment. -->

### 1.1 Localizing ν* on a discrete grid, including censored cases **[R-added, see R2, R3]**

Let the tested grid be ν₁ < ν₂ < … < ν₈, and for each cell *c* let
`B(c) ⊆ {ν₁…ν₈}` be the set of tested ν at which the bound is broken by the
criterion in §3, evaluated at the primary n (§4.0).

Define the **fragility rank** `r(c) ∈ {0, 1, …, 8}`:

    r(c) = 0                       if B(c) = ∅
    r(c) = k  where ν_k = max B(c) otherwise

and read ν* off it as the open interval

    ν*(c) ∈ (ν_r, ν_{r+1}),   with the conventions  ν₀ := 0  and  ν₉ := ∞.

This single integer handles every case, including the two censored ones, with
no post-hoc judgement:

| Outcome | r | Reported as |
|---|---|---|
| Bound never breaks in range | 0 | ν* < ν₁ — **censored below** |
| Bound breaks at some interior ν | 1–7 | ν* ∈ (ν_r, ν_{r+1}) |
| Bound breaks at every tested ν | 8 | ν* > ν₈ — **censored above** |

Censored cells are *not* undefined and are *not* dropped: rank 0 and rank 8 are
ordinary values of r and enter the P1 comparison exactly like any other rank.
A cell censored below is ordered *below* every uncensored cell; a cell censored
above is ordered *above* every uncensored cell. The H1/H0 comparison therefore
remains defined for all 2⁴ combinations of censoring across the four cells.

**Monotonicity precondition.** ν* as a "crossing" presumes that breakage is a
down-set in ν: if the bound breaks at ν_k it also breaks at every ν_j < ν_k.
This is checked, not assumed. If for any cell `B(c)` is not a down-set — i.e.
there exist ν_i < ν_j with ν_i ∉ B(c) and ν_j ∈ B(c) — then that cell is
flagged **non-monotone** in the results table, r(c) is still computed by the
rule above (the definition never becomes ambiguous), but the P1 result is
reported as **inconclusive-exploratory** rather than as confirmatory evidence
for or against H1, and the non-monotone pattern is shown in full. Deciding
after the fact which of several crossings is "the real one" is exactly the
degree of freedom this rule exists to remove.

## 2. Null Hypothesis (H0)

The ratio stays ≤ 1 for all tested ν in both block sizes, or ν*(32) ≈ ν*(16)
(no detectable difference in fragility between block sizes).

### 2.1 Making "≈" and the outcome space exhaustive **[R-added, see R3, R4]**

"≈" is operationalized as **equal fragility rank**: ν*(32) ≈ ν*(16) iff
r(block32, s) = r(block16, s), i.e. the two ν* intervals are the same grid
cell. Any difference of one or more grid steps counts as a detected difference.
The 2×2 design gives two block-size contrasts, one per scale format s ∈ {E4M3,
E8M0}, and the verdict is **conjunctive** across them:

| Pattern across both scale formats s | Verdict |
|---|---|
| r(32,s) > r(16,s) for both s | **H1 supported** |
| r(32,s) = r(16,s) for both s | **H0** — no detectable difference |
| r(32,s) < r(16,s) for both s | **H1 rejected — directional reversal** (block-16 is the more fragile configuration) |
| sign differs across s, or one contrast is 0 | **Interaction-dominated / inconclusive** — reported as such; not counted as support for H1 |

The conjunctive rule is what stops the block-size effect being claimed from
whichever scale-format arm happens to cooperate. The reversal row is stated
explicitly because it is a scientifically important outcome that contradicts
the NVIDIA-gap narrative motivating this study, and it must not be
re-described after the fact as "a null result" or moved into §5.

## 3. Operational Definition of "Bound Broken" (🧠7)

The bound is considered broken at (format, ν, n) if the lower bound of the
95% bootstrap CI of the median of the ratio exceeds 1.0.
- Median, not mean — heavy-tailed inputs may lack finite moments.
- CI lower bound, not point estimate — the conservative choice.
Fixed now; not revisited after looking at the data.

This presupposes a working definition of theoretical_bound(n). Pending formal
resolution in Step 2.3 (🧠1), the provisional definition is:

  u_eff(format, block, ν) := median (and, as an indicative secondary
  statistic, p99) of the per-element relative quantization error,
  measured empirically for that configuration.
  bound(n) = c · √n · u_eff

### 3.1 The bound is PROVISIONAL — status and consequences **[R-added, see R5]**

The definition above is a **placeholder that stands in for unresolved work**,
not a settled result. 🧠1 — what a theoretical backward-error bound even means
for block-scaled formats, where the scale factor is data-dependent and shared
across a block — is scheduled for Step 2.3 and is *not* resolved by this
document. Consequences, committed to now:

1. If 🧠1 resolves to a different bound, the H1 test changes with it. That
   substitution must be filed as a dated amendment in §8 **before** the
   confirmatory analysis is re-run, and both the provisional-bound and
   resolved-bound results must be reported side by side in the paper.
2. This provisional bound is **not** claimed as a theoretical contribution of
   the paper. It is an empirical yardstick.
3. Because `u_eff` is itself indexed by ν, both sides of the ratio move with ν.
   H1 as tested here is therefore a statement about whether the **√n growth
   law** survives heavy tails, *not* about whether heavy tails increase
   absolute error. The paper must say this in those words. See R6 for the
   reviewer's unresolved objection to this construction.

### 3.2 Estimation details that would otherwise be free parameters **[R-added, see R7, R8]**

- **c is fixed a priori at c = 1** for all confirmatory analysis. ⚑ INFERRED
  from the project's framing of the comparator as "the probabilistic √n·u
  bound", whose classical form has c = 1. Any ĉ fitted from data is a
  descriptive, exploratory quantity only (§5(b)) and may never be substituted
  back into the confirmatory ratio: a bound whose constant is fitted to the
  same data it is tested against cannot be falsified by that data.
- **`u_eff` is estimated on an independent calibration sample** — separate
  seeds from the confirmatory sweep, drawn from the same configuration — and
  is frozen and tabulated before the confirmatory ratios are computed. Reusing
  the confirmatory sample to calibrate the denominator would partially
  self-normalize the ratio.
- **Bootstrap:** percentile method, B = 10 000 resamples, RNG seeded
  explicitly per the repo's Generator convention. The **resampling unit is the
  independent trial (seed)**, never the individual matrix element: elements
  within a block share a scale factor and are not independent, so an
  element-level bootstrap would give anticonservative intervals and inflate
  the break rate. ⚑ INFERRED (method and B are conventional defaults; the
  resampling unit follows from the block-scaling structure).
- `qgemm.stats.mean_ci95` is a normal-approximation CI of the **mean** and is
  therefore *not* the estimator specified here. A median bootstrap must be
  implemented before data collection. See R15.

## 4. Primary vs. Exploratory Comparisons

Primary (max 3, these are what answer H1):
- P1 — ν*(block 16) vs ν*(block 32), scale format held constant.
  [block-size causal effect]
- P2 — ν*(scale E4M3) vs ν*(scale E8M0), block size held constant.
  [scale-format causal effect]
- P3 — MXFP4 preset (block32+E8M0) vs NVFP4 preset (block16+E4M3),
  decomposed via P1 + P2 + interaction. [practical bottom line, ties to
  NVIDIA's reported 36%-more-tokens gap]

No multiple-comparison correction is applied across the 8 ν levels within
each of P1–P3: ν*-localization is treated as curve-crossing detection along
a single ordered axis, not as a family of independent hypothesis tests.

Everything else — hypothetical blocks 8/64, RHT on/off, rounding mode,
accumulation precision, real activations — is exploratory and reported as
such, never as evidence for or against H1.

### 4.0 ν* is defined at a single primary n **[R-added, see R9]**

The §3 break criterion is indexed by (format, ν, n), but ν* is not. The
aggregation is fixed here: **ν* is computed at one primary contraction
dimension, `n_primary`** (⚠ OPEN — value to be declared in §7 and frozen in
the sweep config before the first confirmatory run). All other n in the grid
are **robustness checks**: they are reported as a sensitivity table alongside
the primary result, and disagreement between them and n_primary is disclosed,
but they never replace n_primary in the H1 verdict. Without this, "does the
bound break at this ν" would have as many answers as there are n, and the
choice among them would be made after seeing the data.

### 4.1 Which comparisons actually bear on H1 **[R-added, see R10]**

H1, as stated in §1, is a claim about **block size only**. Labelling P1–P3
uniformly as "primary" overstates the evidentiary role of P2 and P3:

- **P1 is the sole confirmatory test of H1**, evaluated by the conjunctive
  rule in §2.1.
- **P2 is a design-validity comparison**, not a test of H1. Its role is to
  establish that scale format is a separable factor, which is what licenses
  the causal reading of P1 in the 2×2. It is prespecified and reported
  whatever it shows.
- **P3 is a derived summary**, not an independent test: it is a function of
  P1, P2 and their interaction, and is reported for the practical
  MXFP4-vs-NVFP4 bottom line. It cannot furnish support for H1 that P1 did
  not already furnish.

This relabelling changes no comparison, statistic or threshold; it fixes which
of them is allowed to move the H1 verdict.

### 4.2 Scope of the no-correction stance **[R-added, see R11]**

The author's stated argument — that ν*-localization along a single ordered ν
axis is a curve-crossing problem rather than a family of independent tests —
is accepted as sound *for the 8 ν levels within one cell*, which is the scope
it was written for. Extending it to the rest of the multiplicity in this
design, with the author's own logic:

- **Across the 4 cells:** no correction. The cells are the arms of a factorial
  design, not competing hypotheses; every cell is reported regardless of
  outcome, and the conjunctive rule in §2.1 already requires *both* block-size
  contrasts to agree, which is stricter than either arm alone.
- **Across P1–P3:** no correction, because per §4.1 only P1 can move the H1
  verdict; P2 and P3 are descriptive and are reported unconditionally.
- **Across n:** no correction, because per §4.0 only n_primary enters the
  verdict.

⚑ INFERRED: the extensions above are the reviewer's reading of the author's
stated rationale, not the author's words. **The residual risk is stated
plainly and is not fully mitigated:** because r(c) is defined by the *largest*
broken ν, a single false-positive break at a high ν raises r(c) directly, so
the no-correction stance is not error-symmetric here — it biases ν* upward
rather than merely widening it. The monotonicity check in §1.1 is the only
safeguard against an isolated spurious break, and it is a flag, not a test.
The number of CIs computed is fixed in advance by the frozen grid and is
reported in the paper. See R11 for the option the author may wish to take
instead.

## 5. Contingency: Null Result (🧠8)

If the ratio stays ≤ 1 for all tested ν in both block sizes, the paper
reframes as:
"Empirical characterization of the effective unit roundoff and scaling
regime for microscaling formats under heavy-tailed inputs."
Contributions in that case: (a) the u_eff table by format/block/ν —
unpublished elsewhere; (b) the empirical constant c; (c) evidence that
classical probabilistic bounds hold even in the adversarial regime — a
clean, useful verification result; (d) the scale-invariance analysis.

### 5.1 Contingencies for the outcomes §5 does not cover **[R-added, see R4, R12]**

- **Directional reversal** (row 3 of §2.1): reported as H1 rejected, in the
  abstract and in the title-level claim. It is not reframed as a null result.
- **All cells censored above** (bound breaks even at the lightest-tailed ν
  tested, r = 8 everywhere): H1 untestable on this grid, because the ν range
  was mis-specified. Reported as such, with the finding that the bound fails
  even in the near-Gaussian regime — a strong result in its own right, but
  labelled a grid mis-specification, not an H1 confirmation.
- **Interaction-dominated** (row 4 of §2.1): reported as an interaction
  between block size and scale format, with P3 as the practical summary and no
  H1 verdict.
- On §5(c): the provisional bound of §3 is **not** the classical probabilistic
  bound — it substitutes an empirically measured `u_eff` for the format's
  nominal unit roundoff. If the null contingency fires, contribution (c) must
  be worded as "the √n scaling law holds with an empirically calibrated
  u_eff", not as a verification of the classical bound. See R13.
- On §5(d): the scale-invariance analysis is listed as a contribution but has
  no analysis plan anywhere in this document. ⚠ OPEN — either specify it in
  §7 or demote it to exploratory. See R16.

## 6. Sweep grid, sample size, stopping rule **[R-added, see R14]**

The draft reviewed on 2026-08-19 dropped these sections, which the earlier
skeleton carried. A preregistration that does not freeze the grid does not
bind: ν levels or n values could be added later without any of it counting as
a deviation.

- **Frozen grid.** The confirmatory grid is the JSON config under `configs/`
  whose sha256 is recorded in §7 once written. Per repo convention, every
  results file is `sweep_{sha256(config)[:12]}.parquet` with the config saved
  alongside, so the frozen grid is verifiable from the artifacts. Any run
  whose config hash differs from the recorded one is exploratory by
  construction. `configs/default.json` is a leftover placeholder and is **not**
  the confirmatory config.
- **Sample size.** Fixed number of independent trials (seeds) per cell,
  declared in the config and never extended after inspecting results.
  ⚠ OPEN — value in §7.
- **Stopping rule.** Data collection stops when the frozen grid has been run to
  its declared trial count. There is no interim look at the ratio, and no
  optional stopping. If a run fails for infrastructure reasons it is re-run at
  the same seeds; a re-run is not an additional sample.

## 7. Open items — must be closed by a dated amendment before the first confirmatory run

These are structurally committed above but numerically unchosen. Each is a
scientific decision that belongs to the author, and each is listed here rather
than filled in by the reviewer.

1. **⚠ The 8 ν values** ν₁ … ν₈, and their range. The entire censoring
   analysis in §1.1 is relative to this grid.
2. **⚠ `n_primary`**, plus the full list of n used as robustness checks (§4.0).
3. **⚠ The primary error metric** — "empirical_error" is never operationally
   defined. The project studies *backward* error, while `qgemm.metrics`
   currently exposes forward-error metrics (`relative_error`, `rms_error`, …)
   and `qgemm.bounds.gamma_n` is a backward-error growth factor. One metric
   must be named as primary, and it must be commensurable with the
   denominator in §3. See R17.
4. **⚠ Trials (seeds) per cell**, and the separate seed range for the `u_eff`
   calibration sample (§3.2).
5. **⚠ The role of FP8** — reference baseline outside the 2×2, or part of the
   primary comparisons. The 2×2 factorial as described covers only the FP4
   microscaling family.
6. **⚠ The scale-invariance analysis** (§5(d)) — specify or demote.
7. **⚠ Confirmatory config path and sha256** (§6).

Until items 1–4 are closed, no run may be labelled confirmatory.

## 8. Amendments and deviations from this preregistration

Append-only. Every entry is dated, states what changed, and states why. Never
edit §1–§7 in place; the tagged commit `preregistration-v1` is the reference
version and any later state is diffable against it.

| Date | Section | Change | Justification | Pre- or post-data |
|---|---|---|---|---|
| 2026-08-20 | §3, §3.1 | 🧠1 resolved: the provisional `bound(n) = c · √n · u_eff` is superseded by `cota(n, format, block, ν) = c · u_eff(format, block, ν) / √n` — a **decaying** form, not a growing one | The primary route accumulates in exact fp64, so the repeated-rounding mechanism that gives γ_n and √n·u their n-dependence is not physically present. The surviving mechanism is partial cancellation of one-time input-quantization errors, which makes `BE` decay as n^(-1/2). Measured: slopes −0.4978 (block 16) / −0.4966 (block 32) at ν = 30. | **Pre-data** |
| 2026-08-22 | §3, §3.2 | Robust-statistics methodology resolved (Step 3.4): median and p90 carry all confirmatory statistical weight (ν* localization included); p99 is reported for every cell but is INDICATIVE ONLY, at any trial count, never the basis of a firm claim | Empirical 95%-CI coverage simulation (t-Student ν=2, 1000 experiments/cell): median coverage 95.5% (n_trials=1000) / 94.8% (n_trials=5000), close to nominal; p99 coverage 92.7% (n_trials=1000) / 94.0% (n_trials=5000), measurably below the median's and below nominal at n_trials=1000, narrower but still short at n_trials=5000 — a percentile bootstrap cannot invent values past the sample's own extreme tail. | **Pre-data** |
| 2026-08-22 | §4.0, §7 item 2 | **`n_primary` declared: 4096** (the largest tested contraction dimension, `configs/sweep_main.yaml`'s `main_grid.n` upper end) | Author's choice, on grounds stated independently of any ν*/H1 outcome: n=4096 is closest to real GEMM contraction dimensions in LLM inference/training (the most externally relevant regime), and is where the √n-decay derivation's asymptotic assumptions (Step 🧠1, cancellation over many roughly-independent blocks) are most likely to hold cleanly. See 8.3 for the full disclosure, including timing. | **Post-data (disclosed, see 8.3)** |
| 2026-08-23 | §1, §4, §4.1 | **Pivot to Step 4.3 as PRIMARY vehicle for the block-vs-scale question, applied to observed `median(BE)` directly rather than to ν*.** P1/P2/P3 as formulated (comparisons of ν*) cannot be evaluated: Step 4.2 found the bound broken at every tested ν, in every one of the 16 families, at every tested n — there is no ν* on this grid for any comparison to be made between. EXPLORATORY, not a resolution of H1 as originally stated. | Step 4.2's grid mis-specification (§5.1's own named contingency) leaves P1/P2/P3 undefined, not merely underpowered — no amount of re-analysis of the existing grid recovers a ν* to compare. The block-vs-scale question motivating H1 is still answerable in a weaker, descriptive form directly from measured `BE`, without `cota(n)` or `c=1` (both of which Step 4.2 showed are the actually-broken part, not the raw error data). See SPEC.md, "Step 4.3 — causal decomposition (EXPLORATORY, post-4.2 pivot)". | **Post-data** |

### 8.1 — 2026-08-20: 🧠1 resolved, bound functional form only

**What changed.** §3.1 recorded `bound(n) = c · √n · u_eff` as an explicit
placeholder pending 🧠1, and committed (§3.1 item 1) that a different
resolution would be filed here as a dated amendment before the confirmatory
analysis is run. 🧠1 is now resolved, and the resolved form is

    cota(n, format, block, ν) = c · u_eff(format, block, ν) / √n

The full derivation, the measurements it is consistent with, the point at which
it breaks down (ν = 1), and its documented limitations are in SPEC.md,
"Theoretical bound definition (🧠1) — RESOLVED".

**This supersedes any earlier informal reference to a √n-growing form**,
including §3.1's own provisional `c · √n · u_eff` and §3.2's parenthetical
framing of the comparator as "the probabilistic √n·u bound". Per the
append-only rule at the head of §8, §§1–7 are **not** edited in place: the
provisional definition stays where it is and is superseded by this entry, not
overwritten by it. Both the provisional-bound and resolved-bound results are
still to be reported side by side, per §3.1 item 1.

**Timing — pre-data, verified.** This resolution was decided **before** the
Phase 3 sweep was run and **before** any `c` was fitted. Verified against the
repository at the time of writing: `results/` contains only `diagnostics/`
subdirectories (`metric_stability`, `n_scaling`, `sanity_gamma_n`,
`sanity_reproduce_2408`, `u_eff`); no `sweep_*.parquet` exists anywhere, and no
fitted constant exists in `src/` or `scripts/`. The evidence the resolution
rests on is diagnostic and exploratory by construction, and none of it is
confirmatory-sweep data.

**c is unaffected by this amendment.** §3.2's `c = 1` a-priori fixing and its
anti-circularity constraint stand exactly as written: a bound whose constant is
fitted to the same data it is tested against cannot be falsified by that data.
Only the bound's **functional form** changes here — `u_eff/√n` replacing the
previously-informal `√n·u` reference — not the constant-fitting rule. Any
ĉ fitted from data remains descriptive and exploratory only (§3.2, §5(b)) and
is never substituted into the confirmatory ratio. Note that §3.2 tags `c = 1`
as ⚑ INFERRED from the superseded √n·u framing; it is retained here as the
reference leading-order constant for the new scaling relationship, and because
the derivation is an order-of-magnitude (CLT/LLN) argument rather than a proven
tight inequality, a poorly-calibrated `c = 1` is a reportable finding rather
than grounds to re-fit.

**R6 remains OPEN and is not resolved by this amendment.** R6 — that a
ν-indexed `u_eff(format, block, ν)` partially defines away the effect H1 is
looking for, since both sides of the ratio then move with ν — is marked
**[major, NOT fixed — author's call]** in §9 and stays that way. The resolved
bound retains the ν-indexed `u_eff`, so R6 applies to it exactly as it applied
to the provisional form. This amendment closes the **functional-form** question
(🧠1) and nothing else; it makes no claim about R6, and §3.1 item 3's required
wording about what H1 does and does not test continues to apply unchanged.

### 8.2 — 2026-08-22: robust statistics methodology resolved (Step 3.4)

**What changed.** §3's provisional `u_eff` definition already named p99 as
"an indicative secondary statistic" alongside the median, but left the
handling of p99 informal and did not say whether trial count changes that
status, and §3.2 (R8) specified the median bootstrap's method and resampling
unit without addressing p99 or p90 at all. This amendment makes the policy
explicit and general, and adds p90 as a second confirmatory statistic:
**median and p90 carry all confirmatory statistical weight in this project,
including ν* localization; p99 is reported for every cell, always, but is
INDICATIVE ONLY and is never the basis of a firm claim, regardless of
whether a cell ran 1000 or 5000 trials.**

**Why -- grounded in measured coverage, not just the a priori mechanism.** A
cell's p99 is set by roughly its most extreme 1% of sample values (10 of
1000 trials, 50 of 5000), and a percentile bootstrap resamples *with
replacement from the observed sample itself*, so it can reweight those
values but can never invent one more extreme than the largest already drawn
-- a structural reason to expect p99's bootstrap CI to undercover relative to
a central statistic like the median. This was checked empirically, not
assumed: a coverage simulation (t-Student ν=2, 1000 experiments per cell, at
both `n_trials=1000` and `n_trials=5000`) found median coverage close to
nominal 95% at both trial counts (95.5%, 94.8%) and p99 coverage measurably
below both the median's and nominal at `n_trials=1000` (92.7%), narrowing but
still short at `n_trials=5000` (94.0%). The gap is real and directionally as
expected, but reported at its actual, modest size rather than a more dramatic
one the a priori mechanism alone might suggest. Full numbers, methodology and
discussion: SPEC.md, "Robust statistics methodology (Step 3.4) — RESOLVED".

**What landed in code.** `qgemm.stats.bootstrap_ci` (percentile bootstrap CI
of an arbitrary statistic, explicit `Generator`, interval reported as-is
rather than forced symmetric) and `qgemm.stats.summarize_cell` (median, p90,
p99, MAD of a cell's `BE` values, each with a `bootstrap_ci`, p99's entry
additionally tagged `p99_ci_indicative_only: True`). Tests in
`tests/test_stats.py`, including the coverage simulation above.

**Timing -- pre-data, verified.** This amendment was written, and the
statistics it is grounded in were implemented and tested, **before** any
aggregation or quantile analysis of the production sweep
(`sweep_e945b87a2395`, SPEC.md "Production run provenance") has been
performed. Verified against the repository at the time of writing: no script
under `scripts/` reads `results/sweep_e945b87a2395*` for any aggregation or
quantile purpose, and neither `bootstrap_ci` nor `summarize_cell` -- both new
in this step -- is imported anywhere outside `tests/test_stats.py`. The
sweep's own parquet file and manifest exist (the run itself already
happened, per the 2026-08-21 provenance entry), but nothing has yet computed
a median, p90, p99, or CI from its contents.

**R8 is not superseded by this amendment.** §3.2's bootstrap specification
(percentile method, B = 10 000, explicit Generator seeding, the independent
trial as the resampling unit) stands unchanged and is exactly what
`bootstrap_ci`'s default `n_resamples=10000` and its `data`-as-trials
contract implement; this amendment adds a p90/p99 handling policy on top of
it, not a replacement for it. The coverage simulation itself used
`n_resamples=1000` for computational tractability (disclosed in SPEC.md), not
because the confirmatory default changed.

### 8.3 — 2026-08-22: n_primary declared (§7 item 2 closed)

**What changed.** §4.0 designates a single `n_primary` for the ν* verdict and
left its *value* ⚠ OPEN, to be declared here before the first confirmatory
run computing ν*. That value is now **n_primary = 4096**, the largest `n` in
the frozen main grid (`configs/sweep_main.yaml`, `main_grid.n = [16, 64,
256, 1024, 4096]`). All other four `n` values remain robustness checks per
§4.0, reported in a disclosed sensitivity table alongside the primary
result, never substituted for it in the H1 verdict.

**Why 4096.** Stated by the author, on grounds independent of the ν*
pattern at any `n`: it is the contraction dimension closest to real GEMM
shapes in LLM inference/training, making it the most externally relevant
regime to report as primary; and it is the `n` at which the √n-decay
derivation's asymptotic assumptions (Step 🧠1 — cancellation over
`n/block_size` roughly-independent blocks) are most likely to hold cleanly,
since the assumption improves as the block count grows.

**Timing — POST-DATA, disclosed rather than hidden.** Unlike the two
amendments above, this one is **not** pre-data, and it would be dishonest to
label it so. Verified against the repository at the time of writing: the
confirmatory sweep (`sweep_e945b87a2395`) had already completed
(2026-08-21) and Step 4.1's n-scaling slope fits — which regress across all
five `n` values, `4096` included, for all 128 sub-configurations — were
already computed and committed (commit `41d6c79`,
`results/analysis/n_scaling_fits/`) before this declaration. So the author
was not blind to how `median(BE)` behaves at `n=4096` when choosing it.

**What this does and does not taint.** Step 4.1 reports a *slope* — how
`median(BE)` scales with `n` — not the *ratio* `median(BE)/cota(n)` at any
single `n`, and not any break/no-break outcome: no bootstrap CI of that
ratio, at `n=4096` or any other `n`, existed anywhere in this repository
before this amendment (verified: no script under `scripts/` reads
`results/sweep_e945b87a2395*` to compute a ratio against `cota`, and
`qgemm.stats.bootstrap_ci` is imported nowhere outside `tests/` and
`scripts/fit_n_scaling.py`, whose output is the slope table, not a ratio).
The specific quantity §3's break criterion is evaluated on was therefore
unseen at declaration time. What *was* seen is the closely related slope
pattern (Step 4.1: gap from -0.5 negligible at ν≥15, material at ν≤3,
uniform across all five `n` including 4096) — which is suggestive of, but
not identical to, whether the ratio's CI at n=4096 specifically clears 1.0
at any given ν. The honest position is that this declaration carries
residual risk of being informed by that adjacent pattern, is disclosed as
such rather than claimed clean, and is the author's call exactly as §7
reserves it to be — not a reviewer- or tool-driven pick made to produce a
particular ν* outcome.

**Scope.** This closes §7 item 2 only. Items 1 (the ν grid), 3 (the primary
error metric), and 4 (trial counts) remain formally unclosed by a dated
amendment, though each is de facto fixed by the frozen `sweep_main.yaml`
grid and by `be_median`'s established use throughout Step 3.3/3.4/4.1 as the
empirical-error quantity. Closing those formally is not done here.

### 8.4 — 2026-08-23: pivot to Step 4.3 as the primary vehicle for the block-vs-scale question

**What changed.** §1's H1 and §4's P1/P2/P3 are all stated in terms of ν* —
a crossing point in the bound-broken/bound-holds pattern across the ν grid.
Step 4.2 (SPEC.md, "Step 4.2 — ν* localization") found the bound broken at
every one of the 8 tested ν, in every one of the 16 (block_size x
scale_format x round_mode x rht) families, at every one of the 5 tested n —
exactly the all-censored-above contingency §5.1 names by name. There is no
crossing anywhere on this grid for any of the 16 families, so ν* is
undefined for all of them, not merely hard to localize precisely. P1/P2/P3
therefore cannot be evaluated as formulated — not "evaluated and found no
effect," but structurally undefined, since both terms of every comparison
(r(32,s), r(16,s), etc.) are pinned at the same censored value by
construction.

This amendment records a pivot: **Step 4.3's originally-planned causal
decomposition (block_size x scale_format, from the Step 1.5 2x2) becomes the
PRIMARY vehicle for the practical block-vs-scale question**, applied
directly to observed `median(BE)` (in log space) rather than to ν* or to
`cota(n)`. Concretely: `effect_block = median(log BE | block=16) -
median(log BE | block=32)` and the symmetric `effect_scale`, each with a
bootstrap CI, computed at every (ν, n, round_mode, rht) cell rather than
collapsed into a single ν* per family.

**This is EXPLORATORY, not a resolution of H1 as originally formulated.**
§1's H1 is specifically a claim about ν* — "there exists a critical ν* such
that ... ν*(block=32) > ν*(block=16)" — and P1-P3 (§4) are specifically
comparisons of ν* (or, for P3, a decomposition of such a comparison). A
direct comparison of `median(BE)` is a different, weaker claim: it can say
which format has more error and by how much at a given (ν, n), but it says
nothing about *where a bound breaks*, because it does not reference any
bound at all. §3.2's anti-circularity constraint (`c` fixed at 1, never
re-fit) and §3's break criterion are both about the *ratio* to `cota(n)`;
Step 4.3 as pivoted here uses neither — no `cota(n)`, no `c`, no bootstrap
CI of a ratio to 1.0 — only directly measured backward error. It therefore
cannot confirm, reject, or reframe-as-null H1 in the sense §1/§2/§5 define;
it is reported as a separate, descriptive finding about the practical
MXFP4-vs-NVFP4 comparison, motivated by the same NVIDIA-gap question that
motivated H1 but not a test of H1's specific ν*-crossing claim.

**Why not simply stop at Step 4.2's null result.** §5.1 already anticipated
exactly this scenario ("bound fails even in the near-Gaussian regime...a
strong result in its own right") and gives it a name (grid
mis-specification) but does not prescribe a next step beyond reporting it.
Step 4.3 was already planned (Step 1.5's 2x2 factorial exists specifically
to support a block-vs-scale decomposition) and needs no new instrumentation
— only a different target quantity (`BE` directly, not `BE`'s ratio to a
now-known-uncalibrated bound). Declaring this pivot before running Step 4.3
keeps the same anti-circularity discipline the rest of this document
enforces: the decision to reframe was made because of Step 4.2's structural
finding (no ν* exists), not because a first look at `median(BE)` differences
suggested a more favorable story.

**Timing — post-data**, same disclosure standard as 8.3: Step 4.2's result
was fully known (committed, `results/analysis/nu_star/`) before this
amendment was written. Nothing about that is hidden; the amendment exists
specifically because of what Step 4.2 found.

## 9. Adversarial Review Notes (2026-08-19)

Review conducted from the perspective of a skeptical reviewer for a
numerical-analysis workshop, against the author's draft of 2026-08-19, before
any data collection. Every issue found is listed, including those fixed above,
so the audit trail is visible. Severity: **[blocking]** would undermine the
confirmatory claim if left as-is; **[major]** leaves a live researcher degree
of freedom; **[minor]** is clarity or completeness.

**R1 [major] — the ν* bracket was stated on the wrong side of the crossing.**
§1 said ν* is the interval "between the largest ν where the bound holds and the
smallest ν where it breaks". Under H1 the bound breaks *below* ν*, so under a
clean step the largest ν where it holds is ν₈ and the smallest where it breaks
is ν₁ — the stated bracket is the entire tested range, which is not what was
meant. *Fixed*: bracket direction corrected in §1 to match H1.

**R2 [blocking] — ν* was undefined when the bound never breaks.** The draft
gave no rule for a cell where the bound holds at every tested ν, or breaks at
every tested ν. In the first case ν* does not exist in the tested range; in the
second it lies above it. Since H1 is the *comparison* ν*(32) > ν*(16), a single
censored cell made the H1/H0 comparison undefined — and, worse, made it
undefined precisely in the scientifically interesting asymmetric case where one
block size breaks and the other never does. *Fixed*: §1.1 introduces the
fragility rank r ∈ {0…8}, which is total, censoring-aware, and computed by a
rule that cannot be adjusted after seeing the data. Censored cells order
correctly against uncensored ones, so every combination of censoring yields a
defined verdict.

**R3 [blocking] — ν* presupposed monotone breakage; 8 noisy levels need not
comply.** With per-level bootstrap decisions, "breaks / holds" can interleave
along ν. The draft's definition of ν* is only coherent for a single crossing,
so an interleaved pattern would have forced a post-hoc choice of which crossing
counts — a large and invisible degree of freedom. *Fixed*: §1.1 makes r
well-defined regardless, adds an explicit down-set monotonicity check, and
prespecifies that non-monotone cells are downgraded to
inconclusive-exploratory rather than adjudicated after the fact.

**R4 [blocking] — H1 and H0 were neither exhaustive nor mutually exclusive.**
The two hypotheses left the directional-reversal outcome — block-16 more
fragile than block-32 — with nowhere to land. That outcome is not H0 (there
*is* a detectable difference) and not H1 (the ordering is inverted), and it is
the outcome that would most sharply contradict the motivating NVIDIA-gap
narrative. Left unhandled, the path of least resistance after seeing such data
is to describe it as "no support for H1" and fall back on the §5 null
reframing, which would hide a positive finding of the opposite sign. *Fixed*:
§2.1 enumerates the full outcome space; §5.1 commits to reporting a reversal as
a reversal, in the abstract.

**R5 [blocking] — the provisional bound was flagged, but its consequences were
not.** Credit where due: the draft *did* mark the `u_eff` / `c·√n·u_eff`
construction as pending 🧠1 rather than stating it as settled, which is more
than most preregistrations manage. What was missing is what follows from that:
that resolving 🧠1 changes the H1 test itself and therefore requires a
pre-analysis amendment; that the provisional bound is not a theoretical
contribution; and that §5(c)'s claim about "classical probabilistic bounds"
does not follow from it (see R13). *Fixed*: §3.1. 🧠1 itself is **not**
resolved here — that remains Step 2.3 and the author's work.

**R6 [major, NOT fixed — author's call] — a ν-indexed `u_eff` partially defines
away the effect H1 is looking for.** `u_eff(format, block, ν)` is measured
empirically *at each ν*. Under heavy tails the per-element relative
quantization error is dominated by the block absmax outlier, so `u_eff` grows
as ν falls — which means the denominator inflates in exactly the regime where
the numerator is expected to inflate. The bound therefore chases the data, and
a reviewer will ask whether a null result is a real verification or an artifact
of a self-adjusting yardstick. The construction is defensible if H1 is
understood narrowly as "does the √n growth law survive heavy tails", which is
what §3.1(3) now commits the paper to saying. An alternative worth considering
— fixing `u_eff` at its Gaussian (ν = ∞) value so the denominator is
ν-independent, and reporting the ν-indexed version as secondary — would test a
stronger and more interesting claim. **Not changed**: this is 🧠1 territory and
a genuine scientific choice, not a wording fix.

**R7 [blocking] — a fitted constant `c` would make H1 unfalsifiable.** §3
defines bound(n) = c·√n·u_eff without fixing c, while §5(b) lists "the
empirical constant c" as a contribution — implying c is estimated from data. If
c is estimated from the same data the ratio is tested on, the break criterion
can be satisfied or defeated at will, and worse, a per-cell ĉ would make ν*
non-comparable across cells, voiding P1. *Fixed*: §3.2 fixes c = 1 for all
confirmatory analysis, marked ⚑ INFERRED from the project's own framing of the
comparator as "the probabilistic √n·u bound"; any fitted ĉ is exploratory and
descriptive only. **The author must confirm c = 1.**

**R8 [major] — the bootstrap was underspecified, and its natural resampling
unit is wrong.** The draft named "95% bootstrap CI of the median" but not the
method, the number of resamples, the seeding, or — most consequentially — the
resampling unit. Resampling matrix elements would be the obvious
implementation and would be invalid: elements sharing a block share a scale
factor, so they are not exchangeable, and element-level intervals would be too
narrow, inflating the break rate and hence ν*. *Fixed*: §3.2 specifies
percentile method, B = 10 000, explicit Generator seeding, and the independent
trial as the resampling unit. Also fixed: `u_eff` must come from an independent
calibration sample, otherwise the ratio is partly self-normalizing.

**R9 [blocking] — the break criterion is indexed by n, but ν* is not.** §3
decides breakage at (format, ν, n); §1 compares a single ν* per block size. No
aggregation rule over n was given, so "does the bound break at this ν" had as
many answers as there are n values in the grid, and the selection among them
would necessarily happen after seeing results. *Fixed*: §4.0 designates a
single `n_primary` for the verdict and demotes all other n to a disclosed
sensitivity table. The *value* of n_primary is left open (§7.2) — choosing it
is the author's scientific decision, but it must be chosen before data.

**R10 [major] — "primary (max 3)" mislabels the evidentiary role of P2 and
P3.** H1 concerns block size only, so P2 (scale format) does not bear on H1 at
all — it is a design-validity check that licenses the causal reading of P1 —
and P3 is explicitly defined as a decomposition of P1 + P2 + interaction, i.e.
a derived quantity, not an independent test. Presenting all three as
co-equal "primary" comparisons invites the paper to draw H1 support from
whichever of the three cooperates, and quietly makes the count four (P1, P2,
interaction, P3) rather than three. *Fixed*: §4.1 designates P1 as the sole
confirmatory test of H1 and relabels P2 and P3 without changing any statistic
or threshold. Additionally, P1 as written ("scale format held constant") does
not say *at which* value — with two scale formats there are two P1 contrasts,
and picking the agreeable one is a live degree of freedom. *Fixed* by the
conjunctive rule in §2.1: both arms must agree.

**R11 [major] — the no-correction stance has a real argument, but it is scoped
narrower than the multiplicity it faces, and the residual bias is
asymmetric.** The draft's justification is genuine, not an assertion:
curve-crossing detection along a single ordered axis is not a family of
independent tests, and that is a sound argument for the 8 ν levels within one
cell. It does not, as written, address multiplicity across the 4 cells, across
P1–P3, or across n. More importantly, the standard "no correction just widens
the uncertainty" defence does not hold here: because r(c) is the index of the
*largest* broken ν, one false-positive break at a high ν shifts ν* upward
directly. With 8 ν × 4 cells × |n| intervals at 95%, at least one spurious
break is likely. *Partially fixed*: §4.2 extends the author's own rationale to
each remaining axis using the design's own structure (all cells reported; only
P1 moves the verdict; only n_primary counts) and states the residual bias
plainly instead of concealing it. **Not fixed, author's call:** the cheap
remedy is to require a break to be confirmed at two adjacent ν levels before it
raises r, or to lower the per-interval level. Both change the operating
characteristics of the primary test, so the reviewer did not impose either.

**R12 [major] — §5 covered one of the four possible outcomes.** The null
contingency was well-designed and specific — it is a genuine strength of the
draft that a null result already has a paper — but reversal, all-censored-above
and interaction-dominated outcomes had no prewritten home, which is where
post-hoc reframing usually enters. *Fixed*: §5.1.

**R13 [major] — §5(c) claims more than the provisional bound can deliver.**
"Evidence that classical probabilistic bounds hold even in the adversarial
regime" would be a strong verification result, but the quantity actually tested
substitutes an empirically measured `u_eff` for the format's nominal unit
roundoff. Holding against a calibrated yardstick is not the same claim as
holding against the classical bound. *Fixed*: §5.1 constrains the wording that
contribution (c) may use. Related to R6.

**R14 [major] — the grid, sample size and stopping rule were absent, and ν was
never defined.** The earlier skeleton carried "Sweep grid / sample size" and
"Stopping rule" sections; the draft dropped them. Without a frozen grid the
document does not bind — ν levels or n values could be appended later with no
deviation logged — and without a stopping rule, optional stopping is
unconstrained. The draft also never states that ν is the t-Student degrees of
freedom or that smaller ν means heavier tails, which the direction of H1
depends on. *Fixed*: §6 restores all three and ties the frozen grid to the
repo's existing `sweep_{sha256(config)[:12]}` convention, which makes the grid
verifiable from the artifacts rather than merely asserted; notation section
added. Values left open in §7.

**R15 [minor] — the specified estimator does not yet exist in the codebase.**
`qgemm.stats` currently provides only `mean_ci95`, a normal-approximation CI of
the *mean* — the estimator §3 explicitly rejects. `qgemm.distributions`
provides Gaussian and uniform samplers but no t-Student sampler, so the ν axis
is not yet implementable either. Not a defect in the preregistration, but both
must be implemented and tested before data collection, and the temptation to
fall back on `mean_ci95` because it is what exists must be resisted. *Noted in
§3.2.*

**R16 [minor] — a claimed contribution with no analysis plan.** §5(d) lists
"the scale-invariance analysis" among the null-result contributions, but no
such analysis is defined anywhere in the document. *Flagged*: §7.6 — specify
or demote.

**R17 [blocking] — "empirical_error" is never operationally defined.** The
numerator of the central ratio has no definition. The project studies backward
error; `qgemm.metrics` exposes forward-error metrics; `qgemm.bounds.gamma_n` is
a backward-error growth factor. Which quantity is measured, and whether it is
commensurable with `c·√n·u_eff`, decides what H1 even means — and leaving it
open until after the sweeps would let the metric be chosen to fit the result.
*Flagged, not fixed*: §7.3. Naming the primary metric is the author's
scientific decision; the reviewer will not choose it. This item alone blocks
labelling any run confirmatory.

### Reviewer's summary judgement

The draft's strengths are real and worth stating: the break criterion is
conservative and was fixed before data (§3), the null-result contingency is
concrete rather than aspirational (§5), the no-correction stance came with an
actual argument rather than an assertion (§4), and the provisional status of
the bound was already flagged rather than being passed off as settled. The
serious problems were all of one kind — **quantities that the document uses as
if they were defined but never defines**: ν* under censoring (R2), ν* under
non-monotonicity (R3), the constant c (R7), the aggregation over n (R9), and
the numerator itself (R17). Four of those are now closed by prespecified rules;
R17 and the §7 items remain with the author, and no run may be called
confirmatory until they are closed by a dated amendment.
