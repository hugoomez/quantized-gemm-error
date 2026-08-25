# Citation worklist

Compiled by scanning `SPEC.md`, `PREREGISTRATION.md`, `README.md`, `paper/README.md`,
`src/qgemm/bounds.py` (the one source module carrying an inline citation),
tracked git history, and the untracked `ultima_conversacion.txt`, for every
claim currently attributed to an external source anywhere in this project's
own documentation.

**Nothing below has been verified, fetched, or confirmed.** This is a
compilation pass only, per the task. "Bibliographic info recorded" reports
exactly what this repo's own documents currently state — no more, no less —
and is explicitly flagged wherever that is incomplete or absent, so that gaps
are visible before the verification pass rather than silently filled in here.

---

## 1. Classical deterministic worst-case error bound (`gamma_n`)

- **Claim:** `gamma_n(n, u) = n*u / (1 - n*u)` is the standard closed-form
  worst-case bound on relative error growth from `n` sequential
  floating-point roundoffs at unit roundoff `u` (sequential-summation model).
  Used both as the implemented function and as the pass/fail criterion in the
  Gamma_n sanity check (zero-violation test against this bound).
- **Bibliographic info recorded:** N. J. Higham, *"Accuracy and Stability of
  Numerical Algorithms"*, 2nd ed., SIAM, 2002. (Author, title, edition,
  publisher, year all present. No DOI/ISBN recorded.)
- **Used in:**
  - `src/qgemm/bounds.py`, module docstring and `gamma_n()` docstring
    (explicit citation line: "Reference for the classical part: N. J. Higham...")
  - `SPEC.md`, "Theoretical bounds (`qgemm.bounds`)"
  - `SPEC.md`, "Gamma_n sanity check — MEASURED, pending human review"

## 2. Probabilistic √n·u (and √(n·log n)·u) high-probability bound

- **Claim:** `sqrt(n)*u` (and, for a refined variant, `sqrt(n log n)*u`)
  is established as a **high-probability upper bound** (not a statement about
  typical/expected growth) on sequential-summation error under random
  rounding — cited to explain why an empirical log-log slope flatter than 0.5
  does not contradict the theorem.
- **Bibliographic info recorded:** "Higham & Mary (2019)" and
  "Connolly-Higham-Mary (2021)" — surnames and years only. **No paper title,
  venue, arXiv ID, or DOI recorded anywhere in this project's documents.**
- **Used in:** `SPEC.md`, "The log-log slope, and why it does not need
  re-litigating here" → "On how to read 'inconsistent with sqrt(n)' above"
  (Gamma_n sanity check section)
- **⚠ CORRECTION (2026-08-25) — the claim as recorded above is
  misattributed.** The "Claim" bullet is left standing as the record of what
  this project's documents asserted at Step 6.1 time; it is not what the
  sources say. Claim-level checking (triggered while drafting the paper's
  Background section) established the correct pairing:
  **Higham & Mary (2019) = `sqrt(n log n)*u`**, a general-case
  high-probability bound under a mean-independence model of the rounding
  errors; **Connolly-Higham-Mary (2021) = `sqrt(n)*u`**, no log factor,
  unconditional, specific to **stochastic rounding**. The two are different
  bounds for different rounding regimes — `sqrt(n log n)` is *not* "a refined
  variant" of `sqrt(n)*u`; if anything the Connolly et al. bound is the
  tighter and later result. `SPEC.md` has been corrected (see its
  "Misattribution of the two probabilistic bounds" entry under Step 6.1
  "Corrections applied"). Note that `paper/refs.bib`'s comments for both
  entries never mention a log factor, and describe `connolly2021stochastic`
  as "specifically the stochastic-rounding refinement" — consistent with the
  corrected attribution, so no `refs.bib` change is required.

## 3. MX / OCP microscaling specification

- **Claim:** MXFP4 (`block_size=32`, `scale_format=e8m0`, no global scale) is
  "a real format (OCP MX)" — i.e. this project's MXFP4 preset implements a
  published Open Compute Project microscaling specification, as opposed to
  the project's own experimental block-size/scale-format controls, which
  correspond to "no hardware, no specification and no vendor format."
- **Bibliographic info recorded:** **None.** The spec is referred to only by
  the acronym "OCP MX" / "the OCP MX specification" — no document title,
  authors/organization byline, version number, publication date, or URL
  recorded anywhere in `SPEC.md`, `PREREGISTRATION.md`, or `README.md`.
- **Used in:**
  - `SPEC.md`, "Block structure (`qgemm.blocks`)" table (🧠-adjacent row:
    `block=32, e8m0 → MXFP4 — a real format (OCP MX)`)
  - `SPEC.md`, "Relationship to `microxcaling`" ("Microsoft's `microxcaling`
    (and the OCP MX specification it implements) computes the shared scale
    as...")
  - `SPEC.md`, "Why the controls exist" / "The controls are not proposals"
    (repeated "no hardware, no specification, no vendor" framing, lines ~213,
    1433, 1717)

## 4. Microsoft `microxcaling` reference implementation

- **Claim:** This project's MXFP4 block-scaling implementation agrees
  bit-for-bit with `microxcaling`'s on every block once the scale-exponent
  rounding rule (ceil vs. floor) is normalized; used as an external
  correctness check on the round-up-vs-round-down scale design decision.
- **Bibliographic info recorded:** Software repository, not a paper —
  `github.com/microsoft/microxcaling`, version **1.1.0**, run against torch
  2.13.0+cpu on 2026-08-20 in a throwaway venv. (Repo URL, version, and run
  date all present; no associated paper/authors cited for the package
  itself, only the OCP MX spec it implements — see item 3.)
- **Used in:** `SPEC.md`, "MXFP4 block scaling" → "Relationship to
  `microxcaling`" and "Verified against `microxcaling` 1.1.0" (lines ~314–359)

## 5. NVIDIA NVFP4 "~36% more tokens" claim — **VENDOR CLAIM**

- **Claim:** "NVIDIA's reported 36%-more-tokens gap" is named as the
  practical motivation tied to comparison P3 (MXFP4 preset vs. NVFP4
  preset), and "the NVIDIA-gap narrative" is referenced repeatedly as the
  motivating story this study's H1/H0 design is built to test rather than
  presuppose.
- **Bibliographic info recorded:** **None whatsoever.** No blog post,
  whitepaper, press release, product page, or URL is recorded anywhere in
  this project's documents — only the bare phrase "NVIDIA's reported
  36%-more-tokens gap."
- **⚠ Flag — this project does not have a single named "vendor claim policy"
  document, but its existing text is consistently skeptical of this figure
  as an independent fact:** `PREREGISTRATION.md` explicitly names "the
  NVIDIA-gap narrative" as something the study must be able to contradict
  without that being mis-described after the fact (§2.1's directional-reversal
  row, §9 R4), and `SPEC.md` repeats "no hardware, no specification and no
  vendor format... nothing here benchmarks or endorses a format" for every
  configuration that is not one of the two real presets. The final paper
  should attribute the 36% figure explicitly to NVIDIA as its source (not
  cite it as independently established), consistent with that existing
  framing — but there is no prewritten "vendor claim policy" section to
  quote verbatim, so this is this compiler's inference from the surrounding
  text, not a located policy statement.
- **Used in:**
  - `PREREGISTRATION.md`, §4, comparison P3 definition (line ~186–188)
  - `PREREGISTRATION.md`, §2.1 ("the NVIDIA-gap narrative motivating this
    study", line ~116)
  - `PREREGISTRATION.md`, §8.4 amendment (line ~548)
  - `PREREGISTRATION.md`, §9 reviewer notes (R4 area, line ~610)

## 6. `torchao` NVFP4 reference implementation

- **Claim:** A restricted golden-reference comparison against `torchao`'s
  NVFP4 (float32-exact tensor, scales landing exactly on E4M3 grid points)
  is used as a sanity check on this project's NVFP4 implementation — explicitly
  described as "a sanity check, not an authority," unlike the `microxcaling`
  comparison, because `torchao`'s NVFP4 lives under `torchao.prototype` and is
  not itself a reference implementation of a published spec.
- **Bibliographic info recorded:** Package name only (`torchao`,
  `torchao.prototype`). **No version pin, no repository URL, no run date
  recorded** (contrast item 4, which has all three for `microxcaling`).
- **Used in:** `SPEC.md`, "NVFP4 block scaling" → "Relationship to
  `torchao`" (lines ~473–480)

## 7. `ml_dtypes` reference casts

- **Claim:** `ml_dtypes` is the bit-exact reference implementation this
  project's E4M3/E5M2/E8M0/E2M1 quantizers and the bf16 accumulation route
  are tested for conformance against (with one documented, deliberate
  deviation: this project correctly-rounds float64→float8 directly, while
  `ml_dtypes` double-rounds via float32).
- **Bibliographic info recorded:** Package name only (`ml_dtypes`). No
  version, no repository URL, no publisher (Google) named in the docs.
- **Used in:** `SPEC.md`, throughout the format sections — "The float8
  reference is `ml_dtypes`" (line 41) through the "Equivalence to
  `ml_dtypes`" subsection (lines ~92–120), and the bf16 accumulation section
  (line ~670, `ml_dtypes.bfloat16`).

## 8. arXiv:2408.02897 — dual role: reproduction target AND scoop-risk differentiator

- **Claim (reproduction role):** This project's `qgemm.metrics.backward_error`
  pipeline qualitatively reproduces the direction of a published finding —
  that FP8 stays much more bounded than INT8 as t-Student tails get heavier —
  from the "methodologically closest prior work" to this project.
- **Claim (scoop-risk role, added 2026-08-24):** Despite being the closest
  methodological antecedent — same backward-error metric
  (`BE = |L·R − Q(L,R)| / (|L|·|R|)`, confirmed verbatim in the source's
  Sec. IV-A during Step 6.1 verification) and the same t-Student
  distributional family — this paper studies only per-tensor/per-vector
  INT8 and FP8, never block-scaled microscaling formats (MXFP4/NVFP4), and
  contains no analogue of this project's H1 (the block-size fragility
  crossing ν*, PREREGISTRATION.md §1). It is therefore the paper closest in
  spirit to this one that does *not* already answer this project's actual
  question, which is exactly what a scoop-risk citation needs to
  demonstrate — not merely "a nearby paper exists," but "the nearest paper
  doesn't already do this." This resolves item 8's status as one of the
  originally-requested "three scoop-risk papers" (see the section below,
  formerly "not found").
- **Bibliographic info recorded:** Rasquinha & Tabak (Google), *"A Metric
  Driven Approach to Mixed Precision Training"*, ASSYST workshop @ ISCA
  2023, arXiv:2408.02897. (Authors, affiliation, venue, title, and arXiv ID
  all present and independently verified — see `paper/refs.bib`'s
  `rasquinha2023metric` entry.)
- **Used in:** `SPEC.md`, "arXiv:2408.02897 reproduction — MEASURED,
  qualitative match" (line 1222 onward); also referenced in `SPEC.md`'s
  `int8_quantize` description as "the classical baseline used by the
  arXiv:2408.02897 reproduction below" (line ~38).

## 9. "Massive activations" / outlier-feature literature — Sun et al.

- **Claim:** A small, fixed set of channel indices dominating the
  extreme-value tail of real GPT-2 hidden-state activations "matches the
  'massive activations' / 'outlier feature' pattern reported in the
  literature on transformer internals."
- **Bibliographic info recorded:** "Sun et al. 2024, 'Massive Activations in
  Large Language Models'" — first-author surname, year, and title present.
  **No full author list, venue, or arXiv ID/DOI recorded.**
- **Used in:** `SPEC.md`, Step 4.4 addendum, "Verdict: CONCENTRATED..."
  (line ~3477–3481)

## 10. Persistent per-channel outlier literature — Liu et al.

- **Claim:** Cited alongside item 9 as independently documenting "persistent
  per-channel outliers in transformer hidden states."
- **Bibliographic info recorded:** **"Liu et al." only — no year, no title,
  no venue, nothing else.** The most bibliographically incomplete entry in
  this worklist.
- **Used in:** `SPEC.md`, same sentence as item 9 (line ~3481–3482)

## 11. GPT-2 (small) language model

- **Claim:** Real transformer hidden-state activations are extracted from
  GPT-2 small (three depths: `h0`, `h6`, `h11`) as the basis of the Step 4.4
  real-activation realism check, contrasted against the synthetic t-Student
  operand model used everywhere else in the study.
- **Bibliographic info recorded:** **None.** Referred to only via its
  HuggingFace `transformers` identifier `gpt2`. The standard citation
  (Radford et al. 2019, "Language Models are Unsupervised Multitask
  Learners") is not recorded anywhere in this project's documents.
- **Used in:** `SPEC.md`, "Step 4.4 — real-activation realism check
  (EXPLORATORY)" (line ~3283 onward, model/layers described ~3302–3321)

## 12. WikiText-2 dataset

- **Claim:** The WikiText-2 raw test split supplies the input text tokenized
  and fed through GPT-2 to produce the real activations used in item 11's
  check.
- **Bibliographic info recorded:** **None.** Referred to only via the
  HuggingFace dataset id `Salesforce/wikitext`, config `wikitext-2-raw-v1`,
  `split="test"` (plus an implementation note that the un-namespaced
  `"wikitext"` loading script is broken under `datasets>=5`, which is a
  code-compatibility note, not a citation). The standard citation (Merity et
  al. 2016, "Pointer Sentinel Mixture Models") is not recorded anywhere.
- **Used in:** `SPEC.md`, Step 4.4, dataset description (line ~3302–3306)

---

## Item, formerly "not found": "the three scoop-risk papers" — RESOLVED 2026-08-24

**Original finding (superseded below, kept for the record):** a full-text,
case-insensitive search for "scoop" across `SPEC.md`, `PREREGISTRATION.md`,
`README.md`, `paper/README.md`, every tracked file's git history, and the
untracked `ultima_conversacion.txt` returned zero matches, and this item was
reported as not found rather than populated with placeholder entries.

**Resolution.** All three scoop-risk papers are now present and verified —
no fourth paper was ever needed, and none was fabricated to reach the count
of three:

1. **`rasquinha2023metric`** (item 8, above) — the closest methodological
   antecedent (same BE metric, same t-Student family), differentiated by
   scope: no block-scaled microscaling, no ν* fragility hypothesis. Playing
   this dual role (reproduction target *and* scoop-risk differentiator) is
   not a stretch reached only by relabeling — it is arguably the single
   most load-bearing scoop-risk citation of the three, since it is the
   paper an outside reader is most likely to ask "doesn't this already do
   what you're doing?" about, and the honest answer ("no, and here's
   specifically why") is what a scoop-risk citation exists to provide.
2. **`fasoli2026finer`** (item 3 of the final refs.bib list) — differentiated
   by mechanism: driven by narrow tensor distributions, not the heavy-tailed
   operand regime this project studies.
3. **`egiazarian2026bridging`** (item 4 of the final refs.bib list) —
   differentiated by metric and operand model: MSE against a Laplace
   assumption (Definitions 1 and 3, confirmed verbatim by direct PDF fetch
   during Step 6.1), not backward error against t-Student.

All three are in `paper/refs.bib` with dated, source-checked verification
comments. No further search is needed for this item.

## Related finding, not a citation to add: the "Appendix A" phantom citation

`SPEC.md` (lines ~1970–1981) contains a documented incident directly
relevant to this exercise: a prior task described a grid cut as following
"this project's own pre-declared cut-order (Appendix A of the roadmap, item
4)," and a full repository and git-history search at the time found **no
such roadmap document, no Appendix A, and no pre-declared cut-order list
anywhere** — only a previous session's commit message that coined the phrase
"cut-line policy" without grounding it in any such document. This is flagged
here not as a citation to add, but as an existing, in-repo precedent for how
this project handles an apparent citation that does not actually resolve to
a real source: name the gap explicitly rather than inventing the document it
implies. The same discipline applies to item "three scoop-risk papers" above.
