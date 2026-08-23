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

## 8. arXiv:2408.02897 reproduction target

- **Claim:** This project's `qgemm.metrics.backward_error` pipeline
  qualitatively reproduces the direction of a published finding — that FP8
  stays much more bounded than INT8 as t-Student tails get heavier — from
  the "methodologically closest prior work" to this project.
- **Bibliographic info recorded:** Rasquinha & Tabak (Google), *"A Metric
  Driven Approach to Mixed Precision Training"*, arXiv:2408.02897. (Authors,
  affiliation, title, and arXiv ID all present — the most bibliographically
  complete entry in this list.)
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

## Item requested but not found: "the three scoop-risk papers"

A full-text, case-insensitive search for "scoop" across `SPEC.md`,
`PREREGISTRATION.md`, `README.md`, `paper/README.md`, every tracked file's
git history (commit messages and diffs), and the untracked
`ultima_conversacion.txt` returned **zero matches**. There is no group of
three (or any number of) papers anywhere in this project's documentation
framed around scoop risk, competing/concurrent work, or a similar concept
that this compiler could locate. Per the task's own instruction not to
fabricate or infer beyond what is actually present, this item is reported as
**not found** rather than populated with placeholder entries. If this refers
to material that exists outside this repository (e.g. discussed verbally or
in a separate document), it will need to be supplied before a citation entry
can be compiled for it.

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
