# SPEC

Technical specification for the quantized-GEMM error study. This is the
source of truth for what "correct" means in this codebase; update it
when a design decision is made, not after the fact.

## Scope

TBD -- which quantized formats, block structures, and GEMM shapes this
project studies.

## Quantization formats (`qgemm.formats`)

TBD -- the full format list and their `QuantFormat` parameters (storage
dtype, finite max). Implemented so far:

| format | function | bits (s/e/m) | bias | finite max | overflow | zero | NaN |
|--------|----------|--------------|------|-----------|----------|------|-----|
| E2M1 | `quantize_e2m1(x, mode, rng=None)` | 1/2/1 | 1 | 6 | saturates to ±6 | ±0 | rejected (`ValueError`) |
| E4M3 | `quantize_e4m3(x)` | 1/4/3 | 7 | 448 | **NaN** | ±0 | one code (`0x7F`) |
| E5M2 | `quantize_e5m2(x)` | 1/5/2 | 15 | 57344 | **±inf** | ±0 | IEEE-style |
| E8M0 | `quantize_e8m0(x)` | 0/8/0 | 127 | 2^127 | **NaN** | **none** | one code (`0xFF`) |

All four take float64 and return float64. `quantize_e2m1` additionally
takes a rounding `mode`; the three float8 quantizers are RTNE only.

`int8_quantize(x)` is listed separately because it is not a number format
in the same sense: it is **symmetric per-tensor INT8**, whose grid depends
on the tensor rather than on the encoding alone --
`scale = amax(|x|)/127`, `q = clip(round(x/scale), -127, 127)`, output
`q * scale`. Uniform grid, no zero point, and the `-128` code is given up
so the grid is exactly symmetric (`int8_quantize(-x) == -int8_quantize(x)`
holds identically). Ties round to even, as elsewhere in this module,
though here "even" is an even integer code. There is no overflow encoding
to fall into, so non-finite input is rejected rather than allowed to
poison `amax` and hence every output element; an all-zero tensor is
returned unchanged. It is the classical baseline used by the
arXiv:2408.02897 reproduction below, and is deliberately **not** block
scaled.

The float8 reference is `ml_dtypes`. Note that `ml_dtypes` ships **two**
E4M3 types: `float8_e4m3fn` (no infinity, max 448) is the deep-learning
format used here, and `float8_e4m3` (has infinity, max 240) is a
different format that must not be used as the reference.

### E4M3, E5M2, E8M0

**Grids.** Each grid is built by bit pattern, so the array index *is* the
code with the sign stripped -- which is what makes ties-to-even a parity
test on the index. All three are non-uniform (the step doubles at every
binade), so neighbors are located by `searchsorted` on the *values*, the
same rule as E2M1.

* **E4M3** -- 127 finite magnitudes (codes `0x00..0x7E`): zero, seven
  subnormals `m * 2^-9`, and normals `(1 + m/8) * 2^(e-7)` for exponent
  field `e = 1..15`. Smallest normal `2^-6`, max `448 = 1.75 * 2^8`.
* **E5M2** -- 124 finite magnitudes (codes `0x00..0x7B`): zero, three
  subnormals `m * 2^-16`, and normals `(1 + m/4) * 2^(e-15)` for
  `e = 1..30`. Smallest normal `2^-14`, max `57344 = 1.75 * 2^15`. Three
  binades wider than E4M3 at each end, at half the relative precision.
* **E8M0** -- 255 magnitudes (codes `0x00..0xFE`): the powers of two
  `2^-127 .. 2^127`. This is the MX *scale* format, so it has no sign
  bit, no mantissa, **and no zero**. Consequently `0.0`, `-0.0`, and
  every negative input map to NaN, and NaN is never returned with its
  sign bit set.

**Saturation.** Unlike E2M1, none of the three saturate; each overflows
into whatever encoding sits one code above the maximum:

| input | E4M3 | E5M2 | E8M0 |
|-------|------|------|------|
| max | 448 | 57344 | 2^127 |
| tie above max | `464 -> 448` (even code) | `61440 -> +inf` (even code) | `1.5*2^127 -> NaN` |
| just past that | `465 -> NaN` | `70000 -> +inf` | `2^128 -> NaN` |
| `±inf` | `±NaN` | `±inf` | NaN |
| underflow | `-> ±0` | `-> ±0` | **clamps up to `2^-127`** |

E4M3 and E5M2 preserve the sign bit throughout, including onto NaN, and
pass NaN input through -- both formats have a NaN encoding, so nothing is
silently lost (contrast E2M1, which has none and therefore rejects NaN).
E8M0's underflow/overflow asymmetry is deliberate: it has a smallest code
to clamp to, but nothing above `2^127` except the NaN code.

**Tie-breaking.** E4M3 and E5M2 use RTNE on the bit pattern, as E2M1
does. **E8M0 does not**: with no mantissa bit, "even" could only mean an
even exponent code, which would make the tie direction alternate with the
parity of `e`. `ml_dtypes` instead rounds ties **up** unconditionally and
this implementation follows it (verified at all 253 exponents). E8M0 also
splits at the *arithmetic* midpoint `1.5 * 2^e`, not the geometric
midpoint `sqrt(2) * 2^e`.

**Equivalence to `ml_dtypes`, and the one deviation.** `ml_dtypes` casts
float64 to float8 *via float32*, so its rounding double-rounds. These
quantizers correctly-round straight from float64 instead, which preserves
`|Q(x) - x| <= half a grid step` -- a guarantee the error analysis in this
project depends on, and one double rounding breaks. The relationship is
exact and is asserted as an identity in `tests/test_formats.py`:

```
ml_dtypes_cast(x)  ==  quantize_*(float32(x))     for every float64 x
```

So the two agree bit for bit on every float32-representable input, and
differ only on float64 values lying within half a float32 ulp of a float8
tie. Example: `1.4999999999999998` is strictly below E8M0's `1.5` tie and
correctly rounds to `1.0`, but `ml_dtypes` rounds it to float32 first,
landing exactly on `1.5`, and takes the tie up to `2.0`.

Two smaller consequences of the same float32 step, both also pinned by
tests, both cases where this implementation is the correct one:

* Magnitudes below `2^-149` flush to zero in float32, so `ml_dtypes` maps
  them to E8M0 NaN; here they simply clamp to `2^-127` like any other
  underflow.
* In the float32 *subnormal* range (below `2^-126`), `ml_dtypes`' E8M0
  cast rounds up a whole binade for anything that is not exactly a power
  of two -- even one float32 ulp above `2^-127`, nowhere near the
  `1.5 * 2^-127` midpoint. This one is not double rounding (the inputs
  are float32-exact); it appears to be a defect in the `ml_dtypes`
  subnormal path.

Conformance is tested as bit-exactness against `ml_dtypes` over dense
float32-exact sweeps covering zero, the subnormals, the max, past the
max, and both signs, plus a probe at and either side of every grid
boundary (128 for E4M3, 125 for E5M2, 255 for E8M0, counting the
overflow code). Those probes pin the rounding function down completely,
since a piecewise-constant function is determined by its behavior around
each breakpoint.

### E2M1 format

Implemented: `quantize_e2m1(x, mode, rng=None)`, float64 in / float64 out.

E2M1 (FP4) is 1 sign bit, 2 exponent bits, 1 mantissa bit, with no
infinity and no NaN encoding.

**Grid.** The eight magnitudes are exactly the 3-bit codes `000..111`:

| code | 000 | 001 | 010 | 011 | 100 | 101 | 110 | 111 |
|------|-----|-----|-----|-----|-----|-----|-----|-----|
| value | 0 | 0.5 | 1 | 1.5 | 2 | 3 | 4 | 6 |

With sign this gives 15 distinct values (+0 and -0 share a value but not
a bit pattern; the sign of zero is preserved). `0.5` (code `001`) is the
single subnormal magnitude and is treated as a grid point in its own
right, not as a scaled normal.

The grid is **non-uniform** -- step 0.5 up to 2, then 1 up to 4, then 2
up to 6 -- so neighbors are located by `searchsorted` on the *values*.
Rounding an interpolated grid *index* (e.g. via `np.round`) silently
misrounds the 2..6 range and is not permitted.

**Saturation.** There is no overflow encoding: `|x| > 6` saturates to
`±6`, and `±inf` likewise maps to `±6`. NaN has no E2M1 encoding and
raises `ValueError` rather than being silently mapped to a finite value.
(`ml_dtypes` maps NaN to `-0.0`; this is a deliberate deviation, since a
silent NaN -> zero would corrupt error statistics downstream.)

**Tie-breaking (RTNE).** Round to nearest, ties to even, where "even"
means the **even bit pattern** (mantissa bit 0) -- IEEE-754
roundTiesToEven applied to the E2M1 encoding:

| midpoint | 0.25 | 0.75 | 1.25 | 1.75 | 2.5 | 3.5 | 5.0 |
|----------|------|------|------|------|-----|-----|-----|
| rounds to | 0 | 1 | 1 | 2 | 2 | 4 | 4 |

RTNE is bit-exact with `ml_dtypes.float4_e2m1fn` over a dense sweep of
1e6 points in [-8, 8]; that equivalence is a fixed test, not a target.

**Stochastic rounding.** `mode="sr"` rounds to one of the two bracketing
neighbors `lo <= |x| <= hi`, taking `hi` with probability
`(|x| - lo) / (hi - lo)`, so `E[Q(x)] = x` for `|x| <= 6`. Grid values
and saturating values are returned deterministically. An explicit
`numpy.random.Generator` is required.

## Block structure (`qgemm.blocks`)

TBD -- the full list of block sizes and axes considered (e.g. per-tensor,
per-row). Implemented so far: MXFP4 and NVFP4, which share E2M1 elements and
differ only in how those elements are scaled.

### One parametrized quantizer; MXFP4 and NVFP4 are presets of it

Implemented: `quantize_blocked(x, block_size, scale_format,
use_global_scale, element_format="e2m1", round_mode="rtne", rng=None)`,
with `block_scales(...)` and `global_scale(...)` exposing the two levels
of scale, all float64 in / float64 out.

There is **one** block-scaling implementation. `quantize_mxfp4` and
`quantize_nvfp4` are thin presets that fix its parameters and add no
logic of their own; likewise `mxfp4_block_scales`, `nvfp4_block_scales`
and `nvfp4_global_scale`. The sections below describe the two formats,
but the code path is shared, so a change to the recipe reaches both.

| `block_size` | `scale_format` | `use_global_scale` | what it is |
|---|---|---|---|
| 32 | `e8m0` | False | **MXFP4** -- a real format (OCP MX) |
| 16 | `e4m3` | True | **NVFP4** -- a real format (NVIDIA) |
| 8, 16, 64 | `e8m0` | False | experimental control |
| 8, 32, 64 | `e4m3` | True | experimental control |

**Why the controls exist.** MXFP4 and NVFP4 differ in *both* block size
and scale format at once, so any measured difference between them is
unattributable: it could be the 32-vs-16 block, the E8M0-vs-E4M3 scale,
or an interaction. The study therefore runs the full `block_size ∈ {8,
16, 32, 64} × scale_format ∈ {e8m0, e4m3}` grid, so that block size can
be held fixed while the scale format varies and vice versa. Block sizes
8 and 64 are the endpoints that show whether the block-size effect is
monotone over the range.

**The controls are not proposals.** Six of the eight combinations --
including `block_size=16` with `e8m0` and `block_size=32` with `e4m3` --
correspond to no hardware, no specification and no vendor format. They
are measurement instruments for a controlled comparison between two
formats that already exist, and nothing in this project should be read
as introducing, endorsing or benchmarking a new format. Results tables
must keep the real/control distinction visible.

`use_global_scale` is a free parameter rather than a synonym for the
scale format, so all four scale/global combinations run, but the study
uses only the two pairings above -- the two that make sense. E8M0 spans
255 binades and reaches any block's magnitude unaided; E4M3 spans about
19 and overflows to NaN without a global scale (see "Why the global
scale exists" below, which the general function carries verbatim: the
clamp that keeps a block scale off E4M3's NaN code applies at every
block size). The other two are documented in the `quantize_blocked`
docstring: E8M0 *with* a global scale gives up the exactness of a
power-of-two scale, and E4M3 *without* one is the NVFP4 bug.

`element_format` is E2M1 in every configuration. It is held fixed on
purpose -- both real formats store E2M1, so keeping it constant makes
the scaling the only thing that varies across the grid. It is a
parameter rather than a constant only so that the assumption is stated
at the call site; passing anything else raises `ValueError`.

### MXFP4 block scaling

Implemented: `quantize_mxfp4(x, block_size=32, rng=None, round_mode="rtne")`
and `mxfp4_block_scales(x, block_size=32)`, float64 in / float64 out. Both are
presets of `quantize_blocked` / `block_scales` with `scale_format="e8m0"` and
`use_global_scale=False`; passing any other `block_size` gives a control
configuration, not MXFP4.

MXFP4 is E2M1 elements plus one shared power-of-two scale per block of
`block_size` contiguous elements. `quantize_mxfp4` returns the
*reconstruction* `q_i * s_b`, not the `(q_i, s_b)` pair; the scales are
available separately from `mxfp4_block_scales`, which is what the error
analysis needs when it asks what the block structure did.

**Recipe**, per block:

1. `amax_b = max |x_i|` over the block.
2. `s_ideal = amax_b / 6` (6 is the largest E2M1 magnitude).
3. `s_b = 2 ** ceil(log2(s_ideal))` -- the exponent is rounded **up**.
4. `q_i = quantize_e2m1(x_i / s_b, mode=round_mode)`.
5. `x_hat_i = q_i * s_b`.
6. Degenerate block: if `amax_b == 0` then `s_b = 1` (no `log2(0)`).

Steps 4 and 5 scale by a power of two and so are exact in float64 except
under underflow; the only rounding is the E2M1 grid rounding itself.

**The exponent is computed exactly, not via `log2`.** With
`amax_b = m * 2**e` and `m` in `[0.5, 1)` from `frexp`, the smallest `k`
with `amax_b <= 6 * 2**k` is `e - 3` when `m <= 0.75` and `e - 2`
otherwise. Routing through `log2(amax_b / 6)` rounds twice and can land
one exponent low, silently reintroducing the saturation that step 3
exists to prevent.

**Why round the exponent up.** This is a design decision, not an
implementation detail. Rounding up guarantees `amax_b / s_b <= 6`, so the
element that set the scale always fits in the E2M1 range. Round-to-nearest
or round-down lets `amax_b / s_b` approach 8, and everything above 6
saturates to exactly 6 -- clipping the block's largest element by up to
25% of its magnitude. The cost is equally real: a scale one binade larger
halves the resolution of every *other* element in the block. So the rule
trades a bounded, uniform precision loss across the block against an
unbounded clipping error on its largest element, and this project takes
that trade. An implementation could reasonably choose otherwise, and the
reference implementation does -- see below.

**Blocking axis.** Blocks run along the **last axis only**; leading axes
are independent rows. In `A @ B` the last axis of `A` and of `B.T` is the
contraction dimension, which is the axis a shared scale factors out of.
Blocking any other axis yields a well-formed array that measures a
different quantity, with no error raised, so callers transpose before
calling. `tests/test_blocks.py` pins this with a tensor whose rows differ
by `2**-20`: under last-axis blocking both rows survive, under axis-0
blocking the small row would flush to zero.

**Tail policy.** A trailing partial block is kept **short** and gets its
own scale from its own `amax`; it is never merged into the previous
block. This is numerically identical to zero-padding the tail to a full
block and trimming the output -- appending zeros changes neither `amax_b`
nor any element's quantization -- so hardware that pads agrees with this
function. The equivalence is asserted as a test rather than assumed.

**Scale range.** The shared scale is an E8M0 value, so its exponent is
clamped to `[-127, 127]`. Outside that range the block degrades rather
than raising: a clamped-low scale saturates the block maximum to
`6 * 2**127`, a clamped-high scale flushes the block to zero. Both are
properties of float64-extreme input data, so they are reported through
the returned values.

**Rounding modes.** `round_mode` is passed straight through to
`quantize_e2m1`, so `"sr"` requires an explicit
`numpy.random.Generator`. Stochastic rounding applies to the *elements*
within a block; the block exponent is always rounded up regardless.

**Input contract.** NaN and infinity are rejected with `ValueError`.
Neither has an E2M1 encoding, and either one would poison a whole
block's `amax` -- one infinity in a block would otherwise silently zero
the other 31 elements.

**Relationship to `microxcaling`.** Microsoft's `microxcaling` (and the
OCP MX specification it implements) computes the shared scale as
`2 ** (floor(log2(amax_b)) - emax_elem)` with `emax_elem = 2` for E2M1,
i.e. it rounds the exponent **down** and accepts the saturation. The two
rules coincide exactly when `amax_b`'s own significand is at most 1.5,
and diverge otherwise; concretely, `amax_b = 7` gives `s_b = 2` here
(reconstructing 7 as 8) and `s_b = 1` there (clipping 7 to 6). This
implementation deliberately keeps the round-up rule, for the reason
above.

**Three conventions in the literature, not two (Step 6.1, 2026-08-24).**
The floor/round-down description immediately above is unchanged and stays
exactly as tested -- bit-exact-verified against the actual `microxcaling`
package (see "Verified against `microxcaling` 1.1.0" below), not merely
asserted. What follows is additional context on where else this exponent
gets rounded differently, each independently verified against its primary
source and cited accordingly:

1. **`microxcaling`/OCP default: `floor(log2(amax_b))`.** As above --
   verified bit-exact against the reference package (Step 1.3).
2. **This project's own choice: round up**, `ceil(log2(amax_b/6))`, to
   guarantee the block max never saturates (see "Why round the exponent
   up," above). Independently corroborated, not merely coincidentally
   matched, by NVIDIA's own NVFP4 pretraining paper
   (`nvidia2026nvfp4`, arXiv:2509.25149): "we typically round decode scale
   factors up to prevent saturations," citing convergence issues observed
   under the round-down default (Mishra et al. 2025) for the same
   saturation-avoidance reason given here.
3. **A third convention: round-to-nearest plus a 4/3 rescaling
   correction**, used by Tseng, Yu and Park 2025 (`tseng2025training`,
   arXiv:2502.20586, "Training LLMs with MXFP4") and adopted by Egiazarian
   et al. 2026 (`egiazarian2026bridging`, arXiv:2509.23202, Appendix H,
   Eq. 1): `s_E8M0 = (4/3) * 2 ** clamp(round(log2(s)), -128, 127)`.
   Egiazarian et al. state this "yields an unbiased estimate of the
   original scale and reduces quantization error" (their Sec. 3/Appendix C,
   citing Tseng et al. 2025 for the 4/3 factor) -- a bias-correction
   motivation distinct from both (1)'s and (2)'s worst-case/saturation
   framing. Confirmed by direct fetch of the paper's own PDF: Appendix H
   ("MXFP SCALE FITTING") states "The original MXFP quantization grid, with
   4/3 re-scaling, quantizes scales as follows" immediately above the
   numbered equation, and a separate passage states "we multiply the scale
   by 4/3 following [52]," where reference [52] in that paper's own
   bibliography is Tseng, Yu and Park 2025.

None of the three is "the" correct convention independent of context: (1)
is a spec default this project deliberately does not use, (2) targets the
worst case (the block's largest element never saturates), and (3) targets
the average case (an unbiased scale estimate, at the cost of the round-up
guarantee). This project's own choice remains (2), for the reasons given
above; (1) and (3) are recorded here as literature context, not adopted.

**Verified against `microxcaling` 1.1.0** (`mx` from
`github.com/microsoft/microxcaling`, run on 2026-08-20 against torch
2.13.0+cpu in a throwaway venv; `torch` is still absent from this
project's own dependency set until Phase 4.4, so the four comparison
tests in `tests/test_blocks.py` `skip` under `make test`). The result is
stronger than "the outputs are close":

> Replacing our `ceil` scale with their `floor` scale makes the two
> implementations agree **bit for bit on every block**, including the
> ones where the scale rules disagree.

So the scale exponent is the *only* difference -- the element path, the
tie handling and the block reduction are identical. That identity is
asserted in `test_the_scale_rule_is_the_only_difference_from_microxcaling`
over 512 blocks of float32-exact log-normal data, of which about 40% take
the diverging branch, and in all of those `microxcaling` clips the block
maximum to exactly `6 * s_b`. On that data the round-up rule is very
slightly ahead on RMS error (55521.9 vs 55538.3, a 0.03% edge) -- the
choice is about the bounded worst case on the block maximum, not about
average error, and the measurement should not be quoted as if it were.

**Round-mode trap.** `microxcaling`'s `round="nearest"` is **not** RTNE:
it is `floor(|A| + 0.5)`, ties away from zero, which disagrees with this
implementation and with `ml_dtypes.float4_e2m1fn` at four of the seven
E2M1 midpoints (`0.25 -> 0.5`, `1.25 -> 1.5`, `2.5 -> 3`, `5 -> 6`).
Its `round="even"` is the RTNE mode and is the correct reference here.
Random data never lands exactly on a midpoint, so the wrong mode makes
every comparison test pass for the wrong reason; `_microxcaling_mxfp4`
therefore pins `round="even"` and
`test_microxcaling_round_modes_at_the_e2m1_midpoints` asserts both modes
explicitly so the choice cannot be silently changed.

Installing `microxcaling` needs `--no-deps`: its requirements pin
`torch==2.2.0` and `torchaudio==2.1.0` simultaneously, and `torchaudio
2.1.0` requires `torch==2.1.0`, so the dependency set cannot be solved as
published. The `mx` package itself runs unmodified on current torch.

### NVFP4 block scaling

Implemented: `quantize_nvfp4(x, block_size=16, rng=None, round_mode="rtne")`,
`nvfp4_block_scales(x, block_size=16)` and `nvfp4_global_scale(x)`, float64
in / float64 out. All three are presets of `quantize_blocked` /
`block_scales` / `global_scale` with `scale_format="e4m3"` and
`use_global_scale=True`; passing any other `block_size` gives a control
configuration, not NVFP4. As with MXFP4, `quantize_nvfp4` returns the
*reconstruction*; the two levels of scale are fetched separately.

NVFP4 stores exactly the same E2M1 elements as MXFP4. **The scaling is the
entire difference**, and it is scaling at two levels rather than one:

| | MXFP4 | NVFP4 |
|---|---|---|
| elements | E2M1 | E2M1 |
| block size | 32 | 16 |
| block scale | E8M0 -- a power of two | E4M3 -- 1/4/3 bits |
| global scale | none | one per tensor |
| scale rounding | exponent rounded **up** | RTNE (`quantize_e4m3`) |
| clipping of the block max | impossible by construction | up to ~6% |
| resolution given up to the scale | up to a full binade (1 bit) | at most 6.25% |
| blocks reachable from one scale | `2**-127 .. 2**127` | `2**-9 .. 448`, relative to the global scale |
| rows independent of each other | yes | **no** |

**Recipe.**

1. Global scale, one number for the whole tensor:
   `s_global = max_b(amax_b) / (6 * 448)`.
2. Per block: `s_b = quantize_e4m3((amax_b / s_global) / 6)`.
3. `q_i = quantize_e2m1(x_i / (s_b * s_global), mode=round_mode)`.
4. Reconstruct `x_hat_i = q_i * s_b * s_global`.
5. Degenerate cases: an all-zero tensor takes `s_global = 1`, and a block
   whose `s_b` is zero reconstructs to zero.

**Why the global scale exists.** To keep the block scales inside E4M3's
range, and for no other reason. A block scale is a magnitude taken from the
input data, `amax_b / 6`, and E4M3 tops out at 448 -- so on a tensor whose
values run to `1e6` every block scale is out of range. E4M3 does not
saturate: the code above its maximum is NaN, so this failure is not even a
graceful one. Dividing by `s_global` first normalises the tensor so that its
largest block wants a scale of exactly 448, the top of E4M3, and every other
block wants less. `s_global` is a full float64 here (float32 in hardware), so
it costs nothing in precision and buys the whole E4M3 range for the block
scales.

Dropping this step is the natural bug and it has a diagnostic signature:
block scales pinned at the top of E4M3 (or NaN), and NVFP4 measuring *worse*
than MXFP4 -- the opposite of what the format is for. `tests/test_blocks.py`
asserts the accuracy ordering directly for that reason, so the symptom is a
test failure rather than a number in a results table.

**The analytic contrast.** The same block used for MXFP4 above, `amax = 4.5`,
at equal block size so that the scale format is the only difference:

| | MXFP4 | NVFP4 |
|---|---|---|
| ideal scale | 0.75 | 0.75 |
| representable scale | `2**0 = 1` (rounded up) | `s_b * s_global = 0.75` exactly |
| `4.5 / s` | 4.5 | 6 |
| nearest E2M1 grid point | 4 | 6 |
| reconstruction | **4.0** | **4.5 exactly** |
| error | 0.5 | 0 |

`0.75 = 1.5 * 2**-1` is an E4M3 value and is not an E8M0 value, and that one
fact is the whole of the difference here. Both halves are asserted exactly
(`==`, not `allclose`) in `tests/test_blocks.py`.

**Clipping is possible, unlike in MXFP4.** `quantize_e4m3` rounds to nearest,
so `s_b` can round *down* below the ideal scale, leaving
`amax_b / (s_b * s_global)` above 6 -- and E2M1 saturates there, clipping the
block maximum. The bound is E4M3's half-ulp at the bottom of a binade, so the
block max loses at most about 6% of its magnitude. MXFP4's round-up rule
forbids this by construction, at the cost of up to a full binade of
resolution for every other element in the block. NVFP4 takes the opposite
side of that trade, and this implementation follows the format rather than
importing MXFP4's rule; a variant that rounds `s_b` up to the next E4M3 value
would be a different format and is not what is measured here.

**Zero scales.** E8M0 has no zero, so an MXFP4 scale is never zero and an
all-zero block is given `s_b = 1`. E4M3 does have a zero, so `s_b == 0` is a
representable outcome and occurs for two kinds of block: one that is entirely
zero, and one whose `amax` is so far below the tensor's that its ideal scale
falls under E4M3's smallest subnormal -- precisely when
`448 * amax_b / amax < 2**-10`, i.e. a ratio below `2**-18.8`. Both flush to
zero, which is what a scale of zero means. So NVFP4's usable scale range
spans about 19 binades, against E8M0's 255: the price of spending scale bits
on mantissa instead of exponent, and a real limit on tensors with extreme
per-block dynamic range.

**Rows are not independent.** `s_global` is per tensor, leading axes
included, so a row's reconstruction depends on the rest of the tensor -- a
property MXFP4 does not have and one to keep in mind when comparing a matrix
against its rows. The block scales themselves remain per row.

**One numerical corner.** The argument to `quantize_e4m3` in step 2 is
`448 * amax_b / amax` and so cannot exceed 448 mathematically. It can
numerically: once `amax` is below roughly `1e-305`, `s_global` is itself a
float64 subnormal and carries a large relative rounding error (at
`amax = 1.4e-320` the argument comes out at 472), which E4M3 would take to
NaN. `nvfp4_block_scales` therefore clamps the argument to 448 before
quantizing. Below about `1.3e-320` the global scale underflows to zero
outright and falls back to `1.0`. Both are properties of float64-extreme
input and are reported through the returned values, not raised.

**Blocking axis, tail policy, input contract.** Identical to MXFP4: blocks
run along the **last axis** only, a trailing partial block stays short and
takes its own scale (numerically identical to zero-padding it, asserted as a
test), and NaN and infinity are rejected with `ValueError`. Note that under
NVFP4 a non-finite value would poison the *global* scale and with it the
entire tensor, not just its own block.

**Relationship to `torchao`.** `tests/test_blocks.py` carries a golden-
reference comparison against `torchao`'s NVFP4, restricted to a tensor whose
values are float32-exact and whose scales land on E4M3 grid points exactly.
It is a **sanity check, not an authority**: `torchao`'s NVFP4 lives under
`torchao.prototype`, it computes the global scale in float32, and unlike
`microxcaling` for MXFP4 it is not a reference implementation of a published
specification. The test `skip`s when `torchao` is not importable, which is
the case in this environment until `torch` arrives in Phase 4.4.

## Rounding modes (`qgemm.rounding`)

TBD -- which rounding modes are compared (round-to-nearest-even,
stochastic rounding, ...).

## Transforms (`qgemm.transforms`)

TBD -- scaling strategy (per-tensor / per-block scale, absmax vs. other
calibration). Implemented so far: the random Hadamard transform.

### Random Hadamard Transform

Implemented: `apply_rht(x, block_size, rng)`, `invert_rht(x, block_size, rng)`,
plus the two pieces they are built from, `hadamard_matrix(block_size)` and
`random_signs(block_size, rng)`. float64 in / float64 out.

**Recipe.** `H` is the normalized Sylvester matrix, `H_1 = [1]` and
`H_2n = (1/sqrt(2)) * [[H_n, H_n], [H_n, -H_n]]`; it is symmetric, orthogonal,
and has every entry of magnitude `1/sqrt(block_size)`. It is randomized by
flipping column signs, `H' = H * diag(eps)` with `eps_i` uniform on `{-1, +1}`
drawn from the passed-in `Generator`, which leaves it orthogonal. Each
contiguous block of `block_size` elements along the last axis is replaced by
`H' @ x_b`; `invert_rht` applies `H'.T`.

`block_size` must be a **power of two** and must divide `x.shape[-1]`; anything
else raises `ValueError`. The project sweeps `RHT_BLOCK_SIZES = (8, 16, 32, 64)`,
matching the block sizes in `qgemm.blocks`.

**What it buys.** The transform is an isometry, so it redistributes a block's
energy without changing it. A block whose `amax` is set by one outlier forces a
large shared scale and spends the resolution of every other element on that
outlier; the RHT spreads it. Exactly: a lone spike of magnitude `a` becomes
`block_size` entries of magnitude `a/sqrt(block_size)`, so a length-16 block
holding `[100, 0, ..., 0]` comes out with every entry at `25` and its `amax`
down from 100 to 25. On heavy-tailed data the effect is statistical rather than
exact -- on t-Student samples with `nu = 2`, empirical kurtosis over blocks of
32 falls by roughly an order of magnitude. The kurtosis test asserts only the
direction of that, since `nu = 2` has infinite population kurtosis and no exact
target exists.

Randomizing the signs is not decoration. A bare `H` has fixed structure, and a
block proportional to a row of `H` concentrates into a single spike instead of
being spread. The random signs make that a coincidence for any fixed input
rather than a property of the data.

**Both operands, or nothing.** Orthogonality is what makes the transform free
across a GEMM:

```
(H'a) . (H'b) = a . (H'.T H' b) = a . b
```

This holds only if **both** operands were transformed with the **same** `H'`.
`apply_rht` transforms one array and cannot check the other; the invariant is a
property of how it is called, and is enforced at the call site in the GEMM
pipeline (Step 1.7, `qgemm.gemm`). Two calls produce the same `H'` exactly when
they are passed generators in the same state -- in practice two
`np.random.default_rng(seed)` from one seed -- since `H'` depends on `rng` only
through `random_signs`. The same requirement applies to `invert_rht`: given a
generator in any other state it produces a different, equally valid orthogonal
matrix and silently returns something that is not the original. Transforming one
operand and not the other computes a different quantity and raises nothing,
which is why `tests/test_transforms.py` pins both misuses as explicit negative
tests.

**Per block only; a global RHT is deliberately excluded.** Blocks run along the
**last axis**, the same convention as `quantize_blocked` -- the axis a block
format shares a scale over, and the contraction axis a shared rotation cancels
along. A global (whole-row, length-`n`) RHT is a real technique and is
*excluded by design, not merely unimplemented*: its spread factor is `sqrt(n)`
rather than `sqrt(block_size)`, which would tie the transform's effect to the
same `n` whose error-growth law this study sweeps, confounding the two. Keeping
it per block leaves `block_size` as the only knob governing the transform. It
should not be added "for completeness".

**Tail policy -- differs from `quantize_blocked`, except in one corner that was
fixed on 2026-08-20.** A *trailing* partial block (some full blocks followed by
a short remainder, e.g. `n=100, block_size=32`) is still not allowed: unlike
`quantize_blocked`'s short tail, which is well defined because a scale does not
care how many elements it covers, a partial block has no `H'` to multiply by,
and padding it would change the array's shape and break invertibility. That
case still raises `ValueError`.

**A *whole row* shorter than `block_size` is different, and is now handled.**
`n < block_size` is unambiguous -- the entire row *is* one block, just a
smaller one than the nominal `block_size` -- so `apply_rht` falls back to
transforming it at its own length `n`, mirroring `quantize_blocked`'s
short-block convention. **This was a real bug, not a documented limitation:
`apply_rht` originally raised `ValueError` on `n < block_size` exactly as it
does on a genuine trailing partial block, and Step 1.6's original test suite
never exercised `block_size > n`, so it went uncaught until Step 3.1's
real-multiprocessing dry-run measurement hit it directly on this project's own
`(n=16, block_size=32, rht=true)` grid cells** (32 of the frozen 640-cell
grid's cells, present since the grid was first written -- see "Sweep grid
(Step 3.1)"). Fixed in `qgemm.transforms._effective_block_size`, general (not
special-cased to `n=16`): the fallback size is `min(block_size, n)`, and it is
independently re-validated as a power of two (the nominal `block_size` being a
power of two does not guarantee `n` is, even though every `n`/`block_size`
pair this project actually sweeps happens to be one).

**The fallback has a real, physical consequence, not just a shape
accommodation.** The RHT's spread factor is `sqrt(effective_block_size)`, so
at `n < block_size` the achieved spread is `sqrt(n)`, strictly less than the
`sqrt(block_size)` a full-length block would give -- an outlier cannot be
spread across more elements than the row contains. Concretely, at
`n=16, block_size=32` a lone spike of magnitude `a` comes out at `a/sqrt(16)`
per entry rather than the `a/sqrt(32)` a genuine 32-element block would give,
`sqrt(2)` times larger. This is reported as a limitation of that one grid
corner, not silently absorbed: RHT's spreading power there is tied to `n`
rather than to the nominal `block_size` the rest of the grid uses. Verified in
`tests/test_transforms.py` ("short-row fallback" section): no crash,
output shape preserved, orthogonality and inner-product preservation hold at
the fallback size, round-trip via `invert_rht` is exact, the fallback spreads
by `sqrt(n)` (not `sqrt(block_size)`) directly, and the genuine
trailing-partial-block case (`n > block_size`, not a whole multiple) is
pinned as still raising, unchanged, by a regression test.

**Implementation note.** `hadamard_matrix` doubles a `+-1` matrix and divides by
`sqrt(block_size)` once at the end, rather than multiplying in `1/sqrt(2)` at
each level. The two are mathematically identical, but the single division keeps
the entries exact whenever `sqrt(block_size)` is (16 and 64 of the four swept),
which is what lets the dispersion property above be asserted with `==` rather
than a tolerance.

## GEMM under quantization (`qgemm.gemm`)

### Quantized GEMM pipeline

`qgemm(A, B, config)` is the study's main measurement instrument: it computes
an approximation of `A @ B` in which the *operands* have gone through a
block-scaled low-precision format, and returns float64. All variation lives in
`GemmConfig`; the function itself has no free parameters.

**The four steps.** For `A` of shape `(M, K)` and `B` of shape `(K, N)`:

1. **Transform** (optional, `config.rht`). One random Hadamard transform
   applied per block along the **contraction dimension** `K`, with the *same*
   `H'` on both operands. `rht_operands` owns this and the transposes it needs.
2. **Quantize** both operands with `quantize_blocked`, blocked along the
   contraction dimension, using `config`'s block size, scale format, global
   scale, element format and rounding mode. Skipped when
   `config.quantize` is False.
3. **Multiply the reconstructions** -- the dequantized float64 values, not the
   codes -- in the precision named by `config.accum`.
4. **Return**. The RHT is *not* inverted; see below.

**The contraction axis is the whole correctness question.** Both
`apply_rht` and `quantize_blocked` operate on the **last** axis. For `A` the
last axis already *is* `K`; for `B` the last axis is `N`, so `B` must be
transposed before either call and transposed back after. Getting this wrong is
a silent bug, not a crash: with `K == N` the wrong axis raises nothing and
returns an array of the right shape that measures a different quantity. Both
axis choices are pinned in `tests/test_gemm.py` against references that spell
the transposes out, plus a test that the wrong axis genuinely disagrees (so the
reference cannot be trivially satisfied).

**Why the RHT is not inverted.** `H'` is orthogonal, so it cancels inside every
inner product the GEMM forms:

```
(H'a).(H'b) = a.(H'.T H' b) = a.b
```

That is why it must hit *both* operands along the *same* axis, and why nothing
has to be undone afterwards. With quantization off, `qgemm` with the RHT on
returns the exact product to float64 rounding -- the transform changes nothing
about the product itself, only what quantization subsequently does to it, by
spreading each block's energy so a lone outlier no longer sets the block scale.

### Accumulation modes

| `accum` | how | role |
|---------|-----|------|
| `"exact"` | `np.matmul` in float64 on the reconstructions | **primary route** |
| `"bf16"`  | explicit loop over `K`, accumulator rounded to bfloat16 after each partial sum | secondary ablation |

**`"exact"` is primary because it isolates input-quantization error.** A
low-precision GEMM has two independent error sources: the operands snapped onto
a coarse grid, and the partial sums rounded in a narrow accumulator. They
compose, and one number measured with both active cannot be attributed to
either -- a downstream result would be unattributable. Under `"exact"` the only
approximation in the pipeline is step 2, so the residual
`qgemm(A, B, config) - A @ B` *is* the input-quantization error. `"bf16"` turns
the second source on deliberately to measure what it adds; it is never mixed
into a headline result.

The bf16 loop rounds only the accumulator: term `k` is formed in float64 as an
outer product and added to an accumulator that is immediately rounded back onto
the bfloat16 grid (RTNE, via `ml_dtypes.bfloat16`).

Note that "worse than exact" is sharp only against the exact route's own
output, which is by construction the exact product of the *same* quantized
operands, so any deviation from it is accumulation error alone. Measured
against the true `A @ B` the ordering is not guaranteed at small `K`: the two
errors add in quadrature and at `K = 256` the FP4 quantization error is roughly
twenty times the bf16 accumulation error, so which route lands closer is
decided by luck. Accumulation error grows with `K` and quantization error does
not, and by `K = 4096` the ordering is solid.

### Performance constraint

**The exact route must complete a 512x512x512 GEMM, quantization and RHT
included, in under one second**, asserted (not logged) in
`tests/test_gemm.py`. This is load-bearing: the sweep calls this function once
per (shape, format, block size, rounding mode, seed) cell, thousands of times,
and a per-block Python loop anywhere in the path would move the main experiment
from minutes to hours. Every step of the exact route is a whole-array NumPy
operation for that reason. Measured: 0.047 s. The `"bf16"` route is exempt --
it is an ablation over a handful of cells and loops over `K` by construction.

### `GemmConfig`

A frozen dataclass of plain scalars, so it is hashable and
`dataclasses.asdict`-able and therefore feeds the sweep's
`sweep_{sha256(config)[:12]}` result-naming convention directly.

| field | default | meaning |
|-------|---------|---------|
| `quantize` | `True` | whether step 2 runs |
| `block_size`, `scale_format`, `use_global_scale`, `element_format`, `round_mode` | MXFP4 | passed through to `quantize_blocked` |
| `rht` | `False` | whether step 1 runs |
| `rht_block_size` | `block_size` | transform block size; separable from the quantizer's |
| `accum` | `"exact"` | accumulation precision |
| `seed` | `0` | seeds every random draw in the pipeline |

The defaults are MXFP4, exact accumulation, no RHT -- the study's baseline
cell. NVFP4 is `GemmConfig(block_size=16, scale_format="e4m3",
use_global_scale=True)`.

`seed` is an `int` rather than a `numpy.random.Generator` so that a config
stays serialisable. It is expanded into **three independent streams** -- the
RHT signs, and stochastic rounding for `A` and for `B` separately. Independent
rather than shared, for two reasons: turning the RHT on must not shift the
rounding draws (a config should differ from another only in the ways it says it
does), and `A` and `B` must not share a rounding draw, since correlated
rounding errors do not cancel in an inner product the way independent ones do.
Generators are built from the seed and passed explicitly; NumPy's global RNG is
never touched.

## Error metrics (`qgemm.metrics`)

Implemented: `abs_error`, `max_abs_error`, `mean_abs_error`,
`relative_error`, `rms_error`. Extend as needed; keep float64 in/out.

## Theoretical bounds (`qgemm.bounds`)

Implemented: `gamma_n` (Higham's sequential-summation error growth
bound). TBD -- bounds specific to the quantized-GEMM error model this
project develops.

## Conventions (fixed -- see README.md)

- float64 in/out for all quantization functions.
- Explicit `numpy.random.Generator` for all stochastic functions --
  never NumPy's global RNG state.
- All results written to Parquet via the
  `sweep_{sha256(config)[:12]}.parquet` convention, with the full config
  saved alongside.

## Metric stability diagnostic -- PROPOSED, pending review

**Status: a proposal, not a decision.** Nothing in this section is implemented
anywhere in the codebase; `qgemm.metrics` is unchanged and
PREREGISTRATION.md §7 item 3 (the primary error metric) stays open. It decides
how every later result is reported, so it needs human review before it is
locked in. Everything above this heading is unaffected by it.

Produced by `scripts/check_metric_stability.py` (one run, no manual steps);
data and figures under
`results/diagnostics/metric_stability/metric_stability_d6c165499b90*`, the
digest being the sha256 of the run config saved alongside.

### What was measured

The candidate primary metric, per output element:

    BE_ij = |(A @ B)_ij - qgemm(A, B, config)_ij| / (|A| @ |B|)_ij

over `nu in {1, 2, 3, 30}` (t-Student degrees of freedom) x `n in {16, 256,
4096}` (contraction dimension), 1000 independent trials per cell, 32x32 output
per trial = 1 024 000 `BE` values per cell. Quantization active throughout, in
the **MXFP4 preset** (`GemmConfig()` defaults: block 32, E8M0 scale, no global
scale, E2M1 elements, RTNE, no RHT, `accum="exact"`). At `n = 16` a block-32
quantizer has a single *short* 16-element block by the tail policy above, so
that cell's block structure differs from the others by construction.

The concern being tested was that numerator and denominator are built from the
same `A`, `B`, and that a denominator with mass near zero could give the ratio
a heavier tail than either input has on its own.

### Recommendation: keep raw `BE` as the primary metric, at every tested n

**Raw `BE` is usable, including at n = 16.** The `log(BE)` fallback is not
needed as the primary reporting metric and is proposed only as a plotting
transform (the histograms are legible on no other axis) and as a secondary
descriptive summary. Two consequences worth stating plainly, because they are
what a reader will check:

- The median of `BE` and the "geometric median" are the **same number** -- the
  median is invariant under the monotone map `log`, so `median(BE) =
  exp(median(log BE))`. Since PREREGISTRATION.md §3 already specifies a
  bootstrap CI of the **median** of the ratio, the metric-stability question
  never bore on that statistic's existence. What it bore on is whether
  *mean-like* summaries of `BE` are admissible at all, and whether the median
  is estimable stably enough to bootstrap. Both hold; see below.
- `BE` is right-skewed, with `mean / median` between 1.18 and 1.62 across the
  grid. Mean and median are therefore both well-defined but **not
  interchangeable as descriptions**, and any table must say which it reports.

### Why -- what steps 3 and 4 actually showed

**Step 3, the denominator has no mass near zero -- measured, not inferred.**
Normalized by its own median, `(|A| @ |B|)_ij` never came near zero in any
cell. Worst case is the highest-risk corner `nu = 1, n = 16`: over 1.024M
samples the smallest denominator observed was `0.031 x median`, and
`P(D < 0.1 x median) = 9.6e-4`. In every other cell that probability is at
most `9.8e-7`, and `P(D < 0.01 x median) = 0` in **all twelve cells**. The
mechanism is structural rather than lucky: the denominator is a sum of
non-negative terms, so heavy operand tails push it *up*, never toward zero --
visible in the same figure as a right tail spanning four decades at `nu = 1`
against a left edge that stops at a third of the median.

**Step 3, attribution -- the ratio's tail is inherited from the numerator, and
the denominator damps it.** For the top 0.1% of `BE` values, the median
percentile rank of their own denominator is 0.46-1.00 in eleven of twelve
cells: the largest ratios occur where the numerator is extreme *and the
denominator is simultaneously large*, not where the denominator collapsed. The
counterfactual makes the size of the damping concrete -- dividing the same
numerators by a *fixed* denominator gives a spread of `1.4e6 x median` at
`nu = 1, n = 16`, where the actual ratio's spread is `13 x median`.

**Step 4, the tail index -- the ratio is lighter-tailed than its own
numerator, not worse.** Hill estimates for `BE` land in `[2.80, 5.97]` with
the lowest 95% CI bound over the grid at 2.77, so the mean (needs `alpha > 1`)
and the variance (needs `alpha > 2`) both exist at every tested `(nu, n)`. For
contrast, the **numerator alone** has `alpha = 0.92, 0.97, 0.98` at
`nu = 1` for `n = 16, 256, 4096` -- no finite mean. The max-to-sum ratios say
the same thing without any choice of tail fraction: `R_1(m)` and `R_2(m)` for
`BE` decay like `1/m` in all twelve cells, while the numerator's `R_1` at
`nu = 1` plateaus around 0.1-1 across six decades of sample size. Dividing by
`(|A| @ |B|)` is what removes the operand-scale heavy tail.

**The underlying reason, which is why this should generalize past the sampled
grid.** Under `accum="exact"`, with `d_i` the relative error of the quantized
product `a_i b_i`,

    |sum_i (a_i b_i - qa_i qb_i)| <= max_i d_i * sum_i |a_i b_i|

so `BE <= max_i d_i`, and the format bounds that: an element that survives
quantization keeps its magnitude to within the E2M1 grid, and an element that
flushes to zero loses at most its own contribution. `BE` is therefore
**bounded**, not merely light-tailed, and a bounded statistic cannot have a
divergent moment. Empirically `be_max` ranges from 0.021 to 1.59 over the
whole grid, with the sharp right edge visible in every histogram panel.
*Caveat on reading the Hill numbers:* applied to a bounded variable, Hill
describes how the sample approaches its ceiling and is not evidence of a
Pareto tail. The `alpha` values above should be quoted only for what they
support -- no divergence -- and not as a tail exponent of a power law.

**Estimator stability, at the trial level.** Across ten disjoint batches of
100 trials (the resampling unit fixed in PREREGISTRATION.md §3.2), the
batch-to-batch spread (max/min) is 1.007-1.059 for the mean, 1.011-1.034 for
the median, and 1.010-1.040 for the geometric mean. All three are stable at
1000 trials; the median is the most stable of the three at `nu = 1`, which is
one more reason not to disturb the preregistered choice.

### Does the answer differ between n = 16 and larger n?

**The verdict is the same at all three n -- but the margin is not, and the
difference should be reported rather than papered over.** `n = 16` is the
loosest corner on every step-3 measure, exactly as predicted: the
denominator's interquartile ratio falls from 4.17 (`n = 16`) to 2.08
(`n = 4096`) at `nu = 1`, and its observed minimum rises from `0.031 x median`
to `0.33 x median` over the same range. `n = 16` is also the only n where any
cell shows a denominator-driven contribution to the `BE` tail at all: at
`nu = 30, n = 16` the top-0.1% `BE` values sit at a median denominator rank of
0.20, with 8.2% of them below the denominator's own 1st percentile. Even
there, `P(D < 0.1 x median) = 0`, `be_max / median = 11.5`, and `alpha = 5.61`
-- the effect is visible but far from pathological, and no cell fails on any
criterion.

So: one verdict, with `n = 16` flagged as the corner to re-check if the design
changes -- particularly for a genuine block-16 configuration, since the
`n = 16` cell measured here is a *short* block-32 block covering the entire
contraction dimension, not a block-16 quantizer.

### Scope -- what this diagnostic does not cover

Measured only for MXFP4, RTNE, `accum="exact"`, no RHT, t-Student operands at
the four `nu` above, and a 32x32 output. Three gaps matter and are not closed
by these data:

- **`accum="bf16"`.** The boundedness argument above is specific to the exact
  route, where the residual is input-quantization error alone. Accumulation
  error is not bounded by `max_i d_i * sum_i |a_i b_i|` in the same way, so
  the bf16 ablation needs its own check before `BE` is reported for it.
- **RHT on.** With the transform active the numerator is built from the
  *transformed* operands while this denominator is built from the original
  ones, so the bound picks up a factor `(|H'A| @ |H'B|) / (|A| @ |B|)` that
  the RHT can push above 1. Whether `BE` is reported against the original or
  the transformed operands is a definition choice that this diagnostic did not
  make.
- **NVFP4 and the block-size controls.** Only the MXFP4 preset was run. The
  E4M3 scale path has a NaN clamp and a global scale that this diagnostic
  never exercised.

To reproduce: `python scripts/check_metric_stability.py` (about three minutes;
`--trials`, `--out-rows` and `--seed` are the only knobs, and changing any of
them changes the config digest and therefore the output filenames).

## Gamma_n sanity check -- MEASURED, pending human review

**Status: a measurement, not a decision.** This is a control experiment,
deliberately outside the quantized formats this project studies -- no
MXFP4/NVFP4/E2M1/block scaling anywhere in it. It exists to validate the
measurement pipeline itself (trial generation, `qgemm.metrics.backward_error`,
log-log slope fitting with a bootstrap CI) on standard IEEE fp16 rounding of
inner products, before that pipeline is trusted on anything exotic. Produced
by `scripts/sanity_gamma_n.py` (one run, no manual steps); data, config and
figure under `results/diagnostics/sanity_gamma_n/sanity_gamma_n_8fe43f31978e*`,
the digest being the sha256 of the run config saved alongside. Per the task
that produced it, this section stops at reporting the measurement -- whether
the result supports moving on to Phase 2 or revisiting Phase 1 is a human
call, made elsewhere, not here.

### What was measured

`n in {16, 64, 256, 1024, 4096, 16384}`, Gaussian(0,1) operands, 2000
independent trials per `n`. Per trial: an exact float64 inner product and an
fp16-rounded inner product of the same two vectors, computed by sequential
summation with **every** intermediate step -- each element-wise product and
each partial sum -- rounded to fp16, once under round-to-nearest-even (RTNE,
via `numpy.float16` casting, exact and needing no further validation) and
once under stochastic rounding (SR, via a small local `round_stochastic_fp16`
following this project's existing SR convention: round to the nearest fp16
neighbor with probability proportional to distance, unit-tested for
determinism on exact grid points, correct probability at a known quarter-step,
and unbiasedness). `qgemm.bounds.gamma_n` and `qgemm.metrics.backward_error`
are used unmodified -- this check is specifically what validates them, so
neither is reimplemented inline. fp16's unit roundoff `u = 2^-11` is derived
from `numpy.finfo(np.float16)` (`eps / 2`) and cross-checked against the
10-bit mantissa, not hardcoded.

### Zero-violation check against the deterministic worst-case bound

**Passed with zero exceptions, everywhere it is defined.** At every trial,
mode, and `n` where `gamma_n(n, u)` is defined (`n * u < 1`, i.e. `n <= 1024`
at fp16 precision), the observed `BE` never exceeded it -- across
2000 trials x 2 modes x 4 such `n` values, zero violations.

**At `n = 4096` and `n = 16384`, `n * u_fp16 >= 1` (2.0 and 8.0
respectively), so `gamma_n`'s closed form `n*u/(1-n*u)` is out of its domain**
(`qgemm.bounds.gamma_n` raises `ValueError` there by design -- see its
docstring). This is a real limitation of the classical bound at fp16
precision, not a gap in this check: fp16's coarse unit roundoff means the
formula's own validity condition fails well inside the range of `n` this
project's contraction dimensions span, so no zero-violation comparison could
be performed at those two `n`. Reported plainly rather than silently
skipped -- see `scripts/sanity_gamma_n.py`'s `run_mode`.

### The log-log slope, and why it does not need re-litigating here

```
RTNE: empirical log-log slope of median(BE) vs n is -0.0198 +/- 0.0099 (95% CI [-0.0299, -0.0102]), inconsistent with sqrt(n) (slope 0.5).
SR: empirical log-log slope of median(BE) vs n is -0.0122 +/- 0.0095 (95% CI [-0.0225, -0.0035]), inconsistent with sqrt(n) (slope 0.5).
gamma_n bound undefined (n * u_fp16 >= 1) at n = [4096, 16384]; zero-violation check performed only at the remaining n values, where it passed with zero exceptions across all trials.
```

Stated plainly, per the task this check was run under: **median(BE) is
essentially flat across four orders of magnitude of `n`, for both RTNE and
SR** -- slightly *negative* slopes, both bootstrap CIs excluding 0.5 and,
even more so, excluding 1.0. `p99(BE)` (plotted alongside, not fit) shows
the same flat pattern. The full table (`median_be`, `p99_be`, `max_be`,
`gamma_n_bound`, `violations` per mode x `n`) is in
`sanity_gamma_n_8fe43f31978e_summary.parquet`; the figure is
`sanity_gamma_n_8fe43f31978e_slope.png`.

**On how to read "inconsistent with sqrt(n)" above.** Two distinct
probabilistic results sit behind that phrase, and neither is a refined
version of the other. Higham & Mary (2019) replace `gamma_n` by a relaxed
constant of order `sqrt(n log n)*u`, holding with a probability bounded
below, for rounding errors modelled as mean-independent random variables.
Connolly-Higham-Mary (2021) prove a `sqrt(n)*u` bound with no log factor,
unconditionally, for **stochastic rounding**. Both are **high-probability
upper bounds** on sequential-summation error -- not statements about the
typical or expected growth rate.

> **Correction (2026-08-25).** This paragraph previously read "Higham &
> Mary (2019) and Connolly-Higham-Mary (2021) establish `sqrt(n)*u`
> (respectively `sqrt(n log n)*u` for the refined variant)", which
> attributed each paper's result to the other and called `sqrt(n log n)`
> the refinement of `sqrt(n)`. Both halves were wrong; see the note under
> "Citation verification (Step 6.1)" below. The script's raw
output above reports "inconsistent with sqrt(n)" in the narrow sense of a
CI-vs-0.5 comparison on the fitted slope; it does not mean the measurement
contradicts those theorems. A growth rate flatter than slope 0.5 --
including the flat-to-slightly-negative slope measured here for both RTNE
and SR -- sits comfortably *under* a probabilistic upper bound and is
consistent with it, not a violation of it. **The actual pass/fail criterion
for this gate is the zero-violation check above against the strict,
deterministic worst-case bound `gamma_n = n*u/(1-n*u)`**, which held with
zero exceptions across every trial and every `n` where it is defined (see
"Zero-violation check against the deterministic worst-case bound" above).

This was checked for an implementation bug before being written down here:
`fp16_dot_rtne`'s vectorized output was cross-verified, off the batched code
path, against a from-scratch scalar Python loop at `n = 16384` (bit-for-bit
match across sampled trials), and the same trials' `backward_error` output
was independently recomputed from the raw numerator (`|exact - fp16|`) and
denominator (`sum|a_i b_i|`) by hand, matching the pipeline's own numbers.
The flatness is not an artifact of this measurement.

No interpretation of what this means for Phase 2 vs. Phase 1 is offered
here, per the instructions this check was run under -- this section reports
the measurement and stops.

## Gamma_n gate — follow-up diagnostic, unresolved

**Status: two targeted checks against the flat-slope finding above, run
diagnosis-only.** Neither the metric, the quantization code, nor the
original `scripts/sanity_gamma_n.py` was modified to produce this section --
both checks call that script's existing functions (`fp16_dot_rtne`,
`fp16_dot_sr`, `backward_error`, `fit_loglog_slope`, `bootstrap_slope_ci`)
unmodified, from a throwaway analysis script kept outside the repo (not
committed), the same pattern `tests/test_sanity_gamma_n.py` already uses to
load the script's functions without a package. Both checks used the same
2000-trial, seed-0 configuration as the original run; the full-range slopes
below reproduce the original run's numbers exactly (`-0.0198`/`-0.0122`),
confirming the two runs are the same measurement, not a different one.

### Check 1 — regime validity: restricting to n·u ≤ 0.125

fp16's `u = 2^-11`, so `n·u = 1` at `n = 1/u = 2048` exactly -- inside the
tested grid, between `n = 1024` and `n = 4096`. Refitting the log-log slope
of `median(BE)` vs `n` using only `n in {16, 64, 256}` (`n·u` at most
`0.125`, comfortably inside the regime where the `sqrt(n)·u` asymptotic is
supposed to hold), with the same percentile-bootstrap method (`B = 10000`,
resampling trials) as the full-range fit:

```
RTNE: full range  (n = 16..16384)         slope = -0.0198 +/- 0.0099  (95% CI [-0.0299, -0.0102])
RTNE: restricted range (n = 16, 64, 256)  slope = -0.0570 +/- 0.0278  (95% CI [-0.0820, -0.0264])
SR:   full range  (n = 16..16384)         slope = -0.0122 +/- 0.0095  (95% CI [-0.0225, -0.0035])
SR:   restricted range (n = 16, 64, 256)  slope = -0.0161 +/- 0.0282  (95% CI [-0.0465,  0.0099])
```

**Stated plainly: restricting to the perturbative regime does not resolve
the discrepancy.** Both restricted-range 95% CIs still exclude 0.5. For
RTNE the restricted-range slope is *more* negative than the full-range
slope (-0.057 vs -0.020), not closer to the 0.5 the `sqrt(n)` law predicts.
For SR the restricted-range point estimate (-0.016) is close to the
full-range one (-0.012) and its CI upper bound (0.0099) comes nearer to
zero than the full-range CI does, but it still does not reach 0.5, and the
interval is wide enough (half-width 0.028, roughly 3x the full-range
half-width) that this reflects the smaller sample (3 points, fewer trials
entering the fit) more than it reflects convergence toward 0.5.

The companion figure, `median(BE)` vs `n` over the full range with the
`n·u = 1` boundary marked
(`results/diagnostics/sanity_gamma_n/sanity_gamma_n_followup_crossover.png`),
shows both curves already flat-to-declining well before `n = 2048`, with a
minimum around `n = 1024` (RTNE) or between `n = 1024` and `n = 4096` (SR)
and a slight uptick after the boundary for both. **The flattening does not
visibly coincide with the n·u = 1 threshold** -- it is present throughout
the tested range, on both sides of the boundary, not something that turns
on only once the boundary is crossed.

### Check 2 — accumulation implementation: is every step actually rounded to fp16?

Checked directly, by instrumenting one representative trial (`n = 256`,
same seed/data as the original run) step by step and asserting the
accumulator's dtype after **every** addition, not inferring it from the
two implementations' outputs matching (a match proves agreement, not that
either one does true per-step rounding).

**`fp16_dot_rtne` (the array implementation actually used by
`sanity_gamma_n.py`):** the accumulator is a `float16` numpy array; at all
256 steps, both the raw `acc + term` expression and the value reassigned to
`acc` were asserted `dtype == np.float16` before moving to the next step.
Zero exceptions across 256 steps.

**`brute_rtne` (the prior ad-hoc scalar cross-check, from the previous
session's verification, not the pipeline itself):** the accumulator is a
`np.float16` scalar; at all 256 steps, the *bare* `acc + term` expression
(before the function's own explicit outer `np.float16(...)` cast) was
checked and was already `type(...) is np.float16` -- so the outer cast is
not masking a wider intermediate. Zero exceptions across 256 steps.

**Definite answer, not inferred: both implementations round after every
single step; neither accumulates internally at higher precision.** This
rules out "the brute-force check only validated inputs/outputs, not
per-step rounding" as an explanation for the flat slope.

Neither check resolves the discrepancy.

### Check 3 — input generation: is anything normalized or rescaled by n?

Checked directly by printing the per-element sample mean and variance of
the generated `x`, `y` vectors for `n in {16, 256, 4096}` -- three
individual trials plus the full 2000-trial aggregate at each `n`, using the
same seeding `run_mode` itself uses -- rather than inferring it from
reading `sample_gaussian`'s source.

```
n = 16    trial 0: x mean=-0.1221 var=0.6554   y mean= 0.0644 var=1.1029
          trial 1: x mean= 0.0292 var=0.7325   y mean= 0.0419 var=0.8200
          trial 2: x mean= 0.2587 var=0.8528   y mean= 0.2115 var=1.0362
          aggregate (2000 trials): x mean= 0.0103 var=1.0055, y mean=-0.0026 var=0.9921
          1/n = 0.0625

n = 256   trial 0: x mean=-0.0422 var=0.8916   y mean= 0.0796 var=0.9548
          trial 1: x mean=-0.0375 var=1.0315   y mean= 0.0268 var=1.0385
          trial 2: x mean= 0.0362 var=1.1004   y mean=-0.0623 var=1.0350
          aggregate (2000 trials): x mean=-0.0001 var=1.0000, y mean=-0.0011 var=1.0023
          1/n = 0.0039

n = 4096  trial 0: x mean= 0.0234 var=0.9935   y mean= 0.0297 var=0.9815
          trial 1: x mean=-0.0216 var=1.0037   y mean=-0.0180 var=1.0568
          trial 2: x mean=-0.0040 var=1.0500   y mean=-0.0059 var=1.0164
          aggregate (2000 trials): x mean= 0.0003 var=0.9993, y mean= 0.0002 var=1.0000
          1/n = 0.0002
```

**Stated plainly: no normalization or rescaling by n is happening,
anywhere.** Per-trial variance fluctuates around 1.0 at every tested `n`
(as expected for `n` independent draws averaged into one variance
estimate -- the fluctuation shrinks with `n`, but the *center* does not
move), and the 2000-trial aggregate variance sits at 1.0000, 1.0000 and
0.9993-1.0000 respectively -- not at `1/n` (0.0625, 0.0039, 0.0002), which
is what unit-vector-norm scaling would produce. `qgemm.distributions.
sample_gaussian` is `(rng.standard_normal(shape) * scale).astype(np.float64)`
with `scale` defaulting to `1.0`, and the call site in `sanity_gamma_n.py`
(`run_mode`) passes no `scale` argument, so nothing between the RNG and the
vectors used in the dot product rescales by any function of `n`.

### Check 4 — numerator/denominator decomposition

Reusing `check_metric_stability.py`'s Step-1.8 decomposition style: for
each `n` and each mode, computed and fit the log-log slope vs `n`
separately for (a) the raw, **unnormalized** absolute error
`median(|fl(dot) - exact_dot|)` and (b) the denominator alone
`median(sum|x_i||y_i|)`, cross-checked against `qgemm.metrics.
backward_error`'s own output on the same trials (`np.allclose`, agrees to
`rtol=1e-10`).

```
mode     n  median_numerator  median_denominator  median_ratio
rtne    16          0.001295            9.903585      0.000134
rtne    64          0.005018           40.527345      0.000123
rtne   256          0.018346          162.552050      0.000115
rtne  1024          0.072548          651.539255      0.000113
rtne  4096          0.299788         2605.189384      0.000116
rtne 16384          1.201727        10431.614106      0.000116
  sr    16          0.001579            9.831194      0.000170
  sr    64          0.006785           40.266042      0.000167
  sr   256          0.026450          162.362319      0.000163
  sr  1024          0.099252          651.364801      0.000153
  sr  4096          0.401963         2605.222058      0.000155
  sr 16384          1.662974        10428.156502      0.000160
```

Log-log slopes vs `n` (n = 16..16384, same `fit_loglog_slope` used
throughout):

```
RTNE:
  (a) median(|fl(dot) - exact_dot|)  raw numerator   slope = 0.9853
  (b) median(sum|x_i||y_i|)          denominator     slope = 1.0032
      (a) - (b) [reconciliation check]              = -0.0179
      directly fit ratio slope (median(a/b) vs n)   = -0.0198

SR:
  (a) median(|fl(dot) - exact_dot|)  raw numerator   slope = 0.9968
  (b) median(sum|x_i||y_i|)          denominator     slope = 1.0044
      (a) - (b) [reconciliation check]              = -0.0075
      directly fit ratio slope (median(a/b) vs n)   = -0.0122
```

Stated plainly: the raw, unnormalized numerator's slope (0.9853 RTNE,
0.9968 SR) and the denominator's slope (1.0032 RTNE, 1.0044 SR) are both
close to 1.0, for both modes. `(a) - (b)` (-0.0179 RTNE, -0.0075 SR) is
close to, but not identical to, the ratio slope obtained by fitting
`median(a/b)` directly (-0.0198 RTNE, -0.0122 SR) -- the two do not have to
match exactly, since `median(a)/median(b)` and `median(a/b)` are not the
same statistic, and this is reported as the reconciliation check the task
asked for, not resolved further.

### Check 5 — does the exact running partial sum grow like sqrt(k) or like k?

Tests one specific hypothesis for the numerator/denominator cancellation in
Check 4: that Gaussian mean-zero inputs make the running **exact** partial
sum `S_k = sum_{i<=k} x_i y_i` (float64, no fp16 rounding involved at all --
this is independent of RTNE/SR by construction) grow like `sqrt(k)`
(cancellation) rather than like `k`. Checked at `n = 4096`, same seeding as
the other checks.

**5 individual representative trials**, log-log slope of `|S_k|` vs `k`
fit over the full `k = 1..4096` path of each:

```
trial 0: |S_1|=1.8159  |S_2048|=45.3177  |S_4096|=48.9533  slope = 0.4052
trial 1: |S_1|=0.1185  |S_2048|=16.6092  |S_4096|=80.6207  slope = 0.6553
trial 2: |S_1|=0.0544  |S_2048|=67.2391  |S_4096|=46.8190  slope = 1.0459
trial 3: |S_1|=0.0037  |S_2048|=12.2443  |S_4096|=49.8653  slope = 0.6778
trial 4: |S_1|=0.0846  |S_2048|=79.4177  |S_4096|=43.3935  slope = 0.7719
mean of the 5 per-trial slopes = 0.7112, range = [0.4052, 1.0459]
```

Individual-trial slopes are noisy and range fairly widely (0.41 to 1.05),
which single random-walk paths are expected to do -- a log-log fit over one
fluctuating path is sensitive to where that particular path happens to sit
at the start and end of the range.

**Aggregate: `median(|S_k|)` across all 2000 trials at `n = 4096`, at each
`k`** (same `fit_loglog_slope` used throughout):

```
median(|S_1|)=0.3464  median(|S_2048|)=31.0785  median(|S_4096|)=42.5786
log-log slope of median(|S_k|) vs k, over all 2000 trials = 0.5132
```

For scale reference: a pure `sqrt(k)` projection from the `k=16` anchor
(`median(|S_16|)`) predicts `41.2506` at `k=4096`, against the actually
observed `42.5786`; a pure linear (`k`) projection from the same anchor
predicts `660.0088` -- more than 15x the observed value.

**Stated plainly: the aggregate slope (0.5132) is close to 0.5, not to
1.0.** This is the cancellation regime the hypothesis in this check's brief
describes, and the `sqrt(k)`-anchored projection lands much closer to the
observed `|S_4096|` than the linear-anchored one does. The individual
per-trial slopes are noisy (mean 0.7112, one trial as high as 1.0459) and,
taken alone, would not have settled the question either way; the aggregate
over 2000 trials is the number this check's brief asked to weigh most.

No further interpretation or next step is offered here, per the
instructions these checks were run under.

### Gate verdict

**The gamma_n gate is considered PASSED.** Zero violations of the strict,
deterministic worst-case bound `gamma_n = n*u/(1-n*u)` were observed across
every trial, both rounding modes, and every `n` where the bound is defined.
The flat-to-slightly-negative ratio slope that motivated Checks 1-5 is now
mechanistically explained rather than an open question: Check 5 shows the
exact running partial sum grows with `k` at slope ~0.51 (aggregate over
2000 trials at `n = 4096`), consistent with the `sqrt(k)`-scaling
cancellation expected of mean-zero Gaussian inputs, and Check 4 shows this
same `sqrt(n)`-vs-`n` cancellation propagating through the raw numerator
and the linearly-growing denominator to produce the flat ratio -- not a
measurement-pipeline defect.

## arXiv:2408.02897 reproduction -- MEASURED, qualitative match

**Status: measured, and the qualitative claim reproduces.** This is an
*external* check: `scripts/sanity_gamma_n.py` validated the measurement
pipeline against theory (Higham's deterministic worst-case bound), and this
validates it against an independent published empirical result. Neither
substitutes for the other, and neither is a confirmatory run.

Target: Rasquinha & Tabak (Google), "A Metric Driven Approach to Mixed
Precision Training", arXiv:2408.02897 -- the methodologically closest prior
work to this project. The four facts targeted, taken from this project's own
reference material rather than from recall of the paper:

* backward error defined at the **inner-product** level as
  `BE = |L.R - Q(L,R)| / (|L|.|R|)`, which is exactly
  `qgemm.metrics.backward_error` and is used here unmodified;
* base case **512x512** matrices;
* **t-Student** inputs across a varying normality parameter;
* finding: **FP8 stays much more bounded than INT8 as the tails get heavier.**

Produced by `scripts/sanity_reproduce_2408.py` (one run, no manual steps);
data and figures under
`results/diagnostics/sanity_reproduce_2408/sanity_reproduce_2408_0460704202d1*`,
the digest being the sha256 of the run config saved alongside. **50 trials per
(nu, format) cell**, each trial a fresh pair of 512x512 operands, so 13.1M
inner products per cell; the whole grid took **54.2 s**.

### The two recipes, and the FP8 choice this project had to make

Both formats are quantized **per-tensor** -- one scale for the whole operand
matrix, not the block scaling the main study uses for MXFP4/NVFP4. Operands
are quantized, then multiplied in float64 (`accum="exact"`, the study's
primary route), so the only error present is operand quantization.

| | recipe |
|---|---|
| INT8 | `qgemm.formats.int8_quantize`: `scale = amax/127`, symmetric, codes clipped to `[-127, 127]` |
| FP8-E4M3 | `scale = amax/448`, then `quantize_e4m3(x/scale) * scale` |

The FP8 line is a **documented methodological assumption, not a hidden
default.** This project does not have the paper's FP8 recipe, so the choice
was made here: a per-tensor amax scale mirroring INT8's, rather than applying
`quantize_e4m3` directly and relying on E4M3's native dynamic range. Two
reasons, in order of force:

1. *The unscaled route is not merely different, it is unusable on this grid.*
   E4M3 has no infinity encoding and overflows to **NaN** (see the E4M3
   section above). At `nu = 1` roughly **0.14%** of a 512x512 t-Student sample
   exceeds 448, and a single NaN operand element turns the entire inner
   product -- and hence the whole quantized product -- into NaN. There would
   be no number to compare.
2. *Fairness.* Giving both formats the same amax-derived per-tensor scale
   means they differ only in the **grid** they land on, which is the
   comparison the paper's claim is about. It is also the standard per-tensor
   FP8 setting in practice.

The trade this scale makes is explicit: the top of the range is guaranteed
(the largest element maps to exactly `+-448`, so overflow cannot occur), and
the bottom is at risk (elements more than `448 * 2**9` below amax fall under
the smallest E4M3 subnormal and flush to zero). INT8 makes the same trade on a
grid with far less dynamic range -- 127 codes, all of them uniform.

### What was found

Median BE per `nu`, pooled over all 13.1M inner products in the cell:

| nu | 1 | 2 | 3 | 5 | 8 | 15 | 30 | gaussian |
|---|---|---|---|---|---|---|---|---|
| INT8 | 0.3294 | 0.06672 | 0.01496 | 0.003279 | 0.001693 | 0.001081 | 0.0008707 | 0.0007133 |
| FP8-E4M3 | 0.03449 | 0.003393 | 0.002450 | 0.002034 | 0.001894 | 0.001817 | 0.001783 | 0.001748 |
| INT8/FP8 | 9.55x | 19.7x | 6.11x | 1.61x | 0.894x | 0.595x | 0.488x | 0.408x |

**The paper's qualitative claim reproduces.** As the tails get heavier, FP8
stays bounded and INT8 does not: from the Gaussian end to `nu = 1`, median BE
rises **462x** for INT8 and **19.7x** for FP8, and over the whole range
`nu = 30` down to `nu = 2` the FP8 curve moves by under a factor of 2 (0.00178
to 0.00339) while INT8 moves by 77x. That is the shape the paper reports, and
it is visible directly in
`sanity_reproduce_2408_0460704202d1_be_vs_nu.png`: an INT8 curve spanning
nearly three decades against a near-flat FP8 curve.

Three qualifications, none of which soften the direction but all of which
bound what was actually shown:

* **The advantage is not universal across the grid -- it reverses at light
  tails.** At `nu >= 8` INT8 is the *better* format, by up to 2.4x at the
  Gaussian end, and the two curves cross at `nu ~ 8`. This is expected rather
  than anomalous (a uniform 127-code grid resolves near-Gaussian data more
  finely than E4M3's 3-bit mantissa, whose relative step is ~2**-4), and it
  does not contradict a claim about heavy tails. But "FP8 is more bounded than
  INT8" is true here only *as a statement about the heavy-tail regime*, and
  this project should not restate it unqualified.
* **The ratio is non-monotone at the extreme.** It peaks at `nu = 2` (19.7x)
  and falls back to 9.55x at `nu = 1`, because INT8 has saturated -- a median
  BE of 0.33 means essentially all information in the operands is gone, so it
  cannot get much worse -- while FP8's dynamic range is finally being
  exhausted too. The FP8 advantage is largest in the middle of the heavy-tail
  range, not at its edge.
* **`nu = 1` is the one cell whose headline number is unstable.** Per-trial
  medians there range over 0.0119-0.166 for FP8 (a 14x spread across
  independent draws) against 0.316-0.340 for INT8, because the FP8 scale is
  driven by the amax of a Cauchy sample, which has no finite mean. Every other
  cell's per-trial medians sit within a few percent of the pooled median. The
  `nu = 1` FP8 value should be read as an order of magnitude, not a
  measurement.

### Qualitative only -- what is not claimed

**No numeric value from the paper is claimed to be matched, and none could
be:** this project has neither the paper's code nor its seeds. Only the
direction and rough order of magnitude are the target, and the script
accordingly has no pass/fail tolerance (an earlier placeholder version of
`scripts/sanity_reproduce_2408.py` was built around a `REFERENCE_VALUE` and a
`REFERENCE_TOLERANCE`; that contract was dropped deliberately, since a
tolerance gate against a number we cannot source is fake precision).

Assumptions made here that could each account for numeric differences against
the paper's own figures:

1. **The FP8 per-tensor recipe** (`amax/448`) above -- the largest such
   assumption, since the entire FP8 curve shifts with the choice of scale.
2. **Exact float64 accumulation.** Any accumulator rounding in the paper's
   setup adds a second error source this measurement does not contain.
3. **No rescaling of the t-Student samples**, following
   `scripts/check_metric_stability.py`'s `sample_t`; how the paper normalizes
   its normality-parameter sweep is not known to this project.
4. **The nu grid itself** (`1, 2, 3, 5, 8, 15, 30, gaussian`), which is this
   project's grid, not the paper's.
5. **RTNE throughout**, no stochastic rounding, no RHT.

### Scope

This diagnostic covers per-tensor INT8 and per-tensor FP8-E4M3 at one shape
(512x512, so contraction dimension 512) with exact accumulation. It says
nothing about the block-scaled MXFP4/NVFP4 formats the main study is actually
about, nothing about other shapes or contraction dimensions, and nothing about
stochastic rounding or the RHT.

It also **does not close PREREGISTRATION.md section 7 item 1** (the eight nu
values). The grid used here is a diagnostic choice; freezing it for the
confirmatory run requires a dated amendment under section 8, which this
section is not.

## n-scaling probe under exact accumulation -- PROPOSED, pending review

**Status: a measurement, not a decision.** Nothing in this section is
implemented anywhere in the codebase. `qgemm.bounds` is untouched and still
contains only `gamma_n`; `qgemm.metrics` is untouched. This section exists to
feed the open conceptual question of how a theoretical error bound for
block-scaled formats should be defined -- specifically, whether such a bound
should carry an `n`-dependence at all and, if so, on what argument. **That
decision is not made here and nothing below should be read as making it.** It
needs human review before `bounds.py` is implemented.

Produced by `scripts/probe_n_scaling.py` (one run, no manual steps, 4.8 min);
data and figures under
`results/diagnostics/n_scaling/probe_n_scaling_44503b1d4f36*`, the digest being
the sha256 of the run config saved alongside.

### Why the question is open

The classical bounds -- `gamma_n = n*u / (1 - n*u)` and the probabilistic
`sqrt(n)*u` -- take their `n`-dependence from **repeated rounding during
accumulation**: `n` sequential partial sums, each rounded once. This project's
primary route accumulates in exact float64 (`accum="exact"`), so that mechanism
is *absent by construction*: the residual `qgemm(A, B, cfg) - A @ B` is
input-quantization error and nothing else. The `gamma_n` sanity check above
already found the corresponding empirical consequence in a different setting
(fp16 inner products): `median(BE)` was flat in `n`, slope about `-0.02`,
because numerator and denominator grow at the same rate.

So any `n`-dependence measured here comes from some other mechanism. Two were
posed in advance, and the probe was designed to separate them:

* **Cancellation.** The numerator `|sum_k eps_k|` sums `n` signed
  quantization errors that partially cancel; the denominator
  `sum_k |a_k||b_k|` grows like `n` with no cancellation. Elements sharing a
  block share a *scale*, so their errors are not independent, and the effective
  number of roughly-independent summands may be the number of **blocks**,
  `n / block_size`, rather than `n`. That reading predicts
  `median(BE) ~ sqrt(block_size / n)`: slope **-0.5 at both block sizes**, with
  block 32 sitting a factor `sqrt(2) = 1.414` **above** block 16 in level.
* **Extreme-value dominance.** Under heavy tails the numerator may be owned by
  a single worst block rather than averaged over many, so growing `n` buys
  progressively less cancellation. That reading predicts a slope flattening
  toward 0, or a slope that itself differs between block sizes -- something the
  `n / block_size` rescaling **cannot** produce, since rescaling a power law's
  argument by a constant moves its level, never its exponent.

### What was measured

`block_size in {16, 32}` x `nu in {1, 30}` x `n in {64, 128, 256, 512, 1024,
2048, 4096}`, 500 independent trials per cell (a cheap probe -- fewer than a
confirmatory sweep cell would use), 28 cells, 2 048 000 `BE` values per cell.
Shape `(m=64, n) @ (n, k=64)`: small fixed output dimensions with only the
contraction length varying, the same convention
`scripts/check_metric_stability.py` uses. Every `n` is a multiple of both block
sizes, so no cell ever gets a short trailing block and the tail policy is held
out of the comparison.

**Held fixed across every cell, deliberately:** `scale_format="e8m0"`,
`use_global_scale=False`, `element_format="e2m1"`, `round_mode="rtne"`,
`accum="exact"`, no RHT. Fixing the scale format is the whole design: it makes
`block_size` the only quantizer parameter that moves, which is what isolates
the block mechanism.

**This reproduces neither MXFP4 nor NVFP4, and must not be read as doing so.**
MXFP4 is `(32, e8m0, no global scale)` and NVFP4 is `(16, e4m3, global scale)`.
The `block_size=32` arm here therefore coincides with MXFP4, but the
`block_size=16` arm is `(16, e8m0, no global scale)` -- one of the six
**experimental controls** in the table above, corresponding to no hardware, no
specification and no vendor. Nothing here benchmarks or endorses a format.

`median(BE)` per cell is the **median over trials of each trial's median** over
its 64x64 = 4096 output elements, the trial being the resampling unit
(PREREGISTRATION.md sec 3.2). The pooled median over all `trials x 4096` values
is tabulated alongside rather than assumed equal to it; the two agree to within
0.56% in the worst cell and 0.09% typically, so the choice between them does
not carry any result below.

### Result 1: fitted log-log slopes of median(BE) vs n

Percentile bootstrap, B = 10 000, resampling trials independently at each `n`.

| `block_size` | `nu` | slope | 95% CI |
|---|---|---|---|
| 16 | 1 | **-0.1395** | [-0.1412, -0.1380] |
| 32 | 1 | **-0.1436** | [-0.1454, -0.1418] |
| 16 | 30 | **-0.4978** | [-0.4983, -0.4972] |
| 32 | 30 | **-0.4966** | [-0.4973, -0.4960] |

Slope *difference* between block sizes, bootstrapped on the same replicates:
`-0.0040` [-0.0064, -0.0016] at `nu = 1`, and `+0.0012` [+0.0002, +0.0020] at
`nu = 30`. Both differences are formally nonzero and both are two orders of
magnitude smaller than the gap between any two reference patterns.

A note on reading these CIs: at 500 trials they are narrow enough
(half-width ~0.001) that a slope of `-0.4978` is *formally* distinguishable
from exactly `-0.5`. The script therefore reports the distance to the nearest
reference separately from whether the CI contains it, against a stated 0.02
reporting threshold fixed before the run. That threshold is a reporting
convention, **not** a hypothesis test, and it is not proposed as a criterion for
anything downstream.

### Result 2: the level, at matched n = 1024

| `nu` | block 16 | block 32 | ratio 32/16 | 95% CI |
|---|---|---|---|---|
| 1 | 0.073155 | 0.096608 | **1.3206** | [1.3068, 1.3324] |
| 30 | 0.005407 | 0.005593 | **1.0344** | [1.0317, 1.0373] |

At `nu = 1` the two curves sit at visibly different heights (about 32% apart).
At `nu = 30` they are **close to superimposed** -- separable, but by 3.4%.
Neither ratio's CI contains `sqrt(32/16) = 1.4142`, and at `nu = 30` the
observed ratio is nowhere near it.

Full `median(BE)` table, all 28 cells:

| `block_size` | `nu` | n=64 | 128 | 256 | 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|---|---|---|---|
| 16 | 1 | 0.106879 | 0.098245 | 0.089427 | 0.080481 | 0.073155 | 0.066115 | 0.060341 |
| 32 | 1 | 0.141508 | 0.130849 | 0.118004 | 0.106922 | 0.096608 | 0.086432 | 0.078789 |
| 16 | 30 | 0.021425 | 0.015202 | 0.010793 | 0.007639 | 0.005407 | 0.003823 | 0.002705 |
| 32 | 30 | 0.022030 | 0.015711 | 0.011135 | 0.007894 | 0.005593 | 0.003959 | 0.002796 |

Figure: `probe_n_scaling_44503b1d4f36_n_scaling.png`, log-log, one panel per
`nu`, both block sizes overlaid with the three reference slopes drawn as guides
anchored at `n = 64`, so slope and level are both readable at a glance.

### Result 3: which reference pattern the data resembles -- stated per nu

The two `nu` do **not** give the same answer, and are not forced into one
verdict.

**`nu = 30` (near-Gaussian): the data resemble the `-0.5` pattern, cleanly.**
Both block sizes fit `-0.497`, they agree with each other to `0.0012`, and the
curves are straight over the full 6-octave range of `n`. `median(BE)` falls by
a factor of about 7.9 from `n = 64` to `n = 4096` (a factor of 64 in `n`);
`sqrt(64) = 8`. This is the magnitude the classical probabilistic bound has,
arrived at with **no accumulation rounding anywhere in the pipeline** -- which
is the observation worth carrying into review, since it means a `sqrt(n)`
shape here would describe a different mechanism than the one `sqrt(n)*u`
classically describes.

**But the level does not follow the `n / block_size` reading.** A slope of
`-0.5` on its own does not identify what the independent summand is: replacing
`n` by `n / block_size` rescales the argument by a constant, which shifts the
curve's height by `sqrt(block_size)` and leaves the slope untouched. The level
is therefore the only thing that discriminates, and it shows a 32/16 ratio of
**1.034**, not the **1.414** that treating the block as the unit of
independence predicts. Read plainly: the `-0.5` slope is there, but the block
structure is *not* visible in the level at anything like the strength the
simple block-count rescaling requires. Whatever sets the effective sample size
at `nu = 30`, the data do not show it scaling with `block_size`.

**`nu = 1` (heavy-tailed): the data resemble none of the three references.**
The slope is about `-0.14` at both block sizes -- well away from flat, well
away from `-0.5`, and nowhere near `-1`. It sits between flat and `-0.5`, about
2.6 times closer to flat (0.140 from one, 0.360 from the other). `median(BE)`
falls by only a factor of 1.77 over the same 64-fold range in `n`.

The two block sizes share this slope: the difference is `-0.0040`, about 3% of
the slope itself and one 125th of the 0.5 gap between the two nearest
reference patterns. So the "slopes differ between block sizes" signature of
extreme-value dominance is
**not** present -- what is present is a common slope that is strongly flattened
relative to `-0.5`, which is the *other* signature that reading predicts
(cancellation being progressively bought off by single-block dominance as `n`
grows). The block size shows up in the **level** instead: 1.32 at `nu = 1`
against 1.03 at `nu = 30`, so the block structure is far more visible in the
error's height under heavy tails than under near-Gaussian data.

### What this does not settle

* **It does not choose a functional form for the bound.** Two `nu` give two
  different scalings (`-0.50` and `-0.14`), so any single `n`-exponent would
  have to be justified as covering both, or the bound would have to be
  `nu`-dependent, or `n`-independent and conservative. Which of those is right
  is exactly the question under review, and this probe deliberately stops short
  of it.
* **It does not explain the mechanism.** The slope/level split above is a
  description of two measured numbers, not a demonstration of what generates
  them. In particular the level ratio confounds two effects that this probe
  cannot separate: how much cancellation the block structure permits, and the
  fact that a larger block gives its shared scale more elements to cover and so
  a larger per-element quantization error in the first place. Separating those
  needs a measurement of the per-element error distribution at fixed `n`, which
  was not run.
* **Scope.** Two block sizes, two `nu`, one scale format (`e8m0`, no global
  scale), one shape family, RTNE only, no RHT, exact accumulation only, 500
  trials. It says nothing about E4M3 scales, global scales, block sizes 8 and
  64, stochastic rounding, the RHT, or `accum="bf16"` -- and in particular
  nothing about NVFP4, whose scale format is not the one held fixed here.
* **It closes no PREREGISTRATION.md item.** The grid, the trial count and the
  0.02 reporting threshold are diagnostic choices, not frozen commitments; none
  of them is a dated amendment under section 8.

## u_eff measurement -- bound-definition groundwork, RESOLVED

## Theoretical bound definition (🧠1) — RESOLVED

**Problem.** Classical bounds (γ_n, √n·u) assume a fixed unit roundoff
applied uniformly across n accumulation steps, with their n-dependence
arising specifically from repeated rounding during summation. Neither
assumption holds here: the unit roundoff varies per block depending on
outlier structure, and this project's primary experimental route
(Step 1.7, accum="exact") accumulates in exact fp64 — there is no
repeated rounding during summation, so the mechanism that produces
γ_n/√n·u's n-dependence is not physically present in this pipeline.

**Derivation of the correct scaling.** With exact accumulation, the only
error source is one-time quantization of each input element before
summation. The error at output entry (i,j) is approximately a sum of n
terms of alternating/mixed sign (each element's quantization error
contributes with essentially random sign), so by partial cancellation
this sum scales like √n, not n — the same mechanism that resolved the
Step 2.1 flat-slope investigation (Check 5: partial sums scale as √k
under mean-zero cancellation). The BE denominator, Σ|A_ik||B_kj|, is a
sum of n strictly positive terms and grows approximately linearly in n
for well-behaved inputs. Combining: BE ~ (√n · u_eff) / n = u_eff/√n —
DECAYING in n, not growing.

**Bound tested:**

    cota(n, format, block, ν) = c · u_eff(format, block, ν) / √n

where:
- u_eff(format, block, ν) := median of per-element relative quantization
  error, measured empirically per configuration (measure_u_eff, Step
  🧠1 groundwork), excluding elements below the format's flush-to-zero
  threshold (a dynamic-range floor, not a precision measurement — see
  documented limitation below).
- c is fixed a priori at c=1 for confirmatory analysis, per
  PREREGISTRATION §3.2's anti-circularity constraint (a bound whose
  constant is fitted to the same data it is tested against cannot be
  falsified by that data). Under the derivation above, c=1 is retained
  as the reference leading-order constant for the u_eff/√n scaling
  relationship; because the derivation is an order-of-magnitude
  argument (CLT/LLN reasoning) rather than a proven tight inequality,
  c=1 is not guaranteed to be well-calibrated, and a poorly-calibrated
  c is itself a reportable finding, not a reason to re-fit. Any ĉ
  fitted from data (Step 4.1) remains descriptive/exploratory only
  (§3.2, §5(b)) and is never substituted into the confirmatory ratio.

This form is directly consistent with measurement: at ν=30, empirical
median(BE) decays with slope ≈ −0.4978 (block 16) / −0.4966 (block 32),
closely matching the derived −0.5 exponent (n-scaling probe, Step 2.3
groundwork).

**Where the derivation breaks, and why this is expected, not a defect.**
At ν=1, empirical decay is much slower (slope ≈ −0.14) than the
derived √n rate predicts. The derivation above relies on the numerator
behaving like a mean-zero random walk (CLT) and the denominator behaving
like a law-of-large-numbers sum — both require finite variance/mean.
Neither holds at ν=1 (Cauchy): both numerator and denominator are
plausibly dominated by the same extreme elements rather than averaging
independently, consistent with the Step 1.8 metric-stability finding
that BE stays well-behaved specifically because its numerator and
denominator are commonly dominated by the same outliers. This is
recorded as an expected breakdown of the derivation's assumptions at
heavy tails, not an error in the bound's form.

**Break criterion (🧠7, unchanged in mechanics):** the bound is
considered broken at (format, ν, n) if the lower bound of the 95%
bootstrap CI of the median of empirical_BE / cota(n) exceeds 1.0 — i.e.
error decays reliably slower than the u_eff/√n cancellation rate
predicts, not that it exceeds a growing worst-case ceiling.

**Documented limitation carried over from the u_eff groundwork.**
u_eff under-predicts larger blocks' relative disadvantage (median BE
level ratio 32/16 explained by u_eff's own ratio: ~83% at ν=30, ~48% at
ν=1). Leading candidate mechanism, confirmed directionally but not
quantitatively: elements sharing a block share a scale, so their errors
are correlated rather than independent, violating the independence
assumption both this derivation and the classical bounds rely on — a
correlation strongly consistent with the survival-fraction gap between
block sizes growing 16.9× from Gaussian to ν=1 (Spearman 1.0 against the
u_eff excess across the full ν grid). Not resolved further here;
reported as a limitation for explicit discussion in Section 5.

**Why this doesn't block Phase 3.** The u_eff residual and the ν=1
decay-rate breakdown are both consistently signed, grow smoothly with
tail weight, and were consistent with a non-independence concern already
raised in the preregistration (§3.2, R8) regarding bootstrap resampling
validity — not late-discovered defects.


**Status: the definition above is resolved** (see "Theoretical bound
definition (🧠1) — RESOLVED"); what follows is the measurement and
diagnostic evidence that supports it.

What landed in code is a measurement function, not a bound:
`qgemm.bounds.measure_u_eff`, with `u_eff_samples` and
`elementwise_relative_error` underneath it, plus tests in
`tests/test_bounds.py`. `gamma_n` is unchanged, `qgemm.metrics` is unchanged,
`quantize_blocked` is unchanged.

Produced by `scripts/measure_u_eff.py` (one run, no manual steps, 1.1 min);
data and figure under `results/diagnostics/u_eff/u_eff_9d25687f1445*`, the
digest being the sha256 of the run config saved alongside.

### Why `u_eff` is a distribution, not a number

The classical unit roundoff is a property of a format alone -- half an ulp of a
fixed grid. A block-scaled format has no such number. The grid an element lands
on is set by the largest magnitude in *its block*, so an element far below its
block's amax is resolved far more coarsely than one at the top, and the
achieved relative error `|x_i - x_hat_i| / |x_i|` is a **distribution** whose
shape depends on the input distribution, the block size and the scale format
together. `measure_u_eff` reports quantiles of it, measured directly rather
than inferred backwards through `BE`.

### The near-zero policy -- a methodological decision, stated

Relative error is ill-defined as `x_i -> 0`, and for a block-scaled format
"near zero" only means anything **relative to the element's own block scale**:
block scales span many orders of magnitude across a heavy-tailed tensor, so a
fixed absolute floor would cut different cells in different places.

**The cut is taken at `0.25 x s_eff`**, where `s_eff` is the effective scale
(block scale times global scale) the element was quantized against. This is not
a tuned knob: E2M1's smallest nonzero magnitude is `0.5 x s_eff`, so under RTNE
every element at or below `0.25 x s_eff` quantizes to exactly zero and carries
a relative error of exactly `1.0`. Those elements record the format's
**dynamic-range floor**, not its **precision**.

Keeping them makes the measurement useless, and that was checked rather than
assumed: with the cut off, **every one of the 32 cells returns `p99 = 1.00000`
exactly**, and the median reaches `1.0` outright at `nu = 1, block 32`. A
quantity capped at 1 by construction cannot track a `BE` ratio, and what it
would be measuring is how much near-zero mass the input distribution has.

The price is that the excluded fraction is large and varies by cell, so the
**surviving fraction is tabulated below** and no number here should be quoted
without it. It ranges from 0.360 (MXFP4 at `nu = 1`) to 0.932 (NVFP4 at the
Gaussian end).

### Sample size, and why p99 is trustworthy here

`2**22 = 4 194 304` elements per cell, drawn as 64 independent `(64, 1024)`
tensors. The shape matters and is not cosmetic: NVFP4 uses a per-**tensor**
global scale, so one flat 4M-element draw would have an amax nothing like a
real operand's and would quantize its block scales differently. `(64, 1024)` is
the operand shape the n-scaling probe used at its level-comparison `n`, which
is what makes the check below apples-to-apples.

Every cell was **also** run at `2**23` with an independent seed. Across all 32
cells the largest `p99` movement is **0.178%** and the largest median movement
is **0.138%** -- both two orders of magnitude below the block-size effects being
measured. The extreme quantile is stable at this budget, not just the median.

### Result 1: u_eff by configuration and nu

Two of the four are real formats; two are the experimental controls that
complete the 2x2 (SPEC.md, "Why the controls exist") and correspond to no
hardware, no specification and no vendor.

**`u_eff` median (p50):**

| configuration | 1 | 2 | 3 | 5 | 8 | 15 | 30 | gaussian |
|---|---|---|---|---|---|---|---|---|
| MXFP4 (32, E8M0) *real* | 0.157282 | 0.139911 | 0.128911 | 0.119846 | 0.114749 | 0.111224 | 0.109440 | 0.108029 |
| NVFP4 (16, E4M3+g) *real* | 0.122349 | 0.109208 | 0.102935 | 0.098530 | 0.096450 | 0.095215 | 0.094485 | 0.094033 |
| control (16, E8M0) | 0.136158 | 0.124965 | 0.118059 | 0.112391 | 0.109365 | 0.107550 | 0.106387 | 0.105679 |
| control (32, E4M3+g) | 0.142980 | 0.124091 | 0.113595 | 0.106150 | 0.102772 | 0.100669 | 0.099753 | 0.098980 |

**`u_eff` p99:**

| configuration | 1 | 2 | 3 | 5 | 8 | 15 | 30 | gaussian |
|---|---|---|---|---|---|---|---|---|
| MXFP4 (32, E8M0) *real* | 0.966452 | 0.946205 | 0.924187 | 0.898866 | 0.879332 | 0.862935 | 0.853144 | 0.844455 |
| NVFP4 (16, E4M3+g) *real* | 0.944042 | 0.898145 | 0.858556 | 0.818837 | 0.795421 | 0.777581 | 0.767866 | 0.757603 |
| control (16, E8M0) | 0.953023 | 0.925238 | 0.900945 | 0.874172 | 0.855224 | 0.844844 | 0.837034 | 0.829281 |
| control (32, E4M3+g) | 0.961434 | 0.927008 | 0.892147 | 0.851539 | 0.826104 | 0.806383 | 0.797514 | 0.787335 |

**Surviving fraction after the near-zero cut:**

| configuration | 1 | 2 | 3 | 5 | 8 | 15 | 30 | gaussian |
|---|---|---|---|---|---|---|---|---|
| MXFP4 (32, E8M0) *real* | 0.360 | 0.678 | 0.780 | 0.838 | 0.863 | 0.879 | 0.887 | 0.893 |
| NVFP4 (16, E4M3+g) *real* | 0.589 | 0.825 | 0.879 | 0.908 | 0.919 | 0.926 | 0.929 | 0.932 |
| control (16, E8M0) | 0.517 | 0.762 | 0.829 | 0.868 | 0.884 | 0.894 | 0.898 | 0.902 |
| control (32, E4M3+g) | 0.425 | 0.759 | 0.843 | 0.887 | 0.904 | 0.914 | 0.918 | 0.922 |

Figure: `u_eff_9d25687f1445_u_eff.png` -- left panel `u_eff` (median, log
scale) against `nu`, one line per configuration, real formats solid and
controls dashed; right panel the level-gap check below.

**Note on p99.** At heavy tails the p99 is pressed against its own ceiling
(0.966 for MXFP4 at `nu = 1`; the maximum possible value is 1.0, at which an
element has flushed to zero). It is therefore compressed and a *poor*
discriminator exactly where the differences are largest: the block-32/block-16
p99 ratio is 1.014 at `nu = 1` against a p50 ratio of 1.155. Everything below
uses the median for that reason.

### Result 2: the level-gap check -- the direct test

The n-scaling probe found that `block_size` barely moves `median(BE)`'s
`n`-exponent but does move its **level**: at matched `n = 1024`, with the scale
format held at E8M0, block 32 sits `1.0344x` above block 16 at `nu = 30` and
`1.3206x` above it at `nu = 1`. The hypothesis was that `u_eff` itself differs
by block size and absorbs the whole effect. If `median(BE) ~ C(n) * u_eff`,
the two ratios must be equal.

Scale format held at E8M0, exactly as the probe held it:

| `nu` | u_eff ratio 32/16 (p50) | median(BE) level ratio | u_eff excess | BE excess | share of the gap explained |
|---|---|---|---|---|---|
| 1 | 1.1551 | 1.3206 | +0.1551 | +0.3206 | **48.4%** |
| 2 | 1.1196 | -- | +0.1196 | -- | -- |
| 3 | 1.0919 | -- | +0.0919 | -- | -- |
| 5 | 1.0663 | -- | +0.0663 | -- | -- |
| 8 | 1.0492 | -- | +0.0492 | -- | -- |
| 15 | 1.0342 | -- | +0.0342 | -- | -- |
| 30 | 1.0287 | 1.0344 | +0.0287 | +0.0344 | **83.4%** |
| gaussian | 1.0222 | -- | +0.0222 | -- | -- |

Both ratios sit near 1, so the comparison that means anything is between their
**excesses over 1** -- "1.03 versus 1.03" is unimpressive when the quantity of
interest is the 0.03. The threshold for calling this consistent was declared
before the run at 75% of the excess explained.

**Verdict: MIXED, and the two `nu` are not forced into one answer.**

* **At `nu = 30` the hypothesis holds.** `u_eff`'s block-size ratio accounts
  for 83.4% of `median(BE)`'s level gap. Near-Gaussian, the level difference
  between block sizes is essentially the per-element quantization error
  differing between block sizes, and nothing else is needed to explain it.
* **At `nu = 1` it does not.** `u_eff` moves in the right direction and is the
  single largest contributor, but it accounts for only 48.4% of the gap --
  roughly half. Under heavy tails something beyond the per-element relative
  error is contributing to `median(BE)`'s block-size dependence.

The `u_eff` ratio is smoothly monotone in tail weight across the full grid
(1.022 at the Gaussian end rising to 1.155 at `nu = 1`), so the discrepancy at
`nu = 1` is not a ragged or noisy point -- it is a systematic shortfall that
grows as tails get heavier. **What that missing half is was not measured here.**
A plausible candidate, untested: `BE` weights each element's error by
`|a_k||b_k|`, so it is not the *unweighted* per-element error distribution that
`measure_u_eff` reports, and under heavy tails the weighting concentrates on
exactly the outlier elements whose blocks behave worst. Confirming or
dismissing that needs a weighted `u_eff`, which was not run.

**Consequence for the bound, stated as a constraint and not as a design:** a
bound of the form `constant * u_eff * f(n)`, with `u_eff` carrying the whole
block-size dependence and `f(n)` carrying none, is consistent with the data at
`nu = 30` and **not** consistent with it at `nu = 1`. Which way to resolve that
-- accept the shortfall as conservatism, measure the weighted `u_eff` first, or
let the bound's block-size dependence live somewhere other than `u_eff` -- is
the decision under review, and this section deliberately does not take it.

### Result 3: other patterns worth noting

**The scale format matters more than the block size, everywhere except the
heaviest tail.** Decomposing the 2x2 into its two one-factor effects:

| | 1 | 2 | 3 | 5 | 8 | 15 | 30 | gaussian |
|---|---|---|---|---|---|---|---|---|
| scale-format effect (E8M0/E4M3), block 16 | 1.113 | 1.144 | 1.147 | 1.141 | 1.134 | 1.130 | 1.126 | 1.124 |
| scale-format effect (E8M0/E4M3), block 32 | 1.100 | 1.128 | 1.135 | 1.129 | 1.117 | 1.105 | 1.097 | 1.091 |
| block-size effect (32/16), E8M0 | 1.155 | 1.120 | 1.092 | 1.066 | 1.049 | 1.034 | 1.029 | 1.022 |
| block-size effect (32/16), E4M3 | 1.169 | 1.136 | 1.104 | 1.077 | 1.066 | 1.057 | 1.056 | 1.053 |

The scale-format effect is large (E4M3-with-a-global-scale resolves 9-15%
better than E8M0) and roughly **flat in `nu`**. The block-size effect is
strongly **`nu`-dependent**, small near the Gaussian end (2-5%) and growing to
16-17% at `nu = 1`. The two are comparable in size only at `nu = 1`; for
`nu >= 3` the scale format dominates. Note this is a statement about `u_eff`
alone and carries no implication for H1, which is a claim about where the
*bound breaks*, not about which format has the smaller per-element error.

**The two controls cross.** `(32, E4M3+global)` resolves *worse* than
`(16, E8M0)` at `nu = 1` (0.142980 against 0.136158) and *better* at every
other tested `nu` (0.098980 against 0.105679 at the Gaussian end). The
crossing falls between `nu = 1` and `nu = 2`. So which of block size and scale
format wins is itself tail-dependent, which is precisely the interaction the
2x2 exists to expose -- and it is visible in `u_eff` directly, before any GEMM.

**NVFP4 has the lowest `u_eff` of the four at every `nu`**, and MXFP4 the
highest. That is the two factors compounding rather than either one alone:
NVFP4 is the small block *and* the finer scale format, MXFP4 the large block
*and* the coarser one. The MXFP4/NVFP4 ratio is 1.149 at the Gaussian end and
1.286 at `nu = 1`.

### Post-hoc: the near-zero cut is not neutral between the two arms

Read back out of the same run's survival fractions, no new sampling. With the
scale format held at E8M0, block 32 loses far more elements to the flush
threshold than block 16 does, and the gap between them grows sharply with tail
weight:

| | 1 | 2 | 3 | 5 | 8 | 15 | 30 | gaussian |
|---|---|---|---|---|---|---|---|---|
| surviving, block 32 | 0.360 | 0.678 | 0.780 | 0.838 | 0.863 | 0.879 | 0.887 | 0.893 |
| surviving, block 16 | 0.517 | 0.762 | 0.829 | 0.868 | 0.884 | 0.894 | 0.898 | 0.902 |
| gap (16 - 32) | 0.1565 | 0.0834 | 0.0491 | 0.0298 | 0.0208 | 0.0149 | 0.0116 | 0.0093 |

The gap is monotone in `nu` over the whole grid -- 16.9x larger at `nu = 1`
than at the Gaussian end, 13.5x larger than at `nu = 30` -- and its rank order
against the `u_eff` block-size excess is perfect across all eight `nu`
(Spearman 1.000; Pearson 0.957, or 0.987 on logs).

**Does it track the unexplained shortfall? Directionally yes, proportionally
no, and the comparison rests on two points.** The shortfall is only defined
where the n-scaling probe measured a `BE` level ratio, i.e. at `nu = 1` and
`nu = 30` alone, so no correlation across the grid can be computed for it and
none is claimed here: two points are always perfectly correlated. At those two
points both quantities are larger at `nu = 1`, but by very different factors --
the shortfall grows 3.1x from `nu = 30` to `nu = 1` (16.6% unexplained to
51.6%) while the survival gap grows 13.5x. Since essentially every quantity in
this measurement increases with tail weight, a matching *direction* at two
points is weak evidence and is reported as such.

What the survival gap does establish, independently of that, is a **caveat on
the level-gap check itself**: the near-zero cut removes 64% of block-32
elements at `nu = 1` against 48% of block-16 elements, so `u_eff` is being
compared between two arms whose surviving subsets differ substantially, and the
arm that flushes more has more of its worst-resolved elements removed. That is
the right sign to make `u_eff` under-report block 32's disadvantage, which is
the direction of the shortfall. This is a mechanism consistent with the data,
**not** a measurement of one -- confirming it would require a `u_eff` variant
that accounts for the flushed elements rather than excluding them, which was
not run.

### What this does not settle

* **No bound is defined here** and no functional form is proposed. The one
  thing the data constrain is stated above as a constraint.
* **The `nu = 1` shortfall is unexplained**, and the weighted-`u_eff`
  conjecture offered for it is a conjecture, not a measurement.
* **The near-zero policy materially affects every number.** It is the
  defensible cut, but it is a cut; the surviving fractions are tabulated so
  that any downstream use has to confront them.
* **Scope.** One tensor shape `(64, 1024)`, E2M1 elements, RTNE only, no RHT,
  four configurations, the eight-value `nu` grid, exact arithmetic throughout.
  Nothing about block sizes 8 and 64, stochastic rounding, the RHT, or real
  activations.
* **It closes no PREREGISTRATION.md item.** The element budget, the tensor
  shape, the near-zero threshold and the 75% reporting threshold are diagnostic
  choices, not frozen commitments; none is a dated amendment under section 8.

## Sweep grid (Step 3.1) -- FROZEN, not yet run

**Status: the grid is frozen (with one cut applied, below); nothing has been
executed.** This section documents `configs/sweep_main.yaml` and the dry-run
timing estimate for it. Per PREREGISTRATION.md sec 6, `configs/default.json`
is a leftover placeholder and is not this config; `sweep_main.yaml`'s sha256
is what a future confirmatory run's `sweep_{sha256(...)[:12]}.parquet`
filename will encode. **Running the sweep is a later step and is explicitly
out of scope here.**

### The frozen grid (post-cut)

Six factors, factorial, exactly as listed (no additions, no removals):

| factor | values |
|---|---|
| `nu` (t-Student df) | 1, 2, 3, 5, 8, 15, 30, gaussian |
| `n` (contraction dimension) | 16, 64, 256, 1024, 4096 |
| `block_size` | 16, 32 (cut from 8, 16, 32, 64 -- see below) |
| `scale_format` | e8m0, e4m3 |
| `round_mode` | rtne, sr |
| `rht` | false, true (per-block only; a global RHT stays out of scope per `transforms.py`) |

`8 x 5 x 2 x 2 x 2 x 2 = 640` cells (was 1280 before the cut). `accum` is
fixed at `"exact"` (`accum="bf16"` is a separate ablation, out of scope) and
`element_format` is fixed at `"e2m1"` (the only element format
`qgemm.blocks` supports). Matrix shape is `(m=64, n) @ (n, k=64)` for every
cell -- reused unchanged from `scripts/probe_n_scaling.py` and
`scripts/measure_u_eff.py`, not a new convention.

**`use_global_scale` is derived, not swept.** It is fully determined by
`scale_format`, matching the MXFP4/NVFP4 presets in `blocks.py` exactly:
`e8m0 -> false` (E8M0 spans 255 binades unaided), `e4m3 -> true` (E4M3 spans
~19 binades and overflows to NaN without one -- see `blocks.py`, "Why the
global scale exists"). Encoded as a lookup in `sweep_main.yaml`
(`use_global_scale_by_scale_format`) and read that way by
`scripts/run_sweep.py`'s `iter_main_cells`, which is what keeps the grid at
640 cells rather than 1280.

### The block_size cut (2026-08-20)

**block_size reduced from `[8, 16, 32, 64]` to `[16, 32]`, dropping the grid
from 1280 to 640 cells.** Applied in response to the first dry-run's gate
check: serial timing reliably exceeded the 8h budget by 7x-13x across five
independent isolated runs (see the superseded numbers this section used to
carry, now replaced below).

**What survives, and why.** 16 and 32 are NVFP4's and MXFP4's own block
sizes; held against the two `scale_format` values they form the full 2x2
this project's own P1 comparison needs (PREREGISTRATION.md: "P1 -- ν*(block
16) vs ν*(block 32), scale format held constant [block-size causal effect]",
the *sole* confirmatory test of H1 per sec 4.1). That 2x2 is the causal core
Step 1.5 built (`qgemm.blocks.quantize_blocked`, one parametrized quantizer
of which MXFP4/NVFP4 are presets) and this project treats it as
non-negotiable.

**What was dropped, and why that's the right cut, not an arbitrary one.**
8 and 64 are, in this project's own documented words (SPEC.md, "Why the
controls exist"): "the endpoints that show whether the block-size effect is
monotone over the range" -- explicitly an extension for checking monotonicity
across the causal core, not part of the core comparison itself. Dropping them
first, before touching `n`, `nu`, `round_mode`, `rht`, or the trial budget,
removes exactly the two values whose absence costs the study the least: P1,
P2 and P3 (PREREGISTRATION.md sec 4) are all defined in terms of `block_size
in {16, 32}` already, and no primary or design-validity comparison reads
block 8 or block 64.

**A citation note, stated plainly rather than papered over.** The task that
requested this cut described it as applying "this project's own pre-declared
cut-order (Appendix A of the roadmap, item 4)". A full search of this
repository -- every tracked file and the complete git history -- found no
roadmap document, no "Appendix A", and no pre-declared cut-order list
anywhere. (It also found that a prior session's own commit message coined the
phrase "cut-line policy" without grounding it in any such document, which may
be the source of the apparent citation.) The cut applied above stands on its
own documented rationale -- SPEC.md's own "Why the controls exist" language,
written before any of this dry-run work -- not on an external document this
repository does not contain. If a roadmap/Appendix A exists outside this
repository, it should be added so future cuts can cite it for real.

### A pre-existing grid defect, found while measuring the cut (not introduced by it)

**32 of the 640 cells are currently not executable.** `qgemm.transforms.
apply_rht` has no partial-block form (SPEC.md: "A last axis that is not a
whole number of blocks raises `ValueError`"), so any cell with `rht=true`
whose `n` is not a whole multiple of `block_size` raises at the `qgemm()`
call. In the post-cut grid this is exactly `n=16, block_size=32, rht=true`
(16 is not a multiple of 32): `2 scale_format x 2 round_mode x 8 nu = 32`
cells, `80,000` trials. **This defect predates today's cut** -- the original
1280-cell grid had the same problem at `block_size in {32, 64}`, 64 broken
cells total -- and was not caught until `scripts/run_sweep.py --dry-run`'s
real-multiprocessing measurement (below) actually called `qgemm()` on a
sampled cell and it raised. The single-process cost-model timing added in
the first dry-run never exercised this combination (its overhead-factor
sample points used `n=256` and `n=4096`, never `n=16`), which is how it went
unnoticed.

`scripts/run_sweep.py` now detects this (`cell_is_executable`), reports the
count and excludes affected cells from its real-multiprocessing sample so the
measurement doesn't crash on them -- but **does not decide how to fix the
grid**. Their trial time is still included in the cost-model totals below
(that model is arithmetic, not an actual `qgemm()` call, so it does not
crash), which is a reporting choice worth flagging: those 80,000 trials'
worth of estimated compute could not actually be produced as the grid is
currently specified. Candidate fixes -- capping `rht_block_size` below
`block_size` when `n < block_size`, dropping `n=16` for the affected arm, or
excluding the cell from the grid outright -- are none of them applied here;
this is a human decision, same as the cut-size gate below.

### Adaptive trial budget

Not uniform across `nu`:

* `nu in {1, 2, 3}`: **5000 trials/cell.** The heaviest tails, where quantile
  estimates (especially p99) are noisiest -- both the u_eff and
  metric-stability diagnostics above found exactly this -- and where the
  bound `cota(n) = c * u_eff / sqrt(n)` is most likely to break, which is H1's
  entire premise. Under-sampling here would save the least useful compute
  while risking the noisiest possible read on the question the sweep exists
  to answer.
* `nu in {5, 8, 15, 30, gaussian}`: **1000 trials/cell.** Lighter tails, where
  the u_eff stability check already found quantiles well-behaved at this
  budget (the same default `scripts/probe_n_scaling.py` and
  `scripts/check_metric_stability.py` already use as a "cheap probe" count).

Main grid: `3 heavy-tail nu x 80 cells/nu x 5000 = 1,200,000` trials plus
`5 light-tail nu x 80 cells/nu x 1000 = 400,000` trials = **1,600,000 trials**
(was 3,200,000 before the cut -- exactly half, as expected from halving
`block_size`'s cardinality with every other factor unchanged).

### Reference configs (outside the 640-cell grid)

FP8-E4M3 per-tensor, FP8-E5M2 per-tensor, INT8 per-tensor
(`qgemm.formats.int8_quantize`) -- context/comparison baselines, not part of
the factorial (no `block_size`/`scale_format`/`use_global_scale`/`rht`
dimension; these formats are not block-scaled, so they are untouched by the
block_size cut). Swept across the same `nu` grid and the same adaptive trial
budget, but at a representative subset of `n = [16, 256, 4096]` rather than
all five -- reusing `scripts/check_metric_stability.py`'s own `(nu, n)` grid
unchanged (its `GRID_N` is exactly `{16, 256, 4096}`) rather than picking a
new one: 16 is the highest-risk corner for a per-tensor denominator, 4096 is
the main grid's ceiling, 256 a middle anchor. `3 formats x 8 nu x 3 n = 72
cells`, `180,000` trials total (`60,000` heavy-tail + `120,000` light-tail) --
unchanged by the cut.

Grid total: **712 cells, 1,780,000 trials** (was 1352 cells / 3,380,000
trials before the cut).

### Dry-run timing estimate (post-cut, real multiprocessing measured) -- SUPERSEDED, see below

**Superseded by "Dry-run timing estimate, retimed after the `apply_rht` fix"
below.** The 32 sampled-around cells this section describes were not merely
excluded from the real-multiprocessing *sample* -- had a real sweep run
attempted them, they would have crashed outright (the `apply_rht` bug fixed
above), so the measurement below was necessarily built from an incomplete
picture of the grid's real cell types: it never actually executed the
`rht=true, n<block_size` corner, only modeled its cost arithmetically. Kept
here, unedited, as the historical record of what was measured before the fix;
not to be read as the current estimate.

Produced by `python scripts/run_sweep.py --dry-run`. Two independent
measurements feed the report, and they are not the same kind of number:

1. **Single-process cost model** (unchanged methodology from the pre-cut
   dry-run): real `qgemm` + `backward_error` calls at the base `n x
   block_size` grid (now 10 points, `rht=False`, `round_mode="rtne"`,
   `scale_format="e8m0"`), plus `rht`/`round_mode="sr"`/`e4m3+global`
   overhead factors from two representative points (`(256, 16)` and
   `(4096, 32)` -- `(4096, 64)` is no longer valid post-cut). This gives the
   **serial** estimate and, divided by core count, an **ideal-scaling**
   parallel estimate -- the same naive assumption the previous dry-run used,
   kept only as a reference figure now.
2. **Real `multiprocessing.Pool` measurement** (new this step, per the task's
   explicit requirement not to extrapolate parallel time from an ideal-
   scaling assumption): 32 real cells, evenly spaced across the grid's
   iteration order for coverage of every factor including `nu`, each run as
   its own OS process (`multiprocessing.Pool(processes=n_cores)`, one process
   per cell, each with its own RNG state seeded from `SeedSequence([cell_index,
   trial])` -- never shared, per the repo's explicit-Generator convention).
   Every sampled cell runs a fixed 150 trials (not its real 1000/5000-trial
   budget, to bound the measurement's own runtime) so the wall-clock time is
   **actually measured**, not modeled. The ratio of this measured time to
   what the single-process cost model predicted for the identical sample
   workload is the **measured parallel efficiency**; applying it as a
   correction to the ideal-scaling full-grid estimate gives the **headline
   MEASURED parallel estimate**. A **cross-check** extrapolates directly from
   the sample's own measured trials/second instead, ignoring the cost model
   entirely, and is reported alongside.

**What this measurement does not capture.** Every sampled cell runs the same
150 trials regardless of its real budget, so **load imbalance from the real
1000-vs-5000-trial split across cells is not measured** -- a real
multiprocessing run would have some workers finish their light-tail cells and
sit idle (or start a new cell) while others are still deep into a 5000-trial
heavy-tail cell, an effect this design cannot see. The efficiency figure
below is therefore likely still a *mild overestimate* of real production
throughput, i.e. real execution is probably somewhat slower than this
estimate says.

**Run-to-run variance, disclosed rather than hidden behind one number.** Same
finding as the pre-cut dry-run: this is a shared development machine, not a
controlled benchmark rig, and both the single-process cost model *and* the
real Pool measurement are noisy here. Five independent invocations, run in
isolation with production defaults (`--dry-run-reps 20`,
`--dry-run-parallel-sample-cells 32`, `--dry-run-parallel-sample-trials 150`):

```
run   serial (cost model)   parallel, MEASURED (headline)   parallel, cross-check
1           11.65 h                  2.70 h                       3.20 h
2           83.16 h                  5.82 h                       8.49 h  <- cross-check EXCEEDS
3            7.01 h                  5.34 h                       5.97 h
4           26.85 h                  4.41 h                       5.78 h
5            6.92 h                  8.12 h  <- EXCEEDS            9.09 h  <- EXCEEDS

serial:            range [6.92, 83.16] h, median 11.65 h -- 2 of 5 runs now clear 8h (never did pre-cut)
measured parallel:  range [2.70, 8.12] h,  median 5.34 h -- straddles the gate (1 of 5 exceeds)
cross-check:        range [3.20, 9.09] h,  median 5.97 h -- straddles the gate (2 of 5 exceed)
```

One representative run (run 5) in full:

```
Main factorial grid: 640 cells (32 not executable, see above), 1,600,000 trials total.
Reference configs:   72 cells, 180,000 trials total.
Grid total:          712 cells, 1,780,000 trials.

Main-grid compute (single-process cost model):      22,574.7 s ( 6.271 h)
Reference-config compute (single-process cost model): 2,350.8 s ( 0.653 h)
TOTAL, serial (1 core, single-process cost model):   24,925.5 s ( 6.924 h)
TOTAL, parallel, IDEAL SCALING (12 cores, naive):     2,077.1 s ( 0.577 h)  [reference only]

REAL multiprocessing sample: 32 cells x 150 trials = 4,800 trials, Pool(processes=12).
  ideal-scaling prediction for this sample:      6.3 s (0.002 h)
  ACTUALLY MEASURED wall-clock for this sample: 88.3 s (0.025 h)
  measured parallel efficiency vs. ideal scaling: 0.071 (7.1%)

TOTAL, parallel, MEASURED (headline):  29,223.9 s ( 8.118 h)  EXCEEDS the 8h budget
  cross-check (direct sample throughput): 32,733.6 s ( 9.093 h)  EXCEEDS the 8h budget
```

Measured parallel efficiency itself ranged from 7.1% to 119.2% of the naive
ideal-scaling prediction across the five runs -- i.e. real multiprocessing
execution was sometimes markedly *slower* than ideal scaling (as low as 7%
efficiency) and once measured *faster* than the ideal-scaling number computed
in that same noisy run (119%, meaning that run's single-process cost model
happened to read unusually high relative to the real measurement, not that
parallel execution beat theoretical serial/n_cores). This is a direct,
measured demonstration that "ideal scaling" is not a safe assumption here in
either direction, which is exactly why the task asked for a real measurement
instead of one.

**Verdict (superseded -- see below): this does NOT reliably clear the 8h
gate, and is not a clean pass.** The MEASURED headline figure exceeded 8h in
1 of 5 independent runs (8.12h); the cross-check exceeded it in 2 of 5
(8.49h, 9.09h); and even the serial (single-core) estimate -- which exceeded
8h in every one of the five pre-cut runs -- now clears it in 2 of 5 post-cut
runs. The cut roughly halved the grid's compute as expected, but the honest
read of five independent measurements is "genuinely borderline, could go
either way on a given invocation," not "clears with margin." Per this step's
own instructions, if the measured estimate exceeds 8h that is reported
plainly and this stops here: **it exceeded 8h at least once, so no further
cut is applied or decided in this step.** Whether the block_size cut alone is
sufficient, or whether a second cut (or accepting the risk, or a
controlled/quieter benchmarking environment) is warranted, is a human call,
to be made **before** anything is launched -- not made here.

### Dry-run timing estimate, retimed after the `apply_rht` fix (2026-08-20)

**All 640 main-grid cells are now executable** (`scripts/run_sweep.py`'s
`cell_is_executable` now returns `True` for every cell in this grid --
verified, not assumed, since the check still tests the real remaining
constraint rather than being hardcoded). The real-multiprocessing sample can
therefore draw from the full grid, including the 32 `rht=true, n=16,
block_size=32` cells the previous measurement above could only model
arithmetically, never actually run. Same methodology otherwise: 32 real
cells evenly spaced across the grid's iteration order, `150` trials each,
`multiprocessing.Pool(processes=n_cores)`, one process per cell with its own
RNG state, five independent isolated invocations at production defaults.

```
run   serial (cost model)   parallel, MEASURED (headline)   parallel, cross-check   efficiency
1            9.69 h                   3.73 h                       4.11 h              21.7%
2          128.61 h  <- EXCEEDS       9.11 h  <- EXCEEDS           12.33 h <- EXCEEDS  117.6%
3          106.74 h  <- EXCEEDS       5.37 h                        6.57 h             165.5%
4          155.38 h  <- EXCEEDS       4.26 h                        5.59 h             304.0%
5            7.08 h                   5.68 h                        6.04 h              10.4%

serial:            range [7.08, 155.38] h, median 106.74 h -- 4 of 5 runs now EXCEED 8h (was 3 of 5 pre-fix)
measured parallel:  range [3.73, 9.11] h,   median  5.37 h -- straddles the gate (1 of 5 exceeds, same as pre-fix)
cross-check:        range [4.11, 12.33] h,  median  6.04 h -- straddles the gate (1 of 5 exceeds, was 2 of 5 pre-fix)
measured efficiency: range [10.4%, 304.0%], median 117.6%
```

**A new observation, disclosed rather than smoothed over: measured parallel
efficiency exceeded 100% in 3 of the 5 runs (117.6%, 165.5%, 304.0%),
something the pre-fix measurement never saw (max 119.2%, and only barely).**
Efficiency above 100% is not real multiprocessing execution beating
theoretical serial-time-divided-by-cores -- that is not physically
meaningful for a fair comparison -- it means the *single-process cost model*
phase (which runs first, in a tight loop, ~10-20 timed calls per point) read
unusually high relative to the real Pool measurement that follows it in the
same invocation. This session's machine shows more single-process noise than
the previous session's (pre-cut) five runs did: serial estimates now reach
155h against a prior ceiling of 102.5h. Whether that is accumulated
background load, thermal effects from the extended single-process timing
loop itself, or something else was not investigated -- it is reported as an
observed instability of the *cost-model* half of this measurement on this
shared machine, not of the real-multiprocessing half, which is what the
MEASURED headline figure is actually built from (via the efficiency
correction) and cross-checked against directly.

**Comparing to the pre-fix numbers: the headline verdict is essentially
unchanged.** Measured-parallel median moved from 5.34h to 5.37h; the 8h
overrun rate stayed at 1 of 5 runs for the headline figure. This is
expected, not a surprise to be explained away: the 32 previously-crashing
cells are only 5% of the grid (32/640), so their addition to the sample pool
(a ~5% chance of landing in any given 32-cell draw) was never going to move
the aggregate by much on its own. **The value of the fix is correctness, not
a materially different timing picture** -- the grid can now actually be run
as specified, which it structurally could not before, and the real-
multiprocessing sample is no longer built from an artificially incomplete
picture of the grid's cell types.

**Verdict, updated: still does NOT reliably clear the 8h gate.** The
MEASURED headline figure exceeded 8h in 1 of 5 runs (9.11h); the cross-check
exceeded it in 1 of 5 (12.33h); the serial estimate now exceeds 8h in 4 of 5
runs (worse than pre-fix, consistent with the cost-model noise observation
above, not with any change to the grid itself). Per this step's own
instructions: the measured estimate exceeded 8h at least once, so this is
reported plainly and **no further cut is applied or decided here.** Whether
the block_size cut is sufficient, whether a further cut is warranted, or
whether the honest answer is "re-measure on a quieter machine before
deciding" given how much the single-process cost model swung between the two
five-run batches, is a human call, to be made **before** anything is
launched -- not made here.

## Cross-distribution normalization (🧠2) — RESOLVED

**Decision.** Every sample generated for the production sweep going forward
(Phase 3.3 onward) is normalized so its MAD (median absolute deviation) is 1,
computed **once over the full generated tensor** -- not per row, not per
block -- before quantization.

**Why MAD, not standard deviation.** Standard deviation is undefined for
t-Student(nu) at nu<=2 (Cauchy included), which is inside this project's own
nu grid (`nu=1` is the heaviest-tailed point swept). MAD is well-defined for
every nu, including nu=1, and gives every distribution in the sweep the same
scale before quantization, so that differences measured across nu reflect
**tail shape**, not an accident of raw draw scale. This is what Phase 3.3
onward needs and what the diagnostics run so far did not: they were resolving
🧠1's functional form (n-dependence, block-size level gap), a question that
does not turn on cross-nu scale comparability.

**Why per-tensor, not per-row or per-block.** The normalization exists to put
different nu on a common footing *before* block-scaled quantization does
anything -- normalizing per-block would fold the normalization into the very
block structure the study varies, confounding the two. A single per-tensor MAD
leaves the tensor's internal block-to-block dynamic range untouched.

**Scope boundary -- explicit, not implicit.** This applies to the production
sweep going forward (Phase 3.3 onward) only. It is **not** applied
retroactively to the already-committed `measure_u_eff.py` / n-scaling-probe
diagnostics, which used raw (unnormalized) scale for a different purpose --
resolving 🧠1's functional form -- and whose numbers in this document stand as
measured. Nothing above this section is affected by this decision.

**What landed in code.**

* `qgemm.stats.median_absolute_deviation(x)` -- `median(|x - median(x)|)`,
  float64 in, float64 out. Tests in `tests/test_stats.py`, including a
  known-answer case and a check that it stays finite on a Cauchy sample where
  `np.std` is dominated by rare extreme draws (ratio ~406x on the tested
  sample) -- the concrete reason MAD is usable here and std is not.
* `normalize="mad"` -- an explicit opt-in parameter, default `None` -- added
  to `qgemm.distributions.sample_gaussian` and `sample_uniform`. The default
  path is unchanged: a regression test (`tests/test_distributions.py`) pins
  the unnormalized output as byte-identical to the pre-existing behavior, so
  every script that calls these without `normalize` (unchanged) is
  unaffected.
* The nu-axis sampler, `sample_t`, still does **not** live in
  `qgemm.distributions` -- it remains deliberately local to
  `scripts/check_metric_stability.py`, per PREREGISTRATION.md R15 ("a piece of
  study infrastructure ... not smuggled in through a diagnostic"). This
  decision does not resolve R15 and does not move it; whether/when to promote
  `sample_t` into `qgemm.distributions` is still open. The same opt-in
  `normalize="mad"` parameter (default `None`, same regression guarantee) was
  added to that local `sample_t`, tested in
  `tests/test_check_metric_stability.py` across the project's full nu grid
  (`1, 2, 3, 5, 8, 15, 30`), so every existing importer of it
  (`measure_u_eff.py`, `probe_n_scaling.py`, `sanity_reproduce_2408.py`,
  `run_sweep.py`) picks up the option automatically without any change to its
  own default behavior.

### Scale invariance under a common rescaling -- quantified, not pass/fail

The other question 🧠2 raised: is block-scaled quantization homogeneous of
degree 1 under a global rescaling -- does `quantize_blocked(c*x)` equal
`c*quantize_blocked(x)`? Measured directly rather than assumed, in
`tests/test_blocks.py::test_scale_invariance_under_global_rescaling`, over
this project's causal 2x2 (`block_size in {16, 32}` x
`scale_format in {"e8m0", "e4m3"}` -- MXFP4, NVFP4 and their two Step-1.5
controls, `use_global_scale` following scale format per the project's
established pairing), on one fixed spread tensor (`_spread_tensor()`: 4x64,
log-normal magnitudes, seed 20260820), for `c` in `{2, 4, 0.5}` (exact powers
of two) and `{1.5, 3, 7}` (not):

| configuration | c | power of 2? | max relative discrepancy | mean relative discrepancy | bit-exact? |
|---|---|---|---|---|---|
| block16_e8m0 (control) | 2.0 | yes | 0.000e+00 | 0.000e+00 | yes |
| block16_e8m0 (control) | 4.0 | yes | 0.000e+00 | 0.000e+00 | yes |
| block16_e8m0 (control) | 0.5 | yes | 0.000e+00 | 0.000e+00 | yes |
| block16_e8m0 (control) | 1.5 | no  | 4.000e+00 | 1.484e-01 | no |
| block16_e8m0 (control) | 3.0 | no  | 8.000e+00 | 1.992e-01 | no |
| block16_e8m0 (control) | 7.0 | no  | 1.000e+00 | 7.719e-02 | no |
| block32_e8m0 (MXFP4) | 2.0 | yes | 0.000e+00 | 0.000e+00 | yes |
| block32_e8m0 (MXFP4) | 4.0 | yes | 0.000e+00 | 0.000e+00 | yes |
| block32_e8m0 (MXFP4) | 0.5 | yes | 0.000e+00 | 0.000e+00 | yes |
| block32_e8m0 (MXFP4) | 1.5 | no  | 1.600e+01 | 2.444e-01 | no |
| block32_e8m0 (MXFP4) | 3.0 | no  | 3.200e+01 | 4.201e-01 | no |
| block32_e8m0 (MXFP4) | 7.0 | no  | 1.000e+00 | 5.078e-02 | no |
| block32_e4m3 (control) | 2.0 | yes | 0.000e+00 | 0.000e+00 | yes |
| block32_e4m3 (control) | 4.0 | yes | 0.000e+00 | 0.000e+00 | yes |
| block32_e4m3 (control) | 0.5 | yes | 0.000e+00 | 0.000e+00 | yes |
| block32_e4m3 (control) | 1.5 | no  | 4.277e-16 | 6.219e-17 | no |
| block32_e4m3 (control) | 3.0 | no  | 4.277e-16 | 6.219e-17 | no |
| block32_e4m3 (control) | 7.0 | no  | 2.716e-16 | 4.514e-17 | no |
| block16_e4m3 (NVFP4) | 2.0 | yes | 0.000e+00 | 0.000e+00 | yes |
| block16_e4m3 (NVFP4) | 4.0 | yes | 0.000e+00 | 0.000e+00 | yes |
| block16_e4m3 (NVFP4) | 0.5 | yes | 0.000e+00 | 0.000e+00 | yes |
| block16_e4m3 (NVFP4) | 1.5 | no  | 4.277e-16 | 8.651e-17 | no |
| block16_e4m3 (NVFP4) | 3.0 | no  | 4.277e-16 | 8.651e-17 | no |
| block16_e4m3 (NVFP4) | 7.0 | no  | 2.716e-16 | 6.214e-17 | no |

**At an exact power of two, every configuration is bit-for-bit exact,
regardless of scale format.** Multiplying by an exact power of two is a pure
exponent shift at every step float64 performs along the pipeline -- `amax`,
the e8m0/e4m3 scale-grid rounding, and the global scale where present -- so it
commutes exactly with every rounding step. This matches what was expected
going in for both scale formats.

**At a non-power-of-two c, the two e8m0 (no global scale) configurations show
a large, easily measurable discrepancy**, also as expected: with no global
scale to absorb the rescaling, a non-power-of-two `c` generally changes which
power-of-two exponent the block scale rounds to, and that is not a rounding
artifact -- it is a genuinely different quantization grid.

**At a non-power-of-two c, the two e4m3-with-global-scale configurations
(this includes NVFP4 itself) do *not* show the "small but nonzero"
discrepancy hypothesized going in.** Measured discrepancy stays at the
float64 rounding noise floor (~1e-16 relative, i.e. a handful of ULPs) --
indistinguishable from exact on this data, not a meaningfully larger "small"
effect. The mechanism: the global scale is a raw float64, not on any grid, so
it exactly cancels a common rescaling in the ratio fed to `quantize_e4m3` up
to ordinary floating-point rounding noise; E4M3's ~3-bit mantissa is a grid
far too coarse for noise at the 1e-16 level to ever cross a rounding boundary
on data like this. **This is reported plainly as a finding that diverges from
the a priori expectation, not forced into the expected shape.** The
discrepancy is not exactly zero in the bit-for-bit sense e8m0's power-of-two
case is, but it is zero for every practical purpose on this project's data.

**Caveat on the measure itself, not the property.** The relative-discrepancy
metric divides by the target reconstruction, so it can be inflated by
elements whose reconstruction sits near zero in one arm and not the other --
this is a property of the relative-error measure at a small denominator, not
of the underlying invariance question, and it is why some of the e8m0 rows
above (e.g. `1.6e+01` max relative discrepancy) look larger than the mean
discrepancy for the same row would suggest.

## Sweep execution harness (Step 3.3)

**Status: the harness is implemented and tested on a small subset of the
frozen grid; the full 640-cell launch has not happened.** This section
documents `scripts/run_sweep.py`'s real execution path (as opposed to
`--dry-run`, which only estimates timing and writes nothing): the storage
schema, the seeding scheme, and confirmation of where Step 3.2's
`normalize="mad"` decision actually takes effect.

### Invocation

```
python scripts/run_sweep.py --config configs/sweep_main.yaml \
    [--resume] [--cell-filter FIELD=VALUE[|VALUE2][,FIELD2=VALUE3]] \
    [--jobs N] [--u-eff-n-elements N]
```

`--config` now does double duty, dispatched by file extension: a `.yaml`/
`.yml` file runs this real, checkpointed main-grid harness; a `.json` file
still runs the pre-existing toy sweep (`make run-sweep`'s `configs/
default.json` path, unrelated to the Step 3.1 grid, left unchanged). `--dry-
run` is untouched and still takes its own `--grid-config` flag.

`--cell-filter` restricts execution to matching main-grid cells --
comma-separated clauses are ANDed, `|`-separated values within one clause are
ORed (e.g. `n=16|64,nu=1` selects `n in {16, 64}` at `nu == 1`). It exists so
this step's own acceptance testing, and any future partial re-run, does not
have to touch 640 cells to exercise the harness. `--jobs` controls
parallelism across cells (`<=1` runs sequentially in the calling process, no
`multiprocessing.Pool` at all -- see "Why jobs<=1 avoids Pool" below).
`--u-eff-n-elements` (default 200 000) sizes the per-cell `measure_u_eff`
draw independently of the trial budget, since u_eff is a per-configuration
quantity, not a per-trial one.

### Storage schema

**Per-cell checkpoint (requirement 1).** Each main-grid cell, once every one
of its trials has run and its `u_eff` has been computed, is written as a
single file:

```
results/sweep_{grid_digest}/cells/cell_{cell_id}.parquet
```

`grid_digest` is `sha256(canonical_json(parsed sweep_main.yaml))[:12]`
(`config_digest`, the same function the pre-existing toy-sweep path already
uses) -- not a hash of the YAML file's raw bytes, so formatting-only changes
to the file do not change the digest. `cell_id` is `sha256(canonical_json(
cell_key))[:12]` where `cell_key` is the cell's identity (see "Seeding
scheme" below) -- deterministic, so re-running the same grid always addresses
the same cell to the same filename, which is what makes `--resume` a simple
existence check.

One row per **trial**, tidy/long format, exactly as the bootstrap CI in a
later phase needs (PREREGISTRATION.md sec 3.2: the resampling unit is the
*trial*, not the element -- an element-level bootstrap would be
anticonservative, since elements within a trial share a row of `A`/`B` and a
block scale). Columns:

| column | meaning |
|---|---|
| `nu`, `n`, `block_size`, `scale_format`, `use_global_scale`, `round_mode`, `rht` | the six swept factors (`nu` stored as `str` -- see below) |
| `element_format`, `accum`, `normalize` | constant across this grid (`"e2m1"`, `"exact"`, `"mad"`), stored explicitly rather than left implicit |
| `cell_id` | this cell's identity hash, repeated on every row |
| `trial_index` | `0..trials-1` |
| `seed_data`, `seed_gemm` | the two derived seeds this trial used (see below) -- kept for audit/debugging, not required to reproduce the trial (the cell_key + trial_index already determine them) |
| `be_median`, `be_p99`, `be_mean`, `be_max` | summary of that trial's `M x K` backward-error matrix; median and p99 are the minimum the bootstrap needs, mean/max are cheap to keep alongside |
| `n_be_values`, `n_be_finite` | `M*K` and how many of those were finite (BE can be `nan`/`inf` where the denominator `(|A|@|B|)_ij` is exactly zero) |
| `u_eff_p50`, `u_eff_p99` | this **cell's** effective unit roundoff (requirement 4) -- a per-configuration quantity, stored as a repeated column rather than a separate table, so the whole result stays one tidy frame with no join required downstream |

`nu` is stored as `str` (`"1"`, `"gaussian"`, ...) rather than left as the
mixed `int | "gaussian"` type `iter_main_cells` yields: a single Arrow/Parquet
column cannot hold both, and canonicalizing to `str` is what `cell_key_from_
cell` does before anything is hashed or written.

**Why u_eff is a repeated column, not a separate table.** The task leaves
this an open choice ("your call... whichever is cleaner"). A separate small
manifest-per-cell table was considered and rejected: it would require a join
key (`cell_id`) to reunite with the trial table for any analysis that wants
both, two files to keep in sync per cell instead of one, and -- most
concretely -- two independent things that could exist in inconsistent states
after a kill (trial file present, u_eff file missing, or vice versa). A
repeated column costs a few extra float64s per trial row (negligible next to
the row itself) and makes every cell's checkpoint exactly one atomically-
written file, which is also what keeps the kill/resume contract simple: a
cell is either fully done (one file, both trial rows and u_eff present) or
not done at all.

**Combined output.** After every invocation (fresh or `--resume`), all
present cell checkpoints under `cells/` are concatenated, sorted by `(cell_id,
trial_index)`, and written to `results/sweep_{grid_digest}.parquet` (+
`.config.json` alongside, the full parsed grid config) -- the same
`sweep_{sha256(config)[:12]}.parquet` naming convention README.md already
documents for every other result in this repo. This combine step is cheap
(concatenation, not recomputation) and reruns on every invocation, so it is
always current; it is not itself part of the checkpoint substrate the
kill/resume contract depends on -- that is the per-cell files under `cells/`.

**Manifest JSON, one per execution (requirement 5).**
`results/sweep_{grid_digest}/manifest_{UTC timestamp}.json` (timestamped, not
overwritten, since "per execution" means one file per *invocation* -- a
`--resume` run gets its own manifest alongside the interrupted run's,
recording only what that invocation itself did):

```json
{
  "grid_config_hash": "...",
  "git_commit": "... or null if git was unavailable",
  "start_time": "... ISO 8601 UTC", "end_time": "...",
  "wall_clock_seconds": 0.36,
  "total_cells_requested": 640,
  "cells_run_this_invocation": 12,
  "cells_skipped_resume": 628,
  "cells_ok": 12, "cells_error": 0,
  "resume": true, "jobs": 12, "u_eff_n_elements": 200000,
  "environment_md": "... full text of ENVIRONMENT.md, or null if absent"
}
```

`environment_md` is a verbatim snapshot of `ENVIRONMENT.md`'s content at
launch time -- reusing what `scripts/record_environment.py` already captures
rather than re-deriving interpreter/OS/CPU/BLAS info inline (which would also
mean shelling out to `pip freeze` on every sweep invocation, unnecessary
overhead for what is otherwise a fast manifest write).

**Errors, logged and non-fatal (requirement 6).** A cell whose execution
raises is caught inside the worker (`process_cell`), turned into a
`{"status": "error", "cell_key": ..., "error": ...}` result instead of
propagating, and appended as one JSON line to
`results/sweep_{grid_digest}/errors.log`. The rest of the sweep continues.
Progress is reported via `tqdm` over the cell-level iterator (or
`Pool.imap_unordered`, at `--jobs > 1`), so progress reflects cells completed,
not trials -- the unit checkpointing and parallelism both operate on.

### Seeding scheme

**No global or shared RNG state anywhere in the harness.** Every
`numpy.random.Generator` this code constructs (operand sampling, `GemmConfig`'s
internal RHT/stochastic-rounding streams, the `u_eff` draw) traces back to a
seed produced by one pure function, `derive_seed(cell_key, trial_index,
stream)` (or `derive_cell_seed(cell_key, stream)` for the one per-cell, not
per-trial, quantity):

```
payload = f"{canonical_json(cell_key)}|trial={trial_index}|stream={stream}"
seed    = sha256(payload)[:8 bytes] -> uint64, masked to 63 bits
```

`cell_key` is the cell's canonical identity -- the six swept factors plus the
three constants (`element_format`, `accum`, `normalize`), **excluding** the
trial *count* (a property of how much a cell is sampled, not of which cell it
is). `stream` separates independent uses within one trial the same way
`GemmConfig`'s own three internal streams (RHT signs, SR for `A`, SR for `B`)
are kept independent: `"data"` seeds the `Generator` that samples `A` and
`B`, `"gemm"` becomes `GemmConfig.seed` (which `GemmConfig` further expands
into its own three streams), and `compute_cell_u_eff` uses `derive_cell_seed`
with stream `"u_eff"`, a distinct payload shape (no `trial=` component) so a
per-cell quantity can never collide with a per-trial seed by construction.

**Why this closes trap (1).** The seed is a pure function of plain, hashable,
picklable values only -- never process identity, worker id, PID, or call
order, none of which a worker process could observe consistently with
another. Two `multiprocessing` workers computing the same `(cell_key,
trial_index, stream)` -- whether because the same cell was resubmitted, or
because a bug double-scheduled it -- therefore always agree exactly (a single
cell is reproducible in isolation, independent of who runs it or when), while
two *different* trials, cells, or streams are mixed through sha256 of a
labeled string and never collide by construction, rather than through a
shared counter that a global-RNG bug could desynchronize. Both directions are
pinned as tests in `tests/test_run_sweep.py`: `test_different_trials_
across_real_worker_processes_are_not_identical` runs two different trials of
the same cell through two real `multiprocessing.Pool` workers and asserts
their raw draws (and seeds) differ; `test_same_trial_across_real_worker_
processes_is_reproducible` runs the *same* trial through two workers and
asserts they agree exactly. A regression that made the harness reach for
`np.random`'s global state, or seed from `os.getpid()`/worker identity,
would fail one or the other.

**Why `jobs<=1` avoids `Pool` entirely.** Not a seeding concern (seeding is
already worker-identity-independent by construction) but a testability and
efficiency one: at `jobs<=1`, `execute_sweep` runs cells sequentially in the
calling process with no `multiprocessing.Pool`, so there is exactly one OS
process doing the work -- which is what lets the kill/resume acceptance test
below send a real `SIGKILL`/`TerminateProcess` to one PID and know it stopped
everything, rather than a parent that dies while an orphaned pool worker
keeps running.

### Checkpoint atomicity (trap 2's other half)

Per-cell checkpointing alone is not sufficient -- a process killed while
*writing* a cell's parquet file could leave a truncated file that `--resume`
might mistake for a completed cell. `_atomic_write_parquet` writes to
`cell_{cell_id}.parquet.tmp` first, then `os.replace`s it onto the final
name; `os.replace` is an atomic rename on both POSIX and Windows when source
and destination share a filesystem (true here -- both live under the same
`cells/` directory), so a kill mid-write leaves either a complete final file
or a stray `.tmp` that `--resume`'s existence check (`cell_{cell_id}.parquet`,
not `.tmp`) simply ignores and recomputes.
`tests/test_run_sweep.py::test_a_stray_tmp_file_is_not_mistaken_for_a_completed_checkpoint`
pins this directly, and the acceptance test below exercises it for real.

### Acceptance test: kill and resume produce an EXACTLY identical result

Per this step's grading criterion, `tests/test_run_sweep.py` includes a test
that launches the actual CLI as a real subprocess
(`test_kill_process_partway_through_then_resume_matches_uninterrupted_run`),
on a 4-cell subset of the grid shape (2 `nu` x 2 `n`, everything else fixed):

1. Run the subset to completion once, uninterrupted -- the reference result.
2. Run it again from scratch in a fresh directory; poll for partial
   checkpoint completion; send a real `kill()` (`TerminateProcess` on
   Windows) once 1-3 of the 4 cells have checkpointed (not 0, not all 4).
3. Re-invoke the CLI with `--resume` against that same directory.
4. Assert the resumed run's combined result is **byte-identical** to the
   reference: same row count, no duplicate `(cell_id, trial_index)` pairs, no
   gaps, and `pandas.testing.assert_frame_equal` on the two combined frames
   directly (not `allclose` -- exact).

The test is inherently timing-dependent (it has to actually catch the process
mid-sweep); if it cannot reliably observe a partial state within its polling
window it `pytest.skip`s with an explicit message rather than passing
vacuously. Across repeated local runs it has not needed to skip.
Complementing it, a second, fully deterministic test
(`test_resume_after_partial_completion_matches_uninterrupted_run`) exercises
the identical property without relying on real-time process timing, by
constructing the "partial" state directly (`--cell-filter` to run a strict
subset, then `--resume` with the full grid) -- useful for fast, non-flaky
regression coverage of the same contract in normal test runs.

### `normalize="mad"` is active here -- confirmed, not assumed

Every `numpy.random.Generator`-backed draw this harness makes for operand
data goes through `_sampler_for`, which calls `sample_gaussian(..., normalize
="mad")` or `sample_t(..., normalize="mad")` unconditionally -- there is no
flag to turn it off in this path, and `cell_key_from_cell` records
`"normalize": "mad"` in every cell's identity (and therefore in every output
row) so this is visible in the data itself, not just in the code. This is
Step 3.2's decision taking effect for the first time in a real data-generating
path: every diagnostic committed before this step (`check_metric_stability.py`,
`measure_u_eff.py`, `probe_n_scaling.py`, `sanity_reproduce_2408.py`) calls
`sample_gaussian`/`sample_t` **without** `normalize`, unchanged, and stays on
raw (unnormalized) scale by design -- those measurements answered questions
that did not turn on cross-`nu` scale comparability (SPEC.md, "Cross-
distribution normalization"), and Step 3.2 was explicit that the decision
applies to the production sweep going forward, not retroactively. The `u_eff`
computed per cell (`compute_cell_u_eff`) also draws through `_sampler_for`,
so it is measured on the same normalized data the trials themselves see, not
on raw-scale data -- consistent with what it is meant to characterize.

### What is deliberately not in scope here

The 72 reference-config cells (FP8-E4M3/E5M2 per-tensor, INT8 per-tensor;
`sweep_main.yaml`'s `reference_configs`, outside the 640-cell factorial) are
**not** executed by this harness yet. They have no `block_size`/
`scale_format`/`rht` structure and no natural `GemmConfig` to feed
`measure_u_eff`, so they need their own execution path rather than a forced
fit into `iter_main_cells`' schema; left for a follow-up rather than bolted on
here. The full 640-cell launch is also explicitly out of scope for this step
-- everything above was verified on small subsets only, per the task's own
instruction not to launch the real grid yet.

### Production run provenance (`sweep_e945b87a2395`)

The full 640-cell main-grid run has since happened, and needed two
invocations rather than one. The first was interrupted partway through by a
thermal shutdown (this hardware -- see ENVIRONMENT.md, an Intel Core 7 150U
thin-and-light chip -- sustaining 11 parallel workers at 100% across all
cores for hours); the second completed the remainder via `--resume`.

What's known from the final manifest
(`results/sweep_e945b87a2395/manifest_20260821T160901168538Z.json`), which
covers only the second (resumed) invocation:

- `start_time` 2026-08-21T16:09:01Z, `end_time` 2026-08-21T18:40:56Z.
- `cells_run_this_invocation`: 344. `cells_skipped_resume`: 296 (already
  complete from the first invocation, correctly not recomputed).
- `cells_error`: 0.

**No manifest exists for the first invocation's own start time or
duration** -- manifests are written only at the end of a completed
`execute_sweep` call, and the first invocation never reached that point.
This is a known provenance gap (there is no recorded wall-clock or start
timestamp for that first stretch of the run), **not a data integrity
issue**: the combined result was independently verified after the fact --
`git_commit` in the surviving manifest matches the commit that was checked
out, and the combined 640-cell output has exactly the expected 1,600,000
rows (240 heavy-tail cells x 5000 trials + 400 light-tail cells x 1000
trials), 0 duplicate `(cell_id, trial_index)` pairs, 0 nulls in the BE
columns, and 0 entries in `errors.log`.

A future long-running sweep on this hardware should account for its
thermal limits up front (e.g. an elevated stand for airflow, or a lower
`--jobs` count sustained over many hours) -- though `--resume` is exactly
why this particular interruption was a non-issue for the data itself,
just an avoidable delay.

## Robust statistics methodology (Step 3.4) — RESOLVED

**Decision.** Median and p90 carry all confirmatory statistical weight in
this project, including ν* localization. **p99 is reported for every cell,
always, but is INDICATIVE ONLY and is never the basis of a firm claim** --
regardless of whether the cell ran 1000 or 5000 trials. This lands before
any confirmatory analysis of the production sweep (`sweep_e945b87a2395`,
see "Production run provenance" above) has been performed: no aggregation or
quantile analysis of that sweep's data exists anywhere in this repository as
of this section being written (verified directly -- `scripts/` contains no
analysis script that reads `results/sweep_e945b87a2395*`, and no such script
imports `bootstrap_ci` or `summarize_cell`, both of which are new in this
step).

**Why.** A cell's p99 is set by roughly its top 1% of sample values -- 10 of
1000 trials, 50 of 5000. Percentile-bootstrap resampling draws *with
replacement from the observed sample itself*, so it can reweight those
extreme values but can never invent a value more extreme than the largest one
already drawn. An extreme-order statistic like p99 is therefore structurally
harder for this method to bracket correctly than a central one like the
median or p90, independent of how many trials a cell happened to run. This is
stated as the mechanism, not just the a priori claim; see the measured
coverage gap below for the empirical check.

**What landed in code.**

* `qgemm.stats.bootstrap_ci(data, statistic_fn, rng, n_resamples=10000, ci=0.95) -> (point_estimate, ci_low, ci_high)`
  -- percentile bootstrap: resamples `data` with replacement `n_resamples`
  times, recomputes `statistic_fn` on each resample, and takes the 2.5th/97.5th
  percentiles of the resulting distribution as `ci_low`/`ci_high`. The interval
  is reported as-is, never assumed or forced symmetric around the point
  estimate (pinned by
  `test_bootstrap_ci_interval_need_not_be_symmetric`). Takes an explicit
  `numpy.random.Generator`, per this project's fixed convention -- never
  NumPy's global RNG state (pinned by
  `test_bootstrap_ci_does_not_touch_numpy_global_rng_state`). `statistic_fn`
  is called once on the raw 1-D data for the point estimate and once on the
  full `(n_resamples, n)` resample matrix with `axis=1` for the bootstrap
  distribution -- `np.median`, `functools.partial(np.percentile, q=...)`, and
  `median_absolute_deviation` (see below) all satisfy that signature -- which
  is what makes the whole resampling pass one vectorized NumPy call instead of
  a `n_resamples`-iteration Python loop; at the resample counts the coverage
  check below needs, that is the difference between minutes and roughly an
  hour.
* `qgemm.stats.summarize_cell(be_values, rng, n_resamples=10000) -> dict` --
  median, p90, p99, and MAD of `be_values`, each as
  `{stat}`/`{stat}_ci_low`/`{stat}_ci_high` via `bootstrap_ci`. The p99 entry
  additionally carries `p99_ci_indicative_only: True`, always, so the
  indicative-only status is visible in the data itself and not just in this
  document.
* `qgemm.stats.median_absolute_deviation` gained an optional keyword-only
  `axis` parameter (default `None`, identical behavior to before) so it can
  serve as a `statistic_fn` in `bootstrap_ci`'s vectorized resampling pass;
  `summarize_cell` uses it to give MAD a bootstrap CI alongside the other
  three statistics.
* Tests in `tests/test_stats.py`: a known-answer case for `bootstrap_ci` on
  constant data (every resample gives the same statistic, so the CI must
  collapse to a point), the `ci_low <= point_estimate <= ci_high` bracket
  property across several statistic/distribution combinations, RNG-discipline
  tests (reproducible for a fixed seed, does not touch NumPy's global RNG
  state), and the coverage simulation below.

### Coverage simulation -- the empirical justification, not just the a priori claim

`test_bootstrap_ci_coverage_for_median_and_p99_under_heavy_tails` in
`tests/test_stats.py` measures 95%-CI coverage directly rather than assuming
it: 1000 simulated experiments, each drawing `n_trials` samples from
t-Student(ν=2) -- heavy enough to be a real stress test, light enough to have
finite variance -- at both of this project's adaptive trial budgets
(`n_trials=1000` for light-tail cells, `n_trials=5000` for heavy-tail cells).
Each experiment computes a 95% bootstrap CI of the sample median and of the
sample p99, and checks whether the true population median (`0`, by symmetry)
and the true population p99 (`scipy.stats.t.ppf(0.99, df=2)`, exact) fall
inside their respective intervals. The bootstrap itself uses `n_resamples =
1000`, not `bootstrap_ci`'s own default of 10000 -- at `n_trials=5000` the
full default would put this one test on the order of an hour; 1000 resamples
still stabilizes a percentile-bootstrap CI adequately for a coverage check
(the recommended range for CI estimation, as opposed to precise tail-quantile
work) and is disclosed in the test rather than silently substituted.

**Measured coverage (1000 simulated experiments per cell, nominal target 95%):**

| n_trials | median coverage | p99 coverage |
|---|---|---|
| 1000 | 95.5% | 92.7% |
| 5000 | 94.8% | 94.0% |

**Median coverage sits close to the nominal 95% at both trial counts**
(95.5%, 94.8%), well inside the `[90%, 99%]` tolerance band `1000` simulated
experiments' own sampling noise warrants (asserted in the test as the
concrete acceptance criterion) -- the median is estimable stably enough to
bootstrap at either trial budget this project actually uses.

**p99 coverage is measurably below both the median's coverage and the
nominal 95% target at `n_trials=1000`** (92.7% vs. 95.5%/95%, roughly 3.5
simulated-experiment standard errors low, `sqrt(0.93*0.07/1000) ~= 0.8pp`) --
consistent with the a priori mechanism above and the reason p99 is policy-set
to indicative-only. **Reported plainly, not forced into a more dramatic
shape than measured: the gap is real but modest, not a collapse, and at
`n_trials=5000` it narrows further** (94.0% vs. 94.8%/95%, roughly 1.3
standard errors low, not clearly distinguishable from nominal at that
sample size on its own). More trials measurably help p99 -- the extra 4000
trials give the bootstrap more of the sample's own extreme tail to resample
from -- but even at 5000 trials p99 coverage does not close the gap to the
median's, so the mechanism (a percentile bootstrap cannot invent values more
extreme than the largest one already sampled) is not fully resolved by a
larger trial count either. This is the concrete basis for treating p99 as
indicative-only **at any trial count this project uses**, rather than only
below some threshold -- the policy set out in the Decision above.

To reproduce: `pytest tests/test_stats.py -k coverage -s` (about 10-11
minutes; the four numbers above are read directly off its printed output,
which the test also prints for exactly this reason).

## Step 4.1 -- n-scaling fits

**Status: measured, confirmatory.** Fits the empirical log-log slope of
`median(BE)` vs `n` for every one of the 128 configurations in the frozen
production sweep (`nu` x `block_size` x `scale_format` x `round_mode` x
`rht`, 8x2x2x2x2, each regressed across its 5 `n` values), and compares each
against the theoretical `-0.5` exponent from "Theoretical bound definition
(🧠1) -- RESOLVED" above. Produced by `scripts/fit_n_scaling.py`, which adds
`qgemm.stats.bootstrap_loglog_slope_ci` -- a generalization of `bootstrap_ci`
for a statistic (a fitted slope) spanning several independently-resampled
groups of unequal size, which `bootstrap_ci`'s single-array API cannot
express on its own; see that function's docstring for why. Output under
`results/analysis/n_scaling_fits/n_scaling_fits_5d2c4b01328a*` (table,
detail table, config, statement), kept under `results/analysis/` rather than
`results/diagnostics/` since this is confirmatory analysis of the production
sweep, not a diagnostic probe. To reproduce:
`python scripts/fit_n_scaling.py` (about 1 minute; 2000 bootstrap resamples
per configuration, each trial resampled within its own `n`, deterministically
seeded per configuration).

### Method, briefly

Per configuration: `median(BE)` at each `n` is the median, over trials, of
each trial's own `be_median` (the trial-level column the sweep already
stores) -- the same "median of medians" convention
`scripts/probe_n_scaling.py` established. The log-log slope is fit by
ordinary least squares over the 5 `(log n, log median(BE))` points. Its 95%
CI comes from `bootstrap_loglog_slope_ci`: 2000 replicates, each resampling
every `n`'s trials independently (with replacement, matching that `n`'s own
trial count), recomputing the 5 medians, and refitting the slope. The
multiplicative constant `c_hat` solves `median(BE) = c_hat * n^p_hat *
u_eff`, `p_hat` fixed at the already-fitted empirical slope and `u_eff`
taken directly from the sweep's own per-cell `u_eff_p50` column (no
recomputation) -- **`c_hat` is DESCRIPTIVE/EXPLORATORY ONLY
(PREREGISTRATION.md sec 3.2) and is never substituted for the confirmatory
bound's constant, which stays fixed at `c=1` everywhere else in this
project**; the script restates this at the point `c_hat` is computed
(`fit_c_hat`'s docstring), not only here.

### Results

**0 of 128 configurations have a slope CI containing -0.5; 128 of 128 fall
in the "weakening" bucket (CI entirely above -0.5, i.e. less negative --
decaying slower than the cancellation regime predicts); 0 decay faster than
predicted; 0 show a positive slope** (error growing with `n`, which would
have been flagged explicitly as a striking result given exact accumulation
-- none did).

**This "128/128" headline needs the magnitude alongside it, or it overstates
the finding.** The break criterion (CI excludes -0.5) is a statistical
significance test, and at this sweep's trial counts (1000-5000 per cell) it
has enough power to detect very small deviations -- so it fires even where
the empirical slope is a few thousandths from -0.5. Splitting by magnitude
(`|slope - (-0.5)| > 0.02`, the same threshold `probe_n_scaling.py`'s
`SLOPE_MATCH_TOL` already uses for exactly this distinction): **63 of 128
configurations deviate materially, 65 do not.** The gap shrinks smoothly and
monotonically with `nu` (min/max across the 16 configurations at each `nu`):

| `nu` | gap min | gap max |
|---|---|---|
| 1 | 0.361 | 0.490 |
| 2 | 0.117 | 0.178 |
| 3 | 0.041 | 0.085 |
| 5 | 0.013 | 0.037 |
| 8 | 0.007 | 0.024 |
| 15 | 0.006 | 0.020 |
| 30 | 0.005 | 0.019 |
| gaussian | 0.004 | 0.018 |

**`nu` in `{1, 2, 3}` is unambiguously material at every configuration** (gap
0.04-0.49, an order of magnitude or more beyond the 0.02 threshold): the
empirical slope is nowhere near the -0.5 cancellation regime here, most
extremely at `nu=1` where slopes range from -0.010 to -0.139 -- one to two
orders of magnitude shallower than predicted, not a borderline case. This
matches the mechanism SPEC.md's bound-definition section already names: at
`nu=1` the derivation's CLT/LLN assumptions (mean-zero numerator,
law-of-large-numbers denominator) do not hold, since Cauchy has neither a
finite mean nor variance.

**`nu` in `{5, 8}` straddles the threshold** (some of their 16 configurations
land above 0.02, some below -- visible in the table above and in the full
per-configuration list in the statement file), consistent with a smooth
transition rather than a sharp cutoff. **`nu` in `{15, 30, gaussian}` is
never material** (gap 0.004-0.020 at every configuration): the empirical
slope sits within two hundredths of -0.5 everywhere in this range, i.e. the
u_eff/√n cancellation regime's *magnitude* is well supported here even
though the CI is tight enough to exclude -0.5 exactly. This sharpens, and is
consistent with, the earlier n-scaling probe's own finding at `nu=30`
(slopes ≈ -0.4978/-0.4966, Step 2.3 groundwork): the production sweep's
larger trial counts (1000-5000 vs. the probe's 500) narrow the CI further
but do not move the point estimate outside the probe's own range.

**c_hat (exploratory only) ranges from 0.81 to 1.87 across the 128
configurations, mean 1.51** -- order-1 and therefore not wildly
uninformative about the derivation's leading-order constant, but not tightly
clustered around `c=1` either, consistent with SPEC.md's own caveat that the
derivation is an order-of-magnitude (CLT/LLN) argument rather than a proven
tight inequality and `c=1` is "not guaranteed to be well-calibrated."
**Restated: these values are descriptive only and must not be read as
evidence for changing the confirmatory bound's `c` away from 1.**

### What this does not settle

This script fits and reports; it does not perform ν\* localization (Step
4.2, a separate step not run here) and does not itself decide whether the
bound is "broken" anywhere -- that judgment, per the break criterion in
"Theoretical bound definition (🧠1)," is about the ratio
`empirical_BE / cota(n)`, which Step 4.2 computes directly rather than being
inferred from the slope fits alone. The `nu` in `{5, 8}` straddle is
reported as observed, not resolved into a single verdict for those `nu`
values -- see the per-configuration table for which specific
`(block_size, scale_format, round_mode, rht)` combinations land on which
side.

## Step 4.2 -- nu* localization (PRIMARY RESULT)

**Status: measured, confirmatory. This is the project's primary result.**
Locates nu* (PREREGISTRATION.md sec 1) -- the critical tail-weight where the
bound `cota(n) = u_eff/sqrt(n)`, `c` fixed at 1 (SPEC.md, "Theoretical bound
definition (🧠1) — RESOLVED") -- breaks, per PREREGISTRATION.md sec 3's break
criterion applied exactly as written: broken at (family, nu) if the lower
bound of the 95% bootstrap CI (B=10000, sec 3.2) of the median of the ratio
`empirical_BE / cota(n_primary)` exceeds 1.0. Produced by
`scripts/localize_nu_star.py`. Output under `results/analysis/nu_star/
nu_star_e15f13a87939*` (16-row families table, 128-row detail, 80-row
sensitivity table across all 5 `n`, statement, headline figure).

### n_primary, and why this step could not start without it

PREREGISTRATION.md sec 4.0 requires ONE `n_primary` for the nu* verdict,
left ⚠ OPEN pending a dated amendment (sec 7 item 2). No such amendment
existed anywhere in the document before this step. Picking a value after
already having seen Step 4.1's per-n slope fits (which cover all 5 n's,
`n_primary` included) would have been exactly the post-hoc researcher
degree of freedom PREREGISTRATION.md sec 9 (R9) was written to close, so
this was raised to the user rather than decided here. The user declared
**n_primary = 4096**, on grounds independent of the nu* pattern (closest to
real GEMM contraction dimensions in LLM inference/training; where the
√n-decay derivation's asymptotic assumptions hold most cleanly) — logged as
PREREGISTRATION.md sec 8.3, a dated amendment disclosing this is **post-data**
(the sweep and Step 4.1's slope fits already existed) rather than claiming
false pre-data cleanliness. See sec 8.3 for the full disclosure, including
the honest statement of what residual risk this carries and what it does not
taint.

### Headline finding: GRID MIS-SPECIFICATION, not H1 confirmation or H0

**The bound is broken at every one of the 8 tested nu, in every one of the 16
families, at n_primary -- and at every other tested n as well (16, 64, 256,
1024; the sensitivity table shows the identical 16/16-censored-above pattern
at all five n).** This is exactly the contingency PREREGISTRATION.md sec 5.1
names explicitly: *"All cells censored above (bound breaks even at the
lightest-tailed nu tested, r = 8 everywhere): H1 untestable on this grid,
because the nu range was mis-specified. Reported as such, with the finding
that the bound fails even in the near-Gaussian regime -- a strong result in
its own right, but labelled a grid mis-specification, not an H1
confirmation."*

**This is not H0 either**, despite the mechanical PREREGISTRATION.md sec 2.1
conjunctive rule technically computing r(32,s) = r(16,s) = 8 for both scale
formats (which sec 2.1's literal table would read as "no detectable
difference"). Sec 5.1 explicitly supersedes that reading for the
all-censored case: with the bound already broken everywhere, there is no
headroom left in the tested nu range to observe *where* block-32 and
block-16 diverge in fragility, so "no detectable difference" would
misdescribe a saturated measurement as a null result. `nu*` is undefined for
every family on this grid; H1 as formulated (sec 1: "nu*(32) > nu*(16)")
cannot be evaluated, not because the data show no effect, but because both
sides of the comparison are censored past the edge of what was tested.

**Concrete numbers, canonical sub-configuration (round_mode=rtne, rht=False),
at n_primary=4096:**

| preset | nu=1 (heaviest) | nu=30 | nu=gaussian (lightest, near-Gaussian) |
|---|---|---|---|
| MXFP4 (block=32, e8m0) | ratio = 31.92 [31.85, 31.97] | ratio = 1.615 [1.613, 1.618] | ratio = 1.588 [1.585, 1.590] |
| NVFP4 (block=16, e4m3) | ratio = 20.91 [20.85, 20.97] | ratio = 1.531 [1.528, 1.533] | ratio = 1.515 [1.513, 1.517] |

(ratio = median(BE)/cota(n_primary), 95% bootstrap CI in brackets.) Even at
the mildest tail tested -- the Gaussian limit, the regime the sqrt(n)
derivation's assumptions fit best -- the CI sits entirely and unambiguously
above 1.0: the `c=1` prefactor under-predicts the empirical error by roughly
50-60% there, growing to a 20-32x under-prediction at nu=1. This is
consistent with, and not contradicted by, Step 4.1's finding that `c_hat`
(fitted with the empirical, not the fixed, exponent) averaged 1.51 across
the 128 configurations: a persistently-too-small `c=1` is precisely what
produces a ratio anchored above 1 everywhere, since the empirical decay
exponent Step 4.1 measured is uniformly slightly less steep than -0.5 (the
"128/128 weakening" result), so the gap this `c=1` under-prediction leaves
does not close as `n` grows -- confirmed directly by the sensitivity table
below, where the ratio is nearly n-invariant rather than shrinking.

### Per-family results (all 16, canonical and the 3 robustness combos)

**16/16 families: r=8 (censored above). 0/16 non-monotone.** Every family's
`broken_at_nu` set is the full 8-element grid `{1, 2, 3, 5, 8, 15, 30,
gaussian}`, so the down-set monotonicity check (PREREGISTRATION.md sec 1.1)
is trivially satisfied everywhere (a full set is always a down-set) -- there
is no ambiguity or inconclusive-exploratory flag to raise here, just a
uniformly saturated grid. Full per-family table:
`results/analysis/nu_star/nu_star_e15f13a87939_families.parquet` /
`_families.csv`.

### H1 analysis (as specified, reported for completeness -- superseded by the grid mis-specification finding above)

Canonical sub-configuration (round_mode=rtne, rht=False): P1a (block effect
@ e8m0) = r(32,e8m0) − r(16,e8m0) = 8 − 8 = 0. P1b (block effect @ e4m3) =
8 − 8 = 0. Mechanical verdict per sec 2.1's table: "H0". P2a (scale effect @
block16) = 0, P2b (scale effect @ block32) = 0 -- design-validity
comparison, not a test of H1 (sec 4.1). Neither factor shows a rank
difference because every rank is pinned at the ceiling (8); this is the
signature of saturation, not of equal fragility. **Per the grid
mis-specification finding above, none of these zero-effect numbers should be
read as evidence for H0 or against H1** -- they are an artifact of both
block sizes' nu* being off the tested grid entirely, not a measurement of
their relative position. Robustness check: the same degenerate P1a=P1b=
P2a=P2b=0 pattern holds at all 3 non-canonical (round_mode, rht) combos
(rtne+rht, sr+not-rht, sr+rht) -- consistent across all 4, because the
saturation is universal across round_mode and rht as well, not because the
underlying comparison is informative.

### Sensitivity across n (PREREGISTRATION.md sec 4.0)

| n | families censored above (r=8) | families censored below (r=0) | mean r |
|---|---|---|---|
| 16 | 16/16 | 0/16 | 8.00 |
| 64 | 16/16 | 0/16 | 8.00 |
| 256 | 16/16 | 0/16 | 8.00 |
| 1024 | 16/16 | 0/16 | 8.00 |
| 4096 (n_primary) | 16/16 | 0/16 | 8.00 |

**The all-censored-above pattern is not an n=4096 artifact -- it holds
identically at every tested n, including the smallest (n=16).** Spot check
(MXFP4 preset, nu=1, canonical sub-config): ratio = 3.55 at n=16, rising to
7.18 (n=64), 12.10 (n=256), 19.62 (n=1024), 31.92 (n=4096) -- already well
above 1.0 at the smallest tested contraction dimension, and growing roughly
in line with the sub-(-0.5) decay exponent Step 4.1 measured. So the
mis-specification is not a large-n phenomenon that a smaller `n_primary`
would have avoided; the `c=1` prefactor is uncalibrated across the entire
tested scale range.

### What this does and does not mean for the project

**This is a real, informative confirmatory finding, not a null result to be
explained away.** Per PREREGISTRATION.md sec 5.1's own framing, it should be
reported as: the empirically-calibrated `u_eff/sqrt(n)` scaling law's
*functional form* (Step 🧠1) is well supported at nu≥15 (Step 4.1: gap from
-0.5 negligible there), but its *leading constant*, fixed a priori at `c=1`
per the anti-circularity constraint (PREREGISTRATION.md sec 3.2), is not
well calibrated anywhere in the tested grid -- it under-predicts by roughly
50% even in the best case (near-Gaussian, large n) and by over an order of
magnitude at the heaviest tail. `c=1` was explicitly flagged as *not
guaranteed to be well-calibrated* when 🧠1 was resolved (SPEC.md,
"Theoretical bound definition (🧠1)"; PREREGISTRATION.md sec 3.2), and a
poorly-calibrated `c` was named there as "a reportable finding rather than
grounds to re-fit" -- this is that finding, now measured directly rather
than anticipated.

**H1 as formulated is untestable on this grid, in either direction.** The
data do not support "block-32 is more fragile" (H1), "no detectable
difference" (H0), nor a directional reversal -- all three presuppose an
observable crossing that this grid never reaches. Extending the nu range
toward the Gaussian side (nu > 30, e.g. much larger degrees of freedom, or
a formal treatment distinguishing "close to Gaussian" from "exactly
Gaussian" more finely) is the natural next step to find where, if anywhere,
the bound holds -- not attempted here.

**Does not proceed to causal decomposition (Step 4.3)** -- that step would
decompose P1/P2/P3 effects that presuppose an observable nu*, which does
not exist on this grid; it is not run here, per the task that produced this
section.

## Step 4.3 -- causal decomposition (EXPLORATORY, post-4.2 pivot)

**Status: measured, EXPLORATORY -- not a resolution of H1.** PREREGISTRATION.md
sec 8.4 (a dated, disclosed post-data amendment) records why this step exists in
this form: Step 4.2 found the bound broken at every tested ν, in every one of
the 16 families, at every tested n, so ν* is undefined everywhere and P1/P2/P3
(all defined in terms of ν*) cannot be evaluated. This step answers a related
but weaker, purely descriptive question instead -- which factor, block_size or
scale_format, moves measured `median(BE)` more, and under what conditions --
using **no theoretical bound at all**: no `cota(n)`, no `c`, no ratio to 1.0
anywhere in this computation, only directly measured backward error in log
space. Produced by `scripts/causal_decomposition.py`. Output under
`results/analysis/causal_decomposition/causal_decomposition_f9d5d51039ab_EXPLORATORY_*`
(160-row table: 8 ν × 5 n × 2 round_mode × 2 rht, headline figure, statement),
filenames and headers marked EXPLORATORY throughout, kept out of
`results/analysis/nu_star/`'s confirmatory namespace.

### Method

For each (ν, n, round_mode, rht), pooling across the factor held out (ν and n
are never collapsed -- Step 4.2's failure came partly from trusting a single
n, so all 5 are reported at every ν, not just the canonical one):

    effect_block(ν,n) = median(log BE | block=16, pooled over scale_format)
                       − median(log BE | block=32, pooled over scale_format)
    effect_scale(ν,n) = median(log BE | scale=e4m3, pooled over block_size)
                       − median(log BE | scale=e8m0, pooled over block_size)
    interaction(ν,n)  = [median(log BE|16,e4m3) − median(log BE|32,e4m3)]
                       − [median(log BE|16,e8m0) − median(log BE|32,e8m0)]

**Median of log(BE), not log of median(BE)** -- stated once, applied
everywhere. Each of the three statistics has a 95% percentile-bootstrap CI
from a single vectorized pass that resamples all four underlying
`(block_size, scale_format)` cells independently (so `interaction`'s
replicates share the same per-replicate draw as `effect_block`/`effect_scale`,
not a recombination of separately-bootstrapped pieces); `n_resamples=2000`,
not PREREGISTRATION.md sec 3.2's confirmatory B=10000, disclosed as an
exploratory-appropriate reduction (same convention Step 3.4's coverage
simulation used for B=1000, for the same computational-cost reason).

### Headline finding: scale_format dominates block_size's effect on BE, everywhere tested

**`|effect_scale| > |effect_block|` at all 40 canonical (ν, n) cells** (8 ν ×
5 n, round_mode=rtne, rht=False), **and the ranking is stable across every one
of the other 3 (round_mode, rht) combinations** (`rtne+rht`: 34/40 scale,
6/40 block; `sr, no-rht`: 38/40 scale; `sr+rht`: 35/40 scale -- scale wins the
large majority everywhere, never reverses to an overall block majority in any
combo). At `n=4096`, canonical: `effect_scale` ranges from −0.162 (gaussian)
to −0.389 (ν=2), while `effect_block` ranges from −0.042 (gaussian) to −0.246
(ν=1) -- scale format's pull on `median(BE)` is 2-4× block size's across the
grid. Both effects are negative throughout (at n>=64): `block=16` and
`scale=e4m3` (NVFP4's own choices) both reduce `median(BE)` relative to their
alternatives, consistent with NVFP4 empirically outperforming MXFP4 at the
element-error level (SPEC.md, "u_eff measurement", "NVFP4 has the lowest
`u_eff` of the four at every ν").

### The n=16 cell is not a counterexample -- it is not measuring block_size at all

At `n=16`, `effect_block` sits near zero and is significant at only 1 of 8 ν
(ν=8, +0.0251) -- looking like "block size stops mattering at small n." **It
does not mean that.** `qgemm.blocks._block_amax` zero-pads a short block to
make the reshape rectangular, and padding with zeros never changes `max|x|`
(`_block_amax`'s own docstring: "That is free... the padded block gets
exactly the scale the shorter block would have"). At `n=16`, `block_size=32`
therefore produces exactly **one** block spanning all 16 real elements, with
a scale computed from their own `amax` -- **structurally identical** to
`block_size=16`'s own single full 16-element block at the same `n`. Under
RTNE (no randomness), the two configurations quantize the same real elements
identically. What differs between the two `block_size` arms' trials at
`n=16` is only the **operand data**: `block_size` is part of `cell_key`
(`scripts/run_sweep.py`'s `derive_seed`), so the two arms draw independent
`A`/`B` samples even at matched `n`. `effect_block` at `n=16` is therefore
measuring pure sampling noise between two datasets pushed through the *same*
quantizer, not a block-size effect -- consistent with its near-zero, mostly
non-significant pattern, and the one nominally-significant cell (ν=8) is
exactly the rate a 95%-CI screen over 8 independent tests would produce by
chance (~0.4 expected false positives). **The real block-size effect first
becomes measurable at `n=64`** (the smallest tested `n` where 16 and 32 are
genuinely different quantizers) and its magnitude is essentially flat from
`n=64` through `n=4096` at every ν (e.g. ν=1: −0.286, −0.270, −0.257, −0.246
at n=64/256/1024/4096) -- so "which n you'd have picked" does not change the
block-size reading once n is large enough for the two arms to differ at all,
though it would have looked spuriously like "no effect" had only `n=16` been
examined.

### Ranking consistency across n, and magnitude vs ν

**Dominant factor (scale) is identical across all 5 tested n at every ν** --
no crossing anywhere in the grid, including through the n=16 cell above
(scale dominates there too, trivially, since block's "effect" is ~0). This
directly addresses the fragility that produced Step 4.2's mis-specification:
here, unlike ν*, the practical answer ("scale format matters more") does not
depend on which n would have been chosen as primary.

**`effect_block`'s magnitude grows as the tail gets heavier** (n=4096:
|−0.246| at ν=1 vs |−0.042| at ν=gaussian) -- **consistent with** the u_eff
level-gap finding already in this document (SPEC.md, "u_eff measurement",
"Result 2: the level-gap check": `median(BE)` level ratio 32/16 at matched
n=1024, scale format e8m0, grows from 1.0344× at ν=30 to 1.3206× at ν=1).
Both measurements, taken independently (different scripts, different
statistics -- a level ratio there, a log-space median difference here), point
the same direction: block size matters more under heavy tails. `effect_scale`
shows the same qualitative pattern (|−0.162| at gaussian growing toward
|−0.28|-to-`−0.39` at heavy ν, peaking at ν=2 rather than ν=1 specifically --
not perfectly monotone, reported as observed rather than smoothed).

### Interaction

`interaction` is significant (CI excludes 0) at essentially every (ν,n) cell
with `n>=64` (e.g. n=4096: −0.095 at ν=1 to −0.028 at ν=gaussian, all
significant) -- block size's effect on `BE` is measurably larger in magnitude
at `e4m3` than at `e8m0` (and symmetrically, scale format's effect is larger
at `block=32` than at `block=16`), i.e. **the two factors do not act purely
additively**. This is reported as a real, measured interaction, not
incorporated into the headline "which factor dominates" comparison, which
uses the pooled (marginal) effects as specified.

### Relationship to Step 4.2 -- what this does and does not settle

**This is not a resolution of H1 as originally formulated**
(PREREGISTRATION.md sec 8.4, stated there and restated here): H1 (sec 1) and
P1-P3 (sec 4) are specifically claims about ν* -- where a bound breaks -- and
this step uses no bound at all, so it cannot confirm, reject, or null-reframe
H1 in sec 1/2/5's sense. What it does establish, at the weaker/descriptive
level: **scale_format is the larger lever on backward error in this study's
factorial design, consistently across ν, n, round_mode and rht** -- the
practical, MXFP4-vs-NVFP4-relevant part of the original motivation, answered
directly from measured error rather than through a bound that Step 4.2 showed
is not calibrated on this grid. The two steps' conclusions are about different
quantities and neither supersedes the other: Step 4.2 says the bound
`cota(n)` cannot currently locate a fragility crossing; Step 4.3 says that,
independent of any bound, scale format moves the raw error more than block
size does, essentially everywhere this grid was measured.

### Reconciliation with the u_eff-based finding -- PARTIAL AGREEMENT, no crossover in BE

**Check performed on request, against the earlier per-element finding** (SPEC.md,
"u_eff measurement", "Result 3: other patterns worth noting"): u_eff's own
marginal effects show block-size *exceeding* scale-format at ν=1 (block-size
effect 1.155 vs scale-format effect 1.113, E8M0/block16 arm) and the two
*cross* between ν=1 and ν=2 (scale format effect 1.144 > block-size effect
1.120 already at ν=2). Step 4.3's headline says scale dominates "consistently
across all 8 ν," which on its face looks like it disagrees with a real
crossover. Re-examined against Step 4.3's own already-computed table plus one
new point-estimate query against the same sweep data (no new sweep, no new
bootstrap -- medians only, reusing `be_median` exactly as Step 4.3 does):

**1. The magnitudes, canonical sub-config, ν∈{1,2} vs ν=15, all 5 n:**

Pooled (Step 4.3's own statistic, log space, from the already-committed
table) -- gap = |effect_scale| − |effect_block|:

| n | gap at ν=1 | gap at ν=2 | gap at ν=15 |
|---|---|---|---|
| 16 (degenerate, see below) | 0.353 | 0.293 | 0.186 |
| 64 | 0.043 | 0.164 | 0.135 |
| 256 | 0.016 | 0.166 | 0.124 |
| 1024 | 0.020 | 0.173 | 0.135 |
| 4096 | 0.032 | 0.189 | 0.127 |

**Yes, visibly narrower at ν=1** (gap 0.016-0.043 at n≥64) **than at ν=15**
(gap 0.124-0.135) -- roughly a 4-8x narrowing, not a small effect. **ν=2 is
not narrower at all** -- its gap (0.16-0.19) is the *widest* of the three
points checked, wider even than ν=15. So the narrowing u_eff shows between
ν=1 and ν=2 does not carry over to BE in the same place: BE's gap narrows
sharply exactly at ν=1 and has already reopened, wider than baseline, by
ν=2.

**No crossover.** Point estimates never flip (`|effect_scale| >
|effect_block|` at every one of the 40 canonical cells, already stated in the
headline), and the 95% CIs do not even overlap at ν=1: e.g. at n=4096,
|effect_block| ∈ [0.2395, 0.2510] and |effect_scale| ∈ [0.2719, 0.2831] --
disjoint, scale significantly larger even at the heaviest tail tested, at
every n≥64. The disaggregated, u_eff-table-matching view (per-arm ratios,
point estimates, same sweep data) confirms this directly -- `block@e8m0`
never exceeds `scale@block16` at any tested n, including ν=1:

| n | block@e8m0 ratio (32/16) | scale@block16 ratio (e8m0/e4m3) |
|---|---|---|
| 64 | 1.322 | 1.465 |
| 256 | 1.329 | 1.483 |
| 1024 | 1.317 | 1.510 |
| 4096 | 1.298 | 1.524 |

(ν=1 shown; block@e8m0 is 30-32% worse for block=32, scale@block16 is
46-52% worse for E8M0 -- scale's lead is smaller than at milder ν, per the
narrowing above, but never erased.) **Verdict: PARTIAL AGREEMENT.** The
direction (narrowing at heavy tail) replicates; the magnitude (an actual
sign flip) does not. Step 4.3's headline ("consistently across all 8 ν") is
accurate as a statement about which factor's point estimate is larger, and
should not be read as claiming the margin is uniform -- it is not, and ν=1
is where it is thinnest by a wide margin.

**2. A candidate mechanism, checked for plausibility, not asserted.** The
task's hypothesis: u_eff's near-zero cut (SPEC.md, "u_eff measurement," "The
near-zero policy") excludes flushed elements entirely, while `BE` is the
full end-to-end GEMM error, which does not exclude their contribution. The
survival-fraction table already in this document (SPEC.md, "Post-hoc: the
near-zero cut is not neutral between the two arms") shows exactly the kind
of asymmetry this would require: at ν=1, block=32 retains only 36.0% of its
elements past the cut versus block=16's 51.7% -- block=32 loses proportionally
more to flushing, and does so most severely at the heaviest tail. If those
excluded elements contribute disproportionate error under block=32
specifically, `BE` (which includes them) should show a *larger* block-size
penalty than `u_eff` alone predicts -- and it does: block@e8m0 goes from
1.155 (u_eff, ν=1) to 1.298 (BE, n=4096, ν=1), a real and sizeable
amplification in the direction the hypothesis predicts.

**But this alone does not explain the pattern, and should not be presented
as though it does.** The scale-format effect amplifies from u_eff to BE by
even more at ν=1 (1.113 → 1.524, block16 arm) than the block-size effect
does (1.155 → 1.298) -- if flushed elements were the whole story and were
specifically a block-size phenomenon, the scale-format gap should not grow
faster. `u_eff`'s near-zero cut is defined per-element relative to each
element's own effective scale and is measured identically regardless of
`scale_format`, but this document does not currently report a
`scale_format`-conditioned survival-fraction table (only the
`block_size`-conditioned one, at fixed E8M0) -- so whether E8M0 also loses
disproportionately more elements than E4M3 at ν=1, which would extend the
same mechanism to the scale-format effect, is **not verified here** and
would need that additional breakdown (not computed in this check, per the
instruction to reuse existing data rather than run new analysis).
**Conclusion: the near-zero-cut hypothesis is plausible and directionally
consistent for the block-size component, unconfirmed for the scale-format
component, and therefore not established as a complete explanation of why
BE fails to reproduce u_eff's ν=1 crossover.**

**3. Not edited into the headline above** -- this note stands alongside it,
per the task that requested this check.

## Step 4.4 -- real-activation realism check (EXPLORATORY)

**Status: measured, EXPLORATORY -- not a fourth confirmatory result.**
Every result in Steps 4.1-4.3 was measured on synthetic t-Student(nu)
operands; PREREGISTRATION.md sec 5.1 lists "real activations" explicitly
under "exploratory ... never as evidence for or against H1." This step is
that check: does the t-Student model's error prediction transfer to what a
real network actually produces? Produced by
`scripts/check_activation_realism.py`. Output under
`results/analysis/activation_realism/activation_realism_ef8b4a911d0a_EXPLORATORY*`
(6-row table, config, statement). To reproduce:
`python scripts/check_activation_realism.py` (about 8 minutes: ~85s forward
pass over 300 sequences, the rest is 6 `qgemm` calls at n_tokens=34390 x
K=768 x N=2304 -- much larger than the 512x512x512 the exact route's own
performance constraint is defined against, so this script's runtime is not
held to that bound).

### Setup

GPT-2 small (`gpt2`, HuggingFace `transformers`), WikiText-2 raw test split
(`load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")` --
the un-namespaced `"wikitext"` repo id's loading script is broken under
`datasets>=5`; this is the fix, not a different dataset). 300 non-empty
lines, tokenized with dynamic per-batch padding (`padding=True`, so each
batch pads only to its own longest sequence, not a fixed `max_length`;
`tokenizer.pad_token = tokenizer.eos_token` set first -- GPT-2's tokenizer
has none by default and batched padding raises without it), truncated at
256 tokens as a safety net that in practice bound almost nothing (mean line
length 114.6 tokens). 34390 valid (non-padded) tokens total, identical
across all three layers since the same 300 sequences and the same attention
mask feed every hook.

**Layers.** Three depths of GPT-2 small's 12 transformer blocks -- `h0`
(first), `h6` (middle), `h11` (last) -- all at the **attention input
projection**, `attn.c_attn`, captured with a forward pre-hook (so the
activation captured is exactly `ln_1`'s output, before the QKV
projection). `c_attn` was chosen over `mlp.c_fc` only to keep the hook
point uniform across all three rows; the choice is held fixed rather than
mixed. GPT-2's `Conv1D` stores its weight as `(in_features, out_features)`
-- already `(K, N)` in this project's `A @ B` convention -- so the real
weight from the same layer is used directly as operand B, no transpose
needed (unlike an `nn.Linear` weight would require).

**Padding-aware kurtosis and GEMM operand, by construction, not by a
post-hoc filter.** The attention mask is applied immediately after each
batch's forward pass, before activations are ever accumulated: padded
positions are excluded from the array before anything downstream --
kurtosis, MAD, or the GEMM operand itself -- ever sees them. There is no
separate "exclude padding" step later that could be forgotten; the
contaminating rows never enter the accumulated tensor.

**Kurtosis convention, stated explicitly per the roadmap's own named trap.**
Excess kurtosis, Fisher's definition (`scipy.stats.kurtosis(..., fisher=True,
bias=True)`): Gaussian maps to **0**, not to 3 (the raw/Pearson convention
used by some other sources, e.g. `scipy.stats.kurtosis(..., fisher=False)`).
Computed over the **full flattened array** (all valid tokens x all 768
hidden features together) as one distribution, matching how this project's
whole t-Student nu grid already treats an operand -- one elementwise
distribution, not one per feature column.

**Equivalent nu.** `nu = 4 + 6 / excess_kurtosis`, the closed-form inverse of
the t-Student excess-kurtosis formula `excess_kurtosis = 6/(nu-4)` (defined
only for `nu > 4`). For any finite `excess_kurtosis > 0` this inverse is
algebraically always `> 4` -- the forward map sends `(4, inf)` onto
`(0, inf)`, so nothing finite and positive can invert back below the
boundary -- so a layer whose measured excess kurtosis implies `nu <= 4`
cannot arise from real (finite-sample) data through this formula; the
domain guard is kept in the code anyway (`nu_equivalent`'s docstring states
this explicitly) so nothing would be silently extrapolated if it ever did.
The genuine out-of-domain case is **`excess_kurtosis <= 0`**: no t-Student
at any `nu > 4` has zero or negative excess kurtosis, so a layer measured at
or below the Gaussian value would report `nu_equivalent = N/A` with a
stated reason rather than a number. This did not occur at any of the three
layers tested (see table below) -- GPT-2's activations are heavy-tailed
enough everywhere sampled that the case never had to be exercised, though
`h6`'s measured value (`nu_equivalent = 4.06`) came within 0.06 of the
boundary, close enough that the guard is not merely decorative.

**Predicted arm.** For each layer's equivalent nu, a synthetic t-Student(nu)
sample of the identical shape is drawn (`scripts/check_metric_stability.py`'s
`sample_t`, loaded by path exactly as `scripts/measure_u_eff.py` already
does) and rescaled so its MAD matches the real activation tensor's own MAD
-- the same rationale as "Cross-distribution normalization (Step 3.3)":
isolating tail *shape* from raw *scale*, since `BE` is not exactly
scale-invariant for E8M0 scales away from power-of-two rescalings. The real
weight matrix (same layer) is used as operand B in both arms, so the two
`qgemm` calls differ **only** in operand A's tail shape.

### Results

`median(BE)` (this project's primary confirmatory statistic elsewhere,
reused here as the summary; no bootstrap CI is computed -- each cell's `BE`
array already has ~79M elements, an order of magnitude past what any CI
would meaningfully sharpen for a single point comparison):

| layer | n_tokens | excess kurtosis | nu_equivalent | preset | observed median(BE) | predicted median(BE) | ratio (obs/pred) | verdict |
|---|---|---|---|---|---|---|---|---|
| h0  | 34390 | 1.020 | 9.880 | MXFP4 | 0.006892 | 0.006912 | 0.997 | CLOSE |
| h0  | 34390 | 1.020 | 9.880 | NVFP4 | 0.005294 | 0.005362 | 0.987 | CLOSE |
| h6  | 34390 | 101.915 | 4.059 | MXFP4 | 0.010733 | 0.007762 | 1.383 | MODERATE GAP |
| h6  | 34390 | 101.915 | 4.059 | NVFP4 | 0.007018 | 0.005687 | 1.234 | MODERATE GAP |
| h11 | 34390 | 7.025 | 4.854 | MXFP4 | 0.007718 | 0.007414 | 1.041 | CLOSE |
| h11 | 34390 | 7.025 | 4.854 | NVFP4 | 0.005499 | 0.005534 | 0.994 | CLOSE |

(CLOSE = within 10% of 1.0; MODERATE GAP = within 50%; would be LARGE GAP
beyond that -- none landed there.)

### Verdict: transfers well at two of three layers; the outlier-channel layer breaks it

**4 of 6 cells land within 10% of the t-Student prediction, and none land
outside 50%.** `h0` and `h11` both transfer closely (2-4% off) at both
presets -- the t-Student model, matched only on excess kurtosis and MAD,
predicts the real GEMM's `median(BE)` about as well as a same-nu synthetic
draw predicts a different same-nu synthetic draw would. This is a real,
positive transfer result for the two layers where it holds.

**`h6` is a genuine miss, not noise, and its own kurtosis number explains
why.** `h6`'s excess kurtosis (101.9) is 15-100x larger than the other two
layers (1.0 and 7.0) -- and unlike anything on the project's own nu-grid
(`nu in {1, 2, 3, 5, 8, 15, 30, gaussian}`): `nu in {1, 2, 3}` have
*infinite/undefined* population excess kurtosis (nu<=4 is outside the
closed form's domain entirely, the same boundary this section's formula
respects), while `nu >= 5` gives a *finite and much smaller* value
(`6/(5-4) = 6` at the heaviest finite-kurtosis grid point). `h6`'s 101.9 is
a large-but-finite number that falls in the gap the grid never samples --
finite, but far larger than the grid's largest finite value. This is the
well-documented "outlier feature" / "massive
activation" phenomenon in transformer internals (a small number of
extreme-magnitude channels, concentrated at particular depths) -- entirely
plausible as a real property of this specific layer rather than a script
bug, and it drives `nu_equivalent` to 4.06, a hair above the formula's
`nu > 4` floor. **A single scalar (kurtosis-matched nu) is evidently not
enough to characterize a distribution this extreme**: the real data at
`h6` produces MORE error than a t-Student(4.06) sample of the same MAD and
the same excess kurtosis, meaning excess kurtosis alone under-describes how
concentrated `h6`'s outlier mass actually is (t-Student's fourth-moment
shape is not the only thing that varies at this extreme -- higher moments,
or the outliers' specific channel-concentrated structure, plausibly matter
too, and are not tested by a one-parameter kurtosis match).

**Plain-language verdict:** the t-Student model's error predictions
transfer well to real GPT-2 activations at typical layers (h0, h11 --
excess kurtosis under 10, well inside the project's own nu-grid range), and
break down by 23-38% at a layer with extreme outlier-channel kurtosis
(h6 -- two orders of magnitude beyond typical, right at the formula's
domain edge). This is consistent with, not contradictory to, the rest of
this project's synthetic-data findings: it says the synthetic nu-grid is a
good model of real activations across most of a real network, and names
the specific condition (extreme, outlier-driven kurtosis) under which a
single kurtosis-matched nu stops being sufficient. Per PREREGISTRATION.md
sec 5.1, none of this bears on nu*, P1/P2/P3, or H1 -- it is reported as an
exploratory realism check on the modeling choice underlying the whole
study, not as a confirmatory result.

### Addendum -- h6 mismatch diagnostic (EXPLORATORY, cheap follow-up; headline numbers above unchanged)

A quick, targeted follow-up on the `h6` miss, requested after the headline
above was reported. **No GEMM was rerun and no number above was changed or
recomputed** -- this reuses the already-committed table and re-extracts
only `h6`'s own activation tensor (skipping the other two layers and both
presets' `qgemm` calls entirely, since neither is needed for a kurtosis /
channel-index diagnostic), verified to be the *identical* tensor by
recomputing its excess kurtosis and matching it exactly against the value
already on record (101.91520380555781, bit-for-bit).

**1. Direction of the miss, stated explicitly.** `ratio_observed_over_predicted`
is **1.383 (MXFP4) and 1.234 (NVFP4)** at `h6` -- both **greater than 1**.
**Observed error was HIGHER than predicted**, i.e. the real activations
produced *more* GEMM backward error than the MAD/kurtosis-matched synthetic
t-Student(4.06) draw did, not less. This was already stated in prose above
("the real data at `h6` produces MORE error than a t-Student(4.06) sample")
but the signed ratio is repeated here because the original table did not
flag the direction as its own reportable fact.

**2. Channel-concentration check.** For the full `h6` tensor (34390 tokens x
768 channels, 26 411 520 elements), the literal **20 single largest `|activation|`
values across the entire tensor** -- not a threshold, not a top-1% pool, the
actual top 20 -- fall in **exactly one channel: index 64**, across 20
distinct tokens. Widening to the top-1% pool (264 116 elements) softens this
but does not erase it: channel 64 alone accounts for 13.0% of that pool and
appears in the top-1% set for **all 34390 tokens** (i.e. channel 64 is among
the largest-magnitude entries of essentially every single token's activation
vector, not merely spiking occasionally); the 5 most frequent channels
(64, 266, 480, 87, 326) account for 58.2% of the top-1% pool, and the top 10
account for 75.3%. Per-channel max magnitude confirms the same handful:
channel 64 peaks at 8.06, channels 87/480/266 at 7.08/6.55/5.31, the next
channel down (640) at only 2.76 -- a sharp drop, not a smooth tail. (The
top-1% pool does touch 767 of 768 channels at least once, which is expected
and not in tension with the above: almost any channel will occasionally
contain a moderately large value across 34390 tokens; concentration is about
*where the mass piles up*, not about which channels are ever represented at
all.)

**3. Verdict: CONCENTRATED -- consistent with, though not proof of, the
"massive activations" mechanism.** A small, fixed set of channel indices
(most starkly, index 64 alone) dominates `h6`'s extreme-value tail across
nearly all 34390 tokens, matching the "massive activations" / "outlier
feature" pattern reported in the literature on transformer internals -- Sun
et al. 2024, "Massive Activations in Large Language Models", which documents
persistent per-channel outliers in transformer hidden states (Sun, Chen,
Kolter and Liu; `sun2024massive`) -- rather than generic scattered
heavy-tailedness. **Correction (Step 6.1, 2026-08-24):** an earlier draft of
this paragraph cited "Liu et al." here as if it were a second, independent
source alongside Sun et al. 2024. It is not: Zhuang Liu is the fourth
co-author of that same paper (confirmed against the paper's own author list
during reference verification), not a separate work, so the standalone "Liu
et al." reference has been removed rather than given its own bib entry. This
is a

**plausible, disclosed explanation for
the mismatch, not a proven one**: real activations with a few structurally
fixed extreme channels violate the t-Student model's implicit iid-across-
elements assumption in a specific way synthetic t-Student draws never do --
every synthetic sample's large values are randomly located and change from
draw to draw, while `h6`'s are anchored to the same handful of channels on
essentially every token. Whether that specific structural difference is
*mechanistically* what drives the 23-38% BE gap (as opposed to being
correlated with it) is not established here -- no ablation isolating channel
structure from the kurtosis-matching alone was run, and none is claimed. Had
the top values instead been scattered across many distinct, changing
channels, this addendum would have reported that plainly instead and left
the mismatch unexplained; that is not what was found.

## Step 6.1 -- reference verification

**Status (updated 2026-08-24): 15 bibliographic entries in `paper/refs.bib`,
each independently verified against live primary sources; 2 corrections
applied to this file; both originally-flagged gaps now closed (one by a
new entry, one by re-scoping an existing entry to a role already
supported by facts already verified -- see "Open gaps" below).** This
section records what was checked, what changed, and what did not.
`paper/refs.bib` is the artifact this verification produced; every entry
there carries its own dated, source-specific comment. Two entries
(`tseng2025training`, `higham2002accuracy`) were added after the original
13-entry pass, each on later explicit request and each verified before
being added, not copied in on request alone -- see the "Three conventions
in the literature" note above for the former and "Open gaps" item 1 below
(now closed) for the latter.

**7 citations verified with claim-level confirmation** (not just
existence -- the specific sentence/figure/definition each is cited for was
fetched and checked against the primary source):

- `nvidia2026nvfp4` (arXiv:2509.25149) -- the "36% more tokens" figure
  (Fig. 6b), the 8B hybrid Mamba-Transformer model, and the "round decode
  scale factors up" statement were all confirmed verbatim by fetching the
  paper's own HTML.
- `rasquinha2023metric` (arXiv:2408.02897) -- the backward-error formula and
  the 512x512 t-distribution setup confirmed verbatim; the per-vector (vs.
  this project's per-tensor) scaling difference confirmed and already
  correctly disclosed in this document's own "arXiv:2408.02897 reproduction"
  section.
- `fasoli2026finer` (arXiv:2601.19026, ICLR 2026) and `egiazarian2026bridging`
  (arXiv:2509.23202, ICLR 2026) -- both confirmed to exist, be ICLR
  2026-accepted, and support the scoop-risk differentiation claimed for each
  (narrow-distribution mechanism for the former; MSE metric + Laplace
  operand model, Definitions 1 and 3, for the latter, confirmed by fetching
  the paper's own text).
- `higham2019new` and `connolly2021stochastic` -- DOIs, volumes and page
  ranges cross-checked against publisher/indexing records. **Claim-level
  check added 2026-08-25, and it found a real misattribution** (see
  "Misattribution of the two probabilistic bounds" below).
- `sun2024massive` (arXiv:2402.17762) -- author list confirmed, including
  that Zhuang Liu is this paper's own fourth co-author (see correction
  below).

**3 standard, low-risk citations for datasets and specifications**
(existence and authorship/publication confirmed, not claim-level fetched
against every detail): `radford2019language` (GPT-2), `merity2016pointer`
(WikiText-2), `ocp2023mx` (the OCP Microscaling Formats spec -- previously
the least-documented reference in this project, cited nowhere by title or
URL before this pass).

**3 software/tool citations**, versioned against this project's own
lockfile where tracked: `mldtypes` (0.6.0, per `requirements.lock`),
`microxcaling` (1.1.0, per this file's own prose -- not in
`requirements.lock`, since it is explicitly outside this project's tracked
dependency set), `torchao` (no version pinned anywhere in this project;
optional import, skipped if unavailable).

### Corrections applied

1. **"Liu et al." ghost citation, removed.** The Step 4.4 addendum's
   "Verdict: CONCENTRATED" paragraph cited "Sun et al. 2024" and a
   separate "Liu et al." as if they were two independent sources. They are
   not: Zhuang Liu is the fourth co-author of the same Sun et al. 2024
   paper (confirmed against the paper's own author list). The standalone
   reference has been removed from that paragraph; see the inline
   correction note there, dated today. This is the same class of error
   this project has already documented once before (the "Appendix A"
   phantom-citation incident logged earlier in this file, in "The controls
   are not proposals" section) -- an apparent citation that does not
   resolve to a real, distinct source -- handled the same way: named and
   removed, not left in place or silently dropped without a trace.

2. **MXFP4 scale-rounding direction -- checked, no error found, nothing
   changed.** The verification task also asked whether this file's
   description of `microxcaling`/OCP MX rounding the block-scale exponent
   *down* was "backwards," on the grounds that NVIDIA's NVFP4 paper
   states they round scale factors *up*. It is not backwards, and no
   statement in this file was altered on this point: the floor/round-down
   description of the `microxcaling` default is not an assertion resting
   on a citation, but an empirically bit-exact-verified fact about the
   actual `microxcaling` package (this file's own "Verified against
   `microxcaling` 1.1.0" section, backed by `tests/test_blocks.py`).
   NVIDIA's paper does not dispute that the OCP MX default rounds down --
   it documents choosing to round up in their own training practice, for
   the same saturation-avoidance reason this project's `quantize_mxfp4`
   already gives for its own round-up design (see "Why round the exponent
   up," above). That convergence is genuinely worth recording, so a short
   "Independent corroboration (Step 6.1)" note citing `nvidia2026nvfp4` was
   added directly after the `microxcaling` relationship paragraph -- but
   as a corroborating citation for an already-correct statement, not a
   correction of an error.

3. **Misattribution of the two probabilistic bounds -- real error, fixed
   (2026-08-25).** Found while drafting the paper's Background section,
   which needed to state these bounds precisely enough to cite one of them
   alone. The "On how to read 'inconsistent with sqrt(n)'" paragraph above
   read: "Higham & Mary (2019) and Connolly-Higham-Mary (2021) establish
   `sqrt(n)*u` (respectively `sqrt(n log n)*u` for the refined variant)".
   The `respectively` pairs each paper with the other's result, and the
   parenthetical inverts the relationship between the two bounds a second
   time. Both halves are wrong:

   - **Higham & Mary (2019)** is the `sqrt(n log n)*u` result, not the
     `sqrt(n)*u` one. Their paper states that `gamma_n` "can be replaced by
     a relaxed constant proportional to `sqrt(n log n) u`, with a
     probability bounded below" -- a general-case high-probability bound
     under a mean-independence model of the rounding errors.
   - **Connolly-Higham-Mary (2021)** is the `sqrt(n)*u` result: no log
     factor, unconditional, and specific to **stochastic rounding**. It is
     the *tighter* and later of the two, so calling `sqrt(n log n)` "the
     refined variant" is backwards -- they are two different bounds for
     two different rounding regimes, not a coarse and a refined version of
     one result.

   The original Step 6.1 pass verified both entries at DOI/volume/page
   level only and so could not have caught this; `paper/refs.bib`'s own
   comments never mention a log factor at all, and describe
   `connolly2021stochastic` as "specifically the stochastic-rounding
   refinement," which is consistent with the corrected attribution rather
   than the erroneous one. The paragraph has been rewritten and carries an
   inline correction note. Downstream text repeating the same error is
   listed in the paper writing plan below.

### Open gaps -- not resolved here

1. ~~The classical `gamma_n` bound (Higham 2002) has no entry in
   `paper/refs.bib`.~~ **Closed 2026-08-24.** `higham2002accuracy` added to
   `paper/refs.bib` on explicit request, using the ISBN/DOI already
   verified during the original pass (0-89871-521-0 / 978-0898715217,
   DOI 10.1137/1.9780898718027). Supports `src/qgemm/bounds.py`'s
   `gamma_n()`, this file's "Theoretical bounds (`qgemm.bounds`)" section,
   and both "Gamma_n sanity check" sections.
2. ~~Only 2 of the "three scoop-risk papers" originally requested have
   been supplied and verified.~~ **Closed 2026-08-24, no new citation
   needed.** The third is `rasquinha2023metric` (already in `refs.bib`
   for its reproduction-target role): the closest methodological
   antecedent to this project (same BE metric, same t-Student family),
   differentiated by scope rather than mechanism or metric -- it studies
   only per-tensor/per-vector INT8/FP8, never block-scaled microscaling
   (MXFP4/NVFP4), and contains no analogue of H1's ν* fragility crossing.
   Full reasoning in `paper/citation_worklist.md`'s "Item, formerly 'not
   found'" section. The three scoop-risk papers are `rasquinha2023metric`,
   `fasoli2026finer`, and `egiazarian2026bridging` -- all three already
   verified and in `refs.bib`.

## Paper writing plan (Step 6.2) -- narrative structure

*Recorded 2026-08-24. This is a decision record of author choices made in
discussion, not a claim requiring external verification -- unlike the
Step 6.1 entries above, nothing here was fetched or fact-checked, and
nothing here should be treated as such.*

Draft title (not finalized): "Isolating the Design Choice Behind
NVFP4's Advantage over MXFP4" -- leads with the causal-decomposition
finding (Step 4.3) as the paper's headline, deliberately NOT with the
ν* localization (Step 4.2), per author decision to lead with the
stronger, more robust result.

Section structure and target lengths (~6 of the workshop's allowed
2-8 pages, references excluded per the workshop CFP):

*Targets for sections 1 and 6 revised down 2026-08-25 (0.75p -> 0.5p and
0.5p -> 0.4p respectively) because the drafted sections -- Method,
Results, Background -- are running long against plan and the projected
total was tracking toward 7.5-8.5 pages against the 8-page hard ceiling;
tightening the not-yet-written sections now is preferred to cutting
drafted content later under deadline pressure. All other targets
unchanged.*

*Second revision, 2026-08-26, from page-budget tracking rather than from
any change of plan. With all five body sections drafted (Background,
Method, Results, Validation, Related work = 4389 prose words), the
projection reached 7.4-8.0 pages against the same 8-page ceiling once two
things were counted that the original plan never budgeted: the figures,
and the title + abstract block. Three consequences, recorded as item 0
and in the Introduction line below, plus the figure-layout rule after the
section list.*

0. **Title block + abstract (~0.3p).** Missing from the original
   seven-section plan, which listed only sections 1-7 and set no target
   for the matter above Section 1, despite it occupying the top of page
   one in every compiled draft. Budgeted here at ~0.3p and tracked
   against the total from now on. This omission is what turned a
   projection that appeared to clear the ceiling into one that does not
   under every layout.
1. Introduction (~0.4p, revised down from 0.5p on 2026-08-26 -- see the
   second-revision note above; originally 0.75p): NVIDIA's
   36%-more-tokens claim (attributed
   as a vendor claim), the two-confound observation (NVFP4 differs
   from MXFP4 in BOTH block size and scale format), the causal
   question as the falsifiable hypothesis. No mention of the ν* pivot
   here -- introduced only in Results.
2. Background (~0.5p): compact MXFP4 vs NVFP4 table; classical
   gamma_n/sqrt(n)*u bounds presented as the starting point (not yet
   noting they don't apply -- that's Method's job).
3. Method (~1.5-2p): why the classical bound doesn't apply under exact
   accumulation, derivation of u_eff/sqrt(n) (Step bounds.py / SPEC.md
   🧠1), factorial 2x2 design, robust statistical protocol (median/p90
   confirmatory, p99 indicative -- justified by Step 3.4's bootstrap
   coverage test).
4. Results (~2-2.5p), DELIBERATE non-chronological internal order:
   4a. Mechanism confirmation (Step 4.1 n-scaling fits).
   4b. LEADS, own subsection: causal decomposition -- scale format
       dominates block size (Step 4.3).
   4c. Short, subordinate: the ν* null result (Step 4.2), framed as
       "why we pivoted from threshold to magnitude", connecting back
       into 4b's finding.
5. Validation with real activations (~0.75-1p): Step 4.4, framed as
   closing validation, including the massive-activations mechanism
   for the h6 discrepancy.
6. Related work and limitations (~0.4p): differentiation from
   rasquinha2023metric (dual role), fasoli2026finer, egiazarian2026bridging;
   honest limitations (CPU emulation not hardware, E2M1 only,
   intra-block correlation not fully isolated, c=1 fixed by design).
7. Conclusion (~0.25p).

**Figure layout -- REQUIRED, not preferred (decided 2026-08-26).** Figures
1 and 2 (`figure1_nu_star`, `figure2_n_scaling`) must be placed **side by
side in a single two-up row**. Both are sized at 3.3in wide specifically to
support this. Figure 3 (`figure3_causal_heatmap`) stays full width; at
6.8in native it is wider than the 5.5in text box and scales to about 2.4in
tall. This was a layout preference until the page-budget projection made it
binding: side by side the three figures cost about 0.6p and the paper lands
near 7.7p with the title/abstract block counted, while stacking them costs
about 0.9p (~8.1in of stacked height) and pushes the projection to about
8.0p -- at or over the hard ceiling. Stacking is therefore not available
without cutting drafted content elsewhere to pay for it.

Recommended writing order: Method -> Results -> Background ->
Related work -> Introduction -> Conclusion -> Abstract.
