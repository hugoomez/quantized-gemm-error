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

**On how to read "inconsistent with sqrt(n)" above.** Higham & Mary (2019)
and Connolly-Higham-Mary (2021) establish `sqrt(n)*u` (respectively
`sqrt(n log n)*u` for the refined variant) as a **high-probability upper
bound** on sequential-summation error under random rounding -- not a
statement about the typical or expected growth rate. The script's raw
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

### Dry-run timing estimate (post-cut, real multiprocessing measured)

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

**Verdict: this does NOT reliably clear the 8h gate, and is not a clean
pass.** The MEASURED headline figure exceeded 8h in 1 of 5 independent runs
(8.12h); the cross-check exceeded it in 2 of 5 (8.49h, 9.09h); and even the
serial (single-core) estimate -- which exceeded 8h in every one of the five
pre-cut runs -- now clears it in 2 of 5 post-cut runs. The cut roughly halved
the grid's compute as expected, but the honest read of five independent
measurements is "genuinely borderline, could go either way on a given
invocation," not "clears with margin." Per this step's own instructions, if
the measured estimate exceeds 8h that is reported plainly and this stops
here: **it exceeded 8h at least once, so no further cut is applied or decided
in this step.** Whether the block_size cut alone is sufficient, or whether a
second cut (or accepting the risk, or a controlled/quieter benchmarking
environment) is warranted, is a human call, to be made **before** anything is
launched -- not made here.
