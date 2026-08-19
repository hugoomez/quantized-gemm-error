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

**Tail policy -- differs from `quantize_blocked`.** A trailing partial block is
*not* allowed. `quantize_blocked` keeps a short tail and gives it its own scale,
which is well defined because a scale does not care how many elements it covers;
a partial block has no `H'` to multiply by, and padding it would change the
array's shape and break invertibility. A last axis that is not a whole number of
blocks raises `ValueError`.

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
