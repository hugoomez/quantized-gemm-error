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
per-row). Implemented so far: MXFP4.

### MXFP4 block scaling

Implemented: `quantize_mxfp4(x, block_size=32, rng=None, round_mode="rtne")`
and `mxfp4_block_scales(x, block_size=32)`, float64 in / float64 out.

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

`tests/test_blocks.py` carries the golden-reference comparison against
`microxcaling`: exact equality over random tensors on the blocks where
the two scale rules agree, and an explicit check that `microxcaling`
clips the block maximum on a block where they do not. Both tests
`skip` when `microxcaling` is not importable -- which is the case in this
environment, since `microxcaling` requires `torch` and `torch` is
excluded from the dependency set until Phase 4.4. **The divergence
described above is therefore derived from the MX scale formula, not yet
observed against a running `microxcaling`;** the tests exist so that
adding `torch` in Phase 4.4 either confirms it or fails loudly.

## Rounding modes (`qgemm.rounding`)

TBD -- which rounding modes are compared (round-to-nearest-even,
stochastic rounding, ...).

## Transforms (`qgemm.transforms`)

TBD -- scaling strategy (per-tensor / per-block scale, absmax vs. other
calibration) and any pre-quantization transforms under consideration.

## GEMM under quantization (`qgemm.gemm`)

TBD -- how operands are quantized, what precision accumulation happens
in, and how `quantized_matmul` composes `formats` / `blocks` /
`rounding` / `transforms`.

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
