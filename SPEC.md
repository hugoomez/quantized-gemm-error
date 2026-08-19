# SPEC

Technical specification for the quantized-GEMM error study. This is the
source of truth for what "correct" means in this codebase; update it
when a design decision is made, not after the fact.

## Scope

TBD -- which quantized formats, block structures, and GEMM shapes this
project studies.

## Quantization formats (`qgemm.formats`)

TBD -- which formats (e.g. float8 variants via `ml_dtypes`, int8) are in
scope, and their `QuantFormat` parameters (storage dtype, finite max).

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

TBD -- block sizes and axes considered (e.g. per-tensor, per-row,
microscaling blocks).

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
