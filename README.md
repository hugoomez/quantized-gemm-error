# quantized-gemm-error

**Deadline: August 29, 2026, 23:59 AoE.**
Source: https://newinml.github.io/NewInML2026NeurIPS/
Checked: August 19, 2026.

Numerical research project studying error behavior of quantized GEMM
(general matrix multiply) operations.

## Repository layout

```
src/qgemm/       -- the qgemm package (formats, blocks, rounding, transforms,
                     gemm, metrics, bounds, distributions, stats)
tests/           -- pytest suite
configs/         -- sweep configs (JSON), consumed by scripts/run_sweep.py
scripts/         -- run_sweep.py, sanity_gamma_n.py, sanity_reproduce_2408.py,
                     make_figures.py, record_environment.py
results/         -- sweep_{sha256(config)[:12]}.parquet + matching config JSON
paper/           -- paper drafts; paper/figures/ holds generated figures
SPEC.md          -- technical design, updated as decisions are made
PREREGISTRATION.md -- fixed analysis plan; deviations logged, not silently edited
ENVIRONMENT.md   -- auto-generated: interpreter/OS/CPU/BLAS backend
requirements.lock -- auto-generated: exact pinned dependency versions
```

## Conventions (fixed -- apply to all code)

- Every quantization function receives and returns NumPy float64.
- Every stochastic function takes an explicit `numpy.random.Generator` --
  never NumPy's global RNG state.
- All results are written to Parquet, never ad-hoc formats.
- Results filenames: `sweep_{sha256(config)[:12]}.parquet`, with the
  full config saved alongside (`sweep_{...}.config.json`).

## Setup

```
make install    # creates .venv, installs deps, writes requirements.lock and ENVIRONMENT.md
make test       # run test suite (pytest discovers tests/, `import qgemm` must work)
make lint       # ruff
make run-sweep  # run configs/default.json -> results/
make figures    # run the sweep and render figures -> paper/figures/
```

`torch` (CPU-only) is intentionally not installed yet -- it's needed
starting Phase 4.4 and will be added as an explicit extra at that point.
