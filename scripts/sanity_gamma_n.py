"""Sanity check: empirical float32 summation error vs. Higham's gamma_n bound.

For each vector length n, sums n uniform(-1, 1) samples in float32 and
compares against a float64 reference. The relative error (normalized by
sum(|x_i|)) must never exceed gamma_n(n, u_float32) -- see qgemm.bounds
for why this holds regardless of summation order.

Writes a result via the same sweep_{sha256(config)[:12]}.parquet
convention as scripts/run_sweep.py.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from qgemm.bounds import gamma_n
from qgemm.distributions import sample_uniform

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"

U_FLOAT32 = 2.0**-24  # unit roundoff for float32, round-to-nearest


def config_digest(config: dict) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def save_result(df: pd.DataFrame, config: dict) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    digest = config_digest(config)
    out_path = RESULTS_DIR / f"sweep_{digest}.parquet"
    df.to_parquet(out_path, engine="pyarrow")
    (RESULTS_DIR / f"sweep_{digest}.config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
    )
    return out_path


def summation_relative_errors(n: int, trials: int, rng: np.random.Generator) -> np.ndarray:
    errors = np.empty(trials, dtype=np.float64)
    for i in range(trials):
        x = sample_uniform((n,), rng, low=-1.0, high=1.0)
        reference = float(np.sum(x, dtype=np.float64))
        approx = float(np.sum(x.astype(np.float32), dtype=np.float32))
        denom = float(np.sum(np.abs(x)))
        errors[i] = abs(approx - reference) / denom if denom > 0 else 0.0
    return errors


def main() -> None:
    config = {"ns": [8, 16, 32, 64, 128, 256], "trials": 200, "seed": 0}
    rng = np.random.default_rng(config["seed"])

    rows = []
    for n in config["ns"]:
        errors = summation_relative_errors(n, config["trials"], rng)
        bound = gamma_n(n, U_FLOAT32)
        rows.append(
            {
                "n": n,
                "max_relative_error": float(np.max(errors)),
                "mean_relative_error": float(np.mean(errors)),
                "gamma_n_bound": bound,
                "within_bound": bool(np.max(errors) <= bound),
            }
        )

    df = pd.DataFrame(rows)
    out_path = save_result(df, config)
    print(df.to_string(index=False))

    if not df["within_bound"].all():
        raise SystemExit(f"gamma_n sanity check FAILED -- see {out_path}")
    print(f"gamma_n sanity check passed. Wrote {out_path}")


if __name__ == "__main__":
    main()
