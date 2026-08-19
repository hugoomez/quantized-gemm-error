"""Entry point for `make run-sweep`.

Sweeps a grid of matmul sizes/dtypes read from a JSON config (default:
configs/default.json). Follows the project's fixed result conventions:
output is Parquet, named sweep_{sha256(config)[:12]}.parquet, with the
full config saved alongside as sweep_{...}.config.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
CONFIGS_DIR = REPO_ROOT / "configs"


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


def run_sweep(config: dict, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for n in tqdm(config["sizes"], desc="sweep"):
        for dtype_name in config["dtypes"]:
            dtype = np.dtype(dtype_name)
            a = rng.standard_normal((n, n)).astype(dtype)
            b = rng.standard_normal((n, n)).astype(dtype)
            c = (a @ b).astype(np.float64)
            rows.append(
                {
                    "n": n,
                    "dtype": dtype_name,
                    "mean_abs": float(np.mean(np.abs(c))),
                    "max_abs": float(np.max(np.abs(c))),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIGS_DIR / "default.json",
        help="Path to a sweep config JSON file.",
    )
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    rng = np.random.default_rng(config["seed"])
    df = run_sweep(config, rng)
    out_path = save_result(df, config)
    print(f"Wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
