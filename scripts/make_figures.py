"""Entry point for `make figures`.

Reads a results/sweep_*.parquet file (default: the most recently written
one) and renders a summary figure to paper/figures/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "paper" / "figures"


def latest_sweep_result() -> Path:
    candidates = sorted(RESULTS_DIR.glob("sweep_*.parquet"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise SystemExit(
            f"No sweep_*.parquet files in {RESULTS_DIR} -- run `make run-sweep` first."
        )
    return candidates[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=None, help="Sweep result parquet file.")
    args = parser.parse_args()

    in_path = args.input or latest_sweep_result()
    df = pd.read_parquet(in_path, engine="pyarrow")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    for dtype, group in df.groupby("dtype"):
        ax.plot(group["n"], group["max_abs"], marker="o", label=dtype)
    ax.set_xlabel("matrix size (n)")
    ax.set_ylabel("max |C|")
    ax.set_title("GEMM output magnitude vs. size")
    ax.legend()
    fig.tight_layout()

    out_path = FIGURES_DIR / f"{in_path.stem}.png"
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
