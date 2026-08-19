"""Sanity check: reproduce a reference number from arXiv:2408.XXXXX.

TODO: fill in the actual paper ID and the specific table/figure value
this reproduces -- placeholder only, do not treat REFERENCE_VALUE below
as a real target.

Follows the same conventions as the other sanity scripts: float64 in/out
for quantization-related calls, explicit RNG, and results written via
the sweep_{sha256(config)[:12]}.parquet convention.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"

PAPER_ID = "2408.XXXXX"  # TODO: fill in
REFERENCE_VALUE = None  # TODO: fill in the target value from the paper
REFERENCE_TOLERANCE = 0.05  # TODO: fill in an appropriate tolerance


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


def reproduce(config: dict, rng: np.random.Generator) -> float:
    """TODO: implement the experiment that should match arXiv:{PAPER_ID}."""
    raise NotImplementedError(
        f"sanity_reproduce_2408: fill in the reproduction of {PAPER_ID} here"
    )


def main() -> None:
    if REFERENCE_VALUE is None:
        raise SystemExit(
            "sanity_reproduce_2408: PAPER_ID/REFERENCE_VALUE are still placeholders -- "
            "fill them in before running this check."
        )

    config = {"paper_id": PAPER_ID, "seed": 0}
    rng = np.random.default_rng(config["seed"])

    reproduced_value = reproduce(config, rng)
    df = pd.DataFrame(
        [
            {
                "paper_id": PAPER_ID,
                "reproduced_value": reproduced_value,
                "reference_value": REFERENCE_VALUE,
                "tolerance": REFERENCE_TOLERANCE,
                "within_tolerance": abs(reproduced_value - REFERENCE_VALUE) <= REFERENCE_TOLERANCE,
            }
        ]
    )
    out_path = save_result(df, config)
    print(df.to_string(index=False))

    if not df["within_tolerance"].all():
        raise SystemExit(f"reproduction of {PAPER_ID} FAILED -- see {out_path}")
    print(f"Reproduction of {PAPER_ID} passed. Wrote {out_path}")


if __name__ == "__main__":
    main()
