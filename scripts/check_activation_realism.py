"""Step 4.4 -- real-activation realism check (EXPLORATORY).

Every result so far in this project (Steps 4.1-4.3) was measured on synthetic
t-Student(nu) operands. That is deliberate study infrastructure -- nu is a
clean, sweepable axis -- but it leaves one question untested: does the
t-Student model's error prediction actually transfer to what a real network
produces? PREREGISTRATION.md sec 5 lists "real activations" explicitly under
"exploratory, reported as such, never as evidence for or against H1" -- this
script is that exploratory check, not a fourth confirmatory result.

What it does
------------
1. Loads GPT-2 small and extracts real input activations at three attention
   input projections (`h[0]`, `h[6]`, `h[11]`'s `attn.c_attn`) over real
   WikiText-2 text.
2. Computes each layer's excess kurtosis (Fisher convention, gaussian=0) on
   the non-padded activations, and maps it to an "equivalent nu" via the
   t-Student closed form `excess_kurtosis = 6/(nu-4)`.
3. Runs this project's own `qgemm` pipeline (MXFP4, NVFP4) with the REAL
   activations as operand A and the REAL weight matrix from the same layer
   as operand B -- not synthetic data -- and measures `median(BE)`.
4. Runs the identical pipeline with synthetic t-Student(nu_equivalent) data
   in A's place, rescaled to match the real activations' own MAD (so the
   comparison isolates tail *shape*, not scale -- the same normalization
   rationale as "Cross-distribution normalization (Step 3.3)" in SPEC.md),
   and the SAME real weight matrix as B.
5. Compares the two `median(BE)` numbers.

Nothing here touches nu*, cota(n), or PREREGISTRATION.md's break criterion --
this is a transfer check of the t-Student model against real data, at the
scale a single forward pass produces, not a bound-violation test.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from scipy.stats import kurtosis as scipy_kurtosis
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from qgemm.gemm import GemmConfig, qgemm
from qgemm.metrics import backward_error
from qgemm.stats import median_absolute_deviation

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "results" / "analysis" / "activation_realism"

# `sample_t` is reused from `scripts/check_metric_stability.py`, exactly the
# way `scripts/measure_u_eff.py` already loads it -- PREREGISTRATION.md R15
# wants the nu-axis sampler implemented and tested on its own terms before it
# moves into `qgemm.distributions`, so it stays local to that script and is
# imported by path rather than duplicated here.
_CMS_PATH = REPO_ROOT / "scripts" / "check_metric_stability.py"
_spec = importlib.util.spec_from_file_location("check_metric_stability", _CMS_PATH)
_cms = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cms)
sample_t = _cms.sample_t

MODEL_NAME = "gpt2"
# Fix (b): the old bare "wikitext" repo id's loading script is broken under
# datasets>=5 (HfHubHTTPError / dataset-script removal); "Salesforce/wikitext"
# is the namespaced, non-script-based repo that still works.
DATASET_REPO = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
DATASET_SPLIT = "test"

N_SEQUENCES = 300  # matches the feasibility scope (~85s full-scope forward pass)
BATCH_SIZE = 16
MAX_SEQ_LENGTH = 256  # truncation safety net; WikiText-2 lines run far shorter on average
SEED = 0

# Layer choice: the attention input projection (`attn.c_attn`), at three
# depths of GPT-2 small's 12 transformer blocks -- first, middle, last.
# `c_attn` (not `mlp.c_fc`) was chosen because it is the very first linear op
# inside each block (applied directly to `ln_1`'s output), which keeps the
# hook point uniform and the comparison as simple as possible; the choice is
# held fixed across all three layers rather than mixed with `mlp.c_fc`, so
# depth is the only thing that varies between rows of the output table.
LAYER_INDICES: dict[str, int] = {"h0": 0, "h6": 6, "h11": 11}

# GPT-2's Conv1D stores its weight as (in_features, out_features) -- i.e.
# already (K, N) in this project's A @ B convention -- so it is used directly
# as operand B with no transpose, unlike an `nn.Linear` weight would need.

PRESETS: dict[str, GemmConfig] = {
    "mxfp4": GemmConfig(block_size=32, scale_format="e8m0", use_global_scale=False),
    "nvfp4": GemmConfig(block_size=16, scale_format="e4m3", use_global_scale=True),
}


def config_digest(config: dict) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def load_texts(n_sequences: int) -> list[str]:
    ds = load_dataset(DATASET_REPO, DATASET_CONFIG, split=DATASET_SPLIT)
    lines = [t for t in ds["text"] if t.strip()]
    if len(lines) < n_sequences:
        raise ValueError(f"only {len(lines)} non-empty lines available, need {n_sequences}")
    return lines[:n_sequences]


def extract_activations(
    texts: list[str], layer_indices: dict[str, int], batch_size: int, max_length: int
) -> dict[str, np.ndarray]:
    """Real, non-padded input activations per layer, float64, shape (n_valid_tokens, hidden).

    Padding-aware by construction: the attention mask is applied per batch,
    immediately after the forward pass, so padded positions never enter the
    accumulated array at all -- there is nothing downstream that could be
    contaminated by them, kurtosis or GEMM alike.
    """
    tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
    # Fix (a): GPT-2's tokenizer has no pad_token by default; batched
    # tokenization with padding raises without this.
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    model.eval()

    captured: dict[str, torch.Tensor] = {}
    handles = []
    for name, idx in layer_indices.items():
        block = model.transformer.h[idx]

        def make_hook(name=name):
            def hook(module, inputs):
                captured[name] = inputs[0]

            return hook

        handles.append(block.attn.c_attn.register_forward_pre_hook(make_hook()))

    rows: dict[str, list[np.ndarray]] = {name: [] for name in layer_indices}
    try:
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            # Dynamic per-batch padding (`padding=True` pads to the longest
            # sequence *in this batch*, not to a fixed max_length) -- shorter
            # sequences elsewhere in the corpus are not padded out for it.
            enc = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            captured.clear()
            with torch.no_grad():
                model(**enc)
            mask = enc["attention_mask"].bool().numpy()  # (batch, seq)
            for name in layer_indices:
                act = captured[name].detach().to(torch.float64).numpy()  # (batch, seq, hidden)
                rows[name].append(act[mask])  # (n_valid_tokens_in_batch, hidden)
    finally:
        for h in handles:
            h.remove()

    weights = {
        name: model.transformer.h[idx].attn.c_attn.weight.detach().to(torch.float64).numpy()
        for name, idx in layer_indices.items()
    }
    activations = {name: np.concatenate(parts, axis=0) for name, parts in rows.items()}
    return activations, weights


def excess_kurtosis(x: np.ndarray) -> float:
    """Fisher's (excess) kurtosis: Gaussian -> 0. NOT the raw/Pearson convention (Gaussian -> 3).

    Computed over the full flattened array (all valid tokens x all hidden
    features together) -- this project's t-Student nu grid treats an operand
    as one elementwise distribution, not one per feature column, so the
    kurtosis compared against it is taken the same way. `bias=True` (scipy's
    default, the plain biased/population-moment estimator) is used throughout;
    with tens of thousands of samples the bias correction is negligible.
    """
    return float(scipy_kurtosis(np.ravel(x), fisher=True, bias=True))


def nu_equivalent(excess_k: float) -> tuple[float | None, str | None]:
    """t-Student nu whose population excess kurtosis equals `excess_k`, or (None, reason).

    Closed form: `excess_kurtosis = 6 / (nu - 4)` for `nu > 4` (undefined,
    i.e. infinite, for `nu <= 4`). Inverting: `nu = 4 + 6 / excess_k`. For any
    finite `excess_k > 0` this is algebraically always `> 4` -- the map
    `nu -> 6/(nu-4)` sends `(4, inf)` onto `(0, inf)`, so its inverse can
    never land back at `nu <= 4` from a finite positive input. The guard
    below is kept anyway (never silently extrapolate past a formula's
    domain), but given finite real data it cannot fire; only `excess_k <= 0`
    is genuinely out of domain -- no t-Student at any nu > 4 has zero or
    negative excess kurtosis, so a layer measured at or below the Gaussian
    value has no equivalent-nu to report.
    """
    if excess_k <= 0.0:
        return None, (
            f"excess kurtosis {excess_k:.4f} <= 0: no t-Student nu (which requires nu>4) "
            "has this little or negative excess kurtosis; undefined, not extrapolated"
        )
    nu = 4.0 + 6.0 / excess_k
    if nu <= 4.0:  # unreachable for finite excess_k > 0; kept as a defensive check, see docstring
        return None, f"implied nu={nu:.4f} <= 4, outside the closed form's domain; not reported"
    return nu, None


def synthetic_operand(
    shape: tuple[int, int], nu: float, target_mad: float, rng: np.random.Generator
) -> np.ndarray:
    """t-Student(nu) sample rescaled to match `target_mad`, isolating tail shape from scale.

    Same rationale as SPEC.md's "Cross-distribution normalization": MAD
    (not std, undefined for nu<=2) is what is matched, so the synthetic and
    real operands differ only in their tail shape, not their raw scale.
    """
    raw = sample_t(shape, nu, rng)  # normalize=None: unscaled, so the rescale below is exact
    return raw / median_absolute_deviation(raw) * target_mad


def run_cell(
    real_a: np.ndarray, weight_b: np.ndarray, nu: float | None, config: GemmConfig, rng
) -> dict:
    """One (layer, preset) cell: observed (real A) and predicted (synthetic A) median(BE)."""
    chat_real = qgemm(real_a, weight_b, config)
    be_real = backward_error(real_a, weight_b, chat_real)
    observed_median_be = float(np.median(be_real))

    if nu is None:
        return {"observed_median_be": observed_median_be, "predicted_median_be": None}

    target_mad = median_absolute_deviation(real_a)
    synth_a = synthetic_operand(real_a.shape, nu, target_mad, rng)
    chat_synth = qgemm(synth_a, weight_b, config)
    be_synth = backward_error(synth_a, weight_b, chat_synth)
    predicted_median_be = float(np.median(be_synth))

    return {"observed_median_be": observed_median_be, "predicted_median_be": predicted_median_be}


def verdict_for(ratio: float | None) -> str:
    if ratio is None:
        return "N/A (no equivalent nu)"
    excess = abs(ratio - 1.0)
    if excess <= 0.10:
        return f"CLOSE (within 10%; observed/predicted = {ratio:.3f})"
    if excess <= 0.50:
        return f"MODERATE GAP (observed/predicted = {ratio:.3f})"
    return f"LARGE GAP (observed/predicted = {ratio:.3f})"


def build_table(
    activations: dict[str, np.ndarray], weights: dict[str, np.ndarray], seed: int
) -> pd.DataFrame:
    rows = []
    for layer_idx, layer_name in enumerate(LAYER_INDICES):
        act = activations[layer_name]
        weight = weights[layer_name]
        ek = excess_kurtosis(act)
        nu, nu_note = nu_equivalent(ek)
        for preset_idx, (preset_name, config) in enumerate(PRESETS.items()):
            # Deterministic seed derivation, not Python's `hash()` (salted per
            # process by default and therefore not reproducible run to run).
            rng = np.random.default_rng([seed, layer_idx, preset_idx])
            cell = run_cell(act, weight, nu, config, rng)
            observed = cell["observed_median_be"]
            predicted = cell["predicted_median_be"]
            ratio = (observed / predicted) if predicted is not None else None
            rows.append(
                {
                    "layer": layer_name,
                    "n_tokens": int(act.shape[0]),
                    "excess_kurtosis": ek,
                    "nu_equivalent": nu,
                    "nu_note": nu_note,
                    "preset": preset_name,
                    "observed_median_be": observed,
                    "predicted_median_be": predicted,
                    "ratio_observed_over_predicted": ratio,
                    "verdict": verdict_for(ratio),
                }
            )
    return pd.DataFrame(rows)


def build_statement(table: pd.DataFrame, n_sequences: int, runtime_s: float) -> str:
    lines: list[str] = []
    lines.append(
        f"Step 4.4 real-activation realism check (EXPLORATORY). GPT-2 small, WikiText-2 "
        f"test split, {n_sequences} sequences, 3 layers x 2 presets = "
        f"{table['layer'].nunique() * table['preset'].nunique()} cells. Runtime {runtime_s:.1f}s."
    )
    lines.append("")
    for _, row in table.iterrows():
        prefix = f"  {row['layer']:>4s} / {row['preset']:<6s}: ek={row['excess_kurtosis']:.3f}, "
        if row["nu_equivalent"] is None:
            lines.append(
                prefix + f"nu_equivalent=N/A ({row['nu_note']}), "
                f"observed median(BE)={row['observed_median_be']:.6f}, predicted=N/A"
            )
        else:
            lines.append(
                prefix + f"nu_equivalent={row['nu_equivalent']:.3f}, "
                f"observed median(BE)={row['observed_median_be']:.6f}, "
                f"predicted median(BE)={row['predicted_median_be']:.6f}, {row['verdict']}"
            )
    lines.append("")
    scored = table[table["ratio_observed_over_predicted"].notna()]
    if len(scored) > 0:
        close = (scored["ratio_observed_over_predicted"].sub(1.0).abs() <= 0.10).sum()
        ratios = scored["ratio_observed_over_predicted"]
        lines.append(
            f"{close}/{len(scored)} cells land within 10% of the t-Student prediction "
            f"(observed/predicted ratio range: {ratios.min():.3f} - {ratios.max():.3f})."
        )
    lines.append("")
    lines.append(
        "EXPLORATORY: this does not touch nu*, cota(n), or the PREREGISTRATION.md sec 3 break "
        "criterion. Per PREREGISTRATION.md sec 5, real activations are exploratory and are not "
        "evidence for or against H1."
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-sequences", type=int, default=N_SEQUENCES)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    assert not torch.cuda.is_available(), "this project's discipline is CPU-only torch"

    run_config = {
        "script": "check_activation_realism.py",
        "version": 1,
        "model": MODEL_NAME,
        "dataset_repo": DATASET_REPO,
        "dataset_config": DATASET_CONFIG,
        "dataset_split": DATASET_SPLIT,
        "n_sequences": args.n_sequences,
        "batch_size": args.batch_size,
        "max_seq_length": MAX_SEQ_LENGTH,
        "layer_indices": LAYER_INDICES,
        "hook_point": "attn.c_attn (input)",
        "presets": {k: k for k in PRESETS},
        "kurtosis_convention": "fisher_excess_gaussian_zero",
        "seed": args.seed,
    }
    digest = config_digest(run_config)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"activation_realism_{digest}_EXPLORATORY"

    started = time.perf_counter()
    texts = load_texts(args.n_sequences)
    activations, weights = extract_activations(
        texts, LAYER_INDICES, args.batch_size, MAX_SEQ_LENGTH
    )
    table = build_table(activations, weights, args.seed)
    runtime_s = time.perf_counter() - started

    table.to_parquet(OUT_DIR / f"{stem}.parquet", engine="pyarrow")
    table.to_csv(OUT_DIR / f"{stem}_table.csv", index=False)
    (OUT_DIR / f"{stem}.config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True), encoding="utf-8"
    )

    statement = build_statement(table, args.n_sequences, runtime_s)
    print("\n" + statement + "\n")
    (OUT_DIR / f"{stem}_statement.txt").write_text(statement + "\n", encoding="utf-8")

    print("TABLE:")
    print(
        table[
            [
                "layer",
                "n_tokens",
                "excess_kurtosis",
                "nu_equivalent",
                "preset",
                "observed_median_be",
                "predicted_median_be",
                "ratio_observed_over_predicted",
                "verdict",
            ]
        ].to_string(index=False)
    )
    print(f"\nWrote table, config, statement to {OUT_DIR} (stem={stem})")


if __name__ == "__main__":
    main()
