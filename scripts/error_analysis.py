"""error_analysis.py -- characterizes the final tuned model's misclassified
samples (false positives / false negatives) on the test split.

Joins predictions back to the original manifest/structural CSV so errors
can be inspected by real attributes (QR version, ECC level, URL length,
whether the QR payload decoded successfully, etc.) rather than just raw
feature indices. Reuses the SAVED tuned model and the SAME fused test split
that produced the headline metrics -- no re-fitting anything.
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from utils import get_logger, load_fused_bundle
from structural import FEATURE_NAMES as STRUCTURAL_FEATURE_NAMES

logger = get_logger("error_analysis")

DEFAULT_THRESHOLD = 0.65


def run(features_dir, models_dir, results_dir, manifest_path, structural_csv,
        split_name="split_70_30", threshold=DEFAULT_THRESHOLD):
    features_dir = Path(features_dir)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_fused_bundle(features_dir, models_dir=models_dir, split_name=split_name, load_model=True)
    model = bundle["model"]
    if model is None:
        raise FileNotFoundError(f"No trained model found in {models_dir}; run train.py first.")

    X_test, y_test, idx_test = bundle["X_test"], bundle["y_test"], bundle["idx_test"]

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)

    manifest = pd.read_csv(manifest_path)
    struct_df = pd.read_csv(structural_csv).set_index("filename").loc[manifest["filename"]].reset_index()

    # idx_test indexes into the ORIGINAL manifest row order (same alignment
    # fusion.py used when it built fused_features.npy), so this join is exact.
    test_rows = struct_df.iloc[idx_test].reset_index(drop=True)
    test_rows["true_label"] = y_test
    test_rows["predicted_proba"] = proba
    test_rows["predicted_label"] = pred

    def outcome(row):
        if row["true_label"] == 1 and row["predicted_label"] == 1:
            return "TP"
        if row["true_label"] == 0 and row["predicted_label"] == 0:
            return "TN"
        if row["true_label"] == 0 and row["predicted_label"] == 1:
            return "FP"
        return "FN"

    test_rows["outcome"] = test_rows.apply(outcome, axis=1)

    out_full = results_dir / "error_analysis_full.csv"
    test_rows.to_csv(out_full, index=False)

    errors = test_rows[test_rows["outcome"].isin(["FP", "FN"])].copy()
    out_errors = results_dir / "error_analysis_errors_only.csv"
    errors.to_csv(out_errors, index=False)

    logger.info(
        f"Outcome counts: {test_rows['outcome'].value_counts().to_dict()} "
        f"(threshold={threshold})"
    )

    # Breakdown of FN/FP by structural protocol-level attributes, since these
    # are cheap, interpretable groupings that map directly to the thesis's
    # protocol-level feature story.
    breakdown_rows = []
    for outcome_type in ["FN", "FP"]:
        subset = test_rows[test_rows["outcome"] == outcome_type]
        if len(subset) == 0:
            continue
        for col in ["version", "ecc_level", "masking_pattern"]:
            if col in subset.columns:
                counts = subset[col].value_counts().to_dict()
                breakdown_rows.append({"outcome": outcome_type, "attribute": col, "value_counts": str(counts)})

    breakdown_df = pd.DataFrame(breakdown_rows)
    out_breakdown = results_dir / "error_analysis_breakdown.csv"
    breakdown_df.to_csv(out_breakdown, index=False)

    # Summary stats: how much lower is the average predicted probability for
    # errors vs. correct predictions? Useful for the "near-miss vs
    # confidently-wrong" framing in a defense Q&A.
    summary = {
        "n_test": len(test_rows),
        "n_fp": int((test_rows["outcome"] == "FP").sum()),
        "n_fn": int((test_rows["outcome"] == "FN").sum()),
        "mean_proba_correct": float(test_rows.loc[test_rows["outcome"].isin(["TP", "TN"]), "predicted_proba"].mean()),
        "mean_proba_fp": float(test_rows.loc[test_rows["outcome"] == "FP", "predicted_proba"].mean()) if (test_rows["outcome"] == "FP").any() else None,
        "mean_proba_fn": float(test_rows.loc[test_rows["outcome"] == "FN", "predicted_proba"].mean()) if (test_rows["outcome"] == "FN").any() else None,
        "threshold": threshold,
    }
    pd.DataFrame([summary]).to_csv(results_dir / "error_analysis_summary.csv", index=False)
    logger.info(f"Error summary: {summary}")

    logger.info(f"Wrote error analysis -> {out_full}, {out_errors}, {out_breakdown}")
    return test_rows


def build_parser():
    p = argparse.ArgumentParser(description="Characterize misclassified test-set samples.")
    p.add_argument("--features-dir", type=Path, required=True)
    p.add_argument("--models-dir", type=Path, required=True)
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--structural-csv", type=Path, required=True)
    p.add_argument("--split-name", type=str, default="split_70_30")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run(
        features_dir=args.features_dir, models_dir=args.models_dir, results_dir=args.results_dir,
        manifest_path=args.manifest, structural_csv=args.structural_csv,
        split_name=args.split_name, threshold=args.threshold,
    )
