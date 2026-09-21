"""recalibrate_threshold.py -- cost-sensitive operating-threshold search.

REBUILD NOTE: an earlier version of this script had an unresolved bug where
manual feature-reassembly diverged from cross_dataset_eval.py's validated
path, producing a nonsensical AUC of 0.46. This rebuild does NOT reassemble
any features by hand. It has exactly two modes, both delegating entirely to
already-validated code:

  Mode 1 (--features-dir, default): reload the already-fused test split via
  utils.load_fused_bundle -- the same function every other extension script
  uses -- and sweep thresholds against its existing predict_proba output.

  Mode 2 (--url-csv): score a brand-new URL set by calling
  cross_dataset_eval.assemble_features() directly (imported, not
  reimplemented) to get (X_fused, y), then sweep thresholds the same way.

The thesis fixes tau=0.65 based on a stated cost asymmetry (false negatives
-- missed phishing -- cost more than false positives). This script makes
that asymmetry explicit and configurable via --fn-cost/--fp-cost, and
reports both the best-F1 threshold (threshold-agnostic) and the minimum-
expected-cost threshold (cost-aware), so you can compare against the
thesis's fixed tau=0.65 choice with an actual number instead of intuition.
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

sys.path.append(str(Path(__file__).resolve().parent))
from server.project.scripts.utils import GLOBAL_SEED, get_logger, load_fused_bundle
import server.project.scripts.cross_dataset_eval as cross_dataset_eval

logger = get_logger("recalibrate_threshold")


def sweep(y_true, proba, thresholds, fn_cost=5.0, fp_cost=1.0):
    rows = []
    for t in thresholds:
        pred = (proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        expected_cost = fn * fn_cost + fp * fp_cost
        rows.append({
            "threshold": round(float(t), 3),
            "precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred, zero_division=0),
            "f1": f1_score(y_true, pred, zero_division=0),
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
            "expected_cost": float(expected_cost),
        })
    return pd.DataFrame(rows)


def run(
    results_dir,
    features_dir=None,
    models_dir=None,
    split_name="split_70_30",
    url_csv=None,
    qr_dir=None,
    manifest_path=None,
    model_path=None,
    tfidf_path=None,
    scaler_path=None,
    structural_scaler_path=None,
    backbone="mobilenet_v2",
    fn_cost=5.0,
    fp_cost=1.0,
    thresholds=None,
    seed=GLOBAL_SEED,
):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.02)

    if url_csv is not None:
        # Mode 2: brand-new dataset, delegate entirely to the validated
        # cross_dataset_eval assembly path.
        logger.info("Mode: new-dataset recalibration via cross_dataset_eval.assemble_features (validated path)")
        X_fused, y_true, _ = cross_dataset_eval.assemble_features(
            url_csv, qr_dir, manifest_path, tfidf_path, scaler_path,
            structural_scaler_path=structural_scaler_path, backbone=backbone, seed=seed,
        )
        model = joblib.load(model_path)
        if model.n_features_in_ != X_fused.shape[1]:
            raise ValueError(
                f"Dimension mismatch: model expects {model.n_features_in_} features, "
                f"got {X_fused.shape[1]}. Run verify_artifacts.py to diagnose."
            )
        proba = cross_dataset_eval.chunked_predict_proba(model, X_fused)
    else:
        # Mode 1: reuse the already-fused, already-split test set.
        logger.info(f"Mode: existing split recalibration (features_dir={features_dir}, split={split_name})")
        bundle = load_fused_bundle(features_dir, models_dir=models_dir, split_name=split_name, load_model=True)
        model = bundle["model"]
        if model is None:
            raise FileNotFoundError(f"No trained model found in {models_dir}; run train.py first.")
        X_test, y_true = bundle["X_test"], bundle["y_test"]
        proba = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_true, proba)
    logger.info(f"AUC on this evaluation set: {auc:.4f} (sanity check -- should NOT be near 0.5 or nonsensically low)")
    if auc < 0.6:
        logger.warning(
            "AUC is suspiciously low for a trained model. This usually means a "
            "feature/model mismatch -- run verify_artifacts.py before trusting "
            "the recalibrated threshold below."
        )

    sweep_df = sweep(y_true, proba, thresholds, fn_cost=fn_cost, fp_cost=fp_cost)
    sweep_df.to_csv(results_dir / "recalibrate_threshold_sweep.csv", index=False)

    best_f1_row = sweep_df.loc[sweep_df["f1"].idxmax()]
    best_cost_row = sweep_df.loc[sweep_df["expected_cost"].idxmin()]

    summary = {
        "auc": auc,
        "thesis_fixed_threshold": 0.65,
        "best_f1_threshold": float(best_f1_row["threshold"]),
        "best_f1_value": float(best_f1_row["f1"]),
        "min_cost_threshold": float(best_cost_row["threshold"]),
        "min_expected_cost": float(best_cost_row["expected_cost"]),
        "fn_cost": fn_cost, "fp_cost": fp_cost,
    }
    pd.DataFrame([summary]).to_csv(results_dir / "recalibrate_threshold_summary.csv", index=False)
    logger.info(f"Recalibration summary: {summary}")

    return summary, sweep_df


def build_parser():
    p = argparse.ArgumentParser(description="Cost-sensitive operating-threshold recalibration.")
    p.add_argument("--results-dir", type=Path, required=True)

    # Mode 1: existing split
    p.add_argument("--features-dir", type=Path, default=None)
    p.add_argument("--models-dir", type=Path, default=None)
    p.add_argument("--split-name", type=str, default="split_70_30")

    # Mode 2: new dataset (mutually exclusive with Mode 1; presence of --url-csv triggers it)
    p.add_argument("--url-csv", type=Path, default=None)
    p.add_argument("--qr-dir", type=Path, default=None)
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--model", type=Path, default=None)
    p.add_argument("--tfidf", type=Path, default=None)
    p.add_argument("--scaler", type=Path, default=None)
    p.add_argument("--structural-scaler", type=Path, default=None)
    p.add_argument("--backbone", type=str, default="mobilenet_v2", choices=["mobilenet_v2", "raw_pixels"])

    p.add_argument("--fn-cost", type=float, default=5.0, help="Relative cost of a missed phishing QR (false negative)")
    p.add_argument("--fp-cost", type=float, default=1.0, help="Relative cost of a wrongly-flagged legitimate QR (false positive)")
    p.add_argument("--seed", type=int, default=GLOBAL_SEED)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.url_csv is not None:
        missing = [n for n, v in [
            ("--qr-dir", args.qr_dir), ("--manifest", args.manifest), ("--model", args.model),
            ("--tfidf", args.tfidf), ("--scaler", args.scaler),
        ] if v is None]
        if missing:
            raise SystemExit(f"--url-csv mode requires: {', '.join(missing)}")
    elif args.features_dir is None or args.models_dir is None:
        raise SystemExit("Existing-split mode requires --features-dir and --models-dir")

    run(
        results_dir=args.results_dir, features_dir=args.features_dir, models_dir=args.models_dir,
        split_name=args.split_name, url_csv=args.url_csv, qr_dir=args.qr_dir, manifest_path=args.manifest,
        model_path=args.model, tfidf_path=args.tfidf, scaler_path=args.scaler,
        structural_scaler_path=args.structural_scaler, backbone=args.backbone,
        fn_cost=args.fn_cost, fp_cost=args.fp_cost, seed=args.seed,
    )
