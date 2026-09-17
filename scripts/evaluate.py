"""evaluate.py -- Phase 5

Standard metrics at threshold 0.65, ROC/PR curves, confusion matrix,
gain-based feature importance, per-sample top-3 explanations, threshold
sweep, optimal-threshold search, and a baseline (Config 1) vs fused
(Config 7) comparison table.

FIX applied here (see REBUILD SPEC section 13): the baseline-vs-proposed
comparison loads the ACTUAL saved tuned model (models/xgboost_fused.pkl)
for the "Ours" row instead of retraining a fresh untuned one.
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
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score, roc_curve,
)

sys.path.append(str(Path(__file__).resolve().parent))
from utils import get_logger
from train import ABLATION_CONFIGS, FIXED_ABLATION_PARAMS, slice_branches

logger = get_logger("evaluate")

DEFAULT_THRESHOLD = 0.65


def compute_metrics(y_true, proba, threshold=DEFAULT_THRESHOLD):
    pred = (proba >= threshold).astype(int)
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "auc": roc_auc_score(y_true, proba),
    }


def threshold_sweep(y_true, proba, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.1, 0.95, 0.05)
    rows = []
    for t in thresholds:
        pred = (proba >= t).astype(int)
        rows.append({
            "threshold": round(float(t), 2),
            "accuracy": accuracy_score(y_true, pred),
            "precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred, zero_division=0),
            "f1": f1_score(y_true, pred, zero_division=0),
        })
    return pd.DataFrame(rows)


def find_optimal_threshold(y_true, proba):
    precisions, recalls, thresholds = precision_recall_curve(y_true, proba)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-12)
    best_idx = int(np.argmax(f1s[:-1])) if len(thresholds) > 0 else 0
    best_threshold = float(thresholds[best_idx]) if len(thresholds) > 0 else DEFAULT_THRESHOLD
    return {"best_f1_threshold": best_threshold, "best_f1": float(f1s[best_idx])}


def feature_importance(model, feature_names, top_n=30):
    booster = model.get_booster()
    gain_scores = booster.get_score(importance_type="gain")
    # xgboost feature names are f0, f1, ... map back to real names
    rows = []
    for fkey, gain in gain_scores.items():
        idx = int(fkey[1:])
        name = feature_names[idx] if idx < len(feature_names) else fkey
        rows.append({"feature": name, "gain": gain})
    df = pd.DataFrame(rows).sort_values("gain", ascending=False).reset_index(drop=True)
    return df.head(top_n)


def per_sample_top3(model, X, feature_names):
    booster = model.get_booster()
    import xgboost as xgb
    dmat = xgb.DMatrix(X, feature_names=[f"f{i}" for i in range(X.shape[1])])
    contribs = booster.predict(dmat, pred_contribs=True)  # (n, n_features+1), last col is bias
    explanations = []
    for row in contribs:
        feat_contribs = row[:-1]
        top_idx = np.argsort(-np.abs(feat_contribs))[:3]
        explanations.append([
            {"feature": feature_names[i] if i < len(feature_names) else f"f{i}", "contribution": float(feat_contribs[i])}
            for i in top_idx
        ])
    return explanations


def baseline_vs_fused(features_dir, models_dir, results_dir, split_name="split_70_30", seed=42):
    from xgboost import XGBClassifier

    features_dir = Path(features_dir)
    models_dir = Path(models_dir)

    split = np.load(features_dir / f"{split_name}.npz")
    X_train, X_test, y_train, y_test = split["X_train"], split["X_test"], split["y_train"], split["y_test"]
    branch_boundaries = np.load(features_dir / "branch_boundaries.npy", allow_pickle=True).item()

    # Baseline: Config 1 (structural only), fixed untuned params
    Xtr_struct = slice_branches(X_train, branch_boundaries, ABLATION_CONFIGS["1_structural_only"])
    Xte_struct = slice_branches(X_test, branch_boundaries, ABLATION_CONFIGS["1_structural_only"])
    baseline_model = XGBClassifier(**FIXED_ABLATION_PARAMS, eval_metric="logloss", random_state=seed, n_jobs=-1)
    baseline_model.fit(Xtr_struct, y_train)
    baseline_proba = baseline_model.predict_proba(Xte_struct)[:, 1]
    baseline_metrics = compute_metrics(y_test, baseline_proba)
    baseline_metrics["config"] = "1_structural_only (baseline)"

    # Ours: load the ACTUAL tuned saved model, do not retrain.
    model_path = models_dir / "xgboost_fused.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Expected trained model at {model_path}; run train.py first.")
    fused_model = joblib.load(model_path)
    fused_proba = fused_model.predict_proba(X_test)[:, 1]
    fused_metrics = compute_metrics(y_test, fused_proba)
    fused_metrics["config"] = "7_all_fused (ours, tuned)"

    comparison = pd.DataFrame([baseline_metrics, fused_metrics])
    return comparison, fused_model, X_test, y_test, fused_proba


def run(features_dir, models_dir, results_dir, split_name="split_70_30", seed=42):
    features_dir = Path(features_dir)
    models_dir = Path(models_dir)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(features_dir / "feature_names.txt") as f:
        feature_names = f.read().splitlines()

    comparison, fused_model, X_test, y_test, fused_proba = baseline_vs_fused(
        features_dir, models_dir, results_dir, split_name, seed
    )
    comparison.to_csv(results_dir / "baseline_vs_fused.csv", index=False)
    logger.info(f"Baseline vs fused:\n{comparison}")

    metrics = compute_metrics(y_test, fused_proba)
    pd.DataFrame([metrics]).to_csv(results_dir / "final_metrics.csv", index=False)

    fpr, tpr, _ = roc_curve(y_test, fused_proba)
    pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(results_dir / "roc_curve.csv", index=False)

    precisions, recalls, _ = precision_recall_curve(y_test, fused_proba)
    pd.DataFrame({"precision": precisions, "recall": recalls}).to_csv(results_dir / "pr_curve.csv", index=False)

    pred = (fused_proba >= DEFAULT_THRESHOLD).astype(int)
    cm = confusion_matrix(y_test, pred)
    pd.DataFrame(cm, index=["actual_legit", "actual_phish"], columns=["pred_legit", "pred_phish"]).to_csv(
        results_dir / "confusion_matrix.csv"
    )

    importance_df = feature_importance(fused_model, feature_names)
    importance_df.to_csv(results_dir / "feature_importance.csv", index=False)

    sweep_df = threshold_sweep(y_test, fused_proba)
    sweep_df.to_csv(results_dir / "threshold_sweep.csv", index=False)

    optimal = find_optimal_threshold(y_test, fused_proba)
    logger.info(f"Optimal threshold search: {optimal}")
    pd.DataFrame([optimal]).to_csv(results_dir / "optimal_threshold.csv", index=False)

    logger.info(f"Evaluation complete. Results -> {results_dir}")
    return comparison, metrics


def build_parser():
    p = argparse.ArgumentParser(description="Evaluate the trained fused model (Phase 5).")
    p.add_argument("--features-dir", type=Path, required=True)
    p.add_argument("--models-dir", type=Path, required=True)
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--split-name", type=str, default="split_70_30")
    p.add_argument("--seed", type=int, default=42)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run(args.features_dir, args.models_dir, args.results_dir, args.split_name, args.seed)
