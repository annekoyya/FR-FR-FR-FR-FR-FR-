"""multi_split_experiment.py -- quantifies result stability across split
configurations and random seeds.

The thesis compares 70/30 vs 80/20 (Table 12) and states multi-split
stability as evidence against overfitting. This script formalizes that:

  Mode A (default, fast): train+evaluate the FINAL saved model's config
  (fixed ablation params, matching train.py's fixed-param philosophy for
  fair comparison) on both existing split_70_30 and split_80_20 files
  already produced by fusion.py, reporting AUC/accuracy for each plus the
  spread between them.

  Mode B (--n-seeds > 1, slower): additionally re-runs fusion.py's split
  step with N different random seeds (holding the underlying structural/
  visual/lexical features fixed) to get a distribution of AUC values and
  report mean +/- std -- directly reproducing the "std ~= 0.00008" style
  stability claim from the original validated run.
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parent))
import server.project.scripts.fusion as fusion_module
from server.project.scripts.train import FIXED_ABLATION_PARAMS
from server.project.scripts.utils import GLOBAL_SEED, get_logger, load_fused_bundle, set_seed

logger = get_logger("multi_split_experiment")


def _train_eval_fixed(X_train, X_test, y_train, y_test, seed=GLOBAL_SEED):
    from xgboost import XGBClassifier

    model = XGBClassifier(**FIXED_ABLATION_PARAMS, eval_metric="logloss", random_state=seed, n_jobs=-1)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.65).astype(int)
    return {"auc": roc_auc_score(y_test, proba), "accuracy": accuracy_score(y_test, pred)}


def mode_a_existing_splits(features_dir, split_names, seed=GLOBAL_SEED):
    rows = []
    for split_name in split_names:
        bundle = load_fused_bundle(features_dir, models_dir=None, split_name=split_name, load_model=False)
        metrics = _train_eval_fixed(bundle["X_train"], bundle["X_test"], bundle["y_train"], bundle["y_test"], seed=seed)
        rows.append({"split_name": split_name, **metrics})
        logger.info(f"{split_name}: {metrics}")
    return pd.DataFrame(rows)


def mode_b_multi_seed(fused_features_path, labels_path, test_size, n_seeds, base_seed=GLOBAL_SEED):
    X = np.load(fused_features_path)
    y = np.load(labels_path)

    rows = []
    for i in range(n_seeds):
        seed = base_seed + i
        idx = np.arange(len(y))
        idx_train, idx_test = train_test_split(idx, test_size=test_size, stratify=y, random_state=seed)
        metrics = _train_eval_fixed(X[idx_train], X[idx_test], y[idx_train], y[idx_test], seed=seed)
        rows.append({"seed": seed, "test_size": test_size, **metrics})
        logger.info(f"seed={seed}, test_size={test_size}: {metrics}")
    return pd.DataFrame(rows)


def run(features_dir, results_dir, split_names=("split_70_30", "split_80_20"),
        n_seeds=1, multi_seed_test_size=0.30, seed=GLOBAL_SEED):
    set_seed(seed)
    features_dir = Path(features_dir)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    df_a = mode_a_existing_splits(features_dir, split_names, seed=seed)
    df_a.to_csv(results_dir / "multi_split_existing_splits.csv", index=False)

    spread = {
        "split_names_compared": list(split_names),
        "auc_mean": float(df_a["auc"].mean()),
        "auc_std": float(df_a["auc"].std()) if len(df_a) > 1 else 0.0,
        "auc_range": float(df_a["auc"].max() - df_a["auc"].min()) if len(df_a) > 1 else 0.0,
        "best_split": str(df_a.loc[df_a["auc"].idxmax(), "split_name"]),
    }
    pd.DataFrame([spread]).to_csv(results_dir / "multi_split_summary.csv", index=False)
    logger.info(f"Cross-split-ratio stability: {spread}")

    df_b = None
    if n_seeds > 1:
        fused_path = features_dir / "fused_features.npy"
        labels_path = features_dir / "labels.npy"
        df_b = mode_b_multi_seed(fused_path, labels_path, multi_seed_test_size, n_seeds, base_seed=seed)
        df_b.to_csv(results_dir / "multi_split_multi_seed.csv", index=False)

        seed_spread = {
            "n_seeds": n_seeds, "test_size": multi_seed_test_size,
            "auc_mean": float(df_b["auc"].mean()), "auc_std": float(df_b["auc"].std()),
        }
        pd.DataFrame([seed_spread]).to_csv(results_dir / "multi_split_multi_seed_summary.csv", index=False)
        logger.info(f"Multi-seed stability (test_size={multi_seed_test_size}): {seed_spread}")

    return df_a, df_b


def build_parser():
    p = argparse.ArgumentParser(description="Quantify result stability across split ratios and/or random seeds.")
    p.add_argument("--features-dir", type=Path, required=True)
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--split-names", type=str, nargs="+", default=["split_70_30", "split_80_20"])
    p.add_argument("--n-seeds", type=int, default=1, help=">1 additionally runs Mode B (multi-seed re-splitting)")
    p.add_argument("--multi-seed-test-size", type=float, default=0.30)
    p.add_argument("--seed", type=int, default=GLOBAL_SEED)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run(
        features_dir=args.features_dir, results_dir=args.results_dir,
        split_names=args.split_names, n_seeds=args.n_seeds,
        multi_seed_test_size=args.multi_seed_test_size, seed=args.seed,
    )
