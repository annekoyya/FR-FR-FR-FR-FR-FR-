"""replicate_baseline_sizes.py -- Reproduce QRiS's exact structural-only,
multi-size XGBoost experiment for direct comparison against their published
Table V numbers. See REBUILD SPEC section 9.
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.append(str(Path(__file__).resolve().parent))
import generate_qr
import load_datasets
import structural
from utils import GLOBAL_SEED, get_logger, set_seed

logger = get_logger("replicate_baseline_sizes")

QRIS_SIZES = [200, 1000, 2000, 6000, 20000, 50000, 100000, 200000]

# QRiS's published Table V numbers (Feat-DataSet-1), for sanity-checking a
# replication run is in the right ballpark.
QRIS_PUBLISHED = {
    200: dict(val_accuracy=0.7642, test_accuracy=0.6000, precision=0.7692, recall=0.6667, f1=0.7143, auc=0.676),
    1000: dict(val_accuracy=0.7285, test_accuracy=0.7333, precision=0.7159, recall=0.8400, f1=0.7730, auc=0.782),
    2000: dict(val_accuracy=0.7321, test_accuracy=0.7600, precision=0.7048, recall=0.7800, f1=0.7405, auc=0.818),
    6000: dict(val_accuracy=0.7571, test_accuracy=0.7811, precision=0.7490, recall=0.8156, f1=0.7809, auc=0.849),
    20000: dict(val_accuracy=0.7847, test_accuracy=0.8143, precision=0.7766, recall=0.8573, f1=0.8150, auc=0.888),
    50000: dict(val_accuracy=0.8059, test_accuracy=0.8197, precision=0.7862, recall=0.8637, f1=0.8231, auc=0.896),
    100000: dict(val_accuracy=0.8212, test_accuracy=0.8232, precision=0.7899, recall=0.8705, f1=0.8282, auc=0.905),
    200000: dict(val_accuracy=0.8305, test_accuracy=0.8318, precision=0.8028, recall=0.8753, f1=0.8375, auc=0.912),
}


def run_one_size(prasad_path, outdir, n_total, seed=GLOBAL_SEED, optuna_trials=100, optuna_timeout=3600):
    from xgboost import XGBClassifier
    import optuna
    from optuna.pruners import MedianPruner

    set_seed(seed)
    n_per_class = n_total // 2
    size_dir = Path(outdir) / f"size_{n_total}"
    size_dir.mkdir(parents=True, exist_ok=True)

    url_csv = size_dir / "urls.csv"
    load_datasets.run(prasad_path, None, url_csv, n_per_class, seed=seed)

    manifest_path = size_dir / "manifest.csv"
    qr_dir = size_dir / "qr_images"
    generate_qr.generate(url_csv, qr_dir, manifest_path, seed=seed)

    struct_path = size_dir / "structural.csv"
    structural.run(manifest_path, qr_dir, struct_path)

    df = pd.read_csv(struct_path)
    feature_cols = [c for c in df.columns if c not in ("filename", "label")]
    X = df[feature_cols].to_numpy(dtype=np.float64)
    y = df["label"].to_numpy()

    # QRiS's exact split: 70/15/15
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=seed)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        }
        # Guard: n_splits can't exceed the smaller class count at tiny sizes (e.g. 200 samples)
        n_splits = min(5, np.bincount(y_train).min())
        if n_splits < 2:
            return 0.5
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        accs = []
        for tr_idx, va_idx in skf.split(X_train, y_train):
            m = XGBClassifier(**params, eval_metric="logloss", random_state=seed, n_jobs=-1)
            m.fit(X_train[tr_idx], y_train[tr_idx])
            accs.append(accuracy_score(y_train[va_idx], m.predict(X_train[va_idx])))
        return float(np.mean(accs))

    pruner = MedianPruner(n_warmup_steps=5, n_startup_trials=10)
    study = optuna.create_study(direction="maximize", pruner=pruner, sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=optuna_trials, timeout=optuna_timeout)

    best_model = XGBClassifier(**study.best_params, eval_metric="logloss", random_state=seed, n_jobs=-1)
    best_model.fit(X_train, y_train)

    val_pred = best_model.predict(X_val)
    test_pred = best_model.predict(X_test)
    test_proba = best_model.predict_proba(X_test)[:, 1]

    result = {
        "samples": n_total,
        "val_accuracy": accuracy_score(y_val, val_pred),
        "test_accuracy": accuracy_score(y_test, test_pred),
        "precision": precision_score(y_test, test_pred, zero_division=0),
        "recall": recall_score(y_test, test_pred, zero_division=0),
        "f1": f1_score(y_test, test_pred, zero_division=0),
        "auc": roc_auc_score(y_test, test_proba),
    }
    logger.info(f"n={n_total}: {result}")

    published = QRIS_PUBLISHED.get(n_total)
    if published:
        logger.info(f"n={n_total}: QRiS published (for comparison): {published}")

    return result


def build_parser():
    p = argparse.ArgumentParser(description="Replicate QRiS's exact multi-size structural-only experiment.")
    p.add_argument("--prasad", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--results-out", type=Path, required=True)
    p.add_argument("--sizes", type=int, nargs="+", default=QRIS_SIZES)
    p.add_argument("--optuna-trials", type=int, default=100)
    p.add_argument("--optuna-timeout", type=int, default=3600)
    p.add_argument("--seed", type=int, default=GLOBAL_SEED)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    rows = [
        run_one_size(args.prasad, args.outdir, n, args.seed, args.optuna_trials, args.optuna_timeout)
        for n in args.sizes
    ]
    df = pd.DataFrame(rows)
    args.results_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.results_out, index=False)
    print(df)
