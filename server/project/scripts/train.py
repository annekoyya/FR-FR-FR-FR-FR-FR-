"""train.py -- Phase 4

Trains a 7-config branch-ablation (fixed, untuned XGBoost params for fair
comparison), then runs Optuna hyperparameter tuning on the full fused config
(Config 7), and saves the final tuned model. Also runs a bonus model
comparison (RF, LogReg, LightGBM, CatBoost) on the fused features.
"""
import argparse
import os
import sys
import time
from pathlib import Path

# Must be set before importing xgboost -- fixes a confirmed macOS OpenMP
# duplicate-runtime segfault during prediction. See REBUILD SPEC section 7.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.append(str(Path(__file__).resolve().parent))
from server.project.scripts.utils import GLOBAL_SEED, get_logger, set_seed

logger = get_logger("train")

ABLATION_CONFIGS = {
    "1_structural_only": ["structural"],
    "2_visual_only": ["visual"],
    "3_lexical_only": ["lexical"],
    "4_structural_visual": ["structural", "visual"],
    "5_structural_lexical": ["structural", "lexical"],
    "6_visual_lexical": ["visual", "lexical"],
    "7_all_fused": ["structural", "visual", "lexical"],
}

FIXED_ABLATION_PARAMS = dict(n_estimators=200, max_depth=6, learning_rate=0.1)

# More conservative than bare XGBoost defaults -- used when --skip-optuna.
SKIP_OPTUNA_PARAMS = dict(
    n_estimators=200, max_depth=5, learning_rate=0.1, subsample=0.8,
    colsample_bytree=0.8, gamma=1.0, min_child_weight=3, reg_alpha=0.1, reg_lambda=1.0,
)


def slice_branches(X, branch_boundaries, branches):
    cols = []
    for b in branches:
        start, end = branch_boundaries[b]
        cols.append(X[:, start:end])
    return np.concatenate(cols, axis=1)


def run_ablation(X_train, X_test, y_train, y_test, branch_boundaries, seed=GLOBAL_SEED):
    from xgboost import XGBClassifier

    rows = []
    for name, branches in ABLATION_CONFIGS.items():
        Xtr = slice_branches(X_train, branch_boundaries, branches)
        Xte = slice_branches(X_test, branch_boundaries, branches)

        model = XGBClassifier(**FIXED_ABLATION_PARAMS, eval_metric="logloss", random_state=seed, n_jobs=-1)
        model.fit(Xtr, y_train)
        proba = model.predict_proba(Xte)[:, 1]
        auc = roc_auc_score(y_test, proba)
        acc = model.score(Xte, y_test)
        rows.append({"config": name, "branches": "+".join(branches), "auc": auc, "accuracy": acc})
        logger.info(f"Ablation {name}: AUC={auc:.4f}, accuracy={acc:.4f}")

    return pd.DataFrame(rows)


def optuna_tune(X_train, y_train, seed=GLOBAL_SEED, n_trials=50, timeout=3600):
    import optuna
    from optuna.pruners import MedianPruner
    from xgboost import XGBClassifier

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
        n_splits = min(5, np.bincount(y_train).min())
        if n_splits < 2:
            return 0.5
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        accs = []
        for tr_idx, va_idx in skf.split(X_train, y_train):
            m = XGBClassifier(**params, eval_metric="logloss", random_state=seed, n_jobs=-1)
            m.fit(X_train[tr_idx], y_train[tr_idx])
            proba = m.predict_proba(X_train[va_idx])[:, 1]
            accs.append(roc_auc_score(y_train[va_idx], proba))
        return float(np.mean(accs))

    pruner = MedianPruner(n_warmup_steps=5, n_startup_trials=10)
    study = optuna.create_study(direction="maximize", pruner=pruner, sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    logger.info(f"Optuna best params: {study.best_params} (best value={study.best_value:.4f})")
    return study.best_params


def train_final_model(X_train, y_train, best_params, seed=GLOBAL_SEED):
    from xgboost import XGBClassifier

    Xtr, Xval, ytr, yval = train_test_split(X_train, y_train, test_size=0.10, stratify=y_train, random_state=seed)
    model = XGBClassifier(**best_params, eval_metric="logloss", random_state=seed, n_jobs=-1, early_stopping_rounds=10)
    model.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
    return model


def model_comparison(X_train, X_test, y_train, y_test, seed=GLOBAL_SEED):
    from xgboost import XGBClassifier

    candidates = {
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=seed),
        "XGBoost": XGBClassifier(**SKIP_OPTUNA_PARAMS, eval_metric="logloss", random_state=seed, n_jobs=-1),
    }
    try:
        from lightgbm import LGBMClassifier
        candidates["LightGBM"] = LGBMClassifier(random_state=seed, n_jobs=-1, verbosity=-1)
    except ImportError:
        logger.warning("lightgbm not installed, skipping from model comparison")
    try:
        from catboost import CatBoostClassifier
        candidates["CatBoost"] = CatBoostClassifier(random_state=seed, verbose=False)
    except ImportError:
        logger.warning("catboost not installed, skipping from model comparison")

    rows = []
    for name, model in candidates.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time_s = time.time() - t0

        t0 = time.time()
        proba = model.predict_proba(X_test)[:, 1]
        inference_ms_per_sample = (time.time() - t0) * 1000 / max(1, len(X_test))

        auc = roc_auc_score(y_test, proba)
        rows.append({
            "model": name, "auc": auc, "train_time_s": train_time_s,
            "inference_ms_per_sample": inference_ms_per_sample,
        })
        logger.info(f"{name}: AUC={auc:.4f}, train_time={train_time_s:.2f}s, inference={inference_ms_per_sample:.3f}ms/sample")

    return pd.DataFrame(rows)


def run(features_dir, models_dir, results_dir, split_name="split_70_30", skip_optuna=False,
        optuna_trials=50, optuna_timeout=3600, seed=GLOBAL_SEED):
    set_seed(seed)
    features_dir = Path(features_dir)
    models_dir = Path(models_dir)
    results_dir = Path(results_dir)

    # BUG FIX: these must be created unconditionally, not only inside the
    # Optuna branch, or --skip-optuna runs crash with FileNotFoundError.
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    split = np.load(features_dir / f"{split_name}.npz")
    X_train, X_test, y_train, y_test = split["X_train"], split["X_test"], split["y_train"], split["y_test"]

    branch_boundaries = np.load(features_dir / "branch_boundaries.npy", allow_pickle=True).item()

    ablation_df = run_ablation(X_train, X_test, y_train, y_test, branch_boundaries, seed=seed)
    ablation_df.to_csv(results_dir / "ablation_results.csv", index=False)

    if skip_optuna:
        best_params = SKIP_OPTUNA_PARAMS
        logger.info(f"--skip-optuna set; using fallback params: {best_params}")
    else:
        best_params = optuna_tune(X_train, y_train, seed=seed, n_trials=optuna_trials, timeout=optuna_timeout)

    final_model = train_final_model(X_train, y_train, best_params, seed=seed)
    joblib.dump(final_model, models_dir / "xgboost_fused.pkl")
    logger.info(f"Saved final tuned model -> {models_dir / 'xgboost_fused.pkl'}")

    proba = final_model.predict_proba(X_test)[:, 1]
    final_auc = roc_auc_score(y_test, proba)
    logger.info(f"Final Config-7 fused model test AUC: {final_auc:.4f}")

    comparison_df = model_comparison(X_train, X_test, y_train, y_test, seed=seed)
    comparison_df.to_csv(results_dir / "model_comparison.csv", index=False)

    return ablation_df, final_model, comparison_df


def build_parser():
    p = argparse.ArgumentParser(description="Train ablation configs + tuned fused model (Phase 4).")
    p.add_argument("--features-dir", type=Path, required=True)
    p.add_argument("--models-dir", type=Path, required=True)
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--split-name", type=str, default="split_70_30")
    p.add_argument("--skip-optuna", action="store_true")
    p.add_argument("--optuna-trials", type=int, default=50)
    p.add_argument("--optuna-timeout", type=int, default=3600)
    p.add_argument("--seed", type=int, default=GLOBAL_SEED)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run(
        features_dir=args.features_dir,
        models_dir=args.models_dir,
        results_dir=args.results_dir,
        split_name=args.split_name,
        skip_optuna=args.skip_optuna,
        optuna_trials=args.optuna_trials,
        optuna_timeout=args.optuna_timeout,
        seed=args.seed,
    )
