"""stats_significance.py -- formal statistical significance testing between
ablation configurations.

The thesis's ablation study (Table 15) reports AUC/accuracy/precision/recall/
F1 per configuration but does not test whether differences between
configurations are statistically significant vs. noise from the specific
train/test split. This script adds that rigor via:

  1. Paired bootstrap confidence intervals on the AUC difference between any
     two configs, resampling the shared test set with replacement.
  2. A paired permutation test (label-shuffle on which model's prediction
     "wins" per sample) for a p-value on the AUC difference.

Always run against the SAME test set for both configs being compared (this
script slices from the shared split, so that's automatic).
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.append(str(Path(__file__).resolve().parent))
from utils import GLOBAL_SEED, get_logger, load_fused_bundle, set_seed
from train import ABLATION_CONFIGS, FIXED_ABLATION_PARAMS, slice_branches

logger = get_logger("stats_significance")


def bootstrap_auc_diff_ci(y_true, proba_a, proba_b, n_boot=2000, seed=GLOBAL_SEED, ci=0.95):
    """Paired bootstrap: resample test-set indices with replacement, recompute
    AUC for both models on the same resample, track the difference.
    Returns (mean_diff, lower, upper, frac_boot_le_zero) where
    frac_boot_le_zero approximates a one-sided p-value for "A is not better
    than B"."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        y_s = y_true[idx]
        if len(np.unique(y_s)) < 2:
            diffs[i] = np.nan
            continue
        auc_a = roc_auc_score(y_s, proba_a[idx])
        auc_b = roc_auc_score(y_s, proba_b[idx])
        diffs[i] = auc_a - auc_b

    diffs = diffs[~np.isnan(diffs)]
    alpha = 1 - ci
    lower = float(np.percentile(diffs, 100 * alpha / 2))
    upper = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    mean_diff = float(np.mean(diffs))
    frac_le_zero = float(np.mean(diffs <= 0))
    return mean_diff, lower, upper, frac_le_zero


def permutation_test_auc_diff(y_true, proba_a, proba_b, n_perm=2000, seed=GLOBAL_SEED):
    """Paired permutation test: for each sample, randomly swap which of
    proba_a/proba_b is treated as "model A" for that sample, recompute the
    AUC difference under the null that the two models are exchangeable.
    Two-sided p-value = fraction of permuted |diff| >= observed |diff|."""
    rng = np.random.RandomState(seed)
    observed_diff = roc_auc_score(y_true, proba_a) - roc_auc_score(y_true, proba_b)

    n = len(y_true)
    perm_diffs = np.empty(n_perm)
    for i in range(n_perm):
        swap_mask = rng.randint(0, 2, size=n).astype(bool)
        perm_a = np.where(swap_mask, proba_b, proba_a)
        perm_b = np.where(swap_mask, proba_a, proba_b)
        perm_diffs[i] = roc_auc_score(y_true, perm_a) - roc_auc_score(y_true, perm_b)

    p_value = float(np.mean(np.abs(perm_diffs) >= abs(observed_diff)))
    return observed_diff, p_value


def get_config_proba(config_name, X_train, X_test, y_train, branch_boundaries, seed=GLOBAL_SEED):
    from xgboost import XGBClassifier

    branches = ABLATION_CONFIGS[config_name]
    Xtr = slice_branches(X_train, branch_boundaries, branches)
    Xte = slice_branches(X_test, branch_boundaries, branches)
    model = XGBClassifier(**FIXED_ABLATION_PARAMS, eval_metric="logloss", random_state=seed, n_jobs=-1)
    model.fit(Xtr, y_train)
    return model.predict_proba(Xte)[:, 1]


def run(features_dir, results_dir, split_name="split_70_30", configs=None, n_boot=2000, n_perm=2000, seed=GLOBAL_SEED):
    set_seed(seed)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_fused_bundle(features_dir, models_dir=None, split_name=split_name, load_model=False)
    X_train, X_test = bundle["X_train"], bundle["X_test"]
    y_train, y_test = bundle["y_train"], bundle["y_test"]
    branch_boundaries = bundle["branch_boundaries"]

    if configs is None:
        configs = list(ABLATION_CONFIGS.keys())

    logger.info(f"Training {len(configs)} ablation configs once each to get comparable test-set probabilities...")
    proba_by_config = {
        name: get_config_proba(name, X_train, X_test, y_train, branch_boundaries, seed=seed)
        for name in configs
    }

    # Primary comparison of interest: baseline (structural only) vs full fused model.
    rows = []
    baseline_name = "1_structural_only"
    fused_name = "7_all_fused"
    if baseline_name in proba_by_config and fused_name in proba_by_config:
        mean_diff, lo, hi, frac_le_zero = bootstrap_auc_diff_ci(
            y_test, proba_by_config[fused_name], proba_by_config[baseline_name], n_boot=n_boot, seed=seed
        )
        obs_diff, p_value = permutation_test_auc_diff(
            y_test, proba_by_config[fused_name], proba_by_config[baseline_name], n_perm=n_perm, seed=seed
        )
        rows.append({
            "comparison": f"{fused_name} vs {baseline_name}",
            "auc_diff": obs_diff, "bootstrap_ci_lower": lo, "bootstrap_ci_upper": hi,
            "bootstrap_p_diff_le_zero": frac_le_zero, "permutation_p_value": p_value,
            "significant_at_0.05": bool(p_value < 0.05),
        })
        logger.info(
            f"{fused_name} vs {baseline_name}: AUC diff={obs_diff:.4f}, "
            f"95% CI=[{lo:.4f}, {hi:.4f}], permutation p={p_value:.4f}"
        )

    # All pairwise comparisons among the requested configs, for completeness.
    for i, name_a in enumerate(configs):
        for name_b in configs[i + 1:]:
            if (name_a, name_b) in [(fused_name, baseline_name), (baseline_name, fused_name)]:
                continue  # already covered above with full detail
            obs_diff, p_value = permutation_test_auc_diff(
                y_test, proba_by_config[name_a], proba_by_config[name_b], n_perm=n_perm, seed=seed
            )
            rows.append({
                "comparison": f"{name_a} vs {name_b}",
                "auc_diff": obs_diff, "bootstrap_ci_lower": None, "bootstrap_ci_upper": None,
                "bootstrap_p_diff_le_zero": None, "permutation_p_value": p_value,
                "significant_at_0.05": bool(p_value < 0.05),
            })

    df = pd.DataFrame(rows)
    out_path = results_dir / "stats_significance.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Wrote significance results -> {out_path}")
    return df


def build_parser():
    p = argparse.ArgumentParser(description="Statistical significance testing between ablation configs.")
    p.add_argument("--features-dir", type=Path, required=True)
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--split-name", type=str, default="split_70_30")
    p.add_argument("--configs", type=str, nargs="*", default=None, help="Subset of ABLATION_CONFIGS keys; default all 7")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--n-perm", type=int, default=2000)
    p.add_argument("--seed", type=int, default=GLOBAL_SEED)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run(
        features_dir=args.features_dir, results_dir=args.results_dir,
        split_name=args.split_name, configs=args.configs,
        n_boot=args.n_boot, n_perm=args.n_perm, seed=args.seed,
    )
