"""evasion_features.py -- feature-space evasion susceptibility analysis.

Operationalizes the thesis's stated limitation: "highly sophisticated,
minimalist phishing campaigns... may occasionally evade detection." This
script measures HOW susceptible the trained model is to small, targeted
perturbations of each branch's features, on correctly-classified phishing
test samples.

Two perturbation strategies, both applied only to already-scaled features
(so magnitudes are in "standard deviations", branch-agnostic):
  1. Uniform Gaussian noise at increasing sigma, per branch (structural /
     visual / lexical / all), measuring what fraction of previously-caught
     phishing samples drop below the operational threshold.
  2. Targeted perturbation: for each of the top-K most important features
     (by gain, from evaluate.feature_importance), nudge that feature alone
     by increasing amounts and measure the probability shift -- this
     identifies which individual features are the "cheapest" to attack.

This is a feature-space proxy, not a physical-QR-image attack (the thesis
explicitly scopes visual tampering evaluation as future work) -- it tells
you where feature-space engineering effort is best spent if you extend this
into an image-level adversarial study later.
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
from server.project.scripts.utils import GLOBAL_SEED, get_logger, load_fused_bundle, set_seed
from server.project.scripts.evaluate import DEFAULT_THRESHOLD, feature_importance

logger = get_logger("evasion_features")


def branch_noise_sweep(model, X_phish, branch_boundaries, feature_dim, sigmas, threshold=DEFAULT_THRESHOLD, seed=GLOBAL_SEED):
    rng = np.random.RandomState(seed)
    rows = []
    baseline_proba = model.predict_proba(X_phish)[:, 1]
    baseline_caught = (baseline_proba >= threshold)
    n_caught = int(baseline_caught.sum())

    for branch_name, (start, end) in list(branch_boundaries.items()) + [("all", (0, feature_dim))]:
        for sigma in sigmas:
            X_perturbed = X_phish.copy()
            noise = rng.normal(loc=0.0, scale=sigma, size=(X_phish.shape[0], end - start))
            X_perturbed[:, start:end] = X_perturbed[:, start:end] + noise
            proba = model.predict_proba(X_perturbed)[:, 1]

            still_caught = baseline_caught & (proba >= threshold)
            evaded = baseline_caught & (proba < threshold)
            evasion_rate = float(evaded.sum() / max(1, n_caught))
            mean_proba_drop = float(np.mean(baseline_proba[baseline_caught] - proba[baseline_caught])) if n_caught else 0.0

            rows.append({
                "branch": branch_name, "sigma": sigma, "n_originally_caught": n_caught,
                "evasion_rate": evasion_rate, "mean_proba_drop": mean_proba_drop,
            })
    return pd.DataFrame(rows)


def targeted_feature_perturbation(model, X_phish, feature_names, top_k_features, magnitudes, threshold=DEFAULT_THRESHOLD):
    rows = []
    baseline_proba = model.predict_proba(X_phish)[:, 1]
    baseline_caught = baseline_proba >= threshold
    n_caught = int(baseline_caught.sum())
    if n_caught == 0:
        logger.warning("No phishing samples were originally caught at this threshold; skipping targeted perturbation.")
        return pd.DataFrame(rows)

    for feat_name in top_k_features:
        if feat_name not in feature_names:
            continue
        col_idx = feature_names.index(feat_name)
        for magnitude in magnitudes:
            X_perturbed = X_phish.copy()
            X_perturbed[:, col_idx] = X_perturbed[:, col_idx] + magnitude
            proba = model.predict_proba(X_perturbed)[:, 1]
            evaded = baseline_caught & (proba < threshold)
            evasion_rate = float(evaded.sum() / n_caught)
            rows.append({"feature": feat_name, "magnitude": magnitude, "evasion_rate": evasion_rate})
    return pd.DataFrame(rows)


def run(features_dir, models_dir, results_dir, split_name="split_70_30",
        sigmas=(0.5, 1.0, 2.0, 4.0), top_k=10, magnitudes=(1.0, 2.0, 4.0),
        threshold=DEFAULT_THRESHOLD, seed=GLOBAL_SEED):
    set_seed(seed)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_fused_bundle(features_dir, models_dir=models_dir, split_name=split_name, load_model=True)
    model = bundle["model"]
    if model is None:
        raise FileNotFoundError(f"No trained model found in {models_dir}; run train.py first.")

    X_test, y_test = bundle["X_test"], bundle["y_test"]
    feature_names = bundle["feature_names"]
    branch_boundaries = bundle["branch_boundaries"]

    phish_mask = y_test == 1
    X_phish = X_test[phish_mask]
    logger.info(f"{X_phish.shape[0]} phishing samples in test set to run evasion sweeps against")

    noise_df = branch_noise_sweep(model, X_phish, branch_boundaries, X_test.shape[1], sigmas, threshold=threshold, seed=seed)
    noise_df.to_csv(results_dir / "evasion_branch_noise_sweep.csv", index=False)
    logger.info(f"Branch noise sweep:\n{noise_df}")

    importance_df = feature_importance(model, feature_names, top_n=top_k)
    top_features = importance_df["feature"].tolist()

    targeted_df = targeted_feature_perturbation(model, X_phish, feature_names, top_features, magnitudes, threshold=threshold)
    targeted_df.to_csv(results_dir / "evasion_targeted_feature_perturbation.csv", index=False)
    logger.info(f"Targeted top-{top_k}-feature perturbation results written.")

    return noise_df, targeted_df


def build_parser():
    p = argparse.ArgumentParser(description="Feature-space evasion susceptibility analysis.")
    p.add_argument("--features-dir", type=Path, required=True)
    p.add_argument("--models-dir", type=Path, required=True)
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--split-name", type=str, default="split_70_30")
    p.add_argument("--sigmas", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0])
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--magnitudes", type=float, nargs="+", default=[1.0, 2.0, 4.0])
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--seed", type=int, default=GLOBAL_SEED)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run(
        features_dir=args.features_dir, models_dir=args.models_dir, results_dir=args.results_dir,
        split_name=args.split_name, sigmas=args.sigmas, top_k=args.top_k,
        magnitudes=args.magnitudes, threshold=args.threshold, seed=args.seed,
    )
