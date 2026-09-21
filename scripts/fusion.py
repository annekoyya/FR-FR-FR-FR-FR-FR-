"""fusion.py -- Phase 3

Concatenates structural(24) + visual(1280 or 4761) + lexical(5010) branches,
aligned by filename order via the manifest. Saves fused_features.npy,
labels.npy, feature_names.txt, branch_boundaries.npy, and stratified splits.

Per the thesis (Figure 1 / Table 10): the 24 structural features are
normalized via a StandardScaler BEFORE fusion, fit on the training split
only (leakage-free, same principle as the lexical branch's scaler/TF-IDF).
The fitted scaler is saved to `structural_scaler_out` so cross_dataset_eval.py
can reuse it via .transform() only.

Design note: the structural scaler is fit once, using the PRIMARY split's
(test_size) train indices, and that same scaled structural array is reused
for any --also-test-sizes secondary splits. This mirrors how the lexical
branch is fit in run_pipeline.py (also anchored to the primary split) and
keeps a single consistent structural_scaler.pkl artifact.
"""
import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parent))
from utils import GLOBAL_SEED, get_logger
from structural import FEATURE_NAMES as STRUCTURAL_FEATURE_NAMES
from lexical import SCALAR_FEATURE_NAMES

logger = get_logger("fusion")


def run(
    manifest_path,
    structural_csv,
    visual_npy,
    lexical_npy,
    out_dir,
    test_size=0.30,
    also_test_sizes=(0.20,),
    seed=GLOBAL_SEED,
    visual_dim=1280,
    tfidf_max_features=5000,
    structural_scaler_out=None,
):
    manifest_path = Path(manifest_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if structural_scaler_out is None:
        structural_scaler_out = out_dir.parent.parent / "models" / "structural_scaler.pkl"
    structural_scaler_out = Path(structural_scaler_out)
    structural_scaler_out.parent.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(manifest_path)
    struct_df = pd.read_csv(structural_csv)

    # Align everything to manifest's filename order.
    struct_df = struct_df.set_index("filename").loc[manifest["filename"]].reset_index()

    X_struct_raw = struct_df[STRUCTURAL_FEATURE_NAMES].to_numpy(dtype=np.float64)
    y = struct_df["label"].to_numpy(dtype=np.int64)

    X_visual = np.load(visual_npy)
    X_lexical = np.load(lexical_npy)

    n = len(manifest)
    assert X_struct_raw.shape[0] == n, f"structural rows {X_struct_raw.shape[0]} != manifest rows {n}"
    assert X_visual.shape[0] == n, f"visual rows {X_visual.shape[0]} != manifest rows {n}"
    assert X_lexical.shape[0] == n, f"lexical rows {X_lexical.shape[0]} != manifest rows {n}"

    # --- Leakage-free StandardScaler fit on the PRIMARY split's train rows only ---
    idx_all = np.arange(n)
    primary_idx_train, _ = train_test_split(idx_all, test_size=test_size, stratify=y, random_state=seed)

    struct_scaler = StandardScaler()
    struct_scaler.fit(X_struct_raw[primary_idx_train])
    X_struct = struct_scaler.transform(X_struct_raw)
    joblib.dump(struct_scaler, structural_scaler_out)
    logger.info(
        f"Fit structural StandardScaler on {len(primary_idx_train)} training rows "
        f"(primary split, test_size={test_size}) -> {structural_scaler_out}"
    )

    fused = np.concatenate([X_struct, X_visual, X_lexical], axis=1)

    struct_dim = X_struct.shape[1]
    visual_dim_actual = X_visual.shape[1]
    lexical_dim = X_lexical.shape[1]

    branch_boundaries = {
        "structural": (0, struct_dim),
        "visual": (struct_dim, struct_dim + visual_dim_actual),
        "lexical": (struct_dim + visual_dim_actual, struct_dim + visual_dim_actual + lexical_dim),
    }

    lexical_names = (
        SCALAR_FEATURE_NAMES
        + [f"tfidf_{i}" for i in range(lexical_dim - len(SCALAR_FEATURE_NAMES) - 1)]
        + ["has_lexical"]
    )
    feature_names = list(STRUCTURAL_FEATURE_NAMES) + [f"visual_{i}" for i in range(visual_dim_actual)] + lexical_names

    np.save(out_dir / "fused_features.npy", fused)
    np.save(out_dir / "labels.npy", y)
    np.save(out_dir / "branch_boundaries.npy", branch_boundaries, allow_pickle=True)
    with open(out_dir / "feature_names.txt", "w") as f:
        f.write("\n".join(feature_names))

    logger.info(f"Fused features: {fused.shape} (structural={struct_dim}, visual={visual_dim_actual}, lexical={lexical_dim})")

    # Primary split (idx_train reuses the same split used to fit the structural scaler above)
    idx = idx_all
    idx_train, idx_test = train_test_split(idx, test_size=test_size, stratify=y, random_state=seed)
    y_train, y_test = y[idx_train], y[idx_test]
    X_train, X_test = fused[idx_train], fused[idx_test]
    split_name = f"split_{int((1 - test_size) * 100)}_{int(test_size * 100)}"
    np.savez(
        out_dir / f"{split_name}.npz",
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
        idx_train=idx_train, idx_test=idx_test,
    )
    np.save(out_dir / f"{split_name}_train_idx.npy", idx_train)
    logger.info(f"Wrote primary split -> {split_name}.npz ({len(idx_train)} train / {len(idx_test)} test)")

    for extra_test_size in also_test_sizes:
        idx_tr2, idx_te2, y_tr2, y_te2 = train_test_split(
            idx, y, test_size=extra_test_size, stratify=y, random_state=seed
        )
        X_tr2, X_te2 = fused[idx_tr2], fused[idx_te2]
        name2 = f"split_{int((1 - extra_test_size) * 100)}_{int(extra_test_size * 100)}"
        np.savez(
            out_dir / f"{name2}.npz",
            X_train=X_tr2, X_test=X_te2, y_train=y_tr2, y_test=y_te2,
            idx_train=idx_tr2, idx_test=idx_te2,
        )
        logger.info(f"Wrote secondary split -> {name2}.npz ({len(idx_tr2)} train / {len(idx_te2)} test)")

    return fused, y, branch_boundaries


def build_parser():
    p = argparse.ArgumentParser(description="Fuse structural+visual+lexical features (Phase 3).")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--structural-csv", type=Path, required=True)
    p.add_argument("--visual-npy", type=Path, required=True)
    p.add_argument("--lexical-npy", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--test-size", type=float, default=0.30)
    p.add_argument("--also-test-sizes", type=float, nargs="*", default=[0.20])
    p.add_argument("--seed", type=int, default=GLOBAL_SEED)
    p.add_argument("--structural-scaler-out", type=Path, default=None)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run(
        manifest_path=args.manifest,
        structural_csv=args.structural_csv,
        visual_npy=args.visual_npy,
        lexical_npy=args.lexical_npy,
        out_dir=args.out_dir,
        test_size=args.test_size,
        also_test_sizes=args.also_test_sizes,
        seed=args.seed,
        structural_scaler_out=args.structural_scaler_out,
    )
