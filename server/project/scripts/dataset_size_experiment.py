"""dataset_size_experiment.py -- data-scaling curve for the PROPOSED
tri-modal model (structural+visual+lexical fused), across multiple sample
sizes.

This is the tri-modal counterpart to replicate_baseline_sizes.py (which
replicates QRiS's own structural-only experiment). Where that script shows
how the STRUCTURAL-ONLY baseline scales with data, this script shows how
the FULL FUSED model scales -- letting you plot both curves on the same
axes for your defense: "does tri-modal fusion still help once you're
correctly at the baseline's own data volumes, or does the fusion advantage
shrink as data grows?"

Uses fixed, untuned XGBoost params (matching the thesis's fixed-classifier
ablation methodology) for speed and a clean, single-variable comparison
across sizes -- Optuna tuning per size would confound "more data" with
"better hyperparameters found by chance."
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
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parent))
import server.project.scripts.generate_qr as generate_qr
import server.project.scripts.lexical as lexical
import server.project.scripts.load_datasets as load_datasets
import server.project.scripts.structural as structural
import server.project.scripts.visual as visual
from server.project.scripts.train import FIXED_ABLATION_PARAMS
from server.project.scripts.utils import GLOBAL_SEED, get_logger, set_seed

logger = get_logger("dataset_size_experiment")

DEFAULT_SIZES = [1000, 2000, 6000, 20000, 50000]


def run_one_size(prasad_path, outdir, n_total, backbone="mobilenet_v2",
                  visual_batch_size=8, tfidf_max_features=5000, seed=GLOBAL_SEED):
    from xgboost import XGBClassifier

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

    visual_path = size_dir / "visual.npy"
    X_visual = visual.run(manifest_path, qr_dir, visual_path, backbone=backbone,
                           batch_size=visual_batch_size, seed=seed)

    manifest = pd.read_csv(manifest_path)
    y_all = pd.read_csv(struct_path).set_index("filename").loc[manifest["filename"], "label"].to_numpy()
    idx_all = np.arange(len(manifest))
    idx_train, idx_test = train_test_split(idx_all, test_size=0.30, stratify=y_all, random_state=seed)
    train_idx_path = size_dir / "train_idx.npy"
    np.save(train_idx_path, idx_train)

    lexical_path = size_dir / "lexical.npy"
    tfidf_out = size_dir / "lexical_tfidf.pkl"
    scaler_out = size_dir / "lexical_scaler.pkl"
    X_lexical = lexical.run(
        manifest_path, qr_dir, lexical_path, tfidf_out, scaler_out,
        split_manifest=train_idx_path, max_features=tfidf_max_features,
    )

    struct_df = pd.read_csv(struct_path).set_index("filename").loc[manifest["filename"]].reset_index()
    X_struct_raw = struct_df[structural.FEATURE_NAMES].to_numpy(dtype=np.float64)

    from sklearn.preprocessing import StandardScaler
    struct_scaler = StandardScaler()
    struct_scaler.fit(X_struct_raw[idx_train])
    X_struct = struct_scaler.transform(X_struct_raw)

    X_fused = np.concatenate([X_struct, X_visual, X_lexical], axis=1)
    y = struct_df["label"].to_numpy(dtype=np.int64)

    X_train, X_test = X_fused[idx_train], X_fused[idx_test]
    y_train, y_test = y[idx_train], y[idx_test]

    model = XGBClassifier(**FIXED_ABLATION_PARAMS, eval_metric="logloss", random_state=seed, n_jobs=-1)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.65).astype(int)

    result = {
        "samples": n_total,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "auc": roc_auc_score(y_test, proba),
    }
    logger.info(f"n={n_total} (tri-modal fused, fixed params): {result}")
    return result


def build_parser():
    p = argparse.ArgumentParser(description="Data-scaling curve for the tri-modal fused model.")
    p.add_argument("--prasad", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--results-out", type=Path, required=True)
    p.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    p.add_argument("--backbone", type=str, default="mobilenet_v2", choices=["mobilenet_v2", "raw_pixels"])
    p.add_argument("--visual-batch-size", type=int, default=8)
    p.add_argument("--tfidf-max-features", type=int, default=5000)
    p.add_argument("--seed", type=int, default=GLOBAL_SEED)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    rows = [
        run_one_size(
            args.prasad, args.outdir, n, backbone=args.backbone,
            visual_batch_size=args.visual_batch_size,
            tfidf_max_features=args.tfidf_max_features, seed=args.seed,
        )
        for n in args.sizes
    ]
    df = pd.DataFrame(rows)
    args.results_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.results_out, index=False)
    print(df)
