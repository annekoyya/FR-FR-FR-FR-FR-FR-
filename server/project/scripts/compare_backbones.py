"""compare_backbones.py -- compares visual-branch backbones on the
visual-only ablation (Config 2), holding structural and lexical branches
constant.

The thesis specifies frozen MobileNetV2 as the visual backbone. This script
reproduces the interesting secondary finding noted during development (see
README "Known-good validated numbers"): a raw-pixel flatten backbone can
outperform MobileNetV2 on visual-only AUC for this specific task, since
QR modules are already near-binary and ImageNet's natural-image-pretrained
filters may not be the ideal inductive bias.

`visual.py` currently supports "mobilenet_v2" and "raw_pixels". To add a
third backbone (e.g. a different torchvision architecture), extend
visual.py's backbone dispatch and this script's BACKBONES list will pick it
up automatically -- no changes needed here.
"""
import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.append(str(Path(__file__).resolve().parent))
from server.project.scripts.utils import GLOBAL_SEED, get_logger, set_seed
import server.project.scripts.visual as visual_module
import server.project.scripts.structural as structural
import server.project.scripts.fusion as fusion_module
from server.project.scripts.train import ABLATION_CONFIGS, FIXED_ABLATION_PARAMS, slice_branches

logger = get_logger("compare_backbones")

BACKBONES = ["mobilenet_v2", "raw_pixels"]


def run(manifest_path, qr_dir, structural_csv, lexical_npy, results_dir,
        backbones=None, test_size=0.30, seed=GLOBAL_SEED, visual_batch_size=8):
    set_seed(seed)
    manifest_path = Path(manifest_path)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if backbones is None:
        backbones = BACKBONES

    manifest = pd.read_csv(manifest_path)
    struct_df = pd.read_csv(structural_csv).set_index("filename").loc[manifest["filename"]].reset_index()
    X_struct = struct_df[structural.FEATURE_NAMES].to_numpy(dtype=np.float64)
    y = struct_df["label"].to_numpy(dtype=np.int64)
    X_lexical = np.load(lexical_npy)

    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier

    idx_all = np.arange(len(manifest))
    idx_train, idx_test = train_test_split(idx_all, test_size=test_size, stratify=y, random_state=seed)
    y_train, y_test = y[idx_train], y[idx_test]

    rows = []
    for backbone in backbones:
        logger.info(f"Extracting visual features with backbone={backbone}...")
        visual_out = results_dir / f"visual_{backbone}.npy"
        t0 = time.time()
        X_visual = visual_module.run(
            manifest_path, qr_dir, visual_out, backbone=backbone,
            batch_size=visual_batch_size, seed=seed,
        )
        extraction_time_s = time.time() - t0

        # Visual-only ablation (Config 2): train on this backbone's features alone.
        Xv_train, Xv_test = X_visual[idx_train], X_visual[idx_test]
        model = XGBClassifier(**FIXED_ABLATION_PARAMS, eval_metric="logloss", random_state=seed, n_jobs=-1)
        model.fit(Xv_train, y_train)
        proba = model.predict_proba(Xv_test)[:, 1]
        auc_visual_only = roc_auc_score(y_test, proba)

        # Also report the fully-fused (structural+this backbone+lexical) AUC,
        # since the visual-only number in isolation doesn't tell you whether
        # a "worse" backbone still contributes usefully when fused.
        X_fused = np.concatenate([X_struct, X_visual, X_lexical], axis=1)
        Xf_train, Xf_test = X_fused[idx_train], X_fused[idx_test]
        fused_model = XGBClassifier(**FIXED_ABLATION_PARAMS, eval_metric="logloss", random_state=seed, n_jobs=-1)
        fused_model.fit(Xf_train, y_train)
        fused_proba = fused_model.predict_proba(Xf_test)[:, 1]
        auc_fused = roc_auc_score(y_test, fused_proba)

        rows.append({
            "backbone": backbone, "visual_dim": X_visual.shape[1],
            "extraction_time_s": extraction_time_s,
            "auc_visual_only": auc_visual_only, "auc_fused_with_this_backbone": auc_fused,
        })
        logger.info(
            f"backbone={backbone}: visual-only AUC={auc_visual_only:.4f}, "
            f"fused AUC={auc_fused:.4f}, extraction_time={extraction_time_s:.1f}s"
        )

    df = pd.DataFrame(rows)
    out_path = results_dir / "compare_backbones.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Wrote backbone comparison -> {out_path}")
    return df


def build_parser():
    p = argparse.ArgumentParser(description="Compare visual-branch backbones on visual-only + fused AUC.")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--qr-dir", type=Path, required=True)
    p.add_argument("--structural-csv", type=Path, required=True)
    p.add_argument("--lexical-npy", type=Path, required=True)
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--backbones", type=str, nargs="+", default=None, choices=BACKBONES)
    p.add_argument("--test-size", type=float, default=0.30)
    p.add_argument("--visual-batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=GLOBAL_SEED)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run(
        manifest_path=args.manifest, qr_dir=args.qr_dir, structural_csv=args.structural_csv,
        lexical_npy=args.lexical_npy, results_dir=args.results_dir, backbones=args.backbones,
        test_size=args.test_size, seed=args.seed, visual_batch_size=args.visual_batch_size,
    )
