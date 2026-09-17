"""run_pipeline.py -- orchestrator

Chains Phases 1.3 -> 5 for the "proposed" config (frozen MobileNetV2,
char_wb(3,5) TF-IDF, both splits, full Optuna, Config 7 final model).

--skip-qr/--skip-structural/--skip-visual/--skip-lexical let you re-run only
later phases. All paths are defined unconditionally at the top of main(),
avoiding the NameError bug documented in the REBUILD SPEC section 15.
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

sys.path.append(str(Path(__file__).resolve().parent / "scripts"))
import fusion
import generate_qr
import lexical
import structural
import train as train_module
import evaluate as evaluate_module
import visual
from utils import GLOBAL_SEED, get_logger, set_seed

logger = get_logger("run_pipeline")


def main():
    p = argparse.ArgumentParser(description="Run the full quishing-detection pipeline (Phases 1.3-5).")
    p.add_argument("--url-csv", type=Path, required=True, help="CSV with url,label columns (output of load_datasets.py)")
    p.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--skip-qr", action="store_true")
    p.add_argument("--skip-structural", action="store_true")
    p.add_argument("--skip-visual", action="store_true")
    p.add_argument("--skip-lexical", action="store_true")
    p.add_argument("--skip-optuna", action="store_true")
    p.add_argument("--visual-backbone", type=str, default="mobilenet_v2", choices=["mobilenet_v2", "raw_pixels"])
    p.add_argument("--visual-batch-size", type=int, default=8)
    p.add_argument("--lexical-max-features", type=int, default=5000)
    p.add_argument("--lexical-min-df", type=int, default=1)
    p.add_argument("--test-size", type=float, default=0.30)
    p.add_argument("--also-test-sizes", type=float, nargs="*", default=[0.20])
    p.add_argument("--optuna-trials", type=int, default=50)
    p.add_argument("--optuna-timeout", type=int, default=3600)
    p.add_argument("--seed", type=int, default=GLOBAL_SEED)
    args = p.parse_args()

    set_seed(args.seed)

    # --- All paths defined unconditionally, regardless of which --skip-* flags are set ---
    root = args.project_root
    qr_dir = root / "data" / "qr_images"
    manifest_path = root / "data" / "features" / "manifest.csv"
    structural_csv = root / "data" / "features" / "structural.csv"
    visual_npy = root / "data" / "features" / "visual.npy"
    lexical_npy = root / "data" / "features" / "lexical.npy"
    tfidf_out = root / "models" / "lexical_tfidf.pkl"
    scaler_out = root / "models" / "lexical_scaler.pkl"
    structural_scaler_out = root / "models" / "structural_scaler.pkl"
    features_dir = root / "data" / "features"
    models_dir = root / "models"
    results_dir = root / "results"
    split_train_idx = features_dir / f"split_{int((1 - args.test_size) * 100)}_{int(args.test_size * 100)}_train_idx.npy"

    for d in (qr_dir, features_dir, models_dir, results_dir):
        d.mkdir(parents=True, exist_ok=True)

    # --- Phase 1.3: QR generation ---
    if not args.skip_qr:
        generate_qr.generate(args.url_csv, qr_dir, manifest_path, seed=args.seed)
    else:
        logger.info("Skipping QR generation (--skip-qr)")

    # --- Phase 2.1: structural features ---
    if not args.skip_structural:
        structural.run(manifest_path, qr_dir, structural_csv)
    else:
        logger.info("Skipping structural extraction (--skip-structural)")

    # --- Phase 2.2: visual features ---
    if not args.skip_visual:
        visual.run(manifest_path, qr_dir, visual_npy, backbone=args.visual_backbone,
                    batch_size=args.visual_batch_size, seed=args.seed)
    else:
        logger.info("Skipping visual extraction (--skip-visual)")

    # --- Fusion pass 1: needed to produce the train-index split BEFORE lexical
    #     can be fit leakage-free. We fuse structural+visual first with a
    #     placeholder lexical array sized correctly, purely to get the split. ---
    # Simpler approach: compute the split directly from structural labels.
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split

    manifest_df = pd.read_csv(manifest_path)
    struct_df = pd.read_csv(structural_csv).set_index("filename").loc[manifest_df["filename"]].reset_index()
    y_all = struct_df["label"].to_numpy()
    idx_all = np.arange(len(manifest_df))
    idx_train, _ = train_test_split(idx_all, test_size=args.test_size, stratify=y_all, random_state=args.seed)
    features_dir.mkdir(parents=True, exist_ok=True)
    np.save(split_train_idx, idx_train)

    # --- Phase 2.3: lexical features (leakage-free fit on the train split) ---
    if not args.skip_lexical:
        lexical.run(
            manifest_path, qr_dir, lexical_npy, tfidf_out, scaler_out,
            split_manifest=split_train_idx,
            max_features=args.lexical_max_features, min_df=args.lexical_min_df,
        )
    else:
        logger.info("Skipping lexical extraction (--skip-lexical)")

    # --- Phase 3: fusion ---
    fusion.run(
        manifest_path, structural_csv, visual_npy, lexical_npy, features_dir,
        test_size=args.test_size, also_test_sizes=args.also_test_sizes, seed=args.seed,
        structural_scaler_out=structural_scaler_out,
    )

    # --- Phase 4: training ---
    train_module.run(
        features_dir, models_dir, results_dir,
        split_name=f"split_{int((1 - args.test_size) * 100)}_{int(args.test_size * 100)}",
        skip_optuna=args.skip_optuna, optuna_trials=args.optuna_trials,
        optuna_timeout=args.optuna_timeout, seed=args.seed,
    )

    # --- Phase 5: evaluation ---
    evaluate_module.run(
        features_dir, models_dir, results_dir,
        split_name=f"split_{int((1 - args.test_size) * 100)}_{int(args.test_size * 100)}",
        seed=args.seed,
    )

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
