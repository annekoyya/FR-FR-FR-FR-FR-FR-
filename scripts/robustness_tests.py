"""robustness_tests.py -- image-level robustness testing against synthetic
camera artifacts.

Directly operationalizes the thesis's stated limitation: "there is no
evaluation of the method's robustness against real-world camera noise,
motion blur, or uneven lighting" (Scope and Limitations, 1.3).

Takes a batch of already-generated, correctly-classified QR images, applies
several synthetic perturbations (Gaussian noise, motion blur, rotation,
brightness/contrast shift, JPEG recompression), re-runs ONLY the affected
branches (structural + visual -- the lexical payload is unaffected since
pyzbar either still decodes the same URL or fails entirely), re-fuses using
the SAME saved scalers/vectorizer, and measures how much accuracy/AUC
degrades at each perturbation level.
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

sys.path.append(str(Path(__file__).resolve().parent))
import lexical
import structural
import visual
from utils import GLOBAL_SEED, get_logger, set_seed

logger = get_logger("robustness_tests")


def apply_gaussian_noise(img, sigma):
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    out = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out


def apply_motion_blur(img, kernel_size):
    if kernel_size < 3:
        return img
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[kernel_size // 2, :] = 1.0 / kernel_size
    return cv2.filter2D(img, -1, kernel)


def apply_rotation(img, degrees):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, degrees, 1.0)
    return cv2.warpAffine(img, matrix, (w, h), borderValue=255)


def apply_brightness_shift(img, delta):
    return np.clip(img.astype(np.int32) + delta, 0, 255).astype(np.uint8)


def apply_jpeg_recompression(img, quality):
    ok, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return img
    return cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)


PERTURBATIONS = {
    "gaussian_noise": {"fn": apply_gaussian_noise, "levels": [5, 15, 30, 50], "param_name": "sigma"},
    "motion_blur": {"fn": apply_motion_blur, "levels": [3, 5, 9, 15], "param_name": "kernel_size"},
    "rotation": {"fn": apply_rotation, "levels": [2, 5, 10, 20], "param_name": "degrees"},
    "brightness_shift": {"fn": apply_brightness_shift, "levels": [-60, -30, 30, 60], "param_name": "delta"},
    "jpeg_recompression": {"fn": apply_jpeg_recompression, "levels": [80, 50, 20, 5], "param_name": "quality"},
}


def perturb_and_save(manifest, src_qr_dir, dst_qr_dir, perturb_fn, level, seed=GLOBAL_SEED):
    set_seed(seed)
    dst_qr_dir = Path(dst_qr_dir)
    dst_qr_dir.mkdir(parents=True, exist_ok=True)
    for _, row in manifest.iterrows():
        img = cv2.imread(str(Path(src_qr_dir) / row["filename"]), cv2.IMREAD_GRAYSCALE)
        perturbed = perturb_fn(img, level)
        cv2.imwrite(str(dst_qr_dir / row["filename"]), perturbed)


def score_perturbed_batch(manifest_path, qr_dir, model, tfidf, scaler, struct_scaler,
                           X_lexical_clean, backbone, seed=GLOBAL_SEED, threshold=0.65):
    """Re-extract structural + visual only (lexical reused from the clean
    baseline extraction, since pixel-level perturbation doesn't change the
    decoded payload unless decoding fails outright)."""
    manifest_path = Path(manifest_path)
    qr_dir = Path(qr_dir)

    struct_out = qr_dir.parent / f"{qr_dir.name}_structural.csv"
    struct_df = structural.run(manifest_path, qr_dir, struct_out, self_check=False)

    visual_out = qr_dir.parent / f"{qr_dir.name}_visual.npy"
    X_visual = visual.run(manifest_path, qr_dir, visual_out, backbone=backbone, seed=seed)

    manifest = pd.read_csv(manifest_path)
    struct_df = struct_df.set_index("filename").loc[manifest["filename"]].reset_index()
    X_struct_raw = struct_df[structural.FEATURE_NAMES].to_numpy(dtype=np.float64)
    X_struct = struct_scaler.transform(X_struct_raw)
    y = struct_df["label"].to_numpy(dtype=np.int64)

    # Check whether pyzbar can still decode under this perturbation -- if not,
    # the lexical branch legitimately degrades to zero-padded per lexical.py's
    # own fallback design, which we mirror here rather than reusing the clean
    # lexical vector blindly.
    has_lexical_now = []
    for _, row in manifest.iterrows():
        decoded = lexical.decode_qr_payload(qr_dir / row["filename"])
        has_lexical_now.append(0 if decoded is None else 1)
    has_lexical_now = np.array(has_lexical_now)

    X_lexical = X_lexical_clean.copy()
    newly_failed = has_lexical_now == 0
    if newly_failed.any():
        X_lexical[newly_failed, :-1] = 0.0
        X_lexical[newly_failed, -1] = 0.0

    X_fused = np.concatenate([X_struct, X_visual, X_lexical], axis=1)

    proba = model.predict_proba(X_fused)[:, 1]
    pred = (proba >= threshold).astype(int)

    return {
        "auc": roc_auc_score(y, proba),
        "accuracy": accuracy_score(y, pred),
        "pyzbar_decode_rate": float(has_lexical_now.mean()),
    }


def run(manifest_path, clean_qr_dir, model_path, tfidf_path, scaler_path, structural_scaler_path,
        lexical_npy, out_dir, results_dir, backbone="mobilenet_v2", perturbations=None, seed=GLOBAL_SEED):
    set_seed(seed)
    manifest_path = Path(manifest_path)
    out_dir = Path(out_dir)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    model = joblib.load(model_path)
    tfidf = joblib.load(tfidf_path)
    scaler = joblib.load(scaler_path)
    struct_scaler = joblib.load(structural_scaler_path)
    X_lexical_clean = np.load(lexical_npy)

    manifest = pd.read_csv(manifest_path)

    # Baseline (unperturbed) score, for comparison.
    baseline = score_perturbed_batch(
        manifest_path, clean_qr_dir, model, tfidf, scaler, struct_scaler,
        X_lexical_clean, backbone, seed=seed,
    )
    logger.info(f"Baseline (clean images): {baseline}")

    if perturbations is None:
        perturbations = list(PERTURBATIONS.keys())

    rows = [{"perturbation": "none", "level": None, **baseline}]
    for pert_name in perturbations:
        spec = PERTURBATIONS[pert_name]
        for level in spec["levels"]:
            dst_qr_dir = out_dir / f"{pert_name}_{level}"
            perturb_and_save(manifest, clean_qr_dir, dst_qr_dir, spec["fn"], level, seed=seed)

            result = score_perturbed_batch(
                manifest_path, dst_qr_dir, model, tfidf, scaler, struct_scaler,
                X_lexical_clean, backbone, seed=seed,
            )
            rows.append({"perturbation": pert_name, "level": level, **result})
            logger.info(f"{pert_name}={level}: {result}")

            shutil.rmtree(dst_qr_dir, ignore_errors=True)  # keep disk usage bounded across many levels

    df = pd.DataFrame(rows)
    out_path = results_dir / "robustness_tests.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Wrote robustness results -> {out_path}")
    return df


def build_parser():
    p = argparse.ArgumentParser(description="Image-level robustness testing against synthetic camera artifacts.")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--clean-qr-dir", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--tfidf", type=Path, required=True)
    p.add_argument("--scaler", type=Path, required=True)
    p.add_argument("--structural-scaler", type=Path, required=True)
    p.add_argument("--lexical-npy", type=Path, required=True, help="Pre-computed clean lexical.npy, aligned to --manifest")
    p.add_argument("--out-dir", type=Path, required=True, help="Scratch dir for perturbed images (cleaned up per-level)")
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--backbone", type=str, default="mobilenet_v2", choices=["mobilenet_v2", "raw_pixels"])
    p.add_argument("--perturbations", type=str, nargs="+", default=None, choices=list(PERTURBATIONS.keys()))
    p.add_argument("--seed", type=int, default=GLOBAL_SEED)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run(
        manifest_path=args.manifest, clean_qr_dir=args.clean_qr_dir, model_path=args.model,
        tfidf_path=args.tfidf, scaler_path=args.scaler, structural_scaler_path=args.structural_scaler,
        lexical_npy=args.lexical_npy, out_dir=args.out_dir, results_dir=args.results_dir,
        backbone=args.backbone, perturbations=args.perturbations, seed=args.seed,
    )
