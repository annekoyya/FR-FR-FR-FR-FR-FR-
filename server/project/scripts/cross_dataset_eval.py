"""cross_dataset_eval.py -- generalizability testing

Scores a trained model against a completely different URL set (e.g.
Kaitholikkal, or a held-out PhiUSIIL slice with zero training overlap),
reusing the SAVED TF-IDF/scaler (.transform() only, never re-fit).

Chunked, single-threaded prediction to avoid the confirmed macOS OpenMP
segfault (see REBUILD SPEC section 7 and 14).
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

sys.path.append(str(Path(__file__).resolve().parent))
from server.project.scripts.utils import get_logger
import server.project.scripts.generate_qr as generate_qr
import server.project.scripts.structural as structural
import server.project.scripts.visual as visual
import server.project.scripts.lexical as lexical

logger = get_logger("cross_dataset_eval")


def chunked_predict_proba(model, X, batch_size=500):
    """Never call predict_proba on the whole array at once -- score in
    chunks to avoid the OpenMP segfault seen on macOS."""
    try:
        model.set_params(n_jobs=1)
    except (ValueError, AttributeError):
        pass

    results = []
    for start in range(0, len(X), batch_size):
        chunk = X[start:start + batch_size]
        proba = model.predict_proba(chunk)[:, 1]
        results.append(proba)
    return np.concatenate(results, axis=0) if results else np.array([])


def assemble_features(
    url_csv,
    qr_dir,
    manifest_path,
    tfidf_path,
    scaler_path,
    structural_scaler_path=None,
    backbone="mobilenet_v2",
    seed=42,
):
    """The SINGLE validated path for turning a raw url_csv into a fused
    X (n, 6314-ish) + y array, reusing every saved artifact via .transform()
    only. This is factored out of run() specifically so other scripts
    (recalibrate_threshold.py, robustness_tests.py, dataset_size_experiment.py,
    etc.) can score new data WITHOUT hand-reassembling features -- that
    exact anti-pattern produced a nonsensical AUC=0.46 in an earlier version
    of recalibrate_threshold.py. Always import and call this function
    instead of reimplementing any part of it.

    Returns (X_fused, y, manifest_df).
    """
    url_csv = Path(url_csv)
    qr_dir = Path(qr_dir)
    manifest_path = Path(manifest_path)

    manifest = generate_qr.generate(url_csv, qr_dir, manifest_path, seed=seed)

    struct_df = structural.run(manifest_path, qr_dir, manifest_path.with_name("cross_structural.csv"))

    visual_out = manifest_path.with_name("cross_visual.npy")
    X_visual = visual.run(manifest_path, qr_dir, visual_out, backbone=backbone, seed=seed)

    tfidf = joblib.load(tfidf_path)
    scaler = joblib.load(scaler_path)

    decoded_urls = []
    has_lexical_flags = []
    for _, row in manifest.iterrows():
        decoded = lexical.decode_qr_payload(qr_dir / row["filename"])
        if decoded is None:
            decoded_urls.append(str(row["url"]))
            has_lexical_flags.append(0)
        else:
            decoded_urls.append(decoded)
            has_lexical_flags.append(1)

    scalar_rows = [lexical.scalar_features(u) for u in decoded_urls]
    scalar_df = pd.DataFrame(scalar_rows)[lexical.SCALAR_FEATURE_NAMES]

    tfidf_matrix = tfidf.transform(decoded_urls).toarray().astype(np.float64)  # transform only, never re-fit
    scaled_scalars = scaler.transform(scalar_df.to_numpy(dtype=np.float64))  # transform only, never re-fit
    has_lexical = np.array(has_lexical_flags, dtype=np.float64).reshape(-1, 1)

    failed_mask = has_lexical.flatten() == 0
    scaled_scalars = scaled_scalars.copy()
    tfidf_matrix = tfidf_matrix.copy()
    scaled_scalars[failed_mask] = 0.0
    tfidf_matrix[failed_mask] = 0.0

    X_lexical = np.concatenate([scaled_scalars, tfidf_matrix, has_lexical], axis=1)

    struct_df = struct_df.set_index("filename").loc[manifest["filename"]].reset_index()
    X_struct_raw = struct_df[structural.FEATURE_NAMES].to_numpy(dtype=np.float64)
    y = struct_df["label"].to_numpy(dtype=np.int64)

    # Reuse the SAVED structural StandardScaler (transform only, never re-fit) --
    # same leakage-free principle as the lexical TF-IDF/scaler.
    if structural_scaler_path is not None and Path(structural_scaler_path).exists():
        struct_scaler = joblib.load(structural_scaler_path)
        X_struct = struct_scaler.transform(X_struct_raw)
    else:
        logger.warning(
            "No structural_scaler_path provided or file not found; using RAW (unscaled) "
            "structural features. This will likely desync from a model trained on "
            "scaled features -- pass models/structural_scaler.pkl."
        )
        X_struct = X_struct_raw

    X_fused = np.concatenate([X_struct, X_visual, X_lexical], axis=1)
    return X_fused, y, manifest


def run(
    url_csv,
    qr_dir,
    manifest_path,
    model_path,
    tfidf_path,
    scaler_path,
    out_results,
    structural_scaler_path=None,
    backbone="mobilenet_v2",
    predict_batch_size=500,
    seed=42,
):
    url_csv = Path(url_csv)
    out_results = Path(out_results)
    out_results.parent.mkdir(parents=True, exist_ok=True)

    X_fused, y, manifest = assemble_features(
        url_csv, qr_dir, manifest_path, tfidf_path, scaler_path,
        structural_scaler_path=structural_scaler_path, backbone=backbone, seed=seed,
    )

    model = joblib.load(model_path)

    if model.n_features_in_ != X_fused.shape[1]:
        raise ValueError(
            f"Dimension mismatch: model expects {model.n_features_in_} features, "
            f"got {X_fused.shape[1]}. Run verify_artifacts.py to diagnose."
        )

    proba = chunked_predict_proba(model, X_fused, batch_size=predict_batch_size)
    pred = (proba >= 0.65).astype(int)

    auc = roc_auc_score(y, proba)
    acc = accuracy_score(y, pred)

    logger.info(f"Cross-dataset eval: AUC={auc:.4f}, accuracy={acc:.4f} on {len(y)} samples")

    result = pd.DataFrame([{"n_samples": len(y), "auc": auc, "accuracy": acc}])
    result.to_csv(out_results, index=False)
    return result


def build_parser():
    p = argparse.ArgumentParser(description="Cross-dataset generalizability evaluation.")
    p.add_argument("--url-csv", type=Path, required=True, help="CSV with url,label columns for the held-out set")
    p.add_argument("--qr-dir", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--tfidf", type=Path, required=True)
    p.add_argument("--scaler", type=Path, required=True)
    p.add_argument("--structural-scaler", type=Path, default=None, help="models/structural_scaler.pkl from fusion.py")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--backbone", type=str, default="mobilenet_v2", choices=["mobilenet_v2", "raw_pixels"])
    p.add_argument("--predict-batch-size", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run(
        url_csv=args.url_csv, qr_dir=args.qr_dir, manifest_path=args.manifest,
        model_path=args.model, tfidf_path=args.tfidf, scaler_path=args.scaler,
        out_results=args.out, structural_scaler_path=args.structural_scaler,
        backbone=args.backbone,
        predict_batch_size=args.predict_batch_size, seed=args.seed,
    )
