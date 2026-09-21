"""profile_pipeline.py -- profiles per-stage runtime and per-sample
inference latency of the pipeline.

The thesis explicitly frames real-time, offline, mobile deployment as a
core design goal ("the practical real-world feasibility of the integrated
pipeline is documented by measuring per-sample inference time"). This
script measures:
  1. Wall-clock time per pipeline stage (QR gen, structural, visual,
     lexical, fusion) for a given batch of URLs.
  2. Per-sample inference latency of the FINAL trained model on already-
     fused test data (the number that actually matters for "is this fast
     enough to run during a QR scan").
  3. Peak resident memory (via `resource`, stdlib-only, no extra deps)
     during the visual-branch extraction step, since that's the heaviest
     stage on constrained hardware.
"""
import argparse
import os
import resource
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
import server.project.scripts.fusion as fusion_module
import server.project.scripts.generate_qr as generate_qr
import server.project.scripts.lexical as lexical
import server.project.scripts.structural as structural
import server.project.scripts.visual as visual
from server.project.scripts.utils import GLOBAL_SEED, get_logger, load_fused_bundle, set_seed

logger = get_logger("profile_pipeline")


def _peak_rss_mb():
    """Peak resident set size in MB. ru_maxrss is KB on Linux, bytes on macOS."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1024 if sys.platform != "darwin" else peak / (1024 * 1024)


def profile_extraction_stages(url_csv, qr_dir, manifest_path, structural_csv, visual_npy,
                               lexical_npy, tfidf_out, scaler_out, split_manifest=None,
                               backbone="mobilenet_v2", visual_batch_size=8, seed=GLOBAL_SEED):
    rows = []

    t0 = time.time()
    manifest = generate_qr.generate(url_csv, qr_dir, manifest_path, seed=seed)
    rows.append({"stage": "qr_generation", "wall_time_s": time.time() - t0, "n_samples": len(manifest)})

    t0 = time.time()
    structural.run(manifest_path, qr_dir, structural_csv)
    rows.append({"stage": "structural_extraction", "wall_time_s": time.time() - t0, "n_samples": len(manifest)})

    rss_before = _peak_rss_mb()
    t0 = time.time()
    visual.run(manifest_path, qr_dir, visual_npy, backbone=backbone, batch_size=visual_batch_size, seed=seed)
    visual_time = time.time() - t0
    rss_after = _peak_rss_mb()
    rows.append({
        "stage": "visual_extraction", "wall_time_s": visual_time, "n_samples": len(manifest),
        "peak_rss_mb_after_stage": rss_after,
    })

    t0 = time.time()
    lexical.run(manifest_path, qr_dir, lexical_npy, tfidf_out, scaler_out, split_manifest=split_manifest)
    rows.append({"stage": "lexical_extraction", "wall_time_s": time.time() - t0, "n_samples": len(manifest)})

    df = pd.DataFrame(rows)
    for col in ("n_samples",):
        pass
    df["seconds_per_sample"] = df["wall_time_s"] / df["n_samples"].replace(0, np.nan)
    return df


def profile_inference_latency(features_dir, models_dir, split_name="split_70_30", n_repeats=200):
    bundle = load_fused_bundle(features_dir, models_dir=models_dir, split_name=split_name, load_model=True)
    model = bundle["model"]
    if model is None:
        raise FileNotFoundError(f"No trained model found in {models_dir}; run train.py first.")
    X_test = bundle["X_test"]

    n = min(n_repeats, len(X_test))
    sample = X_test[:n]

    # Single-sample latency loop (worst-case realistic: one QR scanned at a time).
    single_times = []
    for i in range(n):
        row = sample[i:i + 1]
        t0 = time.time()
        model.predict_proba(row)
        single_times.append(time.time() - t0)

    # Batch latency for comparison.
    t0 = time.time()
    model.predict_proba(sample)
    batch_time = time.time() - t0

    return {
        "n_samples_timed": n,
        "mean_single_sample_latency_ms": float(np.mean(single_times) * 1000),
        "p95_single_sample_latency_ms": float(np.percentile(single_times, 95) * 1000),
        "batch_total_time_ms": float(batch_time * 1000),
        "batch_ms_per_sample": float(batch_time * 1000 / n),
    }


def run(url_csv, qr_dir, manifest_path, structural_csv, visual_npy, lexical_npy,
        tfidf_out, scaler_out, features_dir, models_dir, results_dir,
        split_name="split_70_30", split_manifest=None, backbone="mobilenet_v2",
        visual_batch_size=8, n_inference_repeats=200, seed=GLOBAL_SEED,
        skip_extraction_profile=False):
    set_seed(seed)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if not skip_extraction_profile:
        stage_df = profile_extraction_stages(
            url_csv, qr_dir, manifest_path, structural_csv, visual_npy, lexical_npy,
            tfidf_out, scaler_out, split_manifest=split_manifest, backbone=backbone,
            visual_batch_size=visual_batch_size, seed=seed,
        )
        stage_df.to_csv(results_dir / "profile_extraction_stages.csv", index=False)
        logger.info(f"Extraction stage profile:\n{stage_df}")

    inference_stats = profile_inference_latency(features_dir, models_dir, split_name=split_name, n_repeats=n_inference_repeats)
    pd.DataFrame([inference_stats]).to_csv(results_dir / "profile_inference_latency.csv", index=False)
    logger.info(f"Inference latency profile: {inference_stats}")

    return inference_stats


def build_parser():
    p = argparse.ArgumentParser(description="Profile pipeline stage runtime and model inference latency.")
    p.add_argument("--url-csv", type=Path, required=True)
    p.add_argument("--qr-dir", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--structural-csv", type=Path, required=True)
    p.add_argument("--visual-npy", type=Path, required=True)
    p.add_argument("--lexical-npy", type=Path, required=True)
    p.add_argument("--tfidf-out", type=Path, required=True)
    p.add_argument("--scaler-out", type=Path, required=True)
    p.add_argument("--split-manifest", type=Path, default=None)
    p.add_argument("--features-dir", type=Path, required=True)
    p.add_argument("--models-dir", type=Path, required=True)
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--split-name", type=str, default="split_70_30")
    p.add_argument("--backbone", type=str, default="mobilenet_v2", choices=["mobilenet_v2", "raw_pixels"])
    p.add_argument("--visual-batch-size", type=int, default=8)
    p.add_argument("--n-inference-repeats", type=int, default=200)
    p.add_argument("--skip-extraction-profile", action="store_true")
    p.add_argument("--seed", type=int, default=GLOBAL_SEED)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run(
        url_csv=args.url_csv, qr_dir=args.qr_dir, manifest_path=args.manifest,
        structural_csv=args.structural_csv, visual_npy=args.visual_npy, lexical_npy=args.lexical_npy,
        tfidf_out=args.tfidf_out, scaler_out=args.scaler_out, features_dir=args.features_dir,
        models_dir=args.models_dir, results_dir=args.results_dir, split_name=args.split_name,
        split_manifest=args.split_manifest, backbone=args.backbone, visual_batch_size=args.visual_batch_size,
        n_inference_repeats=args.n_inference_repeats, seed=args.seed,
        skip_extraction_profile=args.skip_extraction_profile,
    )
