"""verify_artifacts.py -- pre-flight consistency check, MANDATORY before
trusting any results.

Exists because of a real, repeated failure mode: silently mismatched
model/feature/vectorizer files that "work" (no crash) but produce garbage
predictions. See REBUILD SPEC section 10.
"""
import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
from utils import get_logger

logger = get_logger("verify_artifacts")

EXPECTED_SCALAR_COUNT = 9


def _mtime_str(path: Path) -> str:
    if not path.exists():
        return "MISSING"
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(path.stat().st_mtime))


def check_tfidf_vocab(models_dir: Path, expected_max_features: int) -> bool:
    path = models_dir / "lexical_tfidf.pkl"
    if not path.exists():
        logger.error(f"[FAIL] TF-IDF vectorizer not found at {path}")
        return False

    tfidf = joblib.load(path)
    vocab_size = len(tfidf.vocabulary_)
    ngram_range = getattr(tfidf, "ngram_range", None)

    logger.info(f"[CHECK] TF-IDF vocab size = {vocab_size} (expected max {expected_max_features}), mtime={_mtime_str(path)}")

    ok = True
    if ngram_range == (1, 1):
        logger.error(
            "[FAIL] TF-IDF ngram_range is (1,1) -- single-character analyzer, capped "
            "regardless of dataset size, NOT a wrong-dataset problem. Fix: use "
            "analyzer='char_wb', ngram_range=(3,5)."
        )
        ok = False
    elif vocab_size > expected_max_features:
        logger.error(f"[FAIL] TF-IDF vocab size {vocab_size} exceeds expected max {expected_max_features}")
        ok = False
    else:
        logger.info("[PASS] TF-IDF vocab size and ngram_range look sane")

    return ok, vocab_size


EXPECTED_STRUCTURAL_COUNT = 24


def check_structural_scaler(models_dir: Path) -> bool:
    path = models_dir / "structural_scaler.pkl"
    if not path.exists():
        logger.error(f"[FAIL] Structural scaler not found at {path} (required per thesis Figure 1/Table 10)")
        return False

    scaler = joblib.load(path)
    n_features = getattr(scaler, "n_features_in_", None)
    logger.info(
        f"[CHECK] Structural scaler n_features_in_ = {n_features} (expected {EXPECTED_STRUCTURAL_COUNT}), "
        f"mtime={_mtime_str(path)}"
    )

    if n_features != EXPECTED_STRUCTURAL_COUNT:
        logger.error(f"[FAIL] Structural scaler expects {n_features} features, expected exactly {EXPECTED_STRUCTURAL_COUNT}")
        return False
    logger.info("[PASS] Structural scaler feature count correct")
    return True


def check_scaler(models_dir: Path) -> bool:
    path = models_dir / "lexical_scaler.pkl"
    if not path.exists():
        logger.error(f"[FAIL] Scaler not found at {path}")
        return False

    scaler = joblib.load(path)
    n_features = getattr(scaler, "n_features_in_", None)
    logger.info(f"[CHECK] Scaler n_features_in_ = {n_features} (expected {EXPECTED_SCALAR_COUNT}), mtime={_mtime_str(path)}")

    if n_features != EXPECTED_SCALAR_COUNT:
        logger.error(f"[FAIL] Scaler expects {n_features} features, expected exactly {EXPECTED_SCALAR_COUNT}")
        return False
    logger.info("[PASS] Scaler feature count correct")
    return True


def check_variance_profile(features_dir: Path, vocab_size: int, expected_max_features: int) -> bool:
    """Checks the LEXICAL branch specifically, not the full fused array.
    Structural protocol columns and visual (especially raw-pixel border)
    columns legitimately have many near-zero-variance columns regardless of
    corruption, so including them would make this check meaningless. We
    isolate the lexical branch via branch_boundaries.npy when available."""
    path = features_dir / "fused_features.npy"
    boundaries_path = features_dir / "branch_boundaries.npy"
    if not path.exists():
        logger.error(f"[FAIL] {path} not found")
        return False

    X = np.load(path)

    if boundaries_path.exists():
        boundaries = np.load(boundaries_path, allow_pickle=True).item()
        lex_start, lex_end = boundaries.get("lexical", (0, X.shape[1]))
        X_check = X[:, lex_start:lex_end]
        scope = "lexical branch"
    else:
        X_check = X
        scope = "full fused array (branch_boundaries.npy not found -- less precise)"

    variances = np.var(X_check, axis=0)
    near_zero_frac = float(np.mean(variances < 1e-10))
    logger.info(f"[CHECK] Near-zero-variance column fraction ({scope}) = {near_zero_frac:.4f}, mtime={_mtime_str(path)}")

    # Legitimate zero-padding happens when vocab_size < expected_max_features;
    # this looks identical to corruption unless we account for it.
    expected_padding_frac = max(0.0, (expected_max_features - vocab_size) / max(1, X_check.shape[1]))
    tolerance = expected_padding_frac + 0.05

    if near_zero_frac > tolerance:
        logger.error(
            f"[FAIL] Near-zero-variance fraction {near_zero_frac:.4f} exceeds tolerance "
            f"{tolerance:.4f} (expected padding fraction {expected_padding_frac:.4f} + 5%% margin). "
            "This may indicate corrupted or zero-padded features beyond what the vocab size explains."
        )
        return False
    logger.info("[PASS] Variance profile consistent with vectorizer's known vocab size")
    return True


def check_feature_name_agreement(features_dir: Path) -> bool:
    names_path = features_dir / "feature_names.txt"
    arr_path = features_dir / "fused_features.npy"
    if not names_path.exists() or not arr_path.exists():
        logger.error(f"[FAIL] Missing {names_path} or {arr_path}")
        return False

    with open(names_path) as f:
        n_names = len(f.read().splitlines())
    X = np.load(arr_path)
    logger.info(f"[CHECK] feature_names.txt lines={n_names} vs fused_features.npy columns={X.shape[1]}, mtime(names)={_mtime_str(names_path)}, mtime(arr)={_mtime_str(arr_path)}")

    if n_names != X.shape[1]:
        logger.error(f"[FAIL] Feature name count {n_names} != array width {X.shape[1]}")
        return False
    logger.info("[PASS] Feature names agree with array width")
    return True


def check_model_dimension(models_dir: Path, features_dir: Path) -> bool:
    model_path = models_dir / "xgboost_fused.pkl"
    arr_path = features_dir / "fused_features.npy"
    if not model_path.exists() or not arr_path.exists():
        logger.error(f"[FAIL] Missing {model_path} or {arr_path}")
        return False

    model = joblib.load(model_path)
    X = np.load(arr_path)
    n_features_in = getattr(model, "n_features_in_", None)
    logger.info(
        f"[CHECK] model.n_features_in_={n_features_in} vs fused_features.npy width={X.shape[1]}, "
        f"mtime(model)={_mtime_str(model_path)}, mtime(features)={_mtime_str(arr_path)}"
    )

    if model_path.stat().st_mtime < arr_path.stat().st_mtime:
        logger.warning(
            "[WARN] Model file is OLDER than the current fused_features.npy -- this staleness is "
            "often the real signal even when dimension checks pass. Consider retraining."
        )

    if n_features_in != X.shape[1]:
        logger.error(f"[FAIL] Model expects {n_features_in} features, current features have {X.shape[1]}")
        return False
    logger.info("[PASS] Model/feature dimension consistency OK")
    return True


def run(features_dir, models_dir, expected_max_features=5000, expected_visual_dims=1280):
    features_dir = Path(features_dir)
    models_dir = Path(models_dir)

    all_ok = True

    tfidf_result = check_tfidf_vocab(models_dir, expected_max_features)
    if isinstance(tfidf_result, tuple):
        tfidf_ok, vocab_size = tfidf_result
    else:
        tfidf_ok, vocab_size = tfidf_result, 0
    all_ok &= tfidf_ok

    all_ok &= check_scaler(models_dir)
    all_ok &= check_structural_scaler(models_dir)
    all_ok &= check_variance_profile(features_dir, vocab_size, expected_max_features)
    all_ok &= check_feature_name_agreement(features_dir)
    all_ok &= check_model_dimension(models_dir, features_dir)

    if all_ok:
        logger.info("=== ALL CHECKS PASSED ===")
    else:
        logger.error("=== ONE OR MORE CHECKS FAILED -- do not trust downstream results until fixed ===")

    return all_ok


def build_parser():
    p = argparse.ArgumentParser(description="Verify model/feature/vectorizer artifact consistency.")
    p.add_argument("--features-dir", type=Path, required=True)
    p.add_argument("--models-dir", type=Path, required=True)
    p.add_argument("--expected-max-features", type=int, default=5000)
    p.add_argument("--expected-visual-dims", type=int, default=1280, help="1280 for mobilenet_v2, 4761 for raw_pixels")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    ok = run(args.features_dir, args.models_dir, args.expected_max_features, args.expected_visual_dims)
    sys.exit(0 if ok else 1)
