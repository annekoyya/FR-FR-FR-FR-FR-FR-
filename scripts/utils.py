"""utils.py -- Shared seeding, path config, and logging utilities used across
every script in the pipeline.
"""
import logging
import random
import sys
from pathlib import Path

import numpy as np

GLOBAL_SEED = 99

# Project root is the parent of the scripts/ directory this file lives in.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

PATHS = {
    "urls": PROJECT_ROOT / "data" / "urls",
    "qr_images": PROJECT_ROOT / "data" / "qr_images",
    "features": PROJECT_ROOT / "data" / "features",
    "models": PROJECT_ROOT / "models",
    "results": PROJECT_ROOT / "results",
}

# Auto-create all known paths on import so downstream scripts never have to
# worry about missing directories.
for _p in PATHS.values():
    _p.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = GLOBAL_SEED) -> None:
    """Seed random, numpy, and torch (if available) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_logger(name: str) -> logging.Logger:
    """Return a configured stdout logger with a compact timestamp format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def load_fused_bundle(features_dir, models_dir=None, split_name="split_70_30", load_model=True):
    """Single, canonical way to load a fused-feature split + its trained
    model. Every extension script (stats_significance, error_analysis,
    evasion_features, recalibrate_threshold, multi_split_experiment, ...)
    MUST go through this helper instead of hand-reassembling features from
    raw structural/visual/lexical arrays.

    This exists specifically because a hand-rolled feature-reassembly path
    in an earlier version of recalibrate_threshold.py diverged from the
    validated fusion.py/cross_dataset_eval.py path and produced a
    nonsensical AUC of 0.46. Reusing this helper (and, for scoring brand-new
    images, cross_dataset_eval.assemble_features) is the fix.

    Returns a dict with: X_train, X_test, y_train, y_test, idx_train,
    idx_test, feature_names (list[str]), branch_boundaries (dict), and
    model (or None if load_model=False / no model found).
    """
    from pathlib import Path as _Path

    features_dir = _Path(features_dir)
    split = np.load(features_dir / f"{split_name}.npz")

    with open(features_dir / "feature_names.txt") as f:
        feature_names = f.read().splitlines()

    branch_boundaries = np.load(features_dir / "branch_boundaries.npy", allow_pickle=True).item()

    model = None
    if load_model and models_dir is not None:
        import joblib
        model_path = _Path(models_dir) / "xgboost_fused.pkl"
        if model_path.exists():
            model = joblib.load(model_path)

    return {
        "X_train": split["X_train"], "X_test": split["X_test"],
        "y_train": split["y_train"], "y_test": split["y_test"],
        "idx_train": split["idx_train"], "idx_test": split["idx_test"],
        "feature_names": feature_names, "branch_boundaries": branch_boundaries,
        "model": model,
    }
