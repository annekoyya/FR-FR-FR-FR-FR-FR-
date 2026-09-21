"""
Quishing Detection — inference API for the mobile app.

Runs the SAME feature assembly path as cross_dataset_eval.py (never re-fits
TF-IDF / scalers, transform() only), so a prediction here matches a prediction
from the thesis pipeline.

Run:
    export KMP_DUPLICATE_LIB_OK=TRUE
    export OMP_NUM_THREADS=1
    uvicorn app:app --host 0.0.0.0 --port 8000
"""

import os

# MUST be set before xgboost is imported anywhere (macOS duplicate-OpenMP segfault).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import csv
import inspect
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------- config
# Point this at the copied-over thesis project (the folder containing scripts/ and models/).
PROJECT_ROOT = Path(os.environ.get("QUISH_PROJECT_ROOT", Path(__file__).parent / "project")).resolve()
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
MODELS_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODELS_DIR / "xgboost_fused.pkl"
TFIDF_PATH = MODELS_DIR / "lexical_tfidf.pkl"
LEXICAL_SCALER_PATH = MODELS_DIR / "lexical_scaler.pkl"
STRUCTURAL_SCALER_PATH = MODELS_DIR / "structural_scaler.pkl"

THRESHOLD = 0.65  # same tau as the thesis (evaluate.py / train.py)
BACKBONE = "mobilenet_v2"

sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

import cross_dataset_eval  # noqa: E402  (from scripts/)

# ---------------------------------------------------------------- load once
with open(MODEL_PATH, "rb") as f:
    MODEL = pickle.load(f)

app = FastAPI(title="Quishing Detection API")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


class ScanRequest(BaseModel):
    url: str


def featurize(url: str) -> np.ndarray:
    """Build the 6,314-dim fused vector for one URL, reusing the validated path."""
    tmp = Path(tempfile.mkdtemp(prefix="quish_"))
    url_csv = tmp / "one.csv"
    with open(url_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["url", "label"])
        w.writerow([url, 0])  # dummy label, ignored for prediction

    # assemble_features()'s exact keyword names may differ slightly between
    # revisions, so only pass the ones it actually accepts.
    candidates = {
        "url_csv": str(url_csv),
        "qr_dir": str(tmp / "qr"),
        "manifest": str(tmp / "manifest.csv"),
        "manifest_path": str(tmp / "manifest.csv"),
        "tfidf": str(TFIDF_PATH),
        "tfidf_path": str(TFIDF_PATH),
        "scaler": str(LEXICAL_SCALER_PATH),
        "scaler_path": str(LEXICAL_SCALER_PATH),
        "lexical_scaler": str(LEXICAL_SCALER_PATH),
        "structural_scaler": str(STRUCTURAL_SCALER_PATH),
        "structural_scaler_path": str(STRUCTURAL_SCALER_PATH),
        "backbone": BACKBONE,
        "visual_backbone": BACKBONE,
        "batch_size": 1,
    }
    sig = inspect.signature(cross_dataset_eval.assemble_features)
    kwargs = {k: v for k, v in candidates.items() if k in sig.parameters}
    out = cross_dataset_eval.assemble_features(**kwargs)

    X = out[0] if isinstance(out, tuple) else out
    return np.asarray(X, dtype=np.float32).reshape(1, -1)


@app.get("/health")
def health():
    return {"status": "ok", "threshold": THRESHOLD, "model": str(MODEL_PATH)}


@app.post("/scan")
def scan(req: ScanRequest):
    X = featurize(req.url)
    prob = float(MODEL.predict_proba(X)[0, 1])
    return {
        "url": req.url,
        "phishing_probability": round(prob, 4),
        "verdict": "phishing" if prob >= THRESHOLD else "legitimate",
        "threshold": THRESHOLD,
    }
