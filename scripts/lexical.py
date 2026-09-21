"""lexical.py -- Phase 2.3

Extracts lexical features: 9 scalar URL-structure features + a 5000-dim
char_wb(3,5) TF-IDF vector decoded from the QR payload + a 1-dim
has_lexical flag = 5010 dims total.

CRITICAL: TF-IDF must use ngram_range=(3,5), analyzer="char_wb" -- NOT the
sklearn default (1,1). A single-character analyzer caps the vocabulary at
~60-70 tokens regardless of dataset size and cannot distinguish
typosquatting. See REBUILD SPEC section 8 for the full diagnosis.
"""
import argparse
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from pyzbar.pyzbar import decode as zbar_decode
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent))
from utils import get_logger

logger = get_logger("lexical")

SCALAR_FEATURE_NAMES = [
    "url_length", "hostname_length", "path_length", "num_subdomains", "num_special_chars",
    "hostname_digit_count", "has_ip_address", "has_https", "has_suspicious_keyword",
]

SUSPICIOUS_KEYWORDS = [
    "login", "verify", "secure", "account", "update", "banking", "confirm",
    "paypal", "signin", "password",
]

IP_RE = re.compile(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$")
SPECIAL_CHARS = set("-_@?=&%")

TFIDF_MAX_FEATURES_DEFAULT = 5000
TFIDF_MIN_DF_DEFAULT = 1


def decode_qr_payload(img_path: Path):
    """Decode the QR payload via pyzbar. Returns the decoded URL string, or
    None if decoding failed or the payload doesn't look like a URL."""
    try:
        img = Image.open(img_path)
        results = zbar_decode(img)
        if not results:
            return None
        text = results[0].data.decode("utf-8", errors="ignore")
        if text.startswith("http://") or text.startswith("https://"):
            return text
        return None
    except Exception:
        return None


def scalar_features(url: str) -> dict:
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").split(":")[0]

    return {
        "url_length": len(url),
        "hostname_length": len(hostname),
        "path_length": len(parsed.path or ""),
        "num_subdomains": max(0, hostname.count(".") - 1),
        "num_special_chars": sum(1 for c in url if c in SPECIAL_CHARS),
        "hostname_digit_count": sum(1 for c in hostname if c.isdigit()),
        "has_ip_address": 1 if IP_RE.match(hostname) else 0,
        "has_https": 1 if parsed.scheme == "https" else 0,
        "has_suspicious_keyword": 1 if any(kw in url.lower() for kw in SUSPICIOUS_KEYWORDS) else 0,
    }


def run(
    manifest_path,
    qr_dir,
    out_path,
    tfidf_out,
    scaler_out,
    split_manifest=None,
    max_features=TFIDF_MAX_FEATURES_DEFAULT,
    min_df=TFIDF_MIN_DF_DEFAULT,
):
    manifest_path = Path(manifest_path)
    qr_dir = Path(qr_dir)
    out_path = Path(out_path)
    tfidf_out = Path(tfidf_out)
    scaler_out = Path(scaler_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tfidf_out.parent.mkdir(parents=True, exist_ok=True)
    scaler_out.parent.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(manifest_path)

    decoded_urls = []
    has_lexical_flags = []
    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Decoding QR payloads"):
        decoded = decode_qr_payload(qr_dir / row["filename"])
        if decoded is None:
            decoded_urls.append(str(row["url"]))  # fall back to known URL for scalar feats
            has_lexical_flags.append(0)
        else:
            decoded_urls.append(decoded)
            has_lexical_flags.append(1)

    scalar_rows = [scalar_features(u) for u in decoded_urls]
    scalar_df = pd.DataFrame(scalar_rows)[SCALAR_FEATURE_NAMES]

    # --- Leakage-free fit: fit TF-IDF + scaler ONLY on training-split rows ---
    if split_manifest is not None:
        train_idx = np.load(split_manifest)
        fit_mask = np.zeros(len(manifest), dtype=bool)
        fit_mask[train_idx] = True
    else:
        fit_mask = np.ones(len(manifest), dtype=bool)
        logger.warning(
            "No --split-manifest provided; fitting TF-IDF/scaler on the FULL set. "
            "This risks leaking test-set statistics -- only acceptable for quick "
            "smoke tests, never for reported results."
        )

    if tfidf_out.exists() or scaler_out.exists():
        logger.warning(
            f"{tfidf_out} or {scaler_out} already exists and will be OVERWRITTEN. "
            "If a trained model depends on the existing vectorizer/scaler, this will "
            "desynchronize it (same dims, different meaning -> near-random predictions). "
            "Use distinct output filenames for experimental variants."
        )

    tfidf = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), lowercase=True,
        max_features=max_features, min_df=min_df,
    )
    scaler = StandardScaler()

    train_urls = [u for u, m in zip(decoded_urls, fit_mask) if m]
    tfidf.fit(train_urls)
    scaler.fit(scalar_df.loc[fit_mask].to_numpy(dtype=np.float64))

    joblib.dump(tfidf, tfidf_out)
    joblib.dump(scaler, scaler_out)
    logger.info(
        f"Fit TF-IDF (vocab size={len(tfidf.vocabulary_)}) and scaler on "
        f"{fit_mask.sum()} training rows -> {tfidf_out}, {scaler_out}"
    )

    # --- Transform the FULL manifest (never re-fit) ---
    tfidf_matrix = tfidf.transform(decoded_urls).toarray().astype(np.float64)
    scaled_scalars = scaler.transform(scalar_df.to_numpy(dtype=np.float64))

    has_lexical = np.array(has_lexical_flags, dtype=np.float64).reshape(-1, 1)

    n_rows = len(manifest)
    tfidf_dim = max_features
    if tfidf_matrix.shape[1] < tfidf_dim:
        pad = np.zeros((n_rows, tfidf_dim - tfidf_matrix.shape[1]))
        tfidf_matrix = np.concatenate([tfidf_matrix, pad], axis=1)

    # Zero-pad the TF-IDF+scalar region for rows where decoding failed, and
    # mark has_lexical=0 so downstream models know this region is intentionally
    # empty, not real zero values.
    failed_mask = has_lexical.flatten() == 0
    scaled_scalars = scaled_scalars.copy()
    tfidf_matrix = tfidf_matrix.copy()
    scaled_scalars[failed_mask] = 0.0
    tfidf_matrix[failed_mask] = 0.0

    fused = np.concatenate([scaled_scalars, tfidf_matrix, has_lexical], axis=1)
    expected_dim = len(SCALAR_FEATURE_NAMES) + tfidf_dim + 1
    assert fused.shape[1] == expected_dim, f"Expected {expected_dim} dims, got {fused.shape[1]}"

    np.save(out_path, fused)
    logger.info(f"Wrote lexical features {fused.shape} -> {out_path}")
    return fused


def build_parser():
    p = argparse.ArgumentParser(description="Extract lexical QR/URL features (Phase 2.3).")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--qr-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tfidf-out", type=Path, required=True)
    p.add_argument("--scaler-out", type=Path, required=True)
    p.add_argument("--split-manifest", type=Path, default=None, help="Path to split_70_30_train_idx.npy")
    p.add_argument("--max-features", type=int, default=TFIDF_MAX_FEATURES_DEFAULT)
    p.add_argument("--min-df", type=int, default=TFIDF_MIN_DF_DEFAULT)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run(
        manifest_path=args.manifest,
        qr_dir=args.qr_dir,
        out_path=args.out,
        tfidf_out=args.tfidf_out,
        scaler_out=args.scaler_out,
        split_manifest=args.split_manifest,
        max_features=args.max_features,
        min_df=args.min_df,
    )
