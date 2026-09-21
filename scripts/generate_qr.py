"""generate_qr.py -- Phase 1.3

Generates QR code PNGs for each URL, with three validation checks:
  (a) encoding mode (alphanumeric charset vs byte-mode fallback)
  (b) ECC level selection, filtered to levels that can actually fit the URL
  (c) version/masking, auto-assigned by the qrcode library (fit=True)

Writes a manifest CSV with columns: filename, url, label, ecc, version, mode.
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import qrcode
from qrcode.constants import ERROR_CORRECT_H, ERROR_CORRECT_L, ERROR_CORRECT_M, ERROR_CORRECT_Q
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent))
from utils import GLOBAL_SEED, get_logger, set_seed

logger = get_logger("generate_qr")

ALPHANUMERIC_CHARSET = set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:")

# Max byte-mode capacity at version 40 for each ECC level (used to filter
# which ECC levels can actually fit a given URL).
MAX_BYTE_CAPACITY_V40 = {"L": 2953, "M": 2331, "Q": 1663, "H": 1273}

ECC_CONSTANTS = {
    "L": ERROR_CORRECT_L,
    "M": ERROR_CORRECT_M,
    "Q": ERROR_CORRECT_Q,
    "H": ERROR_CORRECT_H,
}


def detect_encoding_mode(url: str) -> str:
    """(a) Encoding mode check: alphanumeric charset (uppercase-only) vs byte mode."""
    if all(c in ALPHANUMERIC_CHARSET for c in url.upper()) and url == url.upper():
        return "alphanumeric"
    # QR alphanumeric mode is case-insensitive-only in the sense that it has
    # no lowercase letters; if the URL has lowercase letters it must use byte mode.
    if all(c in ALPHANUMERIC_CHARSET for c in url.upper()):
        # URL uses only chars in the alphanumeric set but has lowercase -> byte mode required
        if any(c.islower() for c in url):
            return "byte"
        return "alphanumeric"
    return "byte"


def choose_ecc(url: str, rng) -> str:
    """(b) ECC level: randomly choose from levels whose max byte-mode capacity
    at version 40 can fit the URL. Returns None if none fit.
    """
    url_len = len(url.encode("utf-8"))
    candidates = [lvl for lvl, cap in MAX_BYTE_CAPACITY_V40.items() if url_len <= cap]
    if not candidates:
        return None
    return rng.choice(candidates)


def generate(url_csv, out_dir, manifest_path, fixed_ecc=None, fixed_version=None, seed=GLOBAL_SEED):
    import numpy as np

    set_seed(seed)
    rng = np.random.RandomState(seed)

    url_csv = Path(url_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(url_csv)
    if "url" not in df.columns or "label" not in df.columns:
        raise ValueError(f"{url_csv} must have 'url' and 'label' columns")

    rows = []
    skipped = 0
    for i, row in enumerate(tqdm(df.itertuples(index=False), total=len(df), desc="Generating QR codes")):
        url = str(row.url)
        label = int(row.label)

        mode = detect_encoding_mode(url)

        if fixed_ecc is not None:
            ecc = fixed_ecc
        else:
            ecc = choose_ecc(url, rng)
            if ecc is None:
                skipped += 1
                continue

        qr = qrcode.QRCode(
            version=fixed_version,
            error_correction=ECC_CONSTANTS[ecc],
            box_size=10,
            border=4,
        )
        qr.add_data(url)
        qr.make(fit=(fixed_version is None))

        img = qr.make_image(fill_color="black", back_color="white").convert("L")

        filename = f"qr_{i:07d}.png"
        img.save(out_dir / filename)

        rows.append(
            {
                "filename": filename,
                "url": url,
                "label": label,
                "ecc": ecc,
                "version": qr.version,
                "mode": mode,
            }
        )

    if skipped:
        logger.warning(f"Skipped {skipped} URLs (too long to fit any ECC level at version 40)")

    manifest = pd.DataFrame(rows)
    manifest.to_csv(manifest_path, index=False)
    logger.info(f"Generated {len(manifest)} QR images -> {out_dir}, manifest -> {manifest_path}")
    return manifest


def build_parser():
    p = argparse.ArgumentParser(description="Generate QR code images from a URL CSV (Phase 1.3).")
    p.add_argument("--url-csv", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--fixed-ecc", type=str, default=None, choices=["L", "M", "Q", "H"])
    p.add_argument("--fixed-version", type=int, default=None)
    p.add_argument("--seed", type=int, default=GLOBAL_SEED)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    generate(
        url_csv=args.url_csv,
        out_dir=args.out_dir,
        manifest_path=args.manifest,
        fixed_ecc=args.fixed_ecc,
        fixed_version=args.fixed_version,
        seed=args.seed,
    )
