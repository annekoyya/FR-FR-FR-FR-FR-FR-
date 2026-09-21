"""structural.py -- Phase 2.1

Extracts 24 structural features from QR code images: 5 protocol-level
(decoded from the QR's own format-info bits) + 19 statistical (computed from
the binarized module grid).

READ THE REBUILD SPEC BEFORE TOUCHING THIS FILE. It documents three real,
previously-diagnosed bugs that are fixed here:
  1. Quiet-zone cropping must happen before module-size estimation, or the
     module count is inflated by exactly 8 (4 per side) and decoded version
     is off by +2 on every sample.
  2. Module-size estimation must only accept runs starting at row 0 (true
     finder-pattern pixels), and take the max run over the first ~30
     columns -- not the first column with any black pixel -- to avoid
     locking onto anti-aliasing artifacts at the crop boundary.
  3. The module matrix must be cast to int16 before any subtraction/diff
     operation, since uint8 wraps around (0-1=255) and silently corrupts
     every symmetry/transition feature.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent))
from server.project.scripts.utils import get_logger

logger = get_logger("structural")

FEATURE_NAMES = [
    "version", "ecc_level", "masking_pattern", "num_alignment_patterns", "required_remainder_bits",
    "num_black_modules", "num_white_modules", "black_white_ratio", "qr_density", "qr_mean_density",
    "std_dev_row_wise", "std_dev_col_wise", "row_transitions_total", "col_transitions_total", "qr_entropy",
    "vertical_asymmetry", "horizontal_asymmetry", "quadrant_density_tl", "quadrant_density_tr",
    "quadrant_density_bl", "quadrant_density_br", "center_density", "row_histogram_peaks", "col_histogram_peaks",
]

ALIGNMENT_POSITIONS = {
    2: [6, 18], 3: [6, 22], 4: [6, 26], 5: [6, 30], 6: [6, 34], 7: [6, 22, 38], 8: [6, 24, 42],
    9: [6, 26, 46], 10: [6, 28, 50], 11: [6, 30, 54], 12: [6, 32, 58], 13: [6, 34, 62],
    14: [6, 26, 46, 66], 15: [6, 26, 48, 70], 16: [6, 26, 50, 74], 17: [6, 30, 54, 78],
    18: [6, 30, 56, 82], 19: [6, 30, 58, 86], 20: [6, 34, 62, 90], 21: [6, 28, 50, 72, 94],
    22: [6, 26, 50, 74, 98], 23: [6, 30, 54, 78, 102], 24: [6, 28, 54, 80, 106], 25: [6, 32, 58, 84, 110],
    26: [6, 30, 58, 86, 114], 27: [6, 34, 62, 90, 118], 28: [6, 26, 50, 74, 98, 122],
    29: [6, 30, 54, 78, 102, 126], 30: [6, 26, 52, 78, 104, 130], 31: [6, 30, 56, 82, 108, 134],
    32: [6, 34, 60, 86, 112, 138], 33: [6, 30, 58, 86, 114, 142], 34: [6, 34, 62, 90, 118, 146],
    35: [6, 30, 54, 78, 102, 126, 150], 36: [6, 24, 50, 76, 102, 128, 154], 37: [6, 28, 54, 80, 106, 132, 158],
    38: [6, 32, 58, 84, 110, 136, 162], 39: [6, 26, 54, 82, 110, 138, 166], 40: [6, 30, 58, 86, 114, 142, 170],
}


def num_alignment_patterns(version: int) -> int:
    if version < 2:
        return 0
    n = len(ALIGNMENT_POSITIONS[version])
    return n * n - 3  # n^2 combos minus 3 finder-pattern overlaps


def required_remainder_bits(version: int) -> int:
    if version == 1:
        return 0
    if 2 <= version <= 6:
        return 7
    if 7 <= version <= 13:
        return 0
    if 14 <= version <= 20:
        return 3
    if 21 <= version <= 27:
        return 4
    if 28 <= version <= 34:
        return 3
    return 0  # 35-40


# ---------------------------------------------------------------------------
# BCH(15,5) format-info encoder/decoder
# ---------------------------------------------------------------------------
_BCH_GENERATOR = 0b10100110111
_BCH_MASK = 0b101010000010010

ECC_ORDINAL = {0b01: 0, 0b00: 1, 0b11: 2, 0b10: 3}  # L=0, M=1, Q=2, H=3
ECC_NAME_BY_BITS = {0b01: "L", 0b00: "M", 0b11: "Q", 0b10: "H"}


def _bch_encode(data5: int) -> int:
    d = data5 << 10
    for i in range(4, -1, -1):
        if d & (1 << (i + 10)):
            d ^= _BCH_GENERATOR << i
    return ((data5 << 10) | d) ^ _BCH_MASK


def _self_test_bch():
    # Published worked example from thonky.com's QR tutorial: ECC=L(01), mask=4(100)
    expected = 0b110011000101111
    actual = _bch_encode(0b01100)
    assert actual == expected, f"BCH self-test failed: got {actual:015b}, expected {expected:015b}"


_self_test_bch()

_ALL_FORMAT_CODEWORDS = {}
for _ecc_bits in (0b01, 0b00, 0b11, 0b10):
    for _mask_bits in range(8):
        _data = (_ecc_bits << 3) | _mask_bits
        _ALL_FORMAT_CODEWORDS[_bch_encode(_data)] = (_ecc_bits, _mask_bits)


def _hamming_distance(a: int, b: int, nbits: int = 15) -> int:
    return bin(a ^ b).count("1")


def decode_format_info(bits15: int):
    """Nearest-Hamming-distance match against all 32 valid (ecc, mask) codewords."""
    best = None
    best_dist = 99
    for codeword, (ecc_bits, mask_bits) in _ALL_FORMAT_CODEWORDS.items():
        dist = _hamming_distance(bits15, codeword)
        if dist < best_dist:
            best_dist = dist
            best = (ecc_bits, mask_bits)
    return best


def _format_copy_a_coords(n):
    return [(8, c) for c in range(6)] + [(8, 7), (8, 8)] + [(7, 8)] + [(r, 8) for r in range(5, -1, -1)]


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def preprocess(img_path: Path) -> np.ndarray:
    """Exact preprocessing sequence: grayscale -> median blur -> gaussian blur
    -> CLAHE -> binary threshold."""
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image at {img_path}")
    img = cv2.medianBlur(img, 3)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    _, binary = cv2.threshold(img, 189, 255, cv2.THRESH_BINARY)
    return binary


def crop_to_content(binary: np.ndarray) -> np.ndarray:
    """BUG FIX #1: crop the white quiet-zone border before module-size
    estimation, or module counts are inflated by exactly 8 (4 per side)."""
    black_mask = binary == 0
    if not black_mask.any():
        return binary
    rows = np.any(black_mask, axis=1)
    cols = np.any(black_mask, axis=0)
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    return binary[r0:r1 + 1, c0:c1 + 1]


def estimate_module_size(binary: np.ndarray) -> int:
    """BUG FIX #2: only accept runs starting at row 0 (guaranteed true finder
    pattern after crop_to_content), take the MAX run across the first ~30
    columns rather than the first column with any black pixel."""
    h, w = binary.shape
    inverted = binary == 0
    best_run = 0
    for col in range(min(30, w)):
        col_pixels = inverted[:, col]
        if not col_pixels[0]:
            continue
        run = 0
        for v in col_pixels:
            if v:
                run += 1
            else:
                break
        best_run = max(best_run, run)
    if best_run > 0:
        return max(1, round(best_run / 7))
    return max(1, h // 25)


def to_module_matrix(binary: np.ndarray, module_px: int) -> np.ndarray:
    """Downsample the cropped binary image into a per-module grid: 1=black, 0=white."""
    h, w = binary.shape
    n_rows = max(1, h // module_px)
    n_cols = max(1, w // module_px)
    matrix = np.zeros((n_rows, n_cols), dtype=np.uint8)
    for r in range(n_rows):
        for c in range(n_cols):
            block = binary[r * module_px:(r + 1) * module_px, c * module_px:(c + 1) * module_px]
            if block.size == 0:
                continue
            matrix[r, c] = 1 if np.mean(block) < 128 else 0
    return matrix


def version_from_matrix_size(n: int) -> int:
    """QR module count: n = 4*version + 17."""
    version = round((n - 17) / 4)
    return max(1, min(40, version))


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def _count_peaks(profile: np.ndarray) -> int:
    peaks = 0
    for i in range(1, len(profile) - 1):
        if profile[i] > profile[i - 1] and profile[i] > profile[i + 1]:
            peaks += 1
    return peaks


def extract_statistical_features(matrix: np.ndarray) -> dict:
    """BUG FIX #3: cast to int16 first, or uint8 subtraction wraps around and
    silently corrupts every symmetry/transition feature."""
    matrix = matrix.astype(np.int16)
    n_rows, n_cols = matrix.shape
    total = matrix.size
    flat = matrix.flatten()

    num_black = int(flat.sum())
    num_white = int(flat.size - num_black)
    black_white_ratio = float(num_black / num_white) if num_white > 0 else 0.0
    qr_density = float(num_black / total) if total > 0 else 0.0

    # qr_mean_density: mean of a 3x3 grid of block-level densities (intentionally
    # distinct from qr_density -- see REBUILD SPEC section 6 for rationale).
    block_densities = []
    r_edges = np.linspace(0, n_rows, 4).astype(int)
    c_edges = np.linspace(0, n_cols, 4).astype(int)
    for i in range(3):
        for j in range(3):
            block = matrix[r_edges[i]:r_edges[i + 1], c_edges[j]:c_edges[j + 1]]
            if block.size > 0:
                block_densities.append(block.mean())
    qr_mean_density = float(np.mean(block_densities)) if block_densities else 0.0

    row_means = matrix.mean(axis=1)
    col_means = matrix.mean(axis=0)
    std_dev_row_wise = float(np.std(row_means))
    std_dev_col_wise = float(np.std(col_means))

    row_transitions_total = float(np.sum(np.abs(np.diff(matrix, axis=1))))
    col_transitions_total = float(np.sum(np.abs(np.diff(matrix, axis=0))))

    p_black = qr_density
    p_white = 1.0 - p_black
    qr_entropy = 0.0
    for p in (p_black, p_white):
        if p > 0:
            qr_entropy -= p * np.log2(p)

    vertical_asymmetry = float(np.mean(np.abs(matrix - np.flipud(matrix))))
    horizontal_asymmetry = float(np.mean(np.abs(matrix - np.fliplr(matrix))))

    half_r, half_c = n_rows // 2, n_cols // 2
    quadrant_density_tl = float(matrix[:half_r, :half_c].mean()) if half_r and half_c else 0.0
    quadrant_density_tr = float(matrix[:half_r, half_c:].mean()) if half_r and (n_cols - half_c) else 0.0
    quadrant_density_bl = float(matrix[half_r:, :half_c].mean()) if (n_rows - half_r) and half_c else 0.0
    quadrant_density_br = float(matrix[half_r:, half_c:].mean()) if (n_rows - half_r) and (n_cols - half_c) else 0.0

    r3a, r3b = n_rows // 3, (2 * n_rows) // 3
    c3a, c3b = n_cols // 3, (2 * n_cols) // 3
    center_block = matrix[r3a:r3b, c3a:c3b]
    center_density = float(center_block.mean()) if center_block.size > 0 else 0.0

    row_histogram_peaks = _count_peaks(row_means)
    col_histogram_peaks = _count_peaks(col_means)

    return {
        "num_black_modules": num_black,
        "num_white_modules": num_white,
        "black_white_ratio": black_white_ratio,
        "qr_density": qr_density,
        "qr_mean_density": qr_mean_density,
        "std_dev_row_wise": std_dev_row_wise,
        "std_dev_col_wise": std_dev_col_wise,
        "row_transitions_total": row_transitions_total,
        "col_transitions_total": col_transitions_total,
        "qr_entropy": qr_entropy,
        "vertical_asymmetry": vertical_asymmetry,
        "horizontal_asymmetry": horizontal_asymmetry,
        "quadrant_density_tl": quadrant_density_tl,
        "quadrant_density_tr": quadrant_density_tr,
        "quadrant_density_bl": quadrant_density_bl,
        "quadrant_density_br": quadrant_density_br,
        "center_density": center_density,
        "row_histogram_peaks": row_histogram_peaks,
        "col_histogram_peaks": col_histogram_peaks,
    }


def extract_protocol_features(matrix: np.ndarray) -> dict:
    n = matrix.shape[0]
    version = version_from_matrix_size(n)

    coords = _format_copy_a_coords(n)
    bits = 0
    for (r, c) in coords:
        if r < matrix.shape[0] and c < matrix.shape[1]:
            bit = int(matrix[r, c])
        else:
            bit = 0
        bits = (bits << 1) | bit

    ecc_bits, mask_bits = decode_format_info(bits)
    ecc_level = ECC_ORDINAL.get(ecc_bits, 0)
    masking_pattern = mask_bits

    return {
        "version": version,
        "ecc_level": ecc_level,
        "masking_pattern": masking_pattern,
        "num_alignment_patterns": num_alignment_patterns(version),
        "required_remainder_bits": required_remainder_bits(version),
    }, ECC_NAME_BY_BITS.get(ecc_bits, "L")


def extract_features(matrix: np.ndarray) -> dict:
    protocol_feats, ecc_name = extract_protocol_features(matrix)
    stat_feats = extract_statistical_features(matrix)
    feats = {**protocol_feats, **stat_feats}
    return feats, ecc_name


def extract_from_image(img_path: Path):
    binary = preprocess(img_path)
    cropped = crop_to_content(binary)
    module_px = estimate_module_size(cropped)
    matrix = to_module_matrix(cropped, module_px)
    feats, ecc_name = extract_features(matrix)
    return feats, ecc_name


def run(manifest_path, qr_dir, out_path, self_check=True):
    manifest_path = Path(manifest_path)
    qr_dir = Path(qr_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(manifest_path)
    rows = []
    version_matches = 0
    ecc_matches = 0
    n_checked = 0

    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Extracting structural features"):
        img_path = qr_dir / row["filename"]
        try:
            feats, ecc_name = extract_from_image(img_path)
        except Exception as e:
            logger.warning(f"Failed to extract features for {img_path}: {e}")
            continue

        feats["filename"] = row["filename"]
        feats["label"] = row["label"]
        rows.append(feats)

        if self_check and "version" in row and "ecc" in row:
            n_checked += 1
            if int(feats["version"]) == int(row["version"]):
                version_matches += 1
            if ecc_name == str(row["ecc"]):
                ecc_matches += 1

    if self_check and n_checked > 0:
        v_acc = version_matches / n_checked
        e_acc = ecc_matches / n_checked
        logger.info(
            f"Self-check vs manifest ground truth: version accuracy={v_acc:.4f} "
            f"({version_matches}/{n_checked}), ecc accuracy={e_acc:.4f} ({ecc_matches}/{n_checked})"
        )
        if v_acc < 0.95 or e_acc < 0.95:
            logger.warning(
                "Structural self-check accuracy below 95%% -- investigate crop_to_content / "
                "estimate_module_size / BCH decoding before trusting downstream results."
            )

    df = pd.DataFrame(rows)
    cols = ["filename"] + FEATURE_NAMES + ["label"]
    df = df[cols]
    df.to_csv(out_path, index=False)
    logger.info(f"Wrote structural features for {len(df)} images -> {out_path}")
    return df


def build_parser():
    p = argparse.ArgumentParser(description="Extract structural QR features (Phase 2.1).")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--qr-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--no-self-check", action="store_true")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run(args.manifest, args.qr_dir, args.out, self_check=not args.no_self_check)
