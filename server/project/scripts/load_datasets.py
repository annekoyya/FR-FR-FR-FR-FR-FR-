"""load_datasets.py -- Phase 1.1 / 1.2

Loads and normalizes URL datasets into a common schema: columns ["url", "label"]
where label==1 means phishing, label==0 means legitimate (pipeline convention).

Known quirks handled here (see REBUILD SPEC for full context):
  * PhiUSIIL's own `label` column is INVERTED relative to the pipeline
    convention (their 1 == legitimate). We invert on load.
  * Kaitholikkal may come as a single file with a Type/type column, OR as two
    separate files (one all-phishing, one all-legitimate) with no label
    column at all. Both layouts are supported.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from server.project.scripts.utils import GLOBAL_SEED, get_logger, set_seed

logger = get_logger("load_datasets")


def load_phiusiil(path: Path) -> pd.DataFrame:
    """Load PhiUSIIL_Phishing_URL_Dataset.csv and invert its label column to
    match the pipeline convention (1 == phishing).
    """
    df = pd.read_csv(path)
    if "URL" in df.columns and "url" not in df.columns:
        df = df.rename(columns={"URL": "url"})
    if "label" not in df.columns:
        raise ValueError(f"PhiUSIIL file at {path} is missing a 'label' column")

    out = pd.DataFrame()
    out["url"] = df["url"].astype(str)
    # PhiUSIIL: label==1 means legitimate. Pipeline convention: 1==phishing.
    out["label"] = 1 - df["label"].astype(int)
    logger.info(f"Loaded PhiUSIIL: {len(out)} rows from {path} (label inverted)")
    return out


def _sniff_delimiter(path: Path) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline()
    return "\t" if "\t" in first_line and "," not in first_line else ","


def load_kaitholikkal(path: Path) -> pd.DataFrame:
    """Load a single Kaitholikkal-style file with a url + Type/type column.
    Handles comma or tab separated, and case-varying Phishing/legitimate
    string labels.
    """
    delim = _sniff_delimiter(path)
    df = pd.read_csv(path, sep=delim, engine="python")
    df.columns = [c.strip() for c in df.columns]

    url_col = next((c for c in df.columns if c.lower() == "url"), None)
    type_col = next((c for c in df.columns if c.lower() == "type"), None)
    if url_col is None or type_col is None:
        raise ValueError(
            f"Kaitholikkal file at {path} must have a url and Type/type column, "
            f"found columns: {list(df.columns)}"
        )

    out = pd.DataFrame()
    out["url"] = df[url_col].astype(str)
    out["label"] = (
        df[type_col].astype(str).str.strip().str.lower().map(
            lambda v: 1 if v == "phishing" else 0
        )
    )
    logger.info(f"Loaded Kaitholikkal (single-file): {len(out)} rows from {path}")
    return out


def load_kaitholikkal_single_label_file(path: Path, forced_label: int) -> pd.DataFrame:
    """Load one of Kaitholikkal's two-file layout: a file containing only
    URLs (no label column), all sharing `forced_label`. Auto-detects
    header-vs-headerless and comma-vs-tab separator.
    """
    delim = _sniff_delimiter(path)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline().strip()

    has_header = not first_line.lower().startswith("http")
    if has_header:
        df = pd.read_csv(path, sep=delim, engine="python")
        url_col = df.columns[0]
        urls = df[url_col].astype(str)
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f, delimiter=delim)
            urls = pd.Series([row[0] for row in reader if row])

    out = pd.DataFrame()
    out["url"] = urls.astype(str)
    out["label"] = forced_label
    logger.info(
        f"Loaded Kaitholikkal (single-label file, label={forced_label}): "
        f"{len(out)} rows from {path} (header={has_header}, delim={'tab' if delim == chr(9) else 'comma'})"
    )
    return out


def sample_balanced(df: pd.DataFrame, n_per_class: int, seed: int = GLOBAL_SEED) -> pd.DataFrame:
    """Sample exactly n_per_class rows from each label. Warns (does not
    error) if fewer rows are available than requested.
    """
    parts = []
    for label in sorted(df["label"].unique()):
        subset = df[df["label"] == label]
        n_avail = len(subset)
        n_take = min(n_per_class, n_avail)
        if n_take < n_per_class:
            logger.warning(
                f"Requested {n_per_class} rows for label={label} but only "
                f"{n_avail} available; taking all {n_avail}."
            )
        parts.append(subset.sample(n=n_take, random_state=seed))
    combined = pd.concat(parts, ignore_index=True)
    combined = combined.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return combined


def run(prasad_path, kaitholikkal_path, out_path, n_per_class, kaitholikkal_holdout_path=None,
        seed=GLOBAL_SEED, kaitholikkal_phishing_path=None, kaitholikkal_legitimate_path=None):
    """Programmatic entry point (used by replicate_baseline_sizes.py and
    run_pipeline.py). Loads PhiUSIIL (required) and optionally Kaitholikkal,
    samples a balanced subset, and writes it to out_path.
    """
    set_seed(seed)
    frames = []
    if prasad_path is not None:
        frames.append(load_phiusiil(Path(prasad_path)))

    if kaitholikkal_path is not None:
        kdf = load_kaitholikkal(Path(kaitholikkal_path))
        if kaitholikkal_holdout_path is not None:
            kdf.to_csv(kaitholikkal_holdout_path, index=False)
            logger.info(f"Kaitholikkal held out (not mixed into training) -> {kaitholikkal_holdout_path}")
        else:
            frames.append(kdf)

    if kaitholikkal_phishing_path is not None and kaitholikkal_legitimate_path is not None:
        kphish = load_kaitholikkal_single_label_file(Path(kaitholikkal_phishing_path), forced_label=1)
        klegit = load_kaitholikkal_single_label_file(Path(kaitholikkal_legitimate_path), forced_label=0)
        kdf = pd.concat([kphish, klegit], ignore_index=True)
        if kaitholikkal_holdout_path is not None:
            kdf.to_csv(kaitholikkal_holdout_path, index=False)
            logger.info(f"Kaitholikkal (two-file) held out -> {kaitholikkal_holdout_path}")
        else:
            frames.append(kdf)

    if not frames:
        raise ValueError("No dataset sources provided; supply at least --prasad")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset="url").reset_index(drop=True)

    if n_per_class is not None:
        combined = sample_balanced(combined, n_per_class, seed)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)
    logger.info(f"Wrote {len(combined)} rows -> {out_path}")
    return combined


def build_parser():
    p = argparse.ArgumentParser(description="Load and normalize URL datasets (Phase 1.1/1.2).")
    p.add_argument("--prasad", type=Path, default=None, help="Path to PhiUSIIL_Phishing_URL_Dataset.csv")
    p.add_argument("--kaitholikkal", type=Path, default=None, help="Single-file Kaitholikkal CSV/TSV")
    p.add_argument("--kaitholikkal-phishing", type=Path, default=None, help="Two-file layout: phishing-only URLs")
    p.add_argument("--kaitholikkal-legitimate", type=Path, default=None, help="Two-file layout: legitimate-only URLs")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-per-class", type=int, default=None)
    p.add_argument(
        "--kaitholikkal-holdout",
        type=Path,
        default=None,
        help="If set, Kaitholikkal is written here and NEVER mixed into --out (for later cross-dataset testing).",
    )
    p.add_argument("--seed", type=int, default=GLOBAL_SEED)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run(
        prasad_path=args.prasad,
        kaitholikkal_path=args.kaitholikkal,
        out_path=args.out,
        n_per_class=args.n_per_class,
        kaitholikkal_holdout_path=args.kaitholikkal_holdout,
        seed=args.seed,
        kaitholikkal_phishing_path=args.kaitholikkal_phishing,
        kaitholikkal_legitimate_path=args.kaitholikkal_legitimate,
    )
