# Quishing Detection: Tri-modal (Structural + Visual + Lexical) XGBoost Framework

Codebase for the thesis *"An Enhanced XGBoost-Based Quishing Detection Framework
Utilizing Tri-modal Feature Fusion of Structural, Visual, and Lexical Indicators."*

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# Optional, kept separate because it can pull in a numba/llvmlite build that
# fails from source on some machines -- never let this abort the core install:
pip install -r requirements-optional.txt
```

### macOS: required before ANY script that predicts with XGBoost

A confirmed, real bug: XGBoost prediction can segfault on macOS due to a
duplicate OpenMP runtime (XGBoost's bundled `libomp.dylib` vs. another copy
loaded elsewhere). `train.py`, `evaluate.py`, and `cross_dataset_eval.py` set
these programmatically via `os.environ.setdefault(...)` before importing
xgboost, but if you ever import xgboost directly yourself, set these first:

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
```

### Long unattended runs (macOS)

Use `caffeinate -i` to prevent the machine from sleeping mid-run. A prior
multi-hour extraction stalled for about an hour, most likely because the
machine slept.

## Data

Place your downloads in `data/urls/raw/`:
- PhiUSIIL: `PhiUSIIL_Phishing_URL_Dataset.csv`
- Kaitholikkal: either a single file with `url` + `Type`/`type` columns, or
  two separate files (all-phishing, all-legitimate) with no label column.

**Important**: PhiUSIIL's own `label` column is inverted relative to this
pipeline's convention (`1 == phishing`). `load_datasets.py` handles this
automatically -- do not "fix" it a second time downstream.

## Pipeline order

```bash
# 1. Load & sample URLs
python3 scripts/load_datasets.py --prasad data/urls/raw/PhiUSIIL_Phishing_URL_Dataset.csv \
    --out data/urls/sampled.csv --n-per-class 25000

# 2-5. Run the full pipeline
python3 run_pipeline.py --url-csv data/urls/sampled.csv
```

Or run phases individually via `scripts/generate_qr.py`, `scripts/structural.py`,
`scripts/visual.py`, `scripts/lexical.py`, `scripts/fusion.py`, `scripts/train.py`,
`scripts/evaluate.py` -- each has a `--help` with full CLI options.

`run_pipeline.py` supports `--skip-qr/--skip-structural/--skip-visual/--skip-lexical`
to re-run only later phases without regenerating everything.

## Structural feature normalization

Per the thesis (Figure 1 / Table 10), the 24 structural features are passed
through a `StandardScaler` before fusion, fit on the training split only
(same leakage-free principle as the lexical TF-IDF/scaler). `fusion.py` does
this automatically and saves the fitted scaler to `models/structural_scaler.pkl`.
`cross_dataset_eval.py` reuses it via `.transform()` only, never re-fitting.

## Before trusting any result

Run `scripts/verify_artifacts.py` -- it catches the most damaging real failure
mode found during development: a trained model whose paired TF-IDF vectorizer
or scaler has been silently overwritten by a later, differently-configured run
(same dimensions, so no crash, but the columns mean something different than
what the model was trained on -- producing near-random or inverted
predictions).

```bash
python3 scripts/verify_artifacts.py --features-dir data/features --models-dir models \
    --expected-max-features 5000 --expected-visual-dims 1280
    
```

## Cross-dataset generalizability

`scripts/cross_dataset_eval.py` scores the trained model against a completely
different URL set (e.g. Kaitholikkal, or a held-out PhiUSIIL slice with zero
training overlap), always calling `.transform()` -- never re-fitting -- on the
saved TF-IDF vectorizer and scaler. On a correctly-configured pipeline from a
prior validated run: in-distribution AUC ~0.998, unseen-but-same-distribution
AUC ~0.998 (no overfitting), Kaitholikkal cross-dataset AUC ~0.82 (a real
generalization gap, consistent with published cross-dataset phishing-detection
literature).

## QRiS baseline replication

`scripts/replicate_baseline_sizes.py` reproduces the QRiS baseline paper's own
structural-only, multi-size XGBoost experiment at their exact 8 sample sizes
(200 to 200,000), for direct comparison against their Table V. This is a
genuinely heavy job at the high end (multi-hour to multi-day) -- start with
smaller sizes first.

## Known-good validated numbers (sanity check for a rebuild)

From a completed real 50k-sample run:

| Ablation config | AUC |
|---|---|
| structural only | ~0.84-0.85 |
| visual only (mobilenet_v2) | ~0.93 |
| visual only (raw_pixels) | ~0.99 |
| lexical only | ~0.998 |
| all fused | ~0.998 |

Bootstrap 95% CI on the final model: AUC = 0.998 +/- 0.0004. 70/30 vs 80/20
split stability: std ~0.00008.

## What's included

Core pipeline: `load_datasets.py`, `generate_qr.py`, `structural.py`,
`visual.py`, `lexical.py`, `fusion.py`, `train.py`, `evaluate.py`,
`verify_artifacts.py`, `cross_dataset_eval.py`, `replicate_baseline_sizes.py`,
`run_pipeline.py`.

Extension scripts (see `MASTERLIST_GUIDE.md` Section 10 for full usage):
`stats_significance.py`, `error_analysis.py`, `evasion_features.py`,
`compare_backbones.py`, `dataset_size_experiment.py`, `profile_pipeline.py`,
`robustness_tests.py`, `recalibrate_threshold.py`, `multi_split_experiment.py`.

## What is NOT included in this rebuild

Nothing from the originally-scoped script list is missing. If you need
something beyond these 21 scripts, describe it and it can be added against
this codebase's now-verified data formats.

## Key design decisions worth remembering

- **TF-IDF must use `analyzer="char_wb", ngram_range=(3,5)`**, not the
  sklearn default `(1,1)`. A single-character analyzer caps the vocabulary at
  ~60-70 tokens regardless of dataset size and cannot distinguish
  typosquatting. This is not a "small dataset" symptom.
- **Structural feature extraction must crop the QR's quiet zone** before
  estimating module size, and module-size estimation must only accept runs
  starting at row 0, taking the max over the first ~30 columns (not the first
  column with any black pixel).
- **The module grid must be cast to `int16`** before any subtraction/diff --
  `uint8` silently wraps around and corrupts every symmetry/transition
  feature.
- **Always fit TF-IDF/scaler on the training split only**, then `.transform()`
  the full set. Fitting on the full set leaks test-set statistics and
  inflates in-distribution numbers.
- **Load the actual saved tuned model for baseline-vs-proposed comparisons.**
  Retraining a fresh model for the "ours" row silently understates the real
  best result.
# FR-FR-FR-FR-FR-FR-
# FR-FR-FR-FR-FR-FR-
