# Code Review Guide: Tri-Modal Quishing Detection

## 1. Purpose of this document

This document is a reviewer-friendly explanation of the project. It describes:

- what the system detects and why the pipeline is divided into stages;
- what each source file does and which functions are most important;
- how data moves from URLs to QR images, features, models, and result tables;
- which files are inputs, generated artifacts, reports, or optional analyses;
- how to verify a run before reporting a result;
- how to export this Markdown review to PDF.

The central model combines three kinds of evidence:

1. **Structural:** QR version, error-correction information, density, transitions,
   entropy, symmetry, and related image statistics.
2. **Visual:** a frozen MobileNetV2 image embedding, or an optional raw-pixel
   representation.
3. **Lexical:** URL scalar features plus character n-gram TF-IDF features.

The final XGBoost classifier receives the concatenated feature vector and predicts
whether the URL is phishing. In the configured model, the feature dimensions are:

```text
24 structural + 1,280 visual + 5,010 lexical = 6,314 fused features
```

Here, `1` means phishing and `0` means legitimate after dataset normalization.

## 2. One-minute architecture summary

```text
raw URL dataset
    |
    v
load_datasets.py       normalize labels and create a balanced url,label CSV
    |
    v
generate_qr.py         render URLs as QR PNGs and save generation metadata
    |
    +--> structural.py  decode QR structure and compute 24 features
    +--> visual.py      compute MobileNetV2 or raw-pixel image features
    +--> lexical.py     decode URL text and compute scalar + TF-IDF features
    |
    v
fusion.py              align rows, scale structural data, concatenate branches
    |
    v
train.py               ablations, Optuna tuning, and final XGBoost model
    |
    v
evaluate.py            metrics, curves, confusion matrix, importance, thresholds
```

`run_pipeline.py` is the orchestrator. It calls the same phase modules in this
order and supports skipping completed extraction stages.

## 3. How to run the main pipeline

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python3 scripts/load_datasets.py \
  --prasad data/urls/raw/PhiUSIIL_Phishing_URL_Dataset.csv \
  --out data/urls/sampled.csv \
  --n-per-class 25000

python3 run_pipeline.py \
  --url-csv data/urls/sampled.csv \
  --visual-backbone mobilenet_v2 \
  --visual-batch-size 8 \
  --optuna-trials 50 \
  --optuna-timeout 3600
```

On macOS, prediction scripts already set the OpenMP environment variables, but
they are useful when importing XGBoost directly:

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
```

For long runs, use `caffeinate -i` so the computer does not sleep while visual
features or hyperparameters are being computed.

## 4. File-by-file review

### Root files

| File | Major role | Important code or review point |
|---|---|---|
| `run_pipeline.py` | End-to-end orchestrator | `main()` parses options, creates directories, creates the primary train index, calls all feature stages, then trains and evaluates. The train index is created before lexical extraction so TF-IDF can be fit only on training rows. |
| `README.md` | Short project overview | Documents setup, label convention, pipeline order, macOS settings, artifact verification, known-good historical numbers, and design decisions. Treat reported AUC values as validation history, not a guarantee for every rebuild. |
| `MASTERLIST_GUIDE.md` | Operational runbook | Contains official commands, expected outputs, paper/table mapping, troubleshooting, and extension-script usage. It complements this reviewer guide. |
| `Pointers.md` | Study and presentation notes | Organizes the project into review units and lists likely questions about each important function. |
| `requirements.txt` | Core dependencies | Installs data processing, image processing, QR, PyTorch, XGBoost, Optuna, and plotting libraries. Several versions are ranges, so exact reproducibility is not fully locked. |
| `requirements-optional.txt` | Optional dependency | Installs SHAP separately because its native dependency chain can fail to build. No core pipeline stage requires it directly. |

### Shared utilities and data preparation

#### `scripts/utils.py`

This is the shared foundation. `GLOBAL_SEED` defines the default reproducibility
seed. `set_seed()` seeds Python, NumPy, and optionally PyTorch. `get_logger()`
provides consistent timestamped logging. `load_fused_bundle()` is the canonical
loader for fused arrays, labels, feature names, branch boundaries, split indices,
and optionally the saved model. Extension scripts should use this helper instead
of rebuilding the feature matrix independently.

#### `scripts/load_datasets.py`

This is the input normalization boundary. `load_phiusiil()` reads PhiUSIIL and
inverts its source label so the project convention is `1 = phishing`.
`load_kaitholikkal()` and `load_kaitholikkal_single_label_file()` support two
optional Kaitholikkal file layouts. `_sniff_delimiter()` handles comma/tab input.
`sample_balanced()` selects the requested number of rows per class. `run()` is
the programmatic entry point used by the command line and size experiments.

Output: a normalized CSV with at least `url` and `label` columns.

#### `scripts/generate_qr.py`

`detect_encoding_mode()` chooses QR alphanumeric mode when the URL qualifies and
byte mode otherwise. `choose_ecc()` selects an error-correction level that can
hold the URL. `generate()` renders each URL, writes a PNG, and records the
filename, URL, label, ECC level, QR version, and encoding mode in `manifest.csv`.

The manifest is important because it preserves ground truth metadata that cannot
be reliably recovered from a processed PNG. `structural.py` later compares its
decoded QR version and ECC values against this manifest.

### Feature extraction

#### `scripts/structural.py`

This module creates the 24-dimensional structural branch. The main public flow is
`run()` -> `extract_from_image()` -> `extract_features()`.

Important helpers include:

- `preprocess()`: denoises and thresholds the QR image;
- `crop_to_content()`: removes the quiet-zone border before module analysis;
- `estimate_module_size()`: estimates QR module size from the cropped image;
- `to_module_matrix()`: converts pixels into a QR module grid;
- `extract_protocol_features()`: decodes protocol-level information;
- `extract_statistical_features()`: computes density, transitions, entropy,
  symmetry, quadrant, and histogram-related measurements;
- `_bch_encode()` and `decode_format_info()`: support format-information
  decoding; `_self_test_bch()` checks the decoder assumptions.

The `int16` conversion before subtraction is essential. Subtracting values in
`uint8` can wrap negative values and corrupt symmetry or transition features.
The quiet-zone crop and module-size rules are similarly correctness-critical.
`run()` writes `structural.csv` and reports version/ECC self-check accuracy.

#### `scripts/visual.py`

`_load_mobilenet()` loads ImageNet MobileNetV2, replaces its classifier with
`nn.Identity()`, and returns a frozen feature extractor. The preprocessing is
224x224 resize, tensor conversion, and ImageNet normalization; no augmentation is
used. `extract_cnn_features()` performs batched extraction and saves checkpoints
so a long run can resume. `extract_raw_pixels()` is a lightweight 69x69 grayscale
fallback. `run()` selects the requested backbone.

Outputs are normally `visual.npy` with 1,280 columns for MobileNetV2 or 4,761
columns for raw pixels. A checkpoint should only be reused with the same manifest
and backbone; the current code does not fingerprint those settings.

#### `scripts/lexical.py`

`decode_qr_payload()` recovers URL text from a QR image. `scalar_features()`
computes nine URL properties such as length, host characteristics, punctuation,
digits, suspicious keywords, and query information. `run()` scales the nine
scalars and creates character TF-IDF features using `analyzer="char_wb"` and
`ngram_range=(3, 5)`.

The lexical output is 5,010 columns: 9 scaled scalar features, up to 5,000 TF-IDF
features, and one `has_lexical` flag. When decoding fails, the lexical values are
zero-filled and the flag identifies the failure. If a train-index file is passed,
the vectorizer and scaler are fit only on training rows, then transformed across
all rows. This prevents test-set leakage.

### Fusion, training, and evaluation

#### `scripts/fusion.py`

`run()` aligns structural, visual, and lexical rows by filename, validates row
counts, fits `StandardScaler` on the primary training rows, transforms the
structural branch, and concatenates the three branches. It saves:

- `fused_features.npy` and `labels.npy`;
- `feature_names.txt`;
- `branch_boundaries.npy`;
- `split_70_30.npz`, `split_80_20.npz`, and their train-index files;
- `structural_scaler.pkl`.

`branch_boundaries` is the contract used by ablation and analysis scripts to
slice structural, visual, and lexical columns. The primary split's train rows are
used for structural-scaler fitting, even when a secondary split is also saved.

#### `scripts/train.py`

`slice_branches()` selects requested branches. `run_ablation()` trains the seven
branch combinations with fixed XGBoost parameters so the comparison is fair.
`optuna_tune()` searches the full fused model using stratified cross-validation,
TPE sampling, and median pruning. `train_final_model()` trains the selected final
model with an internal validation carve-out and early stopping.

The important distinction is that the ablation models are fixed-parameter
comparisons, while the final fused model is tuned. Outputs include
`ablation_results.csv`, optional `model_comparison.csv`, and
`models/xgboost_fused.pkl`.

#### `scripts/evaluate.py`

`compute_metrics()` calculates classification metrics. `threshold_sweep()` and
`find_optimal_threshold()` study operating points. `feature_importance()` maps
XGBoost gain values back to feature names. `per_sample_top3()` provides a small
explainability view. `baseline_vs_fused()` compares a structural-only baseline
with the saved tuned fused model. `run()` writes final metrics, ROC and PR point
tables, confusion matrix counts, feature importance, threshold sweep results, and
the best-F1 threshold.

The fixed classification threshold is `0.65`; AUC uses raw probabilities and is
not dependent on that threshold.

#### `scripts/verify_artifacts.py`

This is the pre-publication safety check. It checks vectorizer vocabulary,
scaler dimensions and statistics, feature-name agreement, model input dimension,
and artifact timestamps. It is designed to catch a model paired with a different
TF-IDF vectorizer or scaler after a later rerun.

Review note: the `expected_visual_dims` option is accepted by the command line but
is not currently enforced by the implementation. The check also cannot prove
semantic compatibility when two artifacts have the same dimensions but different
column meanings.

### Generalization and baseline experiments

#### `scripts/cross_dataset_eval.py`

`assemble_features()` runs QR generation and feature extraction for a new URL
dataset, then uses saved TF-IDF, lexical-scaler, and structural-scaler artifacts
with `transform()` only. `chunked_predict_proba()` limits memory during inference.
`run()` writes cross-dataset predictions and metrics. This is the correct path for
testing an unseen dataset, but a missing structural scaler can make features
incompatible with the trained model.

#### `scripts/replicate_baseline_sizes.py`

`run_one_size()` reproduces the structural-only baseline over sample sizes from
200 to 200,000. It uses fixed QR generation settings, structural extraction, a
train/validation/test split, and XGBoost. This is independent of the full
tri-modal run and exists for comparison with the QRiS baseline experiment.

### Statistical, error, and robustness analyses

#### `scripts/stats_significance.py`

`bootstrap_auc_diff_ci()` estimates confidence intervals for AUC differences.
`permutation_test_auc_diff()` tests whether two configurations differ beyond
random variation. `get_config_proba()` obtains probabilities for selected
ablation configurations. `run()` writes `stats_significance.csv`.

#### `scripts/error_analysis.py`

`run()` joins predictions with the manifest and structural metadata, identifies
false positives and false negatives, and summarizes errors by QR version, ECC,
and other available properties. It produces full outcome, error-only, breakdown,
and summary tables.

#### `scripts/evasion_features.py`

`branch_noise_sweep()` perturbs one feature branch at a time. 
`targeted_feature_perturbation()` changes selected high-impact features. The
script measures how predictions move under synthetic feature-space changes. This
is not a physical QR attack and should be described as a model-sensitivity proxy.

#### `scripts/robustness_tests.py`

The image perturbation helpers apply Gaussian noise, motion blur, rotation,
brightness changes, and JPEG recompression. `perturb_and_save()` creates test
images, `score_perturbed_batch()` re-extracts structural and visual features, and
`run()` measures AUC degradation. The clean lexical array is reused, so this is
primarily an image robustness test rather than a full end-to-end camera test.

#### `scripts/recalibrate_threshold.py`

`sweep()` evaluates thresholds for best F1 and minimum expected cost. `run()` can
operate on an existing fused split or a new dataset. False-negative and
false-positive costs are assumptions supplied by the reviewer or experimenter;
they are not learned from operational data.

### Comparative, scaling, and profiling scripts

#### `scripts/compare_backbones.py`

Runs the visual-only and fused comparisons for MobileNetV2 and raw pixels using a
common split and fixed training configuration. It produces per-backbone feature
arrays and comparison results.

#### `scripts/dataset_size_experiment.py`

Runs the full tri-modal model at several dataset sizes to measure scaling. It is
the companion to `replicate_baseline_sizes.py`, which measures structural-only
scaling. It uses fixed XGBoost parameters and is not a replacement for the main
Optuna-tuned pipeline.

#### `scripts/multi_split_experiment.py`

`mode_a_existing_splits()` compares existing 70/30 and 80/20 bundles.
`mode_b_multi_seed()` repeats fixed-parameter training across seeds.
`_train_eval_fixed()` is the shared training/evaluation helper. The script
measures stability; it does not replace the final saved tuned model.

#### `scripts/profile_pipeline.py`

`profile_extraction_stages()` measures wall time and memory-related information
for QR generation, structural extraction, visual extraction, and lexical
extraction. `profile_inference_latency()` measures prediction latency. The
current implementation does not actually time the fusion stage, so the results
should not be described as a complete end-to-end mobile benchmark.

## 5. Important generated files

| Location | Meaning |
|---|---|
| `data/urls/*.csv` | Normalized URL and label inputs. |
| `data/qr_images/*.png` | Synthetic QR images generated from URLs. |
| `data/features/manifest.csv` | Row identity and QR ground truth metadata. |
| `data/features/structural.csv` | 24 structural features plus identity/label. |
| `data/features/visual.npy` | Visual feature matrix. |
| `data/features/lexical.npy` | Lexical feature matrix. |
| `data/features/fused_features.npy` | Concatenated model input matrix. |
| `data/features/branch_boundaries.npy` | Column ranges for each feature branch. |
| `data/features/split_*.npz` | Train/test indices and arrays for a split. |
| `models/*.pkl` | Trained model, TF-IDF vectorizer, and fitted scalers. |
| `results/*.csv` | Metrics, curves, ablation results, errors, and analysis outputs. |

The default paths are shared. Running a second experiment can overwrite artifacts
from the first experiment, especially the vectorizer, scalers, feature arrays,
model, and result CSVs. Use a separate `--project-root` for each experiment or
archive outputs before starting the next run.

## 6. Reviewer checklist

Before accepting a reported result, check the following:

1. The input labels were normalized so `1` means phishing.
2. The QR manifest, structural CSV, visual array, and lexical array have the
   same row identity and order.
3. Structural scaling was fit on training rows only.
4. TF-IDF and lexical scaling were fit on training rows only.
5. `structural.py` reports near-perfect version/ECC self-check accuracy.
6. `verify_artifacts.py` passes before evaluation.
7. The saved tuned model, vectorizer, and scalers came from the same run.
8. AUC is reported separately from thresholded accuracy, precision, recall, and
   F1.
9. Results from historical generated artifacts are distinguished from a fresh
   reproducible run.
10. Robustness and evasion claims are not overstated: one is synthetic image
    perturbation and the other is feature-space perturbation.

Known review risks are shared output paths, incomplete dependency pinning,
checkpoint reuse without manifest validation, and the current unused visual
dimension check in `verify_artifacts.py`.

## 7. How to send this review as a PDF

The deliverable is intentionally stored as Markdown so it remains editable and
version-controlled. To create a PDF in VS Code:

1. Open `CODE_REVIEW_GUIDE.md`.
2. Install a Markdown PDF extension, such as **Markdown PDF**, if one is not
   already installed.
3. Open the Command Palette with `Cmd+Shift+P`.
4. Run **Markdown PDF: Export (pdf)**.
5. The extension writes `CODE_REVIEW_GUIDE.pdf` beside this Markdown file.

Alternative command-line methods are:

```bash
# With Pandoc installed:
pandoc CODE_REVIEW_GUIDE.md -o CODE_REVIEW_GUIDE.pdf

# Or open the Markdown preview in VS Code and use the preview's print dialog:
# Cmd+Shift+V, then Cmd+P, choose Save as PDF.
```

The PDF is a presentation copy. Keep the Markdown file as the authoritative
review document because it can be updated when the code changes.