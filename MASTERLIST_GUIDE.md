# MASTERLIST GUIDE — How to Run, Verify, and Present This Framework

This is the single "start here" document for the whole codebase: setup,
exact commands for the thesis's official experiments, what each output file
means, and how to map results onto your paper's tables/figures for your
defense or manuscript.

Read this top to bottom once, then use it as a checklist.

---

## 0. Does this codebase match the paper? (honest summary)

Cross-checked against your uploaded thesis PDF. Matches confirmed:

| Paper spec | Code |
|---|---|
| 24 structural features (5 protocol + 19 statistical), Table 3 | `structural.py` `FEATURE_NAMES` — exact match |
| StandardScaler on structural features (Fig 1, Table 10) | `fusion.py` — **added in this revision**, fit on train split only |
| Visual: frozen MobileNetV2, GAP, 1280-dim, ImageNet norm, 224×224 bilinear, no augmentation | `visual.py` — exact match |
| Lexical: 9 scalars + 5000-dim char_wb(3,5) TF-IDF + has_lexical flag = 5010 | `lexical.py` — exact match, including the exact 9 scalar names and 10 suspicious keywords |
| Fused vector: 24+1280+5010 = 6,314 | `fusion.py` — exact match |
| Threshold τ = 0.65 | `evaluate.py`, `train.py` — exact match |
| Optuna: 50 trials, 1hr timeout, 5-fold stratified CV, TPE + MedianPruner, search space incl. reg_alpha/reg_lambda | `train.py` — exact match |
| 7-config ablation (structural/visual/lexical/all pairs/full) | `train.py` `ABLATION_CONFIGS` — exact match |
| PhiUSIIL-only training data (Kaitholikkal excluded per Scope & Limitations) | `load_datasets.py` — Kaitholikkal only loaded if you explicitly pass its flags |
| Both 70/30 and 80/20 splits produced, best one carried forward (Table 12) | `fusion.py` produces both; **you run train+evaluate on each and compare manually** — see Section 4 below |

**Two things flagged in your own paper, not silently "fixed" by guessing:**

1. **Table 3 vs Table 4 disagree.** Table 3 lists the 19 statistical features
   (QR Density, QR Mean Density, row/col transitions, entropy, asymmetry,
   quadrant densities, histogram peaks — this is what's implemented). Table 4
   describes a partially different set with different names (Finder Pattern
   Symmetry Score, Timing Pattern Continuity, Module Edge Contour Count,
   etc.) that don't cleanly map to Table 3. The code follows **Table 3**,
   because it's internally consistent with your dimension math (24 total,
   6,314 fused) and was the version validated against ground truth (100%
   version/ECC self-check accuracy — see Section 5). If your defense panel
   asks about this, you now know it's a real inconsistency between two
   tables in the manuscript, not a coding error — worth tightening in your
   final draft.
2. **Section 2.5.1 says "24 structural, 1,280 visual, and 12 lexical
   features"** — this conflicts with the 5,010-dim lexical branch stated
   everywhere else (Table 8, Table 9, Table 11, the 6,314 total). Almost
   certainly a leftover typo from an earlier draft. Code follows the 5,010
   figure since it's consistent with your stated 6,314 total.

Everything else lines up cleanly.

---

## 1. One-time setup

```bash
cd project
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-optional.txt   # SHAP, optional — safe to skip if it fails to build
```

**macOS only** — do this before running any script that predicts with
XGBoost (train.py/evaluate.py/cross_dataset_eval.py already set these
programmatically, but set them yourself too if you ever `import xgboost`
directly, e.g. in a notebook):
```bash
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
```

**Get your data.** Download PhiUSIIL (`PhiUSIIL_Phishing_URL_Dataset.csv`)
from Prasad & Chandra [16] and place it in `data/urls/raw/`. Kaitholikkal
is NOT needed for the main thesis experiments (your Scope & Limitations
excludes it from training) — only fetch it if you want an optional
cross-dataset generalizability check (Section 6).

---

## 2. Running the official thesis experiment (50k / 100k / 200k)

Your specific objective 1.2.2.1 calls for three stratified, class-balanced
configurations: 50,000, 100,000, and 200,000 total samples. Run each one
as its own pass (they're independent — you don't need to re-download data
between runs).

### Step-by-step for ONE configuration (repeat for each size)

```bash
# --- 50,000-sample configuration (25,000 legit + 25,000 phishing) ---

# 1. Sample the URLs
python3 scripts/load_datasets.py \
    --prasad data/urls/raw/PhiUSIIL_Phishing_URL_Dataset.csv \
    --out data/urls/sampled_50k.csv \
    --n-per-class 25000

# 2-5. Run the full pipeline: QR generation -> structural -> visual -> lexical
#      -> fusion -> Optuna-tuned training -> evaluation
python3 run_pipeline.py \
    --url-csv data/urls/sampled_50k.csv \
    --visual-backbone mobilenet_v2 \
    --visual-batch-size 8 \
    --optuna-trials 50 \
    --optuna-timeout 3600
```

Repeat with `--n-per-class 50000` (100k total) and `--n-per-class 100000`
(200k total) into separate `sampled_100k.csv` / `sampled_200k.csv` files.
**Important**: `run_pipeline.py` writes into shared `data/qr_images/`,
`data/features/`, `models/`, `results/` paths — back up or rename each
configuration's outputs before starting the next one, e.g.:

```bash
mv data/features results/run_50k_features && mkdir data/features
mv models results/run_50k_models && mkdir models
mv results/*.csv results/run_50k_results/ 2>/dev/null
```

Or simpler: pass `--project-root` pointed at a fresh directory per
configuration (`run_50k/`, `run_100k/`, `run_200k/`), each with its own
`data/`, `models/`, `results/` subfolders.

### What this produces (per configuration)

| File | Contents |
|---|---|
| `results/ablation_results.csv` | AUC/accuracy for all 7 branch configs → **Table 15 data** |
| `results/model_comparison.csv` | RF/LogReg/XGBoost(/LightGBM/CatBoost) comparison — bonus, not in thesis body |
| `models/xgboost_fused.pkl` | Final Optuna-tuned Config-7 model |
| `results/baseline_vs_fused.csv` | Structural-only baseline vs. your tuned fused model → **your headline result** |
| `results/final_metrics.csv` | Accuracy/precision/recall/F1/AUC at τ=0.65 |
| `results/roc_curve.csv`, `results/pr_curve.csv` | Raw points for **Figure 11 (AUC-ROC curve)** |
| `results/confusion_matrix.csv` | TP/TN/FP/FN counts → **Figure 10** |
| `results/feature_importance.csv` | Gain-based importance, mapped to real feature names |
| `results/threshold_sweep.csv` | Metrics at τ = 0.10 to 0.90 |
| `results/optimal_threshold.csv` | Best-F1 threshold, for discussion vs. your fixed τ=0.65 choice |

---

## 3. Long-run practicalities (100k / 200k configurations)

- **Time**: QR generation + structural extraction is fast (minutes even at
  200k). The MobileNetV2 visual extraction and Optuna tuning are the slow
  parts — budget several hours at 100k-200k on a laptop CPU.
- **macOS**: run under `caffeinate -i` to stop the machine sleeping
  mid-run:
  ```bash
  caffeinate -i python3 run_pipeline.py --url-csv data/urls/sampled_200k.csv ...
  ```
- **Low RAM**: keep `--visual-batch-size 8` (default). MobileNetV2
  extraction is checkpointed — if it crashes, just re-run the same command;
  it resumes from `data/features/visual.checkpoint.npy` automatically
  instead of starting over.
- **Resuming a full pipeline after a crash**: use the `--skip-*` flags once
  earlier phases have already produced their output files:
  ```bash
  python3 run_pipeline.py --url-csv data/urls/sampled_200k.csv \
      --skip-qr --skip-structural --skip-visual   # only lexical/fusion/train/eval remain
  ```

---

## 4. Choosing between the 70/30 and 80/20 splits (Table 12)

`fusion.py` always produces both `split_70_30.npz` and `split_80_20.npz`.
The thesis methodology is to train+evaluate on both and carry forward
whichever has the higher **validation AUC**. Run both explicitly:

```bash
python3 scripts/train.py --features-dir data/features --models-dir models_70_30 \
    --results-dir results_70_30 --split-name split_70_30 \
    --optuna-trials 50 --optuna-timeout 3600
python3 scripts/evaluate.py --features-dir data/features --models-dir models_70_30 \
    --results-dir results_70_30 --split-name split_70_30

python3 scripts/train.py --features-dir data/features --models-dir models_80_20 \
    --results-dir results_80_20 --split-name split_80_20 \
    --optuna-trials 50 --optuna-timeout 3600
python3 scripts/evaluate.py --features-dir data/features --models-dir models_80_20 \
    --results-dir results_80_20 --split-name split_80_20
```

Compare `results_70_30/final_metrics.csv` vs `results_80_20/final_metrics.csv`
on AUC; report the winner as your primary result, and you can mention the
other as a stability check (your paper already frames std across splits as
evidence against overfitting — see Section 16 of the original build notes).

---

## 5. Sanity-check the structural pipeline (do this once, first)

Before trusting anything at scale, confirm structural extraction is
decoding QR protocol bits correctly. `structural.py` self-checks
automatically against `generate_qr.py`'s own ground truth every time it
runs — watch for this line in the logs:

```
Self-check vs manifest ground truth: version accuracy=1.0000 (.../...),  ecc accuracy=1.0000 (.../...)
```

If either accuracy drops meaningfully below 1.0 at your actual sample size,
stop and investigate before trusting downstream numbers — this exact check
caught real bugs during development (see README's "Key design decisions").

---

## 6. Before presenting/reporting any number: run the artifact check

```bash
python3 scripts/verify_artifacts.py \
    --features-dir data/features --models-dir models \
    --expected-max-features 5000 --expected-visual-dims 1280
```

Exit code 0 + `=== ALL CHECKS PASSED ===` means your model, TF-IDF
vectorizer, structural scaler, and lexical scaler are all mutually
consistent (not silently desynced from a re-run with different settings).
**Run this before every reported number**, not just once at the start —
it's cheap and catches the most damaging real failure mode found during
development (a model paired with the wrong vectorizer, same dimensions,
garbage predictions, no crash).

---

## 7. Optional: cross-dataset generalizability check (not required by thesis scope, but strengthens a defense answer to "how does it generalize?")

Requires Kaitholikkal's dataset (or any other held-out URL set with a
`url,label` CSV).

```bash
python3 scripts/cross_dataset_eval.py \
    --url-csv data/urls/kaitholikkal_holdout.csv \
    --qr-dir data/qr_images_cross \
    --manifest data/features/cross_manifest.csv \
    --model models/xgboost_fused.pkl \
    --tfidf models/lexical_tfidf.pkl \
    --scaler models/lexical_scaler.pkl \
    --structural-scaler models/structural_scaler.pkl \
    --out results/cross_dataset_eval.csv \
    --backbone mobilenet_v2
```

Frame this explicitly as **beyond your stated scope** if you present it —
your Scope & Limitations section says PhiUSIIL-only, so a cross-dataset
number is a bonus robustness discussion, not a claimed contribution.

---

## 8. Optional: reproducing the QRiS baseline's own numbers (Table V of [14])

This is separate from your tri-modal model. It reruns QRiS's own
methodology (structural-only, 8 sample sizes from 200 to 200,000) so you
can show a direct, apples-to-apples baseline comparison table in your
defense, beyond the single-baseline-row your own `baseline_vs_fused.csv`
already gives you.

```bash
python3 scripts/replicate_baseline_sizes.py \
    --prasad data/urls/raw/PhiUSIIL_Phishing_URL_Dataset.csv \
    --outdir data/baseline_replication \
    --results-out results/baseline_replication_results.csv \
    --sizes 200 1000 2000 6000 20000 50000 100000 200000
```

This is heavy at the high end (multi-hour to multi-day) — start with the
smaller sizes to confirm it's working, use `caffeinate -i` on macOS for the
full run. The script logs each size's result next to QRiS's own published
numbers for a quick side-by-side sanity check.

---

## 9. Mapping outputs to your paper's sections, for slides/defense

| Paper element | Where the number/figure comes from |
|---|---|
| Table 12 (Train/Test split counts) | `data/features/split_70_30.npz` and `split_80_20.npz` — `idx_train`/`idx_test` lengths |
| Table 15 / ablation study | `results/ablation_results.csv` |
| Figure 10 (Confusion Matrix) | `results/confusion_matrix.csv` — plot as a heatmap |
| Figure 11 (AUC-ROC curve) | `results/roc_curve.csv` — plot `fpr` vs `tpr` |
| "Baseline vs proposed" headline comparison | `results/baseline_vs_fused.csv` |
| Gain-based feature importance discussion | `results/feature_importance.csv` |
| Optimal threshold discussion (vs. fixed τ=0.65) | `results/optimal_threshold.csv`, `results/threshold_sweep.csv` |
| Per-sample explainability example | `evaluate.per_sample_top3()` — call directly in a notebook/script on a few test rows if you want a worked example slide |

A minimal plotting snippet for your slides (matplotlib, using the raw CSVs
above — no need to rerun anything):

```python
import pandas as pd
import matplotlib.pyplot as plt

roc = pd.read_csv("results/roc_curve.csv")
plt.plot(roc["fpr"], roc["tpr"])
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Tri-Modal Fused Model")
plt.savefig("results/roc_curve.png", dpi=150)
```

---

## 10. Extension scripts (9 additional analyses + stability/replication tooling)

All 9 previously-missing extension scripts are now built and smoke-tested.
Every one of them reuses the validated feature-assembly paths — either
`utils.load_fused_bundle()` (for already-fused data) or
`cross_dataset_eval.assemble_features()` (for brand-new URL sets) — rather
than hand-reassembling features. That's a direct fix for the one script
that had a known bug (`recalibrate_threshold.py`, previously produced a
nonsensical AUC=0.46 from a hand-rolled reassembly path).

Run `verify_artifacts.py` before any of these, same as always.

| Script | What it does | Typical command |
|---|---|---|
| `stats_significance.py` | Bootstrap CI + permutation-test p-values on AUC differences between ablation configs — is fused really better than structural-only, or within noise? | `python3 scripts/stats_significance.py --features-dir data/features --results-dir results --split-name split_70_30` |
| `error_analysis.py` | Breaks down false positives/negatives by QR version/ECC/masking pattern; saves full per-sample outcome table | `python3 scripts/error_analysis.py --features-dir data/features --models-dir models --results-dir results --manifest data/features/manifest.csv --structural-csv data/features/structural.csv` |
| `evasion_features.py` | Feature-space perturbation sweep per branch + targeted top-feature perturbation — how "cheap" would evasion be? | `python3 scripts/evasion_features.py --features-dir data/features --models-dir models --results-dir results` |
| `compare_backbones.py` | Compares MobileNetV2 vs raw-pixel visual backbones on visual-only and fused AUC | `python3 scripts/compare_backbones.py --manifest data/features/manifest.csv --qr-dir data/qr_images --structural-csv data/features/structural.csv --lexical-npy data/features/lexical.npy --results-dir results` |
| `dataset_size_experiment.py` | Data-scaling curve for the FULL tri-modal model (companion to `replicate_baseline_sizes.py`, which only covers structural-only) | `python3 scripts/dataset_size_experiment.py --prasad data/urls/raw/PhiUSIIL_Phishing_URL_Dataset.csv --outdir data/size_exp --results-out results/dataset_size_experiment.csv --sizes 1000 2000 6000 20000 50000` |
| `profile_pipeline.py` | Per-stage extraction runtime + per-sample inference latency — the actual mobile-deployment feasibility numbers | `python3 scripts/profile_pipeline.py --url-csv data/urls/sampled.csv --qr-dir /tmp/p_qr --manifest /tmp/p_manifest.csv --structural-csv /tmp/p_struct.csv --visual-npy /tmp/p_visual.npy --lexical-npy /tmp/p_lex.npy --tfidf-out /tmp/p_tfidf.pkl --scaler-out /tmp/p_scaler.pkl --features-dir data/features --models-dir models --results-dir results` |
| `robustness_tests.py` | Applies synthetic camera noise/blur/rotation/brightness/JPEG artifacts to QR images, measures AUC degradation — directly answers the paper's stated "no real-world camera noise evaluation" limitation | `python3 scripts/robustness_tests.py --manifest data/features/manifest.csv --clean-qr-dir data/qr_images --model models/xgboost_fused.pkl --tfidf models/lexical_tfidf.pkl --scaler models/lexical_scaler.pkl --structural-scaler models/structural_scaler.pkl --lexical-npy data/features/lexical.npy --out-dir /tmp/robust_scratch --results-dir results` |
| `recalibrate_threshold.py` | Cost-sensitive threshold search (best-F1 and min-expected-cost) vs. the paper's fixed τ=0.65; two modes (existing split, or a brand-new dataset) | `python3 scripts/recalibrate_threshold.py --results-dir results --features-dir data/features --models-dir models --split-name split_70_30 --fn-cost 5 --fp-cost 1` |
| `multi_split_experiment.py` | Quantifies AUC stability across 70/30 vs 80/20, and optionally across multiple random seeds | `python3 scripts/multi_split_experiment.py --features-dir data/features --results-dir results --n-seeds 5` |

At real dataset scale (thousands+ of images), `robustness_tests.py` and
`evasion_features.py` in particular can take a while since they re-run
structural+visual extraction per perturbation level — start with a small
`--sizes`/subset to sanity-check before committing to the full run, same
philosophy as everything else long-running in this codebase.

---

## 11. Quick troubleshooting

| Symptom | Fix |
|---|---|
| `FileNotFoundError` on `--skip-optuna` runs | Shouldn't happen — `train.py` creates `models/`/`results/` unconditionally. If it does, you're on a stale copy of the file; re-check you're using this revision. |
| Predictions look ~random / inverted after re-running a phase with different settings | Run `verify_artifacts.py` — almost certainly a stale/mismatched vectorizer or scaler. |
| macOS segfault during prediction | `export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1` before running. |
| `structural.py` self-check accuracy < 1.0 | Check your QR images weren't re-compressed/resized by something outside this pipeline (e.g. an image viewer re-saving them) — the module-size estimation assumes the exact `border=4, box_size=10` output of `generate_qr.py`. |
| Visual extraction crashes partway through a huge run | Just re-run the same command — it resumes from `data/features/visual.checkpoint.npy` automatically. |
| Machine went to sleep mid-run (macOS), extraction stalled | Use `caffeinate -i` for any run over ~30 minutes. |
| `evasion_features.py` / `robustness_tests.py` seem slow | They rerun feature extraction per perturbation level/branch — expected at scale. Reduce `--sigmas`/`--perturbations`/`--levels` for a quicker pass. |
| `recalibrate_threshold.py` reports AUC near 0.5 | That's the built-in sanity check firing — it means a feature/model mismatch, not a valid recalibration. Run `verify_artifacts.py` first. |
