Here is the content converted into a clean, professional Markdown format, perfectly structured for a thesis presentation or codebase walkthrough guide.

***

# Code Study Syllabus: File-by-File Masterlist
*Content for preparing to present and walk through the codebase itself.*

---

## 📚 Study Units

### Unit 0: Orientation — Shared Foundations
- **Files:** `utils.py` + `requirements.txt` + project folder structure
- **Time:** 20 minutes
- **Objective:** Be able to explain, without opening any other file, what every other script imports and relies on: seeding, paths, logging, and the shared `load_fused_bundle()` helper.

**Key Symbols**
| Name | Kind | Description |
| :--- | :--- | :--- |
| `GLOBAL_SEED` | constant | `= 42`. The single seed value every script's `--seed` default points to. |
| `PATHS` | dict | `urls`/`qr_images`/`features`/`models`/`results`, auto-created on import via `.mkdir()`. |
| `set_seed(seed)` | function | Seeds `random`, `numpy`, and `torch` (torch wrapped in try/except since optional). |
| `get_logger(name)` | function | Returns a stdout logger, `%H:%M:%S` timestamps, one per module. |
| `load_fused_bundle(...)` | function | **THE** shared, canonical way every extension script loads an already-fused split + model. Added specifically to prevent the `recalibrate_threshold.py` bug from recurring anywhere else. |

**Pointers**
- Show the auto-mkdir loop at the top of the file — explain why every script can assume its output directories already exist without checking.
- Show `load_fused_bundle()`'s docstring specifically — it names the exact historical bug (AUC=0.46) this function exists to prevent.
- Open `requirements.txt` and `requirements-optional.txt` side by side — be ready to explain **WHY** `shap` is separated out (`llvmlite` build failures) and why `numpy`/`scipy`/`opencv` are hard-pinned (a real, diagnosed dependency conflict).

**Self Check**
- Without scrolling, name the 5 keys in `PATHS`.
- What happens if two scripts call `set_seed()` with different values in the same run — which one wins for a given call?
- Point to the exact line where `load_fused_bundle()` decides whether to load a model at all.

---

### Unit 1: Phase 1.1/1.2 — URL Ingestion and Balanced Sampling
- **Files:** `load_datasets.py`
- **Time:** 25 minutes
- **Objective:** Be able to demo loading PhiUSIIL and explain the label-inversion fix live, plus both Kaitholikkal layouts even though Kaitholikkal isn't used in the main experiments.

**Key Symbols**
| Name | Kind | Description |
| :--- | :--- | :--- |
| `load_phiusiil(path)` | function | Loads PhiUSIIL, inverts label: `pipeline_label = 1 - phiusiil_label`. |
| `load_kaitholikkal(path)` | function | Single-file layout: url + Type/type column, comma or tab. |
| `load_kaitholikkal_single_label_file(...)` | function | Two-file layout: one all-phish, one all-legit, no label column, auto-detects header/delimiter. |
| `sample_balanced(df, n_per_class, seed)` | function | Exact `n_per_class` per label, warns (not errors) if short. |
| `run(...)` | function | Programmatic entry point reused by `run_pipeline.py` and `replicate_baseline_sizes.py`. |

**Pointers**
- Point directly at the line `out['label'] = 1 - df['label'].astype(int)` — this is the single most important line in this file and the one most likely to get a direct "why is this here" question.
- Show the `_sniff_delimiter()` helper — a one-line demonstration of a small, defensive utility.
- Show that `sample_balanced()` warns rather than raises when short on samples — connect this to robustness/graceful-degradation as a design philosophy repeated elsewhere in the codebase.

**Self Check**
- If PhiUSIIL's own file has `label=1` for a row, what value does that row get in the pipeline's output CSV, and why?
- What's the difference in expected file structure between `load_kaitholikkal()` and `load_kaitholikkal_single_label_file()`?
- Where does `--kaitholikkal-holdout` write its output, and does it ever get mixed into `--out`?

---

### Unit 2: Phase 1.3 — Synthetic QR Code Generation
- **Files:** `generate_qr.py`
- **Time:** 20 minutes
- **Objective:** Explain the three validation checks (encoding mode, ECC fit, version) and be able to point to exactly where ground truth gets written, since that ground truth is what `structural.py`'s self-check validates against later.

**Key Symbols**
| Name | Kind | Description |
| :--- | :--- | :--- |
| `detect_encoding_mode(url)` | function | Alphanumeric-charset check with case sensitivity, else byte mode. |
| `MAX_BYTE_CAPACITY_V40` | dict | L:2953, M:2331, Q:1663, H:1273 — max bytes per ECC level at version 40. |
| `choose_ecc(url, rng)` | function | Randomly picks an ECC level from those that can fit the URL; `None` if none fit. |
| `generate(...)` | function | Main entry point; writes PNGs + `manifest.csv` with filename/url/label/ecc/version/mode. |

**Pointers**
- Show the manifest CSV columns live — explicitly say "this is the ground truth `structural.py` checks its own decoding against later."
- Show `qr.make(fit=(fixed_version is None))` — explain the `--fixed-version`/`--fixed-ecc` escape hatch used by `replicate_baseline_sizes.py` for uniform-property baseline replication.
- Show the skip-and-warn behavior when a URL is too long for any ECC level at version 40.

**Self Check**
- What are the exact `box_size` and `border` values passed to `qrcode.QRCode()`, and why does border matter later?
- What's stored in the manifest that a later self-check needs, that couldn't be recovered from the PNG alone?
- How would you generate a batch of QR codes all forced to ECC level H?

---

### Unit 3: Phase 2.1 — The Most Bug-Dense File in the Codebase
- **Files:** `structural.py`
- **Time:** 50 minutes
- **Objective:** This is the file most likely to get deep, specific questions. Be able to name and explain all three historical bugs (crop, module-size, uint8 overflow) from memory, and locate the BCH decoder self-test without searching.

**Key Symbols**
| Name | Kind | Description |
| :--- | :--- | :--- |
| `FEATURE_NAMES` | list[24] | The exact 24 names, in order — memorize the 5 protocol-level ones especially. |
| `preprocess(img_path)` | function | `medianBlur(3)` -> `GaussianBlur((3,3),0)` -> `CLAHE(2.0,(8,8))` -> `threshold(189)`. |
| `crop_to_content(binary)` | function | **BUG FIX #1:** crops the 4-module quiet-zone border before anything else. |
| `estimate_module_size(binary)` | function | **BUG FIX #2:** only accepts runs starting at row 0, takes MAX over first 30 cols. |
| `to_module_matrix` / `extract_features` | functions | **BUG FIX #3:** casts matrix to `int16` before any subtraction. |
| `_bch_encode` / `_self_test_bch()` | function | BCH(15,5) encoder with an assertion-based self-test run on import. |
| `ALIGNMENT_POSITIONS` | dict | Per-version alignment pattern coordinate table, versions 2-40. |
| `run(...)` | function | Runs extraction AND compares decoded version/ECC against the manifest's ground truth. |

**Pointers**
- Open `crop_to_content()` and `estimate_module_size()` side by side — walk through **WHY** the second function depends on the first having already run (the row-0 guarantee).
- Point to `matrix = matrix.astype(np.int16)` at the top of `extract_statistical_features()` — say out loud what `0 - 1` evaluates to in `uint8` vs `int16`.
- Scroll to `_self_test_bch()` and explain it runs automatically on import, not just when called — "fail loud immediately" vs. silent bad decodes buried in a CSV later.
- Show the self-check log line format in `run()` — have this memorized: *"Self-check vs manifest ground truth: version accuracy=X, ecc accuracy=Y"*.
- Be ready to explain `qr_density` vs `qr_mean_density` as two **DIFFERENT** features, not a duplicate.

**Self Check**
- Without looking, what does the quiet-zone bug do to every downstream feature if left unfixed?
- Why does `estimate_module_size()` take the MAX run over 30 columns instead of the FIRST column with any black pixel?
- What published example does the BCH self-test check against, and what does it assert?
- What is the actual generator polynomial and mask constant used in `_bch_encode`?
- If self-check accuracy came back at 80% instead of ~100%, what's the first thing you'd go check?

---

### Unit 4: Phase 2.2 — MobileNetV2 Visual Embeddings
- **Files:** `visual.py`
- **Time:** 30 minutes
- **Objective:** Explain the frozen-CNN pipeline end to end, and be able to justify both the checkpointing design and the `raw_pixels` fallback backbone without notes.

**Key Symbols**
| Name | Kind | Description |
| :--- | :--- | :--- |
| `_load_mobilenet()` | function | `torch.set_num_threads(1)`; loads IMAGENET1K_V1 weights; classifier replaced with `nn.Identity()`; returns `(model, preprocess transform)`. |
| `extract_cnn_features(...)` | function | Checkpointed, resumable batch extraction loop. |
| `extract_raw_pixels(...)` | function | 69x69 grayscale flatten fallback, 4,761 dims, no torch needed. |
| `MOBILENET_DIM` / `RAW_PIXELS_DIM` | constants | 1280 and 4761 respectively. |
| `run(...)` | function | Dispatches to one of the two backbones above. |

**Pointers**
- Show the preprocess transform chain — `Resize(224,224,bilinear)` -> `ToTensor` -> `Normalize(ImageNet mean/std)` — and explicitly state there's no augmentation, then explain why (orientation-dependency).
- Walk through the checkpoint save/resume logic: every `checkpoint_every` batches, `np.save`, then explicit `del` + `gc.collect()`, and `unlink()` on clean completion.
- Point to `torch.set_num_threads(1)` and connect it to the low-RAM design philosophy stated elsewhere.

**Self Check**
- What replaces MobileNetV2's classification head, and why does that specific replacement matter for the output shape?
- If extraction crashes at batch 60 of 200, what happens when you re-run the identical command?
- Why is `raw_pixels`' AUC on visual-only sometimes higher than MobileNetV2's, and why does the thesis still specify MobileNetV2 as primary?

---

### Unit 5: Phase 2.3 — TF-IDF and Scalar URL Features
- **Files:** `lexical.py`
- **Time:** 35 minutes
- **Objective:** Explain the `char_wb(3,5)` TF-IDF choice with the paypal/paypa1 example from memory, and walk through the zero-padding fallback and leakage-free fitting without hesitation — these are high-probability questions.

**Key Symbols**
| Name | Kind | Description |
| :--- | :--- | :--- |
| `SCALAR_FEATURE_NAMES` | list[9] | Exact order matters — memorize it. |
| `SUSPICIOUS_KEYWORDS` | list[10] | login, verify, secure, account, update, banking, confirm, paypal, signin, password. |
| `decode_qr_payload(img_path)` | function | `pyzbar` decode; must start with `http://` or `https://` to count. |
| `scalar_features(url)` | function | Computes all 9 scalars from a `urlparse()` result. |
| `run(...)` | function | The full leakage-free fit-on-train / transform-on-all pipeline; zero-pads on decode failure. |
| `TFIDF_MAX_FEATURES_DEFAULT` / `TFIDF_MIN_DF_DEFAULT` | constants | 5000 and 1. |

**Pointers**
- Show `TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), ...)` and be ready to say the cosine-similarity numbers from memory: `paypal.com` vs `paypa1.com` = 0.725 with this config, vs. no meaningful difference at (1,1).
- Walk through the `fit_mask` / `train_idx` logic — show exactly where `.fit()` vs `.transform()` get called and on which rows.
- Point to the zero-padding block — `scaled_scalars[failed_mask] = 0.0` — and explain `has_lexical` as a signal, not just padding.
- Show the warning that fires if `tfidf_out`/`scaler_out` already exist — this is the file-overwrite-collision bug guard.

**Self Check**
- Why can't a (1,1) character analyzer scale its vocabulary past ~60-70 tokens no matter the dataset size?
- What exact dimension does the lexical branch produce, and how is that number composed?
- What does `has_lexical=0` tell the downstream model that a literal zero value in those columns wouldn't?
- What's the real, confirmed consequence of re-running `lexical.py` with new settings pointed at the SAME output filename as a trained model's vectorizer?

---

### Unit 6: Phase 3 — Concatenation, Structural Scaling, and Splitting
- **Files:** `fusion.py`
- **Time:** 25 minutes
- **Objective:** Explain why the structural `StandardScaler` lives HERE (not in `structural.py` itself), and be able to justify the "primary split anchors the scaler fit" design decision under questioning.

**Key Symbols**
| Name | Kind | Description |
| :--- | :--- | :--- |
| `run(...)` | function | Aligns by filename, fits structural scaler on primary-split train rows, concatenates, saves splits. |
| `branch_boundaries` | dict | `{'structural': (0,24), 'visual': (24,1304), 'lexical': (1304,6314)}` — single source of truth for column slicing everywhere else. |
| `structural_scaler_out` | path | `models/structural_scaler.pkl` — the artifact added in the correctness fix. |

**Pointers**
- Show the exact block: `struct_scaler.fit(X_struct_raw[primary_idx_train])` then `.transform(X_struct_raw)` — this **IS** the fix for the gap found against the thesis's Figure 1/Table 10.
- Show `branch_boundaries` being built and saved via `np.save(..., allow_pickle=True)` — explain why every ablation-slicing script downstream depends on this exact dict.
- Point to the assertion checks (`assert X_struct_raw.shape[0] == n`) — explain these catch a silent misalignment before it becomes a corrupted fused row.

**Self Check**
- Why is the structural scaler fit using the PRIMARY split's train indices even when a secondary 80/20 split is also being produced from the same data?
- What three numbers sum to 6,314, and where in this file does the actual concatenation happen?
- What would happen downstream if `branch_boundaries.npy` and `fused_features.npy` ever disagreed?

---

### Unit 7: Phase 4 — Ablation Training and Optuna Tuning
- **Files:** `train.py`
- **Time:** 35 minutes
- **Objective:** Be able to explain the difference between the fixed-param ablation configs and the Optuna-tuned final model without conflating them, and locate the `--skip-optuna` mkdir fix.

**Key Symbols**
| Name | Kind | Description |
| :--- | :--- | :--- |
| `ABLATION_CONFIGS` | dict | 7 entries, name -> list of branch names to include. |
| `FIXED_ABLATION_PARAMS` | dict | `n_estimators=200`, `max_depth=6`, `learning_rate=0.1` — used for ALL 7 configs. |
| `SKIP_OPTUNA_PARAMS` | dict | More conservative fallback than bare defaults; used only with `--skip-optuna`. |
| `slice_branches(...)` | function | Column-slices and concatenates requested branches. |
| `run_ablation(...)` | function | Trains and scores all 7 configs, returns a DataFrame. |
| `optuna_tune(...)` | function | TPE + MedianPruner(5,10), 5-fold stratified CV on train only. |
| `train_final_model(...)` | function | 90/10 carve from train, `early_stopping_rounds=10`. |

**Pointers**
- Show the unconditional `models_dir.mkdir(...)` / `results_dir.mkdir(...)` calls at the top of `run()` — explicitly say "this used to only happen inside the Optuna branch, and `--skip-optuna` runs crashed because of it."
- Show `FIXED_ABLATION_PARAMS` being reused identically across all 7 `slice_branches()` calls in `run_ablation()` — this is the fairness guarantee for the ablation comparison.
- Show `reg_alpha`/`reg_lambda` in the Optuna search space and explain these were a confirmed earlier gap.

**Self Check**
- Why must every one of the 7 ablation configs use identical hyperparameters?
- What's the exact CV setup inside `optuna_tune()`'s objective function, and why is it restricted to the training set only?
- Where does the FINAL model actually get its validation set from, and is it the same as any CV fold used during tuning?

---

### Unit 8: Phase 5 — Metrics, Curves, Importance, and the Baseline Comparison
- **Files:** `evaluate.py`
- **Time:** 30 minutes
- **Objective:** Explain the baseline-vs-fused fix (loading the saved tuned model, not retraining) and be able to walk through how per-sample explanations are computed.

**Key Symbols**
| Name | Kind | Description |
| :--- | :--- | :--- |
| `DEFAULT_THRESHOLD` | constant | 0.65. |
| `compute_metrics` / `threshold_sweep` / `find_optimal_threshold` | functions | Standard metrics, a 0.1-0.9 sweep, and best-F1 threshold search. |
| `feature_importance(...)` | function | Gain-based, maps f0/f1/... back to real names. |
| `per_sample_top3(...)` | function | Uses `pred_contribs=True` for individual-sample explanations. |
| `baseline_vs_fused(...)` | function | Config 1 (freshly trained) vs. the **SAVED** `xgboost_fused.pkl` — not retrained. |

**Pointers**
- Point directly at `fused_model = joblib.load(model_path)` inside `baseline_vs_fused()` — say out loud "this used to retrain a fresh untuned model here instead, which meant the headline comparison didn't reflect the actual best model."
- Show the `pred_contribs=True` call in `per_sample_top3()` and explain this is a genuinely different API call than `feature_importance()`'s `get_score(importance_type='gain')` — global vs. per-sample.

**Self Check**
- What's the exact bug this file's `baseline_vs_fused()` function was rebuilt to fix?
- What's the difference between what `feature_importance()` and `per_sample_top3()` each answer?
- At what threshold are the headline accuracy/precision/recall/F1 numbers reported, and where else in the codebase can you find the full trade-off curve around that point?

---

### Unit 9: Trust Infrastructure — Consistency Checks and Generalization Testing
- **Files:** `verify_artifacts.py` + `cross_dataset_eval.py`
- **Time:** 30 minutes
- **Objective:** Be able to run `verify_artifacts.py` live and narrate each of its 5 checks, and explain the chunked-prediction / OpenMP fix in `cross_dataset_eval.py`.

**Key Symbols**
| Name | Kind | Description |
| :--- | :--- | :--- |
| `check_tfidf_vocab` / `check_scaler` / `check_structural_scaler` | functions | Vocabulary and dimension sanity checks. |
| `check_variance_profile` | function | Scoped to the LEXICAL branch only, via `branch_boundaries` — fixed after an earlier false-positive on visual/structural columns. |
| `check_feature_name_agreement` / `check_model_dimension` | functions | Width agreement + mtime staleness warning. |
| `assemble_features(...)` | function | **THE** validated new-dataset feature-assembly path, reused by `recalibrate_threshold.py`. |
| `chunked_predict_proba(...)` | function | Scores in batches, sets `n_jobs=1` — avoids the macOS OpenMP segfault. |

**Pointers**
- Actually run `python3 scripts/verify_artifacts.py --features-dir ... --models-dir ...` live if possible and narrate each `[CHECK]`/`[PASS]`/`[FAIL]` line as it prints.
- Show `check_variance_profile()`'s `branch_boundaries` lookup and explain the false-positive it used to produce when scoped to the whole fused array instead of just lexical.
- Show `assemble_features()` being imported and called directly inside `recalibrate_threshold.py` — this is the concrete proof the bug-fix pattern is actually being reused, not just described.

**Self Check**
- Name all 5 checks `verify_artifacts.py` runs, in order.
- Why does checking file mtimes catch bugs that dimension checks alone would miss?
- Why does `chunked_predict_proba()` process data in batches instead of one large `predict_proba()` call?

---

### Unit 10: The Orchestrator
- **Files:** `run_pipeline.py`
- **Time:** 20 minutes
- **Objective:** Be able to trace, without looking, the full call order this script makes across every other core script, and explain the unconditional-path-definition fix.

**Key Symbols**
| Name | Kind | Description |
| :--- | :--- | :--- |
| `main()` | function | Parses args, defines ALL paths unconditionally, then calls each phase in order. |
| `--skip-*` flags | flags | Allow re-running only later phases (e.g., `--skip-qr`, `--skip-structural`, `--skip-visual`, `--skip-lexical`). |

**Pointers**
- Show that every path variable (`qr_dir`, `manifest_path`, `structural_csv`, `visual_npy`, `lexical_npy`, `tfidf_out`, `scaler_out`, `structural_scaler_out`, ...) is defined **BEFORE** any `if not args.skip_*` branch — explain this avoids a `NameError` when a flag skips the phase that would otherwise have defined it.
- Trace the call order out loud: `generate_qr.generate` -> `structural.run` -> `visual.run` -> (split-index computation for leakage-free lexical fit) -> `lexical.run` -> `fusion.run` -> `train.run` -> `evaluate.run`.

**Self Check**
- If `--skip-visual` is passed, what still needs `visual_npy` to already exist on disk, and where does that requirement come from?
- Why does `run_pipeline.py` compute train/test split indices itself, separately from `fusion.py`'s own split logic, before calling `lexical.run()`?

---

### Unit 11: The 9 Extension Scripts (Grouped)
- **Files:** `stats_significance.py`, `error_analysis.py`, `evasion_features.py`, `compare_backbones.py`, `dataset_size_experiment.py`, `profile_pipeline.py`, `robustness_tests.py`, `recalibrate_threshold.py`, `multi_split_experiment.py`
- **Time:** 40 minutes
- **Objective:** You don't need line-by-line recall here — you need to be able to say, for each script, "what real limitation or question does this answer" in one sentence, and know which two shared helpers (`load_fused_bundle`, `assemble_features`) every one of them is built on.

**Key Symbols**
| Script | Description |
| :--- | :--- |
| `stats_significance.py` | Bootstrap CI + permutation p-value on AUC differences between configs. |
| `error_analysis.py` | Joins predictions back to manifest attributes for FP/FN inspection. |
| `evasion_features.py` | Feature-space noise/targeted-perturbation sweep, measures evasion rate. |
| `compare_backbones.py` | MobileNetV2 vs `raw_pixels`, visual-only and fused AUC. |
| `dataset_size_experiment.py` | Tri-modal model's own data-scaling curve, fixed params. |
| `profile_pipeline.py` | Per-stage runtime + per-sample inference latency, peak RSS. |
| `robustness_tests.py` | Synthetic noise/blur/rotation/brightness/JPEG degradation testing. |
| `recalibrate_threshold.py` | Cost-sensitive threshold search; the rebuilt, previously-buggy script. |
| `multi_split_experiment.py` | AUC stability across split ratios and/or seeds. |

**Pointers**
- Open `recalibrate_threshold.py` specifically and show its two modes **never** reassemble features by hand — this is the single most likely "show me the fix" request from a panel that read your earlier documentation.
- Open `robustness_tests.py`'s `PERTURBATIONS` dict and show the 5 perturbation types with their level lists — connect each directly back to a sentence in your Scope & Limitations.
- Have one results CSV per script ready to show on screen (even from a small test run) rather than only describing them abstractly.

**Self Check**
- Which two functions do ALL nine of these scripts route through instead of reassembling features by hand?
- Which extension script most directly answers "is your improvement over the baseline statistically real?"
- Which extension script most directly answers "what happens with a blurry photo of a QR code?"

---

## 🎬 Live Demo Script

| Step | Time | Action / Script |
| :--- | :--- | :--- |
| **1** | 2 min | **Orientation:** Open the folder tree. Say the pipeline in one sentence: *load URLs -> generate QR -> extract structural/visual/lexical in parallel -> fuse -> train -> evaluate.* Point at `utils.py`'s `PATHS` dict as "this is why every script agrees on where things live." |
| **2** | 5 min | **Structural branch deep-dive:** Open `structural.py`. Show `FEATURE_NAMES`. Show `crop_to_content()` and `estimate_module_size()` together. Show the BCH self-test. If you can run it live: run `structural.run()` on a small batch and read the self-check log line out loud as it prints. |
| **3** | 4 min | **Visual + Lexical, faster pass:** Open `visual.py`, show the frozen-model setup and the checkpoint save/resume block. Open `lexical.py`, show the TF-IDF config line and the zero-padding fallback. |
| **4** | 3 min | **Fusion + the structural-scaler fix:** Open `fusion.py`, show the fit-on-train-only scaler block. This is a good place to mention it was a real gap found and fixed against the paper's own Table 10. |
| **5** | 4 min | **Training + evaluation:** Open `train.py`, show `ABLATION_CONFIGS` and the unconditional mkdir fix. Open `evaluate.py`, show `baseline_vs_fused()` loading the saved model, not retraining. |
| **6** | 3 min | **Trust infrastructure:** Run `verify_artifacts.py` live if at all possible — narrate the PASS/FAIL lines as they print. This is the single highest-value 30 seconds of a live demo: it visibly proves you built in a way to catch your own mistakes. |
| **7** | 3 min | **One extension script, your choice:** Pick whichever extension script produced your most interesting real result (likely `stats_significance.py` or `robustness_tests.py`) and show its actual output CSV, not just the code. |
| **Fallback** | N/A | **If live demo isn't possible or something breaks:** Have static outputs (CSVs, the `verify_artifacts.py` PASS log, a screenshot of the structural self-check line) saved and ready to show as images/screenshots — **never debug live in front of a panel.** If asked "can you show me X running," it's completely fine to say "let me show you the saved output from when I ran this" and pull up a file instead. |

---

## ⚡ Quick Reference Cheat Sheet

| File / Component | Description | Key Question to Anticipate |
| :--- | :--- | :--- |
| `utils.py` | Shared seed/paths/logger + `load_fused_bundle()` | Why does every script import this first? |
| `load_datasets.py` | PhiUSIIL/Kaitholikkal loading, label inversion | Why is PhiUSIIL's label inverted? |
| `generate_qr.py` | QR PNG + manifest generation | What's in the manifest that `structural.py` needs later? |
| `structural.py` | 24 protocol/statistical features | Walk me through the quiet-zone crop bug. |
| `visual.py` | MobileNetV2 1280-dim embeddings | Why checkpoint every 25 batches? |
| `lexical.py` | 9 scalars + 5000-dim TF-IDF | Why `char_wb(3,5)` and not the sklearn default? |
| `fusion.py` | Concatenation + structural scaling + splits | Where does the structural scaler get fit, and why there? |
| `train.py` | 7-config ablation + Optuna tuning | Why do all 7 ablation configs share hyperparameters? |
| `evaluate.py` | Metrics, curves, importance, baseline comparison | What bug did `baseline_vs_fused()` have before the fix? |
| `verify_artifacts.py` | 5 consistency checks on saved artifacts | Name the 5 checks. |
| `cross_dataset_eval.py` | Scores new data via saved artifacts, chunked | Why chunked prediction? |
| `run_pipeline.py` | Orchestrates every phase | Why must all paths be defined before any `--skip-*` branch? |
| 9 extension scripts | Significance, error/evasion analysis, robustness, etc. | Which two helper functions do they all share? |