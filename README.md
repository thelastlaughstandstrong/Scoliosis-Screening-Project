# Kidzonia — AI-Assisted Scoliosis Screening System

A proof-of-concept RGB-D screening pipeline for school-based scoliosis screening support. The system captures posture during standardised tasks, extracts surface-based geometric features, and produces interpretable screening indicators to assist existing workflows.

All outputs are camera-derived proxy measurements. They do not represent clinical diagnosis, Cobb angle measurement, or medical-grade assessment. Clinician oversight is required for all referral decisions.

---

## Project Structure

```
Kidzonia Project/
├── src/
│   ├── Data Collection/
│   │   ├── quest_recording_auto.py          # Live RGB-D capture — MediaPipe keypoints
│   │   ├── quest_recording_yolo.py          # Live RGB-D capture — fine-tuned YOLO keypoints
│   │   ├── extract_frames.py                # Extract 1 fps frames from videos for pseudo-labeling
│   │   └── spine_angle_detection_image.py   # Static image capture
│   ├── Data Processing/
│   │   ├── aggregate_all_trials.py          # Frame-level CSV → trial-level dataset
│   │   ├── aggregate_all_subjects.py        # Legacy subject-level aggregation
│   │   ├── add_labels.py                    # Rule-based proxy screening flag (ATR >= 5 deg)
│   │   ├── pseudo_label_mediapipe.py        # Auto-label captured frames with MediaPipe → YOLO format
│   │   └── validate_trial_schema.py         # JSON schema validator
│   ├── Inference/
│   │   └── screen_subject.py               # Run fine-tuned YOLO on image or video → screening output
│   ├── FeatureExtraction/
│   │   └── spine_geometry.py                # Anatomy-aware geometry features + reliability score
│   └── Evaluation/
│       ├── run_evaluation.py                # MediaPipe / YOLO pretrained / YOLO fine-tuned test-set eval (--model, --weights, --tag)
│       ├── train_yolo_scoliosis.py          # Fine-tune YOLO (configurable epochs/imgsz/batch/seed, --augment-profile)
│       ├── train_yolo_combined.py           # Fine-tune on Roboflow + domain pseudo-labeled data
│       ├── dataset_eda.py                   # Read-only dataset EDA / data-quality report
│       ├── dataset_cleaning.py              # Review-first cleaning; exports Data/Processed/scoliosis_cleaned/
│       ├── split_quality_check.py           # Split sizes, class mix, leakage (incl. near-duplicate photos)
│       ├── create_grouped_split.py          # Photo-grouped, class-stratified train/valid/test re-split (no training)
│       ├── create_kfold_splits.py           # OPTIONAL stratified, photo-grouped 3-fold splits (no training)
│       ├── compare_workflows.py             # 3-way comparison: MediaPipe vs YOLO pre vs fine-tuned
│       ├── compare_original_vs_cleaned.py   # Compare any two evaluated fine-tuned models
│       ├── run_original_vs_cleaned_experiment.py  # Print/run the original-vs-cleaned train+eval+compare steps
│       ├── debug_spine_geometry.py          # Annotated overlays of predicted geometry + reliability
│       └── temporal_stability.py           # Frame-to-frame measurement variance on real recordings
├── legacy/
│   └── train_baseline_pending_clinical_labels.py  # LR classifier — valid when independent ATR labels arrive
├── schemas/
│   ├── frame_feature_schema_v1.json         # Per-frame feature contract
│   ├── trial_feature_schema_v1.json         # Per-trial aggregated feature contract
│   └── label_schema_v1.json                 # Proxy screening flag definition
├── Data/
│   ├── Raw/
│   │   ├── raw_data/                        # Live capture output (auto-created, gitignored)
│   │   └── spine_image_dataset/             # Static image capture output (auto-created, gitignored)
│   └── Processed/                           # Generated CSVs (auto-created, gitignored)
├── public_datasets/                         # Source archives only — extracted images are gitignored
├── models/                                  # Trained weights (gitignored)
├── docs/
│   ├── interim_report.md                    # Written project summary (non-diagnostic scope, results, limitations)
│   └── baseline_workflows_comparison.md     # MediaPipe vs pretrained YOLO vs fine-tuned: grouped-split and old-split records
├── workflow_comparison.ipynb                # HISTORICAL/DEBUG notebook: old-split results, affected by data leakage (not final)
├── grouped_workflow_comparison.ipynb        # FINAL results: model retrained + evaluated on the photo-grouped split (leakage-free)
├── example_trial_row.json                   # Example row for schema validation
└── environment.yml                          # Conda environment spec
```

---

## Environment Setup

Requires **Python 3.12** and Conda. See `environment.yml`.

```bash
conda activate env
```

Key dependencies: `openni`, `mediapipe`, `opencv`, `ultralytics`, `pandas`, `scikit-learn`, `numpy`, `scipy`.

The OpenNI driver requires a compatible RGB-D device (e.g. Kinect v1 or Azure Kinect with OpenNI2 wrapper). All scripts except data collection run without a camera.

---

## Pipeline

### Stage 1 — Data Collection (requires camera)

Run a guided screening session for one subject across three standardised postures. A **subject presence gate** blocks recording from starting until a valid pose has been detected for at least 1 second, guarding against false detections on empty scenes.

| Quest ID | Posture | Key Features Extracted |
|---|---|---|
| Q1_standing | Standing upright | Shoulder/hip symmetry, midline offset |
| Q2_forward_bend | Adam's forward bend | ATR proxy, trunk curvature, rib hump proxy |
| Q3_upward_stretch | Arms-up stretch | Trunk inclination, midline offset |

Two detector options — run the same session with both to compare outputs:

| Script | Detector | Extra columns |
|---|---|---|
| `quest_recording_auto.py` | MediaPipe (33 landmarks) | `trunk_curvature_proxy_2d_deg` |
| `quest_recording_yolo.py` | Fine-tuned YOLO 12-pt | `predicted_class`, `mean_kp_confidence`, `yolo_box_confidence` |

```bash
# MediaPipe version (no trained model required)
python "src/Data Collection/quest_recording_auto.py"

# YOLO version (requires trained weights in models/)
python "src/Data Collection/quest_recording_yolo.py"
```

Both scripts require the fine-tuned model to exist for the YOLO version. Run `train_yolo_scoliosis.py` first.

Output → `Data/Raw/raw_data/{subject_id}/{quest_id}/trial_{n}/`:
- `video.avi` — raw RGB video
- `features.csv` — frame-level features (one row per frame)
- `features.json` — same data in JSON
- `meta.json` — session metadata (includes `"detector": "yolo"` for YOLO runs)

---

### Stage 2 — Aggregation

Aggregates frame-level CSVs into one row per trial (median and 95th percentile).

```bash
python "src/Data Processing/aggregate_all_trials.py"
```

Output: `Data/Processed/aggregated_trials.csv`

---

### Stage 3 — Screening Flag

Applies a rule-based proxy screening flag: trials where `atr_proxy_p95_deg >= 5.0` receive `proxy_screening_flag = 1`.

```bash
python "src/Data Processing/add_labels.py"
```

Output: `Data/Processed/screened_trials.csv`

> **Important:** `proxy_screening_flag` is derived from the camera measurements themselves — it is a rule-based output, not a clinical ground truth. It should not be used as a training target without an independent clinical reference (scoliometer ATR, X-ray Cobb angle). See `legacy/train_baseline_pending_clinical_labels.py` for the classifier that becomes valid once clinical labels are available.

---

### Stage 4 — Inference on New Captures

Run the fine-tuned YOLO model on a single image or video to produce a structured screening output.

```bash
python "src/Inference/screen_subject.py" <image_or_video_path>
```

Output is a JSON object including `spine_angle_deg`, `proxy_screening_flag`, `mean_kp_confidence`, `retake_recommended`, and a placeholder `clinical_atr_deg` field for future clinical comparison.

---

### Stage 5 — Temporal Stability (no camera required)

Measures frame-to-frame variance of trunk angle over a real recording to assess measurement reliability.

```bash
python "src/Evaluation/temporal_stability.py" <features.csv | raw_data_root>
```

Reports: angle mean/std, frame-to-frame difference, rolling short-window jitter, pose coverage, and a stability verdict.

---

### Schema Validation (optional QA)

```bash
python "src/Data Processing/validate_trial_schema.py"
```

---

## Fine-Tuning and Evaluation Workflow

The written summary is [`docs/interim_report.md`](docs/interim_report.md); the charts and example images behind it are in [`workflow_comparison.ipynb`](workflow_comparison.ipynb), which is a **historical/debug notebook**: its results are on the old split and affected by data leakage, so they are not final performance claims, and it must not be re-run with grouped-split results. Final results are in [`grouped_workflow_comparison.ipynb`](grouped_workflow_comparison.ipynb) (a model retrained and evaluated on the grouped split). The plan and reasoning for each experiment is in `Data/Processed/evaluation/fine_tuning_plan.md` (generated outputs under `Data/Processed/` are gitignored).

```bash
# 1. Check the data and the splits (read-only)
python src/Evaluation/dataset_cleaning.py --dry-run
python src/Evaluation/split_quality_check.py

# 2. Train (each experiment in its own --project folder). Slow on CPU: ~1 hour each.
python src/Evaluation/train_yolo_scoliosis.py --data-yaml Data/Processed/scoliosis_cleaned/scoliosis_cleaned.yaml \
    --project models/yolo_scoliosis_cleaned_aug --augment-profile conservative

# 3. Evaluate on the fixed test set (fast), one tag per model
python src/Evaluation/run_evaluation.py --model yolo_finetuned \
    --weights models/yolo_scoliosis_cleaned_aug/run/weights/best.pt --tag cleaned_aug_trained

# 4. Compare two models (old-split results — historical/debug only)
python src/Evaluation/compare_original_vs_cleaned.py --baseline-csv Data/Processed/comparison/yolo_finetuned_scoliosis_test_cleaned_trained.csv \
    --candidate-csv Data/Processed/comparison/yolo_finetuned_scoliosis_test_cleaned_aug_trained.csv \
    --baseline-label "Cleaned" --candidate-label "Cleaned + conservative aug" --out-name cleaned_vs_cleaned_aug

# Re-split with no photo shared between train/valid/test (mixed-label photo groups excluded; prepares data only)
python src/Evaluation/create_grouped_split.py
python src/Evaluation/split_quality_check.py --dataset grouped

# Optional: stratified, photo-grouped 3-fold splits (prepares data only; training is up to you)
python src/Evaluation/create_kfold_splits.py --k 3 --seed 42
```

`--augment-profile` is `default_yolo` (Ultralytics defaults, the previous behaviour), `conservative` (small shift/scale/tilt only; no flip, mosaic, erasing or colour changes) or `none`. Epochs, image size, batch size, seed, run name and base weights are also command-line options (`--help`).

### Grouped split (leakage-free evaluation)

The original public split puts near-identical copies of the same photo in train, valid and test, so its valid/test numbers are optimistic. `create_grouped_split.py` builds a new split (`Data/Processed/scoliosis_grouped_split/`) where every photo is in exactly one split; mixed-label photo groups are excluded and listed in its `conflict_report.md`.

- **Old-model results are historical/debug only.** Models trained on the old split should not be interpreted as unbiased on the grouped test set. Retrain on the grouped train/valid split before reporting grouped-split performance.
- `run_evaluation.py` takes `--data-yaml` and `--split` (defaults: the original dataset, `test` — the previous behaviour). Non-default runs write `<workflow>_<dataset>_<split>[_<tag>].csv` plus a `.meta.json` recording the dataset, split and weights; the comparison scripts read it and warn about mixed test sets or old-split models.

```bash
# 1. Train on the grouped train/valid split (slow on CPU; nothing here runs automatically)
python src/Evaluation/train_yolo_scoliosis.py --data-yaml Data/Processed/scoliosis_grouped_split/scoliosis_grouped_split.yaml \
    --project models/yolo_scoliosis_grouped_conservative --augment-profile conservative

# 2. Evaluate on the grouped test split
python src/Evaluation/run_evaluation.py --model yolo_finetuned \
    --weights models/yolo_scoliosis_grouped_conservative/run/weights/best.pt \
    --data-yaml Data/Processed/scoliosis_grouped_split/scoliosis_grouped_split.yaml --split test --tag grouped_conservative

# 3. Compare two grouped-split models (both evaluated with the command above, different --tag)
python src/Evaluation/compare_original_vs_cleaned.py \
    --baseline-csv  Data/Processed/comparison/yolo_finetuned_scoliosis_grouped_split_test_grouped_default.csv \
    --candidate-csv Data/Processed/comparison/yolo_finetuned_scoliosis_grouped_split_test_grouped_conservative.csv \
    --baseline-label "Grouped default aug" --candidate-label "Grouped conservative aug" --out-name grouped_default_vs_conservative
```

**Baseline comparison (MediaPipe vs pretrained YOLO vs fine-tuned):**

```bash
python "src/Evaluation/run_evaluation.py" --model mediapipe
python "src/Evaluation/run_evaluation.py" --model yolo_pretrained
python "src/Evaluation/run_evaluation.py" --model yolo_finetuned
python "src/Evaluation/compare_workflows.py"
```

---

## Screening Flag vs Clinical Label

| | `proxy_screening_flag` | Independent clinical label |
|---|---|---|
| **Source** | Camera-derived ATR proxy >= 5 deg | Scoliometer, X-ray Cobb angle, clinician |
| **Independent?** | No — same pipeline that measures it | Yes |
| **Use as training target?** | No — circular | Yes — meaningful |
| **Use as screening output?** | Yes — as a proxy flag | Yes — as a reference |
| **Status** | Available now | Pending clinical data collection |

---

## Feature Schema Reference

### Frame-level features (`frame_feature_schema_v1.json`)

| Group | Field | Quest | Description |
|---|---|---|---|
| Identifiers | `subject_id`, `quest_id`, `trial_id` | All | Row identifiers |
| Timestamps | `timestamp_epoch`, `t_rel_s` | All | Capture timing |
| Pose quality | `pose_present`, `pose_confidence` | All | Detection validity |
| Standing symmetry | `delta_shoulder_height_px`, `delta_hip_height_px`, `standing_midline_x_offset_px`, `standing_symmetry_index` | Q1 | Left-right asymmetry proxies |
| Forward bend | `atr_proxy_deg`, `rib_hump_proxy_px`, `surface_asymmetry_score`, `atr_exceeds_5deg` | Q2 | ATR and rib hump proxies |
| Trunk geometry | `trunk_inclination_2d_deg`, `trunk_inclination_3d_deg`, `trunk_segment_angle_upper_2d_deg`, `trunk_segment_angle_lower_2d_deg`, `trunk_curvature_proxy_2d_deg` | All/Q2 | Trunk angle and curvature |

### Trial-level aggregated features (`trial_feature_schema_v1.json`)

| Feature | Source |
|---|---|
| `standing_symmetry_median` | Median of `standing_symmetry_index` (Q1) |
| `standing_midline_offset_p95_px` | 95th percentile of midline offset (Q1/Q3) |
| `trunk_curvature_p95_deg` | 95th percentile of trunk curvature proxy (Q2) |
| `atr_proxy_p95_deg` | 95th percentile of ATR proxy (Q2) — used for screening flag |
| `rib_hump_proxy_p95_px` | 95th percentile of rib hump proxy (Q2) |
| `trunk_inclination_2d_median_deg` | Median trunk inclination (all quests) |
| `trunk_inclination_2d_p95_deg` | 95th percentile trunk inclination (all quests) |
| `pose_coverage_ratio` | Fraction of frames with valid pose |

---

## Known Limitations

- **Landmark instability**: Pose landmarks may be unreliable under loose clothing.
- **Posture dependence**: Geometric measurements vary with posture compliance. No posture normalisation implemented yet.
- **Proxy measurements only**: All angular features are camera-derived proxies, not validated clinical measurements.
- **2D-only features**: Angular features are image-plane projections. Depth is captured but not fully integrated.
- **No independent clinical labels yet**: `proxy_screening_flag` cannot be used as a training target until scoliometer/clinical ATR labels are collected.
- **Public scoliosis dataset quality**: small (183 usable images from about 118 distinct photos), with near-duplicate photos across train/valid/test and a few conflicting labels — see `split_quality_check.py`. Test-set metrics are optimistic until the data is re-split by photo.

---

## External Datasets

| Dataset | Source | Keypoints | Classes | License |
|---|---|---|---|---|
| Scoliosis Keypoint Model v3 | [Roboflow Universe — cobacoba](https://universe.roboflow.com/cobacoba-vjxyv/scoliosis-keypoint-model-ahvid/dataset/3) | 12 (spine-specific) | normal / scoliosis | CC BY 4.0 |

Stored under `public_datasets/scoliosis/` (gitignored — re-extract from `.zip`). Used for YOLO fine-tuning and workflow comparison. Dataset-provided labels used for evaluation; not verified against X-ray.

---

## Ethical and Responsible Use

This system is a screening support tool, not a diagnostic device. Outputs must not be interpreted as medical diagnoses. All screening indicators should be reviewed by a qualified clinician before any referral decision is made. Data collection complies with school-based privacy requirements; subject identifiers are anonymised at the point of capture.