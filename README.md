# Kidzonia — AI-Assisted Scoliosis Screening System

A proof-of-concept RGB-D screening pipeline for school-based scoliosis screening support. The system captures posture during standardised tasks, extracts surface-based geometric features, and produces interpretable screening indicators to assist existing workflows.

> **Non-diagnostic:** All outputs are posture-based screening proxies. They do not represent clinical diagnosis, Cobb angle measurement, or medical-grade assessment. Clinician oversight is required for all referral decisions.

---

## Project Structure

```
Kidzonia Project/
├── src/
│   ├── Data Collection/
│   │   ├── quest_recording_auto.py       # Live RGB-D capture (primary)
│   │   ├── quest_recording_manual.py     # Manual capture variant (legacy)
│   │   └── spine_angle_detection_image.py# Static image capture
│   ├── Data Processing/
│   │   ├── aggregate_all_trials.py       # Frame-level CSV → trial-level dataset
│   │   ├── aggregate_all_subjects.py     # Legacy subject-level aggregation
│   │   ├── add_labels.py                 # Adds ATR-based risk labels
│   │   └── validate_trial_schema.py      # JSON schema validator
│   └── ML Modelling/
│       └── train_baseline.py             # Logistic regression baseline
├── schemas/
│   ├── frame_feature_schema_v1.json      # Per-frame feature contract
│   ├── trial_feature_schema_v1.json      # Per-trial aggregated feature contract
│   └── label_schema_v1.json              # Risk label definition
├── raw_data/                             # Live capture output (auto-created)
├── processed_data/                       # Generated trial datasets and CSV artifacts
├── external_datasets/                    # Sourced online datasets and external resources
├── models/                               # Trained model files (auto-created)
├── example_trial_row.json                # Example row for schema validation
└── environment.yml                       # Conda environment spec
```

---

## Environment Setup

Requires **Python 3.12** and Conda.

```bash
conda env create -f environment.yml
conda activate kidzonia
```

Key dependencies: `openni`, `mediapipe`, `opencv`, `pandas`, `scikit-learn`, `numpy`, `scipy`.

The OpenNI driver requires a compatible RGB-D device (e.g. Kinect v1 or Azure Kinect with OpenNI2 wrapper). Data processing scripts run without any device attached.

---

## Pipeline

### Stage 1 — Data Collection (requires camera)

Run a guided screening session for one subject. The system captures three standardised postures:

| Quest ID | Posture | Key Features Extracted |
|---|---|---|
| Q1_standing | Standing upright | Shoulder/hip symmetry, midline offset |
| Q2_forward_bend | Adam's forward bend | ATR proxy, trunk curvature, rib hump proxy |
| Q3_upward_stretch | Arms-up stretch | Trunk inclination, midline offset |

```bash
python "src/Data Collection/quest_recording_auto.py"
```

Output is saved to `raw_data/{subject_id}/{quest_id}/trial_{n}/`:
- `video.avi` — raw RGB video
- `features.csv` — frame-level features (one row per frame)
- `features.json` — same data in JSON format
- `meta.json` — session metadata

---

### Stage 2 — Aggregation (no camera required)

Aggregates frame-level CSVs into one row per trial using robust statistics (median and 95th percentile).

```bash
python "src/Data Processing/aggregate_all_trials.py" raw_data/ processed_data/trial_dataset.csv
```

Output: `processed_data/trial_dataset.csv` — one row per trial across all subjects and quests.

---

### Stage 3 — Labeling (no camera required)

Applies an ATR-based risk label to each trial row. Trials with `atr_proxy_p95_deg >= 5.0` are flagged as `risk_label = 1`.

```bash
python "src/Data Processing/add_labels.py" processed_data/trial_dataset.csv processed_data/trial_dataset_labeled.csv
```

---

### Stage 4 — Model Training (no camera required)

Trains a logistic regression baseline classifier with feature scaling and cross-validation.

```bash
python "src/ML Modelling/train_baseline.py" processed_data/trial_dataset_labeled.csv models/baseline_logreg.pkl
```

Prints cross-validated ROC-AUC, test classification report, and saves the fitted pipeline (scaler + classifier) to `models/`.

---

### Schema Validation (optional QA)

Validates a trial-level JSON row against the schema contract.

```bash
python "src/Data Processing/validate_trial_schema.py"
```

Uses `example_trial_row.json` at the project root as the test input. Replace or extend this file to validate real trial outputs.

---

## Testing the Pipeline with an Online Dataset

The data processing stages (Steps 2–4) only read CSV files and run without any camera. To test with downloaded data you need a bridge script that converts images into the frame-level CSV schema.

**What to look for in a dataset:**

| Quest | Required posture | Typical availability |
|---|---|---|
| Q1 | Posterior or frontal standing view | Common in general pose datasets |
| Q2 | Adam's forward bend test | Rare; some physio/yoga datasets approximate it |
| Q3 | Arms-raised upright | Uncommon |

**Adapter approach (to be implemented):**

1. Place downloaded images under a folder organised by subject and posture type.
2. Run MediaPipe pose estimation on each image (no OpenNI required).
3. Compute the same features as `quest_recording_auto.py` using the landmark coordinates.
4. Write one `features.csv` per trial into `raw_data/{subject_id}/{quest_id}/trial_001/`.
5. Run aggregation, labeling, and training as normal from Stage 2 onward.

**Limitations with image-only datasets:**
- Depth-dependent fields (`rib_hump_proxy_px`, `trunk_inclination_3d_deg`) will be `null` — this is handled gracefully by the pipeline.
- Without Q2 data, `atr_proxy_p95_deg` stays `null` and all labels will be `risk_label = 0`. The classifier cannot be meaningfully trained.
- Clinical labels (e.g. Cobb angles) from the dataset must be manually mapped to `risk_label` instead of using the ATR threshold rule.

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

Aggregated from frame-level using median and 95th percentile. One row per trial per quest per subject.

| Feature | Source |
|---|---|
| `standing_symmetry_median` | Median of `standing_symmetry_index` (Q1) |
| `standing_midline_offset_p95_px` | 95th percentile of midline offset (Q1/Q3) |
| `trunk_curvature_p95_deg` | 95th percentile of trunk curvature proxy (Q2) |
| `atr_proxy_p95_deg` | 95th percentile of ATR proxy (Q2) — used for labeling |
| `rib_hump_proxy_p95_px` | 95th percentile of rib hump proxy (Q2) |
| `trunk_inclination_2d_median_deg` | Median trunk inclination (all quests) |
| `trunk_inclination_2d_p95_deg` | 95th percentile trunk inclination (all quests) |
| `pose_coverage_ratio` | Fraction of frames with valid pose |

---

## Known Limitations

- **Landmark instability**: Pose landmarks may be unreliable under loose clothing. No clothing-correction or confidence filtering is currently applied beyond `pose_present`.
- **Posture dependence**: Geometric measurements vary with posture compliance. No posture normalisation step is implemented yet.
- **False detections**: No subject-presence validation before feature extraction.
- **2D-only features**: All angular features are image-plane projections. Depth is captured but not fully integrated into calibrated 3D measurements.
- **No inference script**: The trained model cannot yet be applied to new recordings without additional code.

---

## Ethical and Responsible Use

This system is a screening support tool, not a diagnostic device. Outputs must not be interpreted as medical diagnoses. All screening indicators should be reviewed by a qualified clinician before any referral decision is made. Data collection complies with school-based privacy requirements; subject identifiers are anonymised at the point of capture.
