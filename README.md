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
│   └── Evaluation/
│       ├── run_evaluation.py                # MediaPipe / YOLO pretrained / YOLO fine-tuned test-set eval (--model)
│       ├── train_yolo_scoliosis.py          # Fine-tune YOLO on scoliosis train/val
│       ├── train_yolo_combined.py           # Fine-tune on Roboflow + domain pseudo-labeled data
│       ├── compare_workflows.py             # 3-way comparison: MediaPipe vs YOLO pre vs fine-tuned
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
├── workflow_comparison.ipynb                # Visual 3-way comparison: MediaPipe vs YOLO
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

## Workflow Comparison Experiment

Compares MediaPipe vs YOLO (pretrained) vs YOLO (fine-tuned) on the scoliosis test set.

```bash
# Run in order:
python "src/Evaluation/run_evaluation.py" --model mediapipe
python "src/Evaluation/run_evaluation.py" --model yolo_pretrained
python "src/Evaluation/train_yolo_scoliosis.py"
python "src/Evaluation/run_evaluation.py" --model yolo_finetuned
python "src/Evaluation/compare_workflows.py"
jupyter lab workflow_comparison.ipynb
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

---

## External Datasets

| Dataset | Source | Keypoints | Classes | License |
|---|---|---|---|---|
| Scoliosis Keypoint Model v3 | [Roboflow Universe — cobacoba](https://universe.roboflow.com/cobacoba-vjxyv/scoliosis-keypoint-model-ahvid/dataset/3) | 12 (spine-specific) | normal / scoliosis | CC BY 4.0 |

Stored under `public_datasets/scoliosis/` (gitignored — re-extract from `.zip`). Used for YOLO fine-tuning and workflow comparison. Dataset-provided labels used for evaluation; not verified against X-ray.

---

## Ethical and Responsible Use

This system is a screening support tool, not a diagnostic device. Outputs must not be interpreted as medical diagnoses. All screening indicators should be reviewed by a qualified clinician before any referral decision is made. Data collection complies with school-based privacy requirements; subject identifiers are anonymised at the point of capture.