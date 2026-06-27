import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ============================================================
# Combined YOLO Fine-Tuning
# Merges two datasets:
#   1. Roboflow scoliosis dataset  — 12-point spine labels, clinical images
#   2. Domain pseudo-labeled data  — 12-point schema, intermediate kps invisible
#
# Why combine: Roboflow teaches spine keypoint geometry.
# Domain data teaches the model your camera/lighting/clothing conditions.
# ============================================================

import shutil
import yaml
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT   = Path(__file__).resolve().parents[2]
ROBOFLOW_DIR   = PROJECT_ROOT / "public_datasets" / "scoliosis"
DOMAIN_DIR     = PROJECT_ROOT / "public_datasets" / "domain_data"
COMBINED_DIR   = PROJECT_ROOT / "public_datasets" / "combined_data"
YAML_PATH      = PROJECT_ROOT / "public_datasets" / "combined_train.yaml"
MODEL_DIR      = PROJECT_ROOT / "models" / "yolo_scoliosis_combined"

# Start from existing fine-tuned weights rather than pretrained
BASE_WEIGHTS   = PROJECT_ROOT / "models" / "yolo_scoliosis" / "run" / "weights" / "best.pt"


def merge_datasets():
    """Copy images + labels from both datasets into one combined folder."""
    for split in ["train", "valid"]:
        for kind in ["images", "labels"]:
            (COMBINED_DIR / split / kind).mkdir(parents=True, exist_ok=True)

    counts = {"roboflow": 0, "domain": 0}

    # Roboflow train
    for kind in ["images", "labels"]:
        src = ROBOFLOW_DIR / "train" / kind
        if src.exists():
            for f in src.iterdir():
                shutil.copy2(f, COMBINED_DIR / "train" / kind / f"roboflow_{f.name}")
                counts["roboflow"] += (1 if kind == "images" else 0)

    # Roboflow valid
    for kind in ["images", "labels"]:
        src = ROBOFLOW_DIR / "valid" / kind
        if src.exists():
            for f in src.iterdir():
                shutil.copy2(f, COMBINED_DIR / "valid" / kind / f"roboflow_{f.name}")

    # Domain pseudo-labeled data (train only — no valid split)
    if DOMAIN_DIR.exists():
        for kind in ["images", "labels"]:
            src = DOMAIN_DIR / "train" / kind
            if src.exists():
                for f in src.iterdir():
                    shutil.copy2(f, COMBINED_DIR / "train" / kind / f"domain_{f.name}")
                    counts["domain"] += (1 if kind == "images" else 0)
    else:
        print(f"[INFO] No domain data found at {DOMAIN_DIR.name} — training on Roboflow only.")
        print(f"       Run extract_frames.py then pseudo_label_mediapipe.py to add domain data.")

    print(f"[INFO] Merged: {counts['roboflow']} Roboflow + {counts['domain']} domain images")
    return counts


def make_yaml():
    data = {
        "path":       str(COMBINED_DIR),
        "train":      "train/images",
        "val":        "valid/images",
        "kpt_shape":  [12, 3],
        "flip_idx":   list(range(12)),
        "nc":         2,
        "names":      {0: "normal", 1: "scoliosis"},
    }
    with open(YAML_PATH, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    print(f"[INFO] Combined yaml written: {YAML_PATH.name}")


def train():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    counts = merge_datasets()
    make_yaml()

    # Use existing fine-tuned weights if available, else fall back to pretrained
    weights = str(BASE_WEIGHTS) if BASE_WEIGHTS.exists() else "yolov8n-pose.pt"
    print(f"[INFO] Starting from: {Path(weights).name}")

    model = YOLO(weights)
    model.train(
        data=str(YAML_PATH),
        epochs=100,
        patience=20,
        imgsz=640,
        batch=8,
        project=str(MODEL_DIR),
        name="run",
        exist_ok=True,
        save=True,
        val=True,
        plots=True,
    )

    best = MODEL_DIR / "run" / "weights" / "best.pt"
    print(f"\n[DONE] Best combined model: {best}")
    print(f"       Update screen_subject.py WEIGHTS path to use this model.")


if __name__ == "__main__":
    train()