# ============================================================
# YOLO Fine-Tuning — Scoliosis Keypoint Model
# Trains yolov8n-pose on the scoliosis dataset (train/valid splits)
# Saves best weights to models/yolo_scoliosis/
# ============================================================

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import yaml
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR  = PROJECT_ROOT / "public_datasets" / "scoliosis"
MODEL_DIR    = PROJECT_ROOT / "models" / "yolo_scoliosis"
YAML_PATH    = PROJECT_ROOT / "public_datasets" / "scoliosis_train.yaml"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.keypoint_schema import NUM_KEYPOINTS


def make_data_yaml():
    """Create a data yaml with absolute paths — the packaged one uses broken relative paths."""
    data = {
        "path": str(DATASET_DIR),
        "train": "train/images",
        "val":   "valid/images",
        "test":  "test/images",
        "kpt_shape": [NUM_KEYPOINTS, 3],
        "flip_idx":  list(range(NUM_KEYPOINTS)),
        "nc": 2,
        "names": {0: "normal", 1: "scoliosis"},
    }
    with open(YAML_PATH, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    print(f"[INFO] Data yaml written to: {YAML_PATH.name}")
    return YAML_PATH


def train():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    yaml_path = make_data_yaml()

    model = YOLO("yolov8n-pose.pt")

    print("[INFO] Starting fine-tuning on scoliosis dataset...")
    print(f"       Train : {DATASET_DIR / 'train' / 'images'} (135 images)")
    print(f"       Val   : {DATASET_DIR / 'valid' / 'images'} (37 images)")
    print(f"       Output: {MODEL_DIR}")

    results = model.train(
        data=str(yaml_path),
        epochs=100,
        patience=20,        # early stopping
        imgsz=640,
        batch=8,
        project=str(MODEL_DIR),
        name="run",
        exist_ok=True,
        save=True,
        val=True,
        plots=True,
    )

    best_weights = MODEL_DIR / "run" / "weights" / "best.pt"
    print(f"\n[DONE] Best model saved to: {best_weights}")
    print(f"       Run the fine-tuned evaluation with: run_evaluation.py --model yolo_finetuned")
    return best_weights


if __name__ == "__main__":
    train()