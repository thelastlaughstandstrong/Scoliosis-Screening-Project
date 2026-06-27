# ============================================================
# Phase 2b Evaluation — Fine-tuned YOLO on Scoliosis Test Set
# Uses spine-specific keypoints (kp[0]=lumbar base, kp[8]=cervical top)
# Also extracts predicted class (normal/scoliosis) from bounding box
# Run train_yolo_scoliosis.py first to produce the model weights
# ============================================================

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
import csv
import time
import math
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT  = Path(__file__).resolve().parents[2]
TEST_IMAGES   = PROJECT_ROOT / "public_datasets" / "scoliosis" / "test" / "images"
TEST_LABELS   = PROJECT_ROOT / "public_datasets" / "scoliosis" / "test" / "labels"
WEIGHTS_PATH  = PROJECT_ROOT / "models" / "yolo_scoliosis" / "run" / "weights" / "best.pt"
OUTPUT_CSV    = PROJECT_ROOT / "Data" / "Processed" / "comparison" / "yolo_finetuned_scoliosis_test.csv"

CLASS_NAMES = {0: "normal", 1: "scoliosis"}
VISIBILITY_THRESHOLD = 0.5

# Spine keypoint indices in the 12-point schema:
#   kp[0]  = lumbar base (bottom of spine)
#   kp[1..7] = intermediate vertebral landmarks
#   kp[8]  = cervical top (top of spine)
#   kp[9]  = left shoulder
#   kp[10] = right shoulder
#   kp[11] = head/neck
SPINE_BOTTOM_IDX = 0
SPINE_TOP_IDX    = 8
KEY_IDX = list(range(9))   # spine points only (0-8) for quality metrics


def load_ground_truth(label_path: Path):
    content = label_path.read_text().strip()
    if not content:
        return None, None
    class_id = int(content.split()[0])
    return class_id, CLASS_NAMES[class_id]


def spine_angle_deg(kps):
    """Angle of the spine line from lumbar base (kp[0]) to cervical top (kp[8])."""
    bottom = kps[SPINE_BOTTOM_IDX, :2]
    top    = kps[SPINE_TOP_IDX,    :2]
    dx = top[0] - bottom[0]
    dy = bottom[1] - top[1]   # flipped: image y increases downward
    return math.degrees(math.atan2(dx, dy))


def run():
    if not WEIGHTS_PATH.exists():
        print(f"[ERROR] Fine-tuned weights not found at: {WEIGHTS_PATH}")
        print("        Run train_yolo_scoliosis.py first.")
        return

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        list(TEST_IMAGES.glob("*.jpg")) + list(TEST_IMAGES.glob("*.png"))
    )
    if not image_files:
        print(f"[ERROR] No images found in {TEST_IMAGES}")
        return

    model = YOLO(str(WEIGHTS_PATH))
    rows = []

    for img_path in image_files:
        label_path = TEST_LABELS / (img_path.stem + ".txt")
        gt_id, gt_class = (
            load_ground_truth(label_path)
            if label_path.exists()
            else (None, None)
        )

        img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] Could not read {img_path.name}")
            continue

        t0 = time.perf_counter()
        results = model(img, verbose=False)
        processing_time = round(time.perf_counter() - t0, 4)

        result = results[0]
        num_detections = len(result.boxes) if result.boxes is not None else 0
        pose_present = (
            num_detections > 0
            and result.keypoints is not None
            and len(result.keypoints.data) > 0
        )

        if pose_present:
            box_confs  = result.boxes.conf.cpu().numpy()
            best_idx   = int(box_confs.argmax())
            det_conf   = float(box_confs[best_idx])
            pred_class_id = int(result.boxes.cls[best_idx].cpu().item())
            pred_class    = CLASS_NAMES.get(pred_class_id, "unknown")
            kps = result.keypoints.data[best_idx].cpu().numpy()  # [12, 3]

            spine_confs      = [float(kps[i, 2]) for i in KEY_IDX]
            mean_confidence  = round(sum(spine_confs) / len(spine_confs), 4)
            missing_kp_count = sum(1 for c in spine_confs if c < VISIBILITY_THRESHOLD)
            valid_measurement = (
                missing_kp_count == 0
                and float(kps[SPINE_BOTTOM_IDX, 2]) >= VISIBILITY_THRESHOLD
                and float(kps[SPINE_TOP_IDX, 2])    >= VISIBILITY_THRESHOLD
            )
            primary_angle = round(spine_angle_deg(kps), 3) if valid_measurement else None
            classification_correct = int(pred_class_id == gt_id) if gt_id is not None else None
        else:
            det_conf               = 0.0
            pred_class             = None
            mean_confidence        = 0.0
            missing_kp_count       = len(KEY_IDX)
            valid_measurement      = False
            primary_angle          = None
            classification_correct = None

        retake_recommended = not valid_measurement

        rows.append({
            "capture_id":             img_path.stem,
            "workflow_type":          "yolo_finetuned",
            "ground_truth_label":     gt_id,
            "ground_truth_class":     gt_class,
            "predicted_class":        pred_class,
            "classification_correct": classification_correct,
            "pose_present":           int(pose_present),
            "valid_measurement":      int(valid_measurement),
            "mean_confidence":        mean_confidence,
            "missing_keypoint_count": missing_kp_count,
            "primary_angle_deg":      primary_angle,
            "retake_recommended":     int(retake_recommended),
            "processing_time_s":      processing_time,
            "detection_confidence":   round(det_conf, 4),
            "num_detections":         num_detections,
        })

        status = "OK  " if valid_measurement else "FAIL"
        correct_str = ("Y" if classification_correct else "N") if classification_correct is not None else "-"
        print(
            f"[{status}] {img_path.name:55s} "
            f"gt={str(gt_class):10s} pred={str(pred_class):10s} cls_ok={correct_str}  "
            f"conf={mean_confidence:.2f}  angle={str(primary_angle):>8}"
        )

    if not rows:
        print("[ERROR] No results produced.")
        return

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # ---------- Summary ----------
    total    = len(rows)
    valid    = sum(r["valid_measurement"] for r in rows)
    n_normal = [r for r in rows if r["ground_truth_class"] == "normal"]
    n_scoli  = [r for r in rows if r["ground_truth_class"] == "scoliosis"]
    cls_rows = [r for r in rows if r["classification_correct"] is not None]
    cls_acc  = sum(r["classification_correct"] for r in cls_rows) / len(cls_rows) if cls_rows else 0

    # Scoliosis-specific: of ground-truth scoliosis, how many predicted scoliosis?
    scoli_detected = [r for r in n_scoli if r["predicted_class"] == "scoliosis"]

    print(f"\n{'='*60}")
    print(f"  Total images processed : {total}")
    print(f"  Valid measurements     : {valid} / {total}  ({100*valid/total:.0f}%)")
    print(f"\n  --- Pose detection by class ---")
    print(f"  Normal    : {sum(r['valid_measurement'] for r in n_normal)} / {len(n_normal)} valid")
    print(f"  Scoliosis : {sum(r['valid_measurement'] for r in n_scoli)} / {len(n_scoli)} valid")
    print(f"\n  --- Classification (predicted class vs ground truth) ---")
    print(f"  Overall accuracy  : {cls_acc:.0%}  ({sum(r['classification_correct'] for r in cls_rows if r['classification_correct'])}/{len(cls_rows)})")
    print(f"  Scoliosis recall  : {len(scoli_detected)}/{len(n_scoli)} correctly predicted as scoliosis")
    print(f"\n  NOTE: Spine angle uses kp[0] (lumbar) to kp[8] (cervical) — direct spine axis.")
    print(f"  Output saved to: {OUTPUT_CSV.name} (in Data/Processed/comparison/)")
    print(f"{'='*60}")


if __name__ == "__main__":
    run()