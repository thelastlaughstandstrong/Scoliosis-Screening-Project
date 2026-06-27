# ============================================================
# Phase 2 Evaluation — Pretrained YOLOv8-pose on Scoliosis Test Set
# Runs yolov8n-pose on public_datasets/scoliosis/test/
# Outputs same comparison schema as mediapipe_scoliosis_test.py
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_IMAGES  = PROJECT_ROOT / "public_datasets" / "scoliosis" / "test" / "images"
TEST_LABELS  = PROJECT_ROOT / "public_datasets" / "scoliosis" / "test" / "labels"
OUTPUT_CSV   = PROJECT_ROOT / "Data" / "Processed" / "comparison" / "yolo_scoliosis_test.csv"
MODEL_NAME   = "yolov8n-pose.pt"   # downloads automatically on first run

CLASS_NAMES = {0: "normal", 1: "scoliosis"}
VISIBILITY_THRESHOLD = 0.5

# COCO keypoint indices used for trunk angle — mirrors MediaPipe landmarks
KEY_IDX = {
    "left_shoulder":  5,
    "right_shoulder": 6,
    "left_hip":       11,
    "right_hip":      12,
}


def load_ground_truth(label_path: Path):
    content = label_path.read_text().strip()
    if not content:
        return None, None
    class_id = int(content.split()[0])
    return class_id, CLASS_NAMES[class_id]


def trunk_angle_deg(kps):
    """kps: array [17, 3] of (x, y, conf) in pixel coords."""
    ls = kps[5, :2]
    rs = kps[6, :2]
    lh = kps[11, :2]
    rh = kps[12, :2]
    spine_top = (ls + rs) / 2
    spine_bot = (lh + rh) / 2
    dx = spine_bot[0] - spine_top[0]
    dy = spine_bot[1] - spine_top[1]
    return math.degrees(math.atan2(dx, dy))


def run():
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        list(TEST_IMAGES.glob("*.jpg")) + list(TEST_IMAGES.glob("*.png"))
    )
    if not image_files:
        print(f"[ERROR] No images found in {TEST_IMAGES}")
        return

    model = YOLO(MODEL_NAME)
    rows = []

    for img_path in image_files:
        label_path = TEST_LABELS / (img_path.stem + ".txt")
        gt_id, gt_class = (
            load_ground_truth(label_path)
            if label_path.exists()
            else (None, None)
        )

        # Use imdecode to handle Unicode paths on Windows
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
            # Take the highest-confidence detection
            box_confs = result.boxes.conf.cpu().numpy()
            best_idx  = int(box_confs.argmax())
            det_conf  = float(box_confs[best_idx])
            kps       = result.keypoints.data[best_idx].cpu().numpy()  # [17, 3]

            key_confs        = [float(kps[i, 2]) for i in KEY_IDX.values()]
            mean_confidence  = round(sum(key_confs) / len(key_confs), 4)
            missing_kp_count = sum(1 for c in key_confs if c < VISIBILITY_THRESHOLD)
            valid_measurement = missing_kp_count == 0
            primary_angle    = round(trunk_angle_deg(kps), 3) if valid_measurement else None
        else:
            det_conf          = 0.0
            mean_confidence   = 0.0
            missing_kp_count  = len(KEY_IDX)
            valid_measurement = False
            primary_angle     = None

        retake_recommended = not valid_measurement

        rows.append({
            "capture_id":             img_path.stem,
            "workflow_type":          "yolo_pretrained",
            "ground_truth_label":     gt_id,
            "ground_truth_class":     gt_class,
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
        print(
            f"[{status}] {img_path.name:55s} "
            f"gt={str(gt_class):10s} conf={mean_confidence:.2f}  "
            f"angle={str(primary_angle):>8}  dets={num_detections}"
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

    print(f"\n{'='*60}")
    print(f"  Total images processed : {total}")
    print(f"  Valid measurements     : {valid} / {total}  ({100*valid/total:.0f}%)")
    print(f"\n  --- Per-class detection ---")
    print(f"  Normal    : {sum(r['valid_measurement'] for r in n_normal)} / {len(n_normal)} detected")
    print(f"  Scoliosis : {sum(r['valid_measurement'] for r in n_scoli)} / {len(n_scoli)} detected")
    print(f"\n  NOTE: Only {len(n_scoli)} scoliosis cases in test set.")
    print(f"  Use per-class recall, not accuracy, to evaluate screening reliability.")
    print(f"\n  Output saved to: {OUTPUT_CSV.name} (in Data/Processed/comparison/)")
    print(f"{'='*60}")


if __name__ == "__main__":
    run()