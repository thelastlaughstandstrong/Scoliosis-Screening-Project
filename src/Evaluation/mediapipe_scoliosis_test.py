# ============================================================
# Phase 1 Evaluation — MediaPipe on Scoliosis Test Set
# Runs MediaPipe pose on public_datasets/scoliosis/test/
# Outputs shared comparison schema CSV for MediaPipe vs YOLO
# ============================================================

import cv2
import numpy as np
import mediapipe as mp
import csv
import time
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_IMAGES  = PROJECT_ROOT / "public_datasets" / "scoliosis" / "test" / "images"
TEST_LABELS  = PROJECT_ROOT / "public_datasets" / "scoliosis" / "test" / "labels"
OUTPUT_CSV   = PROJECT_ROOT / "Data" / "Processed" / "comparison" / "mediapipe_scoliosis_test.csv"

CLASS_NAMES = {0: "normal", 1: "scoliosis"}
VISIBILITY_THRESHOLD = 0.5

# Shoulder + hip landmarks — used as spine-top / spine-base proxies
KEY_LANDMARKS = {
    "left_shoulder":  11,
    "right_shoulder": 12,
    "left_hip":       23,
    "right_hip":      24,
}


def load_ground_truth(label_path: Path):
    content = label_path.read_text().strip()
    if not content:
        return None, None
    class_id = int(content.split()[0])
    return class_id, CLASS_NAMES[class_id]


def trunk_angle_deg(lm, h, w):
    ls = (lm[11].x * w, lm[11].y * h)
    rs = (lm[12].x * w, lm[12].y * h)
    lh = (lm[23].x * w, lm[23].y * h)
    rh = (lm[24].x * w, lm[24].y * h)
    spine_top = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
    spine_bot = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)
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

    rows = []

    with mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.5,
    ) as pose:

        for img_path in image_files:
            label_path = TEST_LABELS / (img_path.stem + ".txt")
            gt_id, gt_class = (
                load_ground_truth(label_path)
                if label_path.exists()
                else (None, None)
            )

            # cv2.imread fails on Windows paths with non-ASCII chars; use imdecode instead
            img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                print(f"[WARN] Could not read {img_path.name}")
                continue

            h, w = img.shape[:2]

            t0 = time.perf_counter()
            result = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            processing_time = round(time.perf_counter() - t0, 4)

            pose_present = result.pose_landmarks is not None

            if pose_present:
                lm = result.pose_landmarks.landmark
                key_vis = [lm[i].visibility for i in KEY_LANDMARKS.values()]
                mean_confidence   = round(sum(key_vis) / len(key_vis), 4)
                missing_kp_count  = sum(1 for v in key_vis if v < VISIBILITY_THRESHOLD)
                valid_measurement = missing_kp_count == 0
                primary_angle     = round(trunk_angle_deg(lm, h, w), 3) if valid_measurement else None
            else:
                mean_confidence   = 0.0
                missing_kp_count  = len(KEY_LANDMARKS)
                valid_measurement = False
                primary_angle     = None

            retake_recommended = not valid_measurement

            rows.append({
                "capture_id":          img_path.stem,
                "workflow_type":       "mediapipe",
                "ground_truth_label":  gt_id,
                "ground_truth_class":  gt_class,
                "pose_present":        int(pose_present),
                "valid_measurement":   int(valid_measurement),
                "mean_confidence":     mean_confidence,
                "missing_keypoint_count": missing_kp_count,
                "primary_angle_deg":   primary_angle,
                "retake_recommended":  int(retake_recommended),
                "processing_time_s":   processing_time,
            })

            status = "OK  " if valid_measurement else "FAIL"
            print(
                f"[{status}] {img_path.name:55s} "
                f"gt={str(gt_class):10s} conf={mean_confidence:.2f}  "
                f"angle={str(primary_angle):>8}"
            )

    if not rows:
        print("[ERROR] No results produced.")
        return

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # ---------- Summary ----------
    total      = len(rows)
    valid      = sum(r["valid_measurement"] for r in rows)
    n_normal   = [r for r in rows if r["ground_truth_class"] == "normal"]
    n_scoli    = [r for r in rows if r["ground_truth_class"] == "scoliosis"]

    print(f"\n{'='*60}")
    print(f"  Total images processed : {total}")
    print(f"  Valid measurements     : {valid} / {total}  ({100*valid/total:.0f}%)")
    print(f"\n  --- Per-class detection (pose found + valid measurement) ---")
    print(f"  Normal    : {sum(r['valid_measurement'] for r in n_normal)} / {len(n_normal)} detected")
    print(f"  Scoliosis : {sum(r['valid_measurement'] for r in n_scoli)} / {len(n_scoli)} detected")
    print(f"\n  NOTE: Only {len(n_scoli)} scoliosis cases in test set.")
    print(f"  Use per-class recall, not accuracy, to evaluate screening reliability.")
    print(f"\n  Output saved to: {OUTPUT_CSV.name} (in Data/Processed/comparison/)")
    print(f"{'='*60}")


if __name__ == "__main__":
    run()