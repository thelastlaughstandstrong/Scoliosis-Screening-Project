# ============================================================
# MediaPipe Pseudo-Labeler
# Runs MediaPipe on captured images and outputs YOLO pose
# labels using the 12-point spine schema.
#
# KEYPOINT MAPPING (12-point schema, matching Roboflow dataset):
#   kp[0]  lumbar base    → hip midpoint (lm[23]+lm[24])/2   vis=1
#   kp[1-7] intermediate  → not labeled                       vis=0
#   kp[8]  cervical top   → shoulder midpoint (lm[11]+lm[12])/2  vis=1
#   kp[9]  left shoulder  → lm[11]                            vis=2
#   kp[10] right shoulder → lm[12]                            vis=2
#   kp[11] head           → nose lm[0]                        vis=1
#
# Intermediate spine points (kp[1-7]) are marked invisible.
# YOLO ignores them in pose loss — no noise introduced.
# Class is set to 0 (normal) — no clinical label available.
# ============================================================

import cv2
import mediapipe as mp
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT   = Path(__file__).resolve().parents[2]
DEFAULT_INPUT  = PROJECT_ROOT / "Data" / "Raw" / "pseudo_label_candidates"
DEFAULT_OUTPUT = PROJECT_ROOT / "public_datasets" / "domain_data"

VISIBILITY_THRESHOLD = 0.5
KPT_SHAPE = 12
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def mediapipe_to_yolo_label(lm, h: int, w: int) -> list[float] | None:
    """Convert MediaPipe landmarks to 12-point YOLO pose label (normalised)."""

    def lm_xy(idx):
        return lm[idx].x, lm[idx].y  # already normalised [0,1]

    def lm_vis(idx):
        return lm[idx].visibility

    # Require key landmarks to be visible
    key_ids = [11, 12, 23, 24]
    if any(lm[i].visibility < VISIBILITY_THRESHOLD for i in key_ids):
        return None

    ls = np.array(lm_xy(11))
    rs = np.array(lm_xy(12))
    lh = np.array(lm_xy(23))
    rh = np.array(lm_xy(24))

    shoulder_mid = (ls + rs) / 2
    hip_mid      = (lh + rh) / 2
    nose         = np.array(lm_xy(0))

    # Build bounding box from all key landmarks (with padding)
    all_pts = np.array([ls, rs, lh, rh, nose])
    x_min, y_min = all_pts.min(axis=0) - 0.05
    x_max, y_max = all_pts.max(axis=0) + 0.05
    x_min, y_min = max(0.0, x_min), max(0.0, y_min)
    x_max, y_max = min(1.0, x_max), min(1.0, y_max)

    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    bw = x_max - x_min
    bh = y_max - y_min

    # 12 keypoints
    kps = []
    def add_kp(x, y, vis):
        kps.extend([round(float(x), 6), round(float(y), 6), int(vis)])

    add_kp(*hip_mid,      1)                     # kp[0]  lumbar base
    for _ in range(7):
        add_kp(0.0, 0.0, 0)                      # kp[1-7] intermediate — invisible
    add_kp(*shoulder_mid, 1)                     # kp[8]  cervical top
    add_kp(ls[0], ls[1],  2)                     # kp[9]  left shoulder
    add_kp(rs[0], rs[1],  2)                     # kp[10] right shoulder
    add_kp(*nose,         1)                     # kp[11] head

    row = [0,                                    # class 0 (normal — no clinical label)
           round(cx, 6), round(cy, 6),
           round(bw, 6), round(bh, 6)]
    row.extend(kps)
    return row


def run(input_dir: Path = DEFAULT_INPUT, output_dir: Path = DEFAULT_OUTPUT):
    img_dir   = output_dir / "train" / "images"
    label_dir = output_dir / "train" / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    images = [f for f in sorted(input_dir.iterdir()) if f.suffix.lower() in IMAGE_EXTS]
    if not images:
        print(f"[ERROR] No images found in {input_dir}")
        return

    mp_pose = mp.solutions.pose
    labeled = 0
    skipped = 0

    with mp_pose.Pose(static_image_mode=True, model_complexity=1,
                      min_detection_confidence=0.5) as pose:

        for img_path in images:
            img = cv2.imdecode(
                np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if img is None:
                skipped += 1
                continue

            h, w = img.shape[:2]
            result = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            if not result.pose_landmarks:
                skipped += 1
                continue

            row = mediapipe_to_yolo_label(result.pose_landmarks.landmark, h, w)
            if row is None:
                skipped += 1
                continue

            # Save image
            out_img = img_dir / img_path.name
            cv2.imencode(".jpg", img)[1].tofile(str(out_img))

            # Save YOLO label
            label_line = " ".join(map(str, row))
            (label_dir / (img_path.stem + ".txt")).write_text(label_line)

            labeled += 1

    print(f"\n[DONE] Labeled: {labeled}  |  Skipped (no pose): {skipped}")
    print(f"       Output: {output_dir}/train/")
    print(f"       Run train_yolo_combined.py next.")


if __name__ == "__main__":
    input_dir  = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INPUT
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUTPUT
    run(input_dir, output_dir)