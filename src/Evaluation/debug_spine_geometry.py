# ============================================================
# Debug / Sanity Script — Spine Geometry Feature Extraction
# Runs the fine-tuned YOLO spine model on a handful of test
# images, prints the anatomy-aware geometry features, and
# optionally saves annotated images showing:
#   - kp[0] -> kp[8] reference line
#   - intermediate spine keypoints (kp[1..7])
#   - perpendicular deviation of each intermediate point
#
# This is a sanity/debug tool, not a formal evaluation script.
# Use run_evaluation.py + compare_workflows.py for the full
# test-set comparison.
#
# Usage:
#   python debug_spine_geometry.py            (first 5 test images, saves annotated images)
#   python debug_spine_geometry.py --n 10
#   python debug_spine_geometry.py --no-save
# ============================================================

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import argparse
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.keypoint_schema import (
    SPINE_BASE_LUMBAR, SPINE_TOP_CERVICAL, INTERMEDIATE_SPINE_IDXS,
    LEFT_SHOULDER, RIGHT_SHOULDER, DEFAULT_CONFIDENCE_THRESHOLD,
)
from src.FeatureExtraction.spine_geometry import extract_spine_geometry, empty_spine_geometry

TEST_IMAGES       = PROJECT_ROOT / "public_datasets" / "scoliosis" / "test" / "images"
FINETUNED_WEIGHTS = PROJECT_ROOT / "models" / "yolo_scoliosis" / "run" / "weights" / "best.pt"
OUTPUT_DIR        = PROJECT_ROOT / "Data" / "Processed" / "comparison" / "debug_spine_geometry"


def imdecode_unicode(img_path: Path):
    # cv2.imread fails on Windows paths with non-ASCII chars; use imdecode instead
    return cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)


def draw_annotated(img, kps, geometry, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD):
    out = img.copy()

    def visible(idx):
        return float(kps[idx, 2]) >= confidence_threshold

    def pt(idx):
        return int(kps[idx, 0]), int(kps[idx, 1])

    if visible(SPINE_BASE_LUMBAR) and visible(SPINE_TOP_CERVICAL):
        base, top = pt(SPINE_BASE_LUMBAR), pt(SPINE_TOP_CERVICAL)
        cv2.line(out, base, top, (0, 255, 255), 2)   # reference axis: yellow

        for idx in INTERMEDIATE_SPINE_IDXS:
            if not visible(idx):
                continue
            p = pt(idx)
            cv2.circle(out, p, 5, (255, 150, 0), -1)  # intermediate kp: orange

            # Perpendicular foot of p onto the base-top line, to draw the deviation segment.
            ax, ay = base
            bx, by = top
            px, py = p
            abx, aby = bx - ax, by - ay
            denom = abx * abx + aby * aby
            if denom > 1e-6:
                t = ((px - ax) * abx + (py - ay) * aby) / denom
                foot = (int(ax + t * abx), int(ay + t * aby))
                cv2.line(out, p, foot, (0, 0, 255), 1)  # lateral deviation: red

    if visible(SPINE_BASE_LUMBAR):
        cv2.circle(out, pt(SPINE_BASE_LUMBAR), 6, (0, 255, 0), -1)
    if visible(SPINE_TOP_CERVICAL):
        cv2.circle(out, pt(SPINE_TOP_CERVICAL), 6, (0, 255, 0), -1)
    if visible(LEFT_SHOULDER) and visible(RIGHT_SHOULDER):
        cv2.line(out, pt(LEFT_SHOULDER), pt(RIGHT_SHOULDER), (255, 0, 255), 2)

    label = "geom OK" if geometry["valid_spine_geometry"] else f"geom INVALID ({geometry['geometry_failure_reason']})"
    color = (0, 200, 0) if geometry["valid_spine_geometry"] else (0, 0, 255)
    cv2.putText(out, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return out


def run(n: int, save: bool):
    if not FINETUNED_WEIGHTS.exists():
        print(f"[ERROR] Fine-tuned weights not found at: {FINETUNED_WEIGHTS}")
        print("        Run train_yolo_scoliosis.py first.")
        return

    from ultralytics import YOLO
    model = YOLO(str(FINETUNED_WEIGHTS))

    image_files = sorted(list(TEST_IMAGES.glob("*.jpg")) + list(TEST_IMAGES.glob("*.png")))[:n]
    if not image_files:
        print(f"[ERROR] No images found in {TEST_IMAGES}")
        return

    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for img_path in image_files:
        img = imdecode_unicode(img_path)
        if img is None:
            print(f"[WARN] Could not read {img_path.name}")
            continue

        result = model(img, verbose=False)[0]
        pose_present = (
            result.boxes is not None and len(result.boxes) > 0
            and result.keypoints is not None and len(result.keypoints.data) > 0
        )

        print(f"\n{'='*60}\n{img_path.name}\n{'='*60}")

        kps = None
        if not pose_present:
            print("  No detection.")
            geometry = empty_spine_geometry("no_pose_detected")
        else:
            best_idx = int(result.boxes.conf.cpu().numpy().argmax())
            kps = result.keypoints.data[best_idx].cpu().numpy()
            geometry = extract_spine_geometry(kps)

        for k, v in geometry.items():
            print(f"  {k:<34}: {v}")

        if save and pose_present:
            annotated = draw_annotated(img, kps, geometry)
            out_path = OUTPUT_DIR / f"{img_path.stem}_geometry.jpg"
            cv2.imencode(".jpg", annotated)[1].tofile(str(out_path))
            print(f"  [SAVED] {out_path.relative_to(PROJECT_ROOT)}")

    if save:
        print(f"\n[DONE] Annotated images saved to: {OUTPUT_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sanity-check spine geometry feature extraction on a few test images.")
    parser.add_argument("--n", type=int, default=5, help="Number of test images to inspect (default: 5)")
    parser.add_argument("--no-save", action="store_true", help="Skip saving annotated images")
    args = parser.parse_args()
    run(args.n, save=not args.no_save)
