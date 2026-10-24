# ============================================================
# Debug / Sanity Script — Spine Geometry Feature Extraction
# Runs the fine-tuned YOLO spine model on a handful of test
# images, prints the anatomy-aware geometry features, and
# optionally saves annotated images showing:
#   - kp[0] -> kp[8] reference line (spine axis)
#   - intermediate spine keypoints (kp[1..7])
#   - perpendicular lateral-deviation line for each intermediate point
#   - shoulder line (kp[9]-kp[10])
#   - reliability_level + anatomy_reliability_score (colour-coded)
#   - retake_or_review_recommended
#   - anatomy_warning_flags / geometry_failure_reason
#
# This is a sanity/debug tool, not a formal evaluation script.
# Use run_evaluation.py + compare_workflows.py for the full
# test-set comparison. Outputs are screening-support indicators
# only — not a diagnosis.
#
# Usage:
#   python debug_spine_geometry.py            (first 5 test images, saves annotated images)
#   python debug_spine_geometry.py --n 10
#   python debug_spine_geometry.py --no-save
#   python debug_spine_geometry.py --data-yaml <data.yaml> --split test    (default: original public dataset, test)
#   python debug_spine_geometry.py --weights models/yolo_scoliosis_cleaned/run/weights/best.pt
#       --out-dir Data/Processed/geometry_debug/cleaned_trained --stems <test_image_stem> [<stem> ...]
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
from src.Evaluation.dataset_paths import resolve_split

FINETUNED_WEIGHTS = PROJECT_ROOT / "models" / "yolo_scoliosis" / "run" / "weights" / "best.pt"
OUTPUT_DIR        = PROJECT_ROOT / "Data" / "Processed" / "geometry_debug"


def imdecode_unicode(img_path: Path):
    # cv2.imread fails on Windows paths with non-ASCII chars; use imdecode instead
    return cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)


RELIABILITY_COLORS = {
    "high":    (0, 200, 0),
    "medium":  (0, 200, 255),
    "low":     (0, 120, 255),
    "invalid": (0, 0, 255),
}


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

    level = geometry.get("reliability_level", "invalid")
    color = RELIABILITY_COLORS.get(level, (0, 0, 255))

    label = "geom OK" if geometry["valid_spine_geometry"] else f"geom INVALID ({geometry['geometry_failure_reason']})"
    cv2.putText(out, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.putText(out, f"reliability: {level}  score={geometry.get('anatomy_reliability_score')}",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    retake_txt = "retake/review: YES" if geometry.get("retake_or_review_recommended") else "retake/review: no"
    cv2.putText(out, retake_txt, (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    flags = geometry.get("anatomy_warning_flags") or ""
    if flags:
        y = 94
        for flag in flags.split("; "):
            cv2.putText(out, f"! {flag}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)
            y += 18

    return out


def run(n: int, save: bool, weights: Path = None, out_dir: Path = None, stems=None, data_yaml=None, split: str = "test"):
    weights = Path(weights).resolve() if weights else FINETUNED_WEIGHTS
    output_dir = Path(out_dir).resolve() if out_dir else OUTPUT_DIR

    if not weights.exists():
        print(f"[ERROR] Fine-tuned weights not found at: {weights}")
        print("        Run train_yolo_scoliosis.py first.")
        return

    from ultralytics import YOLO
    model = YOLO(str(weights))

    try:
        images_dir = resolve_split(data_yaml, split)["images_dir"]
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"[ERROR] {e}")
        return
    image_files = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    if stems:
        image_files = [p for p in image_files if p.stem in set(stems)]
    else:
        image_files = image_files[:n]
    if not image_files:
        print(f"[ERROR] No images found in {images_dir}")
        return

    if save:
        output_dir.mkdir(parents=True, exist_ok=True)

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
            out_path = output_dir / f"{img_path.stem}_geometry.jpg"
            cv2.imencode(".jpg", annotated)[1].tofile(str(out_path))
            print(f"  [SAVED] {out_path.relative_to(PROJECT_ROOT)}")

    if save:
        print(f"\n[DONE] Annotated images saved to: {output_dir.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sanity-check spine geometry feature extraction on a few test images.")
    parser.add_argument("--n", type=int, default=5, help="Number of test images to inspect (default: 5)")
    parser.add_argument("--no-save", action="store_true", help="Skip saving annotated images")
    parser.add_argument("--weights", type=Path, default=None,
                         help="Fine-tuned weights to use instead of the default original-trained model.")
    parser.add_argument("--out-dir", type=Path, default=None,
                         help="Where to save annotated images (default: Data/Processed/geometry_debug).")
    parser.add_argument("--stems", nargs="+", default=None,
                         help="Inspect these test-image filename stems instead of the first --n images.")
    parser.add_argument("--data-yaml", default=None,
                         help="YOLO data yaml whose images to inspect (default: the original public dataset).")
    parser.add_argument("--split", default="test", choices=["train", "valid", "val", "test"],
                         help="Which split of --data-yaml to inspect (default: test).")
    args = parser.parse_args()
    run(args.n, save=not args.no_save, weights=args.weights, out_dir=args.out_dir, stems=args.stems,
        data_yaml=args.data_yaml, split=args.split)
