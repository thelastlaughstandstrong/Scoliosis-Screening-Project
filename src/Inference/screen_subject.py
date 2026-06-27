import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ============================================================
# Kidzonia — Subject Screening Inference
# Runs fine-tuned YOLO on a single image or video and outputs
# a structured proxy screening result.
#
# OUTPUT FRAMING:
# All outputs are camera-derived proxy measurements, not clinical
# diagnoses. proxy_screening_flag signals that a measurement
# exceeded the screening threshold and follow-up may be warranted.
# Accuracy claims require comparison against independent clinical
# ATR/scoliometer labels (not yet collected).
# ============================================================

import cv2
import numpy as np
import json
import math
import time
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = PROJECT_ROOT / "models" / "yolo_scoliosis" / "run" / "weights" / "best.pt"

ATR_PROXY_THRESHOLD_DEG = 5.0
VISIBILITY_THRESHOLD    = 0.5
SPINE_BOTTOM_IDX        = 0
SPINE_TOP_IDX           = 8
CLASS_NAMES             = {0: "normal", 1: "scoliosis"}


def spine_angle_deg(kps: np.ndarray) -> float:
    bottom = kps[SPINE_BOTTOM_IDX, :2]
    top    = kps[SPINE_TOP_IDX,    :2]
    dx = top[0] - bottom[0]
    dy = bottom[1] - top[1]
    return math.degrees(math.atan2(dx, dy))


def screen_frame(model: YOLO, img: np.ndarray) -> dict:
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

    if not pose_present:
        return {
            "pose_present":           False,
            "valid_measurement":      False,
            "spine_angle_deg":        None,
            "proxy_screening_flag":   None,
            "predicted_class":        None,
            "mean_kp_confidence":     0.0,
            "missing_kp_count":       9,
            "retake_recommended":     True,
            "processing_time_s":      processing_time,
            "note": "No subject detected — retake recommended.",
        }

    box_confs   = result.boxes.conf.cpu().numpy()
    best_idx    = int(box_confs.argmax())
    pred_cls_id = int(result.boxes.cls[best_idx].cpu().item())
    kps         = result.keypoints.data[best_idx].cpu().numpy()  # [12, 3]

    spine_confs      = [float(kps[i, 2]) for i in range(9)]
    mean_conf        = round(sum(spine_confs) / len(spine_confs), 4)
    missing_kp_count = sum(1 for c in spine_confs if c < VISIBILITY_THRESHOLD)
    valid = (
        missing_kp_count == 0
        and float(kps[SPINE_BOTTOM_IDX, 2]) >= VISIBILITY_THRESHOLD
        and float(kps[SPINE_TOP_IDX,    2]) >= VISIBILITY_THRESHOLD
    )

    angle   = round(spine_angle_deg(kps), 3) if valid else None
    flag    = int(abs(angle) >= ATR_PROXY_THRESHOLD_DEG) if angle is not None else None
    retake  = not valid

    return {
        "pose_present":           True,
        "valid_measurement":      valid,
        "spine_angle_deg":        angle,
        "proxy_screening_flag":   flag,
        "predicted_class":        CLASS_NAMES.get(pred_cls_id),
        "mean_kp_confidence":     mean_conf,
        "missing_kp_count":       missing_kp_count,
        "retake_recommended":     retake,
        "processing_time_s":      processing_time,
        "note": (
            "Retake recommended — keypoints not fully visible." if retake
            else f"Proxy screening flag: {'RAISED (follow-up may be needed)' if flag else 'NOT raised'}. "
                 "This is a camera-derived proxy, not a clinical diagnosis."
        ),
    }


def screen_image(model: YOLO, image_path: Path) -> dict:
    img = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return {"error": f"Could not read image: {image_path}"}
    result = screen_frame(model, img)
    result["source"] = str(image_path.name)
    result["source_type"] = "image"
    return result


def screen_video(model: YOLO, video_path: Path) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"error": f"Could not open video: {video_path}"}

    frame_results = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_results.append(screen_frame(model, frame))
    cap.release()

    if not frame_results:
        return {"error": "No frames read from video."}

    valid_frames  = [r for r in frame_results if r["valid_measurement"]]
    angles        = [r["spine_angle_deg"] for r in valid_frames if r["spine_angle_deg"] is not None]
    flags         = [r["proxy_screening_flag"] for r in valid_frames if r["proxy_screening_flag"] is not None]
    pose_coverage = sum(1 for r in frame_results if r["pose_present"]) / len(frame_results)

    mean_angle    = round(sum(angles) / len(angles), 3)          if angles else None
    angle_std     = round(float(np.std(angles)), 3)              if len(angles) > 1 else None
    final_flag    = int(sum(flags) / len(flags) >= 0.5)          if flags else None

    return {
        "source":               str(video_path.name),
        "source_type":          "video",
        "total_frames":         len(frame_results),
        "valid_frames":         len(valid_frames),
        "pose_coverage_ratio":  round(pose_coverage, 3),
        "mean_spine_angle_deg": mean_angle,
        "spine_angle_std_deg":  angle_std,
        "proxy_screening_flag": final_flag,
        "retake_recommended":   pose_coverage < 0.5 or len(valid_frames) < 10,
        "note": (
            "Pose coverage below 50% — retake recommended." if pose_coverage < 0.5
            else f"Proxy screening flag: {'RAISED' if final_flag else 'NOT raised'}. "
                 "Camera-derived proxy only — not a clinical diagnosis."
        ),
        # Placeholder for future clinical comparison
        "clinical_atr_deg":              None,
        "clinical_screening_positive":   None,
        "clinical_note": "Awaiting independent clinical ATR measurement for this subject.",
    }


def run(source: str, weights: Path = DEFAULT_WEIGHTS):
    if not weights.exists():
        print(f"[ERROR] Model weights not found: {weights}")
        print("        Run src/Evaluation/train_yolo_scoliosis.py first.")
        return

    model  = YOLO(str(weights))
    source = Path(source)

    if not source.exists():
        print(f"[ERROR] Source not found: {source}")
        return

    if source.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
        result = screen_image(model, source)
    elif source.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
        result = screen_video(model, source)
    else:
        print(f"[ERROR] Unsupported file type: {source.suffix}")
        return

    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python screen_subject.py <image_or_video_path> [weights_path]")
        sys.exit(1)
    weights = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_WEIGHTS
    run(sys.argv[1], weights)