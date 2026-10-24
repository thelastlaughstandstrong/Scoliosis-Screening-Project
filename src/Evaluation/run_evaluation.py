# ============================================================
# Evaluation Runner — MediaPipe / YOLO (pretrained) / YOLO (fine-tuned)
# Runs pose estimation on a split of a YOLO dataset and writes a shared
# comparison-schema CSV per workflow (for compare_workflows.py).
#
# Which images are evaluated is chosen with --data-yaml and --split. With no
# arguments the original public dataset's test split is used (public_datasets/
# scoliosis/test), exactly as before.
#
# Usage:
#   python run_evaluation.py --model mediapipe
#   python run_evaluation.py --model yolo_pretrained
#   python run_evaluation.py --model yolo_finetuned
#   python run_evaluation.py --model all            (default)
#   python run_evaluation.py --model yolo_finetuned \
#       --weights models/yolo_scoliosis_grouped_conservative/run/weights/best.pt \
#       --data-yaml Data/Processed/scoliosis_grouped_split/scoliosis_grouped_split.yaml \
#       --split test --tag grouped_conservative
#
# WARNING: models trained on the old split should not be interpreted as unbiased on the
# grouped test set — the old train split contains photos the grouped split puts in test.
# Retrain on the grouped train/valid split before reporting grouped-split performance.
# Results of old-split models are historical/debug only once the grouped split exists.
#
# Output CSV name (kept backward compatible):
#   default dataset + test split -> <workflow>_scoliosis_test[_<tag>].csv        (unchanged)
#   any other dataset/split      -> <workflow>_<dataset>_<split>[_<tag>].csv
# plus a <csv>.meta.json sidecar recording the dataset, split and weights used.
#
# Run order for the full comparison:
#   1. run_evaluation.py --model mediapipe
#   2. run_evaluation.py --model yolo_pretrained
#   3. train_yolo_scoliosis.py
#   4. run_evaluation.py --model yolo_finetuned
#   5. compare_workflows.py
# ============================================================

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT       = Path(__file__).resolve().parents[2]
COMPARISON_DIR      = PROJECT_ROOT / "Data" / "Processed" / "comparison"
FINETUNED_WEIGHTS   = PROJECT_ROOT / "models" / "yolo_scoliosis" / "run" / "weights" / "best.pt"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.keypoint_schema import SPINE_BASE_LUMBAR, SPINE_TOP_CERVICAL, SPINE_POINT_IDXS
from src.FeatureExtraction.spine_geometry import extract_spine_geometry, empty_spine_geometry
from src.Evaluation.dataset_paths import (
    resolve_split, write_meta, trained_on_yaml, same_path, rel, LEGACY_DATASET_NAMES, OLD_SPLIT_WARNING,
)

CLASS_NAMES = {0: "normal", 1: "scoliosis"}
VISIBILITY_THRESHOLD = 0.5


def load_ground_truth(label_path: Path):
    content = label_path.read_text().strip()
    if not content:
        return None, None
    class_id = int(content.split()[0])
    return class_id, CLASS_NAMES[class_id]


def imdecode_unicode(img_path: Path):
    # cv2.imread fails on Windows paths with non-ASCII chars; use imdecode instead
    return cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)


# ---------------------------------------------------------------
# Workflow: MediaPipe — shoulder + hip landmarks as spine proxy
# ---------------------------------------------------------------

MEDIAPIPE_KEY_LANDMARKS = {"left_shoulder": 11, "right_shoulder": 12, "left_hip": 23, "right_hip": 24}


def mediapipe_setup():
    import mediapipe as mp
    return mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.5)


def mediapipe_angle(lm, h, w):
    ls = (lm[11].x * w, lm[11].y * h)
    rs = (lm[12].x * w, lm[12].y * h)
    lh = (lm[23].x * w, lm[23].y * h)
    rh = (lm[24].x * w, lm[24].y * h)
    spine_top = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
    spine_bot = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)
    dx = spine_bot[0] - spine_top[0]
    dy = spine_bot[1] - spine_top[1]
    return math.degrees(math.atan2(dx, dy))


def mediapipe_process(pose, img, gt_id):
    h, w = img.shape[:2]
    result = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    pose_present = result.pose_landmarks is not None

    if pose_present:
        lm = result.pose_landmarks.landmark
        key_vis = [lm[i].visibility for i in MEDIAPIPE_KEY_LANDMARKS.values()]
        mean_confidence    = round(sum(key_vis) / len(key_vis), 4)
        missing_kp_count   = sum(1 for v in key_vis if v < VISIBILITY_THRESHOLD)
        valid_measurement  = missing_kp_count == 0
        primary_angle      = round(mediapipe_angle(lm, h, w), 3) if valid_measurement else None
    else:
        mean_confidence, missing_kp_count, valid_measurement, primary_angle = (
            0.0, len(MEDIAPIPE_KEY_LANDMARKS), False, None
        )

    return {
        "pose_present":           int(pose_present),
        "valid_measurement":      int(valid_measurement),
        "mean_confidence":        mean_confidence,
        "missing_keypoint_count": missing_kp_count,
        "primary_angle_deg":      primary_angle,
        "retake_recommended":     int(not valid_measurement),
        "processing_time_s":      None,   # filled in by run() after timing inference
    }


# ---------------------------------------------------------------
# Workflow: YOLO pretrained — COCO keypoints 5,6,11,12 mirror MediaPipe
# ---------------------------------------------------------------

YOLO_KEY_IDX = {"left_shoulder": 5, "right_shoulder": 6, "left_hip": 11, "right_hip": 12}


def yolo_pretrained_setup():
    from ultralytics import YOLO
    return YOLO("yolov8n-pose.pt")   # downloads automatically on first run


def yolo_trunk_angle(kps):
    """kps: array [17, 3] of (x, y, conf) in pixel coords."""
    ls, rs, lh, rh = kps[5, :2], kps[6, :2], kps[11, :2], kps[12, :2]
    spine_top, spine_bot = (ls + rs) / 2, (lh + rh) / 2
    dx, dy = spine_bot[0] - spine_top[0], spine_bot[1] - spine_top[1]
    return math.degrees(math.atan2(dx, dy))


def yolo_pretrained_process(model, img, gt_id):
    result = model(img, verbose=False)[0]
    num_detections = len(result.boxes) if result.boxes is not None else 0
    pose_present = num_detections > 0 and result.keypoints is not None and len(result.keypoints.data) > 0

    if pose_present:
        box_confs = result.boxes.conf.cpu().numpy()
        best_idx  = int(box_confs.argmax())
        det_conf  = float(box_confs[best_idx])
        kps       = result.keypoints.data[best_idx].cpu().numpy()  # [17, 3]

        key_confs          = [float(kps[i, 2]) for i in YOLO_KEY_IDX.values()]
        mean_confidence     = round(sum(key_confs) / len(key_confs), 4)
        missing_kp_count    = sum(1 for c in key_confs if c < VISIBILITY_THRESHOLD)
        valid_measurement   = missing_kp_count == 0
        primary_angle       = round(yolo_trunk_angle(kps), 3) if valid_measurement else None
    else:
        det_conf, mean_confidence, missing_kp_count, valid_measurement, primary_angle = (
            0.0, 0.0, len(YOLO_KEY_IDX), False, None
        )

    return {
        "pose_present":           int(pose_present),
        "valid_measurement":      int(valid_measurement),
        "mean_confidence":        mean_confidence,
        "missing_keypoint_count": missing_kp_count,
        "primary_angle_deg":      primary_angle,
        "retake_recommended":     int(not valid_measurement),
        "processing_time_s":      None,
        "detection_confidence":   round(det_conf, 4),
        "num_detections":         num_detections,
    }


# ---------------------------------------------------------------
# Workflow: YOLO fine-tuned — 12-point spine schema + classification
#   kp[0]  = lumbar base (bottom of spine)
#   kp[1..7] = intermediate vertebral landmarks
#   kp[8]  = cervical top (top of spine)
#   kp[9]  = left shoulder, kp[10] = right shoulder, kp[11] = head/neck
# ---------------------------------------------------------------

SPINE_BOTTOM_IDX = SPINE_BASE_LUMBAR
SPINE_TOP_IDX    = SPINE_TOP_CERVICAL
SPINE_KEY_IDX    = SPINE_POINT_IDXS   # spine points only, used for quality metrics


def yolo_finetuned_setup(weights_path=None):
    weights_path = Path(weights_path) if weights_path else FINETUNED_WEIGHTS
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Fine-tuned weights not found at: {weights_path}\n"
            "        Run train_yolo_scoliosis.py first."
        )
    from ultralytics import YOLO
    return YOLO(str(weights_path))


def spine_angle_deg(kps):
    """Angle of the spine line from lumbar base (kp[0]) to cervical top (kp[8])."""
    bottom, top = kps[SPINE_BOTTOM_IDX, :2], kps[SPINE_TOP_IDX, :2]
    dx = top[0] - bottom[0]
    dy = bottom[1] - top[1]   # flipped: image y increases downward
    return math.degrees(math.atan2(dx, dy))


def yolo_finetuned_process(model, img, gt_id):
    result = model(img, verbose=False)[0]
    num_detections = len(result.boxes) if result.boxes is not None else 0
    pose_present = num_detections > 0 and result.keypoints is not None and len(result.keypoints.data) > 0

    if pose_present:
        box_confs     = result.boxes.conf.cpu().numpy()
        best_idx      = int(box_confs.argmax())
        det_conf      = float(box_confs[best_idx])
        pred_class_id = int(result.boxes.cls[best_idx].cpu().item())
        pred_class    = CLASS_NAMES.get(pred_class_id, "unknown")
        kps           = result.keypoints.data[best_idx].cpu().numpy()  # [12, 3]

        spine_confs             = [float(kps[i, 2]) for i in SPINE_KEY_IDX]
        mean_confidence          = round(sum(spine_confs) / len(spine_confs), 4)
        missing_kp_count         = sum(1 for c in spine_confs if c < VISIBILITY_THRESHOLD)
        valid_measurement        = missing_kp_count == 0
        primary_angle            = round(spine_angle_deg(kps), 3) if valid_measurement else None
        classification_correct   = int(pred_class_id == gt_id) if gt_id is not None else None
        geometry                 = extract_spine_geometry(kps)
    else:
        (det_conf, pred_class, mean_confidence, missing_kp_count,
         valid_measurement, primary_angle, classification_correct) = (
            0.0, None, 0.0, len(SPINE_KEY_IDX), False, None, None
        )
        geometry = empty_spine_geometry("no_pose_detected")

    return {
        "predicted_class":        pred_class,
        "classification_correct": classification_correct,
        "pose_present":           int(pose_present),
        "valid_measurement":      int(valid_measurement),
        "mean_confidence":        mean_confidence,
        "missing_keypoint_count": missing_kp_count,
        "primary_angle_deg":      primary_angle,
        "retake_recommended":     int(not valid_measurement),
        "processing_time_s":      None,
        "detection_confidence":   round(det_conf, 4),
        "num_detections":         num_detections,
        **geometry,
    }


# ---------------------------------------------------------------
# Registry + shared runner
# ---------------------------------------------------------------

# "output_csv" is the legacy name used for the default dataset + test split (kept so every older
# command and file still lines up); "prefix" is used to build the name for any other dataset/split.
WORKFLOWS = {
    "mediapipe": {
        "workflow_type": "mediapipe",
        "setup":         mediapipe_setup,
        "process":       mediapipe_process,
        "output_csv":    COMPARISON_DIR / "mediapipe_scoliosis_test.csv",
        "prefix":        "mediapipe",
    },
    "yolo_pretrained": {
        "workflow_type": "yolo_pretrained",
        "setup":         yolo_pretrained_setup,
        "process":       yolo_pretrained_process,
        "output_csv":    COMPARISON_DIR / "yolo_scoliosis_test.csv",
        "prefix":        "yolo_pretrained",
    },
    "yolo_finetuned": {
        "workflow_type": "yolo_finetuned",
        "setup":         yolo_finetuned_setup,
        "process":       yolo_finetuned_process,
        "output_csv":    COMPARISON_DIR / "yolo_finetuned_scoliosis_test.csv",
        "prefix":        "yolo_finetuned",
    },
}


def output_csv_path(wf: dict, source: dict, tag: str = None) -> Path:
    """Results CSV path. Default dataset + test split keeps the legacy name; anything else embeds
    the dataset and split so a grouped-split run can never overwrite an old-split result."""
    if source["is_legacy_default"]:
        path = wf["output_csv"]
    else:
        path = COMPARISON_DIR / f"{wf['prefix']}_{source['dataset_name']}_{source['split']}.csv"
    if tag:
        path = path.with_name(f"{path.stem}_{tag}{path.suffix}")
    return path


def check_training_provenance(model, source: dict, weights) -> dict:
    """Compare the dataset a fine-tuned model was trained on with the one it is evaluated on,
    and warn when evaluating on a grouped/new split a model could have seen part of during training."""
    trained_on = trained_on_yaml(model)
    info = {"trained_on_data_yaml": rel(trained_on), "trained_on_matches_eval": None, "old_split_warning": False}
    if trained_on is None:
        print("[INFO] Could not read the training dataset from the weights; cannot check for train/test overlap.")
        return info
    info["trained_on_matches_eval"] = same_path(trained_on, source["yaml_path"])
    if info["trained_on_matches_eval"]:
        return info
    print(f"[INFO] Model was trained on : {rel(trained_on)}")
    print(f"       Evaluating on        : {rel(source['yaml_path'])} ({source['split']})")
    eval_is_legacy = source["dataset_name"] in LEGACY_DATASET_NAMES
    trained_is_legacy = trained_on.parent.name in LEGACY_DATASET_NAMES
    if trained_is_legacy and not eval_is_legacy:
        info["old_split_warning"] = True
        print(f"[WARN] {OLD_SPLIT_WARNING}")
    elif eval_is_legacy and not trained_is_legacy:
        info["old_split_warning"] = True
        print("[WARN] This model was trained on a grouped/new split but is being evaluated on the OLD test folder, "
              "which shares photos with the grouped train split. Evaluate it on its own grouped test split instead.")
    return info


def run(model_key: str, weights: str = None, tag: str = None, data_yaml=None, split: str = "test"):
    """
    Args:
        weights: only used for model_key == "yolo_finetuned" — overrides
            FINETUNED_WEIGHTS so you can evaluate a model trained on a
            different dataset (e.g. Data/Processed/scoliosis_cleaned) without
            touching the default weights path.
        tag: suffix appended to the output CSV filename so an alternate-weights
            run doesn't overwrite the default run's results. If --weights is
            given without --tag, the weights' parent model directory name is
            used automatically (e.g. "yolo_scoliosis_cleaned").
        data_yaml / split: which YOLO dataset and split to evaluate (default: the original
            public dataset's test split, i.e. the previous behaviour).
    """
    if weights and model_key != "yolo_finetuned":
        print(f"[WARN] --weights is only used by --model yolo_finetuned; ignoring for {model_key}.")
        weights = None
    if weights and not tag:
        tag = Path(weights).resolve().parents[2].name   # .../<model_dir>/run/weights/best.pt

    try:
        source = resolve_split(data_yaml, split)
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"[ERROR] {e}")
        return

    wf = WORKFLOWS[model_key]
    output_csv = output_csv_path(wf, source, tag)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    images_dir, labels_dir = source["images_dir"], source["labels_dir"]
    image_files = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    if not image_files:
        print(f"[ERROR] No images found in {images_dir}")
        return
    if labels_dir is None:
        print(f"[WARN] No labels folder next to {images_dir} — ground truth will be missing.")

    print(f"\n[INFO] Running workflow: {model_key}" + (f"  (weights={weights})" if weights else ""))
    print(f"[INFO] Evaluating on: {rel(source['yaml_path'])}  split={split}  ({len(image_files)} images)")
    if source["split"] == "train":
        print("[WARN] Evaluating on the TRAIN split — a fine-tuned model has seen these images; results are not a test.")
    try:
        model = wf["setup"](weights) if model_key == "yolo_finetuned" else wf["setup"]()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return

    provenance = (check_training_provenance(model, source, weights) if model_key == "yolo_finetuned"
                  else {"trained_on_data_yaml": None, "trained_on_matches_eval": None, "old_split_warning": False})

    rows = []
    try:
        for img_path in image_files:
            label_path = (labels_dir / (img_path.stem + ".txt")) if labels_dir else None
            gt_id, gt_class = load_ground_truth(label_path) if label_path and label_path.exists() else (None, None)

            img = imdecode_unicode(img_path)
            if img is None:
                print(f"[WARN] Could not read {img_path.name}")
                continue

            t0 = time.perf_counter()
            fields = wf["process"](model, img, gt_id)
            fields["processing_time_s"] = round(time.perf_counter() - t0, 4)

            row = {
                "capture_id":         img_path.stem,
                "workflow_type":      wf["workflow_type"],
                "ground_truth_label": gt_id,
                "ground_truth_class": gt_class,
                **fields,
            }
            rows.append(row)
            print_row_status(row, img_path.name, gt_class)
    finally:
        if hasattr(model, "close"):
            model.close()

    if not rows:
        print("[ERROR] No results produced.")
        return

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    write_meta(
        output_csv,
        model=model_key, tag=tag,
        weights=rel(Path(weights)) if weights else None,
        eval_data_yaml=rel(source["yaml_path"]), eval_dataset=source["dataset_name"],
        eval_split=source["split"], n_images=len(rows),
        **provenance,
    )
    print_summary(rows, output_csv, split)


def print_row_status(row, img_name, gt_class):
    status = "OK  " if row["valid_measurement"] else "FAIL"
    extra_bits = []
    if "predicted_class" in row:
        correct_str = ("Y" if row["classification_correct"] else "N") if row["classification_correct"] is not None else "-"
        extra_bits.append(f"pred={str(row['predicted_class']):10s} cls_ok={correct_str}")
    if "num_detections" in row:
        extra_bits.append(f"dets={row['num_detections']}")
    if "valid_spine_geometry" in row:
        extra_bits.append(f"geom_ok={'Y' if row['valid_spine_geometry'] else 'N'}")
    extra_str = ("  " + "  ".join(extra_bits)) if extra_bits else ""

    print(
        f"[{status}] {img_name:55s} "
        f"gt={str(gt_class):10s} conf={row['mean_confidence']:.2f}  "
        f"angle={str(row['primary_angle_deg']):>8}{extra_str}"
    )


def print_summary(rows, output_csv, split="test"):
    total    = len(rows)
    valid    = sum(r["valid_measurement"] for r in rows)
    n_normal = [r for r in rows if r["ground_truth_class"] == "normal"]
    n_scoli  = [r for r in rows if r["ground_truth_class"] == "scoliosis"]

    print(f"\n{'='*60}")
    print(f"  Total images processed : {total}")
    print(f"  Valid measurements     : {valid} / {total}  ({100*valid/total:.0f}%)")
    print(f"\n  --- Per-class detection (pose found + valid measurement) ---")
    print(f"  Normal    : {sum(r['valid_measurement'] for r in n_normal)} / {len(n_normal)} detected")
    print(f"  Scoliosis : {sum(r['valid_measurement'] for r in n_scoli)} / {len(n_scoli)} detected")

    if "classification_correct" in rows[0]:
        cls_rows = [r for r in rows if r["classification_correct"] is not None]
        cls_acc  = sum(r["classification_correct"] for r in cls_rows) / len(cls_rows) if cls_rows else 0
        scoli_detected = [r for r in n_scoli if r["predicted_class"] == "scoliosis"]
        print(f"\n  --- Classification (predicted class vs ground truth) ---")
        print(f"  Overall accuracy  : {cls_acc:.0%}  ({sum(r['classification_correct'] for r in cls_rows if r['classification_correct'])}/{len(cls_rows)})")
        print(f"  Scoliosis recall  : {len(scoli_detected)}/{len(n_scoli)} correctly predicted as scoliosis")

    print(f"\n  NOTE: Only {len(n_scoli)} scoliosis cases in the {split} set.")
    print(f"  Use per-class recall, not accuracy, to evaluate screening reliability.")
    print(f"\n  Output saved to: {output_csv.name} (in Data/Processed/comparison/)")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pose-estimation evaluation on the scoliosis test set.")
    parser.add_argument(
        "--model", choices=[*WORKFLOWS.keys(), "all"], default="all",
        help="Which workflow to evaluate (default: all)",
    )
    parser.add_argument(
        "--weights", default=None,
        help="Fine-tuned YOLO weights to use instead of the default (only applies to "
             "--model yolo_finetuned) — e.g. models/yolo_scoliosis_cleaned/run/weights/best.pt",
    )
    parser.add_argument(
        "--tag", default=None,
        help="Suffix for the output CSV filename, so an alternate --weights run doesn't "
             "overwrite the default results (auto-derived from --weights if omitted).",
    )
    parser.add_argument(
        "--data-yaml", default=None,
        help="YOLO data yaml of the dataset to evaluate on (default: the original public dataset, "
             "public_datasets/scoliosis/data.yaml). e.g. Data/Processed/scoliosis_grouped_split/scoliosis_grouped_split.yaml",
    )
    parser.add_argument(
        "--split", default="test", choices=["train", "valid", "val", "test"],
        help="Which split of --data-yaml to evaluate (default: test).",
    )
    args = parser.parse_args()

    for key in (WORKFLOWS.keys() if args.model == "all" else [args.model]):
        run(key, weights=args.weights, tag=args.tag, data_yaml=args.data_yaml, split=args.split)
