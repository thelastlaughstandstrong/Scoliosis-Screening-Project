import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ============================================================
# Kidzonia School Screening – YOLO Keypoint Recording
# Mirrors quest_recording_auto.py but uses the fine-tuned
# YOLO 12-point spine model instead of MediaPipe.
#
# 12-point keypoint schema:
#   kp[0]  lumbar (hip midpoint)
#   kp[1-7] intermediate spine — invisible, ignored in loss
#   kp[8]  cervical (shoulder midpoint)
#   kp[9]  left shoulder
#   kp[10] right shoulder
#   kp[11] head (nose)
#
# Key additions vs MediaPipe version:
#   - predicted_class  : YOLO classification (0=normal, 1=scoliosis)
#   - mean_kp_confidence: average confidence of visible keypoints
#   - yolo_box_confidence: detection box confidence
#   - spine_angle_deg  : same as trunk_inclination_2d_deg, stored separately
#
# Limitations vs MediaPipe version:
#   - delta_hip_height_px unavailable (schema has only hip midpoint, not L/R hips)
#   - trunk_curvature_proxy_2d_deg unavailable (intermediate spine kps invisible)
#   - ATR proxy in Q2 uses spine inclination angle, not curvature
# ============================================================

import sys
from openni import openni2
from openni import _openni2 as c_api
import numpy as np
import cv2
import math
import time
import csv
import json
import re
from collections import deque
from pathlib import Path
from ultralytics import YOLO

_PROJECT_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT_FOR_IMPORTS))

from src.config.keypoint_schema import (
    SPINE_BASE_LUMBAR, SPINE_TOP_CERVICAL, LEFT_SHOULDER, RIGHT_SHOULDER, NECK_HEAD_CENTER,
)

# ---------------- CONFIG ----------------
QUESTS = [
    ("Q1_standing",      "Stand naturally, arms relaxed."),
    ("Q2_forward_bend",  "Bend forward with arms hanging."),
    ("Q3_upward_stretch","Stand straight and stretch upwards with your arms.")
]

FPS              = 30
_PROJECT_ROOT    = Path(__file__).resolve().parents[2]
OUTPUT_ROOT      = _PROJECT_ROOT / "Data" / "Raw" / "raw_data"
OPERATOR_ID_DEFAULT = "OP01"
FOURCC           = "XVID"

RECORD_SCALE     = 2.0
COUNTDOWN_SEC    = 3
RECORD_SEC       = 5
ANGLE_SMOOTH_WINDOW = 10

PRESENCE_WINDOW_FRAMES = 30
PRESENCE_THRESHOLD     = 0.5

# 12-point keypoint indices (from src/config/keypoint_schema.py)
KP_LUMBAR     = SPINE_BASE_LUMBAR
KP_CERVICAL   = SPINE_TOP_CERVICAL
KP_L_SHOULDER = LEFT_SHOULDER
KP_R_SHOULDER = RIGHT_SHOULDER
KP_HEAD       = NECK_HEAD_CENTER
KP_VISIBLE_IDS = [KP_LUMBAR, KP_CERVICAL, KP_L_SHOULDER, KP_R_SHOULDER, KP_HEAD]

KP_CONF_THRESHOLD = 0.3

# Prefer combined model; fall back to Roboflow-only fine-tuned
_WEIGHTS_COMBINED = _PROJECT_ROOT / "models" / "yolo_scoliosis_combined" / "run" / "weights" / "best.pt"
_WEIGHTS_ROBOFLOW = _PROJECT_ROOT / "models" / "yolo_scoliosis" / "run" / "weights" / "best.pt"
# ----------------------------------------


# ---------------- UTILS ----------------
def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def next_trial_id(path):
    ensure_dir(path)
    ids = [int(m.group(1)) for f in Path(path).iterdir()
           if (m := re.match(r"trial_(\d+)", f.name))]
    return max(ids) + 1 if ids else 1

def trunk_inclination_deg(p_top, p_bottom):
    dx = p_bottom[0] - p_top[0]
    dy = p_bottom[1] - p_top[1]
    return math.degrees(math.atan2(dx, dy))

def update_buffer(buf, val):
    if val is None:
        return None
    buf.append(float(val))
    if len(buf) > ANGLE_SMOOTH_WINDOW:
        buf.pop(0)
    return float(np.median(buf))

def get_depth_safe(depth_img, x, y):
    if depth_img is None:
        return None
    h, w = depth_img.shape
    xi = clamp(int(x), 0, w - 1)
    yi = clamp(int(y), 0, h - 1)
    d = float(depth_img[yi, xi])
    return None if d <= 0 else d

def draw_text(img, text, org, scale, color):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, 2, cv2.LINE_AA)

def load_model():
    for weights in [_WEIGHTS_COMBINED, _WEIGHTS_ROBOFLOW]:
        if weights.exists():
            print(f"[YOLO] Loaded weights: {weights.relative_to(_PROJECT_ROOT)}")
            return YOLO(str(weights))
    raise FileNotFoundError(
        "No fine-tuned YOLO weights found.\n"
        "Run:  python src/Evaluation/train_yolo_scoliosis.py\n"
        "Then: python src/Evaluation/train_yolo_combined.py  (optional, for domain adaptation)"
    )

def extract_pose(result):
    """
    Extract best-scoring detection from a YOLO result.
    Returns (kps, box_conf, pred_class) or (None, None, None).
    kps: np.ndarray (12, 3) — [x_px, y_px, conf]
    """
    if result.keypoints is None or len(result.keypoints) == 0:
        return None, None, None
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None, None, None
    confs = boxes.conf.cpu().numpy()
    best  = int(confs.argmax())
    kps   = result.keypoints.data[best].cpu().numpy()
    box_conf   = float(confs[best])
    pred_class = int(boxes.cls[best].cpu().item()) if boxes.cls is not None else None
    return kps, box_conf, pred_class

def kp_ok(kps, idx):
    return kps[idx][2] >= KP_CONF_THRESHOLD

def kp_xy(kps, idx):
    return float(kps[idx][0]), float(kps[idx][1])
# ----------------------------------------


CSV_HEADER = [
    "timestamp_epoch", "t_rel_s", "subject_id", "quest_id",
    "pose_present", "pose_confidence",
    "delta_shoulder_height_px", "delta_hip_height_px",
    "standing_midline_x_offset_px", "standing_symmetry_index",
    "rib_hump_proxy_px", "surface_asymmetry_score",
    "atr_proxy_deg", "atr_exceeds_5deg",
    "trunk_inclination_2d_deg", "trunk_inclination_3d_deg",
    "trunk_segment_angle_upper_2d_deg", "trunk_segment_angle_lower_2d_deg",
    "trunk_curvature_proxy_2d_deg",
    # YOLO-specific additions
    "spine_angle_deg", "mean_kp_confidence",
    "predicted_class", "yolo_box_confidence",
]

CLASS_NAMES = {0: "Normal", 1: "Scoliosis"}


def main():
    subject_id  = input("Enter subject_id: ").strip() or "S001"
    operator_id = input(f"Enter operator_id ({OPERATOR_ID_DEFAULT}): ").strip() or OPERATOR_ID_DEFAULT

    model = load_model()

    # -------- OpenNI --------
    openni2.initialize()
    dev = openni2.Device.open_any()

    color = dev.create_color_stream()
    color.set_video_mode(c_api.OniVideoMode(
        pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888,
        resolutionX=640, resolutionY=480, fps=30
    ))
    color.start()

    depth = dev.create_depth_stream()
    depth.set_video_mode(c_api.OniVideoMode(
        pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM,
        resolutionX=640, resolutionY=480, fps=30
    ))
    depth.start()

    try:
        dev.set_image_registration_mode(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)
    except Exception:
        pass

    # -------- Window --------
    WIN = "Kidzonia School Screening (YOLO)"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 720)

    quest_idx = 0
    quest_id, quest_prompt = QUESTS[quest_idx]
    mode = "IDLE"
    countdown_start = None
    recording_start = None

    writer     = None
    csv_file   = None
    csv_writer = None
    json_records = []

    buf_2d, buf_3d = [], []
    presence_buffer = deque(maxlen=PRESENCE_WINDOW_FRAMES)

    try:
        while True:
            # -------- Frame --------
            frame_c = color.read_frame()
            img = np.frombuffer(
                frame_c.get_buffer_as_uint8(), dtype=np.uint8
            ).reshape((frame_c.height, frame_c.width, 3))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            frame_d = depth.read_frame()
            depth_img = np.frombuffer(
                frame_d.get_buffer_as_uint16(), dtype=np.uint16
            ).reshape((frame_d.height, frame_d.width))

            img_rec = cv2.resize(img, None, fx=RECORD_SCALE, fy=RECORD_SCALE)
            h, w = img.shape[:2]
            now  = time.time()

            # -------- YOLO inference --------
            results = model(img, verbose=False)
            kps, box_conf, pred_class = extract_pose(results[0])

            key_kps_valid = (
                kps is not None
                and kp_ok(kps, KP_LUMBAR)
                and kp_ok(kps, KP_CERVICAL)
                and kp_ok(kps, KP_L_SHOULDER)
                and kp_ok(kps, KP_R_SHOULDER)
            )
            pose_present = key_kps_valid

            # -------- Subject presence gate --------
            presence_buffer.append(int(pose_present))
            subject_present = (
                len(presence_buffer) == PRESENCE_WINDOW_FRAMES
                and sum(presence_buffer) / PRESENCE_WINDOW_FRAMES >= PRESENCE_THRESHOLD
            )

            # -------- Feature initialisation --------
            delta_shoulder_height_px       = None
            delta_hip_height_px            = None   # unavailable in 12-pt schema
            standing_midline_x_offset_px   = None
            standing_symmetry_index        = None
            rib_hump_proxy_px              = None
            surface_asymmetry_score        = None
            atr_proxy_deg                  = None
            atr_exceeds_5deg               = None
            trunk_inclination_2d_deg       = None
            trunk_inclination_3d_deg       = None
            trunk_segment_angle_upper_2d_deg = None  # unavailable
            trunk_segment_angle_lower_2d_deg = None  # unavailable
            trunk_curvature_proxy_2d_deg   = None    # unavailable
            spine_angle_deg                = None
            mean_kp_confidence             = None
            pose_confidence                = 0.0

            # -------- Feature extraction --------
            if pose_present:
                lum_x, lum_y = kp_xy(kps, KP_LUMBAR)
                cer_x, cer_y = kp_xy(kps, KP_CERVICAL)
                ls_x,  ls_y  = kp_xy(kps, KP_L_SHOULDER)
                rs_x,  rs_y  = kp_xy(kps, KP_R_SHOULDER)

                # Mean confidence of visible keypoints
                vis_confs = [kps[i][2] for i in KP_VISIBLE_IDS if kps[i][2] >= KP_CONF_THRESHOLD]
                mean_kp_confidence = float(np.mean(vis_confs)) if vis_confs else 0.0
                pose_confidence = mean_kp_confidence

                # Spine angle — cervical top relative to lumbar base
                spine_angle_deg = trunk_inclination_deg(
                    (cer_x, cer_y),   # spine top = shoulder midpoint
                    (lum_x, lum_y)    # spine bot = hip midpoint
                )
                trunk_inclination_2d_deg = update_buffer(buf_2d, spine_angle_deg)

                # Shoulder asymmetry (available in all quests)
                delta_shoulder_height_px = abs(ls_y - rs_y)

                # Lateral shift: shoulder midpoint vs hip midpoint
                standing_midline_x_offset_px = abs(cer_x - lum_x)

                if quest_id == "Q1_standing":
                    # delta_hip_height_px unavailable — adjust normalisation
                    standing_symmetry_index = (
                        delta_shoulder_height_px + standing_midline_x_offset_px
                    ) / (h * 2)

                elif quest_id == "Q2_forward_bend":
                    # ATR proxy: spine inclination during forward bend
                    # (replaces curvature-based proxy — intermediate kps not visible)
                    atr_proxy_deg = abs(trunk_inclination_2d_deg) if trunk_inclination_2d_deg is not None else None
                    atr_exceeds_5deg = int(atr_proxy_deg is not None and atr_proxy_deg >= 5.0)
                    surface_asymmetry_score = delta_shoulder_height_px

                    # Depth-based rib hump (using YOLO shoulder / lumbar positions)
                    z_ls  = get_depth_safe(depth_img, ls_x, ls_y)
                    z_rs  = get_depth_safe(depth_img, rs_x, rs_y)
                    z_lum = get_depth_safe(depth_img, lum_x, lum_y)

                    z_top_vals = [z for z in (z_ls, z_rs) if z is not None]
                    z_top = float(np.mean(z_top_vals)) if z_top_vals else None
                    z_bot = z_lum

                    if z_top is not None and z_bot is not None:
                        rib_hump_proxy_px = abs(z_top - z_bot)
                        trunk_inclination_3d_deg = update_buffer(
                            buf_3d,
                            math.degrees(math.atan2(
                                rib_hump_proxy_px,
                                abs(lum_y - cer_y) + 1e-6
                            ))
                        )

                elif quest_id == "Q3_upward_stretch":
                    pass  # trunk_inclination_2d_deg and midline offset already set

                # -------- Draw YOLO keypoints on display frame --------
                kp_colors = {
                    KP_LUMBAR:     (0, 255, 0),
                    KP_CERVICAL:   (0, 255, 0),
                    KP_L_SHOULDER: (255, 150, 0),
                    KP_R_SHOULDER: (255, 150, 0),
                    KP_HEAD:       (255, 255, 0),
                }
                for idx, col in kp_colors.items():
                    if kp_ok(kps, idx):
                        cx = int(kps[idx][0] * RECORD_SCALE)
                        cy = int(kps[idx][1] * RECORD_SCALE)
                        cv2.circle(img_rec, (cx, cy), 7, col, -1)
                        cv2.circle(img_rec, (cx, cy), 7, (255, 255, 255), 2)

                # Spine line
                cv2.line(
                    img_rec,
                    (int(lum_x * RECORD_SCALE), int(lum_y * RECORD_SCALE)),
                    (int(cer_x * RECORD_SCALE), int(cer_y * RECORD_SCALE)),
                    (0, 255, 255), 4
                )

            # ---------- State machine ----------
            if mode == "COUNTDOWN" and now - countdown_start >= COUNTDOWN_SEC:
                quest_dir = OUTPUT_ROOT / subject_id / quest_id
                trial_id  = next_trial_id(quest_dir)
                trial_dir = quest_dir / f"trial_{trial_id:03d}"
                ensure_dir(trial_dir)

                writer = cv2.VideoWriter(
                    str(trial_dir / "video.avi"),
                    cv2.VideoWriter_fourcc(*FOURCC),
                    FPS,
                    img_rec.shape[1::-1]
                )

                csv_file   = open(trial_dir / "features.csv", "w", newline="", encoding="utf-8")
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(CSV_HEADER)
                json_records = []

                with open(trial_dir / "meta.json", "w", encoding="utf-8") as f:
                    json.dump({
                        "subject_id":  subject_id,
                        "quest_id":    quest_id,
                        "trial_id":    f"{trial_id:03d}",
                        "operator_id": operator_id,
                        "detector":    "yolo",
                    }, f, indent=2)

                mode = "RECORDING"
                recording_start = now

            elif mode == "RECORDING":
                t_rel = now - recording_start
                writer.write(img_rec)

                row = [
                    f"{now:.6f}", f"{t_rel:.3f}", subject_id, quest_id,
                    int(pose_present), f"{pose_confidence:.4f}",
                    delta_shoulder_height_px, delta_hip_height_px,
                    standing_midline_x_offset_px, standing_symmetry_index,
                    rib_hump_proxy_px, surface_asymmetry_score,
                    atr_proxy_deg, atr_exceeds_5deg,
                    trunk_inclination_2d_deg, trunk_inclination_3d_deg,
                    trunk_segment_angle_upper_2d_deg,
                    trunk_segment_angle_lower_2d_deg,
                    trunk_curvature_proxy_2d_deg,
                    spine_angle_deg, mean_kp_confidence,
                    pred_class, box_conf,
                ]
                csv_writer.writerow(row)

                json_records.append(dict(zip(CSV_HEADER, row)))

                if t_rel >= RECORD_SEC:
                    writer.release()
                    csv_file.close()
                    with open(trial_dir / "features.json", "w", encoding="utf-8") as f:
                        json.dump(json_records, f, indent=2)
                    print(f"[SAVED] {trial_dir.relative_to(_PROJECT_ROOT)}")
                    mode = "IDLE"

            # ---------- UI ----------
            overlay = img_rec.copy()
            cv2.rectangle(overlay, (0, 0), (img_rec.shape[1], 200), (0, 0, 0), -1)
            img_rec[:] = cv2.addWeighted(overlay, 0.35, img_rec, 0.65, 0)

            if mode == "IDLE":
                if not subject_present:
                    status = "WAITING FOR SUBJECT — ensure person is fully visible"
                else:
                    status = "IDLE | [S] Start  [N] Next  [B] Back  [ESC] Quit"
            elif mode == "COUNTDOWN":
                status = f"COUNTDOWN: {max(0, COUNTDOWN_SEC - int(now - countdown_start))}s"
            else:
                status = f"RECORDING: {max(0.0, RECORD_SEC - (now - recording_start)):.1f}s left"

            draw_text(img_rec, f"Subject: {subject_id}   Quest: {quest_id}",
                      (30, 45), 0.9, (255, 255, 255))
            draw_text(img_rec, f"Instruction: {quest_prompt}",
                      (30, 90), 0.75, (220, 220, 220))
            draw_text(img_rec, f"Status: {status}",
                      (30, 140), 0.75, (0, 255, 255))

            if trunk_inclination_2d_deg is not None:
                draw_text(img_rec, f"Spine angle: {trunk_inclination_2d_deg:.1f} deg",
                          (30, 250), 0.8, (0, 255, 0))
            if mean_kp_confidence is not None:
                draw_text(img_rec, f"KP confidence: {mean_kp_confidence:.2f}",
                          (30, 290), 0.7, (200, 200, 200))
            if pred_class is not None:
                label = CLASS_NAMES.get(pred_class, str(pred_class))
                class_color = (0, 100, 255) if pred_class == 1 else (0, 220, 0)
                draw_text(img_rec, f"YOLO class: {label}",
                          (30, 330), 0.85, class_color)
            if quest_id == "Q2_forward_bend" and atr_proxy_deg is not None:
                draw_text(img_rec, f"ATR proxy (incl.): {atr_proxy_deg:.1f} deg",
                          (30, 370), 0.8, (255, 150, 0))

            cv2.imshow(WIN, img_rec)

            # ---------- Keys ----------
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key in (ord('s'), ord('S')) and mode == "IDLE":
                if not subject_present:
                    print("[GATE] Recording blocked — no valid subject detected in last 1s window.")
                else:
                    mode = "COUNTDOWN"
                    countdown_start = now
            if key in (ord('n'), ord('N')) and mode == "IDLE":
                quest_idx = clamp(quest_idx + 1, 0, len(QUESTS) - 1)
                quest_id, quest_prompt = QUESTS[quest_idx]
            if key in (ord('b'), ord('B')) and mode == "IDLE":
                quest_idx = clamp(quest_idx - 1, 0, len(QUESTS) - 1)
                quest_id, quest_prompt = QUESTS[quest_idx]

    finally:
        try:
            if writer:
                writer.release()
            if csv_file:
                csv_file.close()
        except Exception:
            pass
        color.stop()
        depth.stop()
        openni2.unload()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
