# ============================================================
# Kidzonia School Screening – Multi-Quest Recording (SINGLE FILE)
# Features: surface-based posture & trunk geometry proxies only
# CSV + JSON logging, large display, manual control
# ============================================================

from openni import openni2
from openni import _openni2 as c_api
import numpy as np
import cv2
import mediapipe as mp
import math
import os
import time
import csv
import json
import re

# ---------------- CONFIG ----------------
QUESTS = [
    ("Q1_standing", "Stand naturally, arms relaxed."),
    ("Q2_forward_bend", "Bend forward with arms hanging."),
    ("Q3_upward_stretch", "Stand straight and stretch upwards with your arms.")
]

FPS = 30
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_ROOT = os.path.join(_PROJECT_ROOT, "Data", "Raw", "raw_data")
OPERATOR_ID_DEFAULT = "OP01"
FOURCC = "XVID"

RECORD_SCALE = 2.0
COUNTDOWN_SEC = 3
RECORD_SEC = 5
ANGLE_SMOOTH_WINDOW = 10
# ---------------------------------------


# ---------------- UTILS ----------------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def next_trial_id(path):
    ensure_dir(path)
    ids = []
    for f in os.listdir(path):
        m = re.match(r"trial_(\d+)", f)
        if m:
            ids.append(int(m.group(1)))
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
    x = clamp(int(x), 0, w - 1)
    y = clamp(int(y), 0, h - 1)
    d = float(depth_img[y, x])
    return None if d <= 0 else d

def draw_text(img, text, org, scale, color):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, 2, cv2.LINE_AA)
# ---------------------------------------


def main():
    subject_id = input("Enter subject_id: ").strip() or "S001"
    operator_id = input(f"Enter operator_id ({OPERATOR_ID_DEFAULT}): ").strip() or OPERATOR_ID_DEFAULT

    # -------- MediaPipe --------
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_draw = mp.solutions.drawing_utils

    # -------- OpenNI --------
    openni2.initialize()
    dev = openni2.Device.open_any()

    color = dev.create_color_stream()
    color.set_video_mode(
        c_api.OniVideoMode(
            pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888,
            resolutionX=640,
            resolutionY=480,
            fps=30
        )
    )
    color.start()

    depth = dev.create_depth_stream()
    depth.set_video_mode(
        c_api.OniVideoMode(
            pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM,
            resolutionX=640,
            resolutionY=480,
            fps=30
        )
    )
    depth.start()

    try:
        dev.set_image_registration_mode(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)
    except Exception:
        pass

    # -------- Window --------
    WIN = "Kidzonia School Screening"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 720)

    quest_idx = 0
    quest_id, quest_prompt = QUESTS[quest_idx]

    mode = "IDLE"
    countdown_start = None
    recording_start = None

    writer = None
    csv_file = None
    csv_writer = None
    json_records = []

    buf_2d, buf_3d, buf_curv = [], [], []

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
            h, w, _ = img.shape
            now = time.time()

            # -------- Pose --------
            result = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            pose_present = result.pose_landmarks is not None
            pose_confidence = 1.0 if pose_present else 0.0
            if pose_present:
                lm = result.pose_landmarks.landmark
                ls_vis = lm[11].visibility
                rs_vis = lm[12].visibility
                lh_vis = lm[23].visibility
                rh_vis = lm[24].visibility
                if min(ls_vis, rs_vis, lh_vis, rh_vis) < 0.6:
                    pose_present = False

            # -------- initialise ALL features as None --------
            delta_shoulder_height_px = None
            delta_hip_height_px = None
            standing_midline_x_offset_px = None
            standing_symmetry_index = None

            rib_hump_proxy_px = None
            surface_asymmetry_score = None
            atr_proxy_deg = None
            atr_exceeds_5deg = None

            trunk_inclination_2d_deg = None
            trunk_inclination_3d_deg = None
            trunk_segment_angle_upper_2d_deg = None
            trunk_segment_angle_lower_2d_deg = None
            trunk_curvature_proxy_2d_deg = None

            if pose_present:
                mp_draw.draw_landmarks(img_rec, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                lm = result.pose_landmarks.landmark

                ls = (lm[11].x * w, lm[11].y * h)
                rs = (lm[12].x * w, lm[12].y * h)
                lh = (lm[23].x * w, lm[23].y * h)
                rh = (lm[24].x * w, lm[24].y * h)

                spine_top = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
                spine_bot = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)
                mid_torso = ((spine_top[0] + spine_bot[0]) / 2,
                             (spine_top[1] + spine_bot[1]) / 2)

                # ---------- Q1: Standing ----------
                if quest_id == "Q1_standing":
                    delta_shoulder_height_px = abs(ls[1] - rs[1])
                    delta_hip_height_px = abs(lh[1] - rh[1])

                    mid_sh_x = (ls[0] + rs[0]) / 2
                    mid_hip_x = (lh[0] + rh[0]) / 2
                    standing_midline_x_offset_px = abs(mid_sh_x - mid_hip_x)

                    standing_symmetry_index = (
                        delta_shoulder_height_px +
                        delta_hip_height_px +
                        standing_midline_x_offset_px
                    ) / (h * 3)

                    trunk_inclination_2d_deg = update_buffer(
                        buf_2d, trunk_inclination_deg(spine_top, spine_bot)
                    )

                # ---------- Q2: Forward bend ----------
                elif quest_id == "Q2_forward_bend":
                    trunk_segment_angle_upper_2d_deg = trunk_inclination_deg(spine_top, mid_torso)
                    trunk_segment_angle_lower_2d_deg = trunk_inclination_deg(mid_torso, spine_bot)

                    trunk_curvature_proxy_2d_deg = update_buffer(
                        buf_curv,
                        abs(trunk_segment_angle_upper_2d_deg -
                            trunk_segment_angle_lower_2d_deg)
                    )

                    atr_proxy_deg = trunk_curvature_proxy_2d_deg
                    atr_exceeds_5deg = int(atr_proxy_deg is not None and atr_proxy_deg >= 5.0)

                    trunk_inclination_2d_deg = update_buffer(
                        buf_2d, trunk_inclination_deg(spine_top, spine_bot)
                    )

                    z_ls = get_depth_safe(depth_img, *ls)
                    z_rs = get_depth_safe(depth_img, *rs)
                    z_lh = get_depth_safe(depth_img, *lh)
                    z_rh = get_depth_safe(depth_img, *rh)

                    z_top = np.mean([z for z in (z_ls, z_rs) if z is not None]) if any((z_ls, z_rs)) else None
                    z_bot = np.mean([z for z in (z_lh, z_rh) if z is not None]) if any((z_lh, z_rh)) else None

                    if z_top is not None and z_bot is not None:
                        rib_hump_proxy_px = abs(z_top - z_bot)
                        trunk_inclination_3d_deg = update_buffer(
                            buf_3d,
                            math.degrees(math.atan2(
                                rib_hump_proxy_px,
                                abs(spine_bot[1] - spine_top[1]) + 1e-6
                            ))
                        )

                    surface_asymmetry_score = trunk_curvature_proxy_2d_deg

                # ---------- Q3: Upward stretch ----------
                elif quest_id == "Q3_upward_stretch":
                    mid_sh_x = (ls[0] + rs[0]) / 2
                    mid_hip_x = (lh[0] + rh[0]) / 2
                    standing_midline_x_offset_px = abs(mid_sh_x - mid_hip_x)

                    trunk_inclination_2d_deg = update_buffer(
                        buf_2d, trunk_inclination_deg(spine_top, spine_bot)
                    )

                cv2.line(
                    img_rec,
                    (int(spine_top[0] * RECORD_SCALE), int(spine_top[1] * RECORD_SCALE)),
                    (int(spine_bot[0] * RECORD_SCALE), int(spine_bot[1] * RECORD_SCALE)),
                    (0, 255, 255), 4
                )

            # ---------- State machine ----------
            if mode == "COUNTDOWN" and now - countdown_start >= COUNTDOWN_SEC:
                quest_dir = os.path.join(OUTPUT_ROOT, subject_id, quest_id)
                trial_id = next_trial_id(quest_dir)
                trial_dir = os.path.join(quest_dir, f"trial_{trial_id:03d}")
                ensure_dir(trial_dir)

                writer = cv2.VideoWriter(
                    os.path.join(trial_dir, "video.avi"),
                    cv2.VideoWriter_fourcc(*FOURCC),
                    FPS,
                    img_rec.shape[1::-1]
                )

                csv_file = open(os.path.join(trial_dir, "features.csv"), "w", newline="", encoding="utf-8")
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow([
                    "timestamp_epoch","t_rel_s","subject_id","quest_id",
                    "pose_present","pose_confidence",
                    "delta_shoulder_height_px","delta_hip_height_px",
                    "standing_midline_x_offset_px","standing_symmetry_index",
                    "rib_hump_proxy_px","surface_asymmetry_score",
                    "atr_proxy_deg","atr_exceeds_5deg",
                    "trunk_inclination_2d_deg","trunk_inclination_3d_deg",
                    "trunk_segment_angle_upper_2d_deg","trunk_segment_angle_lower_2d_deg",
                    "trunk_curvature_proxy_2d_deg"
                ])

                json_records = []

                with open(os.path.join(trial_dir, "meta.json"), "w", encoding="utf-8") as f:
                    json.dump({
                        "subject_id": subject_id,
                        "quest_id": quest_id,
                        "trial_id": f"{trial_id:03d}",
                        "operator_id": operator_id
                    }, f, indent=2)

                mode = "RECORDING"
                recording_start = now

            elif mode == "RECORDING":
                t_rel = now - recording_start
                writer.write(img_rec)

                csv_writer.writerow([
                    f"{now:.6f}", f"{t_rel:.3f}", subject_id, quest_id,
                    int(pose_present), f"{pose_confidence:.2f}",
                    delta_shoulder_height_px, delta_hip_height_px,
                    standing_midline_x_offset_px, standing_symmetry_index,
                    rib_hump_proxy_px, surface_asymmetry_score,
                    atr_proxy_deg, atr_exceeds_5deg,
                    trunk_inclination_2d_deg, trunk_inclination_3d_deg,
                    trunk_segment_angle_upper_2d_deg, trunk_segment_angle_lower_2d_deg,
                    trunk_curvature_proxy_2d_deg
                ])

                json_records.append({
                    "timestamp_epoch": now,
                    "t_rel_s": t_rel,
                    "subject_id": subject_id,
                    "quest_id": quest_id,
                    "pose_present": pose_present,
                    "pose_confidence": pose_confidence,
                    "delta_shoulder_height_px": delta_shoulder_height_px,
                    "delta_hip_height_px": delta_hip_height_px,
                    "standing_midline_x_offset_px": standing_midline_x_offset_px,
                    "standing_symmetry_index": standing_symmetry_index,
                    "rib_hump_proxy_px": rib_hump_proxy_px,
                    "surface_asymmetry_score": surface_asymmetry_score,
                    "atr_proxy_deg": atr_proxy_deg,
                    "atr_exceeds_5deg": atr_exceeds_5deg,
                    "trunk_inclination_2d_deg": trunk_inclination_2d_deg,
                    "trunk_inclination_3d_deg": trunk_inclination_3d_deg,
                    "trunk_segment_angle_upper_2d_deg": trunk_segment_angle_upper_2d_deg,
                    "trunk_segment_angle_lower_2d_deg": trunk_segment_angle_lower_2d_deg,
                    "trunk_curvature_proxy_2d_deg": trunk_curvature_proxy_2d_deg
                })

                if t_rel >= RECORD_SEC:
                    writer.release()
                    csv_file.close()
                    with open(os.path.join(trial_dir, "features.json"), "w", encoding="utf-8") as f:
                        json.dump(json_records, f, indent=2)
                    mode = "IDLE"

            # ---------- UI ----------
            overlay = img_rec.copy()
            cv2.rectangle(overlay, (0, 0), (img_rec.shape[1], 190), (0, 0, 0), -1)
            img_rec[:] = cv2.addWeighted(overlay, 0.35, img_rec, 0.65, 0)

            if mode == "IDLE":
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
                      (30, 135), 0.75, (0, 255, 255))

            # Measurements (quest-specific display)
            if trunk_inclination_2d_deg is not None:
                draw_text(img_rec, f"Trunk tilt (2D): {trunk_inclination_2d_deg:.1f} deg",
                          (30, 240), 0.8, (0, 255, 0))
            if quest_id == "Q2_forward_bend" and trunk_curvature_proxy_2d_deg is not None:
                draw_text(img_rec, f"Trunk curvature proxy: {trunk_curvature_proxy_2d_deg:.1f} deg",
                          (30, 280), 0.8, (255, 150, 0))

            cv2.imshow(WIN, img_rec)

            # ---------- Keys ----------
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key in (ord('s'), ord('S')) and mode == "IDLE":
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
