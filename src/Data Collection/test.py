# ============================================================
# Kidzonia School Screening – QUEST-AWARE IMAGE COLLECTION
# Frame-level image + feature logging (CSV + JSON)
# Non-diagnostic, screening-safe features only
# ============================================================

from openni import openni2
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
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_ROOT = os.path.join(_PROJECT_ROOT, "Data", "Raw", "spine_image_dataset")

QUESTS = [
    ("Q1_standing", "Stand naturally, arms relaxed."),
    ("Q2_forward_bend", "Bend forward with arms hanging."),
    ("Q3_upward_stretch", "Stand straight and stretch upwards with your arms.")
]

DISPLAY_SCALE = 2.0
ANGLE_SMOOTH_WINDOW = 10
# --------------------------------------


# ---------------- UTILS ----------------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

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

def draw_text(img, text, org, scale, color):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, 2, cv2.LINE_AA)
# --------------------------------------


def main():
    print("=== Kidzonia Spine IMAGE Collection ===")
    subject_id = input("Enter subject_id: ").strip() or "S001"
    session_id = input("Enter session_id: ").strip() or "session1"

    session_dir = os.path.join(OUTPUT_ROOT, f"{subject_id}_{session_id}")
    ensure_dir(session_dir)

    csv_path = os.path.join(session_dir, "features.csv")
    json_path = os.path.join(session_dir, "features.json")

    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)

    # ---- EXACT FEATURE SCHEMA (as agreed) ----
    csv_writer.writerow([
        "image_file",
        "timestamp_epoch",
        "subject_id",
        "session_id",
        "quest_id",
        "pose_present",
        "pose_confidence",
        "delta_shoulder_height_px",
        "delta_hip_height_px",
        "standing_midline_x_offset_px",
        "standing_symmetry_index",
        "rib_hump_proxy_px",
        "surface_asymmetry_score",
        "atr_proxy_deg",
        "atr_exceeds_5deg",
        "trunk_inclination_2d_deg",
        "trunk_inclination_3d_deg",
        "trunk_segment_angle_upper_2d_deg",
        "trunk_segment_angle_lower_2d_deg",
        "trunk_curvature_proxy_2d_deg",
    ])

    json_records = []

    # -------- MediaPipe --------
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    draw = mp.solutions.drawing_utils

    # -------- OpenNI --------
    openni2.initialize()
    dev = openni2.Device.open_any()
    color = dev.create_color_stream()
    color.start()

    WIN = "Kidzonia – Image Capture (Quest-based)"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 720)

    quest_idx = 0
    quest_id, quest_prompt = QUESTS[quest_idx]

    buf_2d, buf_curv = [], []
    capture_idx = 0

    try:
        while True:
            frame = color.read_frame()
            img = np.frombuffer(
                frame.get_buffer_as_uint8(), dtype=np.uint8
            ).reshape((frame.height, frame.width, 3))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            h, w, _ = img.shape
            img_disp = cv2.resize(img, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)

            now = time.time()

            result = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            pose_present = result.pose_landmarks is not None
            pose_confidence = 1.0 if pose_present else 0.0

            # ---- initialise features ----
            delta_sh = delta_hip = mid_offset = sym_index = None
            rib_hump = surface_asym = atr = atr_flag = None
            t2d = t3d = tu = tl = curv = None

            if pose_present:
                draw.draw_landmarks(img_disp, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                lm = result.pose_landmarks.landmark

                ls = (lm[11].x * w, lm[11].y * h)
                rs = (lm[12].x * w, lm[12].y * h)
                lh = (lm[23].x * w, lm[23].y * h)
                rh = (lm[24].x * w, lm[24].y * h)

                spine_top = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
                spine_bot = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)
                mid_torso = ((spine_top[0] + spine_bot[0]) / 2,
                             (spine_top[1] + spine_bot[1]) / 2)

                # ---------- Q1 ----------
                if quest_id == "Q1_standing":
                    delta_sh = abs(ls[1] - rs[1])
                    delta_hip = abs(lh[1] - rh[1])
                    mid_offset = abs(((ls[0] + rs[0]) / 2) - ((lh[0] + rh[0]) / 2))
                    sym_index = (delta_sh + delta_hip + mid_offset) / (h * 3)
                    t2d = update_buffer(buf_2d, trunk_inclination_deg(spine_top, spine_bot))

                # ---------- Q2 ----------
                elif quest_id == "Q2_forward_bend":
                    tu = trunk_inclination_deg(spine_top, mid_torso)
                    tl = trunk_inclination_deg(mid_torso, spine_bot)
                    curv = update_buffer(buf_curv, abs(tu - tl))
                    atr = curv
                    atr_flag = int(atr is not None and atr >= 5.0)
                    surface_asym = curv
                    t2d = update_buffer(buf_2d, trunk_inclination_deg(spine_top, spine_bot))

                # ---------- Q3 ----------
                elif quest_id == "Q3_upward_stretch":
                    mid_offset = abs(((ls[0] + rs[0]) / 2) - ((lh[0] + rh[0]) / 2))
                    t2d = update_buffer(buf_2d, trunk_inclination_deg(spine_top, spine_bot))

                # draw spine
                cv2.line(
                    img_disp,
                    (int(spine_top[0] * DISPLAY_SCALE), int(spine_top[1] * DISPLAY_SCALE)),
                    (int(spine_bot[0] * DISPLAY_SCALE), int(spine_bot[1] * DISPLAY_SCALE)),
                    (0, 255, 255), 4
                )

            # ---------- UI ----------
            overlay = img_disp.copy()
            cv2.rectangle(overlay, (0, 0), (img_disp.shape[1], 180), (0, 0, 0), -1)
            img_disp[:] = cv2.addWeighted(overlay, 0.35, img_disp, 0.65, 0)

            draw_text(img_disp, f"Subject: {subject_id}  Session: {session_id}",
                      (30, 45), 0.9, (255, 255, 255))
            draw_text(img_disp, f"Quest: {quest_id}",
                      (30, 90), 0.8, (200, 200, 255))
            draw_text(img_disp, f"Instruction: {quest_prompt}",
                      (30, 130), 0.7, (220, 220, 220))
            draw_text(img_disp,
                      "[C] Capture   [N] Next Quest   [B] Back   [ESC] Quit",
                      (30, 170), 0.7, (0, 255, 255))

            if t2d is not None:
                draw_text(img_disp, f"Trunk tilt (2D): {t2d:.1f} deg",
                          (30, 240), 0.9, (0, 255, 0))
            if quest_id == "Q2_forward_bend" and curv is not None:
                draw_text(img_disp, f"Curvature proxy: {curv:.1f} deg",
                          (30, 280), 0.9, (255, 180, 0))

            cv2.imshow(WIN, img_disp)

            # ---------- Keys ----------
            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                break

            if key in (ord('n'), ord('N')):
                quest_idx = clamp(quest_idx + 1, 0, len(QUESTS) - 1)
                quest_id, quest_prompt = QUESTS[quest_idx]

            if key in (ord('b'), ord('B')):
                quest_idx = clamp(quest_idx - 1, 0, len(QUESTS) - 1)
                quest_id, quest_prompt = QUESTS[quest_idx]

            if key in (ord('c'), ord('C')):
                if not pose_present:
                    print("⚠ Capture skipped (no pose detected).")
                    continue

                ts = time.time()
                image_name = f"{subject_id}_{session_id}_{quest_id}_{capture_idx:05d}.jpg"
                image_path = os.path.join(session_dir, image_name)
                cv2.imwrite(image_path, img)

                row = [
                    image_name, ts, subject_id, session_id, quest_id,
                    int(pose_present), pose_confidence,
                    delta_sh, delta_hip, mid_offset, sym_index,
                    rib_hump, surface_asym, atr, atr_flag,
                    t2d, t3d, tu, tl, curv
                ]
                csv_writer.writerow(row)
                csv_file.flush()

                json_records.append(dict(zip(csv_writer.writerow.__self__.fieldnames if False else [
                    "image_file","timestamp_epoch","subject_id","session_id","quest_id",
                    "pose_present","pose_confidence",
                    "delta_shoulder_height_px","delta_hip_height_px",
                    "standing_midline_x_offset_px","standing_symmetry_index",
                    "rib_hump_proxy_px","surface_asymmetry_score",
                    "atr_proxy_deg","atr_exceeds_5deg",
                    "trunk_inclination_2d_deg","trunk_inclination_3d_deg",
                    "trunk_segment_angle_upper_2d_deg","trunk_segment_angle_lower_2d_deg",
                    "trunk_curvature_proxy_2d_deg"
                ], row)))

                capture_idx += 1
                print(f"[SAVED] {image_name}")

    finally:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_records, f, indent=2)

        csv_file.close()
        color.stop()
        openni2.unload()
        cv2.destroyAllWindows()
        print("Image session finished. Data saved.")


if __name__ == "__main__":
    main()
