# ============================================================
# Kidzonia – OpenNI Image Capture (Feature-Compatible)
# Saves images + CSV + JSON
# ============================================================

from openni import openni2
from openni import _openni2 as c_api
import cv2
import mediapipe as mp
import numpy as np
import math
import os
import time
import csv
import json

# ---------------- CONFIG ----------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_ROOT = os.path.join(_PROJECT_ROOT, "Data", "Raw", "spine_image_dataset")

QUESTS = [
    ("Q1_standing", "Stand naturally, arms relaxed."),
    ("Q2_forward_bend", "Bend forward with arms hanging."),
    ("Q3_upward_stretch", "Stand straight and stretch upwards.")
]

DISPLAY_SCALE = 1.6
FONT_SCALE = 0.7
FONT_THICKNESS = 2
# ---------------------------------------

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def angle_deg(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.degrees(math.atan2(dx, dy))

def draw_text(img, text, x, y, color=(255,255,255)):
    cv2.putText(img, text, (x,y),
                cv2.FONT_HERSHEY_SIMPLEX,
                FONT_SCALE, (0,0,0),
                FONT_THICKNESS+2, cv2.LINE_AA)
    cv2.putText(img, text, (x,y),
                cv2.FONT_HERSHEY_SIMPLEX,
                FONT_SCALE, color,
                FONT_THICKNESS, cv2.LINE_AA)

# ---------------- MediaPipe ----------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# ---------------- Main ----------------
def main():
    subject_id = input("Enter subject_id: ").strip() or "S001"
    session_id = input("Enter session_id: ").strip() or "session1"

    session_dir = os.path.join(OUTPUT_ROOT, f"{subject_id}_{session_id}")
    image_dir = os.path.join(session_dir, "images")
    ensure_dir(image_dir)

    csv_path = os.path.join(session_dir, "features.csv")
    json_path = os.path.join(session_dir, "features.json")

    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)

    header = [
        "image_file",
        "timestamp_epoch","t_rel_s","subject_id","quest_id",
        "pose_present","pose_confidence",
        "delta_shoulder_height_px","delta_hip_height_px",
        "standing_midline_x_offset_px","standing_symmetry_index",
        "rib_hump_proxy_px","surface_asymmetry_score",
        "atr_proxy_deg","atr_exceeds_5deg",
        "trunk_inclination_2d_deg","trunk_inclination_3d_deg",
        "trunk_segment_angle_upper_2d_deg","trunk_segment_angle_lower_2d_deg",
        "trunk_curvature_proxy_2d_deg"
    ]
    csv_writer.writerow(header)
    json_records = []

    # -------- OpenNI RGB --------
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

    # -------- UI --------
    WIN = "Kidzonia – Image Feature Capture (OpenNI)"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 720)

    quest_idx = 0
    quest_id, quest_prompt = QUESTS[quest_idx]
    start_time = time.time()
    img_index = 0

    try:
        while True:
            frame = color.read_frame()
            img = np.frombuffer(
                frame.get_buffer_as_uint8(), dtype=np.uint8
            ).reshape((frame.height, frame.width, 3))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            h, w, _ = img.shape
            now = time.time()

            result = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            pose_present = result.pose_landmarks is not None
            pose_confidence = 1.0 if pose_present else 0.0

            # ---------- init features ----------
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
                lm = result.pose_landmarks.landmark
                mp_draw.draw_landmarks(img, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                ls = (lm[11].x * w, lm[11].y * h)
                rs = (lm[12].x * w, lm[12].y * h)
                lh = (lm[23].x * w, lm[23].y * h)
                rh = (lm[24].x * w, lm[24].y * h)

                spine_top = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
                spine_bot = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)
                mid_torso = ((spine_top[0] + spine_bot[0]) / 2,
                             (spine_top[1] + spine_bot[1]) / 2)

                trunk_inclination_2d_deg = angle_deg(spine_top, spine_bot)

                if quest_id == "Q1_standing":
                    delta_shoulder_height_px = abs(ls[1] - rs[1])
                    delta_hip_height_px = abs(lh[1] - rh[1])
                    standing_midline_x_offset_px = abs(
                        (ls[0] + rs[0]) / 2 - (lh[0] + rh[0]) / 2
                    )
                    standing_symmetry_index = (
                        delta_shoulder_height_px +
                        delta_hip_height_px +
                        standing_midline_x_offset_px
                    ) / (h * 3)

                elif quest_id == "Q2_forward_bend":
                    trunk_segment_angle_upper_2d_deg = angle_deg(spine_top, mid_torso)
                    trunk_segment_angle_lower_2d_deg = angle_deg(mid_torso, spine_bot)
                    trunk_curvature_proxy_2d_deg = abs(
                        trunk_segment_angle_upper_2d_deg -
                        trunk_segment_angle_lower_2d_deg
                    )
                    atr_proxy_deg = trunk_curvature_proxy_2d_deg
                    atr_exceeds_5deg = int(atr_proxy_deg >= 5)

                elif quest_id == "Q3_upward_stretch":
                    standing_midline_x_offset_px = abs(
                        (ls[0] + rs[0]) / 2 - (lh[0] + rh[0]) / 2
                    )

            # ---------- Display ----------
            draw_text(img, f"Subject: {subject_id}   Quest: {quest_id}", 30, 40)
            draw_text(img, quest_prompt, 30, 80, (200,200,200))
            draw_text(img, "[C] Capture   [N] Next Quest   [ESC] Quit", 30, h-30)

            img_disp = cv2.resize(img, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
            cv2.imshow(WIN, img_disp)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key in (ord('n'), ord('N')):
                quest_idx = (quest_idx + 1) % len(QUESTS)
                quest_id, quest_prompt = QUESTS[quest_idx]
            if key in (ord('c'), ord('C')):
                image_name = f"img_{img_index:05d}.jpg"
                image_path = os.path.join(image_dir, image_name)
                cv2.imwrite(image_path, img)

                row = [
                    image_name,
                    now, now - start_time, subject_id, quest_id,
                    int(pose_present), pose_confidence,
                    delta_shoulder_height_px, delta_hip_height_px,
                    standing_midline_x_offset_px, standing_symmetry_index,
                    rib_hump_proxy_px, surface_asymmetry_score,
                    atr_proxy_deg, atr_exceeds_5deg,
                    trunk_inclination_2d_deg, trunk_inclination_3d_deg,
                    trunk_segment_angle_upper_2d_deg, trunk_segment_angle_lower_2d_deg,
                    trunk_curvature_proxy_2d_deg
                ]
                csv_writer.writerow(row)
                json_records.append(dict(zip(header, row)))
                csv_file.flush()

                img_index += 1
                print(f"Saved {image_name}")

    finally:
        color.stop()
        openni2.unload()
        csv_file.close()
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_records, f, indent=2)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
