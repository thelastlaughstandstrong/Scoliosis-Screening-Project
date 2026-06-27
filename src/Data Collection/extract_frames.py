# ============================================================
# Frame Extractor — pulls candidate images from recorded videos
# for MediaPipe pseudo-labeling.
# Extracts 1 frame per second by default to avoid near-duplicate
# frames while capturing pose variation across the recording.
# ============================================================

import cv2
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT   = Path(__file__).resolve().parents[2]
DEFAULT_INPUT  = PROJECT_ROOT / "Data" / "Raw" / "raw_data"
DEFAULT_OUTPUT = PROJECT_ROOT / "Data" / "Raw" / "pseudo_label_candidates"
FRAMES_PER_SEC = 1   # how many frames to extract per second of video


def extract_from_video(video_path: Path, output_dir: Path, fps_extract: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [WARN] Cannot open: {video_path.name}")
        return 0

    video_fps  = cap.get(cv2.CAP_PROP_FPS) or 30
    interval   = max(1, int(video_fps / fps_extract))
    frame_idx  = 0
    saved      = 0

    # Tag output filenames with subject/quest/trial from folder structure
    # Expected: .../raw_data/{subject_id}/{quest_id}/{trial_id}/video.avi
    parts   = video_path.parts
    try:
        tag = f"{parts[-4]}_{parts[-3]}_{parts[-2]}"
    except IndexError:
        tag = video_path.stem

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0:
            out_path = output_dir / f"{tag}_f{frame_idx:05d}.jpg"
            cv2.imencode(".jpg", frame)[1].tofile(str(out_path))
            saved += 1
        frame_idx += 1

    cap.release()
    return saved


def run(input_root: Path = DEFAULT_INPUT,
        output_dir: Path = DEFAULT_OUTPUT,
        fps_extract: int = FRAMES_PER_SEC):

    output_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(input_root.rglob("video.avi"))
    if not videos:
        print(f"[WARN] No video.avi files found under {input_root}")
        return

    total_saved = 0
    for v in videos:
        n = extract_from_video(v, output_dir, fps_extract)
        print(f"  {v.parent.relative_to(input_root)}: {n} frames extracted")
        total_saved += n

    print(f"\n[DONE] {total_saved} frames saved to: {output_dir.name}/")
    print(f"       Run pseudo_label_mediapipe.py next to generate YOLO labels.")


if __name__ == "__main__":
    input_root = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INPUT
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUTPUT
    run(input_root, output_dir)