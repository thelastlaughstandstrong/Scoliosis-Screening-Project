# ============================================================
# Temporal Stability Analysis
# Measures frame-to-frame reliability of pose measurements
# over a real 5-second recording (features.csv from a trial).
#
# Why this matters: static image tests show whether a model can
# detect keypoints. Temporal stability shows whether measurements
# are reliable enough to act on across a real capture session.
# ============================================================

import pandas as pd
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Primary angle column to analyse — change to spine_angle_deg when
# YOLO-derived features are added to the capture pipeline
ANGLE_COL        = "trunk_inclination_2d_deg"
SHORT_WINDOW_SEC = 1.0   # rolling window for short-term stability
FPS              = 30


def analyse(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)

    if ANGLE_COL not in df.columns:
        raise ValueError(f"Column '{ANGLE_COL}' not found in {csv_path.name}")

    total_frames    = len(df)
    pose_coverage   = df["pose_present"].mean() if "pose_present" in df.columns else None
    valid_df        = df[df[ANGLE_COL].notna()].copy()
    valid_frames    = len(valid_df)

    if valid_frames < 2:
        return {
            "trial":              csv_path.parent.name,
            "total_frames":       total_frames,
            "valid_frames":       valid_frames,
            "pose_coverage":      pose_coverage,
            "note":               "Too few valid frames for stability analysis.",
        }

    angles = valid_df[ANGLE_COL].values

    # Frame-to-frame absolute difference
    frame_diffs       = np.abs(np.diff(angles))
    mean_frame_diff   = round(float(frame_diffs.mean()), 4)
    max_frame_diff    = round(float(frame_diffs.max()),  4)

    # Overall spread
    angle_mean  = round(float(angles.mean()), 4)
    angle_std   = round(float(angles.std()),  4)
    angle_range = round(float(angles.max() - angles.min()), 4)

    # Rolling short-window std (short-term jitter)
    window_frames = max(2, int(SHORT_WINDOW_SEC * FPS))
    rolling_std   = (
        pd.Series(angles)
        .rolling(window=window_frames, min_periods=2)
        .std()
        .dropna()
        .values
    )
    mean_rolling_std = round(float(rolling_std.mean()), 4) if len(rolling_std) else None

    # Stability verdict
    # Stable if: mean frame-to-frame diff < 1 deg AND rolling std < 2 deg
    is_stable = mean_frame_diff < 1.0 and (mean_rolling_std is None or mean_rolling_std < 2.0)

    return {
        "trial":                  csv_path.parent.name,
        "quest_id":               df["quest_id"].iloc[0] if "quest_id" in df.columns else None,
        "subject_id":             df["subject_id"].iloc[0] if "subject_id" in df.columns else None,
        "total_frames":           total_frames,
        "valid_frames":           valid_frames,
        "pose_coverage_ratio":    round(float(pose_coverage), 4) if pose_coverage is not None else None,
        "angle_mean_deg":         angle_mean,
        "angle_std_deg":          angle_std,
        "angle_range_deg":        angle_range,
        "mean_frame_diff_deg":    mean_frame_diff,
        "max_frame_diff_deg":     max_frame_diff,
        f"mean_rolling_std_{SHORT_WINDOW_SEC}s_deg": mean_rolling_std,
        "measurement_stable":     is_stable,
        "note": (
            "Stable — measurements consistent across capture." if is_stable
            else "Unstable — high frame-to-frame variance. Check clothing, lighting, or subject movement."
        ),
    }


def analyse_all_trials(raw_data_root: Path) -> pd.DataFrame:
    rows = []
    for csv_path in sorted(raw_data_root.rglob("features.csv")):
        try:
            rows.append(analyse(csv_path))
        except Exception as e:
            print(f"[WARN] {csv_path}: {e}")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        # Single trial CSV or raw_data root
        target = Path(sys.argv[1])
        if target.name == "features.csv":
            result = analyse(target)
            for k, v in result.items():
                print(f"  {k:<40}: {v}")
        else:
            df = analyse_all_trials(target)
            if df.empty:
                print("[WARN] No features.csv files found.")
            else:
                print(df.to_string(index=False))
                out = target / "temporal_stability_report.csv"
                df.to_csv(out, index=False)
                print(f"\n[DONE] Report saved to: {out.name}")
    else:
        default_root = PROJECT_ROOT / "Data" / "Raw" / "raw_data"
        print(f"Usage: python temporal_stability.py <features.csv | raw_data_root>")
        print(f"Default root would be: {default_root}")
        sys.exit(1)