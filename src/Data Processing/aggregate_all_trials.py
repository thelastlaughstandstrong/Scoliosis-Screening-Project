import pandas as pd
import json
import numpy as np
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_INPUT  = _PROJECT_ROOT / "Data" / "Raw" / "raw_data"
_DEFAULT_OUTPUT = _PROJECT_ROOT / "Data" / "Processed" / "aggregated_trials.csv"


# ----------------- Single-trial aggregation -----------------

def aggregate_trial(csv_path: Path):
    df = pd.read_csv(csv_path)

    out = {}

    # ---- identifiers ----
    out["subject_id"] = df["subject_id"].iloc[0]
    out["quest_id"] = df["quest_id"].iloc[0]
    out["trial_id"] = csv_path.parent.name

    # ---- pose quality ----
    out["num_frames"] = len(df)
    out["pose_coverage_ratio"] = df["pose_present"].mean()

    # ---- standing symmetry (Q1 / Q3) ----
    if "standing_symmetry_index" in df:
        out["standing_symmetry_median"] = df["standing_symmetry_index"].median()
        out["standing_midline_offset_p95_px"] = df["standing_midline_x_offset_px"].quantile(0.95)
    else:
        out["standing_symmetry_median"] = None
        out["standing_midline_offset_p95_px"] = None

    # ---- forward bend (Q2) ----
    if "trunk_curvature_proxy_2d_deg" in df:
        out["trunk_curvature_p95_deg"] = df["trunk_curvature_proxy_2d_deg"].quantile(0.95)
        out["atr_proxy_p95_deg"] = df["atr_proxy_deg"].quantile(0.95)
        out["rib_hump_proxy_p95_px"] = df["rib_hump_proxy_px"].quantile(0.95)
    else:
        out["trunk_curvature_p95_deg"] = None
        out["atr_proxy_p95_deg"] = None
        out["rib_hump_proxy_p95_px"] = None

    # ---- global trunk ----
    out["trunk_inclination_2d_median_deg"] = df["trunk_inclination_2d_deg"].median()
    out["trunk_inclination_2d_p95_deg"] = df["trunk_inclination_2d_deg"].quantile(0.95)

    # ---- timestamps ----
    out["timestamp_start"] = df["timestamp_epoch"].min()
    out["timestamp_end"] = df["timestamp_epoch"].max()

    return out


# ----------------- Batch aggregation -----------------

def aggregate_all(recordings_root: Path):
    rows = []

    for subject_dir in sorted(recordings_root.iterdir()):
        if not subject_dir.is_dir():
            continue

        for quest_dir in sorted(subject_dir.iterdir()):
            if not quest_dir.is_dir():
                continue

            for trial_dir in sorted(quest_dir.iterdir()):
                if not trial_dir.is_dir():
                    continue

                csv_path = trial_dir / "features.csv"
                if not csv_path.exists():
                    continue

                try:
                    row = aggregate_trial(csv_path)
                    rows.append(row)
                except Exception as e:
                    print(f"[WARN] Failed to process {csv_path}: {e}")

    return pd.DataFrame(rows)


# ----------------- Main -----------------

if __name__ == "__main__":
    if len(sys.argv) == 1:
        recordings_root = _DEFAULT_INPUT
        output_csv = _DEFAULT_OUTPUT
    elif len(sys.argv) == 3:
        recordings_root = Path(sys.argv[1])
        output_csv = Path(sys.argv[2])
    else:
        print("Usage: python aggregate_all_trials.py [<recordings_root> <output_csv>]")
        sys.exit(1)

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    df = aggregate_all(recordings_root)

    df.to_csv(output_csv, index=False)
    print(f"[DONE] Aggregated dataset saved to: {output_csv}")
    print(f"       Rows: {len(df)}")
