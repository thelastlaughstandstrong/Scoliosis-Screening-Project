import os, csv, json
import pandas as pd

RECORDINGS_ROOT = "raw_data"
OUTPUT_DATASET = "processed_data/dataset_v1.csv"

REQUIRED_COLUMNS = [
    "subject_id", "quest_id", "pose_present",
    "atr_proxy_deg", "trunk_curvature_proxy_2d_deg", "trunk_inclination_2d_deg",
]

def validate_trial(df):
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            return False
    if df["pose_present"].sum() == 0:
        return False
    return True

rows = []

for subject in os.listdir(RECORDINGS_ROOT):
    subj_path = os.path.join(RECORDINGS_ROOT, subject)
    if not os.path.isdir(subj_path):
        continue

    for quest in os.listdir(subj_path):
        quest_path = os.path.join(subj_path, quest)
        for trial in os.listdir(quest_path):
            trial_path = os.path.join(quest_path, trial)
            csv_path = os.path.join(trial_path, "features.csv")

            if not os.path.exists(csv_path):
                continue

            df = pd.read_csv(csv_path)

            if not validate_trial(df):
                print(f"[SKIP] {trial_path}")
                continue

            agg = {
                "subject_id": df["subject_id"].iloc[0],
                "quest_id": df["quest_id"].iloc[0],
                "atr_proxy_deg_mean": df["atr_proxy_deg"].mean(),
                "atr_proxy_deg_max": df["atr_proxy_deg"].max(),
                "trunk_curvature_mean": df["trunk_curvature_proxy_2d_deg"].mean(),
                "trunk_tilt_mean": df["trunk_inclination_2d_deg"].mean(),
                "pose_coverage": df["pose_present"].mean()
            }
            rows.append(agg)

os.makedirs("processed_data", exist_ok=True)
pd.DataFrame(rows).to_csv(OUTPUT_DATASET, index=False)
print("✅ processed_data/dataset_v1.csv generated")
