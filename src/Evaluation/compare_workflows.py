# ============================================================
# Workflow Comparison — MediaPipe vs YOLO Pretrained vs YOLO Fine-tuned
# Reads output CSVs and prints a 3-way summary table
#
# Run order:
#   1. run_evaluation.py --model mediapipe
#   2. run_evaluation.py --model yolo_pretrained
#   3. train_yolo_scoliosis.py
#   4. run_evaluation.py --model yolo_finetuned
#   5. compare_workflows.py  (this script)
# ============================================================

import pandas as pd
from pathlib import Path

PROJECT_ROOT   = Path(__file__).resolve().parents[2]
COMPARISON_DIR = PROJECT_ROOT / "Data" / "Processed" / "comparison"

CSVS = {
    "MediaPipe":          COMPARISON_DIR / "mediapipe_scoliosis_test.csv",
    "YOLO (pretrained)":  COMPARISON_DIR / "yolo_scoliosis_test.csv",
    "YOLO (fine-tuned)":  COMPARISON_DIR / "yolo_finetuned_scoliosis_test.csv",
}


def summarise(df, label):
    total    = len(df)
    valid    = df["valid_measurement"].sum()
    retakes  = df["retake_recommended"].sum()
    avg_conf = df["mean_confidence"].mean()
    avg_miss = df["missing_keypoint_count"].mean()
    avg_time = df["processing_time_s"].mean()

    n_rows = df[df["ground_truth_class"] == "normal"]
    s_rows = df[df["ground_truth_class"] == "scoliosis"]

    # Pose detection rate per class (was a valid measurement obtained?)
    n_det_rate = n_rows["valid_measurement"].mean() if len(n_rows) else None
    s_det_rate = s_rows["valid_measurement"].mean() if len(s_rows) else None

    # Classification accuracy — only available for fine-tuned YOLO
    cls_acc      = None
    scoli_recall = None
    if "classification_correct" in df.columns:
        cls_rows = df[df["classification_correct"].notna()]
        if len(cls_rows):
            cls_acc = cls_rows["classification_correct"].mean()
        scoli_cls = df[
            (df["ground_truth_class"] == "scoliosis") & df["predicted_class"].notna()
        ]
        if len(scoli_cls):
            scoli_recall = (scoli_cls["predicted_class"] == "scoliosis").mean()

    # Spine geometry — only present for the fine-tuned YOLO workflow (spine_geometry.py output)
    geometry_valid_pct              = None
    mean_max_lateral_deviation_px   = None
    mean_curve_smoothness_score     = None
    mean_spine_axis_inclination_deg = None
    if "valid_spine_geometry" in df.columns:
        geometry_valid_pct = round(100 * df["valid_spine_geometry"].mean(), 1)
    if "max_lateral_deviation_px" in df.columns:
        vals = df["max_lateral_deviation_px"].dropna()
        mean_max_lateral_deviation_px = round(vals.mean(), 2) if len(vals) else None
    if "curve_smoothness_score" in df.columns:
        vals = df["curve_smoothness_score"].dropna()
        mean_curve_smoothness_score = round(vals.mean(), 2) if len(vals) else None
    if "spine_axis_inclination_deg" in df.columns:
        vals = df["spine_axis_inclination_deg"].dropna()
        mean_spine_axis_inclination_deg = round(vals.mean(), 2) if len(vals) else None

    return {
        "workflow":           label,
        "valid_pct":          round(100 * valid / total, 1),
        "retake_pct":         round(100 * retakes / total, 1),
        "mean_confidence":    round(avg_conf, 3),
        "mean_missing_kp":    round(avg_miss, 2),
        "mean_time_ms":       round(avg_time * 1000, 1),
        "normal_det_rate":    round(n_det_rate, 3) if n_det_rate is not None else None,
        "scoliosis_det_rate": round(s_det_rate, 3) if s_det_rate is not None else None,
        "classification_acc": round(cls_acc, 3) if cls_acc is not None else None,
        "scoliosis_recall":   round(scoli_recall, 3) if scoli_recall is not None else None,
        "geometry_valid_pct":              geometry_valid_pct,
        "mean_max_lateral_deviation_px":   mean_max_lateral_deviation_px,
        "mean_curve_smoothness_score":     mean_curve_smoothness_score,
        "mean_spine_axis_inclination_deg": mean_spine_axis_inclination_deg,
    }


def fmt(val, unit=""):
    if val is None:
        return "N/A"
    if isinstance(val, float):
        return f"{val:.3f}{unit}"
    return f"{val}{unit}"


def run():
    # Load whichever CSVs exist
    frames = {}
    missing = []
    for label, path in CSVS.items():
        if path.exists():
            frames[label] = pd.read_csv(path)
        else:
            missing.append(label)

    if missing:
        print(f"[WARN] Missing results for: {', '.join(missing)}")
        print(f"       Run the corresponding evaluation scripts first.")

    if not frames:
        print("[ERROR] No results found. Nothing to compare.")
        return

    stats = {label: summarise(df, label) for label, df in frames.items()}

    # ---- Header ----
    sample_df = next(iter(frames.values()))
    n_total  = len(sample_df)
    n_normal = (sample_df["ground_truth_class"] == "normal").sum()
    n_scoli  = (sample_df["ground_truth_class"] == "scoliosis").sum()

    print(f"\n{'='*70}")
    print(f"  3-WAY WORKFLOW COMPARISON")
    print(f"  Test set: {n_total} images ({n_normal} normal, {n_scoli} scoliosis)")
    print(f"  WARNING: {n_scoli} scoliosis cases only — treat scoliosis metrics as indicative.")
    print(f"{'='*70}")

    # ---- Per-workflow summary ----
    metrics = [
        ("Valid measurements",         "valid_pct",          "%"),
        ("Retake recommended",          "retake_pct",         "%"),
        ("Mean keypoint confidence",    "mean_confidence",    ""),
        ("Mean missing keypoints",      "mean_missing_kp",    ""),
        ("Mean processing time",        "mean_time_ms",       " ms"),
        ("Normal detection rate",       "normal_det_rate",    ""),
        ("Scoliosis detection rate",    "scoliosis_det_rate", ""),
        ("Classification accuracy",     "classification_acc", "  [fine-tuned only]"),
        ("Scoliosis recall (cls)",      "scoliosis_recall",   "  [fine-tuned only]"),
    ]

    col_labels = list(stats.keys())
    col_width  = 22

    header = f"  {'Metric':<32}" + "".join(f"{c:<{col_width}}" for c in col_labels)
    print(f"\n{header}")
    print(f"  {'-'*32}" + "-" * (col_width * len(col_labels)))

    for name, key, unit in metrics:
        row = f"  {name:<32}"
        for label in col_labels:
            val = stats[label].get(key)
            row += f"{fmt(val, unit):<{col_width}}"
        print(row)

    # ---- Delta table (vs MediaPipe baseline) ----
    if "MediaPipe" in stats and len(stats) > 1:
        print(f"\n  DELTA vs MediaPipe baseline")
        print(f"  {'-'*32}" + "-" * (col_width * len(col_labels)))
        other_labels = [l for l in col_labels if l != "MediaPipe"]
        delta_metrics = [m for m in metrics if m[2] not in ("  [fine-tuned only]",)]

        for name, key, unit in delta_metrics:
            base = stats["MediaPipe"].get(key)
            row  = f"  {name:<32}"
            row += f"{'(baseline)':<{col_width}}"
            for label in other_labels:
                val = stats[label].get(key)
                if base is None or val is None:
                    row += f"{'N/A':<{col_width}}"
                else:
                    delta = val - base
                    sign  = "+" if delta >= 0 else ""
                    row  += f"{sign}{delta:.2f}{unit:<{col_width - len(f'{sign}{delta:.2f}')}}".ljust(col_width)
            print(row)

    # ---- Spine geometry summary (fine-tuned YOLO only — see spine_geometry.py) ----
    geom_labels = [l for l in col_labels if stats[l].get("geometry_valid_pct") is not None]
    if geom_labels:
        geom_metrics = [
            ("Spine geometry valid",         "geometry_valid_pct",              "%"),
            ("Mean max lateral deviation",   "mean_max_lateral_deviation_px",   " px"),
            ("Mean curve smoothness score",  "mean_curve_smoothness_score",     ""),
            ("Mean spine axis inclination",  "mean_spine_axis_inclination_deg", " deg"),
        ]
        print(f"\n  SPINE GEOMETRY  (fine-tuned YOLO only — screening-support proxies, not clinical ATR/Cobb)")
        print(f"  {'-'*32}" + "-" * (col_width * len(geom_labels)))
        header2 = f"  {'Metric':<32}" + "".join(f"{c:<{col_width}}" for c in geom_labels)
        print(header2)
        for name, key, unit in geom_metrics:
            row = f"  {name:<32}"
            for label in geom_labels:
                val = stats[label].get(key)
                row += f"{fmt(val, unit):<{col_width}}"
            print(row)

    # ---- Notes on angle comparability ----
    print(f"\n  NOTES ON ANGLE COMPARABILITY:")
    print(f"  - MediaPipe primary_angle: shoulder-midpoint to hip-midpoint (body axis proxy)")
    print(f"  - YOLO pretrained:         same landmarks (COCO kp 5,6,11,12)")
    print(f"  - YOLO fine-tuned:         kp[0] (lumbar) to kp[8] (cervical) — direct spine axis")
    print(f"  Primary angles are NOT directly comparable across workflows.")
    print(f"  Use detection rate, confidence, and classification accuracy for comparison.")

    # ---- Save summary CSV ----
    out_path = COMPARISON_DIR / "comparison_summary.csv"
    pd.DataFrame(list(stats.values())).to_csv(out_path, index=False)
    print(f"\n  Summary saved to: {out_path.name} (in Data/Processed/comparison/)")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    run()