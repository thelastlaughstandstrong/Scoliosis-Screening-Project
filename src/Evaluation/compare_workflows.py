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
#
# By default it reads the legacy-named CSVs (old public test split) and writes
# comparison_summary.csv. To compare results from another dataset/split, such as
# the grouped split, pass the CSVs explicitly:
#   python compare_workflows.py --finetuned-csv <csv> [--mediapipe-csv <csv>] [--pretrained-csv <csv>] #       --out-name grouped
# It checks each CSV's .meta.json sidecar (written by run_evaluation.py) and warns if the
# results come from different test sets, or were produced by a model trained on the old split.
# ============================================================

import argparse
import sys
import pandas as pd
from pathlib import Path

PROJECT_ROOT   = Path(__file__).resolve().parents[2]
COMPARISON_DIR = PROJECT_ROOT / "Data" / "Processed" / "comparison"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.Evaluation.dataset_paths import read_meta, describe_eval, is_legacy_eval, OLD_SPLIT_WARNING

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

    # Anatomy-aware reliability layer — only present for the fine-tuned YOLO workflow
    # (src/FeatureExtraction/spine_geometry.py output). These describe measurement
    # quality, not disease severity — see that module's docstring.
    mean_reliability_score      = None
    median_reliability_score    = None
    reliability_high_count      = None
    reliability_medium_count    = None
    reliability_low_count       = None
    reliability_invalid_count   = None
    retake_or_review_pct        = None
    curvature_reliable_pct      = None
    if "anatomy_reliability_score" in df.columns:
        vals = df["anatomy_reliability_score"].dropna()
        mean_reliability_score   = round(vals.mean(), 3) if len(vals) else None
        median_reliability_score = round(vals.median(), 3) if len(vals) else None
    if "reliability_level" in df.columns:
        counts = df["reliability_level"].value_counts()
        reliability_high_count    = int(counts.get("high", 0))
        reliability_medium_count  = int(counts.get("medium", 0))
        reliability_low_count     = int(counts.get("low", 0))
        reliability_invalid_count = int(counts.get("invalid", 0))
    if "retake_or_review_recommended" in df.columns:
        retake_or_review_pct = round(100 * df["retake_or_review_recommended"].mean(), 1)
    if "curvature_features_reliable" in df.columns:
        curvature_reliable_pct = round(100 * df["curvature_features_reliable"].mean(), 1)

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
        "mean_reliability_score":          mean_reliability_score,
        "median_reliability_score":        median_reliability_score,
        "reliability_high_count":          reliability_high_count,
        "reliability_medium_count":        reliability_medium_count,
        "reliability_low_count":           reliability_low_count,
        "reliability_invalid_count":       reliability_invalid_count,
        "retake_or_review_pct":            retake_or_review_pct,
        "curvature_reliable_pct":          curvature_reliable_pct,
    }


def normal_vs_scoliosis(df, column):
    """Mean of `column`, split by ground-truth class — for spotting whether a
    geometry feature actually separates normal vs. scoliosis in this sample."""
    if column not in df.columns:
        return None, None
    n_vals = df.loc[df["ground_truth_class"] == "normal", column].dropna()
    s_vals = df.loc[df["ground_truth_class"] == "scoliosis", column].dropna()
    n_mean = round(n_vals.mean(), 4) if len(n_vals) else None
    s_mean = round(s_vals.mean(), 4) if len(s_vals) else None
    return n_mean, s_mean


def fmt(val, unit=""):
    if val is None:
        return "N/A"
    if isinstance(val, float):
        return f"{val:.3f}{unit}"
    return f"{val}{unit}"


def print_provenance(csvs, frames):
    """Say what each results CSV was evaluated on, and warn about mixed test sets / old-split models."""
    metas = {label: read_meta(csvs[label]) for label in frames}
    print("  Evaluated on:")
    for label, meta in metas.items():
        print(f"   - {label:<20} {describe_eval(meta)}")
    datasets = {(m or {}).get("eval_dataset", "legacy") + "/" + (m or {}).get("eval_split", "test") for m in metas.values()}
    if len(datasets) > 1:
        print("  [WARN] These results come from DIFFERENT test sets — they are not directly comparable.")
    if any((m or {}).get("old_split_warning") for m in metas.values()):
        print(f"  [WARN] {OLD_SPLIT_WARNING}")
    if all(is_legacy_eval(m) for m in metas.values()):
        print("  [NOTE] Old public test split: near-duplicate photos are shared with train/valid, so these numbers are "
              "historical/debug only once the grouped split exists.")


def run(csvs=None, out_name="summary"):
    csvs = csvs or CSVS
    # Load whichever CSVs exist
    frames = {}
    missing = []
    for label, path in csvs.items():
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
    print_provenance(csvs, frames)
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

    # ---- Anatomy-aware reliability summary (fine-tuned YOLO only) ----
    reliability_labels = [l for l in col_labels if stats[l].get("mean_reliability_score") is not None]
    if reliability_labels:
        print(f"\n  ANATOMY-AWARE RELIABILITY  (fine-tuned YOLO only — measurement quality, not diagnosis)")
        print(f"  {'-'*32}" + "-" * (col_width * len(reliability_labels)))
        rel_metrics = [
            ("Mean reliability score",       "mean_reliability_score",   ""),
            ("Median reliability score",     "median_reliability_score", ""),
            ("Retake/review recommended",    "retake_or_review_pct",     "%"),
            ("Curvature features reliable",  "curvature_reliable_pct",   "%"),
        ]
        header3 = f"  {'Metric':<32}" + "".join(f"{c:<{col_width}}" for c in reliability_labels)
        print(header3)
        for name, key, unit in rel_metrics:
            row = f"  {name:<32}"
            for label in reliability_labels:
                val = stats[label].get(key)
                row += f"{fmt(val, unit):<{col_width}}"
            print(row)

        print(f"\n  Reliability level counts (high / medium / low / invalid):")
        for label in reliability_labels:
            s = stats[label]
            counts_str = " / ".join(str(s.get(k, 0)) for k in (
                "reliability_high_count", "reliability_medium_count",
                "reliability_low_count", "reliability_invalid_count",
            ))
            print(f"    {label:<22}: {counts_str}")

    # ---- Normal vs. scoliosis feature comparison (fine-tuned YOLO only) ----
    if "YOLO (fine-tuned)" in frames:
        ft_df = frames["YOLO (fine-tuned)"]
        compare_cols = [
            ("normalized_max_lateral_deviation",  "Normalized max lateral deviation"),
            ("normalized_mean_lateral_deviation", "Normalized mean lateral deviation"),
            ("shoulder_height_difference_px",     "Shoulder height difference (px)"),
            ("spine_axis_inclination_deg",        "Spine axis inclination (deg)"),
            ("curve_smoothness_score",            "Curve smoothness score"),
        ]
        available = [(c, name) for c, name in compare_cols if c in ft_df.columns]
        if available:
            name_width = max(34, max(len(name) for _, name in available) + 2)
            print(f"\n  NORMAL vs. SCOLIOSIS  (fine-tuned YOLO only — feature means by ground-truth class)")
            print(f"  Sample sizes are small — read as indicative, not statistically confirmed.")
            print(f"  {'-'*name_width}{'-'*col_width}{'-'*col_width}")
            print(f"  {'Feature':<{name_width}}{'Normal':<{col_width}}{'Scoliosis':<{col_width}}")
            for col, name in available:
                n_mean, s_mean = normal_vs_scoliosis(ft_df, col)
                print(f"  {name:<{name_width}}{fmt(n_mean):<{col_width}}{fmt(s_mean):<{col_width}}")

    # ---- Notes on angle comparability ----
    print(f"\n  NOTES ON ANGLE COMPARABILITY:")
    print(f"  - MediaPipe primary_angle: shoulder-midpoint to hip-midpoint (body axis proxy)")
    print(f"  - YOLO pretrained:         same landmarks (COCO kp 5,6,11,12)")
    print(f"  - YOLO fine-tuned:         kp[0] (lumbar) to kp[8] (cervical) — direct spine axis")
    print(f"  Primary angles are NOT directly comparable across workflows.")
    print(f"  Use detection rate, confidence, and classification accuracy for comparison.")

    # ---- Save summary CSV ----
    out_path = COMPARISON_DIR / f"comparison_{out_name}.csv"
    pd.DataFrame(list(stats.values())).to_csv(out_path, index=False)
    print(f"\n  Summary saved to: {out_path.name} (in Data/Processed/comparison/)")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare MediaPipe / YOLO pretrained / YOLO fine-tuned evaluation results.")
    parser.add_argument("--mediapipe-csv", type=Path, default=CSVS["MediaPipe"], help="MediaPipe results CSV.")
    parser.add_argument("--pretrained-csv", type=Path, default=CSVS["YOLO (pretrained)"], help="Pretrained-YOLO results CSV.")
    parser.add_argument("--finetuned-csv", type=Path, default=CSVS["YOLO (fine-tuned)"], help="Fine-tuned-YOLO results CSV.")
    parser.add_argument("--out-name", default="summary",
                         help="Writes comparison_<out-name>.csv (default: summary, i.e. the existing comparison_summary.csv).")
    a = parser.parse_args()
    run({"MediaPipe": a.mediapipe_csv, "YOLO (pretrained)": a.pretrained_csv, "YOLO (fine-tuned)": a.finetuned_csv}, a.out_name)