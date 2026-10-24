# ============================================================
# Original vs. Cleaned Dataset — Fine-tuned YOLO Comparison
#
# Compares two yolo_finetuned run_evaluation.py CSVs — one from a
# model trained on the original dataset, one from a model trained
# on the cleaned dataset (see dataset_cleaning.py) — on the same
# test set. Focuses on measurement quality (confidence, geometry
# validity, anatomy-aware reliability), not just classification
# accuracy, since the test set is small.
#
# The two CSVs can be ANY two evaluated models (e.g. cleaned vs. cleaned + conservative
# augmentation): pass --baseline-csv / --candidate-csv with matching --baseline-label /
# --candidate-label / --out-name. Defaults reproduce the original-vs-cleaned comparison.
#
# This does NOT train anything. Full workflow (or just run
# run_original_vs_cleaned_experiment.py --no-dry-run instead):
#   1. python train_yolo_scoliosis.py --model-dir models/yolo_scoliosis
#        (original dataset -> models/yolo_scoliosis/run/weights/best.pt)
#   2. python train_yolo_scoliosis.py \
#        --data-yaml Data/Processed/scoliosis_cleaned/scoliosis_cleaned.yaml \
#        --model-dir models/yolo_scoliosis_cleaned
#        (cleaned dataset -> models/yolo_scoliosis_cleaned/run/weights/best.pt)
#   3. python run_evaluation.py --model yolo_finetuned \
#        --weights models/yolo_scoliosis/run/weights/best.pt --tag original_trained
#        (writes yolo_finetuned_scoliosis_test_original_trained.csv)
#   4. python run_evaluation.py --model yolo_finetuned \
#        --weights models/yolo_scoliosis_cleaned/run/weights/best.pt --tag cleaned_trained
#        (writes yolo_finetuned_scoliosis_test_cleaned_trained.csv)
#   5. python compare_original_vs_cleaned.py
# ============================================================

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT   = Path(__file__).resolve().parents[2]
COMPARISON_DIR = PROJECT_ROOT / "Data" / "Processed" / "comparison"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.Evaluation.dataset_eda import df_to_markdown
from src.Evaluation.dataset_paths import read_meta, describe_eval, is_legacy_eval, OLD_SPLIT_WARNING

DEFAULT_ORIGINAL_CSV = COMPARISON_DIR / "yolo_finetuned_scoliosis_test_original_trained.csv"
DEFAULT_CLEANED_CSV  = COMPARISON_DIR / "yolo_finetuned_scoliosis_test_cleaned_trained.csv"

# +1 -> higher is better, -1 -> lower is better (used only for the markdown report's
# plain-language interpretation; the console table just shows raw numbers).
METRIC_DIRECTION = {
    "mean_keypoint_confidence":        1,
    "valid_spine_geometry_pct":        1,
    "mean_anatomy_reliability_score":  1,
    "curvature_features_reliable_pct": 1,
    "retake_or_review_pct":            -1,
    "shoulder_warning_pct":            -1,
    "classification_accuracy":         1,
    "scoliosis_recall":                1,
}


def summarise(df: pd.DataFrame, label: str) -> dict:
    total = len(df)
    stats = {"model": label, "n_samples": total}

    stats["mean_keypoint_confidence"] = round(df["mean_confidence"].mean(), 3) if "mean_confidence" in df else None

    if "valid_spine_geometry" in df:
        stats["valid_spine_geometry_pct"] = round(100 * df["valid_spine_geometry"].mean(), 1)
    if "anatomy_reliability_score" in df:
        vals = df["anatomy_reliability_score"].dropna()
        stats["mean_anatomy_reliability_score"] = round(vals.mean(), 3) if len(vals) else None
    if "curvature_features_reliable" in df:
        stats["curvature_features_reliable_pct"] = round(100 * df["curvature_features_reliable"].mean(), 1)
    if "retake_or_review_recommended" in df:
        stats["retake_or_review_pct"] = round(100 * df["retake_or_review_recommended"].mean(), 1)
    if "anatomy_warning_flags" in df and "valid_spine_geometry" in df:
        # Only images where geometry was computed can raise a shoulder warning, so measure over those —
        # otherwise a model that misses detections looks artificially "better".
        has_geom = df[df["valid_spine_geometry"].astype(bool)]
        stats["shoulder_warning_pct"] = (round(100 * has_geom["anatomy_warning_flags"].fillna("").str.contains("shoulder").mean(), 1)
                                         if len(has_geom) else None)

    if "classification_correct" in df:
        cls_rows = df[df["classification_correct"].notna()]
        stats["classification_accuracy"] = round(cls_rows["classification_correct"].mean(), 3) if len(cls_rows) else None
        scoli_rows = df[(df["ground_truth_class"] == "scoliosis") & df["predicted_class"].notna()]
        stats["scoliosis_recall"] = round((scoli_rows["predicted_class"] == "scoliosis").mean(), 3) if len(scoli_rows) else None

    return stats


def fmt(val, unit=""):
    if val is None:
        return "N/A"
    if isinstance(val, float):
        return f"{val:.3f}{unit}"
    return f"{val}{unit}"


def interpret_delta(key, o_val, c_val, unit, cand_label="the cleaned-trained model"):
    """Plain-language read of a metric's change, accounting for which direction is 'better'."""
    if o_val is None or c_val is None:
        return None
    delta = c_val - o_val
    if abs(delta) < 1e-9:
        return "no change"
    direction = METRIC_DIRECTION.get(key, 1)
    verb = "better" if (delta * direction) > 0 else "worse"
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.3f}{unit} ({verb}) for {cand_label}"


def provenance_notes(base_label, cand_label, base_csv: Path, cand_csv: Path):
    """What each CSV was evaluated on, plus warnings: mixed test sets, old-split models, historical results."""
    metas = {base_label: read_meta(base_csv), cand_label: read_meta(cand_csv)}
    notes = [f"- **Evaluated on** — {label}: {describe_eval(m)}" for label, m in metas.items()]
    sets = {(m or {}).get("eval_dataset", "legacy") + "/" + (m or {}).get("eval_split", "test") for m in metas.values()}
    warnings = []
    if len(sets) > 1:
        warnings.append("These two results come from **different test sets** and are not directly comparable.")
    if any((m or {}).get("old_split_warning") for m in metas.values()):
        warnings.append(OLD_SPLIT_WARNING)
    legacy = all(is_legacy_eval(m) for m in metas.values())
    if legacy:
        warnings.append("These results are on the **old public test split**, which shares near-identical photos with "
                        "train/valid. Treat them as **historical/debug only**; final evaluation needs models retrained "
                        "and evaluated on the grouped split.")
    return notes, warnings, legacy


def write_markdown_report(stats, metrics, original_csv: Path, cleaned_csv: Path,
                          base_label="Original-trained", cand_label="Cleaned-trained", out_name="original_vs_cleaned") -> Path:
    original = stats[base_label]
    cleaned  = stats[cand_label]

    table_df = pd.DataFrame([
        {
            "Metric": name,
            base_label: fmt(original.get(key), unit),
            cand_label: fmt(cleaned.get(key), unit),
        }
        for name, key, unit in metrics
    ])

    lines = []
    lines.append(f"# {base_label} vs. {cand_label} — Fine-tuned YOLO Comparison Report\n")
    lines.append("Auto-generated by `compare_original_vs_cleaned.py`. Both models are the same architecture "
                  "(`yolov8n-pose`) and are evaluated on the same held-out test set — any difference below "
                  "comes from how they were trained (data and/or augmentation; see the training setup in "
                  "`workflow_comparison.ipynb`).\n")
    lines.append(f"- **{base_label}**: `{original_csv.name}`")
    lines.append(f"- **{cand_label}**: `{cleaned_csv.name}`")
    notes, warnings, legacy = provenance_notes(base_label, cand_label, original_csv, cleaned_csv)
    lines.extend(notes)
    lines.append("- Dataset cleaning details: `Data/Processed/eda/data_cleaning_report.md`\n")
    for w in warnings:
        lines.append(f"> **Warning:** {w}\n")

    lines.append("## Metrics\n")
    lines.append(df_to_markdown(table_df))
    lines.append("")

    lines.append("## What changed\n")
    for name, key, unit in metrics:
        interp = interpret_delta(key, original.get(key), cleaned.get(key), unit, cand_label=cand_label)
        if interp is not None:
            lines.append(f"- **{name}**: {interp}")
    lines.append("")

    lines.append("## How to read this\n")
    lines.append(f"- Test set size: **{original.get('n_samples')} images** — small enough that "
                  "`classification_accuracy` / `scoliosis_recall` can swing a lot from one or two "
                  "predictions. Treat those two as indicative, not conclusive.")
    lines.append("- The measurement-quality metrics (keypoint confidence, valid spine geometry, "
                  "anatomy reliability score, curvature-feature reliability, shoulder-warning and "
                  "retake/review rates) are the more informative comparison here; accuracy on a test "
                  "set this small is the least stable number.")
    lines.append("- `retake_or_review_recommended` and the shoulder-warning rate — **lower is better**: they are the "
                  "fraction of test images the anatomy-aware layer itself flagged as needing a human look.")
    if legacy:
        lines.append("- Near-duplicate copies of some test photos also appear in the training data "
                      "(see `split_quality_report_cleaned.md`), so absolute numbers are optimistic; compare models "
                      "against each other, not against a target.")
    else:
        lines.append("- This test set comes from a photo-grouped split, so no test photo is shared with the training "
                      "photos **for models trained on that same split**. The test set is still small; compare models "
                      "against each other and do not quote a single accuracy figure as final.")
    lines.append("- All figures are screening-support indicators, not a clinical diagnosis, Cobb angle, "
                  "or true angle-of-trunk-rotation measurement.\n")

    lines.append("## Related files\n")
    lines.append("- `Data/Processed/eda/data_cleaning_report.md` — what was removed during cleaning, and why")
    lines.append("- `Data/Processed/evaluation/fine_tuning_plan.md` — why these experiments exist and how results are judged")
    lines.append(f"- `{original_csv.name}` / `{cleaned_csv.name}` — per-image raw evaluation output")
    lines.append(f"- `comparison_{out_name}.csv` — the metrics table above, as CSV\n")

    out_path = COMPARISON_DIR / f"{out_name}_report.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def run(original_csv: Path, cleaned_csv: Path, base_label="Original-trained", cand_label="Cleaned-trained", out_name="original_vs_cleaned"):
    missing = [p for p in (original_csv, cleaned_csv) if not p.exists()]
    if missing:
        print("[ERROR] Missing CSV(s):")
        for p in missing:
            print(f"         {p}")
        print("\n         Generate both with run_evaluation.py --model yolo_finetuned")
        print("         (once per model — see this script's module docstring for the full sequence).")
        return

    original_df = pd.read_csv(original_csv)
    cleaned_df  = pd.read_csv(cleaned_csv)

    stats = {
        base_label: summarise(original_df, base_label),
        cand_label: summarise(cleaned_df, cand_label),
    }

    metrics = [
        ("Samples evaluated",              "n_samples",                        ""),
        ("Mean keypoint confidence",       "mean_keypoint_confidence",         ""),
        ("Valid spine geometry",            "valid_spine_geometry_pct",         "%"),
        ("Mean anatomy reliability score",  "mean_anatomy_reliability_score",   ""),
        ("Curvature features reliable",     "curvature_features_reliable_pct",  "%"),
        ("Retake/review recommended",       "retake_or_review_pct",             "%"),
        ("Shoulder warnings (of images with geometry)", "shoulder_warning_pct",     "%"),
        ("Classification accuracy",         "classification_accuracy",          ""),
        ("Scoliosis recall",                "scoliosis_recall",                 ""),
    ]

    col_width = 22
    labels = list(stats.keys())

    print(f"\n{'='*70}")
    print(f"  {base_label.upper()} vs. {cand_label.upper()} — FINE-TUNED YOLO COMPARISON")
    print(f"  {base_label}: {original_csv.name}")
    print(f"  {cand_label}: {cleaned_csv.name}")
    print("  Same test set for both — differences reflect how the models were trained.")
    _notes, _warnings, _ = provenance_notes(base_label, cand_label, original_csv, cleaned_csv)
    for n in _notes:
        print("  " + n.replace("**", "").lstrip("- "))
    for w in _warnings:
        print("  [WARN] " + w.replace("**", ""))
    print(f"{'='*70}\n")

    header = f"  {'Metric':<34}" + "".join(f"{l:<{col_width}}" for l in labels)
    print(header)
    print(f"  {'-'*34}" + "-" * (col_width * len(labels)))
    for name, key, unit in metrics:
        row = f"  {name:<34}"
        for label in labels:
            row += f"{fmt(stats[label].get(key), unit):<{col_width}}"
        print(row)

    print(f"\n  NOTE: read this as a training ablation, not a certified benchmark — the test")
    print(f"  set is small, so treat differences as indicative unless they're large and consistent.")

    out_path = COMPARISON_DIR / f"comparison_{out_name}.csv"
    pd.DataFrame(list(stats.values())).to_csv(out_path, index=False)
    print(f"\n  Summary saved to: {out_path.name} (in Data/Processed/comparison/)")

    report_path = write_markdown_report(stats, metrics, original_csv, cleaned_csv, base_label, cand_label, out_name)
    print(f"  Report saved to : {report_path.name} (in Data/Processed/comparison/)")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two fine-tuned YOLO models' run_evaluation.py results "
                                                 "(default: trained on the original vs. cleaned dataset).")
    parser.add_argument("--baseline-csv", "--original-csv", dest="baseline_csv", type=Path, default=DEFAULT_ORIGINAL_CSV,
                         help=f"run_evaluation.py output for the baseline model (default: {DEFAULT_ORIGINAL_CSV.name})")
    parser.add_argument("--candidate-csv", "--cleaned-csv", dest="candidate_csv", type=Path, default=DEFAULT_CLEANED_CSV,
                         help=f"run_evaluation.py output for the candidate model (default: {DEFAULT_CLEANED_CSV.name})")
    parser.add_argument("--baseline-label", default="Original-trained", help="Display name for the baseline model.")
    parser.add_argument("--candidate-label", default="Cleaned-trained", help="Display name for the candidate model.")
    parser.add_argument("--out-name", default="original_vs_cleaned",
                         help="Output file stem: writes comparison_<out-name>.csv and <out-name>_report.md "
                              "(default: original_vs_cleaned, i.e. the existing files).")
    args = parser.parse_args()
    run(args.baseline_csv, args.candidate_csv, args.baseline_label, args.candidate_label, args.out_name)
