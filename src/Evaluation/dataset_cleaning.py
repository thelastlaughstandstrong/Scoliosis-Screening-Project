# ============================================================
# Dataset Cleaning — Review-First, Export-On-Request
# public_datasets/scoliosis (YOLOv8-pose, 12-point spine schema)
#
# Classifies every image/label sample as hard_error / warning / ok,
# and (only when explicitly asked) exports a cleaned COPY of the
# dataset containing only samples without a hard error.
#
# This script NEVER modifies or deletes anything under
# public_datasets/scoliosis/ — it only reads from there. A cleaned
# copy is written to a separate output directory, and only when
# --export-cleaned is passed (and --dry-run is not).
#
# Severity model:
#   hard_error — objective, unsafe-for-training issue -> excluded from export
#   warning    — worth a human look, but kept by default
#   ok         — no issue detected
#
# Usage:
#   python dataset_cleaning.py --dry-run
#   python dataset_cleaning.py --export-cleaned
#   python dataset_cleaning.py --export-cleaned --strict-multiple-objects
#   python dataset_cleaning.py --export-cleaned --min-visible-keypoints 4 --output-dir Data/Processed/scoliosis_cleaned_v2
# ============================================================

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import re
import math
import argparse
import hashlib
import shutil
import collections
from pathlib import Path

import yaml
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.keypoint_schema import (
    NUM_KEYPOINTS, SPINE_BASE_LUMBAR, SPINE_TOP_CERVICAL,
    INTERMEDIATE_SPINE_IDXS, LEFT_SHOULDER, RIGHT_SHOULDER, SPINE_POINT_IDXS,
)
# Reuse tested, low-level primitives from the EDA module instead of redefining
# them — cleaning-specific *decisions* (severity, export) stay in this file.
from src.Evaluation.dataset_eda import (
    DATASET_DIR, DATA_YAML, SPLITS, IMAGE_EXTS, EXPECTED_TOKENS,
    VALID_VISIBILITY_VALUES, load_class_names, imdecode_unicode, cv2, df_to_markdown,
)
from src.FeatureExtraction.spine_geometry import MIN_VALID_SPINE_KEYPOINTS

EDA_OUTPUT_DIR      = PROJECT_ROOT / "Data" / "Processed" / "eda"
DEFAULT_OUTPUT_DIR  = PROJECT_ROOT / "Data" / "Processed" / "scoliosis_cleaned"

VISIBLE_FLAG = 2.0   # YOLO-pose convention: v == 2 means "labeled and visible"

# --- Soft, documented thresholds (heuristics, not clinical cutoffs) ---
MIN_INTERMEDIATE_FOR_READY        = 2     # kp[1..7] visible needed to trust curve-smoothness at all
INTERMEDIATE_MOSTLY_INVISIBLE_MAX = 3     # warn if fewer than this many of the 7 intermediates are visible
UNUSUAL_INCLINATION_DEG           = 40.0  # soft "does this look like a normal standing photo" check

SPLIT_PRIORITY = {"train": 0, "valid": 1, "test": 2}   # which split "wins" a cross-split duplicate


def normalize_tag(tag: str) -> str:
    """Group a detailed issue tag (e.g. 'invalid_class_id=5') down to its type
    (e.g. 'invalid_class_id') for count-by-type reporting."""
    return re.split(r"[:=(]", tag, maxsplit=1)[0]


# ---------------------------------------------------------------
# Per-object-row parsing (hard-error checks only — see dataset_eda.py
# for the softer, EDA-oriented version of this same parsing)
# ---------------------------------------------------------------

def parse_object_row(line: str, valid_class_ids):
    """Parse one YOLO-pose label line. Returns (obj_dict_or_None, hard_error_tags).

    A row with ANY hard error is treated as unusable (obj=None) — the whole
    sample it belongs to is excluded from the cleaned export, since a partially
    broken label file can't be trusted to describe the surviving rows either.
    """
    hard = []
    toks = line.split()
    if len(toks) != EXPECTED_TOKENS:
        return None, [f"wrong_value_count={len(toks)}"]

    try:
        values = [float(t) for t in toks]
    except ValueError:
        return None, ["non_numeric_value"]

    class_id = int(values[0])
    if valid_class_ids is not None and class_id not in valid_class_ids:
        hard.append(f"invalid_class_id={class_id}")

    xc, yc, bw, bh = values[1:5]
    if any(v < 0 or v > 1 for v in (xc, yc, bw, bh)):
        hard.append("bbox_out_of_range")
    if bw <= 0 or bh <= 0:
        hard.append("bbox_non_positive_size")

    keypoints = []
    for i in range(NUM_KEYPOINTS):
        x, y, v = values[5 + 3 * i], values[6 + 3 * i], values[7 + 3 * i]
        if v not in VALID_VISIBILITY_VALUES:
            hard.append(f"invalid_visibility_value:kp{i}")
        elif v > 0 and (x < 0 or x > 1 or y < 0 or y > 1):
            # Only checked when the point is claimed labeled (v>0) — an
            # unlabeled point (v==0) commonly carries placeholder (0,0) coords.
            hard.append(f"keypoint_out_of_range:kp{i}")
        keypoints.append((x, y, v))

    if hard:
        return None, hard

    return {"class_id": class_id, "keypoints": keypoints}, []


# ---------------------------------------------------------------
# Per-sample (one image/label pair) classification
# ---------------------------------------------------------------

def classify_sample(split, stem, image_dir, label_dir, valid_class_ids, class_names, min_visible_keypoints):
    image_path = next((p for ext in IMAGE_EXTS if (p := image_dir / f"{stem}{ext}").exists()), None)
    label_path = label_dir / f"{stem}.txt"
    label_exists = label_path.exists()

    hard, warn = [], []

    record = {
        "split": split, "stem": stem,
        "image_path": str(image_path) if image_path else "",
        "label_path": str(label_path) if label_exists else "",
        "n_objects": 0, "class_id": None, "class_name": None,
        "visible_keypoint_count": None, "visible_spine_count": None, "visible_intermediate_count": None,
        "geometry_ready_basic": False, "geometry_ready_intermediate": False,
        "image_hash": None,
    }

    if image_path is None:
        hard.append("missing_image")
    else:
        if cv2 is not None and imdecode_unicode(image_path) is None:
            hard.append("unreadable_image")
        try:
            record["image_hash"] = hashlib.md5(image_path.read_bytes()).hexdigest()
        except Exception:
            pass

    if not label_exists:
        hard.append("missing_label")

    objects = []
    if label_exists:
        text = label_path.read_text(encoding="utf-8").strip()
        if not text:
            hard.append("empty_label_file")
        else:
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                obj, row_hard = parse_object_row(line, valid_class_ids)
                if row_hard:
                    hard.extend(row_hard)
                else:
                    objects.append(obj)

    record["n_objects"] = len(objects)
    if len(objects) > 1:
        warn.append(f"multiple_objects_n={len(objects)}")

    primary = objects[0] if objects else None
    if primary is not None:
        record["class_id"] = primary["class_id"]
        record["class_name"] = class_names.get(primary["class_id"], "unknown")
        kps = primary["keypoints"]

        visible_total        = [i for i in range(NUM_KEYPOINTS) if kps[i][2] >= VISIBLE_FLAG]
        visible_spine         = [i for i in SPINE_POINT_IDXS if kps[i][2] >= VISIBLE_FLAG]
        visible_intermediate  = [i for i in INTERMEDIATE_SPINE_IDXS if kps[i][2] >= VISIBLE_FLAG]

        record["visible_keypoint_count"]    = len(visible_total)
        record["visible_spine_count"]       = len(visible_spine)
        record["visible_intermediate_count"] = len(visible_intermediate)

        base_ok = kps[SPINE_BASE_LUMBAR][2] >= VISIBLE_FLAG
        top_ok  = kps[SPINE_TOP_CERVICAL][2] >= VISIBLE_FLAG
        ls_ok   = kps[LEFT_SHOULDER][2] >= VISIBLE_FLAG
        rs_ok   = kps[RIGHT_SHOULDER][2] >= VISIBLE_FLAG

        # geometry_ready_basic needs shoulders too (shoulder-asymmetry features);
        # geometry_ready_intermediate only needs the base/top reference line plus
        # enough intermediate points — shoulder visibility is irrelevant to it.
        record["geometry_ready_basic"] = bool(base_ok and top_ok and ls_ok and rs_ok)
        record["geometry_ready_intermediate"] = bool(
            base_ok and top_ok and len(visible_intermediate) >= MIN_INTERMEDIATE_FOR_READY
        )

        if len(visible_spine) < min_visible_keypoints:
            warn.append(f"low_visible_keypoint_count={len(visible_spine)}")
        if not base_ok:
            warn.append("spine_base_not_visible")
        if not top_ok:
            warn.append("spine_top_not_visible")
        if not ls_ok or not rs_ok:
            warn.append("shoulder_not_visible")
        if len(visible_intermediate) < INTERMEDIATE_MOSTLY_INVISIBLE_MAX:
            warn.append(f"intermediate_keypoints_mostly_invisible={len(visible_intermediate)}/7")

        if base_ok and top_ok:
            bx, by = kps[SPINE_BASE_LUMBAR][:2]
            tx, ty = kps[SPINE_TOP_CERVICAL][:2]
            if (by - ty) <= 0:
                warn.append("questionable_anatomical_ordering")
            else:
                angle = math.degrees(math.atan2(tx - bx, by - ty))
                if abs(angle) > UNUSUAL_INCLINATION_DEG:
                    warn.append(f"unusual_geometry_inclination={angle:.1f}deg")

    record["hard_errors"] = hard
    record["warnings"] = warn
    return record


# ---------------------------------------------------------------
# Cross-split duplicate-content check (mutates records in place)
# ---------------------------------------------------------------

def flag_cross_split_duplicates(records):
    hash_to_records = collections.defaultdict(list)
    for r in records:
        if r["image_hash"]:
            hash_to_records[r["image_hash"]].append(r)

    flagged = 0
    for h, group in hash_to_records.items():
        splits_present = {r["split"] for r in group}
        if len(splits_present) <= 1:
            continue
        keeper_split = min(splits_present, key=lambda s: SPLIT_PRIORITY.get(s, 99))
        for r in group:
            if r["split"] != keeper_split:
                r["hard_errors"].append(f"duplicate_image_hash_cross_split(kept_in={keeper_split})")
                flagged += 1
    return flagged


def finalize_record(r, strict_multiple_objects):
    if strict_multiple_objects and r["n_objects"] > 1:
        r["hard_errors"].append("multiple_objects_excluded_strict")

    if r["hard_errors"]:
        r["severity"], r["status"] = "hard_error", "excluded"
    elif r["warnings"]:
        r["severity"], r["status"] = "warning", "kept"
    else:
        r["severity"], r["status"] = "ok", "kept"

    r["issues"] = "; ".join(r["hard_errors"] + r["warnings"])
    return r


# ---------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------

def collect_records(valid_class_ids, class_names, min_visible_keypoints, strict_multiple_objects):
    records = []
    for split in SPLITS:
        image_dir = DATASET_DIR / split / "images"
        label_dir = DATASET_DIR / split / "labels"

        image_stems = {p.stem for p in image_dir.glob("*") if p.suffix.lower() in IMAGE_EXTS} if image_dir.exists() else set()
        label_stems = {p.stem for p in label_dir.glob("*.txt")} if label_dir.exists() else set()

        for stem in sorted(image_stems | label_stems):
            records.append(classify_sample(split, stem, image_dir, label_dir, valid_class_ids, class_names, min_visible_keypoints))

    flag_cross_split_duplicates(records)
    for r in records:
        finalize_record(r, strict_multiple_objects)
    return records


def build_manifest_df(records):
    cols = [
        "split", "stem", "image_path", "label_path", "status", "severity", "issues",
        "class_id", "class_name", "n_objects",
        "visible_keypoint_count", "visible_spine_count", "visible_intermediate_count",
        "geometry_ready_basic", "geometry_ready_intermediate",
    ]
    return pd.DataFrame(records)[cols]


def class_distribution(manifest_df, kept_only):
    df = manifest_df[manifest_df["status"] == "kept"] if kept_only else manifest_df
    df = df[df["class_id"].notna()]
    if df.empty:
        return pd.DataFrame(columns=["split", "class_name", "count"])
    return df.groupby(["split", "class_name"]).size().reset_index(name="count")


def issue_counts(manifest_df, column):
    tags = manifest_df[column].str.split("; ").explode()
    tags = tags[tags != ""].dropna()
    if tags.empty:
        return pd.Series(dtype=int)
    return tags.map(normalize_tag).value_counts()


def hard_error_tags_only(manifest_df):
    # "issues" mixes hard errors and warnings; hard-error tags are exactly the
    # ones that also appear for excluded rows and aren't in the warning vocabulary.
    warning_prefixes = (
        "multiple_objects_n", "low_visible_keypoint_count", "spine_base_not_visible",
        "spine_top_not_visible", "shoulder_not_visible", "intermediate_keypoints_mostly_invisible",
        "questionable_anatomical_ordering", "unusual_geometry_inclination",
    )
    tags = manifest_df["issues"].str.split("; ").explode()
    tags = tags[tags != ""].dropna()
    tags = tags[~tags.map(lambda t: normalize_tag(t).startswith(warning_prefixes))]
    if tags.empty:
        return pd.Series(dtype=int)
    return tags.map(normalize_tag).value_counts()


def warning_tags_only(manifest_df):
    warning_prefixes = (
        "multiple_objects_n", "low_visible_keypoint_count", "spine_base_not_visible",
        "spine_top_not_visible", "shoulder_not_visible", "intermediate_keypoints_mostly_invisible",
        "questionable_anatomical_ordering", "unusual_geometry_inclination",
    )
    tags = manifest_df["issues"].str.split("; ").explode()
    tags = tags[tags != ""].dropna()
    tags = tags[tags.map(lambda t: normalize_tag(t).startswith(warning_prefixes))]
    if tags.empty:
        return pd.Series(dtype=int)
    return tags.map(normalize_tag).value_counts()


# ---------------------------------------------------------------
# Cleaned dataset export (only reached with --export-cleaned, no --dry-run)
# ---------------------------------------------------------------

def export_cleaned_dataset(manifest_df, output_dir: Path, class_names, nc):
    for split in SPLITS:
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    kept = manifest_df[manifest_df["status"] == "kept"]
    copied = 0
    for _, row in kept.iterrows():
        if not row["image_path"] or not row["label_path"]:
            continue   # shouldn't happen for "kept" rows, but stay defensive
        split = row["split"]
        shutil.copy2(row["image_path"], output_dir / split / "images" / Path(row["image_path"]).name)
        shutil.copy2(row["label_path"], output_dir / split / "labels" / Path(row["label_path"]).name)
        copied += 1

    data_yaml = {
        "path":       str(output_dir),
        "train":      "train/images",
        "val":        "valid/images",
        "test":       "test/images",
        "kpt_shape":  [NUM_KEYPOINTS, 3],
        "flip_idx":   list(range(NUM_KEYPOINTS)),
        "nc":         nc,
        "names":      class_names,
    }
    with open(output_dir / "scoliosis_cleaned.yaml", "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

    return copied


# ---------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------

def write_cleaning_report(manifest_df, args):
    total = len(manifest_df)
    kept = int((manifest_df["status"] == "kept").sum())
    excluded = int((manifest_df["status"] == "excluded").sum())
    n_warning = int((manifest_df["severity"] == "warning").sum())
    n_ok = int((manifest_df["severity"] == "ok").sum())

    before_dist = class_distribution(manifest_df, kept_only=False)
    after_dist  = class_distribution(manifest_df, kept_only=True)

    hard_counts = hard_error_tags_only(manifest_df)
    warn_counts = warning_tags_only(manifest_df)

    geom_before = manifest_df[manifest_df["class_id"].notna()]
    geom_after  = manifest_df[(manifest_df["status"] == "kept") & (manifest_df["class_id"].notna())]

    def geom_pct(df, col):
        return round(100 * df[col].mean(), 1) if len(df) else None

    lines = []
    lines.append("# Data Cleaning Report — public_datasets/scoliosis\n")
    lines.append("Auto-generated by `src/Evaluation/dataset_cleaning.py`. Read-only classification; "
                  "the original dataset under `public_datasets/scoliosis/` is never modified or deleted.\n")

    lines.append("## Run configuration\n")
    lines.append(f"- `--min-visible-keypoints` = {args.min_visible_keypoints}")
    lines.append(f"- `--strict-multiple-objects` = {args.strict_multiple_objects}")
    lines.append(f"- Export requested: {args.export_cleaned and not args.dry_run} "
                  f"(`--export-cleaned={args.export_cleaned}`, `--dry-run={args.dry_run}`)")
    lines.append(f"- Output directory: `{args.output_dir}`\n")

    lines.append("## 1. Overview\n")
    lines.append(f"- Total samples checked: **{total}**")
    lines.append(f"- Kept: **{kept}** ({n_ok} ok, {n_warning} warning-only)")
    lines.append(f"- Excluded (hard error): **{excluded}**\n")

    lines.append("## 2. Hard errors by type (excluded samples)\n")
    if hard_counts.empty:
        lines.append("No hard errors found.\n")
    else:
        lines.append(df_to_markdown(hard_counts.rename_axis("hard_error_type").reset_index(name="count")))
    lines.append("")

    lines.append("## 3. Warnings by type (kept, but flagged for review)\n")
    if warn_counts.empty:
        lines.append("No warnings found.\n")
    else:
        lines.append(df_to_markdown(warn_counts.rename_axis("warning_type").reset_index(name="count")))
    lines.append("")

    lines.append("## 4. Class distribution — before vs. after cleaning\n")
    lines.append("Note: a sample only has a determined class if at least one label row parsed cleanly, so "
                  "\"before\" already excludes unparseable rows (e.g. `empty_label_file`, `keypoint_out_of_range`) "
                  "— these two tables only differ when a *parseable* sample is excluded for a reason unrelated "
                  "to row parsing (a cross-split duplicate, or `--strict-multiple-objects`).\n")
    lines.append("**Before (all samples with a determined class):**\n")
    lines.append(df_to_markdown(before_dist) if not before_dist.empty else "_(no data)_")
    lines.append("\n**After (kept samples only):**\n")
    lines.append(df_to_markdown(after_dist) if not after_dist.empty else "_(no data)_")
    lines.append("")

    lines.append("## 5. Geometry-readiness summary\n")
    lines.append(f"- `geometry_ready_basic` — before: **{geom_pct(geom_before, 'geometry_ready_basic')}%**, "
                  f"after: **{geom_pct(geom_after, 'geometry_ready_basic')}%**")
    lines.append(f"- `geometry_ready_intermediate` — before: **{geom_pct(geom_before, 'geometry_ready_intermediate')}%**, "
                  f"after: **{geom_pct(geom_after, 'geometry_ready_intermediate')}%**\n")

    lines.append("## 6. Recommendation\n")
    if excluded == 0:
        lines.append("No hard errors were found — the cleaned export (if generated) is identical in content "
                      "to the original dataset. **Retraining/re-evaluating is not necessary** unless you also "
                      "change `--min-visible-keypoints` or `--strict-multiple-objects`.")
    else:
        pct_excluded = round(100 * excluded / total, 1)
        lines.append(f"{excluded} of {total} samples ({pct_excluded}%) had an objective hard error and were "
                      f"excluded from the cleaned export. If that's a meaningful fraction of `train`, "
                      f"**re-running `train_yolo_scoliosis.py` on the cleaned data and re-running "
                      f"`run_evaluation.py` / `compare_workflows.py`** is worth doing to see whether removing "
                      f"these objectively-broken labels changes training stability or evaluation numbers. "
                      f"If it's only a handful of samples, the effect is likely to be small.")
    lines.append("")
    lines.append("> **Important:** this cleaning pass only removes *objective format/integrity errors* "
                  "(missing files, malformed rows, out-of-range coordinates, invalid ids, cross-split "
                  "duplicates). It does **not** remove realistic, anatomically difficult, or ambiguous "
                  "cases — those are exactly the kind of hard examples a screening model needs to be "
                  "evaluated against, and are only ever flagged here as *warnings* for manual review, "
                  "never auto-excluded.\n")

    (EDA_OUTPUT_DIR / "data_cleaning_report.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------

def display_path(p: Path) -> str:
    """Render a path for console printing, relative to PROJECT_ROOT when possible.

    Falls back to just the path's own name rather than an absolute path — this
    project's root directory contains non-ASCII characters, and printing an
    absolute path through it can raise UnicodeEncodeError on a Windows console
    using a legacy (cp1252) codepage.
    """
    try:
        return str(p.relative_to(PROJECT_ROOT))
    except ValueError:
        return p.name


def print_console_summary(manifest_df, args, exported_count):
    total = len(manifest_df)
    kept = int((manifest_df["status"] == "kept").sum())
    excluded = int((manifest_df["status"] == "excluded").sum())

    print(f"\n{'='*60}")
    print("  DATASET CLEANING SUMMARY")
    print(f"{'='*60}")
    print(f"  Total samples checked : {total}")
    print(f"  Kept                  : {kept}")
    print(f"  Excluded (hard error) : {excluded}")

    hard_counts = hard_error_tags_only(manifest_df)
    if not hard_counts.empty:
        print(f"\n  Hard errors by type:")
        for tag, n in hard_counts.items():
            print(f"    {tag:<40}: {n}")

    if args.export_cleaned and not args.dry_run:
        print(f"\n  Cleaned dataset exported to: {display_path(args.output_dir)}")
        print(f"  Files copied (image+label pairs): {exported_count}")
    else:
        print(f"\n  No files exported (dry-run / --export-cleaned not set).")

    print(f"\n  Manifest   : {display_path(args.output_dir / 'cleaning_manifest.csv')}")
    print(f"  Report     : {display_path(EDA_OUTPUT_DIR / 'data_cleaning_report.md')}")
    print(f"\n  Suggested next commands:")
    print(f"    python src/Evaluation/dataset_cleaning.py --dry-run")
    print(f"    python src/Evaluation/dataset_cleaning.py --export-cleaned")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def run(args):
    # Resolve to absolute so later `.relative_to(PROJECT_ROOT)` calls work
    # regardless of whether --output-dir was passed as relative or absolute.
    args.output_dir = args.output_dir.resolve()

    class_names, nc = load_class_names(DATA_YAML)
    valid_class_ids = set(class_names.keys()) if class_names else None
    if not class_names:
        print(f"[WARN] Could not load class names from {display_path(DATA_YAML)} — class-id validity check disabled.")

    records = collect_records(valid_class_ids, class_names, args.min_visible_keypoints, args.strict_multiple_objects)
    manifest_df = build_manifest_df(records)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(args.output_dir / "cleaning_manifest.csv", index=False)
    manifest_df[manifest_df["status"] == "excluded"].to_csv(args.output_dir / "excluded_samples.csv", index=False)
    manifest_df[manifest_df["severity"] == "warning"].to_csv(args.output_dir / "warning_samples.csv", index=False)

    EDA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_cleaning_report(manifest_df, args)

    exported_count = 0
    if args.export_cleaned and not args.dry_run:
        exported_count = export_cleaned_dataset(manifest_df, args.output_dir, class_names, nc)
    elif args.export_cleaned and args.dry_run:
        print("[INFO] --dry-run set — skipping file export even though --export-cleaned was passed.")

    print_console_summary(manifest_df, args, exported_count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Review-first data cleaning for public_datasets/scoliosis.")
    parser.add_argument("--export-cleaned", action="store_true",
                         help="Copy kept samples (image+label) into --output-dir as a cleaned dataset.")
    parser.add_argument("--dry-run", action="store_true",
                         help="Classify and report only — never copy files, even if --export-cleaned is set.")
    parser.add_argument("--strict-multiple-objects", action="store_true",
                         help="Treat multi-object label files as a hard error (excluded) instead of a warning.")
    parser.add_argument("--min-visible-keypoints", type=int, default=MIN_VALID_SPINE_KEYPOINTS,
                         help=f"Warn if fewer than this many of the 9 spine keypoints (kp0..8) are visible "
                              f"(default: {MIN_VALID_SPINE_KEYPOINTS}, matching spine_geometry.py).")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                         help=f"Where to write the manifest/CSVs and (if requested) the cleaned dataset "
                              f"(default: {DEFAULT_OUTPUT_DIR.relative_to(PROJECT_ROOT)}).")
    args = parser.parse_args()
    run(args)
