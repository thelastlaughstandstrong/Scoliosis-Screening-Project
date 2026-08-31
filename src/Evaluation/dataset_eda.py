# ============================================================
# Dataset EDA & Data-Quality Report
# public_datasets/scoliosis (YOLOv8-pose, 12-point spine schema)
#
# Read-only inspection of the train/valid/test splits: file
# integrity, label validity, class balance, keypoint visibility,
# and whether ground-truth keypoints support the anatomy-aware
# geometry layer in src/FeatureExtraction/spine_geometry.py.
#
# Does NOT touch the model, training config, or geometry module.
#
# Usage:
#   python dataset_eda.py                (full EDA + sample overlays)
#   python dataset_eda.py --no-overlays  (skip saving overlay images)
# ============================================================

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import argparse
import hashlib
import collections
from pathlib import Path

import yaml
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.keypoint_schema import (
    KEYPOINT_MAP, NUM_KEYPOINTS,
    SPINE_BASE_LUMBAR, SPINE_TOP_CERVICAL, INTERMEDIATE_SPINE_IDXS,
    SHOULDER_IDXS, NECK_HEAD_CENTER, DEFAULT_CONFIDENCE_THRESHOLD,
)
from src.FeatureExtraction.spine_geometry import extract_spine_geometry

try:
    import cv2
except ImportError:
    cv2 = None

DATASET_DIR = PROJECT_ROOT / "public_datasets" / "scoliosis"
DATA_YAML   = PROJECT_ROOT / "public_datasets" / "scoliosis_train.yaml"
OUTPUT_DIR  = PROJECT_ROOT / "Data" / "Processed" / "eda"
OVERLAY_DIR = OUTPUT_DIR / "sample_overlays"

SPLITS       = ["train", "valid", "test"]
IMAGE_EXTS   = (".jpg", ".jpeg", ".png")
N_BBOX_VALS  = 4
EXPECTED_TOKENS = 1 + N_BBOX_VALS + NUM_KEYPOINTS * 3   # class + bbox + 12*(x,y,v)

# YOLO-pose visibility convention: 0 = not labeled, 1 = labeled but not visible, 2 = labeled and visible.
VALID_VISIBILITY_VALUES = (0.0, 1.0, 2.0)

# Simulated per-keypoint confidence for feeding ground-truth labels through
# extract_spine_geometry(), so "geometry readiness" reflects the *labels*,
# not a trained model's runtime confidence. Coordinates stay normalised
# [0,1] here (no image dims needed) — only the boolean/validity outcome
# and visible-count are trustworthy at that scale, not the *_px magnitudes.
SIM_CONF_VISIBLE  = 1.0   # v == 2
SIM_CONF_OCCLUDED = 0.3   # v == 1  (below DEFAULT_CONFIDENCE_THRESHOLD on purpose)
SIM_CONF_ABSENT   = 0.0   # v == 0

KP_GROUPS = {
    "spine_base_top":       [SPINE_BASE_LUMBAR, SPINE_TOP_CERVICAL],
    "spine_intermediate":   INTERMEDIATE_SPINE_IDXS,
    "shoulders":            SHOULDER_IDXS,
    "head_neck":            [NECK_HEAD_CENTER],
}


# ---------------------------------------------------------------
# Small IO helpers
# ---------------------------------------------------------------

def imdecode_unicode(img_path: Path):
    # cv2.imread fails on Windows paths with non-ASCII chars; use imdecode instead.
    try:
        return cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None


def load_class_names(yaml_path: Path):
    if not yaml_path.exists():
        return {}, None
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    names = data.get("names", {}) or {}
    return {int(k): v for k, v in names.items()}, data.get("nc")


# ---------------------------------------------------------------
# Label parsing
# ---------------------------------------------------------------

def parse_label_line(line: str, valid_class_ids):
    """Parse one YOLO-pose label line into a validated object dict, or None if malformed."""
    toks = line.split()
    if len(toks) != EXPECTED_TOKENS:
        return None, f"wrong_token_count={len(toks)} (expected {EXPECTED_TOKENS})"

    try:
        values = [float(t) for t in toks]
    except ValueError:
        return None, "non_numeric_token"

    class_id = int(values[0])
    bbox = values[1:5]

    keypoints = []
    kp_out_of_range = 0
    invalid_visibility = 0
    for i in range(NUM_KEYPOINTS):
        x, y, v = values[5 + 3 * i], values[6 + 3 * i], values[7 + 3 * i]
        if v not in VALID_VISIBILITY_VALUES:
            invalid_visibility += 1
        if x < 0 or x > 1 or y < 0 or y > 1:
            kp_out_of_range += 1
        keypoints.append((x, y, v))

    obj = {
        "class_id":               class_id,
        "invalid_class_id":       valid_class_ids is not None and class_id not in valid_class_ids,
        "bbox_out_of_range":      any(v < 0 or v > 1 for v in bbox),
        "kp_out_of_range_count":  kp_out_of_range,
        "invalid_visibility_count": invalid_visibility,
        "keypoints":              keypoints,
    }
    return obj, None


def parse_label_file(label_path: Path, valid_class_ids):
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return {"empty": True, "objects": [], "malformed_lines": []}

    objects, malformed_lines = [], []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        obj, err = parse_label_line(line, valid_class_ids)
        if obj is None:
            malformed_lines.append(err)
        else:
            objects.append(obj)
    return {"empty": False, "objects": objects, "malformed_lines": malformed_lines}


def simulate_confidence_array(keypoints):
    """Map GT (x, y, v) label triples to a [12, 3] array for extract_spine_geometry()."""
    arr = np.zeros((NUM_KEYPOINTS, 3), dtype=float)
    for i, (x, y, v) in enumerate(keypoints):
        if v >= 2:
            conf = SIM_CONF_VISIBLE
        elif v >= 1:
            conf = SIM_CONF_OCCLUDED
        else:
            conf = SIM_CONF_ABSENT
        arr[i] = [x, y, conf]
    return arr


# ---------------------------------------------------------------
# Per-split processing
# ---------------------------------------------------------------

def process_split(split: str, valid_class_ids):
    img_dir = DATASET_DIR / split / "images"
    lbl_dir = DATASET_DIR / split / "labels"

    images = {p.stem: p for p in img_dir.glob("*") if p.suffix.lower() in IMAGE_EXTS} if img_dir.exists() else {}
    labels = {p.stem: p for p in lbl_dir.glob("*.txt")} if lbl_dir.exists() else {}

    missing_labels = sorted(set(images) - set(labels))   # image with no matching label file
    missing_images = sorted(set(labels) - set(images))   # label with no matching image file
    paired_stems   = sorted(set(images) & set(labels))

    issue_rows = []
    for stem in missing_labels:
        issue_rows.append({"split": split, "stem": stem, "n_objects": 0, "issues": "missing_label_file"})
    for stem in missing_images:
        issue_rows.append({"split": split, "stem": stem, "n_objects": 0, "issues": "missing_image_file"})

    class_counter          = collections.Counter()
    objects_per_image      = []
    kp_visible_by_idx       = collections.Counter()
    kp_total_objects        = 0
    total_visible_kp_sum    = 0
    unreadable_images       = []
    empty_label_files       = []
    malformed_label_files   = []
    geometry_valid_count    = 0
    geometry_checked_count  = 0
    image_hashes            = {}   # stem -> md5, for leakage check
    sample_pool             = {"normal": [], "scoliosis": [], "problematic": []}

    for stem in sorted(set(images) | set(labels)):
        if stem not in paired_stems:
            continue

        img_path, lbl_path = images[stem], labels[stem]

        try:
            image_hashes[stem] = hashlib.md5(img_path.read_bytes()).hexdigest()
        except Exception:
            pass

        readable = None
        if cv2 is not None:
            readable = imdecode_unicode(img_path) is not None
            if not readable:
                unreadable_images.append(stem)

        parsed = parse_label_file(lbl_path, valid_class_ids)
        row_issues = []

        if parsed["empty"]:
            empty_label_files.append(stem)
            issue_rows.append({"split": split, "stem": stem, "n_objects": 0, "issues": "empty_label_file"})
            continue

        if parsed["malformed_lines"]:
            malformed_label_files.append(stem)
            row_issues.append(f"malformed_lines={len(parsed['malformed_lines'])}")

        n_objects = len(parsed["objects"])
        objects_per_image.append(n_objects)

        obj_class_for_sample = None
        for obj in parsed["objects"]:
            class_counter[obj["class_id"]] += 1
            obj_class_for_sample = obj["class_id"]

            if obj["invalid_class_id"]:
                row_issues.append(f"invalid_class_id={obj['class_id']}")
            if obj["bbox_out_of_range"]:
                row_issues.append("bbox_coord_out_of_range")
            if obj["kp_out_of_range_count"]:
                row_issues.append(f"kp_coord_out_of_range_x{obj['kp_out_of_range_count']}")
            if obj["invalid_visibility_count"]:
                row_issues.append(f"invalid_visibility_x{obj['invalid_visibility_count']}")

            kp_total_objects += 1
            n_visible = 0
            for i, (x, y, v) in enumerate(obj["keypoints"]):
                if v >= 2:
                    kp_visible_by_idx[i] += 1
                    n_visible += 1
            total_visible_kp_sum += n_visible

            geometry_checked_count += 1
            sim_kps = simulate_confidence_array(obj["keypoints"])
            geom = extract_spine_geometry(sim_kps, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD)
            if geom["valid_spine_geometry"]:
                geometry_valid_count += 1

        if readable is False:
            row_issues.append("unreadable_image")

        if row_issues:
            issue_rows.append({
                "split": split, "stem": stem, "n_objects": n_objects,
                "issues": "; ".join(row_issues),
            })

        # Bucket a few images for the optional overlay export
        if cv2 is not None and readable is not False:
            bucket = "problematic" if row_issues else ("scoliosis" if obj_class_for_sample == 1 else "normal")
            if len(sample_pool[bucket]) < 3:
                sample_pool[bucket].append((stem, img_path, parsed["objects"]))

    n_images, n_labels = len(images), len(labels)
    class_total = sum(class_counter.values())
    imbalance_ratio = (
        round(max(class_counter.values()) / max(1, min(class_counter.values())), 2)
        if len(class_counter) >= 2 and min(class_counter.values()) > 0 else None
    )

    summary = {
        "split":                    split,
        "num_images":               n_images,
        "num_label_files":          n_labels,
        "missing_label_files":      len(missing_labels),
        "missing_image_files":      len(missing_images),
        "unreadable_images":        len(unreadable_images) if cv2 is not None else "cv2_unavailable",
        "empty_label_files":        len(empty_label_files),
        "malformed_label_files":    len(malformed_label_files),
        "num_objects":              class_total,
        "avg_objects_per_image":    round(float(np.mean(objects_per_image)), 3) if objects_per_image else None,
        "max_objects_per_image":    max(objects_per_image) if objects_per_image else None,
        "images_with_multi_objects": sum(1 for n in objects_per_image if n > 1),
        "class_imbalance_ratio":    imbalance_ratio,
        "avg_visible_kp_per_object": round(total_visible_kp_sum / kp_total_objects, 3) if kp_total_objects else None,
        "geometry_valid_pct":       round(100 * geometry_valid_count / geometry_checked_count, 1) if geometry_checked_count else None,
        "geometry_checked_count":   geometry_checked_count,
        "geometry_valid_count":     geometry_valid_count,
    }

    return {
        "summary":            summary,
        "class_counter":      class_counter,
        "kp_visible_by_idx":  kp_visible_by_idx,
        "kp_total_objects":   kp_total_objects,
        "issue_rows":         issue_rows,
        "image_hashes":       image_hashes,
        "sample_pool":        sample_pool,
    }


# ---------------------------------------------------------------
# Cross-split checks
# ---------------------------------------------------------------

def check_leakage(split_results):
    name_to_splits = collections.defaultdict(set)
    hash_to_splits = collections.defaultdict(set)

    for split, res in split_results.items():
        for stem, h in res["image_hashes"].items():
            name_to_splits[stem].add(split)
            hash_to_splits[h].add(split)

    dupe_names = {n: sorted(s) for n, s in name_to_splits.items() if len(s) > 1}
    dupe_hashes = {h: sorted(s) for h, s in hash_to_splits.items() if len(s) > 1}
    return dupe_names, dupe_hashes


# ---------------------------------------------------------------
# Aggregation into output tables
# ---------------------------------------------------------------

def build_keypoint_visibility_table(split_results):
    rows = []
    for split, res in split_results.items():
        total = res["kp_total_objects"]
        for idx in range(NUM_KEYPOINTS):
            visible = res["kp_visible_by_idx"].get(idx, 0)
            group = next((g for g, idxs in KP_GROUPS.items() if idx in idxs), "other")
            rows.append({
                "split": split,
                "kp_index": idx,
                "kp_name": KEYPOINT_MAP[idx],
                "group": group,
                "visible_count": visible,
                "total_objects": total,
                "visibility_rate_pct": round(100 * visible / total, 1) if total else None,
            })

    # "ALL" split aggregate
    total_all = sum(r["kp_total_objects"] for r in split_results.values())
    for idx in range(NUM_KEYPOINTS):
        visible_all = sum(res["kp_visible_by_idx"].get(idx, 0) for res in split_results.values())
        group = next((g for g, idxs in KP_GROUPS.items() if idx in idxs), "other")
        rows.append({
            "split": "ALL", "kp_index": idx, "kp_name": KEYPOINT_MAP[idx], "group": group,
            "visible_count": visible_all, "total_objects": total_all,
            "visibility_rate_pct": round(100 * visible_all / total_all, 1) if total_all else None,
        })
    return pd.DataFrame(rows)


def build_class_distribution_table(split_results, class_names):
    rows = []
    all_counter = collections.Counter()
    for split, res in split_results.items():
        total = sum(res["class_counter"].values())
        for class_id, count in sorted(res["class_counter"].items()):
            all_counter[class_id] += count
            rows.append({
                "split": split, "class_id": class_id,
                "class_name": class_names.get(class_id, "unknown"),
                "count": count,
                "pct_of_split": round(100 * count / total, 1) if total else None,
            })
    total_all = sum(all_counter.values())
    for class_id, count in sorted(all_counter.items()):
        rows.append({
            "split": "ALL", "class_id": class_id,
            "class_name": class_names.get(class_id, "unknown"),
            "count": count,
            "pct_of_split": round(100 * count / total_all, 1) if total_all else None,
        })
    return pd.DataFrame(rows)


def build_geometry_readiness(split_results, kp_vis_df):
    def group_rate(split, group):
        sub = kp_vis_df[(kp_vis_df["split"] == split) & (kp_vis_df["group"] == group)]
        if sub.empty or sub["total_objects"].iloc[0] == 0:
            return None
        return round(sub["visible_count"].sum() / (sub["total_objects"].iloc[0] * len(sub)) * 100, 1)

    total_checked = sum(res["summary"]["geometry_checked_count"] for res in split_results.values())
    total_valid   = sum(res["summary"]["geometry_valid_count"] for res in split_results.values())
    all_geometry_valid_pct = round(100 * total_valid / total_checked, 1) if total_checked else None

    rows = []
    for split in list(split_results.keys()) + ["ALL"]:
        geometry_valid_pct = (
            split_results[split]["summary"]["geometry_valid_pct"] if split != "ALL" else all_geometry_valid_pct
        )
        rows.append({
            "split": split,
            "base_top_visibility_pct":     group_rate(split, "spine_base_top"),
            "intermediate_visibility_pct": group_rate(split, "spine_intermediate"),
            "shoulder_visibility_pct":     group_rate(split, "shoulders"),
            "geometry_valid_pct":          geometry_valid_pct,
        })
    return pd.DataFrame(rows)


def geometry_reliability_verdict(intermediate_visibility_pct):
    if intermediate_visibility_pct is None:
        return "unknown (no data)"
    if intermediate_visibility_pct >= 90:
        return "reliable — treat lateral-deviation/curve-smoothness features as trustworthy"
    if intermediate_visibility_pct >= 60:
        return "weak — usable for screening-support trends, not per-subject decisions"
    return "experimental — intermediate points are too often missing/occluded in ground truth to trust curvature features"


# ---------------------------------------------------------------
# Sample overlays
# ---------------------------------------------------------------

def draw_overlay(img, objects):
    out = img.copy()
    h, w = out.shape[:2]
    colors = {
        "spine_base_top":     (0, 255, 0),
        "spine_intermediate": (255, 150, 0),
        "shoulders":          (255, 0, 255),
        "head_neck":          (0, 255, 255),
    }
    for obj in objects:
        base = top = None
        for i, (x, y, v) in enumerate(obj["keypoints"]):
            if v < 2:
                continue
            px, py = int(x * w), int(y * h)
            group = next((g for g, idxs in KP_GROUPS.items() if i in idxs), "other")
            cv2.circle(out, (px, py), 5, colors.get(group, (200, 200, 200)), -1)
            if i == SPINE_BASE_LUMBAR:
                base = (px, py)
            if i == SPINE_TOP_CERVICAL:
                top = (px, py)
        if base and top:
            cv2.line(out, base, top, (0, 200, 200), 2)
    return out


def save_sample_overlays(split_results, class_names):
    if cv2 is None:
        print("[WARN] cv2 not available in this interpreter — skipping sample overlays.")
        return

    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    saved = 0
    for split, res in split_results.items():
        for bucket, samples in res["sample_pool"].items():
            for stem, img_path, objects in samples:
                img = imdecode_unicode(img_path)
                if img is None:
                    continue
                annotated = draw_overlay(img, objects)
                label = bucket if bucket == "problematic" else class_names.get(
                    objects[0]["class_id"] if objects else -1, bucket
                )
                cv2.putText(annotated, f"{split}/{label}: {stem}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                out_path = OVERLAY_DIR / f"{bucket}_{split}_{stem}.jpg"
                cv2.imencode(".jpg", annotated)[1].tofile(str(out_path))
                saved += 1
    print(f"[INFO] Saved {saved} sample overlay image(s) to {OVERLAY_DIR.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------

def df_to_markdown(df: pd.DataFrame) -> str:
    """Minimal pipe-table formatter — avoids adding a `tabulate` dependency
    just for pandas.DataFrame.to_markdown()."""
    if df.empty:
        return "_(no rows)_"
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |"]
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def write_markdown_report(summary_df, class_df, kp_vis_df, issues_df, geom_df, dupe_names, dupe_hashes, class_names):
    lines = []
    lines.append("# Dataset EDA Report — public_datasets/scoliosis\n")
    lines.append("Auto-generated by `src/Evaluation/dataset_eda.py`. Read-only inspection of the "
                  "train/valid/test splits; does not modify the dataset, model, or geometry code.\n")

    lines.append("## 1. Split overview\n")
    lines.append(df_to_markdown(summary_df))
    lines.append("")

    lines.append("## 2. Class distribution & imbalance\n")
    lines.append(df_to_markdown(class_df))
    lines.append("")
    lines.append("Classes: " + ", ".join(f"{k}={v}" for k, v in sorted(class_names.items())))
    lines.append("")

    lines.append("## 3. Label quality issues\n")
    if issues_df.empty:
        lines.append("No label/image issues found across any split.\n")
    else:
        lines.append(f"{len(issues_df)} file(s) flagged. See `label_quality_issues.csv` for the full list. "
                      f"First {min(15, len(issues_df))} shown below:\n")
        lines.append(df_to_markdown(issues_df.head(15)))
    lines.append("")

    lines.append("## 4. Train/valid/test leakage check\n")
    lines.append(f"- Duplicate filenames across splits: **{len(dupe_names)}**")
    lines.append(f"- Duplicate image content (md5) across splits: **{len(dupe_hashes)}**")
    if dupe_hashes:
        lines.append("\n  Content-identical images across splits (possible leakage):")
        for h, splits in list(dupe_hashes.items())[:10]:
            lines.append(f"  - `{h[:10]}...` present in: {', '.join(splits)}")
    lines.append("")

    lines.append("## 5. Keypoint visibility by index\n")
    lines.append(df_to_markdown(kp_vis_df[kp_vis_df["split"] == "ALL"]))
    lines.append("")

    lines.append("## 6. Geometry readiness (see spine_geometry.py)\n")
    lines.append(df_to_markdown(geom_df))
    lines.append("")

    all_row = geom_df[geom_df["split"] == "ALL"].iloc[0]
    verdict = geometry_reliability_verdict(all_row["intermediate_visibility_pct"])
    lines.append("**Answers:**\n")
    lines.append(f"- kp[0]/kp[8] (spine base/top) visible enough for spine-axis inclination: "
                  f"**{all_row['base_top_visibility_pct']}%** of objects.")
    lines.append(f"- kp[1..7] (intermediate spine) visible enough for lateral deviation / curve smoothness: "
                  f"**{all_row['intermediate_visibility_pct']}%** average per-point visibility.")
    lines.append(f"- kp[9]/kp[10] (shoulders) visible enough for shoulder asymmetry: "
                  f"**{all_row['shoulder_visibility_pct']}%** of objects.")
    lines.append(f"- Labels that would satisfy `valid_spine_geometry` if fed through as ground truth: "
                  f"**{all_row['geometry_valid_pct']}%**.")
    lines.append(f"- Reliability verdict for intermediate-point-dependent features "
                  f"(`max/mean_lateral_deviation_px`, `curve_smoothness_score`): **{verdict}**.")
    lines.append("")
    lines.append("> Note: this measures **ground-truth label completeness**, not the trained model's "
                  "runtime keypoint confidence. A well-labeled dataset does not guarantee the model will "
                  "predict those points with high confidence at inference — that is a separate question "
                  "answered by `run_evaluation.py` / `compare_workflows.py`.\n")

    (OUTPUT_DIR / "dataset_eda_report.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------

def print_console_summary(summary_df, class_df, geom_df, issues_df):
    total_images = int(summary_df["num_images"].sum())
    total_issues = len(issues_df)

    print(f"\n{'='*60}")
    print("  DATASET EDA SUMMARY")
    print(f"{'='*60}")
    print(f"  Total images checked : {total_images}")
    for _, row in summary_df.iterrows():
        print(f"    {row['split']:<6}: {row['num_images']} images, {row['num_objects']} objects, "
              f"imbalance {row['class_imbalance_ratio']}")

    print(f"\n  Class counts (ALL splits):")
    all_classes = class_df[class_df["split"] == "ALL"]
    for _, row in all_classes.iterrows():
        print(f"    {row['class_name']:<12}: {row['count']} ({row['pct_of_split']}%)")

    all_geom = geom_df[geom_df["split"] == "ALL"].iloc[0]
    print(f"\n  Keypoint visibility (ALL splits):")
    print(f"    spine base/top (kp0,kp8) : {all_geom['base_top_visibility_pct']}%")
    print(f"    intermediate  (kp1..7)   : {all_geom['intermediate_visibility_pct']}%")
    print(f"    shoulders     (kp9,kp10) : {all_geom['shoulder_visibility_pct']}%")
    print(f"    valid_spine_geometry rate: {all_geom['geometry_valid_pct']}%")
    print(f"    intermediate-kp reliability: {geometry_reliability_verdict(all_geom['intermediate_visibility_pct'])}")

    print(f"\n  Problematic labels/images flagged: {total_issues}")
    print(f"\n  Full report: Data/Processed/eda/dataset_eda_report.md")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def run(save_overlays: bool = True):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    class_names, nc = load_class_names(DATA_YAML)
    valid_class_ids = set(class_names.keys()) if class_names else None
    if not class_names:
        print(f"[WARN] Could not load class names from {DATA_YAML} — class-id validity check disabled.")

    split_results = {split: process_split(split, valid_class_ids) for split in SPLITS}

    summary_df = pd.DataFrame([res["summary"] for res in split_results.values()])
    class_df   = build_class_distribution_table(split_results, class_names)
    kp_vis_df  = build_keypoint_visibility_table(split_results)
    geom_df    = build_geometry_readiness(split_results, kp_vis_df)

    issue_rows = [row for res in split_results.values() for row in res["issue_rows"]]
    issues_df  = pd.DataFrame(issue_rows) if issue_rows else pd.DataFrame(
        columns=["split", "stem", "n_objects", "issues"]
    )

    dupe_names, dupe_hashes = check_leakage(split_results)

    summary_df.to_csv(OUTPUT_DIR / "dataset_eda_summary.csv", index=False)
    kp_vis_df.to_csv(OUTPUT_DIR / "keypoint_visibility_by_split.csv", index=False)
    class_df.to_csv(OUTPUT_DIR / "class_distribution_by_split.csv", index=False)
    issues_df.to_csv(OUTPUT_DIR / "label_quality_issues.csv", index=False)

    write_markdown_report(summary_df, class_df, kp_vis_df, issues_df, geom_df, dupe_names, dupe_hashes, class_names)

    if save_overlays:
        save_sample_overlays(split_results, class_names)

    print_console_summary(summary_df, class_df, geom_df, issues_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset EDA & data-quality report for public_datasets/scoliosis.")
    parser.add_argument("--no-overlays", action="store_true", help="Skip saving sample overlay images.")
    args = parser.parse_args()
    run(save_overlays=not args.no_overlays)
