# ============================================================
# Split-Quality Check — original and cleaned scoliosis datasets
#
# Read-only inspection of the train/valid/test splits, to answer: "is the
# held-out evaluation actually held out, and is it big/balanced enough to trust?"
#
# Reports, per split: image/label counts, class distribution, normal/scoliosis
# ratio, missing/invalid labels. Across splits: leakage, checked three ways —
#   1. identical filename in two splits
#   2. identical file bytes (SHA-1) in two splits
#   3. near-identical *content*: same source photo re-exported under a different
#      Roboflow hash (e.g. `300_jpg.rf.<hashA>` vs `300_jpg.rf.<hashB>`), found by
#      comparing small greyscale thumbnails. Byte-hash checks miss these.
# Then warns about severe class imbalance, a very small test set, and splits
# whose class mix differs a lot from train.
#
# Does NOT modify any dataset. Writes only under Data/Processed/evaluation/.
#
# Usage:
#   python split_quality_check.py                  (original + cleaned)
#   python split_quality_check.py --dataset original
#   python split_quality_check.py --dataset cleaned
#   python split_quality_check.py --dataset grouped      (after create_grouped_split.py)
# ============================================================

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import argparse
import hashlib
import collections
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.Evaluation.dataset_eda import (
    SPLITS, IMAGE_EXTS, parse_label_file, df_to_markdown,
)

DATASETS = {
    "original": PROJECT_ROOT / "public_datasets" / "scoliosis",
    "cleaned":  PROJECT_ROOT / "Data" / "Processed" / "scoliosis_cleaned",
    "grouped":  PROJECT_ROOT / "Data" / "Processed" / "scoliosis_grouped_split",   # from create_grouped_split.py
}
CLASS_NAMES = {0: "normal", 1: "scoliosis"}
OUTPUT_DIR  = PROJECT_ROOT / "Data" / "Processed" / "evaluation"

THUMB_SIZE            = 32      # greyscale thumbnail side used for the near-duplicate check
NEAR_DUP_CORRELATION  = 0.97    # thumbnail correlation above this = "same photo"
SEVERE_MINORITY_SHARE = 0.25    # a split whose rarer class is below this share is severely imbalanced
SMALL_TEST_IMAGES     = 30      # fewer test images than this = very small
SMALL_TEST_PER_CLASS  = 10      # fewer test images of one class than this = too few to judge that class
MIX_DIFFERENCE_PTS    = 15.0    # a split's scoliosis share this many points off train's is flagged


# ---------------------------------------------------------------
# Shared helpers (also used by create_kfold_splits.py)
# ---------------------------------------------------------------

def source_key(stem: str) -> str:
    """Roboflow names files `<source>.rf.<hash>`; the part before `.rf.` identifies the source photo."""
    return stem.split(".rf.")[0]


def list_split(root: Path, split: str):
    """One record per image in a split: stem, image path, label path (or None if missing)."""
    image_dir, label_dir = root / split / "images", root / split / "labels"
    records = []
    if not image_dir.exists():
        return records
    for img in sorted(image_dir.iterdir()):
        if img.suffix.lower() not in IMAGE_EXTS:
            continue
        label = label_dir / f"{img.stem}.txt"
        records.append({"split": split, "stem": img.stem, "image": img, "label": label if label.exists() else None})
    return records


def label_summary(label_path: Path):
    """(class_id or None, n_objects, problem or None) for a label file."""
    if label_path is None:
        return None, 0, "missing_label_file"
    parsed = parse_label_file(label_path, set(CLASS_NAMES))
    if parsed["empty"]:
        return None, 0, "empty_label_file"
    problems = []
    if parsed["malformed_lines"]:
        problems.append("malformed_row")
    if any(o["invalid_class_id"] for o in parsed["objects"]):
        problems.append("invalid_class_id")
    if any(o["kp_out_of_range_count"] or o["bbox_out_of_range"] for o in parsed["objects"]):
        problems.append("coordinate_out_of_range")
    if not parsed["objects"]:
        return None, 0, "; ".join(problems) or "no_valid_rows"
    # An image's class = class of its first valid object (matches how evaluation reads labels).
    return parsed["objects"][0]["class_id"], len(parsed["objects"]), ("; ".join(problems) or None)


def sha1_of(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def thumbnail_vector(path: Path):
    """Zero-mean, unit-norm greyscale thumbnail — a dot product of two of these is their correlation."""
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    v = cv2.resize(img, (THUMB_SIZE, THUMB_SIZE), interpolation=cv2.INTER_AREA).astype(np.float64).ravel()
    v -= v.mean()
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else None


def near_duplicate_pairs(records, threshold: float = NEAR_DUP_CORRELATION):
    """All (i, j, correlation) pairs of records whose thumbnails correlate above `threshold`."""
    vecs = [thumbnail_vector(r["image"]) for r in records]
    ok = [i for i, v in enumerate(vecs) if v is not None]
    if len(ok) < 2:
        return []
    mat = np.stack([vecs[i] for i in ok])
    corr = mat @ mat.T
    pairs = []
    for a in range(len(ok)):
        for b in range(a + 1, len(ok)):
            if corr[a, b] >= threshold:
                pairs.append((ok[a], ok[b], float(corr[a, b])))
    return pairs


# ---------------------------------------------------------------
# Per-dataset analysis
# ---------------------------------------------------------------

def analyse(name: str, root: Path):
    records = [r for s in SPLITS for r in list_split(root, s)]

    for r in records:
        r["class_id"], r["n_objects"], r["problem"] = label_summary(r["label"])
        r["sha1"] = sha1_of(r["image"])
        r["source"] = source_key(r["stem"])

    # --- per-split table ---
    rows = []
    for split in SPLITS:
        recs = [r for r in records if r["split"] == split]
        label_files = len(list((root / split / "labels").glob("*.txt"))) if (root / split / "labels").exists() else 0
        counts = collections.Counter(r["class_id"] for r in recs if r["class_id"] is not None)
        n_normal, n_scoli = counts.get(0, 0), counts.get(1, 0)
        labelled = n_normal + n_scoli
        problems = [r for r in recs if r["problem"]]
        rows.append({
            "dataset": name, "split": split,
            "images": len(recs), "label_files": label_files,
            "normal": n_normal, "scoliosis": n_scoli,
            "scoliosis_share_pct": round(100 * n_scoli / labelled, 1) if labelled else None,
            "normal_to_scoliosis": f"{n_normal}:{n_scoli}",
            "missing_or_invalid_labels": len(problems),
        })
    summary = pd.DataFrame(rows)

    # --- leakage across splits ---
    by_split = {s: [r for r in records if r["split"] == s] for s in SPLITS}
    name_splits, hash_splits, source_splits = (collections.defaultdict(set) for _ in range(3))
    for r in records:
        name_splits[r["stem"]].add(r["split"])
        hash_splits[r["sha1"]].add(r["split"])
        source_splits[r["source"]].add(r["split"])
    dup_names  = {k: sorted(v) for k, v in name_splits.items() if len(v) > 1}
    dup_hashes = {k: sorted(v) for k, v in hash_splits.items() if len(v) > 1}
    shared_sources = {k: sorted(v) for k, v in source_splits.items() if len(v) > 1}

    all_pairs = [(records[i], records[j], c) for i, j, c in near_duplicate_pairs(records)]
    near_pairs = [(a, b, c) for a, b, c in all_pairs if a["split"] != b["split"]]
    conflicts = [(a, b, c) for a, b, c in all_pairs
                 if a["class_id"] is not None and b["class_id"] is not None and a["class_id"] != b["class_id"]]
    leaked = {s: set() for s in SPLITS}      # images in `s` that have a near-duplicate in another split
    for a, b, _ in near_pairs:
        leaked[a["split"]].add(a["stem"])
        leaked[b["split"]].add(b["stem"])

    leakage = {
        "dup_names": dup_names, "dup_hashes": dup_hashes, "shared_sources": shared_sources,
        "near_pairs": near_pairs, "leaked": leaked, "conflicts": conflicts,
        "n_sources": len(source_splits), "n_images": len(records),
    }
    for row in rows:
        row["images_with_near_duplicate_in_other_split"] = len(leaked[row["split"]])
    summary = pd.DataFrame(rows)

    return {"name": name, "root": root, "records": records, "summary": summary, "leakage": leakage}


def build_warnings(result):
    warnings = []
    summary, leak = result["summary"], result["leakage"]
    srow = {r["split"]: r for r in summary.to_dict("records")}

    for split in SPLITS:
        r = srow[split]
        labelled = r["normal"] + r["scoliosis"]
        if labelled and min(r["normal"], r["scoliosis"]) / labelled < SEVERE_MINORITY_SHARE:
            warnings.append(f"**{split}**: severe class imbalance ({r['normal_to_scoliosis']} normal:scoliosis).")
        if r["missing_or_invalid_labels"]:
            warnings.append(f"**{split}**: {r['missing_or_invalid_labels']} image(s) with a missing/empty/invalid label.")
        if r["images"] != r["label_files"]:
            warnings.append(f"**{split}**: {r['images']} images but {r['label_files']} label files.")

    test = srow["test"]
    if test["images"] < SMALL_TEST_IMAGES:
        warnings.append(f"**test**: only {test['images']} images — one or two predictions move accuracy by 5–10 points.")
    for cls in ("normal", "scoliosis"):
        if 0 < test[cls] < SMALL_TEST_PER_CLASS:
            warnings.append(f"**test**: only {test[cls]} {cls} images — too few to judge {cls} performance reliably.")

    train_share = srow["train"]["scoliosis_share_pct"]
    for split in ("valid", "test"):
        share = srow[split]["scoliosis_share_pct"]
        if train_share is not None and share is not None and abs(share - train_share) >= MIX_DIFFERENCE_PTS:
            warnings.append(f"**{split}**: scoliosis share is {share}% vs {train_share}% in train — class mix differs between splits.")

    if leak["dup_names"]:
        warnings.append(f"{len(leak['dup_names'])} filename(s) appear in more than one split.")
    if leak["dup_hashes"]:
        warnings.append(f"{len(leak['dup_hashes'])} byte-identical image(s) appear in more than one split.")
    for split in ("valid", "test"):
        n_leaked, n_total = len(leak["leaked"][split]), srow[split]["images"]
        if n_leaked:
            warnings.append(
                f"**LEAKAGE — {split}**: {n_leaked} of {n_total} images have a near-identical copy of the same "
                f"photo in another split (re-exported under a different file hash). "
                f"Metrics on `{split}` are optimistic.")
    if leak["conflicts"]:
        warnings.append(f"**LABEL CONFLICT**: {len(leak['conflicts'])} pair(s) of near-identical photos carry different "
                        f"class labels (normal vs scoliosis) — at least one label in each pair is wrong.")
    return warnings


# ---------------------------------------------------------------
# Report output
# ---------------------------------------------------------------

def write_report(result, out_path: Path):
    summary, leak = result["summary"], result["leakage"]
    show = summary[["split", "images", "label_files", "normal", "scoliosis", "scoliosis_share_pct",
                    "normal_to_scoliosis", "missing_or_invalid_labels",
                    "images_with_near_duplicate_in_other_split"]].rename(columns={
        "label_files": "labels", "scoliosis_share_pct": "scoliosis %", "normal_to_scoliosis": "normal:scoliosis",
        "missing_or_invalid_labels": "bad labels", "images_with_near_duplicate_in_other_split": "near-dup in other split"})

    warnings = build_warnings(result)
    rel_root = result["root"].relative_to(PROJECT_ROOT)
    lines = [f"# Split-Quality Report — {result['name']} dataset\n",
             f"Auto-generated by `split_quality_check.py` from `{rel_root.as_posix()}`. Read-only; nothing was modified.\n",
             "## Split sizes and class mix\n", df_to_markdown(show), ""]

    lines.append("## Leakage between splits\n")
    lines.append(f"- {leak['n_images']} images come from **{leak['n_sources']} distinct source photos** "
                 f"(the part of the filename before `.rf.`).")
    lines.append(f"- Identical filenames across splits: **{len(leak['dup_names'])}**")
    lines.append(f"- Byte-identical images across splits: **{len(leak['dup_hashes'])}**")
    lines.append(f"- Source photos present in more than one split: **{len(leak['shared_sources'])}**")
    lines.append(f"- Near-identical image pairs across splits (thumbnail correlation ≥ {NEAR_DUP_CORRELATION}): "
                 f"**{len(leak['near_pairs'])}**\n")
    if leak["near_pairs"]:
        example = leak["near_pairs"][:5]
        ex = pd.DataFrame([{"image A": f"{a['split']}/{source_key(a['stem'])}", "image B": f"{b['split']}/{source_key(b['stem'])}",
                            "correlation": round(c, 3)} for a, b, c in example])
        lines.append("Examples:\n")
        lines.append(df_to_markdown(ex))
        lines.append("")

    if leak["conflicts"]:
        lines.append("Near-identical photos with conflicting class labels:\n")
        cf = pd.DataFrame([{"image A": f"{a['split']}/{a['stem']}", "class A": CLASS_NAMES[a["class_id"]],
                            "image B": f"{b['split']}/{b['stem']}", "class B": CLASS_NAMES[b["class_id"]],
                            "correlation": round(c, 3)} for a, b, c in leak["conflicts"]])
        lines.append(df_to_markdown(cf))
        lines.append("")

    lines.append("## Warnings\n")
    lines.extend([f"- {w}" for w in warnings] or ["- None."])
    lines.append("")
    lines.append("## How to read this\n")
    lines.append("- Near-duplicates mean the same photo sits in train and in valid/test, so the model is partly "
                 "being tested on images it has effectively seen. Treat valid/test metrics as optimistic until the "
                 "splits are rebuilt grouped by source photo (see `create_kfold_splits.py`, which groups this way).")
    lines.append("- These are screening-support checks on dataset quality, not a clinical assessment.")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(which):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    names = ["original", "cleaned"] if which == "both" else [which]
    summaries = []
    for name in names:
        root = DATASETS[name]
        if not root.exists():
            print(f"[SKIP] {name}: {root} not found")
            continue
        result = analyse(name, root)
        out_path = OUTPUT_DIR / f"split_quality_report_{name}.md"
        write_report(result, out_path)
        summaries.append(result["summary"])

        print(f"\n{'='*60}\n  SPLIT QUALITY — {name.upper()}\n{'='*60}")
        print(result["summary"].drop(columns=["dataset"]).to_string(index=False))
        warnings = build_warnings(result)
        print(f"\n  {len(warnings)} warning(s):")
        for w in warnings:
            print(f"   - {w.replace('**', '')}")
        print(f"\n  Report: {out_path.relative_to(PROJECT_ROOT)}")

    if summaries:
        csv_path = OUTPUT_DIR / "split_quality_summary.csv"
        new = pd.concat(summaries, ignore_index=True)
        if csv_path.exists():   # keep rows for datasets that were not re-run this time
            old = pd.read_csv(csv_path)
            new = pd.concat([old[~old["dataset"].isin(new["dataset"])], new], ignore_index=True)
        new["dataset"] = pd.Categorical(new["dataset"], [d for d in DATASETS if d in set(new["dataset"])], ordered=True)
        new.sort_values(["dataset", "split"], key=lambda c: c.map({"train": 0, "valid": 1, "test": 2}) if c.name == "split" else c,
                        inplace=True)
        new.to_csv(csv_path, index=False)
        print(f"  Summary: {csv_path.relative_to(PROJECT_ROOT)}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check train/valid/test split quality for the scoliosis datasets.")
    parser.add_argument("--dataset", choices=["original", "cleaned", "grouped", "both"], default="both",
                         help="Which dataset to inspect (default: both = original + cleaned; "
                              "grouped = the split from create_grouped_split.py).")
    run(parser.parse_args().dataset)
