# ============================================================
# Photo-Grouped, Class-Stratified Train/Valid/Test Re-Split
#
# The original train/valid/test split puts copies of the same photo in
# different splits (see split_quality_check.py), so valid/test metrics are
# optimistic. This script builds a NEW split of the cleaned dataset in which
# every photo lives in exactly one split.
#
# How it works:
#   1. Group images by photo: same filename before `.rf.`, OR near-identical pixels
#      under another name (same grouping create_kfold_splits.py uses).
#   2. Photo groups whose images carry DIFFERENT class labels (normal vs scoliosis)
#      are excluded from the split and written to a separate conflict report —
#      the safe choice, because their true class is unknown. By default ANY mixed-label
#      group is excluded (--conflict-scope any). --conflict-scope pixel excludes only
#      groups containing a near-identical pair with opposite labels, and keeps the
#      other mixed groups (still listed in the report for manual review).
#   3. Remaining groups are assigned WHOLE (never image by image) with a grouped,
#      class-stratified splitter: first a test cut, then a validation cut from the rest.
#   4. Train/valid/test folders and a YOLO yaml are written; the split is re-checked
#      for leakage with the independent checks in split_quality_check.py.
#
# Does NOT train anything and never modifies the source dataset.
#
# Usage:
#   python create_grouped_split.py
#   python create_grouped_split.py --seed 7 --test-fraction 0.15 --valid-fraction 0.15
#   python create_grouped_split.py --force          (rebuild; replaces generated files in --output-dir only)
# ============================================================

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import shutil
import argparse
import collections
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.Evaluation.dataset_eda import SPLITS, df_to_markdown
from src.Evaluation.create_kfold_splits import build_groups, resolve_dataset_root
from src.Evaluation.split_quality_check import (
    DATASETS, CLASS_NAMES, NEAR_DUP_CORRELATION,
    list_split, label_summary, near_duplicate_pairs, source_key,
)

DEFAULT_YAML   = PROJECT_ROOT / "Data" / "Processed" / "scoliosis_cleaned" / "scoliosis_cleaned.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "Data" / "Processed" / "scoliosis_grouped_split"

MIN_TEST_PER_CLASS = 10   # warn if the test set has fewer images of either class than this


def grouped_stratified_cut(indices, y, groups, n_splits, seed):
    """One cut: returns (kept_indices, held_out_indices). Whole groups go to one side, classes stay balanced."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    keep, held = next(sgkf.split(np.zeros(len(indices)), y[indices], groups[indices]))
    return indices[keep], indices[held]


def folds_for(fraction_of_total, already_removed_share):
    """Number of folds whose single held-out fold is ~fraction_of_total of ALL images."""
    share_of_remaining = fraction_of_total / (1.0 - already_removed_share)
    return max(2, int(round(1.0 / share_of_remaining)))


def find_conflict_groups(records, groups):
    """Groups whose images carry more than one class label."""
    by_group = collections.defaultdict(list)
    for i, g in enumerate(groups):
        by_group[g].append(i)
    return {g: idxs for g, idxs in by_group.items() if len({records[i]["class_id"] for i in idxs}) > 1}


def copy_split(out_dir: Path, split_name: str, recs):
    (out_dir / split_name / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / split_name / "labels").mkdir(parents=True, exist_ok=True)
    for r in recs:
        shutil.copy2(r["image"], out_dir / split_name / "images" / r["image"].name)
        shutil.copy2(r["label"], out_dir / split_name / "labels" / r["label"].name)


def write_yaml(out_dir: Path, source_cfg: dict):
    cfg = {
        "path":      str(out_dir),
        "train":     "train/images",
        "val":       "valid/images",
        "test":      "test/images",
        "kpt_shape": source_cfg.get("kpt_shape", [12, 3]),
        "flip_idx":  source_cfg.get("flip_idx", list(range(12))),
        "nc":        source_cfg.get("nc", 2),
        "names":     source_cfg.get("names", CLASS_NAMES),
    }
    path = out_dir / "scoliosis_grouped_split.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    return path


def verify_no_leakage(out_dir: Path):
    """Independent re-check on the written folders (does not reuse the grouping that made the split)."""
    recs = [r for s in SPLITS for r in list_split(out_dir, s)]
    names, sources = collections.defaultdict(set), collections.defaultdict(set)
    for r in recs:
        names[r["stem"]].add(r["split"])
        sources[source_key(r["stem"])].add(r["split"])
    cross_pairs = [(recs[i], recs[j], c) for i, j, c in near_duplicate_pairs(recs) if recs[i]["split"] != recs[j]["split"]]
    return {
        "images": len(recs),
        "filenames_in_multiple_splits": sum(len(v) > 1 for v in names.values()),
        "source_photos_in_multiple_splits": sum(len(v) > 1 for v in sources.values()),
        "near_duplicate_pairs_across_splits": len(cross_pairs),
    }


def split_table(records, split_idx, groups):
    rows = []
    for name, idxs in split_idx.items():
        classes = collections.Counter(records[i]["class_id"] for i in idxs)
        n, s = classes.get(0, 0), classes.get(1, 0)
        rows.append({"split": name, "images": len(idxs), "normal": n, "scoliosis": s,
                     "scoliosis %": round(100 * s / (n + s), 1) if n + s else None,
                     "photo groups": len({groups[i] for i in idxs})})
    total_n = sum(r["normal"] for r in rows); total_s = sum(r["scoliosis"] for r in rows)
    rows.append({"split": "total", "images": total_n + total_s, "normal": total_n, "scoliosis": total_s,
                 "scoliosis %": round(100 * total_s / (total_n + total_s), 1),
                 "photo groups": sum(r["photo groups"] for r in rows)})
    return pd.DataFrame(rows)


def write_reports(out_dir, args, data_yaml, table, conflict_rows, skipped, leak, warnings, manifest_path):
    rel = lambda p: p.relative_to(PROJECT_ROOT).as_posix() if PROJECT_ROOT in p.parents else str(p)
    ok = (leak["filenames_in_multiple_splits"] == 0 and leak["source_photos_in_multiple_splits"] == 0
          and leak["near_duplicate_pairs_across_splits"] == 0)

    lines = ["# Grouped Re-Split Report\n",
             f"Auto-generated by `create_grouped_split.py` from `{rel(data_yaml)}` "
             f"(seed {args.seed}, test ≈ {args.test_fraction:.0%}, valid ≈ {args.valid_fraction:.0%}). "
             "Nothing was trained; the source dataset was not modified.\n",
             "## Class counts per split\n", df_to_markdown(table), "",
             "Whole photo groups were assigned to one split each (grouped, class-stratified). "
             "A group = images with the same filename before `.rf.`, or near-identical pixels "
             f"(thumbnail correlation ≥ {NEAR_DUP_CORRELATION}).\n",
             "## Leakage check on the written folders\n",
             f"- Result: **{'PASS — no photo appears in more than one split' if ok else 'FAIL — see counts below'}**",
             f"- Filenames in more than one split: **{leak['filenames_in_multiple_splits']}**",
             f"- Source photos in more than one split: **{leak['source_photos_in_multiple_splits']}**",
             f"- Near-identical image pairs across splits: **{leak['near_duplicate_pairs_across_splits']}**\n",
             "## Excluded images\n",
             f"- **{sum(len(r['images']) for r in conflict_rows if r['excluded'])} images in "
             f"{sum(r['excluded'] for r in conflict_rows)} conflicting photo group(s)** are not in any split — see "
             f"`conflict_report.md` (conflict scope: `{args.conflict_scope}`; "
             f"{sum(not r['excluded'] for r in conflict_rows)} mixed-label group(s) kept).",
             f"- {len(skipped)} image(s) skipped for having no usable label.\n"]
    lines.append("## Warnings\n")
    lines.extend([f"- {w}" for w in warnings] or ["- None."])
    lines += ["", "## Using the split\n",
              "Nothing has been trained. To train on it (slow on CPU), for example:\n",
              "```bash",
              f"python src/Evaluation/train_yolo_scoliosis.py --data-yaml {rel(out_dir / 'scoliosis_grouped_split.yaml')} \\",
              "    --project models/yolo_scoliosis_grouped_conservative --augment-profile conservative",
              "```\n",
              "Evaluate a model trained on this split on its own test folder (`run_evaluation.py` reads the split from the yaml):\n",
              "```bash",
              "python src/Evaluation/run_evaluation.py --model yolo_finetuned --weights <best.pt> \\",
              f"    --data-yaml {rel(out_dir / 'scoliosis_grouped_split.yaml')} --split test --tag <name>",
              "```\n",
              "**Models trained on the old split should not be interpreted as unbiased on this grouped test set** — their "
              "training data contains photos that are now in test. Their results are historical/debug only. Retrain on the "
              "grouped train/valid split before reporting grouped-split performance.\n",
              f"Per-image assignments: `{rel(manifest_path)}`. Screening-support research only — not a diagnosis."]
    (out_dir / "split_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    c = ["# Conflicting-Label Photo Groups\n",
         "These photo groups contain near-identical or same-named images that carry **different class labels** "
         "(normal vs scoliosis). At least one label in each group is wrong, so the group was left out of the "
         "grouped split rather than guessed. Review each by hand, then either fix the labels or keep them excluded.\n"]
    if not conflict_rows:
        c.append("_No conflicting groups found._")
    for k, r in enumerate(conflict_rows, 1):
        c.append(f"## Group {k} — {'EXCLUDED' if r['excluded'] else 'kept (review)'}\n")
        c.append(df_to_markdown(pd.DataFrame(r["images"])))
        c.append("")
        if r["why"]:
            c.append(f"_{r['why']}_\n")
    (out_dir / "conflict_report.md").write_text("\n".join(c) + "\n", encoding="utf-8")


def run(args):
    data_yaml = args.data_yaml.resolve()
    out_dir = args.output_dir.resolve()
    if not data_yaml.exists():
        print(f"[ERROR] data yaml not found: {data_yaml}")
        return 1
    source_cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    root = resolve_dataset_root(data_yaml)

    protected = {root.resolve(), *[p.resolve() for k, p in DATASETS.items() if k != "grouped"]}
    if any(out_dir == p or p in out_dir.parents or out_dir in p.parents for p in protected):
        print(f"[ERROR] --output-dir must be separate from the source datasets: {out_dir}")
        return 1
    generated = [out_dir / s for s in SPLITS] + [out_dir / n for n in
                 ("split_report.md", "conflict_report.md", "split_manifest.csv", "scoliosis_grouped_split.yaml")]
    if any(p.exists() for p in generated) and not args.force:
        print(f"[ERROR] {out_dir} already contains a split. Re-run with --force to rebuild it.")
        return 1

    records = [r for s in SPLITS for r in list_split(root, s)]
    usable, skipped = [], []
    for r in records:
        r["class_id"], _, _ = label_summary(r["label"])
        (skipped if r["class_id"] is None or r["label"] is None else usable).append(r)

    groups_all = np.array(build_groups(usable))

    # --- conflicting groups: excluded, reported separately ---
    conflicts = find_conflict_groups(usable, groups_all)
    conflict_rows, excluded = [], set()
    for g, idxs in conflicts.items():
        pairs = {(i, j): c for i, j, c in near_duplicate_pairs([usable[k] for k in idxs])}
        has_pixel_dup = any(usable[idxs[a]]["class_id"] != usable[idxs[b]]["class_id"] for (a, b) in pairs)
        exclude = has_pixel_dup or args.conflict_scope == "any"
        why = ("Near-identical pixels with opposite labels." if has_pixel_dup
               else "Mixed labels in a group linked by shared filenames and/or chains of near-identical images, "
                    "but no single near-identical pair disagrees — these may be different people grouped together by "
                    "mistake. Review by hand.")
        conflict_rows.append({"why": why + ("" if exclude else " KEPT in the split (--conflict-scope pixel); the whole "
                                            "group is assigned to one split."), "excluded": exclude,
                              "images": [{"split (original)": usable[i]["split"], "image": usable[i]["stem"],
                                          "class": CLASS_NAMES[usable[i]["class_id"]]} for i in idxs]})
        if exclude:
            excluded.update(idxs)

    keep = np.array([i for i in range(len(usable)) if i not in excluded])
    y = np.array([r["class_id"] for r in usable])

    # --- two grouped, stratified cuts: test, then valid ---
    n_test_folds = folds_for(args.test_fraction, 0.0)
    rest, test = grouped_stratified_cut(keep, y, groups_all, n_test_folds, args.seed)
    n_valid_folds = folds_for(args.valid_fraction, len(test) / len(keep))
    train, valid = grouped_stratified_cut(rest, y, groups_all, n_valid_folds, args.seed)
    split_idx = {"train": train, "valid": valid, "test": test}

    # --- internal sanity: no group in two splits ---
    seen = {}
    for name, idxs in split_idx.items():
        for g in {groups_all[i] for i in idxs}:
            if g in seen and seen[g] != name:
                print(f"[ERROR] photo group {g} assigned to both {seen[g]} and {name}.")
                return 1
            seen[g] = name

    # --- write ---
    if args.force:
        for p in generated:
            shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, idxs in split_idx.items():
        copy_split(out_dir, name, [usable[i] for i in idxs])
    yaml_path = write_yaml(out_dir, source_cfg)

    manifest = pd.DataFrame([{"image": usable[i]["stem"], "assigned_split": name, "original_split": usable[i]["split"],
                              "class": CLASS_NAMES[usable[i]["class_id"]], "photo_group": int(groups_all[i])}
                             for name, idxs in split_idx.items() for i in idxs]
                            + [{"image": usable[i]["stem"], "assigned_split": "EXCLUDED (label conflict)",
                                "original_split": usable[i]["split"], "class": CLASS_NAMES[usable[i]["class_id"]],
                                "photo_group": int(groups_all[i])} for i in sorted(excluded)])
    manifest_path = out_dir / "split_manifest.csv"
    manifest.sort_values(["assigned_split", "image"]).to_csv(manifest_path, index=False)

    table = split_table(usable, split_idx, groups_all)
    leak = verify_no_leakage(out_dir)

    warnings = []
    trows = {r["split"]: r for r in table.to_dict("records")}
    for cls in ("normal", "scoliosis"):
        if trows["test"][cls] < MIN_TEST_PER_CLASS:
            warnings.append(f"Test set has only {trows['test'][cls]} {cls} images (< {MIN_TEST_PER_CLASS}) — still small.")
    for name in SPLITS:
        if trows[name]["normal"] == 0 or trows[name]["scoliosis"] == 0:
            warnings.append(f"The {name} split is missing a class.")
    if trows["test"]["images"] < 30:
        warnings.append(f"Test set has {trows['test']['images']} images; one image still moves accuracy by about "
                        f"{100 / trows['test']['images']:.0f} points.")
    if any(v for k, v in leak.items() if k != "images"):
        warnings.append("Leakage check FAILED — do not use this split.")
    warnings.append("Models trained on the OLD split must be retrained before being compared on this test set.")

    write_reports(out_dir, args, data_yaml, table, conflict_rows, skipped, leak, warnings, manifest_path)

    print(f"\n{'='*64}\n  PHOTO-GROUPED RE-SPLIT (seed {args.seed})\n{'='*64}")
    print(f"  Source   : {data_yaml.relative_to(PROJECT_ROOT) if PROJECT_ROOT in data_yaml.parents else data_yaml}")
    print(f"  Images   : {len(usable)} usable -> {len(keep)} split + {len(excluded)} excluded "
          f"({sum(r['excluded'] for r in conflict_rows)} of {len(conflict_rows)} mixed-label photo group(s), "
          f"scope '{args.conflict_scope}'); {len(skipped)} skipped (no usable label)\n")
    print(table.to_string(index=False))
    ok = not any(v for k, v in leak.items() if k != "images")
    print(f"\n  Leakage check: {'PASS' if ok else 'FAIL'}  {leak}")
    for w in warnings:
        print(f"  ! {w}")
    print(f"\n  Output : {out_dir.relative_to(PROJECT_ROOT) if PROJECT_ROOT in out_dir.parents else out_dir}")
    print("  Reports: split_report.md, conflict_report.md, split_manifest.csv. Nothing was trained.\n")
    return 0 if ok else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a photo-grouped, class-stratified train/valid/test split of the cleaned dataset.")
    parser.add_argument("--data-yaml", type=Path, default=DEFAULT_YAML, help="Cleaned dataset yaml (default: scoliosis_cleaned.yaml).")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="Where to write the split (default: Data/Processed/scoliosis_grouped_split).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument("--test-fraction", type=float, default=0.15, help="Approximate share of images in test (default: 0.15).")
    parser.add_argument("--valid-fraction", type=float, default=0.15, help="Approximate share of images in valid (default: 0.15).")
    parser.add_argument("--conflict-scope", choices=["any", "pixel"], default="any",
                         help="any (default, most conservative): exclude every photo group with mixed class labels. "
                              "pixel: exclude only groups with a near-identical pair that disagrees; keep other mixed groups.")
    parser.add_argument("--force", action="store_true", help="Rebuild: replace the generated files inside --output-dir.")
    sys.exit(run(parser.parse_args()))
