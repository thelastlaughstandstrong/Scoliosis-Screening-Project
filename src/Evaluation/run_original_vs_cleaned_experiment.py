# ============================================================
# Original vs. Cleaned Dataset — Experiment Runner
#
# A thin, print-first wrapper around the existing scripts — it does
# not reimplement training or evaluation. By default it only PRINTS
# the exact commands for the full experiment; pass --no-dry-run to
# actually execute them.
#
# Model output folders (kept separate on purpose — see
# train_yolo_scoliosis.py):
#   Original: models/yolo_scoliosis/run/weights/best.pt        (existing default)
#   Cleaned : models/yolo_scoliosis_cleaned/run/weights/best.pt
#
# Evaluation CSVs (kept separate via --tag, in the existing
# Data/Processed/comparison/ convention used by run_evaluation.py):
#   Original: yolo_finetuned_scoliosis_test_original_trained.csv
#   Cleaned : yolo_finetuned_scoliosis_test_cleaned_trained.csv
#
# Usage:
#   python run_original_vs_cleaned_experiment.py                 (dry-run, shows every step)
#   python run_original_vs_cleaned_experiment.py --train --no-dry-run
#   python run_original_vs_cleaned_experiment.py --evaluate --no-dry-run
#   python run_original_vs_cleaned_experiment.py --compare --no-dry-run
#   python run_original_vs_cleaned_experiment.py --train --evaluate --compare --no-dry-run   (full pipeline, for real)
# ============================================================

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_SCRIPT   = PROJECT_ROOT / "src" / "Evaluation" / "train_yolo_scoliosis.py"
EVAL_SCRIPT    = PROJECT_ROOT / "src" / "Evaluation" / "run_evaluation.py"
COMPARE_SCRIPT = PROJECT_ROOT / "src" / "Evaluation" / "compare_original_vs_cleaned.py"

# Original model dir is the EXISTING default (models/yolo_scoliosis) — kept as-is
# so already-trained weights are reused rather than silently orphaned by a rename.
ORIGINAL_MODEL_DIR = PROJECT_ROOT / "models" / "yolo_scoliosis"
CLEANED_MODEL_DIR  = PROJECT_ROOT / "models" / "yolo_scoliosis_cleaned"

ORIGINAL_WEIGHTS = ORIGINAL_MODEL_DIR / "run" / "weights" / "best.pt"
CLEANED_WEIGHTS  = CLEANED_MODEL_DIR / "run" / "weights" / "best.pt"

CLEANED_DATA_YAML = PROJECT_ROOT / "Data" / "Processed" / "scoliosis_cleaned" / "scoliosis_cleaned.yaml"

ORIGINAL_TAG = "original_trained"
CLEANED_TAG  = "cleaned_trained"


def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p)


def build_commands():
    return {
        "train_original": (
            "Train YOLO on the ORIGINAL dataset",
            [sys.executable, _rel(TRAIN_SCRIPT), "--model-dir", _rel(ORIGINAL_MODEL_DIR)],
        ),
        "train_cleaned": (
            "Train YOLO on the CLEANED dataset",
            [sys.executable, _rel(TRAIN_SCRIPT),
             "--data-yaml", _rel(CLEANED_DATA_YAML), "--model-dir", _rel(CLEANED_MODEL_DIR)],
        ),
        "evaluate_original": (
            "Evaluate the ORIGINAL-trained model",
            [sys.executable, _rel(EVAL_SCRIPT), "--model", "yolo_finetuned",
             "--weights", _rel(ORIGINAL_WEIGHTS), "--tag", ORIGINAL_TAG],
        ),
        "evaluate_cleaned": (
            "Evaluate the CLEANED-trained model",
            [sys.executable, _rel(EVAL_SCRIPT), "--model", "yolo_finetuned",
             "--weights", _rel(CLEANED_WEIGHTS), "--tag", CLEANED_TAG],
        ),
        "compare": (
            "Compare original-trained vs. cleaned-trained results",
            [sys.executable, _rel(COMPARE_SCRIPT)],
        ),
    }


def execute(name, description, cmd, dry_run):
    mode = "DRY-RUN" if dry_run else "RUNNING"
    print(f"\n[{mode}] {name} — {description}")
    print(f"  $ {' '.join(cmd)}")
    if dry_run:
        return
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print(f"[ERROR] '{name}' exited with code {result.returncode} — stopping.")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="Print (and, optionally, run) the original-vs-cleaned dataset experiment "
                     "for the fine-tuned YOLO spine-keypoint model. Defaults to printing only."
    )
    parser.add_argument("--train", action="store_true", help="Include the training steps (original + cleaned).")
    parser.add_argument("--evaluate", action="store_true", help="Include the evaluation steps (original + cleaned).")
    parser.add_argument("--compare", action="store_true", help="Include the comparison step.")
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=True,
                         help="Print commands only (default). Pass --no-dry-run to actually execute them.")
    args = parser.parse_args()

    # No stage flag given -> show/run the full pipeline (train -> evaluate -> compare).
    run_train    = args.train    or not (args.train or args.evaluate or args.compare)
    run_evaluate = args.evaluate or not (args.train or args.evaluate or args.compare)
    run_compare  = args.compare  or not (args.train or args.evaluate or args.compare)

    commands = build_commands()

    print("=" * 70)
    print("  ORIGINAL vs. CLEANED DATASET EXPERIMENT")
    print(f"  Mode: {'DRY-RUN (printing only)' if args.dry_run else 'EXECUTING'}")
    print("=" * 70)

    if run_train:
        if not args.dry_run and not CLEANED_DATA_YAML.exists():
            print(f"[ERROR] Cleaned data yaml not found: {_rel(CLEANED_DATA_YAML)}")
            print("        Run: python src/Evaluation/dataset_cleaning.py --export-cleaned")
            sys.exit(1)
        for key in ("train_original", "train_cleaned"):
            execute(key, *commands[key], dry_run=args.dry_run)

    if run_evaluate:
        for key in ("evaluate_original", "evaluate_cleaned"):
            if not args.dry_run:
                weights = ORIGINAL_WEIGHTS if key == "evaluate_original" else CLEANED_WEIGHTS
                if not weights.exists():
                    print(f"[ERROR] Weights not found: {_rel(weights)} — train that model first.")
                    sys.exit(1)
            execute(key, *commands[key], dry_run=args.dry_run)

    if run_compare:
        execute("compare", *commands["compare"], dry_run=args.dry_run)

    if args.dry_run:
        print("\nThis was a dry run — nothing was trained, evaluated, or overwritten.")
        print("Re-run with --no-dry-run to actually execute the commands above.")
    print("=" * 70)


if __name__ == "__main__":
    main()
