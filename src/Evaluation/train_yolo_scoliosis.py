# ============================================================
# YOLO Fine-Tuning — Scoliosis Keypoint Model
# Trains yolov8n-pose on the scoliosis dataset (train/valid splits)
# Saves best weights to models/yolo_scoliosis/ (or --project)
#
# Defaults reproduce the original behaviour exactly (100 epochs, patience 20,
# imgsz 640, batch 8, seed 0, Ultralytics' own augmentation). Everything is
# overridable from the command line — see --help.
#
# Usage:
#   python train_yolo_scoliosis.py                                   (original dataset)
#   python train_yolo_scoliosis.py --data-yaml Data/Processed/scoliosis_cleaned/scoliosis_cleaned.yaml \
#       --project models/yolo_scoliosis_cleaned_aug --augment-profile conservative
# ============================================================

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import argparse
import yaml
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR  = PROJECT_ROOT / "public_datasets" / "scoliosis"
MODEL_DIR    = PROJECT_ROOT / "models" / "yolo_scoliosis"
YAML_PATH    = PROJECT_ROOT / "public_datasets" / "scoliosis_train.yaml"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.keypoint_schema import NUM_KEYPOINTS, LEFT_SHOULDER, RIGHT_SHOULDER

# Defaults — unchanged from the previous hard-coded values.
DEFAULT_EPOCHS   = 100
DEFAULT_PATIENCE = 20
DEFAULT_IMGSZ    = 640
DEFAULT_BATCH    = 8
DEFAULT_SEED     = 0
DEFAULT_BASE     = "yolov8n-pose.pt"
DEFAULT_RUN_NAME = "run"


# ---------------------------------------------------------------
# Augmentation profiles
#
# These are Ultralytics train() overrides. "default_yolo" passes nothing, so the
# run uses Ultralytics' own defaults — exactly what every run so far has used.
#
# Conservative augmentation is intended as simple synthetic variation from the
# available images (a slightly shifted / scaled / tilted copy of a real photo),
# NOT as aggressive real-world noise simulation. Occlusion, lighting and other
# heavy modifications are deliberately left out for now: each one can quietly
# make an image/label pair invalid (a keypoint hidden or moved with no label
# change), which is hard to spot in a dataset this small.
#
# Choices in "conservative", and why:
#   degrees=5, translate=0.05, scale=0.10  small geometric jitter only; keypoints
#                                           are transformed together with the image
#   shear=0, perspective=0                 these distort apparent spine angles, which
#                                           the geometry layer later measures
#   fliplr=0, flipud=0                     horizontal flip DISABLED. This dataset's
#                                           flip_idx is the identity, but kp[9]/kp[10]
#                                           are left/right shoulders: a flipped image
#                                           would keep each label on the wrong side.
#                                           (Ultralytics' default fliplr=0.5 therefore
#                                           flips half of default_yolo's images with
#                                           swapped shoulders.) Flip is only safe with
#                                           flip_idx = [0..8, 10, 9, 11] in the data yaml.
#   mosaic=0, mixup=0, cutmix=0,           mosaic/mixup/cutmix stitch or blend several
#   copy_paste=0                            people — anatomically implausible for a
#                                           single-subject spine task
#   erasing=0                              no random erasing / occlusion simulation
#   hsv_h=hsv_s=hsv_v=0                    no colour / brightness / lighting changes
# ---------------------------------------------------------------
_OFF = dict(
    hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
    degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0,
    flipud=0.0, fliplr=0.0,
    mosaic=0.0, mixup=0.0, cutmix=0.0, copy_paste=0.0, erasing=0.0,
)

AUGMENT_PROFILES = {
    "default_yolo": {},
    "none": dict(_OFF),
    "conservative": {**_OFF, "degrees": 5.0, "translate": 0.05, "scale": 0.10},
}

# Ultralytics defaults that matter when judging a profile (see `ultralytics.cfg`).
_ULTRALYTICS_FLIP_DEFAULT = 0.5


def expected_flip_idx():
    """The flip_idx that WOULD make a horizontal flip label-correct for this schema."""
    idx = list(range(NUM_KEYPOINTS))
    idx[LEFT_SHOULDER], idx[RIGHT_SHOULDER] = RIGHT_SHOULDER, LEFT_SHOULDER
    return idx


def check_flip_safety(data_yaml: Path, overrides: dict):
    """Warn if a profile leaves horizontal flip on while the yaml's flip_idx is wrong."""
    fliplr = overrides.get("fliplr", _ULTRALYTICS_FLIP_DEFAULT)
    if fliplr <= 0:
        return
    try:
        with open(data_yaml, "r", encoding="utf-8") as f:
            flip_idx = yaml.safe_load(f).get("flip_idx")
    except (OSError, yaml.YAMLError):
        return
    if flip_idx != expected_flip_idx():
        print(f"[WARN] Horizontal flip is ON (fliplr={fliplr}) but flip_idx in the data yaml is {flip_idx}.")
        print(f"       kp[{LEFT_SHOULDER}]/kp[{RIGHT_SHOULDER}] are left/right shoulders, so flipped images get")
        print(f"       their shoulder labels on the wrong side (correct flip_idx: {expected_flip_idx()}).")
        print(f"       --augment-profile conservative or none disables flipping.")


def make_data_yaml():
    """Create a data yaml with absolute paths — the packaged one uses broken relative paths.

    Only used for the ORIGINAL dataset. A cleaned/other dataset should already have
    its own ready-to-use yaml (see dataset_cleaning.py's scoliosis_cleaned.yaml) —
    pass it directly via --data-yaml instead of regenerating one here.
    """
    data = {
        "path": str(DATASET_DIR),
        "train": "train/images",
        "val":   "valid/images",
        "test":  "test/images",
        "kpt_shape": [NUM_KEYPOINTS, 3],
        "flip_idx":  list(range(NUM_KEYPOINTS)),
        "nc": 2,
        "names": {0: "normal", 1: "scoliosis"},
    }
    with open(YAML_PATH, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    print(f"[INFO] Data yaml written to: {YAML_PATH.name}")
    return YAML_PATH


def train(data_yaml: Path = None, model_dir: Path = None, epochs: int = DEFAULT_EPOCHS,
          patience: int = DEFAULT_PATIENCE, imgsz: int = DEFAULT_IMGSZ, batch: int = DEFAULT_BATCH,
          seed: int = DEFAULT_SEED, run_name: str = DEFAULT_RUN_NAME,
          augment_profile: str = "default_yolo", base_weights: str = DEFAULT_BASE):
    """Fine-tune yolov8n-pose on a scoliosis keypoint dataset.

    Args:
        data_yaml: path to a ready-made data yaml (e.g. the cleaned dataset's
            scoliosis_cleaned.yaml). If omitted, the original dataset's yaml is
            (re)generated at YAML_PATH, exactly as before.
        model_dir: Ultralytics `project` directory (default: models/yolo_scoliosis).
            Use a different directory for each experiment so earlier weights are
            never overwritten.
        augment_profile: one of AUGMENT_PROFILES (default_yolo = unchanged behaviour).
    """
    if augment_profile not in AUGMENT_PROFILES:
        raise ValueError(f"Unknown augment profile {augment_profile!r}; choose from {list(AUGMENT_PROFILES)}")
    overrides = AUGMENT_PROFILES[augment_profile]

    # Resolve to absolute — a relative `project=` path passed to Ultralytics gets
    # silently nested under its own default runs/<task>/ directory instead of
    # landing where it looks like it should (e.g. models/yolo_scoliosis_cleaned/
    # ends up as runs/pose/models/yolo_scoliosis_cleaned/). MODEL_DIR is already
    # absolute; this just makes a CLI-supplied --project behave the same way.
    model_dir = Path(model_dir or MODEL_DIR).resolve()
    model_dir.mkdir(parents=True, exist_ok=True)

    if data_yaml is None:
        data_yaml = make_data_yaml()
        train_desc, val_desc = f"{DATASET_DIR / 'train' / 'images'} (135 images)", f"{DATASET_DIR / 'valid' / 'images'} (37 images)"
    else:
        data_yaml = Path(data_yaml)
        if not data_yaml.exists():
            raise FileNotFoundError(f"--data-yaml not found: {data_yaml}")
        train_desc, val_desc = str(data_yaml), "(see data yaml)"

    model = YOLO(base_weights)

    print("[INFO] Starting fine-tuning on scoliosis dataset...")
    print(f"       Data yaml : {data_yaml}")
    print(f"       Train     : {train_desc}")
    print(f"       Val       : {val_desc}")
    print(f"       Base model: {base_weights}")
    print(f"       Settings  : epochs={epochs} patience={patience} imgsz={imgsz} batch={batch} seed={seed}")
    print(f"       Augment   : {augment_profile}" + (f" -> {overrides}" if overrides else " (Ultralytics defaults)"))
    print(f"       Output    : {model_dir / run_name}")
    check_flip_safety(data_yaml, overrides)

    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        patience=patience,  # early stopping
        imgsz=imgsz,
        batch=batch,
        seed=seed,
        project=str(model_dir),
        name=run_name,
        exist_ok=True,
        save=True,
        val=True,
        plots=True,
        **overrides,
    )

    best_weights = model_dir / run_name / "weights" / "best.pt"
    print(f"\n[DONE] Best model saved to: {best_weights}")
    print(f"       Run the fine-tuned evaluation with:")
    print(f"       python src/Evaluation/run_evaluation.py --model yolo_finetuned --weights {best_weights} --tag <name>")
    return best_weights


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune yolov8n-pose on a scoliosis keypoint dataset.")
    parser.add_argument("--data-yaml", type=Path, default=None,
                         help="Path to a ready-made data yaml (e.g. Data/Processed/scoliosis_cleaned/"
                              "scoliosis_cleaned.yaml). Omit to train on the original dataset "
                              "(public_datasets/scoliosis), regenerating its yaml as before.")
    parser.add_argument("--project", "--model-dir", dest="model_dir", type=Path, default=None,
                         help=f"Output directory for this experiment (default: {MODEL_DIR.relative_to(PROJECT_ROOT)}). "
                              f"Use a distinct directory per experiment so you don't overwrite earlier models. "
                              f"--model-dir is the older name for the same option.")
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME,
                         help=f"Sub-folder inside --project for this run (default: {DEFAULT_RUN_NAME}). "
                              f"Weights land in <project>/<run-name>/weights/best.pt.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help=f"Max epochs (default: {DEFAULT_EPOCHS}).")
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE,
                         help=f"Early-stopping patience in epochs (default: {DEFAULT_PATIENCE}).")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ, help=f"Training image size (default: {DEFAULT_IMGSZ}).")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH, help=f"Batch size (default: {DEFAULT_BATCH}).")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"Random seed (default: {DEFAULT_SEED}).")
    parser.add_argument("--base-weights", default=DEFAULT_BASE,
                         help=f"Pretrained weights to fine-tune from (default: {DEFAULT_BASE}).")
    parser.add_argument("--augment-profile", choices=list(AUGMENT_PROFILES), default="default_yolo",
                         help="default_yolo = Ultralytics' own augmentation (unchanged behaviour); "
                              "conservative = small shift/scale/tilt only, no flip/mosaic/erasing/colour changes; "
                              "none = no augmentation at all.")
    args = parser.parse_args()
    train(data_yaml=args.data_yaml, model_dir=args.model_dir, epochs=args.epochs, patience=args.patience,
          imgsz=args.imgsz, batch=args.batch, seed=args.seed, run_name=args.run_name,
          augment_profile=args.augment_profile, base_weights=args.base_weights)
