# ============================================================
# Dataset path + provenance helpers for the evaluation scripts
#
# One place that answers "which images/labels does this evaluation read?",
# so run_evaluation.py, debug_spine_geometry.py and the comparison scripts
# never hardcode a test folder:
#
#   resolve_split(data_yaml, split)  -> images dir, labels dir, dataset name
#   write_meta / read_meta           -> a small <csv>.meta.json "sidecar" recording
#                                       which dataset/split/weights produced a results CSV
#
# WARNING — models trained on the old split should not be interpreted as unbiased
# on the grouped test set. The old train split contains photos that the grouped
# split puts in test. Retrain on the grouped train/valid split before reporting
# grouped-split performance (see create_grouped_split.py).
# ============================================================

import json
import os
from datetime import datetime
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# The original public dataset — evaluating against this reproduces every older result.
DEFAULT_DATA_YAML = PROJECT_ROOT / "public_datasets" / "scoliosis" / "data.yaml"

# Datasets whose test folders are the OLD split (near-duplicate photos shared with train/valid).
# ("public_datasets" is the folder of the generated scoliosis_train.yaml the original model was trained from.)
LEGACY_DATASET_NAMES = {"scoliosis", "scoliosis_cleaned", "public_datasets"}

# The yaml key for validation is "val" in Ultralytics but the folders are called "valid".
_SPLIT_KEYS = {"train": ["train"], "valid": ["valid", "val"], "val": ["val", "valid"], "test": ["test"]}

OLD_SPLIT_WARNING = (
    "Models trained on the old split should not be interpreted as unbiased on the grouped test set. "
    "Retrain on the grouped train/valid split before reporting grouped-split performance."
)


def _rel(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def resolve_data_yaml(data_yaml) -> Path:
    path = Path(data_yaml) if data_yaml else DEFAULT_DATA_YAML
    if not path.is_absolute():
        path = (PROJECT_ROOT / path) if (PROJECT_ROOT / path).exists() else path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"data yaml not found: {path}")
    return path.resolve()


def resolve_split(data_yaml=None, split: str = "test") -> dict:
    """Locate a split's image and label folders from a YOLO data yaml.

    Returns a dict: yaml_path, dataset_name (the yaml's folder name), split, images_dir,
    labels_dir (may be None if no sibling `labels` folder exists), is_legacy_default.
    """
    yaml_path = resolve_data_yaml(data_yaml)
    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}

    key = next((k for k in _SPLIT_KEYS.get(split, [split]) if k in cfg), None)
    if key is None:
        raise KeyError(f"{yaml_path.name} has no '{split}' entry (keys: {[k for k in ('train', 'val', 'valid', 'test') if k in cfg]}).")
    value = cfg[key]
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"'{key}' in {yaml_path.name} lists several folders; only a single images folder is supported.")
        value = value[0]
    value = str(value)

    root = Path(cfg["path"]) if cfg.get("path") else yaml_path.parent
    if not root.is_absolute():
        root = (yaml_path.parent / root)
    images_dir = (root / value)
    if not images_dir.exists():
        # The packaged Roboflow yaml uses '../test/images' relative to a folder one level too deep;
        # fall back to the same path relative to the yaml's own folder.
        stripped = Path(*[p for p in Path(value).parts if p != ".."])
        images_dir = yaml_path.parent / stripped
    images_dir = images_dir.resolve()
    if not images_dir.is_dir():
        raise FileNotFoundError(f"{split} images folder not found for {yaml_path.name}: {images_dir}")
    if value.lower().endswith(".txt"):
        raise ValueError(f"'{key}' in {yaml_path.name} points to an image list file; point it at an images folder.")

    labels_dir = images_dir.parent / "labels" if images_dir.name == "images" else None
    if labels_dir is not None and not labels_dir.is_dir():
        labels_dir = None

    return {
        "yaml_path": yaml_path,
        "dataset_name": yaml_path.parent.name,
        "split": split,
        "images_dir": images_dir,
        "labels_dir": labels_dir,
        "is_legacy_default": yaml_path == DEFAULT_DATA_YAML.resolve() and split == "test",
    }


# ---------------------------------------------------------------
# Sidecar metadata — which dataset/split/weights produced a results CSV
# ---------------------------------------------------------------

def meta_path(csv_path: Path) -> Path:
    csv_path = Path(csv_path)
    return csv_path.with_name(csv_path.stem + ".meta.json")


def write_meta(csv_path: Path, **fields) -> Path:
    fields.setdefault("created", datetime.now().isoformat(timespec="seconds"))
    path = meta_path(csv_path)
    path.write_text(json.dumps(fields, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def read_meta(csv_path: Path):
    """The sidecar dict, or None for results CSVs produced before sidecars existed (legacy runs)."""
    path = meta_path(csv_path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def describe_eval(meta) -> str:
    """One line saying what a results CSV was evaluated on."""
    if meta is None:
        return "legacy run, no metadata — assumed public_datasets/scoliosis/test (the OLD split)"
    return f"{meta.get('eval_dataset')}/{meta.get('eval_split')} ({meta.get('n_images')} images)"


def is_legacy_eval(meta) -> bool:
    """True if the results were produced on an old-split test folder (so they are historical/debug only
    once the grouped split exists)."""
    return meta is None or meta.get("eval_dataset") in LEGACY_DATASET_NAMES


def trained_on_yaml(model):
    """The data yaml a loaded Ultralytics model was trained on (from its checkpoint), or None if unknown."""
    ckpt = getattr(model, "ckpt", None)
    if not isinstance(ckpt, dict):
        return None
    data = (ckpt.get("train_args") or {}).get("data")
    if not data:
        return None
    path = Path(str(data))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def same_path(a, b) -> bool:
    return a is not None and b is not None and os.path.normcase(str(Path(a).resolve())) == os.path.normcase(str(Path(b).resolve()))


def rel(path) -> str:
    return _rel(path) if path is not None else None
