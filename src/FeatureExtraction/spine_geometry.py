# ============================================================
# Spine Geometry Feature Extraction
# Anatomy-aware post-processing for the fine-tuned YOLOv8
# 12-point spine keypoint model.
#
# SCOPE / NAMING — read before using these numbers anywhere:
# These are camera-derived, screening-support proxies only.
# They are NOT clinical ATR (angle of trunk rotation) or Cobb
# angle measurements, and must not be reported as such.
#
# Reliability note: intermediate spine keypoints (kp[1..7]) are
# marked "invisible" and excluded from training loss (see
# train_yolo_scoliosis.py / quest_recording_yolo.py comments),
# so their predicted positions are lower-confidence than
# kp[0] (base), kp[8] (top), and the shoulders. Treat
# max/mean_lateral_deviation_px and curve_smoothness_score as
# indicative, not precise.
# ============================================================

import sys
import math
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config.keypoint_schema import (
    SPINE_BASE_LUMBAR,
    SPINE_TOP_CERVICAL,
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    SPINE_POINT_IDXS,
    INTERMEDIATE_SPINE_IDXS,
    DEFAULT_CONFIDENCE_THRESHOLD,
)

MIN_VALID_SPINE_KEYPOINTS = 3   # base + top + >=1 intermediate, minimum to trust geometry
MIDLINE_SPLIT_IDX         = 4   # spine_intermediate_4 — splits the spine into lower/upper halves


# ---------------------------------------------------------------
# Small geometry helpers
# ---------------------------------------------------------------

def _conf_ok(kps, idx, threshold):
    return float(kps[idx, 2]) >= threshold


def _xy(kps, idx):
    return float(kps[idx, 0]), float(kps[idx, 1])


def _segment_angle_deg(p_bottom, p_top):
    """Angle of the line from a lower spine point to a higher one.

    Convention matches the existing spine_angle_deg() in run_evaluation.py /
    screen_subject.py: 0 deg = perfectly vertical, image y increases downward.
    """
    dx = p_top[0] - p_bottom[0]
    dy = p_bottom[1] - p_top[1]
    return math.degrees(math.atan2(dx, dy))


def _angle_diff_deg(a, b):
    """Smallest signed difference a-b, wrapped to [-180, 180]."""
    return (a - b + 180) % 360 - 180


def _point_to_line_distance(point, line_a, line_b):
    """Perpendicular distance from `point` to the infinite line through line_a/line_b."""
    ax, ay = line_a
    bx, by = line_b
    px, py = point
    line_len = math.hypot(bx - ax, by - ay)
    if line_len < 1e-6:
        return math.hypot(px - ax, py - ay)
    return abs((bx - ax) * (ay - py) - (ax - px) * (by - ay)) / line_len


def visible_spine_indices(kps, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD):
    """Spine indices (0..8) with confidence >= threshold, in schema order (base -> top)."""
    return [i for i in SPINE_POINT_IDXS if _conf_ok(kps, i, confidence_threshold)]


# ---------------------------------------------------------------
# Feature dict scaffolding
# ---------------------------------------------------------------

def _default_features(reason="ok"):
    return {
        "spine_axis_inclination_deg":     None,
        "shoulder_height_difference_px":  None,
        "shoulder_line_angle_deg":        None,
        "spine_midline_offset_px":        None,
        "valid_spine_keypoint_count":     0,
        "mean_spine_keypoint_confidence": 0.0,
        "max_lateral_deviation_px":       None,
        "mean_lateral_deviation_px":      None,
        "curve_smoothness_score":         None,
        "segment_angle_upper_deg":        None,
        "segment_angle_lower_deg":        None,
        "valid_spine_geometry":           False,
        "geometry_failure_reason":        reason,
    }


def empty_spine_geometry(reason="no_pose_detected"):
    """Default feature dict for when no detection/keypoints are available at all."""
    return _default_features(reason)


# ---------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------

def extract_spine_geometry(kps, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD):
    """
    Compute anatomy-aware spine geometry features from one detection's keypoints.

    Args:
        kps: array-like, shape [12, 3], rows = [x_px, y_px, confidence].
        confidence_threshold: minimum per-keypoint confidence to treat a point
            as visible/usable.

    Returns:
        dict of features (see module docstring for definitions and caveats).
        Fields that cannot be computed are None — always check
        `valid_spine_geometry` / `geometry_failure_reason` first.
    """
    features = _default_features("ok")

    spine_confs = [float(kps[i, 2]) for i in SPINE_POINT_IDXS]
    features["mean_spine_keypoint_confidence"] = round(sum(spine_confs) / len(spine_confs), 4)

    visible_idxs = visible_spine_indices(kps, confidence_threshold)
    features["valid_spine_keypoint_count"] = len(visible_idxs)

    base_ok = _conf_ok(kps, SPINE_BASE_LUMBAR, confidence_threshold)
    top_ok  = _conf_ok(kps, SPINE_TOP_CERVICAL, confidence_threshold)

    if not (base_ok and top_ok):
        features["geometry_failure_reason"] = "missing_or_low_confidence_base_or_top"
        return features

    if len(visible_idxs) < MIN_VALID_SPINE_KEYPOINTS:
        features["geometry_failure_reason"] = "insufficient_spine_keypoints"
        return features

    base = _xy(kps, SPINE_BASE_LUMBAR)
    top  = _xy(kps, SPINE_TOP_CERVICAL)

    if (base[1] - top[1]) <= 0:
        # Top of spine is not above (image-y-wise) the base — anatomically
        # implausible for an upright subject; likely a bad detection.
        features["geometry_failure_reason"] = "implausible_keypoint_ordering"
        return features

    features["valid_spine_geometry"] = True

    # ---- Core axis: base -> top chord ----
    features["spine_axis_inclination_deg"] = round(_segment_angle_deg(base, top), 3)

    # ---- Shoulders ----
    if _conf_ok(kps, LEFT_SHOULDER, confidence_threshold) and _conf_ok(kps, RIGHT_SHOULDER, confidence_threshold):
        ls, rs = _xy(kps, LEFT_SHOULDER), _xy(kps, RIGHT_SHOULDER)
        features["shoulder_height_difference_px"] = round(abs(ls[1] - rs[1]), 3)
        features["shoulder_line_angle_deg"] = round(
            math.degrees(math.atan2(rs[1] - ls[1], rs[0] - ls[0])), 3
        )

    # ---- Midline offset: horizontal shift of spine top vs base ----
    features["spine_midline_offset_px"] = round(abs(top[0] - base[0]), 3)

    # ---- Lateral deviation of intermediate points from the base-top reference line ----
    visible_intermediate = [i for i in visible_idxs if i in INTERMEDIATE_SPINE_IDXS]
    if visible_intermediate:
        deviations = [_point_to_line_distance(_xy(kps, i), base, top) for i in visible_intermediate]
        features["max_lateral_deviation_px"]  = round(max(deviations), 3)
        features["mean_lateral_deviation_px"] = round(sum(deviations) / len(deviations), 3)

    # ---- Curve smoothness: mean change between adjacent segment angles ----
    if len(visible_idxs) >= 3:
        points = [_xy(kps, i) for i in visible_idxs]
        segment_angles = [_segment_angle_deg(points[i], points[i + 1]) for i in range(len(points) - 1)]
        if len(segment_angles) >= 2:
            diffs = [
                abs(_angle_diff_deg(segment_angles[i + 1], segment_angles[i]))
                for i in range(len(segment_angles) - 1)
            ]
            features["curve_smoothness_score"] = round(sum(diffs) / len(diffs), 3)

    # ---- Upper / lower segment angles ----
    lower_idxs = [i for i in visible_idxs if i <= MIDLINE_SPLIT_IDX]
    upper_idxs = [i for i in visible_idxs if i >= MIDLINE_SPLIT_IDX]

    if len(lower_idxs) >= 2:
        features["segment_angle_lower_deg"] = round(
            _segment_angle_deg(_xy(kps, min(lower_idxs)), _xy(kps, max(lower_idxs))), 3
        )
    if len(upper_idxs) >= 2:
        features["segment_angle_upper_deg"] = round(
            _segment_angle_deg(_xy(kps, min(upper_idxs)), _xy(kps, max(upper_idxs))), 3
        )

    return features
