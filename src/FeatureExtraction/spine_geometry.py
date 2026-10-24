# ============================================================
# Spine Geometry Feature Extraction
# Anatomy-aware POST-PROCESSING for the fine-tuned YOLOv8
# 12-point spine keypoint model.
#
# Pipeline this module implements (no YOLO architecture change):
#   YOLOv8-pose keypoints + confidence
#     -> confidence filtering
#     -> anatomical plausibility checks
#     -> transparent geometric screening-support indicators
#     -> rule-based reliability score / retake-or-review flag
#
# SCOPE / NAMING — read before using these numbers anywhere:
# These are camera-derived, screening-support proxies only.
# They are NOT clinical ATR (angle of trunk rotation) or Cobb
# angle measurements, are NOT a diagnosis, and must not be
# reported as such. `anatomy_reliability_score` and
# `retake_or_review_recommended` describe measurement quality,
# not disease severity.
#
# Reliability note: intermediate spine keypoints (kp[1..7]) are
# marked "invisible" and excluded from training loss for the
# pseudo-labeled domain data (see quest_recording_yolo.py), so
# their predicted positions can be lower-confidence than kp[0]
# (base), kp[8] (top), and the shoulders — `curvature_features_reliable`
# says whether enough of them were usable on this sample.
# ============================================================

import sys
import math
import statistics
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

MIN_VALID_SPINE_KEYPOINTS = 3   # base + top + >=1 intermediate, minimum to compute anything
MIDLINE_SPLIT_IDX         = 4   # spine_intermediate_4 — splits the spine into lower/upper halves

# --- Soft, documented heuristic thresholds (screening-support, not clinical cutoffs) ---
MIN_INTERMEDIATE_FOR_RELIABLE_CURVATURE = 3      # >= this many of kp[1..7] visible -> "reliable" curvature
SPAN_TOLERANCE_RATIO      = 0.10   # how far outside the base-top span a point may project before flagging
ORDERING_TOLERANCE_RATIO  = 0.05   # allowed backward drift (as a fraction of spine length) between consecutive intermediates
JUMP_RATIO_THRESHOLD      = 2.5    # a segment longer than this x the median segment length is an "excessive jump"
STRADDLE_MIN_OFFSET_RATIO = 0.02   # shoulder offset below this fraction of spine length is too small to judge straddling
UNUSUAL_INCLINATION_DEG   = 45.0   # |spine_axis_inclination_deg| beyond this is unusual, not necessarily invalid
UNUSUAL_SHOULDER_ANGLE_DEG = 20.0  # |shoulder_line_angle_deg| beyond this is unusual, not necessarily invalid

# Reliability-score component weights (must sum to 1.0 — see anatomy_reliability_score())
RELIABILITY_WEIGHTS = {
    "base_top_visible":       0.25,
    "shoulders_visible":      0.15,
    "spine_coverage":         0.15,
    "mean_confidence":        0.15,
    "anatomy_valid_basic":    0.10,
    "anatomy_valid_intermediate": 0.10,
    "intermediate_coverage":  0.10,
}
WARNING_PENALTY_PER_FLAG = 0.03
WARNING_PENALTY_MAX      = 0.15

RELIABILITY_LEVEL_THRESHOLDS = {"high": 0.85, "medium": 0.60}   # >= high, >= medium else low


# ---------------------------------------------------------------
# Small geometry helpers
# ---------------------------------------------------------------

def _conf_ok(kps, idx, threshold):
    return float(kps[idx, 2]) >= threshold


def _xy(kps, idx):
    return float(kps[idx, 0]), float(kps[idx, 1])


def _dist(p, q):
    return math.hypot(q[0] - p[0], q[1] - p[1])


def _segment_angle_deg(p_bottom, p_top):
    """Angle of the line from a lower spine point to a higher one.

    Convention matches spine_angle_deg() in run_evaluation.py / screen_subject.py:
    0 deg = perfectly vertical, image y increases downward.
    """
    dx = p_top[0] - p_bottom[0]
    dy = p_bottom[1] - p_top[1]
    return math.degrees(math.atan2(dx, dy))


def _angle_diff_deg(a, b):
    """Smallest signed difference a-b, wrapped to [-180, 180]."""
    return (a - b + 180) % 360 - 180


def _point_to_line_distance(point, line_a, line_b):
    """Perpendicular (unsigned) distance from `point` to the line through line_a/line_b."""
    ax, ay = line_a
    bx, by = line_b
    px, py = point
    line_len = math.hypot(bx - ax, by - ay)
    if line_len < 1e-6:
        return math.hypot(px - ax, py - ay)
    return abs((bx - ax) * (ay - py) - (ax - px) * (by - ay)) / line_len


def _signed_offset_from_line(point, line_a, line_b):
    """Signed perpendicular offset — sign indicates which side of the line the point is on."""
    ax, ay = line_a
    bx, by = line_b
    px, py = point
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax)


def _project_param(point, line_a, line_b):
    """Where `point` projects onto the line a->b, as a fraction t (0 at a, 1 at b)."""
    ax, ay = line_a
    bx, by = line_b
    px, py = point
    abx, aby = bx - ax, by - ay
    denom = abx * abx + aby * aby
    if denom < 1e-9:
        return 0.0
    return ((px - ax) * abx + (py - ay) * aby) / denom


def visible_indices(kps, idxs, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD):
    """Which of `idxs` have confidence >= threshold, preserving their given order."""
    return [i for i in idxs if _conf_ok(kps, i, confidence_threshold)]


def visible_spine_indices(kps, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD):
    """Spine indices (0..8) with confidence >= threshold, in schema order (base -> top)."""
    return visible_indices(kps, SPINE_POINT_IDXS, confidence_threshold)


# ---------------------------------------------------------------
# Feature dict scaffolding — every key extract_spine_geometry() can
# ever produce, so callers can rely on the schema even on failure.
# ---------------------------------------------------------------

def _default_features(reason="ok"):
    return {
        # --- Task 1: anatomical validity ---
        "valid_spine_geometry":            False,
        "anatomy_valid_basic":             False,
        "anatomy_valid_intermediate":      False,
        "geometry_failure_reason":         reason,
        "anatomy_warning_flags":           "",
        "visible_spine_keypoint_count":    0,
        "visible_intermediate_count":      0,
        "mean_spine_keypoint_confidence":  0.0,
        "min_required_keypoints_met":      False,

        # --- Task 2: confidence-aware reliability of feature groups ---
        "spine_axis_confidence":           0.0,
        "shoulder_feature_confidence":     0.0,
        "intermediate_feature_confidence": None,
        "curvature_features_reliable":     False,
        "curvature_features_reason":       reason,

        # --- Task 3: transparent geometry features ---
        "spine_axis_inclination_deg":              None,
        "shoulder_height_difference_px":            None,
        "shoulder_line_angle_deg":                   None,
        "spine_midline_offset_px":                   None,
        "max_lateral_deviation_px":                  None,
        "mean_lateral_deviation_px":                 None,
        "confidence_weighted_mean_lateral_deviation_px": None,
        "curve_smoothness_score":                    None,
        "segment_angle_upper_deg":                   None,
        "segment_angle_lower_deg":                   None,
        "upper_lower_segment_angle_difference_deg":  None,
        "spine_length_px":                           None,
        "normalized_max_lateral_deviation":           None,
        "normalized_mean_lateral_deviation":          None,

        # --- Task 4: rule-based reliability score ---
        "anatomy_reliability_score":        0.0,
        "reliability_level":                "invalid",
        "retake_or_review_recommended":     True,
        "retake_or_review_reason":          reason,
    }


def empty_spine_geometry(reason="no_pose_detected"):
    """Default feature dict for when no detection/keypoints are available at all."""
    return _default_features(reason)


# ---------------------------------------------------------------
# Anatomical plausibility checks (Task 1)
# Each check appends a short, non-clinical tag to `flags` when it
# fires. Checks here only ever *flag* — none of them raise or crash;
# the caller decides how much weight to give each flag.
# ---------------------------------------------------------------

def _check_intermediate_span_and_ordering(kps, base, top, visible_intermediate, spine_length, flags):
    """'Between base and top' + 'roughly bottom-to-top ordering' + 'no excessive jump'."""
    if not visible_intermediate or spine_length < 1e-6:
        return

    params = {i: _project_param(_xy(kps, i), base, top) for i in visible_intermediate}

    if any(t < -SPAN_TOLERANCE_RATIO or t > 1 + SPAN_TOLERANCE_RATIO for t in params.values()):
        flags.append("intermediate_outside_base_top_span")

    ordered = sorted(params.items())  # by keypoint index, which should track base->top order
    for (_, t_prev), (_, t_next) in zip(ordered, ordered[1:]):
        if t_next < t_prev - ORDERING_TOLERANCE_RATIO:
            flags.append("intermediate_ordering_violation")
            break

    # Excessive jump between neighbouring *visible* spine points (base/top included).
    all_visible_spine = sorted(set(visible_intermediate) | {SPINE_BASE_LUMBAR, SPINE_TOP_CERVICAL})
    points = [_xy(kps, i) for i in all_visible_spine]
    seg_lengths = [_dist(points[i], points[i + 1]) for i in range(len(points) - 1)]
    if len(seg_lengths) >= 3:
        median_len = statistics.median(seg_lengths)
        if median_len > 1e-6 and max(seg_lengths) > JUMP_RATIO_THRESHOLD * median_len:
            flags.append("excessive_segment_jump")


def _check_shoulders_straddle_axis(ls, rs, base, top, spine_length, flags):
    if spine_length < 1e-6:
        return
    left_offset  = _signed_offset_from_line(ls, base, top)
    right_offset = _signed_offset_from_line(rs, base, top)
    min_offset = STRADDLE_MIN_OFFSET_RATIO * spine_length
    if abs(left_offset) < min_offset or abs(right_offset) < min_offset:
        return   # too close to the axis to judge either way — not flagged
    if (left_offset > 0) == (right_offset > 0):
        flags.append("shoulders_not_straddling_spine_axis")


# ---------------------------------------------------------------
# Task 4: rule-based reliability score
# ---------------------------------------------------------------

def _score_reliability(features, flags):
    w = RELIABILITY_WEIGHTS
    score = (
        w["base_top_visible"] * 1.0   # gated upstream — always true when this is called
        + w["shoulders_visible"] * (1.0 if features["shoulder_height_difference_px"] is not None else 0.0)
        + w["spine_coverage"] * (features["visible_spine_keypoint_count"] / len(SPINE_POINT_IDXS))
        + w["mean_confidence"] * features["mean_spine_keypoint_confidence"]
        + w["anatomy_valid_basic"] * (1.0 if features["anatomy_valid_basic"] else 0.0)
        + w["anatomy_valid_intermediate"] * (1.0 if features["anatomy_valid_intermediate"] else 0.0)
        + w["intermediate_coverage"] * (features["visible_intermediate_count"] / len(INTERMEDIATE_SPINE_IDXS))
    )
    penalty = min(WARNING_PENALTY_MAX, WARNING_PENALTY_PER_FLAG * len(flags))
    return max(0.0, min(1.0, score - penalty))


def _reliability_level(score):
    if score >= RELIABILITY_LEVEL_THRESHOLDS["high"]:
        return "high"
    if score >= RELIABILITY_LEVEL_THRESHOLDS["medium"]:
        return "medium"
    return "low"


# ---------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------

def extract_spine_geometry(kps, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD):
    """
    Compute anatomy-aware spine geometry features from one detection's keypoints.

    Args:
        kps: array-like, shape [12, 3], rows = [x_px, y_px, confidence].
        confidence_threshold: minimum per-keypoint confidence to treat a point
            as visible/usable — the single configurable knob for every
            visibility-gated check in this function.

    Returns:
        dict of features (see module docstring for definitions/caveats). Fields
        that cannot be computed are None — always check `valid_spine_geometry`
        (can we compute anything at all) and `reliability_level` /
        `retake_or_review_recommended` (should this measurement be trusted)
        before using the rest.
    """
    features = _default_features("ok")
    flags = []

    spine_confs = [float(kps[i, 2]) for i in SPINE_POINT_IDXS]
    features["mean_spine_keypoint_confidence"] = round(sum(spine_confs) / len(spine_confs), 4)

    visible_idxs = visible_spine_indices(kps, confidence_threshold)
    visible_intermediate = [i for i in visible_idxs if i in INTERMEDIATE_SPINE_IDXS]

    features["visible_spine_keypoint_count"] = len(visible_idxs)
    features["visible_intermediate_count"]   = len(visible_intermediate)
    features["min_required_keypoints_met"]   = len(visible_idxs) >= MIN_VALID_SPINE_KEYPOINTS

    if visible_intermediate:
        inter_confs = [float(kps[i, 2]) for i in visible_intermediate]
        features["intermediate_feature_confidence"] = round(sum(inter_confs) / len(inter_confs), 4)

    base_ok = _conf_ok(kps, SPINE_BASE_LUMBAR, confidence_threshold)
    top_ok  = _conf_ok(kps, SPINE_TOP_CERVICAL, confidence_threshold)
    ls_ok   = _conf_ok(kps, LEFT_SHOULDER, confidence_threshold)
    rs_ok   = _conf_ok(kps, RIGHT_SHOULDER, confidence_threshold)

    if base_ok:
        features["spine_axis_confidence"] = round(float(kps[SPINE_BASE_LUMBAR, 2]), 4)
    if base_ok and top_ok:
        features["spine_axis_confidence"] = round(
            (float(kps[SPINE_BASE_LUMBAR, 2]) + float(kps[SPINE_TOP_CERVICAL, 2])) / 2, 4
        )
    if ls_ok and rs_ok:
        features["shoulder_feature_confidence"] = round(
            (float(kps[LEFT_SHOULDER, 2]) + float(kps[RIGHT_SHOULDER, 2])) / 2, 4
        )
    if not ls_ok or not rs_ok:
        flags.append("shoulder_not_visible")

    if not (base_ok and top_ok):
        features["geometry_failure_reason"] = "missing_or_low_confidence_base_or_top"
        features["curvature_features_reason"] = "missing_or_low_confidence_base_or_top"
        features["anatomy_warning_flags"] = "; ".join(flags)
        features["retake_or_review_reason"] = features["geometry_failure_reason"]
        return features

    if len(visible_idxs) < MIN_VALID_SPINE_KEYPOINTS:
        flags.append("insufficient_spine_keypoints")
        features["geometry_failure_reason"] = "insufficient_spine_keypoints"
        features["curvature_features_reason"] = "insufficient_spine_keypoints"
        features["anatomy_warning_flags"] = "; ".join(flags)
        features["retake_or_review_reason"] = features["geometry_failure_reason"]
        return features

    base = _xy(kps, SPINE_BASE_LUMBAR)
    top  = _xy(kps, SPINE_TOP_CERVICAL)

    if (base[1] - top[1]) <= 0:
        # Top of spine is not above (image-y-wise) the base — anatomically
        # implausible for an upright subject; likely a bad detection.
        flags.append("spine_top_not_above_base")
        features["geometry_failure_reason"] = "implausible_keypoint_ordering"
        features["curvature_features_reason"] = "implausible_keypoint_ordering"
        features["anatomy_warning_flags"] = "; ".join(flags)
        features["retake_or_review_reason"] = features["geometry_failure_reason"]
        return features

    features["valid_spine_geometry"] = True
    spine_length = _dist(base, top)
    features["spine_length_px"] = round(spine_length, 3)

    # ---- Core axis: base -> top chord ----
    features["spine_axis_inclination_deg"] = round(_segment_angle_deg(base, top), 3)
    if abs(features["spine_axis_inclination_deg"]) > UNUSUAL_INCLINATION_DEG:
        flags.append("unusual_spine_axis_inclination")

    # ---- Shoulders ----
    ls = rs = None
    if ls_ok and rs_ok:
        ls, rs = _xy(kps, LEFT_SHOULDER), _xy(kps, RIGHT_SHOULDER)
        features["shoulder_height_difference_px"] = round(abs(ls[1] - rs[1]), 3)
        features["shoulder_line_angle_deg"] = round(
            math.degrees(math.atan2(rs[1] - ls[1], rs[0] - ls[0])), 3
        )
        if abs(features["shoulder_line_angle_deg"]) > UNUSUAL_SHOULDER_ANGLE_DEG:
            flags.append("unusual_shoulder_line_angle")
        _check_shoulders_straddle_axis(ls, rs, base, top, spine_length, flags)

    # ---- Midline offset: horizontal shift of spine top vs base ----
    features["spine_midline_offset_px"] = round(abs(top[0] - base[0]), 3)

    # ---- Anatomical plausibility of intermediate points ----
    _check_intermediate_span_and_ordering(kps, base, top, visible_intermediate, spine_length, flags)

    # ---- Lateral deviation of intermediate points from the base-top reference line ----
    if visible_intermediate:
        deviations = [_point_to_line_distance(_xy(kps, i), base, top) for i in visible_intermediate]
        weights    = [float(kps[i, 2]) for i in visible_intermediate]

        features["max_lateral_deviation_px"]  = round(max(deviations), 3)
        features["mean_lateral_deviation_px"] = round(sum(deviations) / len(deviations), 3)

        weight_sum = sum(weights)
        if weight_sum > 1e-6:
            weighted = sum(d * w for d, w in zip(deviations, weights)) / weight_sum
            features["confidence_weighted_mean_lateral_deviation_px"] = round(weighted, 3)

        if spine_length > 1e-6:
            features["normalized_max_lateral_deviation"]  = round(features["max_lateral_deviation_px"] / spine_length, 4)
            features["normalized_mean_lateral_deviation"] = round(features["mean_lateral_deviation_px"] / spine_length, 4)

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
    if features["segment_angle_lower_deg"] is not None and features["segment_angle_upper_deg"] is not None:
        features["upper_lower_segment_angle_difference_deg"] = round(
            abs(_angle_diff_deg(features["segment_angle_upper_deg"], features["segment_angle_lower_deg"])), 3
        )

    # ---- Task 1: anatomy_valid_basic / anatomy_valid_intermediate ----
    features["anatomy_valid_basic"] = bool(
        ls_ok and rs_ok
        and "shoulders_not_straddling_spine_axis" not in flags
    )
    features["anatomy_valid_intermediate"] = bool(
        features["anatomy_valid_basic"]
        and visible_intermediate
        and "intermediate_outside_base_top_span" not in flags
        and "intermediate_ordering_violation" not in flags
        and "excessive_segment_jump" not in flags
    )

    # ---- Task 2: curvature feature reliability ----
    if not visible_intermediate:
        features["curvature_features_reliable"] = False
        features["curvature_features_reason"] = "no_intermediate_keypoints_visible"
    elif any(f in flags for f in ("intermediate_outside_base_top_span", "intermediate_ordering_violation", "excessive_segment_jump")):
        features["curvature_features_reliable"] = False
        features["curvature_features_reason"] = "anatomically_implausible_intermediate_points"
    elif len(visible_intermediate) < MIN_INTERMEDIATE_FOR_RELIABLE_CURVATURE:
        features["curvature_features_reliable"] = False
        features["curvature_features_reason"] = "too_few_intermediate_keypoints_experimental"
    else:
        features["curvature_features_reliable"] = True
        features["curvature_features_reason"] = "ok"

    features["anatomy_warning_flags"] = "; ".join(dict.fromkeys(flags))   # de-duplicate, keep order

    # ---- Task 4: reliability score ----
    score = _score_reliability(features, flags)
    features["anatomy_reliability_score"] = round(score, 4)
    features["reliability_level"] = _reliability_level(score)
    features["retake_or_review_recommended"] = features["reliability_level"] != "high"
    if features["reliability_level"] == "high":
        features["retake_or_review_reason"] = "ok"
    elif flags:
        features["retake_or_review_reason"] = features["anatomy_warning_flags"]
    else:
        features["retake_or_review_reason"] = f"low_reliability_score={features['anatomy_reliability_score']}"

    return features
