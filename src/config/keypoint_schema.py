# ============================================================
# Central Keypoint Schema — Fine-Tuned YOLOv8 Spine Model
#
# Single source of truth for the 12-point keypoint layout used
# by the fine-tuned spine model (train_yolo_scoliosis.py /
# train_yolo_combined.py). Other scripts should import these
# constants instead of redefining keypoint indices locally.
#
# NOTE: The pretrained COCO-pose YOLO workflow (yolov8n-pose.pt,
# 17 keypoints) uses a different, unrelated schema and is not
# covered here.
#
#   kp[0]  spine_base_lumbar      kp[6]  spine_intermediate_6
#   kp[1]  spine_intermediate_1   kp[7]  spine_intermediate_7
#   kp[2]  spine_intermediate_2   kp[8]  spine_top_cervical
#   kp[3]  spine_intermediate_3   kp[9]  left_shoulder
#   kp[4]  spine_intermediate_4   kp[10] right_shoulder
#   kp[5]  spine_intermediate_5   kp[11] neck_head_center
# ============================================================

NUM_KEYPOINTS = 12

KEYPOINT_MAP = {
    0:  "spine_base_lumbar",
    1:  "spine_intermediate_1",
    2:  "spine_intermediate_2",
    3:  "spine_intermediate_3",
    4:  "spine_intermediate_4",
    5:  "spine_intermediate_5",
    6:  "spine_intermediate_6",
    7:  "spine_intermediate_7",
    8:  "spine_top_cervical",
    9:  "left_shoulder",
    10: "right_shoulder",
    11: "neck_head_center",
}

# Named constants
SPINE_BASE_LUMBAR    = 0
SPINE_INTERMEDIATE_1 = 1
SPINE_INTERMEDIATE_2 = 2
SPINE_INTERMEDIATE_3 = 3
SPINE_INTERMEDIATE_4 = 4
SPINE_INTERMEDIATE_5 = 5
SPINE_INTERMEDIATE_6 = 6
SPINE_INTERMEDIATE_7 = 7
SPINE_TOP_CERVICAL    = 8
LEFT_SHOULDER         = 9
RIGHT_SHOULDER        = 10
NECK_HEAD_CENTER      = 11

# Grouped index lists
SPINE_POINT_IDXS        = list(range(9))    # kp[0..8] — full spine incl. base + top anchors
INTERMEDIATE_SPINE_IDXS = list(range(1, 8)) # kp[1..7] — excludes base/top anchors
SHOULDER_IDXS           = [LEFT_SHOULDER, RIGHT_SHOULDER]

# Shared default for "is this keypoint reliable enough to use" checks
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
