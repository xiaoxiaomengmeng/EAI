"""常量与路径:关节表、调参默认值、资产与 wuji-retargeting 位置。"""

from __future__ import annotations

import os
from pathlib import Path
import sys

FPS = 50
BODY_JOINT_SPEED_LIMIT = 6.0
# G1 model + meshes are vendored inside this repo (assets/g1_description/).
# wuji-retargeting is fetched by create_env.sh into <repo>/3rd. Both roots can
# be overridden with the G1_DESCRIPTION / WUJI_ROOT environment variables; nothing
# is read from $HOME.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

ASSETS = REPO_ROOT / "assets"
G1_DESCRIPTION = Path(os.environ.get("G1_DESCRIPTION", ASSETS / "g1_description"))
G1_XML = G1_DESCRIPTION / "g1_gear_wbc.xml"
G1_MESHES = G1_DESCRIPTION / "meshes"
XIXI_SCENE = ASSETS / "xixi_mujoco" / "environment.xml"
WUJI_ROOT = Path(os.environ.get("WUJI_ROOT", REPO_ROOT / "3rd" / "wuji-retargeting"))
WUJI_DESCRIPTION = WUJI_ROOT / "wuji_retargeting" / "wuji-description"
TUNING_PATH = Path(__file__).with_name("tuning.json")
TUNING_DEFAULTS = {
    "arm_scale": 1.0,
    "leg_scale": 1.0,
    "shoulder_width": 1.0,
    "hip_width": 1.0,
    "hand_response": 0.2,
    "damping": 0.04,
    "wrist_pos": 1.2,
    "ankle_pos": 1.2,
    "knee_pos": 0.8,
    "elbow_pos": 0.8,
    "torso_rot": 0.45,
    "wrist_rot": 0.25,
    "ankle_rot": 0.2,
    "left_wrist_x": 0.0,
    "left_wrist_y": 0.0,
    "left_wrist_z": 0.0,
    "right_wrist_x": 0.0,
    "right_wrist_y": 0.0,
    "right_wrist_z": 0.0,
}
WUJI_JOINTS = tuple(
    f"finger{finger}_joint{joint}" for finger in range(1, 6) for joint in range(1, 5)
)
JOINT_NAMES = (
    "waist", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hand", "right_hand",
)
HAND_TRACKER_NAMES = ("left_controller", "right_controller")
AXIS_COLORS = ("#ff5964", "#59d98e", "#54a7ff")
ROTATION_TASKS = (
    ("torso", "spine3", "torso_link", "torso_rot"),
    ("left_wrist", "left_wrist", "left_wrist_yaw_link", "wrist_rot"),
    ("right_wrist", "right_wrist", "right_wrist_yaw_link", "wrist_rot"),
    ("left_ankle", "left_foot", "left_ankle_roll_link", "ankle_rot"),
    ("right_ankle", "right_foot", "right_ankle_roll_link", "ankle_rot"),
)
GLOVE_JOINT_NAMES = (
    "Wrist",
    "ThumbProximal", "ThumbIntermediate", "ThumbDistal", "ThumbEnd",
    "IndexProximal", "IndexIntermediate", "IndexDistal", "IndexEnd",
    "MiddleProximal", "MiddleIntermediate", "MiddleDistal", "MiddleEnd",
    "RingProximal", "RingIntermediate", "RingDistal", "RingEnd",
    "PinkyProximal", "PinkyIntermediate", "PinkyDistal", "PinkyEnd",
)
BODY_EDGES = tuple(
    (JOINT_NAMES.index(start), JOINT_NAMES.index(end))
    for start, end in (
        ("waist", "spine1"), ("spine1", "spine2"), ("spine2", "spine3"),
        ("spine3", "neck"), ("neck", "head"),
        ("waist", "left_hip"), ("left_hip", "left_knee"),
        ("left_knee", "left_ankle"), ("left_ankle", "left_foot"),
        ("waist", "right_hip"), ("right_hip", "right_knee"),
        ("right_knee", "right_ankle"), ("right_ankle", "right_foot"),
        ("spine3", "left_collar"), ("left_collar", "left_shoulder"),
        ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
        ("left_wrist", "left_hand"),
        ("spine3", "right_collar"), ("right_collar", "right_shoulder"),
        ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
        ("right_wrist", "right_hand"),
    )
)
HAND_EDGES = tuple(
    (0, 1 + 4 * finger)
    for finger in range(5)
) + tuple(
    (1 + 4 * finger + joint, 2 + 4 * finger + joint)
    for finger in range(5)
    for joint in range(3)
)
