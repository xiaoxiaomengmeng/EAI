"""Shared constants and paths for the G1 SONIC simulator.

The REPO_ROOT path is inserted into sys.path on first import so that
`from sim_scenes import ...` works from any module of this package."""
from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation



FPS = 50
FUTURE_FRAMES = 10
HISTORY_FRAMES = 10
SIM_SUBSTEPS = 10
CAMERA_FPS = 30
ROOT = Path(__file__).resolve().parent
# G1 model + meshes and the human skeleton are vendored inside this repo
# (assets/). wuji-retargeting and the SONIC ONNX models are fetched by setup.sh
# (into <repo>/3rd and ./models). G1_DESCRIPTION / WUJI_ROOT env vars can
# override the roots; nothing is read from $HOME.
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

ASSETS = REPO_ROOT / "assets"
G1_DESCRIPTION = Path(os.environ.get("G1_DESCRIPTION", ASSETS / "g1_description"))
G1_XML = G1_DESCRIPTION / "g1_gear_wbc.xml"
G1_MESHES = G1_DESCRIPTION / "meshes"
XIXI_SCENE = ASSETS / "xixi_mujoco" / "environment.xml"
WUJI_ROOT = Path(os.environ.get("WUJI_ROOT", REPO_ROOT / "3rd" / "wuji-retargeting"))
WUJI_DESCRIPTION = WUJI_ROOT / "wuji_retargeting" / "wuji-description"
SKELETON_PATH = ASSETS / "human_joints_info.pkl"
ROTATION_OFFSETS_PATH = ROOT / "rotation_offsets.json"

JOINT_NAMES = (
    "waist", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hand", "right_hand",
)
HAND_TRACKER_NAMES = ("left_controller", "right_controller")
GLOVE_JOINT_NAMES = (
    "Wrist",
    "ThumbProximal", "ThumbIntermediate", "ThumbDistal", "ThumbEnd",
    "IndexProximal", "IndexIntermediate", "IndexDistal", "IndexEnd",
    "MiddleProximal", "MiddleIntermediate", "MiddleDistal", "MiddleEnd",
    "RingProximal", "RingIntermediate", "RingDistal", "RingEnd",
    "PinkyProximal", "PinkyIntermediate", "PinkyDistal", "PinkyEnd",
)
ROTATION_OFFSET_JOINTS = (
    ("left_elbow", "左肘"), ("right_elbow", "右肘"),
    ("left_wrist", "左腕 tracker"), ("right_wrist", "右腕 tracker"),
)
AXIS_COLORS = ("#ff5964", "#59d98e", "#54a7ff")
WUJI_JOINTS = tuple(
    f"finger{finger}_joint{joint}" for finger in range(1, 6) for joint in range(1, 5)
)
PARENTS_24 = np.array(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
)
SMPL_OUTPUT_JOINTS = np.array([*range(22), 39, 54])
Y_180 = Rotation.from_euler("y", 180.0, degrees=True).as_matrix()
Y_UP_TO_Z_UP = Rotation.from_euler("x", 90.0, degrees=True).as_matrix()
SMPL_BASE_INV = Rotation.from_quat([-0.5, -0.5, -0.5, 0.5]).as_matrix()

ISAACLAB_TO_MUJOCO = np.array(
    [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28]
)
MUJOCO_TO_ISAACLAB = np.array(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28]
)
WRIST_ISAACLAB = np.array([23, 24, 25, 26, 27, 28])
DEFAULT_ANGLES = np.array(
    [-0.312, 0, 0, 0.669, -0.363, 0, -0.312, 0, 0, 0.669, -0.363, 0,
     0, 0, 0, 0.2, 0.2, 0, 0.6, 0, 0, 0, 0.2, -0.2, 0, 0.6, 0, 0, 0],
    dtype=np.float32,
)
ARMATURE = {"5020": 0.003609725, "7520_14": 0.010177520,
            "7520_22": 0.025101925, "4010": 0.00425}
EFFORT = {"5020": 25.0, "7520_14": 88.0, "7520_22": 139.0, "4010": 5.0}
MOTOR_TYPES = (
    "7520_22", "7520_22", "7520_14", "7520_22", "5020", "5020",
    "7520_22", "7520_22", "7520_14", "7520_22", "5020", "5020",
    "7520_14", "5020", "5020",
    "5020", "5020", "5020", "5020", "5020", "4010", "4010",
    "5020", "5020", "5020", "5020", "5020", "4010", "4010",
)
NATURAL_FREQ = 20.0 * np.pi
KP = np.array([ARMATURE[k] * NATURAL_FREQ**2 for k in MOTOR_TYPES], dtype=np.float32)
KD = np.array([4.0 * ARMATURE[k] * NATURAL_FREQ for k in MOTOR_TYPES], dtype=np.float32)
KP[[4, 5, 10, 11, 13, 14]] *= 2.0
KD[[4, 5, 10, 11, 13, 14]] *= 2.0
ACTION_SCALE = np.array(
    [0.25 * EFFORT[k] / (ARMATURE[k] * NATURAL_FREQ**2) for k in MOTOR_TYPES],
    dtype=np.float32,
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
HAND_EDGES = tuple((0, 1 + 4 * finger) for finger in range(5)) + tuple(
    (1 + 4 * finger + joint, 2 + 4 * finger + joint)
    for finger in range(5) for joint in range(3)
)
