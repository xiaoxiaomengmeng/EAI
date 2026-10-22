"""Shared constants and paths for the SONIC data bridge."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation



FPS = 50
WINDOW = 5
HEADER_SIZE = 1280
SONIC_ENDPOINT = os.environ.get("SONIC_ENDPOINT", "tcp://*:5556")
SONIC_TOPIC = "pose"
# The human skeleton and Dex3 hand urdfs are vendored inside this repo (assets/).
# ASSETS_DIR can override the location; nothing is read from $HOME or the other
# two projects.
REPO_ROOT = Path(__file__).resolve().parent.parent
ROOT = Path(__file__).resolve().parent
ASSETS = Path(os.environ.get("ASSETS_DIR", REPO_ROOT / "assets"))
SKELETON_PATH = ASSETS / "human_joints_info.pkl"
ROTATION_OFFSETS_PATH = ROOT / "rotation_offsets.json"
HAND_FIELDS = ("left_hand_joints", "right_hand_joints")
GLOVE_POINTS = ("wrist", "thumbend", "indexend", "middleend")
HAND_ORIGIN = np.array([2, 3, 3, 0, 0, 0])
HAND_TASK = np.array([1, 1, 2, 1, 2, 3])
GLOVE_TO_DEX3 = {
    "left": np.array([[-1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32),
    "right": np.array([[1, 0, 0], [0, 0, -1], [0, -1, 0]], dtype=np.float32),
}

JOINT_NAMES = (
    "waist",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hand",
    "right_hand",
)
HAND_TRACKER_NAMES = ("left_controller", "right_controller")
ROTATION_OFFSET_JOINTS = (
    "left_elbow", "right_elbow", "left_wrist", "right_wrist",
)
GLOVE_JOINT_NAMES = (
    "Wrist",
    "ThumbProximal", "ThumbIntermediate", "ThumbDistal", "ThumbEnd",
    "IndexProximal", "IndexIntermediate", "IndexDistal", "IndexEnd",
    "MiddleProximal", "MiddleIntermediate", "MiddleDistal", "MiddleEnd",
    "RingProximal", "RingIntermediate", "RingDistal", "RingEnd",
    "PinkyProximal", "PinkyIntermediate", "PinkyDistal", "PinkyEnd",
)
PARENTS_24 = np.array(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
)
Y_180 = Rotation.from_euler("y", 180.0, degrees=True).as_matrix()
Y_UP_TO_Z_UP = Rotation.from_euler("x", 90.0, degrees=True).as_matrix()
SMPL_BASE_INV = Rotation.from_quat([-0.5, -0.5, -0.5, 0.5]).as_matrix()
SMPL_OUTPUT_JOINTS = np.array([*range(22), 39, 54])
OUTPUT_FIELDS = (
    "joint_pos",
    "joint_vel",
    "body_quat",
    "smpl_joints",
    "smpl_pose",
)
DTYPE_TO_NAME = {
    np.dtype(np.float32): "f32",
    np.dtype(np.float64): "f64",
    np.dtype(np.int32): "i32",
    np.dtype(np.int64): "i64",
    np.dtype(np.bool_): "bool",
    np.dtype(np.uint8): "u8",
}
NAME_TO_DTYPE = {name: dtype for dtype, name in DTYPE_TO_NAME.items()}
