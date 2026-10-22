"""录制数据:parquet 加载、设备消息解析与时间戳重采样。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from g1_config import FPS, GLOVE_JOINT_NAMES, HAND_TRACKER_NAMES, JOINT_NAMES


def load_recording(path: Path) -> tuple[np.ndarray, np.ndarray]:
    table = pq.read_table(path, columns=["timestamp_ns", "available", *JOINT_NAMES])
    timestamps = table["timestamp_ns"].to_numpy(zero_copy_only=False).astype(np.int64)
    available = table["available"].to_numpy(zero_copy_only=False).astype(bool)
    poses = np.stack(
        [np.asarray(table[name].to_pylist(), dtype=np.float32) for name in JOINT_NAMES],
        axis=1,
    )
    valid = (
        available
        & np.isfinite(poses).all(axis=(1, 2))
        & (np.linalg.norm(poses[..., 3:], axis=-1) > 1e-8).all(axis=1)
    )
    timestamps, poses = timestamps[valid], poses[valid]
    if len(timestamps) < 2 or np.any(np.diff(timestamps) <= 0):
        raise ValueError("recording must contain increasing timestamps and valid frames")
    return timestamps, poses


def load_hand_trackers(path: Path) -> tuple[np.ndarray, np.ndarray]:
    table = pq.read_table(path, columns=["timestamp_ns", *HAND_TRACKER_NAMES])
    timestamps = table["timestamp_ns"].to_numpy(zero_copy_only=False).astype(np.int64)
    poses = np.stack(
        [np.asarray(table[name].to_pylist(), dtype=np.float32) for name in HAND_TRACKER_NAMES],
        axis=1,
    )
    valid = (
        np.isfinite(poses).all(axis=(1, 2))
        & (np.linalg.norm(poses[..., 3:], axis=-1) > 1e-8).all(axis=1)
    )
    timestamps, poses = timestamps[valid], poses[valid]
    if len(timestamps) < 2 or np.any(np.diff(timestamps) <= 0):
        raise ValueError("hand tracker recording must contain increasing timestamps")
    return timestamps, poses


def slerp(q0: np.ndarray, q1: np.ndarray, alpha: np.ndarray | float) -> np.ndarray:
    q0 = q0.astype(np.float64)
    q1 = q1.astype(np.float64)
    q0 /= np.linalg.norm(q0, axis=-1, keepdims=True)
    q1 /= np.linalg.norm(q1, axis=-1, keepdims=True)
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0.0, -q1, q1)
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    weight = np.asarray(alpha, dtype=np.float64)
    while weight.ndim < dot.ndim:
        weight = weight[..., None]
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    denominator = np.where(sin_theta > 1e-6, sin_theta, 1.0)
    spherical = (
        np.sin((1.0 - weight) * theta) / denominator * q0
        + np.sin(weight * theta) / denominator * q1
    )
    linear = (1.0 - weight) * q0 + weight * q1
    result = np.where(sin_theta > 1e-6, spherical, linear)
    return (result / np.linalg.norm(result, axis=-1, keepdims=True)).astype(np.float32)


def interpolate(pose0: np.ndarray, pose1: np.ndarray, alpha: np.ndarray | float) -> np.ndarray:
    weight = np.asarray(alpha, dtype=np.float64)
    while weight.ndim < pose0[..., :3].ndim:
        weight = weight[..., None]
    positions = (1.0 - weight) * pose0[..., :3] + weight * pose1[..., :3]
    rotations = slerp(pose0[..., 3:], pose1[..., 3:], alpha)
    return np.concatenate((positions, rotations), axis=-1).astype(np.float32)


def resample(timestamps: np.ndarray, poses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    step_ns = int(round(1e9 / FPS))
    target = np.arange(timestamps[0], timestamps[-1] + 1, step_ns, dtype=np.int64)
    right = np.clip(np.searchsorted(timestamps, target, side="left"), 1, len(timestamps) - 1)
    left = right - 1
    alpha = (
        (target - timestamps[left]).astype(np.float64)
        / (timestamps[right] - timestamps[left]).astype(np.float64)
    )
    return target, interpolate(poses[left], poses[right], alpha[:, None])


def resample_poses_to(
    timestamps: np.ndarray, poses: np.ndarray, target: np.ndarray
) -> np.ndarray:
    right = np.clip(
        np.searchsorted(timestamps, target, side="left"), 1, len(timestamps) - 1
    )
    left = right - 1
    alpha = np.clip(
        (target - timestamps[left]).astype(np.float64)
        / (timestamps[right] - timestamps[left]).astype(np.float64),
        0.0,
        1.0,
    )
    return interpolate(poses[left], poses[right], alpha[:, None])


def apply_hand_trackers(poses: np.ndarray, trackers: np.ndarray) -> np.ndarray:
    result = poses.copy()
    for side, tracker_name in enumerate(HAND_TRACKER_NAMES):
        wrist = JOINT_NAMES.index(f"{tracker_name.split('_')[0]}_wrist")
        result[..., wrist, 3:] = trackers[..., side, 3:]
    return result


def resample_points(
    timestamps: np.ndarray, points: np.ndarray, target: np.ndarray
) -> np.ndarray:
    right = np.clip(np.searchsorted(timestamps, target, side="left"), 1, len(timestamps) - 1)
    left = right - 1
    alpha = np.clip(
        (target - timestamps[left]).astype(np.float64)
        / (timestamps[right] - timestamps[left]).astype(np.float64),
        0.0,
        1.0,
    )
    return (
        (1.0 - alpha[:, None, None]) * points[left]
        + alpha[:, None, None] * points[right]
    ).astype(np.float32)


def named_poses(data: list[dict], names: tuple[str, ...], label: str) -> np.ndarray:
    poses_by_name = {item.get("name"): item for item in data}
    missing = [name for name in names if name not in poses_by_name]
    if missing:
        raise ValueError(f"{label} missing joints: {', '.join(missing)}")
    poses = np.asarray(
        [poses_by_name[name]["pos"] + poses_by_name[name]["rot"] for name in names],
        dtype=np.float32,
    )
    if poses.shape != (len(names), 7) or not np.isfinite(poses).all():
        raise ValueError(f"{label} poses must contain finite [x,y,z,qx,qy,qz,qw]")
    if np.any(np.linalg.norm(poses[:, 3:], axis=1) <= 1e-8):
        raise ValueError(f"{label} contains invalid quaternions")
    return poses


def parse_vr_data(data: dict) -> tuple[int, np.ndarray, np.ndarray]:
    body = data.get("body_tracking") or {}
    if not body.get("available"):
        raise ValueError("VR body tracking is unavailable")
    return (
        int(data["timestamp_ns"]),
        named_poses(body.get("joints") or [], JOINT_NAMES, "VR body"),
        named_poses(data.get("joints") or [], HAND_TRACKER_NAMES, "VR hand trackers"),
    )


def parse_glove_data(data: dict) -> dict[str, np.ndarray]:
    gloves = {}
    for side in ("left", "right"):
        hand = data.get(side) or {}
        if hand.get("calibration") != 3:
            raise ValueError(f"{side} glove is not calibrated: {hand.get('calibration')}")
        gloves[side] = named_poses(
            hand.get("pose_joints") or [], GLOVE_JOINT_NAMES, f"{side} glove"
        )
    return gloves


def load_glove_recording(path: Path) -> tuple[np.ndarray, np.ndarray]:
    names = [name.lower() for name in GLOVE_JOINT_NAMES]
    table = pq.read_table(path, columns=["recv_timestamp_ns", "calibration_state", *names])
    timestamps = table["recv_timestamp_ns"].to_numpy(zero_copy_only=False).astype(np.int64)
    calibrated = table["calibration_state"].to_numpy(zero_copy_only=False) == 3
    points = np.stack(
        [np.asarray(table[name].to_pylist(), dtype=np.float32)[..., :3] for name in names],
        axis=1,
    )
    valid = calibrated & np.isfinite(points).all(axis=(1, 2))
    timestamps, points = timestamps[valid], points[valid]
    if len(timestamps) < 2 or np.any(np.diff(timestamps) <= 0):
        raise ValueError(f"{path.name} must contain increasing calibrated frames")
    return timestamps, points
