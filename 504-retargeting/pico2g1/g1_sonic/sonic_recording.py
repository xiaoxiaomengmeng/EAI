"""Recording and live-frame loading: parquet IO, resampling, device parsing.

load_skeleton reads the vendored torch.save archive as numpy without a torch
install (see the note above _TorchArchiveUnpickler)."""
from __future__ import annotations

import json
from pathlib import Path
import pickle
import zipfile

import numpy as np
import pyarrow.parquet as pq

from sonic_config import (
    FPS,
    GLOVE_JOINT_NAMES,
    HAND_TRACKER_NAMES,
    JOINT_NAMES,
    ROTATION_OFFSET_JOINTS,
    ROTATION_OFFSETS_PATH,
    SKELETON_PATH,
)



# torch.save writes a zip archive: a data.pkl holding the object graph, with each
# tensor's payload in a sibling data/<key> entry and referenced by a persistent id
# instead of being inlined. human_joints_info.pkl is such an archive, and its
# tensors (J and rot_mats) are contiguous little-endian float32 at offset 0, so
# numpy can map them directly. Unpickling it here rather than via torch.load
# keeps a ~590 MB torch install out of this venv for the sake of one 4 KB file:
# torch is not used for anything else here, and this simulator runs its policy
# through onnxruntime. Note that g1_sonic_bridge still needs torch -- not for
# its own code, but because dex_retargeting refuses to import without it.
STORAGE_DTYPES = {"FloatStorage": "<f4", "DoubleStorage": "<f8", "LongStorage": "<i8"}


def _rebuild_tensor(storage: np.ndarray, offset: int, size, stride, *_) -> np.ndarray:
    if not size:
        return storage[offset:offset + 1].reshape(())
    return np.lib.stride_tricks.as_strided(
        storage[offset:], shape=tuple(size),
        strides=tuple(s * storage.itemsize for s in stride),
    )


class _TorchArchiveUnpickler(pickle.Unpickler):
    """Reads the tensors in a torch.save archive as numpy arrays, without torch."""

    def __init__(self, file, archive: zipfile.ZipFile, prefix: str):
        super().__init__(file)
        self.archive = archive
        self.prefix = prefix

    def find_class(self, module: str, name: str):
        if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
            return _rebuild_tensor
        # Storage classes only ever appear inside a persistent id, as a dtype tag.
        if module == "torch" and name in STORAGE_DTYPES:
            return name
        if module.startswith("torch"):
            raise pickle.UnpicklingError(f"unsupported torch object {module}.{name}")
        return super().find_class(module, name)

    def persistent_load(self, pid):
        kind, storage_type, key, _location, count = pid
        if kind != "storage":
            raise pickle.UnpicklingError(f"unsupported persistent id {kind}")
        raw = self.archive.read(f"{self.prefix}/data/{key}")
        return np.frombuffer(raw, dtype=STORAGE_DTYPES[storage_type], count=count)


def load_skeleton() -> tuple[np.ndarray, np.ndarray]:
    with zipfile.ZipFile(SKELETON_PATH) as archive:
        pickle_name = next(n for n in archive.namelist() if n.endswith("/data.pkl"))
        prefix = pickle_name.rsplit("/", 1)[0]
        byteorder = archive.read(f"{prefix}/byteorder").decode().strip()
        if byteorder != "little":
            raise ValueError(f"{SKELETON_PATH.name} is {byteorder}-endian")
        with archive.open(pickle_name) as stream:
            data = _TorchArchiveUnpickler(stream, archive, prefix).load()
        joints = np.array(data["J"], dtype=np.float32)
    parents = np.asarray(data["parents_list"], dtype=np.int64)
    if joints.shape != (55, 3) or parents.shape != (55,):
        raise ValueError("invalid human_joints_info.pkl")
    return joints, parents


def load_rotation_offsets() -> dict[str, np.ndarray]:
    offsets = {
        name: np.zeros(3, dtype=np.float64) for name, _ in ROTATION_OFFSET_JOINTS
    }
    try:
        saved = json.loads(ROTATION_OFFSETS_PATH.read_text())
        if not isinstance(saved, dict):
            return offsets
        for name in offsets.keys() & saved.keys():
            value = np.asarray(saved[name], dtype=np.float64)
            if value.shape == (3,) and np.isfinite(value).all():
                offsets[name][:] = value
    except (FileNotFoundError, OSError, TypeError, ValueError):
        pass
    return offsets


def slerp(q0: np.ndarray, q1: np.ndarray, alpha: np.ndarray | float) -> np.ndarray:
    q0 = q0.astype(np.float64) / np.linalg.norm(q0, axis=-1, keepdims=True)
    q1 = q1.astype(np.float64) / np.linalg.norm(q1, axis=-1, keepdims=True)
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0.0, -q1, q1)
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    weight = np.asarray(alpha, dtype=np.float64)
    while weight.ndim < dot.ndim:
        weight = weight[..., None]
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    denominator = np.where(sin_theta > 1e-6, sin_theta, 1.0)
    spherical = (np.sin((1.0 - weight) * theta) / denominator * q0
                 + np.sin(weight * theta) / denominator * q1)
    linear = (1.0 - weight) * q0 + weight * q1
    result = np.where(sin_theta > 1e-6, spherical, linear)
    return (result / np.linalg.norm(result, axis=-1, keepdims=True)).astype(np.float32)


def interpolate(pose0: np.ndarray, pose1: np.ndarray, alpha: np.ndarray | float) -> np.ndarray:
    weight = np.asarray(alpha, dtype=np.float64)
    while weight.ndim < pose0[..., :3].ndim:
        weight = weight[..., None]
    positions = (1.0 - weight) * pose0[..., :3] + weight * pose1[..., :3]
    return np.concatenate((positions, slerp(pose0[..., 3:], pose1[..., 3:], alpha)), axis=-1).astype(np.float32)


def resample(timestamps: np.ndarray, values: np.ndarray, quaternions: bool) -> tuple[np.ndarray, np.ndarray]:
    step_ns = round(1e9 / FPS)
    target = np.arange(timestamps[0], timestamps[-1] + 1, step_ns, dtype=np.int64)
    right = np.clip(np.searchsorted(timestamps, target, side="left"), 1, len(timestamps) - 1)
    left = right - 1
    alpha = np.clip((target - timestamps[left]) / (timestamps[right] - timestamps[left]), 0.0, 1.0)
    if quaternions:
        return target, interpolate(values[left], values[right], alpha[:, None])
    return target, ((1.0 - alpha[:, None, None]) * values[left] + alpha[:, None, None] * values[right]).astype(np.float32)


def load_recording(path: Path) -> tuple[np.ndarray, np.ndarray]:
    table = pq.read_table(path, columns=["timestamp_ns", "available", *JOINT_NAMES])
    timestamps = table["timestamp_ns"].to_numpy(zero_copy_only=False).astype(np.int64)
    available = table["available"].to_numpy(zero_copy_only=False).astype(bool)
    poses = np.stack([np.asarray(table[name].to_pylist(), dtype=np.float32) for name in JOINT_NAMES], axis=1)
    valid = (available & np.isfinite(poses).all(axis=(1, 2))
             & (np.linalg.norm(poses[..., 3:], axis=-1) > 1e-8).all(axis=1))
    timestamps, poses = timestamps[valid], poses[valid]
    if len(timestamps) < 2 or np.any(np.diff(timestamps) <= 0):
        raise ValueError("body recording must contain increasing timestamps and valid frames")
    return timestamps, poses


def load_hand_trackers(path: Path) -> tuple[np.ndarray, np.ndarray]:
    table = pq.read_table(path, columns=["timestamp_ns", *HAND_TRACKER_NAMES])
    timestamps = table["timestamp_ns"].to_numpy(zero_copy_only=False).astype(np.int64)
    poses = np.stack([
        np.asarray(table[name].to_pylist(), dtype=np.float32)
        for name in HAND_TRACKER_NAMES
    ], axis=1)
    valid = (np.isfinite(poses).all(axis=(1, 2))
             & (np.linalg.norm(poses[..., 3:], axis=-1) > 1e-8).all(axis=1))
    timestamps, poses = timestamps[valid], poses[valid]
    if len(timestamps) < 2 or np.any(np.diff(timestamps) <= 0):
        raise ValueError("hand tracker recording must contain increasing timestamps and valid frames")
    return timestamps, poses


def load_glove_recording(path: Path) -> tuple[np.ndarray, np.ndarray]:
    names = [name.lower() for name in GLOVE_JOINT_NAMES]
    table = pq.read_table(path, columns=["recv_timestamp_ns", "calibration_state", *names])
    timestamps = table["recv_timestamp_ns"].to_numpy(zero_copy_only=False).astype(np.int64)
    calibrated = table["calibration_state"].to_numpy(zero_copy_only=False) == 3
    points = np.stack([np.asarray(table[name].to_pylist(), dtype=np.float32)[..., :3] for name in names], axis=1)
    valid = calibrated & np.isfinite(points).all(axis=(1, 2))
    timestamps, points = timestamps[valid], points[valid]
    if len(timestamps) < 2 or np.any(np.diff(timestamps) <= 0):
        raise ValueError(f"{path.name} must contain increasing calibrated frames")
    return timestamps, points


def named_poses(data: list[dict], names: tuple[str, ...], label: str) -> np.ndarray:
    by_name = {item.get("name"): item for item in data}
    missing = [name for name in names if name not in by_name]
    if missing:
        raise ValueError(f"{label} missing joints: {', '.join(missing)}")
    poses = np.asarray([by_name[name]["pos"] + by_name[name]["rot"] for name in names], dtype=np.float32)
    if poses.shape != (len(names), 7) or not np.isfinite(poses).all():
        raise ValueError(f"{label} must contain finite [x,y,z,qx,qy,qz,qw] poses")
    return poses


def parse_vr(data: dict) -> tuple[int, np.ndarray, np.ndarray]:
    body = data.get("body_tracking") or {}
    if not body.get("available"):
        raise ValueError("VR body tracking is unavailable")
    return (
        int(data["timestamp_ns"]),
        named_poses(body.get("joints") or [], JOINT_NAMES, "VR body"),
        named_poses(data.get("joints") or [], HAND_TRACKER_NAMES, "VR hand trackers"),
    )


def parse_gloves(data: dict) -> dict[str, np.ndarray]:
    result = {}
    for side in ("left", "right"):
        hand = data.get(side) or {}
        if hand.get("calibration") != 3:
            raise ValueError(f"{side} glove is not calibrated")
        result[side] = named_poses(hand.get("pose_joints") or [], GLOVE_JOINT_NAMES, f"{side} glove")
    return result


def resample_to(timestamps: np.ndarray, points: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    right = np.clip(np.searchsorted(timestamps, target, side="left"), 1, len(timestamps) - 1)
    left = right - 1
    alpha = np.clip((target - timestamps[left]) / (timestamps[right] - timestamps[left]), 0.0, 1.0)
    values = (1.0 - alpha[:, None, None]) * points[left] + alpha[:, None, None] * points[right]
    return target, values.astype(np.float32)


def resample_poses_to(timestamps: np.ndarray, poses: np.ndarray, target: np.ndarray) -> np.ndarray:
    right = np.clip(np.searchsorted(timestamps, target, side="left"), 1, len(timestamps) - 1)
    left = right - 1
    alpha = np.clip((target - timestamps[left]) / (timestamps[right] - timestamps[left]), 0.0, 1.0)
    return interpolate(poses[left], poses[right], alpha[:, None])
