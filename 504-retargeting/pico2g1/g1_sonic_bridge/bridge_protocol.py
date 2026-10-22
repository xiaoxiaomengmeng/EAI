"""ZMQ message packing/unpacking, window validation and the publisher socket."""
from __future__ import annotations

import json
import time

import numpy as np
import zmq

from bridge_config import (
    DTYPE_TO_NAME,
    HAND_FIELDS,
    HEADER_SIZE,
    NAME_TO_DTYPE,
    OUTPUT_FIELDS,
    SONIC_ENDPOINT,
    WINDOW,
)



def pack_message(arrays: dict[str, np.ndarray], topic: str, version: int = 3) -> bytes:
    fields = []
    payload = []
    for name, value in arrays.items():
        array = np.asarray(value)
        dtype_name = DTYPE_TO_NAME.get(array.dtype)
        if dtype_name is None:
            array = array.astype(np.float32)
            dtype_name = "f32"
        array = np.ascontiguousarray(array.astype(array.dtype.newbyteorder("<"), copy=False))
        fields.append({"name": name, "dtype": dtype_name, "shape": list(array.shape)})
        payload.append(array.tobytes())
    header = json.dumps(
        {"v": version, "endian": "le", "count": 1, "fields": fields},
        separators=(",", ":"),
    ).encode()
    if len(header) > HEADER_SIZE:
        raise ValueError("ZMQ header exceeds 1280 bytes")
    return topic.encode() + header.ljust(HEADER_SIZE, b"\0") + b"".join(payload)


def unpack_message(message: bytes, topic: str) -> dict[str, np.ndarray]:
    prefix = topic.encode()
    if not message.startswith(prefix) or len(message) < len(prefix) + HEADER_SIZE:
        raise ValueError(f"invalid {topic!r} message")
    offset = len(prefix)
    header = json.loads(message[offset : offset + HEADER_SIZE].rstrip(b"\0"))
    payload = memoryview(message)[offset + HEADER_SIZE :]
    arrays = {}
    cursor = 0
    for field in header["fields"]:
        dtype = NAME_TO_DTYPE.get(field["dtype"])
        if dtype is None:
            raise ValueError(f"unsupported dtype {field['dtype']!r}")
        shape = tuple(field["shape"])
        size = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        if cursor + size > len(payload):
            raise ValueError("truncated ZMQ payload")
        arrays[field["name"]] = np.frombuffer(
            payload[cursor : cursor + size],
            dtype=dtype.newbyteorder("<"),
        ).reshape(shape)
        cursor += size
    if cursor != len(payload):
        raise ValueError("unexpected trailing ZMQ payload")
    return arrays


def make_window(frames: dict[str, np.ndarray], start: int, indexes: np.ndarray) -> dict:
    window = {name: frames[name][start : start + WINDOW] for name in OUTPUT_FIELDS}
    for name in HAND_FIELDS:
        if name in frames:
            window[name] = frames[name][start + WINDOW - 1]
    window["frame_index"] = np.asarray(indexes, dtype=np.int64)
    return window


def validate_window(window: dict[str, np.ndarray]) -> None:
    shapes = {
        "joint_pos": (WINDOW, 29),
        "joint_vel": (WINDOW, 29),
        "body_quat": (WINDOW, 4),
        "smpl_joints": (WINDOW, 24, 3),
        "smpl_pose": (WINDOW, 21, 3),
        "frame_index": (WINDOW,),
    }
    for name, shape in shapes.items():
        if name not in window or window[name].shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
        if not np.isfinite(window[name]).all():
            raise ValueError(f"{name} contains invalid values")
    for name in HAND_FIELDS:
        if name in window and (
            window[name].shape != (7,) or not np.isfinite(window[name]).all()
        ):
            raise ValueError(f"{name} must contain 7 finite joint angles")
    if not np.allclose(np.linalg.norm(window["body_quat"], axis=1), 1.0, atol=1e-4):
        raise ValueError("body_quat must contain unit quaternions")


def open_publisher() -> tuple[zmq.Context, zmq.Socket]:
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.setsockopt(zmq.SNDHWM, 2)
    socket.bind(SONIC_ENDPOINT)
    time.sleep(1.0)
    return context, socket
