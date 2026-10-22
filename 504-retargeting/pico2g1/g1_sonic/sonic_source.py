"""Frame sources: recorded sessions and the live device, plus the frame bus."""
from __future__ import annotations

import json
from pathlib import Path
import threading
import time
import traceback

import numpy as np
from websockets.sync.client import connect

from sonic_config import FPS, FUTURE_FRAMES
from sonic_recording import (
    load_glove_recording,
    load_hand_trackers,
    load_recording,
    load_skeleton,
    parse_gloves,
    parse_vr,
    resample,
    resample_poses_to,
    resample_to,
)
from sonic_smpl import WujiMapper, apply_rotation_sources, convert



def resolve_body_path(path: Path) -> Path:
    if path.is_file():
        return path
    matches = sorted(path.glob("data/chunk-*/observation.data.vr_full_body_*.parquet"))
    if not matches:
        raise FileNotFoundError(f"no full-body parquet found under {path}")
    return matches[0]


def replay_frames(
    path: Path,
    rotation_offsets: dict[str, np.ndarray],
    pause_event: threading.Event,
):
    path = resolve_body_path(path)
    timestamps, raw_poses = load_recording(path)
    timestamps, raw_poses = resample(timestamps, raw_poses, True)
    tracker_path = path.with_name(path.name.replace("vr_full_body", "vr"))
    tracker_time, trackers = load_hand_trackers(tracker_path)
    trackers = resample_poses_to(tracker_time, trackers, timestamps)
    rest, parents = load_skeleton()
    gloves = {}
    for side in ("left", "right"):
        glove_path = path.with_name(path.name.replace("vr_full_body", f"glove_{side}"))
        glove_time, points = load_glove_recording(glove_path)
        _, gloves[side] = resample_to(glove_time, points, timestamps)
    mapper = WujiMapper()
    started = time.monotonic()
    for index, pose in enumerate(raw_poses):
        if pause_event.is_set():
            paused_at = time.monotonic()
            while pause_event.is_set():
                time.sleep(0.02)
            started += time.monotonic() - paused_at
        delay = started + index / FPS - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        future = np.minimum(index + np.arange(FUTURE_FRAMES), len(raw_poses) - 1)
        raw_gloves = {side: points[index] for side, points in gloves.items()}
        control_pose = apply_rotation_sources(
            raw_poses[future], trackers[future], rotation_offsets
        )
        reference = convert(control_pose, rest, parents)
        yield reference, pose, trackers[index], raw_gloves, mapper.retarget(raw_gloves)


def live_frames(host: str, token: str, rotation_offsets: dict[str, np.ndarray]):
    rest, parents = load_skeleton()
    mapper = WujiMapper()
    gloves = None
    last_timestamp = 0
    uri = f"ws://{host}:9990/subscribe?token={token}"
    with connect(uri, max_size=None) as source:
        hello = json.loads(source.recv())
        if hello.get("channel") != "hello":
            raise ValueError(f"expected hello message, got {hello}")
        source.send(json.dumps({"op": "subscribe", "channels": ["glove", "vr"], "fps": {"glove": 90, "vr": 90}}))
        for message in source:
            if isinstance(message, bytes):
                continue
            envelope = json.loads(message)
            channel, data = envelope.get("channel"), envelope.get("data") or {}
            if channel == "error":
                raise RuntimeError(data)
            try:
                if channel == "glove":
                    gloves = parse_gloves(data)
                elif channel == "vr" and gloves is not None:
                    timestamp, pose, trackers = parse_vr(data)
                    if timestamp - last_timestamp >= 1_000_000_000 // FPS:
                        last_timestamp = timestamp
                        control_pose = apply_rotation_sources(
                            pose[None], trackers[None], rotation_offsets
                        )
                        converted = convert(control_pose, rest, parents)
                        reference = {name: np.repeat(value, FUTURE_FRAMES, axis=0) for name, value in converted.items()}
                        yield reference, pose, trackers, gloves, mapper.retarget(gloves)
            except (KeyError, TypeError, ValueError) as error:
                print(f"dropped {channel} message: {error}")


class LatestFrame:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None
        self.error = None
        self.finished = False

    def put(self, frame) -> None:
        with self.lock:
            self.frame = frame

    def take(self):
        with self.lock:
            frame, error = self.frame, self.error
            self.frame = None
            self.error = None
            return frame, error, self.finished


def publish_frames(frames, latest: LatestFrame) -> None:
    try:
        for frame in frames:
            latest.put(frame)
    except Exception:
        with latest.lock:
            latest.error = traceback.format_exc()
    finally:
        with latest.lock:
            latest.finished = True
