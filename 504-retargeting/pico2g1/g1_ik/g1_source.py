"""数据源:replay/live 帧生成、调参读写与跨线程帧总线。"""

from __future__ import annotations

import json
import threading
import time
import traceback
from pathlib import Path

import numpy as np
from websockets.sync.client import connect

from g1_config import FPS, TUNING_DEFAULTS, TUNING_PATH
from g1_ik import WujiMapper
from g1_recording import (
    apply_hand_trackers, load_glove_recording, load_hand_trackers, load_recording,
    parse_glove_data, parse_vr_data, resample, resample_points, resample_poses_to,
)


def replay_frames(
    path: Path,
    tuning: dict[str, float],
    pause_event: threading.Event,
):
    timestamps, raw_poses = load_recording(path)
    timestamps, raw_poses = resample(timestamps, raw_poses)
    tracker_path = path.with_name(path.name.replace("vr_full_body", "vr"))
    tracker_timestamps, trackers = load_hand_trackers(tracker_path)
    trackers = resample_poses_to(tracker_timestamps, trackers, timestamps)
    poses = apply_hand_trackers(raw_poses, trackers)
    mapper = WujiMapper(tuning)
    gloves = {}
    for side in ("left", "right"):
        glove_path = path.with_name(path.name.replace("vr_full_body", f"glove_{side}"))
        glove_timestamps, points = load_glove_recording(glove_path)
        gloves[side] = resample_points(glove_timestamps, points, timestamps)
    started = time.monotonic()
    for index, pose in enumerate(poses):
        if pause_event.is_set():
            paused_at = time.monotonic()
            while pause_event.is_set():
                time.sleep(0.02)
            started += time.monotonic() - paused_at
        delay = started + index / FPS - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        raw_gloves = {side: points[index] for side, points in gloves.items()}
        yield pose, raw_poses[index], trackers[index], raw_gloves, mapper.retarget(raw_gloves)


def live_frames(host: str, token: str, tuning: dict[str, float]):
    mapper = WujiMapper(tuning)
    gloves = None
    last_timestamp = 0
    uri = f"ws://{host}:9990/subscribe?token={token}"
    with connect(uri, max_size=None) as source:
        hello = json.loads(source.recv())
        if hello.get("channel") != "hello":
            raise ValueError(f"expected hello message, got {hello}")
        source.send(json.dumps({
            "op": "subscribe",
            "channels": ["glove", "vr"],
            "fps": {"glove": 90, "vr": 90},
        }))
        for message in source:
            if isinstance(message, bytes):
                continue
            envelope = json.loads(message)
            channel = envelope.get("channel")
            data = envelope.get("data") or {}
            if channel == "error":
                raise RuntimeError(data)
            try:
                if channel == "glove":
                    gloves = parse_glove_data(data)
                elif channel == "vr" and gloves is not None:
                    timestamp, pose, trackers = parse_vr_data(data)
                    if timestamp - last_timestamp >= 1_000_000_000 // FPS:
                        last_timestamp = timestamp
                        yield (
                            apply_hand_trackers(pose, trackers),
                            pose,
                            trackers,
                            gloves,
                            mapper.retarget(gloves),
                        )
            except (KeyError, TypeError, ValueError) as error:
                print(f"dropped {channel} message: {error}")


def load_tuning() -> dict[str, float]:
    tuning = TUNING_DEFAULTS.copy()
    try:
        saved = json.loads(TUNING_PATH.read_text())
        if not isinstance(saved, dict):
            raise TypeError("tuning.json must contain an object")
        for key in tuning.keys() & saved.keys():
            value = float(saved[key])
            if np.isfinite(value):
                tuning[key] = value
    except (FileNotFoundError, OSError, TypeError, ValueError):
        pass
    return tuning


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
