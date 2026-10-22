"""The two drivers: replay a recorded session or stream the live device."""
from __future__ import annotations

from collections import deque
import json
from pathlib import Path
import time

import numpy as np
from websockets.sync.client import connect

from bridge_config import FPS, HAND_FIELDS, OUTPUT_FIELDS, SONIC_ENDPOINT, SONIC_TOPIC, WINDOW
from bridge_hands import (
    build_hand_retargeter,
    hand_references,
    load_hand_joints,
)
from bridge_protocol import (
    make_window,
    open_publisher,
    pack_message,
    unpack_message,
    validate_window,
)
from bridge_smpl import convert
from bridge_source import (
    apply_rotation_sources,
    interpolate,
    load_hand_trackers,
    load_recording,
    load_rotation_offsets,
    load_skeleton,
    parse_glove_data,
    parse_vr_data,
    resample,
    resample_poses_to,
)



def replay(path: Path) -> None:
    rest_joints, parents = load_skeleton()
    timestamps, poses = load_recording(path)
    timestamps, poses = resample(timestamps, poses)
    tracker_path = path.with_name(path.name.replace("vr_full_body", "vr"))
    tracker_timestamps, trackers = load_hand_trackers(tracker_path)
    trackers = resample_poses_to(tracker_timestamps, trackers, timestamps)
    poses = apply_rotation_sources(poses, trackers, load_rotation_offsets())
    frames = convert(poses, rest_joints, parents)
    frames.update(load_hand_joints(path, timestamps))
    first = make_window(frames, 0, np.arange(WINDOW))
    validate_window(first)
    validate_window(unpack_message(pack_message(first, SONIC_TOPIC), SONIC_TOPIC))
    print(f"converted {len(poses)} body and Dex3 hand frames; publishing on {SONIC_ENDPOINT}")

    context, socket = open_publisher()
    try:
        started = time.monotonic()
        for index in range(WINDOW - 1, len(poses)):
            delay = started + index / FPS - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            start = index + 1 - WINDOW
            window = make_window(frames, start, np.arange(start, index + 1))
            socket.send(pack_message(window, SONIC_TOPIC))
    except KeyboardInterrupt:
        pass
    finally:
        socket.close(0)
        context.term()


def live(host: str, token: str) -> None:
    rest_joints, parents = load_skeleton()
    rotation_offsets = load_rotation_offsets()
    hand_retargeters = {
        side: build_hand_retargeter(side) for side in ("left", "right")
    }
    output_context, output = open_publisher()
    buffer: deque[dict[str, np.ndarray]] = deque(maxlen=WINDOW)
    current_gloves = None
    previous_timestamp = None
    previous_pose = None
    previous_trackers = None
    previous_gloves = None
    next_timestamp = None
    frame_index = 0
    step_ns = int(round(1e9 / FPS))
    uri = f"ws://{host}:9990/subscribe?token={token}"
    print(f"receiving glove and VR data from {host}:9990; publishing on {SONIC_ENDPOINT}")

    try:
        with connect(uri, max_size=None) as source:
            hello = json.loads(source.recv())
            if hello.get("channel") != "hello":
                raise ValueError(f"expected hello message, got {hello}")
            source.send(
                json.dumps(
                    {
                        "op": "subscribe",
                        "channels": ["glove", "vr"],
                        "fps": {"glove": 90, "vr": 90},
                    }
                )
            )
            for message in source:
                if isinstance(message, bytes):
                    continue
                envelope = json.loads(message)
                channel = envelope.get("channel")
                data = envelope.get("data") or {}
                if channel == "error":
                    raise RuntimeError(data)
                if channel == "subscribed":
                    print(f"subscribed: {data}")
                    continue
                try:
                    if channel == "glove":
                        glove_poses = parse_glove_data(data)
                        current_gloves = {
                            side: glove_poses[side][[0, 4, 8, 12], :3]
                            for side in ("left", "right")
                        }
                        continue
                    if channel != "vr" or current_gloves is None:
                        continue
                    timestamp, pose, trackers = parse_vr_data(data)
                except (KeyError, TypeError, ValueError) as error:
                    print(f"dropped {channel} message: {error}")
                    continue

                if previous_timestamp is None:
                    previous_timestamp, previous_pose = timestamp, pose
                    previous_trackers = trackers
                    previous_gloves = current_gloves
                    next_timestamp = timestamp
                    continue
                gap = timestamp - previous_timestamp
                if gap <= 0 or gap > 200_000_000:
                    previous_timestamp, previous_pose = timestamp, pose
                    previous_trackers = trackers
                    previous_gloves = current_gloves
                    next_timestamp = timestamp
                    buffer.clear()
                    for retargeter, _ in hand_retargeters.values():
                        retargeter.reset()
                        if retargeter.filter is not None:
                            retargeter.filter.reset()
                    continue
                while next_timestamp is not None and next_timestamp <= timestamp:
                    alpha = (next_timestamp - previous_timestamp) / gap
                    if alpha >= 0.0:
                        alpha = np.clip(alpha, 0.0, 1.0)
                        sampled = interpolate(previous_pose, pose, alpha)
                        sampled_trackers = interpolate(
                            previous_trackers, trackers, alpha
                        )
                        sampled = apply_rotation_sources(
                            sampled[None], sampled_trackers[None], rotation_offsets
                        )[0]
                        converted = convert(sampled[None], rest_joints, parents)
                        frame = {name: converted[name][0] for name in OUTPUT_FIELDS}
                        for side in ("left", "right"):
                            points = (
                                (1.0 - alpha) * previous_gloves[side]
                                + alpha * current_gloves[side]
                            )
                            retargeter, order = hand_retargeters[side]
                            frame[f"{side}_hand_joints"] = retargeter.retarget(
                                hand_references(points, side)
                            )[order].astype(np.float32)
                        buffer.append(frame)
                        if len(buffer) == WINDOW:
                            window = {
                                name: np.stack([frame[name] for frame in buffer])
                                for name in OUTPUT_FIELDS
                            }
                            for name in HAND_FIELDS:
                                window[name] = buffer[-1][name]
                            window["frame_index"] = np.arange(
                                frame_index + 1 - WINDOW,
                                frame_index + 1,
                                dtype=np.int64,
                            )
                            output.send(pack_message(window, SONIC_TOPIC))
                        frame_index += 1
                    next_timestamp += step_ns
                previous_timestamp, previous_pose = timestamp, pose
                previous_trackers = trackers
                previous_gloves = current_gloves
    except KeyboardInterrupt:
        pass
    finally:
        output.close(0)
        output_context.term()
