#!/usr/bin/env python3
"""Save one complete glove message and one complete VR message from the device."""

import argparse
import asyncio
import json
from pathlib import Path

import websockets


GLOVE_JOINTS = (
    "Wrist",
    "ThumbProximal", "ThumbIntermediate", "ThumbDistal", "ThumbEnd",
    "IndexProximal", "IndexIntermediate", "IndexDistal", "IndexEnd",
    "MiddleProximal", "MiddleIntermediate", "MiddleDistal", "MiddleEnd",
    "RingProximal", "RingIntermediate", "RingDistal", "RingEnd",
    "PinkyProximal", "PinkyIntermediate", "PinkyDistal", "PinkyEnd",
)

VR_BODY_JOINTS = (
    "waist", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hand", "right_hand",
)


def print_glove_summary(data: dict) -> None:
    print(f"glove: timestamp_ms={data.get('timestamp_ms')} frame={data.get('frame_index')}")
    for side in ("left", "right"):
        hand = data.get(side) or {}
        joints = hand.get("pose_joints") or []
        names = {joint.get("name") for joint in joints}
        missing = [name for name in GLOVE_JOINTS if name not in names]
        print(
            f"  {side}: joints={len(joints)} calibration={hand.get('calibration')} "
            f"missing={missing or 'none'}"
        )


def print_vr_summary(data: dict) -> None:
    body = data.get("body_tracking") or {}
    joints = body.get("joints") or []
    names = {joint.get("name") for joint in joints}
    missing = [name for name in VR_BODY_JOINTS if name not in names]
    print(
        f"vr: timestamp_ns={data.get('timestamp_ns')} available={body.get('available')} "
        f"device_poses={len(data.get('joints') or [])} body_joints={len(joints)} "
        f"missing={missing or 'none'}"
    )


async def capture(host: str, token: str) -> None:
    uri = f"ws://{host}:9990/subscribe?token={token}"
    async with websockets.connect(uri, max_size=None) as websocket:
        hello = json.loads(await websocket.recv())
        print(f"connected: {hello}")
        await websocket.send(
            json.dumps(
                {
                    "op": "subscribe",
                    "channels": ["glove", "vr"],
                    "fps": {"glove": 90, "vr": 90},
                }
            )
        )

        pending = {
            "glove": Path("glove_pose_sample.json"),
            "vr": Path("vr_pose_sample.json"),
        }
        while pending:
            message = await asyncio.wait_for(websocket.recv(), timeout=30)
            if isinstance(message, bytes):
                continue
            message_data = json.loads(message)
            channel = message_data.get("channel")
            data = message_data.get("data") or {}
            if channel == "error":
                raise RuntimeError(data)
            if channel == "subscribed":
                print(f"subscribed: {data}")
            if channel in pending:
                if channel == "glove":
                    print_glove_summary(data)
                else:
                    print_vr_summary(data)
                path = pending.pop(channel)
                path.write_text(
                    json.dumps(message_data, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                print(f"saved: {path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("host", help="RK3588 IP address, for example 192.168.2.111")
    parser.add_argument("--token", default="rynn_dev")
    args = parser.parse_args()
    asyncio.run(capture(args.host, args.token))


if __name__ == "__main__":
    main()
