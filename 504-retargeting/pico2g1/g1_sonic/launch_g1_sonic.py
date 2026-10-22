#!/usr/bin/env python3
"""Run SONIC v1.1 ONNX with recorded or live PICO/glove data."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import threading

import numpy as np
from PySide6 import QtWidgets

# Re-exports below: tests/test_scenes.py loads this entry by file path and
# takes build_model / Simulator / load_skeleton / SMPL_OUTPUT_JOINTS from it.
from sonic_config import SMPL_OUTPUT_JOINTS  # noqa: F401
from sonic_model import build_model  # noqa: F401
from sonic_policy import Simulator  # noqa: F401
from sonic_recording import load_rotation_offsets, load_skeleton  # noqa: F401
from sonic_source import live_frames, replay_frames
from sonic_window import Window

from sim_scenes import add_scene_arguments
from sim_scenes.live_stream import add_stream_arguments, stream_config_from_args


def run(frames, xixi_scene: bool = False, *, scene: str | None = None,
        render_shadows: bool = False,
        rotation_offsets: dict[str, np.ndarray] | None = None,
        pause_event: threading.Event | None = None, stream_config=None,
        stream_enabled: bool = False) -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    window = Window(frames, xixi_scene, scene=scene, render_shadows=render_shadows,
                    rotation_offsets=rotation_offsets, pause_event=pause_event,
                    stream_config=stream_config, stream_enabled=stream_enabled)
    print(f"MuJoCo ready: {window.sim.model.nq} qpos (robot 76, environment {window.sim.model.nq - 76})")
    window.show()
    app.exec()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    replay_parser = modes.add_parser("replay", help="play a recorded session")
    replay_parser.add_argument("input", type=Path, help="session directory or full-body parquet")
    live_parser = modes.add_parser("live", help="subscribe to the device in real time")
    live_parser.add_argument("host", help="device IP address")
    live_parser.add_argument("--token", default="rynn_dev")
    for mode in (replay_parser, live_parser):
        add_scene_arguments(mode)
        add_stream_arguments(mode)
    args = parser.parse_args()
    rotation_offsets = load_rotation_offsets()
    pause_event = threading.Event() if args.mode == "replay" else None
    frames = (
        replay_frames(args.input, rotation_offsets, pause_event)
        if args.mode == "replay"
        else live_frames(args.host, args.token, rotation_offsets)
    )
    run(frames, args.xixi_scene, scene=args.scene, render_shadows=args.render_shadows,
        rotation_offsets=rotation_offsets, pause_event=pause_event,
        stream_config=stream_config_from_args(args), stream_enabled=args.stream)


if __name__ == "__main__":
    main()
