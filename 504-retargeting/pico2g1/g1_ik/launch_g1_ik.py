#!/usr/bin/env python3
"""Lightweight device/replay to G1 + Wuji Hand MuJoCo visualization."""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

# 兄弟模块 g1_*.py 按名导入;首次导入 g1_config 时其模块级代码会把 REPO_ROOT
# (pico2g1/) 插入 sys.path,后续 sim_scenes 的导入依赖它,顺序不可调整。
sys.path.insert(0, str(Path(__file__).resolve().parent))
from g1_config import TUNING_DEFAULTS  # noqa: E402
from g1_ik import FullBodyIK  # noqa: E402
from g1_model import build_model  # noqa: E402
from g1_source import live_frames, load_tuning, replay_frames  # noqa: E402
from g1_window import TuningWindow  # noqa: E402
from sim_scenes import add_scene_arguments  # noqa: E402
from sim_scenes.live_stream import add_stream_arguments, stream_config_from_args  # noqa: E402

import mujoco  # noqa: E402
from PySide6 import QtWidgets  # noqa: E402

# 拆分前这些符号定义于本文件;re-export 保持 tests/test_scenes.py 等外部
# 按入口模块取用的路径不变。
__all__ = ["FullBodyIK", "TUNING_DEFAULTS", "build_model"]


def run(frames, tuning: dict[str, float], xixi_scene: bool = False,
        pause_event: threading.Event | None = None, *, scene: str | None = None,
        render_shadows: bool = False, stream_config=None, stream_enabled: bool = False) -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    model = build_model(xixi_scene, scene=scene)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    print(f"MuJoCo model ready: {model.nq} qpos (robot 76, environment {model.nq - 76})")
    window = TuningWindow(model, data, frames, tuning, pause_event, render_shadows=render_shadows,
                          stream_config=stream_config, stream_enabled=stream_enabled)
    window.show()
    app.exec()
    print(f"last IK residual: {window.last_error:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    replay_parser = modes.add_parser("replay")
    replay_parser.add_argument("input", type=Path)
    live_parser = modes.add_parser("live")
    live_parser.add_argument("host")
    live_parser.add_argument("--token", default="rynn_dev")
    for mode in (replay_parser, live_parser):
        add_scene_arguments(mode)
        add_stream_arguments(mode)
    args = parser.parse_args()
    tuning = load_tuning()
    pause_event = threading.Event() if args.mode == "replay" else None
    frames = (
        replay_frames(args.input, tuning, pause_event)
        if args.mode == "replay"
        else live_frames(args.host, args.token, tuning)
    )
    run(frames, tuning, args.xixi_scene, pause_event,
        scene=args.scene, render_shadows=args.render_shadows,
        stream_config=stream_config_from_args(args), stream_enabled=args.stream)


if __name__ == "__main__":
    main()
