from __future__ import annotations

import numpy as np

from .config import ControlConfig
from .motion_mode import MotionMode, build_motion_mode
from .robots.base import RobotConfig


class ActionAdapter:
    def __init__(self, config: ControlConfig, robot_config: RobotConfig):
        self.config = config
        self.robot_config = robot_config
        self.motion_mode: MotionMode = build_motion_mode(config, robot_config)

    def to_ctrl(self, env, current_ctrl: np.ndarray, action: np.ndarray, distance: float | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
        return self.motion_mode.to_ctrl(env, current_ctrl, action, distance)

    def arm_delta_by_joint(self, arm_delta: np.ndarray) -> dict[str, float]:
        return self.motion_mode.arm_delta_by_joint(arm_delta)