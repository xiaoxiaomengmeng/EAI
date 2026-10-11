from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .config import ControlConfig
from .robots.base import RobotConfig


class MotionMode(ABC):
    name: str

    def __init__(self, config: ControlConfig, robot_config: RobotConfig):
        self.config = config
        self.robot_config = robot_config

    @abstractmethod
    def to_ctrl(self, env, current_ctrl: np.ndarray, action: np.ndarray, distance: float | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
        raise NotImplementedError

    def arm_delta_by_joint(self, arm_delta: np.ndarray) -> dict[str, float]:
        joint_names = self.robot_config.arm_joint_names[:6]
        return {name: float(value) for name, value in zip(joint_names, arm_delta)}

    def gripper_ctrl(self, gripper_action: float, distance: float | None = None, aligned: bool = True, force_closed: bool = False) -> float:
        gripper_open = self.robot_config.gripper_open
        gripper_closed = self.robot_config.gripper_closed
        if force_closed:
            return gripper_closed
        if distance is not None and distance > self.config.gripper_close_threshold:
            return gripper_open
        if not aligned:
            return gripper_open
        return gripper_open if gripper_action > 0.5 else gripper_closed


def build_motion_mode(config: ControlConfig, robot_config: RobotConfig) -> MotionMode:
    if config.motion_mode == "ik":
        from .motion_modes.ik import IKMotionMode

        return IKMotionMode(config, robot_config)
    if config.motion_mode == "demo":
        from .motion_modes.demo import DemoMotionMode

        return DemoMotionMode(config, robot_config)
    if config.motion_mode == "direct":
        from .motion_modes.direct import DirectMotionMode

        return DirectMotionMode(config, robot_config)
    raise ValueError(f"Unsupported motion_mode: {config.motion_mode}")