from __future__ import annotations

import numpy as np

from ..motion_mode import MotionMode


class DirectMotionMode(MotionMode):
    """直接关节增量运动模式。

    该模式将 OpenVLA action[:6] 直接作为 6 个机械臂关节的增量，主要用于 baseline 对照。
    """

    name = "direct"

    def to_ctrl(self, env, current_ctrl: np.ndarray, action: np.ndarray, distance: float | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
        target = current_ctrl.copy()
        arm_delta = np.clip(
            action[:6] * self.config.joint_gain,
            -self.config.max_joint_delta,
            self.config.max_joint_delta,
        )
        target[:6] = target[:6] + arm_delta
        target[6] = self.gripper_ctrl(action[6], distance)
        return env.clamp_ctrl(target), arm_delta, {}
