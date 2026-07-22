from __future__ import annotations

import numpy as np

from ..motion_mode import MotionMode


class DemoMotionMode(MotionMode):
    """演示友好运动模式。

    该模式用手写规则放大并混合 OpenVLA 的 xyz/rpy，目的是让视频里机械臂主体运动更明显。
    它不是严格机器人学控制，适合演示和调试。
    """

    name = "demo"

    def to_ctrl(self, env, current_ctrl: np.ndarray, action: np.ndarray, distance: float | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
        target = current_ctrl.copy()
        arm_delta = self._demo_delta(action)
        target[:6] = target[:6] + arm_delta
        target[6] = self.gripper_ctrl(action[6], distance)
        return env.clamp_ctrl(target), arm_delta, {}

    def _demo_delta(self, action: np.ndarray) -> np.ndarray:
        dx, dy, dz, droll, dpitch, dyaw = action[:6]
        raw_delta = np.array(
            [
                2.4 * dy,
                -1.8 * dx + 2.2 * dz - self.config.reach_bias,
                1.4 * dx - 2.0 * dz + self.config.reach_bias,
                0.8 * droll,
                1.0 * dpitch,
                1.0 * dyaw,
            ],
            dtype=np.float64,
        )
        return np.clip(raw_delta * self.config.joint_gain, -self.config.max_joint_delta, self.config.max_joint_delta)
