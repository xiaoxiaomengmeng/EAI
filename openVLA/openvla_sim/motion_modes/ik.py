from __future__ import annotations

import numpy as np

from ..motion_mode import MotionMode


class IKMotionMode(MotionMode):
    name = "ik"

    def to_ctrl(self, env, current_ctrl: np.ndarray, action: np.ndarray, distance: float | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
        gripper_open = self.robot_config.gripper_open
        gripper_closed = self.robot_config.gripper_closed
        gripper_index = len(self.robot_config.actuator_names) - 1
        is_gripper_closed = current_ctrl[gripper_index] < (gripper_open + gripper_closed) / 2

        target, arm_delta, info = env.solve_ik_ctrl(
            action_xyz=action[:3],
            current_ctrl=current_ctrl,
            max_joint_delta=self.config.max_joint_delta,
            position_gain=self.config.ik_position_gain,
            max_position_delta=self.config.ik_max_position_delta,
            damping=self.config.ik_damping,
            reach_bias=self.config.reach_bias,
            grip_down_offset=self.config.grip_down_offset,
            gripper_closed=is_gripper_closed,
        )
        target[gripper_index] = self.gripper_ctrl(action[6], distance)
        return env.clamp_ctrl(target), arm_delta, info
on[6], distance)
        return env.clamp_ctrl(target), arm_delta, info