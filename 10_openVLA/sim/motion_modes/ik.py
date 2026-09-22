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

        # latch 后 runner 传 distance=0：即使爪未闭也激活 grip_down，settle 阶段边居中边下压
        grip_active = bool(is_gripper_closed or (distance is not None and distance <= 0.0))

        target, arm_delta, info = env.solve_ik_ctrl(
            action_xyz=action[:3],
            current_ctrl=current_ctrl,
            max_joint_delta=self.config.max_joint_delta,
            position_gain=self.config.ik_position_gain,
            max_position_delta=self.config.ik_max_position_delta,
            damping=self.config.ik_damping,
            reach_bias=self.config.reach_bias,
            grip_down_offset=self.config.grip_down_offset,
            gripper_closed=grip_active,
        )
        target[gripper_index] = self.gripper_ctrl(action[6], distance, self._aligned(env), is_gripper_closed)
        return env.clamp_ctrl(target), arm_delta, info

    def _aligned(self, env) -> bool:
        diag = env.diagnostics()
        ee_p = np.asarray(diag.get("ee_site_pos", (0.0, 0.0, 0.0)))
        cu_p = np.asarray(diag.get("cube_pos", (0.0, 0.0, 0.0)))
        horiz = float(np.linalg.norm(ee_p[:2] - cu_p[:2]))
        vert = float(abs(ee_p[2] - cu_p[2]))
        return horiz <= self.config.grasp_horizontal_tolerance and vert <= self.config.grasp_vertical_tolerance