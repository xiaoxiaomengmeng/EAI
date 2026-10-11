from __future__ import annotations

import numpy as np
from pathlib import Path

from .base import RobotConfig

FRANKA_CONFIG = RobotConfig(
    name="franka",
    robot_dir=Path("/home/creator/ws/assets/mujoco/mujoco_menagerie/franka_emika_panda"),
    actuator_names=[
        "actuator1",
        "actuator2",
        "actuator3",
        "actuator4",
        "actuator5",
        "actuator6",
        "actuator7",
        "actuator8",
    ],
    arm_joint_names=[
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7",
    ],
    home_ctrl=np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853, 255.0], dtype=np.float64),
    gripper_body_name="hand",
    finger_link_marker='                      <body name="left_finger" pos="0 0 0.0584">',
    ee_site_pos=(0.0, 0.0, 0.108),
    gripper_open=255.0,
    gripper_closed=0.0,
)