from __future__ import annotations

import numpy as np
from pathlib import Path

from .base import RobotConfig

WX250S_CONFIG = RobotConfig(
    name="wx250s",
    robot_dir=Path("/home/creator/ws/assets/mujoco/mujoco_menagerie/trossen_wx250s"),
    actuator_names=[
        "waist",
        "shoulder",
        "elbow",
        "forearm_roll",
        "wrist_angle",
        "wrist_rotate",
        "gripper",
    ],
    arm_joint_names=[
        "waist",
        "shoulder",
        "elbow",
        "forearm_roll",
        "wrist_angle",
        "wrist_rotate",
    ],
    home_ctrl=np.array([0.0, -0.96, 1.16, 0.0, -0.3, 0.0, 0.025], dtype=np.float64),
    gripper_body_name="wx250s/gripper_link",
    finger_link_marker='                  <body name="wx250s/left_finger_link" pos="0.066 0 0">',
    ee_site_pos=(0.108, 0.0, 0.0),
    gripper_open=0.037,
    gripper_closed=0.015,
)