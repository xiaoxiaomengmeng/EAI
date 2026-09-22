from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class RobotConfig:
    name: str
    robot_dir: Path
    actuator_names: list[str]
    home_ctrl: np.ndarray
    gripper_body_name: str
    finger_link_marker: str
    ee_site_pos: tuple[float, float, float]
    arm_joint_names: Optional[list[str]] = None
    gripper_open: float = 0.037
    gripper_closed: float = 0.015