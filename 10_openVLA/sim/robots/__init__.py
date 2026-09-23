from .base import RobotConfig
from .franka import FRANKA_CONFIG

ROBOT_CONFIGS = {
    "franka": FRANKA_CONFIG,
}

DEFAULT_ROBOT = "franka"