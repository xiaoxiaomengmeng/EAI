from .base import RobotConfig
from .franka import FRANKA_CONFIG
from .wx250s import WX250S_CONFIG

ROBOT_CONFIGS = {
    "franka": FRANKA_CONFIG,
    "wx250s": WX250S_CONFIG,
}

DEFAULT_ROBOT = "franka"