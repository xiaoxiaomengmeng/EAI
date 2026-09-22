from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# 由本文件位置推导项目根（sim/config.py 上两级），避免硬编码绝对路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = "/home/creator/.cache/modelscope/models/zixiaoBios--openvla-7b/snapshots/master"


@dataclass(frozen=True)
class SceneConfig:
    cube_x: float = 0.42
    cube_y: float = -0.16
    cube_z: float = 0.045
    cube_size: float = 0.025
    table_x: float = 0.30
    table_y: float = 0.0
    table_z: float = 0.015
    policy_camera_name: str = "agentview"
    video_camera_name: str = "overview"
    wrist_camera_name: str = "wristview"
    camera_name: str = "agentview"


@dataclass(frozen=True)
class RenderConfig:
    width: int = 640
    height: int = 480


@dataclass(frozen=True)
class PolicyConfig:
    model_path: str = DEFAULT_MODEL_PATH
    instruction: str = "pick up the red cube"
    unnorm_key: str = "bridge_orig"
    attn_implementation: str = "sdpa"


@dataclass(frozen=True)
class ControlConfig:
    motion_mode: str = "ik"
    max_joint_delta: float = 0.08
    joint_gain: float = 3.0
    reach_bias: float = 0.03
    ik_position_gain: float = 1.0
    ik_max_position_delta: float = 0.05
    ik_damping: float = 0.05
    gripper_close_threshold: float = 0.05
    grip_down_offset: float = 0.025
    grasp_assist: bool = True
    grasp_trigger_distance: float = 0.055
    grasp_hold_steps: int = 8
    grasp_horizontal_tolerance: float = 0.02
    grasp_vertical_tolerance: float = 0.035
    grasp_settle_steps: int = 4
    grasp_lift_steps: int = 8
    grasp_lift_delta: float = 0.045
    early_stop_on_grasp: bool = True


@dataclass(frozen=True)
class RolloutConfig:
    policy_steps: int = 20
    sim_steps_per_policy: int = 30
    output_dir: str = str(PROJECT_ROOT / "outputs" / "franka_openvla")
    save_frames: bool = False
    save_policy_inputs: bool = True
    save_wrist_debug: bool = False
    dry_run: bool = False
