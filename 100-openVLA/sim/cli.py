from __future__ import annotations

import argparse

from .config import (
    DEFAULT_MODEL_PATH,
    ControlConfig,
    PolicyConfig,
    RenderConfig,
    RolloutConfig,
    SceneConfig,
)
from .runner import RolloutRunner
from .robots import DEFAULT_ROBOT, ROBOT_CONFIGS


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run OpenVLA base + MuJoCo closed-loop simulation."
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Local OpenVLA checkpoint path or HF repo id.")
    parser.add_argument("--robot", default=DEFAULT_ROBOT, choices=list(ROBOT_CONFIGS.keys()), help="Robot type (Franka Panda only).")
    parser.add_argument("--output-dir", default=None, help="Directory for rollout video, frames, and logs.")
    parser.add_argument("--instruction", default="pick up the red cube", help="Language command.")
    parser.add_argument("--unnorm-key", default="bridge_orig", help="OpenVLA action un-normalization key.")
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        choices=["eager", "sdpa", "flash_attention_2"],
        help="Attention backend. sdpa is the safe default; flash_attention_2 is faster if available.",
    )
    parser.add_argument("--policy-camera", default="agentview", help="MuJoCo camera used as OpenVLA policy input.")
    parser.add_argument("--video-camera", default="overview", help="MuJoCo camera used for rollout video/debug frames.")
    parser.add_argument("--wrist-camera", default="wristview", help="Optional wrist/near-gripper debug camera; not used as policy input.")
    parser.add_argument("--camera", default=None, help="Backward-compatible alias: sets both policy and video camera.")
    parser.add_argument("--width", type=int, default=640, help="Render width.")
    parser.add_argument("--height", type=int, default=480, help="Render height.")
    parser.add_argument("--policy-steps", type=int, default=50, help="Number of OpenVLA policy queries.")
    parser.add_argument("--sim-steps-per-policy", type=int, default=30, help="MuJoCo steps after each policy action.")
    parser.add_argument("--max-joint-delta", type=float, default=0.08, help="Clamp each arm joint delta per policy step.")
    parser.add_argument("--joint-gain", type=float, default=3.0, help="Scale OpenVLA-derived arm deltas before applying to joints.")
    parser.add_argument("--reach-bias", type=float, default=0.02, help="Small demo-mode bias toward the front tabletop target.")
    parser.add_argument(
        "--motion-mode",
        default="ik",
        choices=["ik", "demo", "direct"],
        help=(
            "Motion mode. ik: use ee_site Jacobian IK; demo: amplified/remixed joint motion; "
            "direct: action[:6] maps directly to six arm joints."
        ),
    )
    parser.add_argument("--ik-position-gain", type=float, default=1.0, help="Scale OpenVLA xyz before IK target computation.")
    parser.add_argument("--ik-max-position-delta", type=float, default=0.05, help="Clamp IK target position delta per policy step.")
    parser.add_argument("--ik-damping", type=float, default=0.05, help="Damping factor for damped least-squares IK.")
    parser.add_argument("--gripper-close-threshold", type=float, default=0.05, help="Distance threshold (m) below which gripper is allowed to close.")
    parser.add_argument("--grip-down-offset", type=float, default=0.015, help="Vertical offset (m) for gripper to press down after closing.")
    parser.add_argument("--grasp-assist", action=argparse.BooleanOptionalAction, default=True, help="Latch close-gripper control near the cube to make grasp completion visible.")
    parser.add_argument("--grasp-trigger-distance", type=float, default=0.055, help="Distance (m) below which grasp assist closes and holds the gripper.")
    parser.add_argument("--grasp-hold-steps", type=int, default=5, help="Number of policy steps to hold a latched grasp before optional early stop.")
    parser.add_argument("--early-stop-on-grasp", action=argparse.BooleanOptionalAction, default=True, help="Stop rollout after grasp assist has held the object for enough steps.")
    parser.add_argument("--cube-x", type=float, default=0.42, help="Red cube x position in world coordinates.")
    parser.add_argument("--cube-y", type=float, default=-0.16, help="Red cube y position in world coordinates.")
    parser.add_argument("--cube-z", type=float, default=0.045, help="Red cube z position in world coordinates.")
    parser.add_argument("--cube-size", type=float, default=0.025, help="Red cube half-size.")
    parser.add_argument("--dry-run", action="store_true", help="Skip OpenVLA loading and use deterministic test actions.")
    parser.add_argument("--save-frames", action="store_true", help="Also save individual debug/video RGB frames.")
    parser.add_argument("--save-policy-inputs", action=argparse.BooleanOptionalAction, default=True, help="Save OpenVLA policy input images.")
    parser.add_argument("--save-wrist-debug", action=argparse.BooleanOptionalAction, default=False, help="Save optional wrist/near-gripper debug camera images; not used for policy inference.")
    return parser.parse_args()


def build_runner(args) -> RolloutRunner:
    robot_config = ROBOT_CONFIGS[args.robot]
    
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = f"/home/creator/ws/100-openVLA/outputs/{robot_config.name}_openvla"
    
    policy_camera = args.camera or args.policy_camera
    video_camera = args.camera or args.video_camera

    scene = SceneConfig(
        cube_x=args.cube_x,
        cube_y=args.cube_y,
        cube_z=args.cube_z,
        cube_size=args.cube_size,
        policy_camera_name=policy_camera,
        video_camera_name=video_camera,
        wrist_camera_name=args.wrist_camera,
        camera_name=policy_camera,
    )
    render = RenderConfig(width=args.width, height=args.height)
    policy = PolicyConfig(
        model_path=args.model_path,
        instruction=args.instruction,
        unnorm_key=args.unnorm_key,
        attn_implementation=args.attn_implementation,
    )
    control = ControlConfig(
        motion_mode=args.motion_mode,
        max_joint_delta=args.max_joint_delta,
        joint_gain=args.joint_gain,
        reach_bias=args.reach_bias,
        ik_position_gain=args.ik_position_gain,
        ik_max_position_delta=args.ik_max_position_delta,
        ik_damping=args.ik_damping,
        gripper_close_threshold=args.gripper_close_threshold,
        grip_down_offset=args.grip_down_offset,
        grasp_assist=args.grasp_assist,
        grasp_trigger_distance=args.grasp_trigger_distance,
        grasp_hold_steps=args.grasp_hold_steps,
        early_stop_on_grasp=args.early_stop_on_grasp,
    )
    rollout = RolloutConfig(
        policy_steps=args.policy_steps,
        sim_steps_per_policy=args.sim_steps_per_policy,
        output_dir=output_dir,
        save_frames=args.save_frames,
        save_policy_inputs=args.save_policy_inputs,
        save_wrist_debug=args.save_wrist_debug,
        dry_run=args.dry_run,
    )
    return RolloutRunner(
        robot_config=robot_config,
        scene=scene,
        render=render,
        policy=policy,
        control=control,
        rollout=rollout,
    )


def main():
    args = parse_args()
    runner = build_runner(args)
    runner.run()


if __name__ == "__main__":
    main()
