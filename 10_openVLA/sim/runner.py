from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import imageio.v2 as imageio
from PIL import Image
import numpy as np

from .action_adapter import ActionAdapter
from .config import ControlConfig, PolicyConfig, RenderConfig, RolloutConfig, SceneConfig
from .mujoco_env import MuJoCoEnv
from .openvla_policy import DryRunPolicy, OpenVLAPolicy
from .robots.base import RobotConfig


_logger = logging.getLogger("sim")


def log(message: str):
    _logger.info(message)


class RolloutRunner:
    def __init__(
        self,
        robot_config: RobotConfig,
        scene: SceneConfig,
        render: RenderConfig,
        policy: PolicyConfig,
        control: ControlConfig,
        rollout: RolloutConfig,
    ):
        self.robot_config = robot_config
        self.scene_config = scene
        self.render_config = render
        self.policy_config = policy
        self.control_config = control
        self.rollout_config = rollout
        self.adapter = ActionAdapter(control, robot_config)

    def run(self):
        total_start = time.perf_counter()
        output_dir = Path(self.rollout_config.output_dir).expanduser().resolve()
        frames_dir = output_dir / "frames"
        policy_inputs_dir = output_dir / "policy_inputs"
        wrist_debug_dir = output_dir / "wrist_debug"
        output_dir.mkdir(parents=True, exist_ok=True)
        if self.rollout_config.save_frames:
            frames_dir.mkdir(parents=True, exist_ok=True)
        if self.rollout_config.save_policy_inputs:
            policy_inputs_dir.mkdir(parents=True, exist_ok=True)
        if self.rollout_config.save_wrist_debug:
            wrist_debug_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S") + f"_{time.time_ns() % 1_000_000_000:09d}"
        robot_name = self.robot_config.name
        log_file_path = output_dir / f"{robot_name}_openvla_{timestamp}.log"

        self._setup_logging(log_file_path)

        env_start = time.perf_counter()
        env = MuJoCoEnv(self.robot_config, self.scene_config, self.render_config)
        env_init_sec = time.perf_counter() - env_start

        policy_start = time.perf_counter()
        policy = self._load_policy()
        policy_load_sec = time.perf_counter() - policy_start
        policy_load_timing = dict(getattr(policy, "load_timing", {}))
        policy_load_timing["policy_load_sec"] = policy_load_sec

        records = []
        video_frames = []
        current_ctrl = env.current_ctrl()
        grasp_latched = False
        grasp_latched_step = None
        grasp_hold_count = 0
        grasp_lifting = False
        grasp_lift_count = 0
        stop_reason = None

        log(f"Starting rollout")
        log(f"  Robot: {robot_name}")
        log(f"  Instruction: '{self.policy_config.instruction}'")
        log(f"  Policy steps: {self.rollout_config.policy_steps}")
        log(f"  Sim steps per policy: {self.rollout_config.sim_steps_per_policy}")
        log(f"  Motion mode: {self.control_config.motion_mode}")
        log(f"  Joint gain: {self.control_config.joint_gain}")
        log(f"  Max joint delta: {self.control_config.max_joint_delta}")
        log(f"  Reach bias: {self.control_config.reach_bias}")
        log(f"  IK position gain: {self.control_config.ik_position_gain}")
        log(f"  IK max position delta: {self.control_config.ik_max_position_delta}")
        log(f"  IK damping: {self.control_config.ik_damping}")
        log(f"  Grasp assist: {self.control_config.grasp_assist}")
        if self.control_config.grasp_assist:
            log(f"  Grasp trigger distance: {self.control_config.grasp_trigger_distance}")
            log(f"  Grasp hold steps: {self.control_config.grasp_hold_steps}")
            log(f"  Early stop on grasp: {self.control_config.early_stop_on_grasp}")

        try:
            for policy_step in range(self.rollout_config.policy_steps):
                step_start = time.perf_counter()

                render_start = time.perf_counter()
                policy_frame = env.render_rgb(self.scene_config.policy_camera_name)
                render_sec = time.perf_counter() - render_start

                image_start = time.perf_counter()
                image = Image.fromarray(policy_frame).convert("RGB")
                frame_to_image_sec = time.perf_counter() - image_start

                save_frame_sec = 0.0
                policy_input_path = None
                wrist_debug_path = None
                if self.rollout_config.save_policy_inputs:
                    save_policy_input_start = time.perf_counter()
                    policy_input_path = policy_inputs_dir / f"policy_input_{policy_step:04d}_{self.scene_config.policy_camera_name}.png"
                    image.save(policy_input_path)
                    save_frame_sec += time.perf_counter() - save_policy_input_start
                if self.rollout_config.save_wrist_debug:
                    save_wrist_debug_start = time.perf_counter()
                    wrist_frame = env.render_rgb(self.scene_config.wrist_camera_name)
                    wrist_debug_path = wrist_debug_dir / f"wrist_debug_{policy_step:04d}_{self.scene_config.wrist_camera_name}.png"
                    Image.fromarray(wrist_frame).convert("RGB").save(wrist_debug_path)
                    save_frame_sec += time.perf_counter() - save_wrist_debug_start
                if self.rollout_config.save_frames:
                    save_frame_start = time.perf_counter()
                    debug_frame = env.render_rgb(self.scene_config.video_camera_name)
                    Image.fromarray(debug_frame).convert("RGB").save(frames_dir / f"frame_{policy_step:04d}_{self.scene_config.video_camera_name}.png")
                    save_frame_sec += time.perf_counter() - save_frame_start

                predict_start = time.perf_counter()
                if self.rollout_config.dry_run:
                    action = policy.predict(image, self.policy_config.instruction, step=policy_step)
                else:
                    action = policy.predict(image, self.policy_config.instruction)
                policy_predict_sec = time.perf_counter() - predict_start
                policy_internal_timing = dict(getattr(policy, "last_timing", {}))

                pre_ctrl_diagnostics = env.diagnostics()
                distance = pre_ctrl_diagnostics.get("gripper_cube_distance")
                ee_p = np.asarray(pre_ctrl_diagnostics.get("ee_site_pos", (0.0, 0.0, 0.0)))
                cu_p = np.asarray(pre_ctrl_diagnostics.get("cube_pos", (0.0, 0.0, 0.0)))
                horiz = float(np.linalg.norm(ee_p[:2] - cu_p[:2]))
                vert = float(abs(ee_p[2] - cu_p[2]))

                effective_action = action.copy()
                distance_for_ctrl = distance
                grasp_event = "none"
                if self.control_config.grasp_assist and distance is not None:
                    aligned = horiz <= self.control_config.grasp_horizontal_tolerance and vert <= self.control_config.grasp_vertical_tolerance
                    if not grasp_latched and aligned and distance <= self.control_config.grasp_trigger_distance:
                        grasp_latched = True
                        grasp_latched_step = policy_step
                        grasp_event = "latched"
                        log(f"Grasp assist latched at step={policy_step:03d}, dist={distance:.4f}, horiz={horiz:.4f}, vert={vert:.4f}")
                    if grasp_latched:
                        grasp_hold_count = policy_step - int(grasp_latched_step) + 1
                        distance_for_ctrl = 0.0
                        effective_action[:3] = 0.0
                        # settle 阶段先保持开爪，让 reach_bias 把 ee 拉到方块正上方再闭合，避免打飞方块
                        effective_action[6] = 0.0 if grasp_hold_count > self.control_config.grasp_settle_steps else 1.0
                        if grasp_event == "none":
                            grasp_event = "holding"
                        if grasp_hold_count >= self.control_config.grasp_hold_steps and self.control_config.grasp_lift_steps > 0:
                            grasp_lifting = True
                        if grasp_lifting:
                            grasp_lift_count += 1
                            effective_action[:3] = np.array([0.0, 0.0, self.control_config.grasp_lift_delta])
                            grasp_event = "lifting"

                motion_start = time.perf_counter()
                target_ctrl, arm_delta, ik_info = self.adapter.to_ctrl(env, current_ctrl, effective_action, distance_for_ctrl)
                motion_mode_sec = time.perf_counter() - motion_start

                mujoco_start = time.perf_counter()
                step_frames = env.step_ctrl(target_ctrl, self.rollout_config.sim_steps_per_policy, camera_name=self.scene_config.video_camera_name)
                mujoco_step_sec = time.perf_counter() - mujoco_start
                video_frames.extend(step_frames)
                current_ctrl = target_ctrl

                diagnostics_start = time.perf_counter()
                diagnostics = env.diagnostics()
                diagnostics_sec = time.perf_counter() - diagnostics_start

                timing = {
                    "render_sec": render_sec,
                    "frame_to_image_sec": frame_to_image_sec,
                    "save_frame_sec": save_frame_sec,
                    "policy_predict_sec": policy_predict_sec,
                    "motion_mode_sec": motion_mode_sec,
                    "mujoco_step_sec": mujoco_step_sec,
                    "mujoco_step_avg_sec": mujoco_step_sec / max(self.rollout_config.sim_steps_per_policy, 1),
                    "diagnostics_sec": diagnostics_sec,
                    "step_total_sec": time.perf_counter() - step_start,
                    **policy_internal_timing,
                }
                if ik_info.get("ik_timing"):
                    timing.update(ik_info["ik_timing"])

                gripper_ctrl_val = float(target_ctrl[-1])
                if robot_name == "franka":
                    gripper_open = 255.0
                    gripper_closed = 0.0
                else:
                    gripper_open = 0.037
                    gripper_closed = 0.015
                gripper_state = "open" if gripper_ctrl_val > (gripper_open + gripper_closed) / 2 else "closed"
                grasp_completed = grasp_latched and grasp_hold_count >= self.control_config.grasp_hold_steps
                lift_completed = grasp_lifting and grasp_lift_count >= self.control_config.grasp_lift_steps
                if self.control_config.early_stop_on_grasp:
                    if lift_completed:
                        stop_reason = f"grasp_lifted_after_{grasp_lift_count}_lift_steps"
                    elif grasp_completed and self.control_config.grasp_lift_steps <= 0:
                        stop_reason = f"grasp_completed_after_{grasp_hold_count}_hold_steps"

                record = {
                    "policy_step": policy_step,
                    "policy_camera": self.scene_config.policy_camera_name,
                    "video_camera": self.scene_config.video_camera_name,
                    "wrist_camera": self.scene_config.wrist_camera_name,
                    "policy_input_image": str(policy_input_path) if policy_input_path is not None else None,
                    "wrist_debug_image": str(wrist_debug_path) if wrist_debug_path is not None else None,
                    "action": action.tolist(),
                    "effective_action": effective_action.tolist(),
                    "action_xyz": action[:3].tolist(),
                    "effective_action_xyz": effective_action[:3].tolist(),
                    "action_rot": action[3:6].tolist(),
                    "effective_action_rot": effective_action[3:6].tolist(),
                    "gripper_action": float(action[6]),
                    "effective_gripper_action": float(effective_action[6]),
                    "grasp_assist": {
                        "enabled": self.control_config.grasp_assist,
                        "event": grasp_event,
                        "latched": grasp_latched,
                        "latched_step": grasp_latched_step,
                        "hold_count": grasp_hold_count,
                        "completed": grasp_completed,
                        "lifting": grasp_lifting,
                        "lift_count": grasp_lift_count,
                        "trigger_distance": self.control_config.grasp_trigger_distance,
                    },
                    "motion_mode": self.control_config.motion_mode,
                    "arm_delta": arm_delta.tolist(),
                    "arm_delta_by_joint": self.adapter.arm_delta_by_joint(arm_delta),
                    "ik_info": ik_info,
                    "target_ctrl": target_ctrl.tolist(),
                    "gripper_ctrl": gripper_ctrl_val,
                    "gripper_state": gripper_state,
                    "qpos": env.data.qpos.tolist(),
                    "qvel": env.data.qvel.tolist(),
                    "pre_ctrl_diagnostics": pre_ctrl_diagnostics,
                    "post_ctrl_diagnostics": diagnostics,
                    "timing": timing,
                    **diagnostics,
                }
                records.append(record)
                log(
                    f"step={policy_step:03d} "
                    f"action={np.array2string(action, precision=4)} "
                    f"effective_action={np.array2string(effective_action, precision=4)} "
                    f"arm_delta={np.array2string(arm_delta, precision=4)} "
                    f"ctrl={np.array2string(target_ctrl, precision=4)} "
                    f"dist={diagnostics['gripper_cube_distance']:.4f} "
                    f"grasp={grasp_event}/hold={grasp_hold_count} "
                    f"predict={policy_predict_sec:.4f}s motion={motion_mode_sec:.4f}s sim={mujoco_step_sec:.4f}s"
                )
                if stop_reason is not None:
                    log(f"Early stop: {stop_reason}, step={policy_step:03d}, dist={diagnostics['gripper_cube_distance']:.4f}")
                    break
        finally:
            env.close()

        timing_summary = self._summarize_timing(records)
        timing_summary.update(policy_load_timing)
        timing_summary["env_init_sec"] = env_init_sec
        timing_summary["pre_save_runtime_sec"] = time.perf_counter() - total_start
        video_path, json_log_path = self._save_outputs(output_dir, video_frames, records, timing_summary, total_start, timestamp, stop_reason)
        log(f"Timing summary: total={timing_summary['total_runtime_sec']:.3f}s, policy_load={policy_load_sec:.3f}s")
        log(f"Saved video: {video_path}")
        log(f"Saved JSON log: {json_log_path}")
        log(f"Saved text log: {log_file_path}")
        return video_path, json_log_path, log_file_path

    def _setup_logging(self, log_file_path: Path):
        robot_name = self.robot_config.name
        formatter = logging.Formatter(f"[%(asctime)s] [{robot_name.capitalize()} OpenVLA Sim] %(message)s")

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        _logger.setLevel(logging.INFO)
        _logger.handlers.clear()
        _logger.addHandler(stream_handler)

        try:
            file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            _logger.addHandler(file_handler)
        except (IOError, PermissionError):
            _logger.warning(f"Failed to create log file: {log_file_path}, using console-only logging.")

    def _load_policy(self):
        if self.rollout_config.dry_run:
            log("Dry run enabled: using deterministic test actions instead of OpenVLA.")
            return DryRunPolicy()
        return OpenVLAPolicy(
            model_path=self.policy_config.model_path,
            attn_implementation=self.policy_config.attn_implementation,
            unnorm_key=self.policy_config.unnorm_key,
        )

    @staticmethod
    def _summarize_timing(records: list[dict]) -> dict:
        timing_rows = [record.get("timing", {}) for record in records]
        if not timing_rows:
            return {}

        timing_keys = sorted({key for row in timing_rows for key in row})
        summary = {}
        for key in timing_keys:
            values = [float(row[key]) for row in timing_rows if key in row]
            if not values:
                continue
            summary[f"avg_{key}"] = sum(values) / len(values)
            summary[f"total_{key}"] = sum(values)
        return summary

    def _save_outputs(self, output_dir: Path, video_frames: list[np.ndarray], records: list[dict], timing: dict, total_start: float, timestamp: str, stop_reason: str | None):
        robot_name = self.robot_config.name
        video_path = output_dir / f"{robot_name}_openvla_{timestamp}.mp4"
        log_path = output_dir / f"{robot_name}_openvla_{timestamp}.json"

        save_outputs_start = time.perf_counter()
        save_video_start = time.perf_counter()
        if video_frames:
            imageio.mimsave(video_path, video_frames, fps=20)
        timing["save_video_sec"] = time.perf_counter() - save_video_start

        payload = {
            "validation_goal": "Franka custom MuJoCo scene instruction inference with openvla-7b; grasp success is not required.",
            "instruction": self.policy_config.instruction,
            "model_path": self.policy_config.model_path,
            "robot": robot_name,
            "unnorm_key": self.policy_config.unnorm_key,
            "attn_implementation": self.policy_config.attn_implementation,
            "policy_camera": self.scene_config.policy_camera_name,
            "video_camera": self.scene_config.video_camera_name,
            "wrist_camera": self.scene_config.wrist_camera_name,
            "stop_reason": stop_reason,
            "scene": self.scene_config.__dict__,
            "control": self.control_config.__dict__,
            "rollout": self.rollout_config.__dict__,
            "timing": timing,
            "records": records,
        }

        save_json_start = time.perf_counter()
        log_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        timing["save_json_sec"] = time.perf_counter() - save_json_start
        timing["save_outputs_sec"] = time.perf_counter() - save_outputs_start
        timing["total_runtime_sec"] = time.perf_counter() - total_start

        payload["timing"] = timing
        log_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return video_path, log_path