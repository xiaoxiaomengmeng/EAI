from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

from .config import RenderConfig, SceneConfig
from .robots.base import RobotConfig


class MuJoCoEnv:
    def __init__(self, robot_config: RobotConfig, scene: SceneConfig, render: RenderConfig):
        self.robot_config = robot_config
        self.scene_config = scene
        self.render_config = render
        self.model, self.data = self._compile_scene()
        self.policy_camera_id = self._get_camera_id(scene.policy_camera_name)
        self.video_camera_id = self._get_camera_id(scene.video_camera_name)
        self.camera_id = self.policy_camera_id
        self.ee_site_id = self._site_id("ee_site")
        arm_joint_names = robot_config.arm_joint_names or robot_config.actuator_names[:6]
        self.arm_joint_ids = [self._joint_id(name) for name in arm_joint_names[:6]]
        self.arm_qpos_ids = np.array([self.model.jnt_qposadr[joint_id] for joint_id in self.arm_joint_ids], dtype=np.int32)
        self.arm_dof_ids = np.array([self.model.jnt_dofadr[joint_id] for joint_id in self.arm_joint_ids], dtype=np.int32)
        self.renderer = mujoco.Renderer(self.model, height=render.height, width=render.width)
        self.gripper_body_id = self._body_id(robot_config.gripper_body_name)
        self.cube_body_id = self._body_id("red_cube")
        self.reset()

    def close(self):
        self.renderer.close()

    def reset(self):
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if key_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        else:
            mujoco.mj_resetData(self.model, self.data)
        self.data.ctrl[:] = np.clip(
            self.robot_config.home_ctrl[: self.model.nu],
            self.model.actuator_ctrlrange[:, 0],
            self.model.actuator_ctrlrange[:, 1],
        )
        mujoco.mj_forward(self.model, self.data)

    def render_rgb(self, camera_name: str | None = None) -> np.ndarray:
        camera_id = self.camera_id if camera_name is None else self._get_camera_id(camera_name)
        self.renderer.update_scene(self.data, camera=camera_id)
        return np.asarray(self.renderer.render(), dtype=np.uint8)

    def step_ctrl(self, ctrl: np.ndarray, sim_steps: int, camera_name: str | None = None) -> list[np.ndarray]:
        self.data.ctrl[:] = ctrl
        frames = []
        for _ in range(sim_steps):
            mujoco.mj_step(self.model, self.data)
            frames.append(self.render_rgb(camera_name))
        return frames

    def current_ctrl(self) -> np.ndarray:
        return self.data.ctrl.copy()

    def clamp_ctrl(self, ctrl: np.ndarray) -> np.ndarray:
        low = self.model.actuator_ctrlrange[:, 0]
        high = self.model.actuator_ctrlrange[:, 1]
        return np.clip(ctrl, low, high)

    def diagnostics(self) -> dict:
        ee_pos = self.data.site_xpos[self.ee_site_id].copy()
        gripper_pos = self.data.xpos[self.gripper_body_id].copy()
        cube_pos = self.data.xpos[self.cube_body_id].copy()
        distance = float(np.linalg.norm(ee_pos - cube_pos))
        return {
            "ee_site_pos": ee_pos.tolist(),
            "gripper_pos": gripper_pos.tolist(),
            "cube_pos": cube_pos.tolist(),
            "gripper_cube_distance": distance,
        }

    def solve_ik_ctrl(
        self,
        action_xyz: np.ndarray,
        current_ctrl: np.ndarray,
        max_joint_delta: float,
        position_gain: float,
        max_position_delta: float,
        damping: float,
        reach_bias: float,
        grip_down_offset: float = 0.0,
        gripper_closed: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        ik_start = time.perf_counter()
        forward_start = time.perf_counter()
        mujoco.mj_forward(self.model, self.data)
        ik_forward_sec = time.perf_counter() - forward_start
        ee_pos = self.data.site_xpos[self.ee_site_id].copy()
        cube_pos = self.data.xpos[self.cube_body_id].copy()

        model_delta = np.asarray(action_xyz, dtype=np.float64) * position_gain
        model_delta = np.clip(model_delta, -max_position_delta, max_position_delta)
        to_cube = cube_pos - ee_pos
        to_cube_norm = float(np.linalg.norm(to_cube))
        cube_delta = np.zeros(3, dtype=np.float64)
        if to_cube_norm > 1e-8 and reach_bias > 0:
            cube_delta = to_cube / to_cube_norm * min(reach_bias, to_cube_norm)

        desired_delta = model_delta + cube_delta
        delta_norm = float(np.linalg.norm(desired_delta))
        if delta_norm > max_position_delta:
            desired_delta = desired_delta / delta_norm * max_position_delta
        target_pos = ee_pos + desired_delta

        if gripper_closed and grip_down_offset > 0:
            target_pos[2] -= grip_down_offset

        error = target_pos - ee_pos

        jacobian_start = time.perf_counter()
        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site_id)
        arm_jacobian = jacp[:, self.arm_dof_ids]
        ik_jacobian_sec = time.perf_counter() - jacobian_start

        solve_start = time.perf_counter()
        lhs = arm_jacobian @ arm_jacobian.T + (damping**2) * np.eye(3)
        joint_delta = arm_jacobian.T @ np.linalg.solve(lhs, error)
        joint_delta = np.clip(joint_delta, -max_joint_delta, max_joint_delta)
        ik_linear_solve_sec = time.perf_counter() - solve_start

        target_ctrl = current_ctrl.copy()
        current_qpos = self.data.qpos[self.arm_qpos_ids].copy()
        target_ctrl[:6] = current_qpos + joint_delta
        target_ctrl = self.clamp_ctrl(target_ctrl)
        info = {
            "ee_pos": ee_pos.tolist(),
            "target_pos": target_pos.tolist(),
            "model_delta": model_delta.tolist(),
            "cube_delta": cube_delta.tolist(),
            "ik_error": error.tolist(),
            "ik_timing": {
                "ik_solve_sec": time.perf_counter() - ik_start,
                "ik_forward_sec": ik_forward_sec,
                "ik_jacobian_sec": ik_jacobian_sec,
                "ik_linear_solve_sec": ik_linear_solve_sec,
            },
        }
        return target_ctrl, joint_delta, info

    def _compile_scene(self):
        robot_dir = self.robot_config.robot_dir
        robot_xml_name = "panda.xml" if self.robot_config.name == "franka" else f"{self.robot_config.name}.xml"
        if not (robot_dir / robot_xml_name).is_file():
            raise FileNotFoundError(f"{robot_xml_name} not found in {robot_dir}")

        scene_xml = self._make_scene_xml()
        with tempfile.NamedTemporaryFile("w", suffix=".xml", prefix=f"openvla_{self.robot_config.name}_", dir=robot_dir, delete=False) as scene_file:
            scene_file.write(scene_xml)
            scene_path = Path(scene_file.name)

        try:
            model = mujoco.MjModel.from_xml_path(str(scene_path))
        finally:
            scene_path.unlink(missing_ok=True)
        data = mujoco.MjData(model)
        return model, data

    def _get_camera_id(self, camera_name: str) -> int:
        camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if camera_id < 0:
            raise ValueError(f"Camera {camera_name!r} not found in MuJoCo model.")
        return camera_id

    def _body_id(self, body_name: str) -> int:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"Body {body_name!r} not found in MuJoCo model.")
        return body_id

    def _joint_id(self, joint_name: str) -> int:
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"Joint {joint_name!r} not found in MuJoCo model.")
        return joint_id

    def _site_id(self, site_name: str) -> int:
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id < 0:
            raise ValueError(f"Site {site_name!r} not found in MuJoCo model.")
        return site_id

    def _make_robot_xml_with_ee_site(self) -> str:
        robot_dir = self.robot_config.robot_dir
        robot_xml_name = f"{self.robot_config.name}.xml"
        if self.robot_config.name == "franka":
            robot_xml_name = "panda.xml"
        robot_xml = (robot_dir / robot_xml_name).read_text(encoding="utf-8")
        if 'name="ee_site"' in robot_xml:
            return robot_xml
        marker = self.robot_config.finger_link_marker
        if marker not in robot_xml:
            raise ValueError(f"Could not find insertion point for ee_site in {robot_xml_name}")
        ee_x, ee_y, ee_z = self.robot_config.ee_site_pos
        site_xml = f'                      <site name="ee_site" pos="{ee_x} {ee_y} {ee_z}" size="0.008" rgba="0 1 0 1"/>\n'
        return robot_xml.replace(marker, site_xml + marker, 1)

    def _make_scene_xml(self) -> str:
        robot_xml = self._make_robot_xml_with_ee_site()
        robot_xml = self._inject_scene_assets(robot_xml)
        return self._inject_scene_worldbody(robot_xml)

    def _inject_scene_assets(self, robot_xml: str) -> str:
        scene_asset_xml = """
    <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1="0.3 0.3 0.3" rgb2="0.4 0.4 0.4"/>
    <material name="grid" texture="grid" texrepeat="1 1" reflectance="0.2"/>"""
        if "<asset>" not in robot_xml:
            raise ValueError("Robot XML does not contain an <asset> section for scene assets.")
        return robot_xml.replace("<asset>", "<asset>" + scene_asset_xml, 1)

    def _inject_scene_worldbody(self, robot_xml: str) -> str:
        scene = self.scene_config
        scene_worldbody_xml = f"""
    <camera name="agentview" pos="0.75 -0.65 0.55" xyaxes="0.65 0.75 0 -0.35 0.30 0.89"/>
    <camera name="overview" pos="0.75 0.75 0.70" xyaxes="-0.71 0.71 0 -0.40 -0.40 0.82"/>
    <camera name="wristview" pos="0.55 -0.50 0.38" xyaxes="0.65 0.75 0 -0.35 0.30 0.89"/>
    <light name="ambient_light" pos="0 0 3" dir="0 0 -1" diffuse="0.6 0.6 0.6" specular="0.2 0.2 0.2" ambient="0.4 0.4 0.4"/>
    <light name="point_light_1" pos="0.5 0.5 1.0" dir="-0.5 -0.5 -1" diffuse="0.8 0.8 0.8" specular="0.5 0.5 0.5"/>
    <light name="point_light_2" pos="-0.5 0.5 1.0" dir="0.5 -0.5 -1" diffuse="0.8 0.8 0.8" specular="0.5 0.5 0.5"/>
    <geom name="floor" type="plane" size="1.5 1.5 0.02" rgba="0.7 0.7 0.7 1" material="grid"/>
    <body name="table" pos="{scene.table_x} {scene.table_y} 0">
      <geom name="table_top" type="box" size="0.35 0.35 0.02" rgba="0.5 0.4 0.3 1"/>
      <geom name="table_leg_1" type="box" size="0.02 0.02 0.48" pos="-0.33 0.33 -0.5" rgba="0.4 0.3 0.2 1"/>
      <geom name="table_leg_2" type="box" size="0.02 0.02 0.48" pos="-0.33 -0.33 -0.5" rgba="0.4 0.3 0.2 1"/>
      <geom name="table_leg_3" type="box" size="0.02 0.02 0.48" pos="0.33 0.33 -0.5" rgba="0.4 0.3 0.2 1"/>
      <geom name="table_leg_4" type="box" size="0.02 0.02 0.48" pos="0.33 -0.33 -0.5" rgba="0.4 0.3 0.2 1"/>
    </body>
    <body name="red_cube" pos="{scene.cube_x} {scene.cube_y} {scene.cube_z}">
      <geom name="cube_geom" type="box" size="{scene.cube_size} {scene.cube_size} {scene.cube_size}" rgba="1.0 0.0 0.0 1.0" mass="0.1"/>
    </body>"""
        if "<worldbody>" not in robot_xml:
            raise ValueError("Robot XML does not contain a <worldbody> section for scene bodies.")
        return robot_xml.replace("<worldbody>", "<worldbody>" + scene_worldbody_xml, 1)