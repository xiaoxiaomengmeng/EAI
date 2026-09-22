from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import mink
import numpy as np

from .config import RenderConfig, SceneConfig
from .robots.base import RobotConfig

IK_DT = 0.1
IK_MAX_ITERS = 40
IK_POS_THRESHOLD = 1e-3
IK_SOLVER = "daqp"


class MuJoCoEnv:
    def __init__(self, robot_config: RobotConfig, scene: SceneConfig, render: RenderConfig):
        self.robot_config = robot_config
        self.scene_config = scene
        self.render_config = render
        self.model, self.data = self._compile_scene()
        self.policy_camera_id = self._get_camera_id(scene.policy_camera_name)
        self.video_camera_id = self._get_camera_id(scene.video_camera_name)
        self.wrist_camera_id = self._get_camera_id(scene.wrist_camera_name)
        self.camera_id = self.policy_camera_id
        self.ee_site_id = self._site_id("ee_site")
        arm_joint_names = robot_config.arm_joint_names or robot_config.actuator_names[:6]
        self.arm_joint_ids = [self._joint_id(name) for name in arm_joint_names[:6]]
        self.arm_qpos_ids = np.array([self.model.jnt_qposadr[joint_id] for joint_id in self.arm_joint_ids], dtype=np.int32)
        self.arm_dof_ids = np.array([self.model.jnt_dofadr[joint_id] for joint_id in self.arm_joint_ids], dtype=np.int32)
        self.renderer = mujoco.Renderer(self.model, height=render.height, width=render.width)
        self.gripper_body_id = self._body_id(robot_config.gripper_body_name)
        self.cube_body_id = self._body_id("red_cube")
        arm_joint_names7 = robot_config.arm_joint_names or robot_config.actuator_names[:7]
        self.arm_joint_ids7 = [self._joint_id(name) for name in arm_joint_names7]
        self.arm_qpos_ids7 = np.array([self.model.jnt_qposadr[j] for j in self.arm_joint_ids7], dtype=np.int32)
        self.ik_model = self._compile_robot_model()
        # mink 默认遵守关节限位会挡住折叠下探；IK 模型放开限位，执行端仍由 clamp_ctrl 保护
        self.ik_model.jnt_limited[:] = 0
        self.ik_config = mink.Configuration(self.ik_model)
        self.ik_frame_task = mink.FrameTask("ee_site", "site", position_cost=1.0, orientation_cost=0.0, lm_damping=1.0)
        self.ik_posture_task = mink.PostureTask(self.ik_model, cost=1e-2)
        self.ik_arm_qpos_ids = np.array(
            [self.ik_model.jnt_qposadr[mujoco.mj_name2id(self.ik_model, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in arm_joint_names7],
            dtype=np.int32,
        )
        ik_joint_names = [mujoco.mj_id2name(self.ik_model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(self.ik_model.njnt)]
        finger_names = ik_joint_names[len(arm_joint_names7):]
        self.finger_qpos_ids = np.array([self.model.jnt_qposadr[self._joint_id(n)] for n in finger_names], dtype=np.int32)
        self.reset()
        self._setup_policy_camera()
        self._setup_video_camera()

    def close(self):
        self.renderer.close()

    def reset(self):
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if key_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        else:
            mujoco.mj_resetData(self.model, self.data)
        # 场景注入 cube freejoint 后 robot keyframe qpos 会错位，显式回填 cube 与机械臂 home 位
        cube_jnt = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_freejoint")
        if cube_jnt >= 0:
            adr = int(self.model.jnt_qposadr[cube_jnt])
            self.data.qpos[adr : adr + 3] = [self.scene_config.cube_x, self.scene_config.cube_y, self.scene_config.cube_z]
            self.data.qpos[adr + 3 : adr + 7] = [1.0, 0.0, 0.0, 0.0]
        ik_key = mujoco.mj_name2id(self.ik_model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if ik_key >= 0:
            home_q = self.ik_model.key_qpos[ik_key]
            self.data.qpos[self.arm_qpos_ids7] = home_q[: len(self.arm_qpos_ids7)]
            if len(self.finger_qpos_ids) and len(home_q) > len(self.arm_qpos_ids7):
                self.data.qpos[self.finger_qpos_ids] = home_q[len(self.arm_qpos_ids7) : len(self.arm_qpos_ids7) + len(self.finger_qpos_ids)]
        self.data.ctrl[:] = np.clip(
            self.robot_config.home_ctrl[: self.model.nu],
            self.model.actuator_ctrlrange[:, 0],
            self.model.actuator_ctrlrange[:, 1],
        )
        mujoco.mj_forward(self.model, self.data)

    def render_rgb(self, camera_name: str | None = None) -> np.ndarray:
        camera_id = self.camera_id if camera_name is None else self._get_camera_id(camera_name)
        if camera_id == self.wrist_camera_id:
            self._update_wrist_camera()
        self.renderer.update_scene(self.data, camera=camera_id)
        return np.asarray(self.renderer.render(), dtype=np.uint8)

    def step_ctrl(self, ctrl: np.ndarray, sim_steps: int, camera_name: str | None = None) -> list[np.ndarray]:
        self.data.ctrl[:] = ctrl
        frames = []
        for _ in range(sim_steps):
            mujoco.mj_step(self.model, self.data)
            frames.append(self.render_rgb(camera_name))
        return frames

    def _update_wrist_camera(self) -> None:
        # 腕部相机动态跟随夹爪：置于夹爪后上方，沿 ee_site 前进方向 look-at，形成第一人称腕部视角
        gripper_pos = self.data.xpos[self.gripper_body_id].copy()
        ee_pos = self.data.site_xpos[self.ee_site_id].copy()
        forward = ee_pos - gripper_pos
        norm = float(np.linalg.norm(forward))
        forward = forward / norm if norm > 1e-8 else np.array([0.0, 0.0, -1.0])
        cam_pos = gripper_pos + np.array([-0.10, -0.12, 0.10])
        target = ee_pos + forward * 0.20
        self.model.cam_pos[self.wrist_camera_id] = cam_pos
        self.model.cam_quat[self.wrist_camera_id] = self._lookat_quat(cam_pos, target)
        mujoco.mj_forward(self.model, self.data)

    def _setup_policy_camera(self) -> None:
        # 策略输入相机一次性对准工作区（ee 与红方块中点），保证夹爪+方块居中清晰
        ee_pos = self.data.site_xpos[self.ee_site_id].copy()
        cube_pos = self.data.xpos[self.cube_body_id].copy()
        target = 0.5 * (ee_pos + cube_pos)
        cam_pos = target + np.array([0.65, -1.05, 0.70])
        self.model.cam_pos[self.policy_camera_id] = cam_pos
        self.model.cam_quat[self.policy_camera_id] = self._lookat_quat(cam_pos, target)
        mujoco.mj_forward(self.model, self.data)

    def _setup_video_camera(self) -> None:
        # 视频/调试相机一次性对准全景（base+ee+cube），保证完整机械臂运动趋势可见
        ee_pos = self.data.site_xpos[self.ee_site_id].copy()
        cube_pos = self.data.xpos[self.cube_body_id].copy()
        target = 0.5 * (ee_pos + cube_pos)
        cam_pos = target + np.array([1.0, -1.4, 1.1])
        self.model.cam_pos[self.video_camera_id] = cam_pos
        self.model.cam_quat[self.video_camera_id] = self._lookat_quat(cam_pos, target)
        mujoco.mj_forward(self.model, self.data)

    @staticmethod
    def _lookat_quat(pos: np.ndarray, target: np.ndarray, up: np.ndarray | None = None) -> np.ndarray:
        up = np.array([0.0, 0.0, 1.0]) if up is None else np.asarray(up, dtype=np.float64)
        pos = np.asarray(pos, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        forward = target - pos
        norm = float(np.linalg.norm(forward))
        forward = forward / norm if norm > 1e-8 else np.array([0.0, 0.0, -1.0])
        z_axis = -forward
        x_axis = np.cross(up, z_axis)
        x_norm = float(np.linalg.norm(x_axis))
        x_axis = x_axis / x_norm if x_norm > 1e-8 else np.array([1.0, 0.0, 0.0])
        y_axis = np.cross(z_axis, x_axis)
        rot = np.stack([x_axis, y_axis, z_axis], axis=1)
        quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quat, rot.reshape(-1))
        return quat

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
        mujoco.mj_forward(self.model, self.data)
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

        # 将场景臂关节同步到 robot-only mink 模型，保持姿态并求解位置+姿态 QP IK
        q_ik = self.ik_config.q.copy()
        q_ik[self.ik_arm_qpos_ids] = self.data.qpos[self.arm_qpos_ids7]
        self.ik_config.update(q_ik)
        self.ik_posture_task.set_target(self.ik_config.q)
        current_rotation = self.ik_config.get_transform_frame_to_world("ee_site", "site").rotation()
        self.ik_frame_task.set_target(mink.SE3.from_rotation_and_translation(current_rotation, target_pos))

        solve_start = time.perf_counter()
        converged = False
        iters = 0
        for iters in range(1, IK_MAX_ITERS + 1):
            vel = mink.solve_ik(self.ik_config, [self.ik_frame_task, self.ik_posture_task], IK_DT, IK_SOLVER, damping=1e-3)
            self.ik_config.integrate_inplace(vel, IK_DT)
            error = self.ik_frame_task.compute_error(self.ik_config)
            if float(np.linalg.norm(error[:3])) <= IK_POS_THRESHOLD:
                converged = True
                break
        ik_qp_solve_sec = time.perf_counter() - solve_start

        current_q = self.data.qpos[self.arm_qpos_ids7].copy()
        joint_delta = self.ik_config.q[self.ik_arm_qpos_ids] - current_q
        joint_delta = np.clip(joint_delta, -max_joint_delta, max_joint_delta)
        target_ctrl = current_ctrl.copy()
        target_ctrl[: len(joint_delta)] = current_q + joint_delta
        target_ctrl = self.clamp_ctrl(target_ctrl)
        info = {
            "ee_pos": ee_pos.tolist(),
            "target_pos": target_pos.tolist(),
            "model_delta": model_delta.tolist(),
            "cube_delta": cube_delta.tolist(),
            "ik_error": (target_pos - ee_pos).tolist(),
            "ik_converged": converged,
            "ik_iters": iters,
            "ik_timing": {
                "ik_solve_sec": time.perf_counter() - ik_start,
                "ik_qp_solve_sec": ik_qp_solve_sec,
            },
        }
        return target_ctrl, joint_delta, info

    def _compile_robot_model(self):
        robot_dir = self.robot_config.robot_dir
        robot_xml = self._make_robot_xml_with_ee_site()
        with tempfile.NamedTemporaryFile("w", suffix=".xml", prefix=f"openvla_ik_{self.robot_config.name}_", dir=robot_dir, delete=False) as robot_file:
            robot_file.write(robot_xml)
            robot_path = Path(robot_file.name)
        try:
            return mujoco.MjModel.from_xml_path(str(robot_path))
        finally:
            robot_path.unlink(missing_ok=True)

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
      <freejoint name="cube_freejoint"/>
      <geom name="cube_geom" type="box" size="{scene.cube_size} {scene.cube_size} {scene.cube_size}" rgba="1.0 0.0 0.0 1.0" mass="0.02" friction="1.5 0.005 0.0001"/>
    </body>"""
        if "<worldbody>" not in robot_xml:
            raise ValueError("Robot XML does not contain a <worldbody> section for scene bodies.")
        return robot_xml.replace("<worldbody>", "<worldbody>" + scene_worldbody_xml, 1)