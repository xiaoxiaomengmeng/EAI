"""求解:全身 IK(FullBodyIK)与 Wuji Hand 重定向(WujiMapper/set_hands)。"""

from __future__ import annotations

import sys

import mujoco
import numpy as np
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation
import yaml

from g1_config import (
    BODY_JOINT_SPEED_LIMIT, FPS, JOINT_NAMES, ROTATION_TASKS,
    WUJI_DESCRIPTION, WUJI_JOINTS, WUJI_ROOT,
)
from g1_model import object_id


class FullBodyIK:
    def __init__(
        self, model: mujoco.MjModel, data: mujoco.MjData, tuning: dict[str, float]
    ):
        self.model = model
        self.data = data
        self.tuning = tuning
        self.body_qpos = []
        self.body_dofs = []
        self.lower = []
        self.upper = []
        joint_indices = {}
        for actuator in range(model.nu):
            joint = model.actuator_trnid[actuator, 0]
            joint_indices[
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
            ] = actuator
            self.body_qpos.append(model.jnt_qposadr[joint])
            self.body_dofs.append(model.jnt_dofadr[joint])
            self.lower.append(model.jnt_range[joint, 0])
            self.upper.append(model.jnt_range[joint, 1])
        self.body_qpos = np.asarray(self.body_qpos)
        self.body_dofs = np.asarray(self.body_dofs)
        self.lower = np.asarray(self.lower)
        self.upper = np.asarray(self.upper)
        if len(self.body_qpos) != 29:
            raise ValueError(f"expected 29 G1 body joints, got {len(self.body_qpos)}")
        for side in ("left", "right"):
            knee = joint_indices[f"{side}_knee_joint"]
            elbow = joint_indices[f"{side}_elbow_joint"]
            self.lower[knee] = max(self.lower[knee], 0.0)
            self.upper[elbow] = min(self.upper[elbow], np.pi / 2)

        self.body_ids = {
            name: object_id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in (
                "pelvis",
                "torso_link",
                "left_hip_pitch_link",
                "left_knee_link",
                "left_ankle_roll_link",
                "right_hip_pitch_link",
                "right_knee_link",
                "right_ankle_roll_link",
                "left_shoulder_pitch_link",
                "left_elbow_link",
                "left_wrist_roll_link",
                "left_wrist_yaw_link",
                "right_shoulder_pitch_link",
                "right_elbow_link",
                "right_wrist_roll_link",
                "right_wrist_yaw_link",
            )
        }
        self.source_ids = {name: JOINT_NAMES.index(name) for name in JOINT_NAMES}
        mujoco.mj_forward(model, data)
        self.neutral_pos = {
            name: data.xpos[body_id].copy() for name, body_id in self.body_ids.items()
        }
        self.source_first = None
        self.source_first_rot = None
        self.rotation_alignment = {}
        self.scales = {}

    def reset_calibration(self) -> None:
        self.source_first = None
        self.source_first_rot = None
        self.rotation_alignment.clear()
        self.scales.clear()

    @staticmethod
    def rotation(pose: np.ndarray, index: int) -> np.ndarray:
        return Rotation.from_quat(pose[index, 3:]).as_matrix()

    def calibrate(self, pose: np.ndarray) -> None:
        self.source_first = pose.copy()
        self.source_first_rot = np.stack(
            [self.rotation(pose, index) for index in range(len(pose))]
        )
        left = np.mean([
            pose[self.source_ids[f"left_{joint}"], :3]
            - pose[self.source_ids[f"right_{joint}"], :3]
            for joint in ("shoulder", "hip")
        ], axis=0)
        left[1] = 0.0
        left /= np.linalg.norm(left)
        up = np.array([0.0, 1.0, 0.0])
        self.source_to_g1 = np.stack((np.cross(left, up), left, up))
        for limb in ("leg", "arm"):
            values = []
            for side in ("left", "right"):
                if limb == "leg":
                    source_names = (f"{side}_hip", f"{side}_knee", f"{side}_ankle")
                    robot_names = (
                        f"{side}_hip_pitch_link",
                        f"{side}_knee_link",
                        f"{side}_ankle_roll_link",
                    )
                else:
                    source_names = (
                        f"{side}_shoulder",
                        f"{side}_elbow",
                        f"{side}_wrist",
                    )
                    robot_names = (
                        f"{side}_shoulder_pitch_link",
                        f"{side}_elbow_link",
                        f"{side}_wrist_roll_link",
                    )
                source = pose[[self.source_ids[name] for name in source_names], :3]
                robot = np.stack([self.neutral_pos[name] for name in robot_names])
                values.append(np.linalg.norm(np.diff(robot, axis=0), axis=1).sum() /
                              np.linalg.norm(np.diff(source, axis=0), axis=1).sum())
            self.scales[limb] = float(np.clip(np.mean(values), 0.6, 1.4))

    def relative_rotation(self, pose: np.ndarray, source_name: str) -> np.ndarray:
        index = self.source_ids[source_name]
        delta = self.rotation(pose, index) @ self.source_first_rot[index].T
        return self.source_to_g1 @ delta @ self.source_to_g1.T

    def targets(self, pose: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        waist = self.source_ids["waist"]
        root_rotation = self.relative_rotation(pose, "waist")
        root_position = self.neutral_pos["pelvis"] + (
            self.scales["leg"] * self.tuning["leg_scale"]
        ) * (
            self.source_to_g1 @ (pose[waist, :3] - self.source_first[waist, :3])
        )
        self.data.qpos[:3] = root_position
        root_quat = Rotation.from_matrix(root_rotation).as_quat()
        self.data.qpos[3:7] = root_quat[[3, 0, 1, 2]]

        positions = {}
        for side in ("left", "right"):
            for limb, middle, end in (
                ("leg", "knee", "ankle"),
                ("arm", "elbow", "wrist"),
            ):
                source_anchor = f"{side}_{'hip' if limb == 'leg' else 'shoulder'}"
                robot_anchor = f"{side}_{'hip_pitch' if limb == 'leg' else 'shoulder_pitch'}_link"
                anchor_offset = (
                    self.neutral_pos[robot_anchor] - self.neutral_pos["pelvis"]
                ).copy()
                anchor_offset[1] *= self.tuning[
                    "hip_width" if limb == "leg" else "shoulder_width"
                ]
                anchor = root_position + root_rotation @ anchor_offset
                source_anchor_pos = pose[self.source_ids[source_anchor], :3]
                for joint in (middle, end):
                    source_name = f"{side}_{joint}"
                    source_delta = (
                        pose[self.source_ids[source_name], :3] - source_anchor_pos
                    )
                    positions[f"{side}_{joint}"] = (
                        anchor
                        + self.scales[limb]
                        * self.tuning[f"{limb}_scale"]
                        * (self.source_to_g1 @ source_delta)
                    )

        rotations = {}
        for task, source_name, _, _ in ROTATION_TASKS:
            if task in self.rotation_alignment:
                target = (
                    self.source_to_g1
                    @ self.rotation(pose, self.source_ids[source_name])
                    @ self.rotation_alignment[task]
                )
                if task.endswith("wrist"):
                    side = task.split("_", 1)[0]
                    target = target @ Rotation.from_euler(
                        "XYZ",
                        [self.tuning[f"{side}_wrist_{axis}"] for axis in "xyz"],
                        degrees=True,
                    ).as_matrix()
                rotations[task] = target
        return positions, rotations

    def solve(self, pose: np.ndarray) -> float:
        first = self.source_first is None
        if first:
            self.calibrate(pose)
        positions, rotations = self.targets(pose)
        previous = self.data.qpos[self.body_qpos].copy()
        frame_lower = self.lower
        frame_upper = self.upper
        if not first:
            frame_lower = np.maximum(
                frame_lower, previous - BODY_JOINT_SPEED_LIMIT / FPS
            )
            frame_upper = np.minimum(
                frame_upper, previous + BODY_JOINT_SPEED_LIMIT / FPS
            )
        for _ in range(30 if first else 8):
            mujoco.mj_forward(self.model, self.data)
            errors = []
            jacobians = []
            for side in ("left", "right"):
                for joint in ("knee", "ankle", "elbow", "wrist"):
                    weight = self.tuning[f"{joint}_pos"]
                    body_name = f"{side}_{joint if joint in ('knee', 'elbow') else joint + '_roll'}_link"
                    body_id = self.body_ids[body_name]
                    jac_pos = np.zeros((3, self.model.nv))
                    mujoco.mj_jacBody(self.model, self.data, jac_pos, None, body_id)
                    errors.append(weight * (positions[f"{side}_{joint}"] - self.data.xpos[body_id]))
                    jacobians.append(weight * jac_pos[:, self.body_dofs])
            for task, _, body_name, weight_key in ROTATION_TASKS:
                if task not in rotations:
                    continue
                weight = self.tuning[weight_key]
                body_id = self.body_ids[body_name]
                current = self.data.xmat[body_id].reshape(3, 3)
                error = Rotation.from_matrix(rotations[task] @ current.T).as_rotvec()
                jac_rot = np.zeros((3, self.model.nv))
                mujoco.mj_jacBody(self.model, self.data, None, jac_rot, body_id)
                errors.append(weight * error)
                jacobians.append(weight * jac_rot[:, self.body_dofs])
            error = np.concatenate(errors)
            jacobian = np.vstack(jacobians)
            damping = self.tuning["damping"]
            q = self.data.qpos[self.body_qpos]
            step = lsq_linear(
                np.vstack((jacobian, damping * np.eye(len(self.body_dofs)))),
                np.concatenate((error, np.zeros(len(self.body_dofs)))),
                bounds=(frame_lower - q, frame_upper - q),
            ).x
            self.data.qpos[self.body_qpos] = q + step
            if np.linalg.norm(error) < 0.01:
                break
        mujoco.mj_forward(self.model, self.data)
        if first:
            for task, source_name, body_name, _ in ROTATION_TASKS:
                source_rotation = self.source_to_g1 @ self.rotation(
                    pose, self.source_ids[source_name]
                )
                body_id = self.body_ids[body_name]
                self.rotation_alignment[task] = (
                    source_rotation.T
                    @ self.data.xmat[body_id].reshape(3, 3)
                )
        return float(np.linalg.norm(error))


class WujiMapper:
    def __init__(self, tuning: dict[str, float]):
        sys.path.insert(0, str(WUJI_ROOT))
        from wuji_retargeting import Retargeter

        self.tuning = tuning
        self.retargeters = {}
        for side in ("left", "right"):
            config_path = (
                WUJI_ROOT
                / f"example/config/adaptive_analytical_wuji_glove_{side}.yaml"
            )
            with config_path.open() as file:
                config = yaml.safe_load(file)
            config["optimizer"]["urdf_path"] = str(
                WUJI_DESCRIPTION / f"hand/body/urdf/{side}.urdf"
            )
            retargeter = Retargeter.from_config(config, side)
            expected = [f"{side}_{name}" for name in WUJI_JOINTS]
            if retargeter.optimizer.robot.dof_joint_names != expected:
                raise ValueError(f"unexpected {side} Wuji Hand joint order")
            self.retargeters[side] = retargeter

    def retarget(self, gloves: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        for retargeter in self.retargeters.values():
            retargeter.lp_filter.alpha = self.tuning["hand_response"]
        return {
            side: retargeter.retarget(
                gloves[side][..., :3] * (-1.0, 1.0, 1.0)
            ).astype(np.float32)
            for side, retargeter in self.retargeters.items()
        }


def set_hands(model: mujoco.MjModel, data: mujoco.MjData, hands: dict[str, np.ndarray]) -> None:
    for side, joints in hands.items():
        for value, short_name in zip(joints, WUJI_JOINTS):
            name = f"{side}_{short_name}"
            joint = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            data.qpos[model.jnt_qposadr[joint]] = np.clip(value, *model.jnt_range[joint])
