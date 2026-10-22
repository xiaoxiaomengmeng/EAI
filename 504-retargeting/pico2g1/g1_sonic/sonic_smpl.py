"""Convert PICO body poses into SMPL references and drive Wuji hand retargeting."""
from __future__ import annotations

import sys

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from sonic_config import (
    HAND_TRACKER_NAMES,
    JOINT_NAMES,
    PARENTS_24,
    ROTATION_OFFSET_JOINTS,
    SMPL_BASE_INV,
    SMPL_OUTPUT_JOINTS,
    WUJI_DESCRIPTION,
    WUJI_JOINTS,
    WUJI_ROOT,
    Y_180,
    Y_UP_TO_Z_UP,
)



def forward_kinematics(rotations: np.ndarray, rest: np.ndarray, parents: np.ndarray) -> np.ndarray:
    offsets = rest.copy()
    offsets[1:] -= rest[parents[1:]]
    world_rot = np.empty_like(rotations)
    world_pos = np.empty(rotations.shape[:-2] + (3,), dtype=np.float64)
    world_rot[:, 0] = rotations[:, 0]
    world_pos[:, 0] = rest[0]
    for joint in range(1, len(parents)):
        parent = parents[joint]
        world_rot[:, joint] = world_rot[:, parent] @ rotations[:, joint]
        world_pos[:, joint] = world_pos[:, parent] + np.einsum("nij,j->ni", world_rot[:, parent], offsets[joint])
    return world_pos


def swing_euler(axis_angle: np.ndarray) -> np.ndarray:
    quat_xyzw = Rotation.from_rotvec(axis_angle).as_quat()
    quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]]
    twist = np.zeros_like(quat_wxyz)
    twist[:, (0, 2)] = quat_wxyz[:, (0, 2)]
    norm = np.linalg.norm(twist, axis=1, keepdims=True)
    twist = np.divide(twist, norm, out=np.tile([1.0, 0, 0, 0], (len(twist), 1)), where=norm > 1e-8)
    return (Rotation.from_quat(twist[:, [1, 2, 3, 0]]).inv() * Rotation.from_quat(quat_xyzw)).as_euler("XYZ")


def convert(
    poses: np.ndarray,
    rest: np.ndarray,
    parents: np.ndarray,
) -> dict[str, np.ndarray]:
    poses = np.asarray(poses, dtype=np.float32)
    if poses.ndim != 3 or poses.shape[1:] != (24, 7):
        raise ValueError(f"expected body poses [N,24,7], got {poses.shape}")
    count = len(poses)
    global_rot = Rotation.from_quat(poses[..., 3:].reshape(-1, 4)).as_matrix().reshape(count, 24, 3, 3) @ Y_180
    local_rot = np.empty_like(global_rot)
    local_rot[:, 0] = global_rot[:, 0]
    for joint in range(1, 24):
        local_rot[:, joint] = np.swapaxes(global_rot[:, PARENTS_24[joint]], 1, 2) @ global_rot[:, joint]
    smpl_pose = Rotation.from_matrix(local_rot[:, 1:22].reshape(-1, 3, 3)).as_rotvec().reshape(count, 21, 3)
    full_rot = np.broadcast_to(np.eye(3), (count, 55, 3, 3)).copy()
    root_z_up = Y_UP_TO_Z_UP @ global_rot[:, 0]
    full_rot[:, 0] = root_z_up
    full_rot[:, 1:22] = local_rot[:, 1:22]
    joints_world = forward_kinematics(full_rot, rest, parents)
    body_rot = root_z_up @ SMPL_BASE_INV
    smpl_joints = np.einsum("nij,nkj->nki", np.swapaxes(body_rot, 1, 2), joints_world[:, SMPL_OUTPUT_JOINTS])
    left_elbow, right_elbow = swing_euler(smpl_pose[:, 17]), swing_euler(smpl_pose[:, 18])
    left_wrist = Rotation.from_rotvec(smpl_pose[:, 19]).as_euler("XYZ")
    right_wrist = Rotation.from_rotvec(smpl_pose[:, 20]).as_euler("XYZ")
    joint_pos = np.zeros((count, 29), dtype=np.float32)
    joint_pos[:, 23] = left_elbow[:, 0] + left_wrist[:, 0]
    joint_pos[:, 25] = left_wrist[:, 1]
    joint_pos[:, 27] = left_elbow[:, 2] + left_wrist[:, 2]
    joint_pos[:, 24] = -(right_elbow[:, 0] + right_wrist[:, 0])
    joint_pos[:, 26] = -right_wrist[:, 1]
    joint_pos[:, 28] = right_elbow[:, 2] + right_wrist[:, 2]
    quat_xyzw = Rotation.from_matrix(body_rot).as_quat()
    return {
        "smpl_joints": smpl_joints.astype(np.float32),
        "body_quat": quat_xyzw[:, [3, 0, 1, 2]].astype(np.float32),
        "joint_pos": joint_pos,
    }


def apply_rotation_sources(
    poses: np.ndarray,
    trackers: np.ndarray,
    rotation_offsets: dict[str, np.ndarray],
) -> np.ndarray:
    result = poses.copy()
    for side, name in enumerate(HAND_TRACKER_NAMES):
        wrist = JOINT_NAMES.index(f"{name.split('_')[0]}_wrist")
        result[:, wrist, 3:] = trackers[:, side, 3:]
    for name, _ in ROTATION_OFFSET_JOINTS:
        joint = JOINT_NAMES.index(name)
        result[:, joint, 3:] = (
            Rotation.from_quat(result[:, joint, 3:])
            * Rotation.from_euler("XYZ", rotation_offsets[name].copy(), degrees=True)
        ).as_quat().astype(np.float32)
    return result


class WujiMapper:
    def __init__(self):
        sys.path.insert(0, str(WUJI_ROOT))
        from wuji_retargeting import Retargeter

        self.retargeters = {}
        for side in ("left", "right"):
            with (WUJI_ROOT / f"example/config/adaptive_analytical_wuji_glove_{side}.yaml").open() as file:
                config = yaml.safe_load(file)
            config["optimizer"]["urdf_path"] = str(WUJI_DESCRIPTION / f"hand/body/urdf/{side}.urdf")
            retargeter = Retargeter.from_config(config, side)
            expected = [f"{side}_{name}" for name in WUJI_JOINTS]
            if retargeter.optimizer.robot.dof_joint_names != expected:
                raise ValueError(f"unexpected {side} Wuji Hand joint order")
            self.retargeters[side] = retargeter

    def retarget(self, gloves: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {
            side: retargeter.retarget(gloves[side][..., :3] * (-1.0, 1.0, 1.0)).astype(np.float32)
            for side, retargeter in self.retargeters.items()
        }
