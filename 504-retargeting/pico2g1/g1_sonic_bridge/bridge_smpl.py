"""Convert PICO body poses into the SMPL-style fields SONIC consumes."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from bridge_config import (
    PARENTS_24,
    SMPL_BASE_INV,
    SMPL_OUTPUT_JOINTS,
    Y_180,
    Y_UP_TO_Z_UP,
)



def forward_kinematics(
    rotations: np.ndarray,
    rest_joints: np.ndarray,
    parents: np.ndarray,
) -> np.ndarray:
    offsets = rest_joints.copy()
    offsets[1:] -= rest_joints[parents[1:]]
    global_rotations = np.empty_like(rotations)
    global_positions = np.empty(rotations.shape[:-2] + (3,), dtype=np.float64)
    global_rotations[:, 0] = rotations[:, 0]
    global_positions[:, 0] = rest_joints[0]
    for joint in range(1, len(parents)):
        parent = parents[joint]
        global_rotations[:, joint] = global_rotations[:, parent] @ rotations[:, joint]
        global_positions[:, joint] = (
            global_positions[:, parent]
            + np.einsum("nij,j->ni", global_rotations[:, parent], offsets[joint])
        )
    return global_positions


def swing_euler(axis_angle: np.ndarray) -> np.ndarray:
    quaternion_xyzw = Rotation.from_rotvec(axis_angle).as_quat()
    quaternion_wxyz = quaternion_xyzw[:, [3, 0, 1, 2]]
    twist = np.zeros_like(quaternion_wxyz)
    twist[:, 0] = quaternion_wxyz[:, 0]
    twist[:, 2] = quaternion_wxyz[:, 2]
    norm = np.linalg.norm(twist, axis=1, keepdims=True)
    twist = np.divide(
        twist,
        norm,
        out=np.tile([1.0, 0.0, 0.0, 0.0], (len(twist), 1)),
        where=norm > 1e-8,
    )
    swing = Rotation.from_quat(twist[:, [1, 2, 3, 0]]).inv() * Rotation.from_quat(
        quaternion_xyzw
    )
    return swing.as_euler("XYZ", degrees=False)


def wrist_references(smpl_pose: np.ndarray) -> np.ndarray:
    joint_pos = np.zeros((len(smpl_pose), 29), dtype=np.float32)
    left_elbow = swing_euler(smpl_pose[:, 17])
    right_elbow = swing_euler(smpl_pose[:, 18])
    left_wrist = Rotation.from_rotvec(smpl_pose[:, 19]).as_euler("XYZ", degrees=False)
    right_wrist = Rotation.from_rotvec(smpl_pose[:, 20]).as_euler("XYZ", degrees=False)
    joint_pos[:, 23] = left_elbow[:, 0] + left_wrist[:, 0]
    joint_pos[:, 25] = left_wrist[:, 1]
    joint_pos[:, 27] = left_elbow[:, 2] + left_wrist[:, 2]
    joint_pos[:, 24] = -(right_elbow[:, 0] + right_wrist[:, 0])
    joint_pos[:, 26] = -right_wrist[:, 1]
    joint_pos[:, 28] = right_elbow[:, 2] + right_wrist[:, 2]
    return joint_pos


def convert(
    poses: np.ndarray,
    rest_joints: np.ndarray,
    parents: np.ndarray,
) -> dict[str, np.ndarray]:
    poses = np.asarray(poses, dtype=np.float32)
    if poses.ndim != 3 or poses.shape[1:] != (24, 7):
        raise ValueError(f"expected body poses [N,24,7], got {poses.shape}")
    if not np.isfinite(poses).all() or np.any(
        np.linalg.norm(poses[..., 3:], axis=-1) <= 1e-8
    ):
        raise ValueError("body poses contain invalid values")

    count = len(poses)
    global_rotations = Rotation.from_quat(poses[..., 3:].reshape(-1, 4)).as_matrix().reshape(
        count, 24, 3, 3
    )
    global_rotations = global_rotations @ Y_180
    local_rotations = np.empty_like(global_rotations)
    local_rotations[:, 0] = global_rotations[:, 0]
    for joint in range(1, 24):
        local_rotations[:, joint] = (
            np.swapaxes(global_rotations[:, PARENTS_24[joint]], 1, 2)
            @ global_rotations[:, joint]
        )

    smpl_pose = Rotation.from_matrix(local_rotations[:, 1:22].reshape(-1, 3, 3))
    smpl_pose = smpl_pose.as_rotvec().reshape(count, 21, 3).astype(np.float32)

    full_rotations = np.broadcast_to(np.eye(3), (count, 55, 3, 3)).copy()
    root_z_up = Y_UP_TO_Z_UP @ global_rotations[:, 0]
    full_rotations[:, 0] = root_z_up
    full_rotations[:, 1:22] = local_rotations[:, 1:22]
    joints_world = forward_kinematics(full_rotations, rest_joints, parents)

    body_rotation = root_z_up @ SMPL_BASE_INV
    smpl_joints = np.einsum(
        "nij,nkj->nki",
        np.swapaxes(body_rotation, 1, 2),
        joints_world[:, SMPL_OUTPUT_JOINTS],
    ).astype(np.float32)
    body_quat_xyzw = Rotation.from_matrix(body_rotation).as_quat().astype(np.float32)
    joint_pos = wrist_references(smpl_pose)
    return {
        "joint_pos": joint_pos,
        "joint_vel": np.zeros((count, 29), dtype=np.float32),
        "body_quat": body_quat_xyzw[:, [3, 0, 1, 2]],
        "smpl_joints": smpl_joints,
        "smpl_pose": smpl_pose,
    }
