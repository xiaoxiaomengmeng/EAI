"""SONIC v1.1 ONNX policy inference and the MuJoCo simulation loop around it."""
from __future__ import annotations

from collections import deque
import time

import mujoco
import numpy as np
import onnxruntime as ort
from scipy.spatial.transform import Rotation

from sonic_config import (
    ACTION_SCALE,
    DEFAULT_ANGLES,
    HISTORY_FRAMES,
    ISAACLAB_TO_MUJOCO,
    KD,
    KP,
    MUJOCO_TO_ISAACLAB,
    ROOT,
    SIM_SUBSTEPS,
    WUJI_JOINTS,
    WRIST_ISAACLAB,
)
from sonic_model import build_model, object_id

from sim_scenes import PulseSession



def heading(rotation: Rotation) -> Rotation:
    matrix = rotation.as_matrix()
    return Rotation.from_euler("z", np.arctan2(matrix[1, 0], matrix[0, 0]))


class SonicPolicy:
    def __init__(self, model: mujoco.MjModel):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        self.encoder = ort.InferenceSession(str(ROOT / "models/model_encoder.onnx"), options, providers=["CPUExecutionProvider"])
        self.decoder = ort.InferenceSession(str(ROOT / "models/model_decoder.onnx"), options, providers=["CPUExecutionProvider"])
        self.qpos = np.array([model.jnt_qposadr[model.actuator_trnid[i, 0]] for i in range(29)])
        self.dofs = np.array([model.jnt_dofadr[model.actuator_trnid[i, 0]] for i in range(29)])
        sensor = object_id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu-pelvis-angular-velocity")
        self.gyro = slice(model.sensor_adr[sensor], model.sensor_adr[sensor] + model.sensor_dim[sensor])
        self.history = deque(maxlen=HISTORY_FRAMES)
        self.last_action = np.zeros(29, dtype=np.float32)
        self.heading_alignment = None

    def reset(self) -> None:
        self.history.clear()
        self.last_action.fill(0.0)
        self.heading_alignment = None

    def robot_observation(self, data: mujoco.MjData) -> tuple[np.ndarray, ...]:
        root = Rotation.from_quat(data.qpos[3:7][[1, 2, 3, 0]])
        return (
            np.asarray(data.sensordata[self.gyro], dtype=np.float32).copy(),
            (data.qpos[self.qpos] - DEFAULT_ANGLES)[MUJOCO_TO_ISAACLAB].astype(np.float32),
            data.qvel[self.dofs][MUJOCO_TO_ISAACLAB].astype(np.float32),
            self.last_action.copy(),
            root.inv().apply([0.0, 0.0, -1.0]).astype(np.float32),
        )

    def encoder_input(self, reference: dict[str, np.ndarray], data: mujoco.MjData) -> np.ndarray:
        ref_rot = Rotation.from_quat(reference["body_quat"][:, [1, 2, 3, 0]])
        robot_rot = Rotation.from_quat(data.qpos[3:7][[1, 2, 3, 0]])
        if self.heading_alignment is None:
            self.heading_alignment = heading(robot_rot) * heading(ref_rot[0]).inv()
        relative = heading(robot_rot).inv() * self.heading_alignment * ref_rot
        orientation = relative.as_matrix()[..., :2].reshape(-1)
        obs = np.zeros(1751, dtype=np.float32)
        obs[0] = 2.0
        obs[911:1631] = reference["smpl_joints"].reshape(-1)
        obs[1631:1691] = orientation
        obs[1691:1751] = reference["joint_pos"][:, WRIST_ISAACLAB].reshape(-1)
        return obs[None]

    def target(self, reference: dict[str, np.ndarray], data: mujoco.MjData) -> tuple[np.ndarray, float]:
        started = time.perf_counter()
        token = self.encoder.run(None, {"obs_dict": self.encoder_input(reference, data)})[0][0]
        observation = self.robot_observation(data)
        if not self.history:
            self.history.extend([observation] * HISTORY_FRAMES)
        else:
            self.history.append(observation)
        decoder_input = np.concatenate((token, *(np.stack([frame[field] for frame in self.history]).reshape(-1) for field in range(5))))
        if decoder_input.shape != (994,):
            raise RuntimeError(f"decoder observation has shape {decoder_input.shape}, expected (994,)")
        action = self.decoder.run(None, {"obs_dict": decoder_input[None].astype(np.float32)})[0][0]
        if not np.isfinite(action).all():
            raise RuntimeError("SONIC returned non-finite actions")
        self.last_action = action.astype(np.float32)
        target = DEFAULT_ANGLES + action[ISAACLAB_TO_MUJOCO] * ACTION_SCALE
        return target, (time.perf_counter() - started) * 1000.0


class Simulator:
    def __init__(self, xixi_scene: bool = False, *, scene: str | None = None):
        self.model = build_model(xixi_scene, scene=scene)
        self.pulse = PulseSession.for_model(self.model)
        self.data = mujoco.MjData(self.model)
        self.policy = SonicPolicy(self.model)
        self.qpos = self.policy.qpos
        self.dofs = self.policy.dofs
        self.lower = np.array([self.model.jnt_range[self.model.actuator_trnid[i, 0], 0] for i in range(29)])
        self.upper = np.array([self.model.jnt_range[self.model.actuator_trnid[i, 0], 1] for i in range(29)])
        self.hand_qpos = {}
        self.hand_dofs = {}
        self.hand_limits = {}
        for side in ("left", "right"):
            joints = [object_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_{name}") for name in WUJI_JOINTS]
            self.hand_qpos[side] = np.array([self.model.jnt_qposadr[joint] for joint in joints])
            self.hand_dofs[side] = np.array([self.model.jnt_dofadr[joint] for joint in joints])
            self.hand_limits[side] = self.model.jnt_range[joints].T
        self.pelvis = object_id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.reset()

    def control(self, target: np.ndarray) -> None:
        self.data.ctrl[:] = KP * (target - self.data.qpos[self.qpos]) - KD * self.data.qvel[self.dofs]

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qpos] = DEFAULT_ANGLES
        mujoco.mj_forward(self.model, self.data)
        self.commanded_qpos = self.data.qpos.copy()
        self.policy.reset()
        if self.pulse:
            self.pulse.reset()

    def plan(self, reference: dict[str, np.ndarray]) -> tuple[np.ndarray, float]:
        """Run the ONNX policy on the current state.

        Inference only reads `data`; it does not step physics, so the caller
        may run it outside the lock that keeps render snapshots out of the
        substep loop (the GUI thread is the only mutator of `data`).
        """
        target, inference_ms = self.policy.target(reference, self.data)
        return np.clip(target, self.lower, self.upper), inference_ms

    def advance(self, target: np.ndarray, hands: dict[str, np.ndarray]) -> None:
        for _ in range(SIM_SUBSTEPS):
            self.control(target)
            mujoco.mj_step(self.model, self.data)
            if self.pulse:
                self.pulse.update(self.data)
        for side, joints in hands.items():
            indexes = self.hand_qpos[side]
            self.data.qpos[indexes] = np.clip(joints, *self.hand_limits[side])
            self.data.qvel[self.hand_dofs[side]] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.commanded_qpos = self.data.qpos.copy()
        self.commanded_qpos[self.qpos] = target

    def step(self, reference: dict[str, np.ndarray], hands: dict[str, np.ndarray]) -> float:
        target, inference_ms = self.plan(reference)
        self.advance(target, hands)
        return inference_ms
