"""OSC_POSE 等价控制器 (自建 libero_spatial 场景版, M3)。

复刻 robosuite controllers/osc.py + utils/control_utils.py 的**目标位姿合成语义**
(osc_pose.json: output_max=[0.05,0.05,0.05,0.5,0.5,0.5], control_delta=true,
uncouple_pos_ori=true, control_freq=20):

- 缩放   : scaled = clip(action[:6], -1, 1) * output_max
- 位置目标: goal_pos = ee_pos + scaled[:3]                (世界系直接相加, set_goal_position)
- 姿态目标: goal_R  = R(axisangle(scaled[3:6])) @ ee_R    (世界系左乘, set_goal_orientation)
- 每步从**当前** EE 位姿重算目标 (非在旧 goal 上累加), 与 robosuite set_goal 一致
- 夹爪   : action[-1] ∈ {-1,+1} (-1=open, +1=close, 即 libero_obs.libero_action_postprocess 产物)
           -> 双指位置执行器目标 [±0.04, ∓0.04] / [0, 0]
- 步频   : 1 个 policy action = 1 个 robosuite control step = 0.05s = 25 个 0.002s 物理步

近似点 (与 robosuite 的差异, 属"等价控制"而非"复制控制律"):
robosuite 用 OSC 阻抗力矩律 (kp=150, damping_ratio=1) 跟踪 goal; 本控制器改用
mink 位置+姿态 QP IK 解出关节目标、由 position 执行器跟踪。跟踪刚度由执行器 kp 决定,
稳态目标与增量语义与 OSC 一致, 瞬态响应近似。
"""

from __future__ import annotations

import mujoco
import mink
import numpy as np

# osc_pose.json 的 output_max (归一化动作 -> 物理增量)。
OSC_OUTPUT_MAX = np.array([0.05, 0.05, 0.05, 0.5, 0.5, 0.5])
# control_freq=20, timestep=0.002 -> 每个 control step 25 个物理步。
PHYSICS_STEPS_PER_ACTION = 25
# mink 求解参数 (对齐 sim/mujoco_env.py 的 IK 配置风格)。
IK_DT = 0.1
IK_MAX_ITERS = 20
IK_POS_THRESHOLD = 1e-3
IK_SOLVER = "daqp"
# 单次物理步内关节增量上限, 防 IK 跳变 (25 步累计上限 0.5rad, 远大于单步所需)。
MAX_JOINT_DELTA_PER_STEP = 0.02

GRIPPER_OPEN_CTRL = np.array([0.04, -0.04])
GRIPPER_CLOSE_CTRL = np.array([0.0, 0.0])

ARM_JOINT_NAMES = [f"joint{i}" for i in range(1, 8)]
FINGER_ACTUATOR_NAMES = ["gripper_finger_joint1", "gripper_finger_joint2"]


def axisangle2mat(aa: np.ndarray) -> np.ndarray:
    """轴角 -> 旋转矩阵 (Rodrigues), 复刻 robosuite trans.quat2mat(trans.axisangle2quat(aa))。"""
    angle = float(np.linalg.norm(aa))
    if angle < 1e-12:
        return np.eye(3)
    k = aa / angle
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


class LiberoOscEquivalentController:
    """把 LIBERO 约定 (后处理完) 的 7D action 执行到自建场景上的等价 OSC 控制器。"""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, ee_site: str = "grip_site"):
        self.model = model
        self.data = data
        self.ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ee_site)
        if self.ee_site_id < 0:
            raise ValueError(f"场景缺少 EE site: {ee_site}")
        self.arm_qpos_ids = np.array(
            [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in ARM_JOINT_NAMES]
        )
        self.arm_dof_ids = np.array(
            [model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in ARM_JOINT_NAMES]
        )
        self.arm_ctrl_ids = np.array(
            [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"actuator{i}") for i in range(1, 8)]
        )
        self.finger_ctrl_ids = np.array(
            [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in FINGER_ACTUATOR_NAMES]
        )

        # mink 直接在完整场景模型上求解; 只积分臂 dof, 物体 freejoint 不受 IK 影响。
        self.ik_config = mink.Configuration(model)
        self.frame_task = mink.FrameTask(ee_site, "site", position_cost=1.0, orientation_cost=1.0, lm_damping=1.0)
        self.posture_task = mink.PostureTask(model, cost=1e-2)

        self.goal_pos = self.ee_pos.copy()
        self.goal_rot = self.ee_rot.copy()
        self.gripper_ctrl = GRIPPER_OPEN_CTRL.copy()

    # ---- EE 状态 (对齐 robosuite robot0_eef_pos / eef_quat 的 site 约定) ----
    @property
    def ee_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.ee_site_id].copy()

    @property
    def ee_rot(self) -> np.ndarray:
        return self.data.site_xmat[self.ee_site_id].reshape(3, 3).copy()

    def reset_goal_to_current(self):
        """沉降结束后调用: goal 初始化为当前 EE 位姿 (对齐 OSC 的 initial_ee_pos/ori)。"""
        self.goal_pos = self.ee_pos
        self.goal_rot = self.ee_rot
        self.gripper_ctrl = GRIPPER_OPEN_CTRL.copy()

    def set_action(self, action7: np.ndarray):
        """接收**后处理完**的 7D action (env 约定: 前6维归一化增量, 末维 ±1 夹爪)。"""
        action = np.asarray(action7, dtype=np.float64).reshape(-1)
        if action.shape != (7,):
            raise ValueError(f"期望 7D action, 实际 {action.shape}")
        scaled = np.clip(action[:6], -1.0, 1.0) * OSC_OUTPUT_MAX
        # set_goal_position: 世界系直接相加
        self.goal_pos = self.ee_pos + scaled[:3]
        # set_goal_orientation: 世界系左乘 delta 旋转
        self.goal_rot = axisangle2mat(scaled[3:6]) @ self.ee_rot
        # 夹爪: -1=open, +1=close
        self.gripper_ctrl = GRIPPER_CLOSE_CTRL.copy() if action[6] > 0 else GRIPPER_OPEN_CTRL.copy()

    def _solve_ik_to_goal(self):
        """mink 位置+姿态 QP IK 解到当前 goal, 只积分臂 dof。"""
        q = self.ik_config.q.copy()
        q[self.arm_qpos_ids] = self.data.qpos[self.arm_qpos_ids]
        self.ik_config.update(q)
        self.posture_task.set_target(self.ik_config.q)
        target = mink.SE3.from_rotation_and_translation(
            mink.SO3.from_matrix(self.goal_rot), self.goal_pos
        )
        self.frame_task.set_target(target)
        for _ in range(IK_MAX_ITERS):
            vel = mink.solve_ik(self.ik_config, [self.frame_task, self.posture_task], IK_DT, IK_SOLVER, damping=1e-3)
            q = self.ik_config.q.copy()
            delta = np.clip(vel[self.arm_dof_ids] * IK_DT, -MAX_JOINT_DELTA_PER_STEP, MAX_JOINT_DELTA_PER_STEP)
            q[self.arm_qpos_ids] += delta
            self.ik_config.update(q)
            error = self.frame_task.compute_error(self.ik_config)
            if float(np.linalg.norm(error[:3])) <= IK_POS_THRESHOLD:
                break
        return self.ik_config.q[self.arm_qpos_ids].copy()

    def step_action(self):
        """执行 1 个 policy action = 25 物理步 (1 个 robosuite control step)。"""
        for _ in range(PHYSICS_STEPS_PER_ACTION):
            arm_q = self._solve_ik_to_goal()
            self.data.ctrl[self.arm_ctrl_ids] = arm_q
            self.data.ctrl[self.finger_ctrl_ids] = self.gripper_ctrl
            mujoco.mj_step(self.model, self.data)

    def tracking_error(self) -> dict:
        """当前 EE 与 goal 的差距, 供 M3 校验/ M4 日志。"""
        pos_err = float(np.linalg.norm(self.ee_pos - self.goal_pos))
        # 姿态误差用旋转差矩阵的轴角模长, 避免 scipy 依赖
        r = self.ee_rot.T @ self.goal_rot
        cos_angle = (np.trace(r) - 1.0) / 2.0
        rot_err = float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
        return {"pos_err_m": pos_err, "rot_err_rad": rot_err}
