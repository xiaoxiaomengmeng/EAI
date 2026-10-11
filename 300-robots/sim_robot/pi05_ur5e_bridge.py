#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
π0.5-LIBERO 策略 → UR5e MuJoCo 末端 OSC 桥接闭环（免训练零样本泛化实验）

链路：
  MuJoCo 20Hz 观测（2 路 256² 图像 + 8 维 LIBERO 风格 state）
    → PI05Policy（复用 101-pi-serial-models/.venv312 中的 lerobot 代码与权重）
    → 7 维 relative OSC delta 动作 [dx,dy,dz,drx,dry,drz,grip]（归一化 ±1）
    → UR5e 侧：6D 位姿 DLS 逆运动学 → 6 关节目标角 → 模型自带位置伺服
    → 夹爪：grip>0 闭合（模型 ctrl=0），grip<0 张开（ctrl=255）

LIBERO 侧语义（与 envs/libero.py、processor/env_processor.py 对齐）：
  state = [ee_pos(3, m), axis-angle(3, rad), finger_qpos(2)]
  action = 归一化末端增量，20Hz，relative OSC；rotation 用 axis-angle 增量
  图像：agentview + eye-in-hand，训练时做了 180° 翻转（本脚本可选）

运行（必须用训练 venv，torch+lerobot+mujoco 3.8 全齐）：
  # 无显示器/服务器：离屏渲染（看不到实时窗口，可用 --record 存视频）
  cd ~/ws/300-robots/sim_robot
  MUJOCO_GL=egl HF_HUB_OFFLINE=1 \\
  /home/creator/ws/101-pi-serial-models/.venv312/bin/python \\
      pi05_ur5e_bridge.py --prompt "put the cube into the mug" --record

  # 有桌面、想看实时 MuJoCo 画面：必须用 glfw，不能设 egl
  MUJOCO_GL=glfw HF_HUB_OFFLINE=1 \\
  /home/creator/ws/101-pi-serial-models/.venv312/bin/python \\
      pi05_ur5e_bridge.py --prompt "put the cube into the mug" --view

  无 GPU/无渲染时的链路自检（不加载模型、不渲染）：
  /usr/bin/python3 pi05_ur5e_bridge.py --dry-run
  /usr/bin/python3 pi05_ur5e_bridge.py --dry-run --view   # 带实时画面
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import mujoco

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
SIM_DT = 0.001                # MuJoCo 物理步长 1ms
CONTROL_HZ = 20               # π0.5 LIBERO 策略频率
SUBSTEPS = int(round(1.0 / CONTROL_HZ / SIM_DT))   # 每个控制周期 50 个物理步 = 50ms

IMG_SIZE = 256                # LIBERO 输入特征 256²（模型内部再 resize 到 224）

# OSC delta → 物理增量的增益（对齐 robosuite OSC_POSE 的 output_limits）
POS_GAIN = 0.05               # action=1 → 每 tick 末端移动 5cm（robosuite 位置限幅 ±0.05）
ROT_GAIN = 0.25               # action=1 → 每 tick 旋转 0.25rad（robosuite 姿态限幅 ±0.5）
GRIP_OPEN_TH = 0.2            # robosuite 约定：action > 0 = 张开
GRIP_CLOSE_TH = -0.2          #                   action < 0 = 闭合（迟滞 ±0.2）

# ---- UR5e 世界 → 伪 LIBERO/Franka 世界的标定（结构性对齐，唯一转换边界）----
# 位置平移：UR 桌面 z=0 / 物体 x≈0.4 → LIBERO 桌面 z≈0.82 / 物体 x≈0.0
POLICY_POS_OFFSET = np.array([-0.40, 0.0, 0.82])
# 工具姿态标定：home 位 UR EE 姿态 Rz(π)（site z 朝上）→ Franka 抓握姿态 Rx(π)（z 朝下）
# R_policy = Ry(π) @ R_ur
R_OFFSET = np.array([[-1.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0],
                     [0.0, 0.0, -1.0]])

HOME_Q = np.array([0.0, -1.6, 1.7, -1.7, -1.57, 0.0])

# 马克杯（场景固定道具，world 坐标）
MUG_XY = np.array([0.20, -0.25])
MUG_INNER_R = 0.036
MUG_RIM_Z = 0.075

SCENE_DIR = (
    Path(__file__).resolve().parent.parent
    / "RynnMotion/models/3.robot_arm/21.ur5e/scene"
)
SCENE_FILE = "scene_pickplace.xml"
DEFAULT_CKPT = (
    "/home/creator/ws/101-pi-serial-models/outputs/"
    "pi05_libero_expert/checkpoints/last/pretrained_model"
)
OUT_DIR = Path(__file__).resolve().parent / "outputs"


# ---------------------------------------------------------------------------
# 四元数 / 旋转工具
# ---------------------------------------------------------------------------
def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """MuJoCo w,x,y,z 四元数乘法。"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def axisangle_to_quat(aa: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(aa))
    if angle < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = aa / angle
    return np.concatenate([[np.cos(angle / 2)], axis * np.sin(angle / 2)])


def quat_to_axisangle(q: np.ndarray) -> np.ndarray:
    """MuJoCo wxyz → axis-angle 3 向量。LIBERO processor 用 xyzw 公式，这里直接标准实现。"""
    q = q / np.linalg.norm(q)
    w = np.clip(q[0], -1.0, 1.0)
    den = np.sqrt(max(0.0, 1.0 - w * w))
    if den < 1e-8:
        return np.zeros(3)
    return q[1:4] / den * (2.0 * np.arccos(w))


def mat_to_quat(mat: np.ndarray) -> np.ndarray:
    """3×3 旋转矩阵 → MuJoCo wxyz 四元数。"""
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, mat.flatten())
    return q


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """MuJoCo wxyz 四元数 → 3×3 旋转矩阵。"""
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, q / np.linalg.norm(q))
    return m.reshape(3, 3)


def orientation_error(cur_r: np.ndarray, tgt_r: np.ndarray) -> np.ndarray:
    """MuJoCo 官方推荐的 3×3 旋转矩阵误差（世界系下的轴角误差）。"""
    return 0.5 * (
        np.cross(cur_r[:, 0], tgt_r[:, 0])
        + np.cross(cur_r[:, 1], tgt_r[:, 1])
        + np.cross(cur_r[:, 2], tgt_r[:, 2])
    )


# ---------------------------------------------------------------------------
# UR5e MuJoCo 环境（对外模仿 LIBERO 的观测/动作接口）
# ---------------------------------------------------------------------------
class UR5ePickPlaceEnv:
    def __init__(self, render: bool = True, flip_images: bool = False,
                 pose_align: bool = True, verbose: bool = True):
        self.render_enabled = render
        self.flip_images = flip_images
        self.pose_align = pose_align
        self.verbose = verbose

        # MjSpec 在 wrist_3_link 上挂腕部相机（不改原始 XML）；
        # meshdir 相对路径按 CWD 解析，编译期间临时切到 scene 目录。
        cwd = os.getcwd()
        os.chdir(SCENE_DIR)
        try:
            spec = mujoco.MjSpec.from_file(SCENE_FILE)
            body = spec.body("wrist_3_link")
            if not any(c.name == "wrist_camera" for c in body.cameras):
                body.add_camera(
                    name="wrist_camera", pos=[0.0, 0.0, 0.12],
                    euler=[0.0, np.pi, 0.0], fovy=60, resolution=[IMG_SIZE, IMG_SIZE],
                )
            self.model = spec.compile()
        finally:
            os.chdir(cwd)

        self.data = mujoco.MjData(self.model)

        # id 索引
        self.arm_act_ids = []
        for prefix in ["shoulder_pan", "shoulder_lift", "elbow",
                       "wrist_1", "wrist_2", "wrist_3"]:
            for i in range(self.model.nu):
                n = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                if n and n.startswith(prefix):
                    self.arm_act_ids.append(i)
        assert len(self.arm_act_ids) == 6
        self.arm_jnt_ids = [self.model.actuator_trnid[a, 0] for a in self.arm_act_ids]
        self.arm_qadr = np.array([self.model.jnt_qposadr[j] for j in self.arm_jnt_ids])

        self.grip_id = next(
            i for i in range(self.model.nu)
            if mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i).startswith("grip")
        )
        self.ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "EE")
        self.drv_l = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "left_driver_joint")
        self.drv_r = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "right_driver_joint")

        # 相机
        self.cam_agent = self._find_camera("ext_camera")
        self.cam_wrist = self._find_camera("wrist_camera")

        # 自由方块
        self.cubes = []
        for i in range(1, self.model.nbody):
            if (self.model.body_jntnum[i] > 0 and
                    self.model.jnt_type[self.model.body_jntadr[i]] == mujoco.mjtJoint.mjJNT_FREE):
                self.cubes.append(i)

        # 工作空间（基座系）
        self.jnt_lo = self.model.jnt_range[self.arm_jnt_ids, 0]
        self.jnt_hi = self.model.jnt_range[self.arm_jnt_ids, 1]

        self.renderer = None
        if render:
            self.renderer = mujoco.Renderer(self.model, height=IMG_SIZE, width=IMG_SIZE)

        self.grip_open = True       # 当前夹爪状态（带迟滞）
        self._success_count = 0

    def _find_camera(self, prefix: str) -> str:
        for i in range(self.model.ncam):
            n = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            if n and (n == prefix or n.startswith(prefix)):
                return n
        raise RuntimeError(f"找不到相机 {prefix}")

    # ---- UR5e ↔ 伪 LIBERO 坐标转换（唯一边界，观测与动作都必须经过这里）----
    def ur_pose_to_policy(self, pos: np.ndarray, rot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """UR EE 世界位姿（pos + 3×3 R）→ 策略侧 Franka 风格位姿。"""
        if not self.pose_align:
            return pos.copy(), rot.copy()
        return pos + POLICY_POS_OFFSET, R_OFFSET @ rot

    def policy_rot_delta_to_ur(self, dmat_policy: np.ndarray, rot_ur: np.ndarray) -> np.ndarray:
        """策略侧世界系旋转增量 → UR 侧目标 3×3 姿态。
        R_p,tgt = dR_p @ R_p  ⇒  R_u,tgt = R_offᵀ dR_p R_off @ R_u（共轭）。"""
        if not self.pose_align:
            return dmat_policy @ rot_ur
        d_ur = R_OFFSET.T @ dmat_policy @ R_OFFSET
        return d_ur @ rot_ur

    # ---- 状态 -------------------------------------------------------------
    def get_state(self) -> np.ndarray:
        """组装 8 维 LIBERO 风格 state：[ee_xyz, ee_axisangle, finger_l, finger_r]。"""
        ee_pos = self.data.site_xpos[self.ee_id].copy()
        ee_rot = self.data.site_xmat[self.ee_id].reshape(3, 3).copy()
        p_pol, r_pol = self.ur_pose_to_policy(ee_pos, ee_rot)
        aa = quat_to_axisangle(mat_to_quat(r_pol))

        ql = self.data.qpos[self.model.jnt_qposadr[self.drv_l]]
        qr = self.data.qpos[self.model.jnt_qposadr[self.drv_r]]
        # driver 行程 [0, 0.8]：0 闭合 → 映射到 LIBERO Franka 风格 ±0.04（镜像）
        f = np.clip((ql + qr) / 2.0 / 0.8, 0.0, 1.0)
        fingers = np.array([0.04 * f, -0.04 * f], dtype=np.float32)
        return np.concatenate([p_pol, aa, fingers]).astype(np.float32)

    def get_images(self) -> tuple[np.ndarray, np.ndarray]:
        """返回 (agentview, wrist) HWC uint8 256²。"""
        imgs = []
        for cam in (self.cam_agent, self.cam_wrist):
            self.renderer.update_scene(self.data, camera=cam)
            img = self.renderer.render()
            if self.flip_images:
                img = img[::-1, ::-1]
            imgs = imgs + [np.ascontiguousarray(img)]
        return imgs[0], imgs[1]

    def get_observation_numpy(self) -> dict:
        state = self.get_state()
        obs = {"state": state}
        if self.renderer is not None:
            obs["image"], obs["image2"] = self.get_images()
        return obs

    # ---- 重置 -------------------------------------------------------------
    def reset(self, seed: int | None = None) -> dict:
        if seed is not None:
            np.random.seed(seed)
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        # 伺服到 home 位、张开夹爪，等待稳定
        self.data.ctrl[self.arm_act_ids] = HOME_Q
        self.data.ctrl[self.grip_id] = 255.0
        for _ in range(3000):
            mujoco.mj_step(self.model, self.data)
        self.grip_open = True
        self._success_count = 0
        return self.get_observation_numpy()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    # ---- IK ---------------------------------------------------------------
    def solve_ik(self, tgt_pos: np.ndarray, tgt_quat: np.ndarray,
                 iters: int = 15, damp: float = 0.05) -> np.ndarray:
        """6D 位姿 DLS 微分逆运动学，返回 6 个目标关节角。"""
        tgt_quat = tgt_quat / np.linalg.norm(tgt_quat)
        tgt_r = np.zeros(9)
        mujoco.mju_quat2Mat(tgt_r, tgt_quat)
        tgt_r = tgt_r.reshape(3, 3)

        q = self.data.qpos[self.arm_qadr].copy()
        nv = self.model.nv
        jacp = np.zeros((3, nv))
        jacr = np.zeros((3, nv))
        for _ in range(iters):
            self.data.qpos[self.arm_qadr] = q
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)
            cur_pos = self.data.site_xpos[self.ee_id]
            cur_r = self.data.site_xmat[self.ee_id].reshape(3, 3)

            e_pos = tgt_pos - cur_pos
            e_rot = orientation_error(cur_r, tgt_r)
            err = np.concatenate([e_pos, e_rot])
            if np.linalg.norm(err[:3]) < 2e-4 and np.linalg.norm(err[3:]) < 2e-3:
                break

            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_id)
            J = np.vstack([jacp[:, self.arm_qadr], jacr[:, self.arm_qadr]])
            dq = J.T @ np.linalg.solve(J @ J.T + damp ** 2 * np.eye(6), err)
            dq = np.clip(dq, -0.2, 0.2)
            q = np.clip(q + dq, self.jnt_lo, self.jnt_hi)

        return q

    # ---- 动作执行 ----------------------------------------------------------
    def apply_action(self, action: np.ndarray):
        """消费一个 7 维 LIBERO relative OSC delta 动作，步进 50ms。"""
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        dpos = action[:3] * POS_GAIN
        dmat_pol = quat_to_mat(axisangle_to_quat(action[3:6] * ROT_GAIN))

        # 当前 EE（UR 世界系）；旋转增量经标定边界共轭回 UR 工具约定
        cur_pos = self.data.site_xpos[self.ee_id].copy()
        cur_rot = self.data.site_xmat[self.ee_id].reshape(3, 3).copy()
        tgt_pos = cur_pos + dpos                 # 两世界系同向，平移不影响 delta
        tgt_rot = self.policy_rot_delta_to_ur(dmat_pol, cur_rot)
        tgt_quat = mat_to_quat(tgt_rot)

        q_target = self.solve_ik(tgt_pos, tgt_quat)
        self.data.ctrl[self.arm_act_ids] = q_target

        # 夹爪迟滞：robosuite/LIBERO 约定 +1 张开（UR ctrl=255）/ -1 闭合（ctrl=0）
        g = action[6]
        if (not self.grip_open) and g > GRIP_OPEN_TH:
            self.grip_open = True
        elif self.grip_open and g < GRIP_CLOSE_TH:
            self.grip_open = False
        self.data.ctrl[self.grip_id] = 255.0 if self.grip_open else 0.0

        for _ in range(SUBSTEPS):
            mujoco.mj_step(self.model, self.data)

    # ---- 成功判定 ----------------------------------------------------------
    def success(self) -> bool:
        """任一 2cm 方块进入杯口内并低于杯沿，连续 10 个控制周期判定成功。"""
        inside = False
        for bid in self.cubes:
            jid = self.model.body_jntadr[bid]
            qadr = self.model.jnt_qposadr[jid]
            pos = self.data.qpos[qadr:qadr + 3]
            if (np.linalg.norm(pos[:2] - MUG_XY) < MUG_INNER_R and pos[2] < MUG_RIM_Z):
                inside = True
                break
        self._success_count = self._success_count + 1 if inside else 0
        return self._success_count >= 10


# ---------------------------------------------------------------------------
# π0.5 策略运行时（复用训练 venv 中安装的 lerobot PI05Policy + processor 链）
# ---------------------------------------------------------------------------
class Pi05Runtime:
    def __init__(self, ckpt: str, device: str = "cuda"):
        import torch
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        from lerobot.processor.pipeline import DataProcessorPipeline

        self.torch = torch
        self.device = device
        self.policy = PI05Policy.from_pretrained(ckpt, local_files_only=True)
        self.policy.to(device)
        self.policy.eval()

        self.pre = DataProcessorPipeline.from_pretrained(
            ckpt, config_filename="policy_preprocessor.json",
            overrides={"device_processor": {"device": device}},
        )
        self.post = DataProcessorPipeline.from_pretrained(
            ckpt, config_filename="policy_postprocessor.json",
            overrides={"device_processor": {"device": "cpu"}},
        )

    def reset(self):
        self.policy.reset()

    def infer(self, obs_numpy: dict, prompt: str) -> np.ndarray:
        torch = self.torch
        batch = {
            "observation.images.image": torch.from_numpy(obs_numpy["image"]).permute(2, 0, 1)[None].float() / 255.0,
            "observation.images.image2": torch.from_numpy(obs_numpy["image2"]).permute(2, 0, 1)[None].float() / 255.0,
            "observation.state": torch.from_numpy(obs_numpy["state"])[None].float(),
            "task": [prompt],
        }
        batch = self.pre(batch)
        with torch.inference_mode():
            action = self.policy.select_action(batch)
        # DataProcessorPipeline 要求 EnvTransition(dict) 输入，不能直接喂裸 Tensor
        action = self.post({"action": action})
        if isinstance(action, dict):
            action = action["action"]
        return action.to("cpu").numpy()[0]


# ---------------------------------------------------------------------------
# 主循环
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="π0.5-LIBERO → UR5e MuJoCo OSC 桥接")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT, help="PI05 checkpoint 目录")
    ap.add_argument("--prompt", default="put the cube into the mug", help="语言指令")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=200, help="最多控制周期（20Hz，200=10s）")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--flip-images", action="store_true", help="图像 180° 翻转（对齐 LIBERO 训练约定）")
    ap.add_argument("--raw-pose", action="store_true",
                    help="关闭 UR5e→Franka 位姿/位置标定（A/B 对比用，默认开启对齐）")
    ap.add_argument("--record", action="store_true", help="录制 agentview 视频到 outputs/")
    ap.add_argument("--view", action="store_true",
                    help="打开 MuJoCo 实时查看器（需 glfw 后端：MUJOCO_GL=glfw，不能用 egl）")
    ap.add_argument("--dry-run", action="store_true",
                    help="不加载模型、不渲染，用脚本动作验证环境+IK 链路")
    args = ap.parse_args()

    if args.dry_run:
        args.record = False

    render = not args.dry_run
    env = UR5ePickPlaceEnv(render=render, flip_images=args.flip_images,
                           pose_align=not args.raw_pose)
    obs = env.reset(seed=args.seed)
    print(f"[env] 复位完成，state={obs['state'].round(3)}，方块数={len(env.cubes)}，"
          f"控制频率={CONTROL_HZ}Hz，子步={SUBSTEPS}")

    viewer = None
    if args.view:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(env.model, env.data)
        print("[view] MuJoCo 查看器已启动（空格暂停，鼠标拖拽调整视角）")

    frames = []
    if args.dry_run:
        # 脚本化动作：向前 → 下降 → 闭合 → 抬起 → 向杯 → 下降 → 张开（验证 IK/夹爪物理）
        plan = [
            (30, [0.6, 0, 0, 0, 0, 0, 1]),     # x 前移（张开）
            (20, [0, 0, -0.6, 0, 0, 0, 1]),    # 下降接近桌面（张开）
            (10, [0, 0, 0, 0, 0, 0, -1]),      # 闭合
            (25, [0, 0, 0.8, 0, 0, 0, -1]),    # 抬起（保持闭合）
            (30, [-0.3, -0.8, 0, 0, 0, 0, -1]),# 向杯方向（杯在 x0.2 y-0.25）
            (15, [0, 0, -0.7, 0, 0, 0, -1]),   # 下降入杯
            (10, [0, 0, 0, 0, 0, 0, 1]),       # 张开
        ]
        seq = []
        for n, a in plan:
            seq += [np.array(a, dtype=np.float32)] * n
        print(f"[dry-run] 执行 {len(seq)} 步脚本动作（无策略、无渲染）")
        for t, act in enumerate(seq):
            if viewer is not None and not viewer.is_running():
                break
            env.apply_action(act)
            if viewer is not None:
                viewer.sync()
            if t % 20 == 0:
                s = env.get_state()
                print(f"  t={t:3d} ee_xyz={s[:3].round(3)} grip_open={env.grip_open}")
        print(f"[dry-run] 成功判定: {env.success()}（脚本轨迹仅验证链路，不保证入杯）")
        if viewer is not None:
            input("[view] 回车关闭查看器...")
            viewer.close()
        env.close()
        return

    policy = Pi05Runtime(args.ckpt, device=args.device)
    policy.reset()
    print(f"[policy] {args.ckpt}")
    print(f"[policy] prompt: {args.prompt!r}")

    t0 = time.time()
    for t in range(args.steps):
        if viewer is not None and not viewer.is_running():
            break
        if render and args.record:
            img, _ = env.get_images()
            frames.append(img)
        act = policy.infer(obs, args.prompt)
        env.apply_action(act)
        obs = env.get_observation_numpy()
        if viewer is not None:
            viewer.sync()
        if t % 10 == 0:
            print(f"  t={t:3d} act={act.round(2)} ee={obs['state'][:3].round(3)} "
                  f"grip={'open' if env.grip_open else 'close'}")
        if env.success():
            print(f"[成功] 第 {t} 个控制周期（{t / CONTROL_HZ:.1f}s）方块入杯")
            break
    dt = time.time() - t0
    print(f"[done] {t + 1} 周期，墙钟 {dt:.1f}s，平均 {(t + 1) / max(dt, 1e-6):.1f} tick/s")

    if viewer is not None:
        input("[view] 回车关闭查看器...")
        viewer.close()
    env.close()

    if frames:
        OUT_DIR.mkdir(exist_ok=True)
        out = OUT_DIR / f"pi05_ur5e_{int(time.time())}.mp4"
        try:
            import imageio.v2 as imageio
            imageio.mimsave(out, frames, fps=CONTROL_HZ)
            print(f"[record] {out}")
        except Exception as e:
            print(f"[record] 视频保存失败（{e}），帧已在内存中")


if __name__ == "__main__":
    main()
