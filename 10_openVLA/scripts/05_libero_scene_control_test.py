"""阶段⑤(M3) · OSC 等价控制保真测试 (自建 libero_spatial 场景, 不经模型)。

用合成 7D action (LIBERO env 约定: 前6维归一化增量 ∈[-1,1], 末维 ±1 夹爪) 开环驱动
sim/libero_controller.py 的 LiberoOscEquivalentController, 逐项校验:

  A. 位置增量: 重复 [+x] action -> EE 沿世界 +x 累计位移 ≈ n * a * 0.05 m
  B. 位置增量: 重复 [-z] action -> EE 沿世界 -z 累计位移
  C. 姿态增量: 重复 [yaw] action -> EE 绕世界 z 累计旋转 ≈ n * a * 0.5 rad (世界系左乘)
  D. 夹爪映射: +1 -> 闭 (finger_joint1→0), -1 -> 开 (→0.04)
  E. 每段结束打印 tracking_error (EE 与 goal 差距), 校验跟踪收敛

用法:
    cd /home/creator/ws/10_openVLA && source .venv/bin/activate
    MUJOCO_GL=egl python scripts/05_libero_scene_control_test.py
"""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import mujoco
import numpy as np

from sim.libero_controller import (
    OSC_OUTPUT_MAX,
    PHYSICS_STEPS_PER_ACTION,
    LiberoOscEquivalentController,
)
from sim.libero_obs import LIBERO_DUMMY_ACTION

SCENE_XML = Path(__file__).resolve().parents[1] / "sim" / "scenes" / "libero_spatial_scene.xml"
OUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "libero_scene_control_test"

REPEAT = 5  # 每段合成 action 重复次数 (LIBERO 里即连续 5 个 policy step 同动作)


def log(msg: str):
    print(f"[M3] {msg}", flush=True)


def rot_angle_axis(r0: np.ndarray, r1: np.ndarray):
    """r0->r1 的旋转角与轴 (世界系): R = r1 @ r0.T。"""
    r = r1 @ r0.T
    cos = (np.trace(r) - 1.0) / 2.0
    angle = float(np.arccos(np.clip(cos, -1.0, 1.0)))
    if angle < 1e-9:
        return 0.0, np.array([0.0, 0.0, 1.0])
    axis = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]]) / (2 * np.sin(angle))
    return angle, axis


def run_segment(ctrl: LiberoOscEquivalentController, action7, repeat: int) -> dict:
    p0, r0 = ctrl.ee_pos, ctrl.ee_rot
    for _ in range(repeat):
        ctrl.set_action(action7)
        ctrl.step_action()
    p1, r1 = ctrl.ee_pos, ctrl.ee_rot
    angle, axis = rot_angle_axis(r0, r1)
    return {
        "dpos": (p1 - p0).tolist(),
        "dpos_norm": float(np.linalg.norm(p1 - p0)),
        "drot_rad": angle,
        "drot_axis": axis.tolist(),
        "tracking": ctrl.tracking_error(),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    ctrl = LiberoOscEquivalentController(model, data)

    # 沉降: num_steps_wait=10 个 control step 的 dummy action (机器人不动, 夹爪开)
    log(f"沉降 {10} 个 control step (dummy={LIBERO_DUMMY_ACTION})...")
    for _ in range(10):
        ctrl.set_action(np.array(LIBERO_DUMMY_ACTION, dtype=np.float64))
        ctrl.step_action()
    ctrl.reset_goal_to_current()
    log(f"沉降后 EE pos={np.round(ctrl.ee_pos, 4).tolist()}")

    results = {}
    checks = {}

    # A: +x 位置增量, 期望累计 REPEAT*0.4*0.05 = 0.10 m
    exp = REPEAT * 0.4 * OSC_OUTPUT_MAX[0]
    res = run_segment(ctrl, np.array([0.4, 0, 0, 0, 0, 0, -1.0]), REPEAT)
    results["A_pos_plus_x"] = res
    got = res["dpos"][0]
    checks["A"] = bool(got > 0.6 * exp and res["dpos_norm"] < 1.5 * exp + 1e-3)
    log(f"A +x: 期望≈{exp:.3f}m, 实测 dx={got:.4f} (总位移 {res['dpos_norm']:.4f}), 跟踪误差={res['tracking']} -> {'PASS' if checks['A'] else 'FAIL'}")

    # B: -z 位置增量
    exp = REPEAT * 0.4 * OSC_OUTPUT_MAX[2]
    res = run_segment(ctrl, np.array([0, 0, -0.4, 0, 0, 0, -1.0]), REPEAT)
    results["B_pos_minus_z"] = res
    got = -res["dpos"][2]
    checks["B"] = bool(got > 0.6 * exp and res["dpos_norm"] < 1.5 * exp + 1e-3)
    log(f"B -z: 期望≈{exp:.3f}m, 实测 -dz={got:.4f}, 跟踪误差={res['tracking']} -> {'PASS' if checks['B'] else 'FAIL'}")

    # C: 绕世界 z 姿态增量, 期望累计 REPEAT*0.4*0.5 = 1.0 rad, 轴≈[0,0,±1]
    exp = REPEAT * 0.4 * OSC_OUTPUT_MAX[5]
    res = run_segment(ctrl, np.array([0, 0, 0, 0, 0, 0.4, -1.0]), REPEAT)
    results["C_yaw_world_z"] = res
    axis_z = abs(res["drot_axis"][2])
    checks["C"] = bool(res["drot_rad"] > 0.6 * exp and axis_z > 0.9)
    log(f"C yaw: 期望≈{exp:.3f}rad, 实测={res['drot_rad']:.4f}rad, 轴z分量={axis_z:.3f} -> {'PASS' if checks['C'] else 'FAIL'}")

    # D: 夹爪 闭->开 (指关节 damping=100+frictionloss, 闭合约需 0.2s≈4 个 control step,
    #    与 robosuite 指动力学一致, 故各给 10 个 control step 再断言)
    for _ in range(10):
        ctrl.set_action(np.array([0, 0, 0, 0, 0, 0, 1.0]))
        ctrl.step_action()
    q_close = data.qpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1")]
    for _ in range(10):
        ctrl.set_action(np.array([0, 0, 0, 0, 0, 0, -1.0]))
        ctrl.step_action()
    q_open = data.qpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1")]
    results["D_gripper"] = {"q_close": float(q_close), "q_open": float(q_open)}
    checks["D"] = bool(q_close < 0.02 and q_open > 0.03)
    log(f"D gripper: 闭后 finger1={q_close:.4f} (<0.02), 开后={q_open:.4f} (>0.03) -> {'PASS' if checks['D'] else 'FAIL'}")

    # E: 物体未被 IK 误驱动 (freejoint 应保持原位附近)
    bowl = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "akita_black_bowl_1")]
    results["E_bowl_still"] = bowl.tolist()
    checks["E"] = bool(abs(bowl[0]) < 0.01 and abs(bowl[1]) < 0.01)
    log(f"E 碗未漂移: pos={np.round(bowl, 4).tolist()} -> {'PASS' if checks['E'] else 'FAIL'}")

    payload = {
        "physics_steps_per_action": PHYSICS_STEPS_PER_ACTION,
        "repeat_per_segment": REPEAT,
        "results": results,
        "checks": checks,
        "all_pass": bool(all(checks.values())),
    }
    (OUT_DIR / "control_test.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log(f"落盘: {OUT_DIR / 'control_test.json'}")
    log(f"总结: {'全部 PASS' if payload['all_pass'] else '存在 FAIL'} -> {checks}")


if __name__ == "__main__":
    main()
