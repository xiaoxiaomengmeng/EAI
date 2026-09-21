#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UR5e 模型资产逐项验证（无窗口 / headless，系统 python3 即可运行）

对 RynnMotion/models/3.robot_arm/21.ur5e 下的资产做能力核查：
  1. 四个 MJCF 入口均可加载
  2. 拓扑：14 关节 / 7 actuator / body 数
  3. 总质量与各连杆质量
  4. 关节限位（elbow ±π，其余 ±2π）
  5. 伺服增益与力限（大臂 2000/±150Nm，腕部 500/±28Nm）
  6. 25 路传感器
  7. ur5e_robot.xml 的 3 个 keyframe
  8. 夹爪真实性：ctrl=255 闭合 vs ctrl=0 张开，指尖距离数值实验
  9. pickplace 场景：mug + 6 个 freejoint 方块
 10. 6 对自碰撞排除
 11. mesh 资产文件完整（20 OBJ + 8 STL）
 12. pinocchio 版：6 轴纯运动学、带 EE site、无夹爪
 13. 物理稳定性：给 home 位 ctrl 跑 1000 步不发散
 14. 离屏渲染（可选，需要 EGL/osmesa）

用法：
    python3 verify_ur5e.py
"""

import re
import sys
from pathlib import Path

import numpy as np
import mujoco

MODEL_DIR = Path(__file__).resolve().parent.parent / "RynnMotion/models/3.robot_arm/21.ur5e"
SCENE = MODEL_DIR / "scene/scene.xml"
PICKPLACE = MODEL_DIR / "scene/scene_pickplace.xml"
ROBOT = MODEL_DIR / "mjcf/ur5e_robot.xml"
PINO = MODEL_DIR / "mjcf/ur5e_pinocchio.xml"
ASSETS_XML = MODEL_DIR / "mjcf/ur5e_assets.xml"
ROBOTIQ_DIR = MODEL_DIR.parent.parent / "2.end_effector/11.robotiq/assets"

passed = 0
failed = 0


def check(name, ok, detail=""):
    global passed, failed
    if ok:
        passed += 1
        print(f"  PASS  {name}" + (f"  ({detail})" if detail else ""))
    else:
        failed += 1
        print(f"  FAIL  {name}" + (f"  ({detail})" if detail else ""))


# 各类对象数量属性名映射（MuJoCo 命名不统一）
_COUNT_ATTR = {
    "body": "nbody", "joint": "njnt", "actuator": "nu", "sensor": "nsensor",
    "site": "nsite", "camera": "ncam", "key": "nkey",
}
# 各类对象 mjtObj 枚举后缀
_TYPE_SUFFIX = {
    "body": "BODY", "joint": "JOINT", "actuator": "ACTUATOR", "sensor": "SENSOR",
    "site": "SITE", "camera": "CAMERA", "key": "KEY",
}


def find_id(model, objtype, prefix):
    """按名字前缀查对象 id（兼容 replicate 加 _0 后缀）。"""
    count = getattr(model, _COUNT_ATTR[objtype])
    enum_type = getattr(mujoco.mjtObj, f"mjOBJ_{_TYPE_SUFFIX[objtype]}")
    for i in range(count):
        n = mujoco.mj_id2name(model, enum_type, i)
        if n and (n == prefix or n.startswith(prefix)):
            return i, n
    return -1, None


print("=" * 60)
print("  UR5e 模型资产逐项验证")
print(f"  路径: {MODEL_DIR}")
print("=" * 60)

# ---------- 1. 四个入口可加载 ----------
print("\n[1] MJCF 入口加载")
models = {}
for label, path in [("scene", SCENE), ("pickplace", PICKPLACE), ("robot", ROBOT), ("pinocchio", PINO)]:
    try:
        models[label] = mujoco.MjModel.from_xml_path(str(path))
        check(f"{path.name} 可加载", True)
    except Exception as e:
        check(f"{path.name} 可加载", False, str(e)[:80])

m = models.get("scene")
if m is None:
    sys.exit("主场景加载失败，后续检查终止")

# ---------- 2. 拓扑 ----------
print("\n[2] 拓扑结构")
n_gripper_jnt = sum(
    1 for i in range(m.njnt)
    if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) or "").startswith(("left", "right"))
)
check("6 臂关节 + 8 夹爪铰链 = 14 关节", m.njnt == 14, f"实际 {m.njnt}（夹爪 {n_gripper_jnt}）")
check("7 个 actuator（6 臂 + 1 grip）", m.nu == 7, f"实际 {m.nu}")
check("body 数 = 23", m.nbody == 23, f"实际 {m.nbody}")
check("split 腱传动 1 条", m.ntendon == 1, f"实际 {m.ntendon}")

# ---------- 3. 质量 ----------
print("\n[3] 动力学参数")
total_mass = float(sum(m.body_mass))
check("总质量 ≈ 22.05 kg", abs(total_mass - 22.05) < 0.5, f"{total_mass:.2f} kg")
base_id, _ = find_id(m, "body", "base")
check("基座质量 4.0 kg", abs(m.body_mass[base_id] - 4.0) < 0.01, f"{m.body_mass[base_id]:.3f}")

# ---------- 4. 关节限位 ----------
print("\n[4] 关节限位")
for prefix, expect_half in [("shoulder_pan_joint", 2 * np.pi), ("elbow_joint", np.pi),
                            ("wrist_3_joint", 2 * np.pi)]:
    jid, name = find_id(m, "joint", prefix)
    lo, hi = m.jnt_range[jid]
    check(f"{name} 限位 ±{expect_half/np.pi:.0f}π",
          abs(lo + expect_half) < 0.01 and abs(hi - expect_half) < 0.01,
          f"[{lo:.2f}, {hi:.2f}]")

# ---------- 5. 伺服参数 ----------
print("\n[5] 伺服增益 / 力限")
for prefix, kp, fmax in [("shoulder_pan", 2000, 150), ("wrist_1", 500, 28)]:
    aid, name = find_id(m, "actuator", prefix)
    check(f"{name} kp={kp} 力限 ±{fmax}Nm",
          abs(m.actuator_gainprm[aid, 0] - kp) < 1
          and abs(m.actuator_forcerange[aid, 1] - fmax) < 0.1,
          f"kp={m.actuator_gainprm[aid,0]:.0f} fmax={m.actuator_forcerange[aid,1]:.0f}")
gid, gname = find_id(m, "actuator", "grip")
cr = m.actuator_ctrlrange[gid]
check(f"{gname} 控制量 0–255（Robotiq 协议）", cr[0] == 0 and cr[1] == 255, f"[{cr[0]},{cr[1]}]")

# ---------- 6. 传感器 ----------
print("\n[6] 传感器（25 路）")
snames = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SENSOR, i) for i in range(m.nsensor)]
check("传感器总数 25", m.nsensor == 25, f"实际 {m.nsensor}")
for s in ["eePos", "eeQuat", "eeForce", "eeGyro", "elbow_joint_pos", "wrist_3_joint_vel",
          "shoulder_pan_frc"]:
    sid, real = find_id(m, "sensor", s)
    check(f"传感器 {s}", sid >= 0)

# ---------- 7. keyframes（仅 robot.xml） ----------
print("\n[7] 预设关键位")
mr = models["robot"]
check("ur5e_robot.xml 含 3 个 keyframe", mr.nkey == 3,
      f"{[mujoco.mj_id2name(mr, mujoco.mjtObj.mjOBJ_KEY, i) for i in range(mr.nkey)]}")
check("scene.xml 无 keyframe", m.nkey == 0, f"nkey={m.nkey}")

# ---------- 8. 夹爪开合数值实验 ----------
print("\n[8] 夹爪机构（0–255 驱动四连杆 + 弹簧回位）")

def pad_gap(model, grip_ctrl, steps=2000):
    d = mujoco.MjData(model)
    mujoco.mj_forward(model, d)
    # 臂关节保持当前角，避免塌落
    for p in ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]:
        aid, _ = find_id(model, "actuator", p)
        jid = model.actuator_trnid[aid, 0]
        d.ctrl[aid] = d.qpos[model.jnt_qposadr[jid]]
    d.ctrl[gid] = grip_ctrl
    for _ in range(steps):
        mujoco.mj_step(model, d)
    lp, _ = find_id(model, "body", "left_pad")
    rp, _ = find_id(model, "body", "right_pad")
    return float(np.linalg.norm(d.xpos[lp] - d.xpos[rp]))

gaps = {c: pad_gap(m, c) for c in (0, 100, 200, 255)}
for c, g in gaps.items():
    print(f"       ctrl={c:3d}  指垫间距 = {g*1000:6.1f} mm")
stroke = abs(gaps[255] - gaps[0])
check(f"夹爪全行程 > 50mm（实测 {stroke*1000:.1f} mm）", stroke > 0.05)
closed_ctrl = min(gaps, key=gaps.get)
open_ctrl = max(gaps, key=gaps.get)
print(f"       语义：ctrl={closed_ctrl} 闭合（弹簧回位），ctrl={open_ctrl} 张开（腱拉力克服弹簧）")

# ---------- 9. pickplace 场景物体 ----------
print("\n[9] pick-place 任务场景")
mp = models["pickplace"]
free_bodies = [
    mujoco.mj_id2name(mp, mujoco.mjtObj.mjOBJ_BODY, i)
    for i in range(mp.nbody)
    if mp.body_jntnum[i] > 0
    and mp.jnt_type[mp.body_jntadr[i]] == mujoco.mjtJoint.mjJNT_FREE
]
check("6 个 freejoint 方块（可抓取）", len(free_bodies) == 6, f"{free_bodies}")
mug_id, mug_name = find_id(mp, "body", "mug")
check("马克杯为固定道具（无关节）", mug_id >= 0 and mp.body_jntnum[mug_id] == 0,
      "static" if mug_id >= 0 else "缺失")
cam_id, cam_name = find_id(mp, "camera", "ext_camera")
check("1080p 固定机位", cam_id >= 0 and mp.cam_resolution[cam_id, 1] == 1080,
      f"{mp.cam_resolution[cam_id].tolist() if cam_id>=0 else 'N/A'}")

# ---------- 10. 接触排除 ----------
print("\n[10] 自碰撞排除")
check("6 对内部接触排除", m.nexclude == 6, f"实际 {m.nexclude}")

# ---------- 11. mesh 资产 ----------
print("\n[11] 网格资产文件")
obj_files = list((MODEL_DIR / "assets").glob("*.obj"))
check("臂部 20 个 OBJ", len(obj_files) == 20, f"实际 {len(obj_files)}")
xml_text = ASSETS_XML.read_text(encoding="utf-8")
stl_names = re.findall(r'file="[^"]*?([\w_]+\.stl)"', xml_text)
missing = [n for n in stl_names if not (ROBOTIQ_DIR / n).exists()]
check(f"夹爪 {len(stl_names)} 个 STL（引用 11.robotiq）", len(missing) == 0,
      "全部存在" if not missing else f"缺失 {missing}")

# ---------- 12. pinocchio 纯运动学版 ----------
print("\n[12] Pinocchio 运动学模型")
mpo = models["pinocchio"]
arm_jnts_pino = [i for i in range(mpo.njnt)]
ee_pino, _ = find_id(mpo, "site", "EE")
check("仅 6 轴、无夹爪", mpo.njnt == 6, f"njnt={mpo.njnt} nu={mpo.nu}")
check("含 EE site", ee_pino >= 0)

# ---------- 13. 物理稳定性 ----------
print("\n[13] 物理稳定性（home 位保持 1000 步）")
d = mujoco.MjData(mr)
mujoco.mj_resetDataKeyframe(mr, d, 0)
mujoco.mj_forward(mr, d)
ee0 = d.site_xpos[find_id(mr, "site", "EE")[0]].copy()
for _ in range(1000):
    mujoco.mj_step(mr, d)
ee1 = d.site_xpos[find_id(mr, "site", "EE")[0]].copy()
drift = float(np.linalg.norm(ee1 - ee0))
check("home 位 1 秒末端漂移 < 1mm", drift < 0.001, f"{drift*1000:.3f} mm")

# ---------- 14. 离屏渲染（可选） ----------
print("\n[14] 离屏渲染（可选）")
try:
    import os
    os.environ.setdefault("MUJOCO_GL", "egl")
    renderer = mujoco.Renderer(m, height=360, width=640)
    dd = mujoco.MjData(m)
    mujoco.mj_forward(m, dd)
    renderer.update_scene(dd)
    img = renderer.render()
    check("EGL 离屏渲染成功", True, f"图像 {img.shape}")
    # 存盘是可选项，不强制依赖 imageio/PIL
    out = Path(__file__).parent / "outputs/ur5e_render.png"
    out.parent.mkdir(exist_ok=True)
    saved = False
    try:
        import imageio.v2 as imageio
        imageio.imwrite(out, img)
        saved = True
    except Exception:
        try:
            from PIL import Image
            Image.fromarray(img).save(out)
            saved = True
        except Exception:
            pass
    if saved:
        print(f"       已保存: {out}")
except Exception as e:
    check("EGL 离屏渲染（跳过）", True, f"不可用: {str(e)[:60]}")

# ---------- 汇总 ----------
print("\n" + "=" * 60)
print(f"  结果：{passed} PASS / {failed} FAIL")
print("=" * 60)
sys.exit(1 if failed else 0)
