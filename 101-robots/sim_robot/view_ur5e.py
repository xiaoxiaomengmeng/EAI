#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UR5e MuJoCo 独立查看脚本（不依赖 RynnMotion C++ 编译）

直接复用 RynnMotion 仓库中的 UR5e + Robotiq 2F85 模型：
    ../RynnMotion/models/3.robot_arm/21.ur5e/scene/scene.xml

用法：
    python3 view_ur5e.py              # 交互模式：物理仿真运行，右侧 Control 面板可拖滑条
    python3 view_ur5e.py --demo       # 演示模式：6 个关节正弦摆动
    python3 view_ur5e.py --drag       # 拖拽模式：拖动红色目标球，机械臂末端实时 IK 跟随
    python3 view_ur5e.py --paused     # 启动即暂停，纯静态查看/拖拽
    python3 view_ur5e.py --scene pickplace   # 加载抓取场景（马克杯 + 6 方块）

拖拽模式操作：
    双击红色小球选中它，然后按住 Ctrl + 鼠标左键拖动（平移）
    机械臂末端会通过阻尼最小二乘 IK 实时跟随目标球

通用交互：
    鼠标              旋转/平移视角
    右侧 Control 面板 直接拖动 actuator 滑条控制 6 关节 + 夹爪（非 drag 模式）
    空格              暂停 / 继续仿真
    R                 重置到初始位姿
    ESC / 关窗口      退出

依赖：mujoco（pip install mujoco），系统 python3 已自带。
注意：需要图形桌面环境（有 DISPLAY），纯 SSH 终端无法弹出窗口。
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

import mujoco
import mujoco.viewer

# scene.xml 相对本脚本的路径：
# 101-robots/sim_robot/view_ur5e.py -> 101-robots/RynnMotion/models/...
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = (
    SCRIPT_DIR.parent
    / "RynnMotion"
    / "models"
    / "3.robot_arm"
    / "21.ur5e"
)
SCENES = {
    "basic": MODEL_DIR / "scene/scene.xml",
    "pickplace": MODEL_DIR / "scene/scene_pickplace.xml",
}

# UR5e 6 个机械臂关节（夹爪 grip actuator 不参与摆动演示）
ARM_JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
]


def parse_args():
    p = argparse.ArgumentParser(description="UR5e MuJoCo 独立查看脚本")
    p.add_argument("--demo", action="store_true", help="关节正弦摆动演示")
    p.add_argument("--drag", action="store_true", help="末端拖拽 IK 跟随模式")
    p.add_argument("--paused", action="store_true", help="启动即暂停")
    p.add_argument("--scene", choices=list(SCENES), default="basic",
                   help="场景：basic（默认）或 pickplace（马克杯+方块）")
    return p.parse_args()


def load_model_with_mocap(scene_xml: Path):
    """在 scene.xml 的 worldbody 中注入一个 mocap 目标球，写入同目录临时文件后加载。

    必须放在 scene 同目录，include 的 ../mjcf/*.xml 相对路径才能解析。
    """
    xml_text = scene_xml.read_text(encoding="utf-8")
    mocap_body = (
        '  <body mocap="true" name="ee_target" pos="0.5 0 0.5">\n'
        '    <geom type="sphere" size="0.02" rgba="1 0.1 0.1 0.8" '
        'contype="0" conaffinity="0"/>\n'
        "  </body>\n"
    )
    if "<body mocap" not in xml_text:
        xml_text = xml_text.replace("</worldbody>", mocap_body + "</worldbody>")
    tmp_xml = scene_xml.parent / ".scene_drag_tmp.xml"
    tmp_xml.write_text(xml_text, encoding="utf-8")
    try:
        model = mujoco.MjModel.from_xml_path(str(tmp_xml))
    finally:
        tmp_xml.unlink(missing_ok=True)
    return model


def main():
    args = parse_args()

    scene_xml = SCENES[args.scene]
    if not scene_xml.exists():
        sys.exit(f"找不到模型文件: {scene_xml}")

    # drag 模式需要注入 mocap 目标；其余模式直接加载原 scene
    model = load_model_with_mocap(scene_xml) if args.drag else mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0) if model.nkey > 0 else mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    print(f"模型加载成功: {scene_xml.name}（场景: {args.scene}）")
    print(f"  nq={model.nq}  nv={model.nv}  nu={model.nu}  nbody={model.nbody}")

    # 取臂关节 actuator 地址
    # 注意：scene.xml 用了 <replicate>，actuator 名会被加后缀，如 shoulder_pan -> shoulder_pan_0
    all_act_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)
    ]
    arm_adrs = []
    for name in ARM_JOINTS:
        adr = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if adr < 0:  # 兜底：匹配 replicate 加后缀的名字
            for real_name in all_act_names:
                if real_name and real_name.startswith(name):
                    adr = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, real_name)
                    break
        if adr >= 0:
            arm_adrs.append(adr)
    print(f"  找到 {len(arm_adrs)}/6 个臂关节 actuator")

    # ---- drag 模式：定位 EE site、mocap 目标、臂关节自由度地址 ----
    ee_site_id = -1
    if args.drag:
        for i in range(model.nsite):
            sname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
            if sname == "EE_0" or (sname or "").startswith("EE"):
                ee_site_id = i
                break
        mocap_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ee_target")
        mocap_id = model.body_mocapid[mocap_body]
        # 每个臂 actuator 绑定的 joint id -> qpos / dof 地址
        arm_jids = [model.actuator_trnid[a, 0] for a in arm_adrs]
        arm_qadr = np.array([model.jnt_qposadr[j] for j in arm_jids])
        arm_dadr = np.array([model.jnt_dofadr[j] for j in arm_jids])
        jnt_lo = model.jnt_range[arm_jids, 0]
        jnt_hi = model.jnt_range[arm_jids, 1]

    start_time = time.time()

    def reset_pose():
        """重置关节，drag 模式下同时把目标球对齐到当前末端位置。"""
        nonlocal start_time
        if model.nkey > 0:
            mujoco.mj_resetDataKeyframe(model, data, 0)
        else:
            mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        if args.drag:
            data.mocap_pos[mocap_id] = data.site_xpos[ee_site_id].copy()
            # ctrl 对齐当前关节角，位置伺服不会猛拽
            for a, qa in zip(arm_adrs, arm_qadr):
                data.ctrl[a] = data.qpos[qa]
        start_time = time.time()

    reset_pose()

    if args.drag:
        print("  拖拽模式：双击红色目标球选中，Ctrl + 左键拖动；机械臂末端 IK 跟随")

    paused = args.paused
    last_seconds = time.time()

    def key_callback(keycode):
        nonlocal paused
        # glfw 键码：空格=32, R=82
        if keycode == 32:
            paused = not paused
        elif keycode == 82:
            reset_pose()

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            step_start = time.time()

            if not paused:
                if args.drag and ee_site_id >= 0:
                    # 阻尼最小二乘 IK：末端位置 -> 6 关节角
                    target = data.mocap_pos[mocap_id]
                    for _ in range(3):  # 每个物理步迭代 3 次，跟随更跟手
                        mujoco.mj_forward(model, data)
                        jacp = np.zeros((3, model.nv))
                        mujoco.mj_jacSite(model, data, jacp, None, ee_site_id)
                        err = target - data.site_xpos[ee_site_id]
                        J = jacp[:, arm_dadr]                    # 3x6
                        damp = 0.05
                        dq = J.T @ np.linalg.solve(
                            J @ J.T + damp ** 2 * np.eye(3), err
                        )
                        dq = np.clip(dq, -0.1, 0.1)
                        q = data.qpos[arm_qadr] + dq
                        data.qpos[arm_qadr] = np.clip(q, jnt_lo, jnt_hi)
                    # 位置伺服 actuator：ctrl 即目标关节角
                    for a, qa in zip(arm_adrs, arm_qadr):
                        data.ctrl[a] = data.qpos[qa]
                elif args.demo and arm_adrs:
                    # 6 关节错相位正弦摆动，幅度 0.5 rad
                    t = time.time() - start_time
                    for i, adr in enumerate(arm_adrs):
                        data.ctrl[adr] = 0.5 * math.sin(0.8 * t + i * 0.6)
                mujoco.mj_step(model, data)

            # 约 30Hz 同步渲染即可，物理 timestep=1ms 内部步进
            if time.time() - last_seconds > 1.0 / 30:
                viewer.sync()
                last_seconds = time.time()

            # 按物理时钟节奏等待
            elapsed = time.time() - step_start
            sleep_for = model.opt.timestep - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)


if __name__ == "__main__":
    main()
