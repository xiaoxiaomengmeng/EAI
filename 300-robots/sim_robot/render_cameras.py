#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
相机链路验证：为 UR5e pickplace 场景增加腕部相机，渲染两路观测图像。

- base 视角：场景已有的第三人称固定机位 camera（≈ LIBERO agentview）
- wrist 视角：运行时注入到 wrist_3_link 上的 eye-in-hand 相机（朝法兰前方 +z）

不改 RynnMotion 原始 XML，注入方式与 view_ur5e.py 的 mocap 注入一致。

用法：
    MUJOCO_GL=egl python3 render_cameras.py        # 无窗口离屏渲染（服务器）
    python3 render_cameras.py                      # 默认尝试 EGL
输出：outputs/cam_base.png / cam_wrist.png（256×256，LIBERO 模型输入尺寸）
"""

from pathlib import Path
import sys

import numpy as np
import mujoco

MODEL_DIR = Path(__file__).resolve().parent.parent / "RynnMotion/models/3.robot_arm/21.ur5e"
SCENE = MODEL_DIR / "scene/scene_pickplace.xml"
OUT_DIR = Path(__file__).resolve().parent / "outputs"

# LIBERO: 256×256；pi0_base: 224×224
SIZE_LIBERO = 256
SIZE_PI0 = 224

WRIST_CAMERA_XML = (
    '<camera name="wrist_camera" pos="0 0 0.12" euler="0 3.14159265 0" '
    'fovy="60" resolution="256 256" sensorsize="5.37 4.04" focal="2.9 2.9" />'
)


def load_with_wrist_camera(scene_path: Path):
    """用 MjSpec 在 wrist_3_link body 上挂随动相机（无需改任何原始 XML）。"""
    spec = mujoco.MjSpec.from_file(str(scene_path))
    try:
        body = spec.body("wrist_3_link")
    except Exception:
        raise RuntimeError("找不到 wrist_3_link body")
    if not any(c.name == "wrist_camera" for c in body.cameras):
        body.add_camera(
            name="wrist_camera",
            pos=[0.0, 0.0, 0.12],
            euler=[0.0, np.pi, 0.0],
            fovy=60,
            resolution=[256, 256],
        )
    return spec.compile()


def save_png(img: np.ndarray, path: Path):
    path.parent.mkdir(exist_ok=True)
    try:
        from PIL import Image
        Image.fromarray(img).save(path)
    except ImportError:
        import imageio.v2 as imageio
        imageio.imwrite(path, img)
    print(f"  已保存 {path} ({img.shape[1]}x{img.shape[0]})")


def main():
    print(f"加载场景: {SCENE}")
    model = load_with_wrist_camera(SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    # 给一个合理的悬停姿态（无 keyframe，直接设目标角后伺服到位）
    home = np.array([0.0, -1.6, 1.7, -1.7, -1.57, 0.0])
    arm_ids = []
    for p in ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]:
        for i in range(model.nu):
            n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if n and n.startswith(p):
                arm_ids.append(i)
    grip_id = next(
        i for i in range(model.nu)
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i).startswith("grip")
    )
    data.ctrl[arm_ids] = home
    data.ctrl[grip_id] = 255  # 张开
    for _ in range(3000):
        mujoco.mj_step(model, data)

    def find_cam(prefix):
        for i in range(model.ncam):
            n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            if n and (n == prefix or n.startswith(prefix)):
                return i, n
        return -1, None

    cam_base, n_base = find_cam("ext_camera")  # pickplace 场景的固定第三人称机位
    cam_wrist, n_wrist = find_cam("wrist_camera")
    assert cam_base >= 0 and cam_wrist >= 0, "相机注入失败"
    print(f"相机就绪: {n_base}(base)={cam_base}, {n_wrist}(wrist)={cam_wrist}")

    try:
        renderer = mujoco.Renderer(model, height=SIZE_LIBERO, width=SIZE_LIBERO)
    except Exception as e:
        sys.exit(f"Renderer 创建失败（尝试 MUJOCO_GL=egl 或在桌面运行）: {e}")

    # base 视角
    renderer.update_scene(data, camera=n_base)
    img_base = renderer.render()
    save_png(img_base, OUT_DIR / "cam_base_256.png")

    # wrist 视角
    renderer.update_scene(data, camera=n_wrist)
    img_wrist = renderer.render()
    save_png(img_wrist, OUT_DIR / "cam_wrist_256.png")

    # 简单内容自检：两张图不应全黑/全同色
    for name, img in [("base", img_base), ("wrist", img_wrist)]:
        std = float(img.std())
        print(f"  {name} 图像像素标准差 = {std:.1f}（>10 说明有内容）")

    print("\n下一步：目视检查两张图，base 应看到桌面+方块+杯，wrist 应是夹爪朝前的第一人称视角。")


if __name__ == "__main__":
    main()
