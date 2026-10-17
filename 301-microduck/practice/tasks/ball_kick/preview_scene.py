"""场景预览（headless 离屏渲染）：把「细桩 tee + 球 + 机器人脚」渲成一张多角度拼图。

为什么需要它：本机通常**无 DISPLAY**（headless），mjlab 的 `native` viewer 起不来；
`viser` 网页 viewer 在官方 play 管线的 velocity-command slider 处有 assert bug 会崩。
所以最稳的"看一眼场景几何"方式是 **EGL 离屏渲染成 PNG**，直接读图即可。

做法（工程点）：
  - `render_mode="rgb_array"` 建 env → mjlab 内部建 `OffscreenRenderer`（EGL）；
  - `num_envs=1` + `max_extra_envs=0`：画面只有单个场景，干净不杂；
  - reset 后读球的世界坐标（含 env_origin 偏移），把相机 `lookat` 对准球；
    相机对象是 `env._offline_renderer._cam`（私有但稳定），reset 后改它即可，
    因为 `render()` 每次都用 `self._cam` 重新 `update_scene`；
  - 机器人朝向随机（set_random_ground_state 随机 yaw），故从 4 个方位角各渲一帧
    拼成 2×2，确保无论朝向都能看清细桩 + 球 + 脚。

用法（本目录下）：
    MUJOCO_GL=egl ../../../.venv/bin/python preview_scene.py [输出png路径]
默认输出 ./preview_scene.png（已被 .gitignore 忽略，属可再生产物）。
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")  # headless 渲染：必须在 import mujoco 前

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.viewer.viewer_config import ViewerConfig  # noqa: E402

import high_kick_cfg as C  # noqa: E402

DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "preview_scene.png")
AZIMUTHS = (90, 180, 270, 0)  # 四个方位角，覆盖随机朝向


def main(out_path: str = DEFAULT_OUT) -> None:
    cfg = C.make_high_kick_env_cfg(play=True)
    cfg.scene.num_envs = 1
    # 单场景、近距离、略俯视，框住细桩(0~0.045) + 球(球心 0.08) + 脚。
    cfg.viewer = ViewerConfig(
        origin_type=ViewerConfig.OriginType.WORLD,
        lookat=(0.0, 0.0, 0.05),  # reset 后按球实际世界坐标覆盖
        distance=0.32,
        elevation=-18.0,
        azimuth=90.0,
        height=360,
        width=480,
        max_extra_envs=0,
        enable_reflections=False,
        enable_shadows=True,
    )

    env = ManagerBasedRlEnv(cfg, device="cuda:0", render_mode="rgb_array")
    env.reset()
    for _ in range(5):  # settle，让球稳坐桩顶
        env.step(torch.zeros(1, env.action_manager.total_action_dim, device=env.device))

    bp = env.scene["ball"].data.root_link_pos_w[0].cpu().numpy()
    cam = env._offline_renderer._cam
    cam.lookat[:] = [float(bp[0]), float(bp[1]), 0.05]  # 对准球，z 压到桩中段
    print("ball world pos =", np.round(bp, 4).tolist())

    frames = []
    for az in AZIMUTHS:
        cam.azimuth = az
        frames.append(np.asarray(env.render()))

    h, w = frames[0].shape[:2]
    grid = Image.new("RGB", (w * 2, h * 2))
    for i, f in enumerate(frames):
        grid.paste(Image.fromarray(f), ((i % 2) * w, (i // 2) * h))
    grid.save(out_path)
    print("saved:", out_path, "size=", grid.size)
    env.close()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT)
