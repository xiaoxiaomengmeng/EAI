"""headless 回放训练好的策略 → mp4（H.264）：真正「看到」microduck 把球挑起来。

为什么需要它（和 preview_scene.py 的分工）：
  - `play_high_kick.py` 复用官方 `run_play`，但它最后**必进 `viewer.run()`**：headless 下
    `native` 要 DISPLAY 起不来、`viser` 在 velocity-command slider 处有 assert bug 会崩；
    即使 `--video True`，录像也是被 viewer 循环驱动的，headless 依旧跑不通。
  - `preview_scene.py` 只出**静态几何拼图**（随机/零动作，球不动），看不到「踢球动作」。
  - 本脚本用**训练好的 checkpoint** 驱动策略 rollout，逐帧 EGL 离屏渲染，叠加球高标注，
    导出**可直接播放的 H.264 mp4** —— 这是 headless 机器上「如何 Play」的可行路径。

做法（工程点）：
  - `render_mode="rgb_array"` 建单环境（`num_envs=1`，画面干净）；
  - `RslRlVecEnvWrapper` 包一层喂 runner，`MicroduckOnPolicyRunner.load(ckpt)` 取推理策略；
  - 相机固定**侧视角**框住 0~0.30m 竖直范围（球峰值约 0.21m），`lookat` 对准球的初始 xy；
  - 每帧用 PIL 叠加 `step / 球相对起始高 Δh(cm) / 峰值(cm)`（对齐「实时图像 + 标注」偏好）；
  - imageio + 内置 ffmpeg 写 `libx264 / yuv420p`（规避 AV1 编码播放器打不开的坑）。

用法（本目录下）：
    MUJOCO_GL=egl ../../../.venv/bin/python play_to_video.py [checkpoint.pt] [输出.mp4]
默认 checkpoint = 大训练最新 `model_23750.pt`，默认输出 `./play_high_kick.mp4`（已 gitignore）。
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")  # headless 渲染：必须在 import mujoco 前

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataclasses import asdict  # noqa: E402

import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.viewer.viewer_config import ViewerConfig  # noqa: E402

import high_kick_cfg as C  # noqa: E402
from mjlab_microduck.tasks import MicroduckOnPolicyRunner  # noqa: E402
from train_high_kick import register_high_kick_task  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CKPT = os.path.join(
    HERE,
    "logs/rsl_rl/ball_kick_aggr_right/2026-10-17_01-15-28_ball_kick_aggr_right/model_23750.pt",
)
DEFAULT_OUT = os.path.join(HERE, "play_high_kick.mp4")
STEPS = 300          # 录多少步（回合长约 230，300 步能看到 1 次完整挑球 + 重开）
FPS = 50             # 与控制频率一致，播放为实时速度
DEVICE = "cuda:0"


def _font(size: int = 18):
    try:
        return ImageFont.load_default(size=size)   # PIL>=10 支持带字号的默认字体
    except TypeError:
        return ImageFont.load_default()


def main(ckpt: str = DEFAULT_CKPT, out: str = DEFAULT_OUT) -> None:
    register_high_kick_task()  # 保证 registry 里有我们的 task（回放管线一致性）
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"checkpoint 不存在：{ckpt}")

    cfg = C.make_high_kick_env_cfg(play=True)
    cfg.scene.num_envs = 1
    # 侧视角、框住 0~0.30m 竖直范围，能完整看到球从桩顶(0.08)被挑到峰值(~0.21)。
    cfg.viewer = ViewerConfig(
        origin_type=ViewerConfig.OriginType.WORLD,
        lookat=(0.0, 0.0, 0.11),   # reset 后按球实际 xy 覆盖
        distance=0.55,
        elevation=-12.0,
        azimuth=90.0,
        height=480,
        width=640,
        max_extra_envs=0,
        enable_reflections=False,
        enable_shadows=True,
    )

    raw = ManagerBasedRlEnv(cfg, device=DEVICE, render_mode="rgb_array")
    env = RslRlVecEnvWrapper(raw, clip_actions=C.HIGH_KICK_RL_CFG.clip_actions)

    runner = MicroduckOnPolicyRunner(env, asdict(C.HIGH_KICK_RL_CFG), device=DEVICE)
    runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location=DEVICE)
    policy = runner.get_inference_policy(device=DEVICE)

    ball = raw.scene["ball"]
    org_z = float(raw.scene.env_origins[0, 2])
    cam = raw._offline_renderer._cam
    font = _font(18)

    obs = env.get_observations()
    bp = ball.data.root_link_pos_w[0].cpu().numpy()
    cam.lookat[:] = [float(bp[0]), float(bp[1]), 0.11]  # 初始对准球；循环内水平跟随

    peak_dh = 0.0
    writer = imageio.get_writer(
        out, format="FFMPEG", fps=FPS,
        codec="libx264", pixelformat="yuv420p",
        macro_block_size=1, output_params=["-crf", "20"],
    )
    print("checkpoint:", os.path.basename(ckpt))
    with torch.no_grad():
        for s in range(STEPS):
            actions = policy(obs)
            obs, _rew, _dones, _info = env.step(actions)
            bpos = ball.data.root_link_pos_w[0]
            z = float(bpos[2]) - org_z                        # 球相对本地 origin 的高度
            dh = (z - C.BALL_START_Z) * 100.0                 # 相对起始垫高高的增量(cm)
            peak_dh = max(peak_dh, dh)
            # 水平方向逐帧跟随球（v4 策略会摔倒翻滚、横向移动，固定机位会丢目标）；
            # 竖直 lookat 固定 0.11 不动，保留地面/细桩作参考，"挑高"才看得出来。
            cam.lookat[0] = float(bpos[0])
            cam.lookat[1] = float(bpos[1])

            frame = np.asarray(raw.render())
            img = Image.fromarray(frame)
            d = ImageDraw.Draw(img)
            d.text((8, 6), f"step {s + 1:03d}/{STEPS}", fill=(255, 255, 0), font=font)
            d.text((8, 28), f"ball dh = {dh:+5.1f} cm", fill=(0, 255, 128), font=font)
            d.text((8, 50), f"peak   = {peak_dh:5.1f} cm", fill=(255, 255, 255), font=font)
            writer.append_data(np.asarray(img))
    writer.close()
    raw.close()
    print(f"峰值挑高 = {peak_dh:.2f} cm（球心最高 ≈ {C.BALL_START_Z + peak_dh / 100:.3f} m）")
    print("saved:", out)


if __name__ == "__main__":
    main(
        sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CKPT,
        sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUT,
    )
