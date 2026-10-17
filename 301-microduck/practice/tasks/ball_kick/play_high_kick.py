"""高踢任务回放入口（practice 侧，本地 checkpoint）。

复用官方 `run_play(task_id, PlayConfig)`——它内部 `load_env_cfg(play=True)` +
`load_rl_cfg` + `load_runner_cls` 全走 registry，所以我们只要保证高踢 task 已
注册（直接复用 train_high_kick.register_high_kick_task），回放管线原样可用。

与官方 scripts/play_latest.py 的区别：那个绑定了 wandb user=coralie 从云端找
run；我们**从本地 checkpoint 回放**，用 PlayConfig.checkpoint_file。

用法（在本目录下执行）：
    MUJOCO_GL=egl python play_high_kick.py \
        --checkpoint-file logs/rsl_rl/ball_kick_high_right/<时间戳>/model_300.pt
可选：
    --num-envs 4            少量环境看得清楚
    --viewer native|viser   有显示器用 native；headless 用 viser（浏览器）
    --video                 录 mp4 到 <run>/videos/play/
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import sys  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tyro  # noqa: E402

import mjlab  # noqa: E402
from mjlab.scripts.play import PlayConfig, run_play  # noqa: E402

from train_high_kick import TASK_ID, register_high_kick_task  # noqa: E402


def main() -> None:
    register_high_kick_task()
    args = tyro.cli(
        PlayConfig,
        default=PlayConfig(),
        prog=f"play_high_kick {TASK_ID}",
        config=mjlab.TYRO_FLAGS,
    )
    run_play(TASK_ID, args)


if __name__ == "__main__":
    main()
