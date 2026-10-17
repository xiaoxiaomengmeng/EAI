"""高踢任务训练入口（practice 侧）。

一句话原理：官方 `train` = 「import 触发注册 → list_tasks → tyro 选 task →
launch_training」。我们**照搬这套骨架**，只是把「选 task」换成固定的高踢
task，并在选之前先把我们的自定义变体 `register_mjlab_task` 进那个全局注册表
（`_REGISTRY` 是模块级 dict，运行时可追加）——于是不改子模块一行代码，就能
新增一个任务入口，并复用官方整条训练管线（GPU 选择 / EGL / wandb / checkpoint
/ VideoRecorder 全在 launch_training 里）。

用法（在本目录下执行）：
    # 小规模验证链路
    WANDB_MODE=offline python train_high_kick.py \
        --env.scene.num-envs 512 --agent.max-iterations 300
日志落在 ./logs/rsl_rl/ball_kick_lift_right/<时间戳>/（相对当前 cwd）。
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")  # headless 渲染：必须在 import mujoco 前

import sys  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tyro  # noqa: E402

import mjlab  # noqa: E402
from mjlab.scripts.train import TrainConfig, launch_training  # noqa: E402
from mjlab.tasks.registry import list_tasks, register_mjlab_task  # noqa: E402

TASK_ID = "Mjlab-BallKickHigh-Flat-MicroDuck"


def register_high_kick_task() -> None:
    """把高踢变体注册进全局 registry（幂等；play 入口也复用它）。"""
    # 触发官方插件加载：注册全部官方 task，并导出 MicroduckOnPolicyRunner。
    import mjlab_microduck.tasks  # noqa: F401
    from mjlab_microduck.tasks import MicroduckOnPolicyRunner

    from high_kick_cfg import HIGH_KICK_RL_CFG, make_high_kick_env_cfg

    if TASK_ID in list_tasks():
        return
    register_mjlab_task(
        task_id=TASK_ID,
        env_cfg=make_high_kick_env_cfg(),
        play_env_cfg=make_high_kick_env_cfg(play=True),
        rl_cfg=HIGH_KICK_RL_CFG,
        runner_cls=MicroduckOnPolicyRunner,
    )


def main() -> None:
    register_high_kick_task()
    args = tyro.cli(
        TrainConfig,
        default=TrainConfig.from_task(TASK_ID),
        prog=f"train_high_kick {TASK_ID}",
        config=mjlab.TYRO_FLAGS,
    )
    launch_training(task_id=TASK_ID, args=args)


if __name__ == "__main__":
    main()
