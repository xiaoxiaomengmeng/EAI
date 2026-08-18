#!/usr/bin/env python3
"""ALoHa 仿真环境冒烟测试：随机动作跑通 gym_aloha/AlohaTransferCube-v0。

训练前必做，用于排除渲染/依赖问题。包含观测/动作空间断言与帧率统计。

用法：
    python scripts/smoke_test_env.py                # 台式机带显示器：默认 glfw 渲染
    MUJOCO_GL=egl python scripts/smoke_test_env.py  # 无头/SSH：EGL GPU 渲染

参照 py310 环境实测输出（EGL）：
    obs keys: ['top']（480x640x3 uint8）、action 空间: Box(-1.0, 1.0, (14,))、
    冒烟测试 OK | 步数: 100 | 帧率: 241 FPS
"""
from __future__ import annotations

import argparse
import time

import gymnasium as gym
import gym_aloha  # noqa: F401（注册环境）


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", default="gym_aloha/AlohaTransferCube-v0")
    parser.add_argument("--steps", type=int, default=100, help="随机动作步数")
    parser.add_argument("--render", action="store_true", help="开启 human 渲染窗口")
    args = parser.parse_args()

    env = gym.make(args.env_id, render_mode="human" if args.render else None)

    # 动作空间断言：ALOHA 双臂 14 维（每臂 6 关节 + 1 夹爪）
    assert env.action_space.shape == (14,), f"action 空间维度异常: {env.action_space}"
    print(f"action 空间: {env.action_space}")

    obs, _ = env.reset()
    print("obs keys:", list(obs.keys()))
    for key, value in obs.items():
        # 图像观测断言：uint8、HWC
        assert value.dtype == "uint8" and value.ndim == 3 and value.shape[2] == 3, (
            f"图像观测 {key} 格式异常: dtype={value.dtype}, shape={value.shape}"
        )
        print(f"  {key}: shape={value.shape}, dtype={value.dtype}")

    n_steps = 0
    t0 = time.perf_counter()
    for _ in range(args.steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        n_steps += 1
        if terminated or truncated:
            break
    fps = n_steps / (time.perf_counter() - t0)

    env.close()
    print(f"冒烟测试 OK | 步数: {n_steps} | 帧率: {fps:.0f} FPS")


if __name__ == "__main__":
    main()
