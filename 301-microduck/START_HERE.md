# 301-microduck

一句话：Pollen Robotics 的 Microduck 微型双足机器人（25cm / 800g）。`microduck_rl` 用 mjlab + PPO 训练运动策略并导出 ONNX，`microduck`（Rust）是跑在真机 RK3566 上加载 ONNX 的板载运行时——两者构成 sim2real 闭环。

## 双子仓库分工

- **`microduck_rl/`（训练场）**：Python 3.12 · mjlab · PPO · MuJoCo Warp，**uv 管理**；仿真里训练策略 → 导出 ONNX。
- **`microduck/`（鸭子的大脑）**：Rust · cargo；部署在 Rockchip RK3566，50Hz 控制循环驱动 15 个舵机、加载 ONNX 策略。

## 启动环境（训练侧 microduck_rl）

```bash
cd ~/ws/301-microduck
source .venv/bin/activate   # 根目录 .venv 即训练环境：Python 3.12 · torch 2.9.1+cu128 · mjlab 1.3.0 · mjlab_microduck 为 editable（→ microduck_rl/src）
train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 16 --agent.max_iterations 5   # 冒烟
```

- 根 `.venv` 已于 2026-10-12 目录改名后修复并实测：16 env × 1 迭代 exit 0。`train` 是 microduck_rl 覆盖版入口（`mjlab_microduck.train_cli`），`play`/`export` 走 mjlab；wandb 记录为 offline 模式，无需登录。
- `microduck_rl/.venv` 是个空壳（只有 virtualenv 引导文件、无任何包）：在里面 `uv run` 会报 `Failed to spawn: train`；要恢复 AGENTS.md 的 uv 工作流需 `cd microduck_rl && uv sync`（重建约 2GB），不用 uv 可无视它。
- Rust 侧用 `cargo build`，真机部署见 [`microduck/deploy/`](microduck/deploy)。

## 从这里开始

- ⭐ **训练全链路**（本机 RTX 5090 已实测跑通 §1~§6）：[docs/microduck_rl/01-quickstart/desktop_quickstart.md](docs/microduck_rl/01-quickstart/desktop_quickstart.md) —— 装 → 验证 → smoke → 正式训练 → 导出 ONNX；赶时间直接看末尾 §10 三条命令 TL;DR。
- **文档中心总索引**：[docs/README.md](docs/README.md)（中文学习笔记，分 microduck / microduck_rl / diagrams 三块）；RL 认知阶梯 [docs/microduck_rl/README.md](docs/microduck_rl/README.md)（00→07）与路线图 [docs/learning_roadmap.md](docs/learning_roadmap.md)。
- **Rust 运行时**：[docs/microduck/README.md](docs/microduck/README.md)（架构 / IPC 契约 / robotd / updater）；上游说明 [microduck/README.md](microduck/README.md)。
- **RL 入门练习**：[learning_RL/](learning_RL)（session_01 PyTorch 基础、session_02 MDP · Q-learning · DQN）。
- 任务认准 `-MicroDuck` 后缀（`uv run list-envs` 共 45 个，其余为 mjlab 内置的 Cartpole / Unitree / Yam）；首推 walking `Mjlab-Velocity-Flat-MicroDuck`。
