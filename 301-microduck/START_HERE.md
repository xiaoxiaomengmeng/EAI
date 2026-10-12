# 301-microduck

一句话：Pollen Robotics 的 Microduck 微型双足机器人（25cm / 800g）。`microduck_rl` 用 mjlab + PPO 训练运动策略并导出 ONNX，`microduck`（Rust）是跑在真机 RK3566 上加载 ONNX 的板载运行时——两者构成 sim2real 闭环。

## 双子仓库分工

- **`microduck_rl/`（训练场）**：Python 3.12 · mjlab · PPO · MuJoCo Warp，**uv 管理**；仿真里训练策略 → 导出 ONNX。
- **`microduck/`（鸭子的大脑）**：Rust · cargo；部署在 Rockchip RK3566，50Hz 控制循环驱动 15 个舵机、加载 ONNX 策略。

## 启动环境（训练侧 microduck_rl）

```bash
cd ~/ws/301-microduck/microduck_rl
uv sync                 # 首次：拉 torch 2.9.1 / warp / mjlab（~2GB CUDA wheel），装本仓库为 editable
uv run list-envs        # 之后一切命令走 uv run（自动用 .venv，Python 3.12）
```

- ⚠️ 本项目用 **uv**，不是 `source .venv/bin/activate`；临时依赖用 `uv run --with <pkg>`（如 pytest）；别用 `(base)` 的 python 直接跑测试/脚本。
- 根目录无 `.venv`（重组前 baseline 里记录的那个已失效）；Rust 侧用 `cargo build`，真机部署见 [`microduck/deploy/`](microduck/deploy)。

## 从这里开始

- ⭐ **训练全链路**（本机 RTX 5090 已实测跑通 §1~§6）：[docs/microduck_rl/01-quickstart/desktop_quickstart.md](docs/microduck_rl/01-quickstart/desktop_quickstart.md) —— 装 → 验证 → smoke → 正式训练 → 导出 ONNX；赶时间直接看末尾 §10 三条命令 TL;DR。
- **文档中心总索引**：[docs/README.md](docs/README.md)（中文学习笔记，分 microduck / microduck_rl / diagrams 三块）；RL 认知阶梯 [docs/microduck_rl/README.md](docs/microduck_rl/README.md)（00→07）与路线图 [docs/learning_roadmap.md](docs/learning_roadmap.md)。
- **Rust 运行时**：[docs/microduck/README.md](docs/microduck/README.md)（架构 / IPC 契约 / robotd / updater）；上游说明 [microduck/README.md](microduck/README.md)。
- **RL 入门练习**：[learning_RL/](learning_RL)（session_01 PyTorch 基础、session_02 MDP · Q-learning · DQN）。
- 任务认准 `-MicroDuck` 后缀（`uv run list-envs` 共 45 个，其余为 mjlab 内置的 Cartpole / Unitree / Yam）；首推 walking `Mjlab-Velocity-Flat-MicroDuck`。
