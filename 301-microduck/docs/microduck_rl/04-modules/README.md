# 04-modules · 模块击破

> 认知阶梯第 ⑤ 层：以 **velocity 实例为脊柱**，逐个吃透 `microduck_rl` 的源码模块。

## 核心原则

**不孤立读模块**——每读一个模块，都回到"它在训练 `Mjlab-Velocity-Flat-MicroDuck`（鸭子走路）时扮演什么角色"。大文件（如 7188 行的 `mdp.py`）抓主线 + 按需查，**不通读**。每份导读都保留代码里的**真实调参案例与踩坑推理**。

## 填充计划（13 份，按依赖顺序）

| # | 模块 | 核心文件 |
|---|---|---|
| 1 | robot 身体 | `robot/microduck_constants.py` + XML + `add_backlash.py` |
| 2 | actuator 怎么动 | `actuator/friction_dr_bam.py`（BAM 舵机模型） |
| 3 | tasks 总览 | `tasks/__init__.py`（注册机制） |
| 4 ★ | mdp 词库 | `tasks/mdp.py`（7188 行，怎么查） |
| 5 ★ | [velocity 精讲](05-velocity-deep-dive.md) | `tasks/microduck_velocity_env_cfg.py`（950 行，逐段读） |
| 6 | 任务变体 | `symmetry.py` + `backlash.py` + `slope_terrain.py` |
| 7 | 其余任务对照 | 13 个 `*_env_cfg.py` 的差异地图 |
| 8 | export | `export.py`（ONNX 导出） |
| 9 | publish | `publish/cli.py` + `manifest.py`（发布关卡） |
| 10 | 训练入口 | `train_cli.py` + `train_hook.py` + `hf_jobs.py` |
| 11 | sim | `sim/body_server.py`（反向 sim2real） |
| 12 | scripts | `scripts/infer_policy.py`（1524 行）+ `testbench_sim2real.py` 等 |
| 13 | tests | `tests/` 19 个 CPU-only 不变量测试 |

## 现状

- ✅ **#5 velocity 精讲已完成** → [`05-velocity-deep-dive.md`](05-velocity-deep-dive.md)（逐段拆解 950 行 env_cfg + 完整"改/删/加"速查表 + 真实 run 复盘）。
- 🚧 其余 12 份待写。**建议下一份写 #4（mdp 词库）**——velocity 精讲里出现的 `head_pose_bias_penalty`、`robot_state_is_nan`、`reward_weight` 等函数的实现都在 `mdp.py`，两份配套读最佳。

每份"导读要回答什么"的完整清单，见 [总导航 README 第五节](../README.md)。
