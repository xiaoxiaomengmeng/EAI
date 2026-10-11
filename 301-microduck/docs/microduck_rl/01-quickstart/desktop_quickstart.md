# 台式机快速部署与首次训练指南

本文档面向想在**一台本地台式机**上从零跑通 microduck_rl 训练的用户，覆盖：环境准备、依赖安装、验证、smoke test、正式训练、监控要点、训练后处理、故障排查。

若只想尽快开跑，直接跳到 [§10 TL;DR](#10-tldr三条命令跑通整个栈)。

---

## 1. 前置条件

### 1.1 硬件

| 项 | 要求 |
|---|---|
| GPU | **CUDA GPU 必需**（MuJoCo Warp 强依赖）。显存 ≥ 8 GB 可跑 4096 envs 的 walking 任务；≥ 12 GB 更从容 |
| 架构 | `x86_64` 或 `linux-aarch64`（DGX Spark / GB10 / Jetson）都受支持 |
| 内存 | 建议 ≥ 32 GB（4096 envs + warp kernel 编译时会吃内存） |
| 磁盘 | 首次 `uv sync` 会拉 ~2 GB CUDA wheel，加上 wandb 日志和 checkpoint，预留 ≥ 20 GB |

### 1.2 操作系统

- **Linux**：最顺的路径，HF Jobs 容器就是 Linux。
- **macOS**：**不能训**（无 CUDA），只能跑 `tests/` 里的 CPU-only 测试和 `scripts/infer_policy.py` 部署演练。
- **Windows**：建议在 WSL2 里跑；原生 Windows 未验证。

### 1.3 Python

`pyproject.toml` 死 pin `>=3.12,<3.13`：
- 上游 `bam` 要求 `<3.13`
- HF Jobs 曾因浮动上界选到 3.13.14（2026-07-21 事件）

`uv` 会自己拉一个 3.12 解释器，**不用管系统 Python**。

### 1.4 无 GPU 场景

任意 `train` 命令加 `--hf-jobs`，会在 Hugging Face Jobs 上跑（详见 [`scripts/hf/README.md`](../../../microduck_rl/scripts/hf/README.md)）：

```bash
uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096 --hf-jobs
```

默认 flavor `l4x1`，可选 `a10g-large` / `a100-large`。前置：`hf auth login` + `wandb login`。

---

## 2. 安装 uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**版本建议**：[`src/mjlab_microduck/hf_jobs.py`](../../../microduck_rl/src/mjlab_microduck/hf_jobs.py) 里 pin 的是 uv `0.11.30`，原因是持久化 uv cache bucket 跨 HF Job 共享，新版 uv 读旧版 cache 会报 `Missing .dist-info directory`。本地不共享 cache 影响不大；如果之后转 HF Jobs，或撞上类似 wheel-invalid 错误，可以 pin 到相同版本：

```bash
curl -LsSf https://astral.sh/uv/0.11.30/install.sh | sh
```

---

## 3. Clone 与 sync

```bash
git clone https://github.com/pollen-robotics/microduck_rl
cd microduck_rl

# ARM 机器（DGX Spark / GB10 / Jetson）必须加超时
# 首次 sync 拉 ~2 GB CUDA wheel，uv 默认 30 s HTTP 超时会中途断
export UV_HTTP_TIMEOUT=600

uv sync
```

`uv sync` 会做几件事：
- 拉 `torch==2.9.1`：x86_64 走 PyPI，aarch64 走 `download.pytorch.org/whl/cu129`（`[tool.uv.sources]` 里的路由）
- 拉 `warp-lang==1.12.0`
- 从 git 拉 `better-actuator-models` 的 `mjlab_frictionloss` 分支
- 装本仓库为 editable，注册 `mjlab.tasks` plugin 入口

首次 sync 大概 **3-8 分钟**看网速。

### 3.1 `pyproject.toml` 里几处**不能删**的配置

每一处都是踩过的坑（文件顶部注释写得很清楚），改之前务必读：

| 配置 | 作用 | 删掉的后果 |
|---|---|---|
| `[tool.uv] override-dependencies` 里 `zmq ; python_version < '3.0'` | 用永假 marker 屏蔽 bam 依赖的 PyPI stub `zmq==0.0.0` | 全新装失败："The wheel is invalid: Missing .dist-info directory" |
| `[tool.uv] override-dependencies` 里 `protobuf>=4.0,<7.0` | 阻止 bam 的 `protobuf<4.0` pin 级联 | onnx 被拖到 1.17.0，无预编译 wheel，从源码构建失败 |
| `[tool.uv] override-dependencies` 里 `onnx>=1.20.1` | 保证 onnx 有预编译 wheel | 同上 |
| `dependencies` 里 `scipy>=1.16` | mjlab 1.3.0 用了 scipy 但没声明 | `import mjlab_microduck` 直接失败（HF Jobs 上发现的 2026-07-21） |
| `dependencies` 里 `torch==2.9.1`（作为**直接**依赖） | 让 `[tool.uv.sources]` 生效（uv 只对直接依赖应用 sources） | aarch64 上路由变 no-op，装到 CPU-only torch |
| `torch==2.9.1` 用 `==` 不用 `>=` | CUDA 索引带的构建比 PyPI 新 | `>=` 曾静默把 torch 从 2.9.1 拖到 2.13.0 |
| `[tool.uv.sources]` 里的 `pytorch-cu129` 索引 + aarch64 marker | 路由 aarch64 的 torch 到 CUDA 版本 | aarch64 上装到 CPU-only torch，`torch.cuda.device_count() == 0` |
| `[project.scripts]` 里 `train = "mjlab_microduck.train_cli:main"` | 与 mjlab 的 `train` 故意冲突（last-writer-wins） | 删掉后 `uv sync` 会**卸载** `.venv/bin/train` 而**不重装** mjlab 的，`uv run train` 落到 PATH 上任何叫 `train` 的东西（曾落到 liblinear 的） |

这些都被 [`tests/test_aarch64_cuda_torch.py`](../../../microduck_rl/tests/test_aarch64_cuda_torch.py) 和 [`tests/test_hf_jobs_flag.py`](../../../microduck_rl/tests/test_hf_jobs_flag.py) 锁住，删了 CI 会红。

---

## 4. 配置 wandb（强烈推荐）

训练脚本默认把日志写到 wandb 项目 `mjlab_microduck`（`wandb_project="mjlab_microduck"`），每个任务族有自己的 `experiment_name`：`velocity`、`velocity_swizzle`、`roller_standup`、`roulade`……

```bash
uv run wandb login
```

想禁用 wandb 用 tensorboard：训练时加 `--agent.logger tensorboard`（[`docs/superpowers/plans/2026-08-04-roller-standup.md:1095`](../../../microduck_rl/docs/superpowers/plans/2026-08-04-roller-standup.md) 里就是这么建议做一次性验证 run 的，避免污染 wandb 项目）。

---

## 5. 验证安装

### 5.1 任务注册表能列出（30 秒）

```bash
uv run list-envs
```

会打印 27 个任务（含 `-Backlash-` 变体）。看到列表就代表：mjlab + warp + torch + bam + 本仓库 plugin 入口都通了。

### 5.2 torch 能看到 GPU

**x86_64**：

```bash
uv run python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda, torch.cuda.device_count())"
# 期望: True 12.x ≥ 1
```

**linux-aarch64（DGX Spark / GB10）**：跑专门的锁测试：

```bash
uv run pytest tests/test_aarch64_cuda_torch.py -v
```

`test_installed_torch_actually_sees_the_gpu` 只在 linux-aarch64 + `nvidia-smi` 有 GPU 时才跑，直接调 `torch.cuda.device_count()`——**这正是 mjlab 的 `select_gpus()` 崩之前读的东西**。

如果这一步失败，多半是 `[tool.uv.sources]` 的 torch 路由被破坏；参见 §3.1。

### 5.3 CPU-only 测试全通

```bash
uv run --with pytest pytest tests/
```

`tests/` 里全是 CPU-only 的 cfg 不变式和 reward 符号回归测试，无 GPU 也能跑（aarch64-specific 的会 skip）。跑通说明 mdp 补丁、publish gate、manifest schema 都在正常状态。

---

## 6. Smoke test（**永远不要跳过这一步**）

[`AGENTS.md`](../../../microduck_rl/AGENTS.md) 原话：

> A 5-iteration smoke test at 64 envs catches ~95% of config errors for cents. **Never launch a long run without one.**

```bash
uv run train Mjlab-Velocity-Flat-MicroDuck \
    --env.scene.num-envs 64 \
    --agent.max_iterations 5 \
    --agent.logger tensorboard
```

这一步验证：
- 环境构建、GPU 编译 warp kernel
- 61D obs 布局对齐
- 每个 reward term 都能算
- 没 NaN
- checkpoint 能写

**几分钟内就该结束**。挂了立刻看 traceback，别去开长跑。

启动日志里应该能看到这两行（说明 NaN 保护补丁生效）：

```
[mdp] Patches 1-2 active: NaN-safe reward/advantage
[mdp] Patch 4 active: ONNX export filters passive_* joints
```

---

## 7. 跑一次真正的训练

Smoke 通过后，选一个任务开跑。**首推 walking**（`Mjlab-Velocity-Flat-MicroDuck`）：它是最成熟的、其他 env 都建立在它之上的模板。

```bash
uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096
```

### 7.1 时间预算

[`AGENTS.md`](../../../microduck_rl/AGENTS.md#L215) 里的实测数字：

| 任务类型 | 4096 envs 下的迭代数 | 大概时长 |
|---|---|---|
| Walking 出可用步态 | ~1500-2500 iter | **1-2 h** |
| Walking 完整 curriculum 收敛 | 4000-6000 iter | 3-5 h |
| 简单 episodic trick（BallKick、Roulade） | ~1000 iter | ~1 h |
| 步态 + curriculum 重的恢复（VelStand、StandUp、RollerStandUp） | 4000-6000 iter | 3-5 h |

每 iteration = `NUM_STEPS_PER_ENV (24) × num_envs` env steps。4096 envs 下每 iteration ≈ 100k env steps。

### 7.2 显存不够

调 `--env.scene.num-envs`：

```bash
uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 2048    # 或 1024
```

也能训，只是慢一点、经验多样性差点。curriculum 里的 `step` 阈值是按 env steps 算的（`iteration × 24`），与 num_envs **无关**，所以 stage 触发时机不变，只是每 iter 的样本量少一半。

### 7.3 续训

```bash
uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096 \
    --agent.run-name resume \
    --agent.load-checkpoint model_2999.pt \
    --agent.resume True
```

Checkpoint 每 250 iter 存一次（`save_interval=250`），路径：

```
logs/rsl_rl/<experiment_name>/<timestamp>_<run_name>/model_<iter>.pt
```

### 7.4 其他任务

```bash
uv run train Mjlab-StandUp-Flat-MicroDuck --env.scene.num-envs 4096          # 从趴/仰/坐起身
uv run train Mjlab-SitStand-Flat-MicroDuck --env.scene.num-envs 4096         # 命令式 sit↔stand
uv run train Mjlab-BallKick-Flat-MicroDuck --env.scene.num-envs 4096         # 踢球（右脚）
uv run train Mjlab-Roulade-Flat-MicroDuck --env.scene.num-envs 4096          # 前滚翻
uv run train Mjlab-Velocity-Swizzle-MicroDuck --env.scene.num-envs 4096      # 轮滑 swizzle
uv run train Mjlab-Velocity-Flat-Backlash-MicroDuck --env.scene.num-envs 4096 # 含齿隙的 walking
```

完整列表 `uv run list-envs`。

---

## 8. 训练时该盯什么

[`AGENTS.md`](../../../microduck_rl/AGENTS.md#L207-L223) 里的清单，打开 wandb（或 tensorboard）后每 iteration 检查：

### 8.1 mean reward 上升 **且** episode length 按任务需要动作

- Walking：episode length 应该稳定接近最大值（不摔）
- StandUp / RollerStandUp：随恢复成功率上升
- Roulade：完成翻滚后 episode length 会稳定到 5 s（`EPISODE_LENGTH_S`）

### 8.2 每个 penalty 项 ≤ 0（**铁律**）

wandb 里 `Episode_Reward/<penalty>` 都必须 ≤ 0。**如果发现某项是正的，就是双重否定 bug**——立刻停 run 检查权重符号。

背景：`mdp.py` 里有两种惩罚风格：
- mjlab 基础成本函数返回 ≥0 → **负权重**
- microduck 自写的 `*_penalty` / `*_l1` 返回 ≤0（已自 negated）→ **正权重**

给自 negated 的惩罚再加负权重 = 双重否定 = **对违规行为的奖励**，策略会去 farm 它（butt-hopping、crash-sits）。

真实案例：`roller_standup` 里 `gentle_rise` 用 `trunk_vertical_accel_penalty`（返回 `-|a_z|`），乘上从 standup 继承的 `-0.02`，等于 `+0.02·|a_z|`——**躯干越暴力加速，policy 越被付钱**。日志确认 `Episode_Reward/gentle_rise = +0.0118`（run `vweolw91`，唯一记录为正值的惩罚项）。详见 [`docs/roller_standup_policy_summary.md`](../../../microduck_rl/docs/roller_standup_policy_summary.md#L119-L138)。

### 8.3 主任务项**真的在长**

Total reward 可能纯靠 regularizer 上升而 trick 从未发生。对 walking 主要看：
- `Episode_Reward/track_linear_velocity`
- `Episode_Reward/track_angular_velocity`
- `Episode_Reward/head_pose_tracking`

### 8.4 权重为 0 的项读作 0 不管行为怎样

`Episode_Reward/<term>` 记的是**加权后**的值，要对着 curriculum 权重调度解读。

Velocity env 的权重 curriculum：
- `action_rate_l2`：`-0.1 → -1.0`（iter 0→1500）
- `head_pose_bias`：`0 → 3.0`（iter 600→1500，之前是 0）
- `standing_envs`：`2% → 25%`（iter 0→2000）

### 8.5 wandb 指标恰好在 curriculum stage 边界处下降 = 节奏错了

**别硬扛**，把 stage 拉长或者推迟引入。规则：`Phase-align every stage with what the policy has actually learned`（[`AGENTS.md`](../../../microduck_rl/AGENTS.md#L196-L200)）。

### 8.6 报告 rollout 实际显示什么

不是 "it works!"，而是 "rolls but face-plants 1 in 3"。**用户决定什么时候够好**。Sim 指标可以过而视频在人眼上失败——看视频，也检查哪个 geom/轴接触。

---

## 9. 训练后处理

### 9.1 在 viewer 里看训练好的 policy

```bash
uv run play Mjlab-Velocity-Flat-MicroDuck \
    --wandb-run-path <entity/project/run_id>
```

或用最新 checkpoint：

```bash
uv run scripts/play_latest.py
```

### 9.2 导出 ONNX（**唯一安全路径**）

```bash
uv run scripts/export.py Mjlab-Velocity-Flat-MicroDuck \
    --wandb-run-path <entity/project/run_id>
```

`runner.export_policy_to_onnx` 发射 `actor(normalizer(obs))`，obs normalizer **烘焙进图**。

**永远不要手转 checkpoint**：in-sim `play` 自己也应用 normalizer，会**掩盖**手转 checkpoint 忘了烘焙的 bug。真机上策略看到的是未归一化 obs，行为完全崩掉。

### 9.3 CPU MuJoCo 里键盘驱动 ONNX 演练部署

```bash
uv run scripts/infer_policy.py --walking output.onnx
```

不需要 GPU，用 BAM M6 执行器模拟训练时的物理。键位：
- 速度命令：WASD 类
- `G` ground pick、`Y` sit/stand、`R` roulade、`K`/`L` kicks
- `--debug` / `--save-csv` / `--record` 支持 sim2real 对比

多 policy 热切换演练：

```bash
uv run scripts/infer_policy.py --walking walk.onnx --standing stand.onnx \
    --sitstand sitstand.onnx --roulade roulade.onnx --new-cmd-obs
```

**关键**：posture flag 住在 **twist vx 槽**，喂全零意味着 "stand"。看起来就像 "policy 忽略按钮" 的 bug 其实是命令槽写错了。

### 9.4 发布到 HF Hub（可选）

```bash
uv run publish --task Mjlab-Velocity-Flat-MicroDuck \
    --wandb-run-path <entity/project/run_id> --checkpoint 3000 \
    --repo <user>/microduck-my-walk --kind perpetual --slot walk
```

上传前会 gate：图必须是 `[1,61] → [1,14]`、smoke run 50 步不出 NaN 且输出非常数、manifest schema-2 合法。详见 [`src/mjlab_microduck/publish/cli.py`](../../../microduck_rl/src/mjlab_microduck/publish/cli.py)。

---

## 10. TL;DR：三条命令跑通整个栈

```bash
# 1. 装
git clone https://github.com/pollen-robotics/microduck_rl && cd microduck_rl
UV_HTTP_TIMEOUT=600 uv sync && uv run wandb login

# 2. Smoke（永远别跳）
uv run train Mjlab-Velocity-Flat-MicroDuck \
    --env.scene.num-envs 64 --agent.max_iterations 5 --agent.logger tensorboard

# 3. 真训
uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096
```

跑通这三条，整个栈（uv → torch/CUDA → warp → mjlab → BAM → mdp 补丁 → wandb → checkpoint）就都验证过了。之后玩其他任务就是改一下 task id 的事。

---

## 11. 故障速查

| 症状 | 原因 / 修法 |
|---|---|
| `uv sync` 卡在拉 CUDA wheel | ARM 机器没设 `UV_HTTP_TIMEOUT=600` |
| `uv sync` 报 `zmq` wheel invalid | `[tool.uv] override-dependencies` 里的 zmq 屏蔽被误删（见 §3.1） |
| `uv sync` 报 onnx 从源码构建失败 | protobuf 或 onnx override 被误删（见 §3.1） |
| `train` 命令找不到 / 报 `can't open input file Mjlab-Velocity-Flat-MicroDuck` | PATH 上有别的 `train`（如 liblinear 的）；检查 `.venv/bin/train` 是否存在，不在就 `uv sync --reinstall-package mjlab-microduck`（[`tests/test_hf_jobs_flag.py`](../../../microduck_rl/tests/test_hf_jobs_flag.py) 有细节） |
| `IndexError: list index out of range` @ `mjlab/utils/gpu.py:70` | torch 看不到 GPU；aarch64 上多半是 `[tool.uv.sources]` 的 torch 路由被破坏（见 §3.1 和 §5.2） |
| Iteration 0 之前就崩，报 NaN | 大概率是 mjlab 版本或某个 patch 没生效；确认启动日志里 `[mdp] Patches 1-2 active` 和 `[mdp] Patch 4 active` 都打印了 |
| 训练几百 iter 后 reward 突然 NaN 崩 | Rough terrain + 接触爆炸；velocity env 已经调过 `nconmax=200` / `iterations=30` / `ls_iterations=50`；flat 版本没这问题。若自建 rough env 崩，参考 [`microduck_velocity_env_cfg.py:762-769`](../../../microduck_rl/src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py#L762-L769) |
| 某个 penalty 项在 wandb 里读作正值 | 双重否定 bug（见 §8.2）——立刻停 run，检查权重符号 |
| `--hf-jobs` 报 `Unrecognized options: --hf-jobs` | `train_hook.py` 的入口拦截没跑；确认 `mjlab_microduck.tasks` 被 mjlab plugin loader 加载了。也可能是 `[project.scripts]` 里的 `train` 声明被删（见 §3.1） |
| Rough terrain 上 OOM | `BoxRandomGridTerrainCfg` 的 `grid_width` 太小；velocity env 用 0.45 m（17×17 boxes/patch），若改成 0.12 m 会变 66×66 = 4356 boxes/patch → 总 ~261K → OOM |

---

## 12. 延伸阅读

- [`README.md`](../../../microduck_rl/README.md) — 项目全景、任务清单、actuator 模型、robot 模型变体、publish 流程
- [`AGENTS.md`](../../../microduck_rl/AGENTS.md) — env-building workflow、reward 设计规则、curriculum 规则、sim2real footguns（**读完再改 env cfg**）
- [`docs/roller_standup_policy_summary.md`](../../../microduck_rl/docs/roller_standup_policy_summary.md) — 一个 policy 从设计到调试的完整案例（含符号 bug 复盘）
- [`docs/superpowers/specs/`](../../../microduck_rl/docs/superpowers/specs/) — 每个新 env 的设计文档
- [`docs/superpowers/plans/`](../../../microduck_rl/docs/superpowers/plans/) — 每个新 env 的实现计划（含具体 iteration 数和验收标准）
- [`scripts/hf/README.md`](../../../microduck_rl/scripts/hf/README.md) — HF Jobs 远程训练
- 上游 [mjlab](https://github.com/mujocolab/mjlab) / [BAM](https://github.com/Rhoban/bam) / [microduck runtime](https://github.com/pollen-robotics/microduck)
