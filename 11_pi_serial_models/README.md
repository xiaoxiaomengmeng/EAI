# 11_pi_serial_models — Pi 系列与 ACT 策略模型学习目录

Pi 系列（**Pi0 / Pi0.5**）与 **ACT** 策略模型的本地训练、微调与仿真评估，基于
**lerobot-main v0.6.2**（PyTorch）。包含两条评测链路：

- **ALOHA 双臂**（gym-aloha / MuJoCo）：搬方块交接、插孔任务，覆盖 ACT 与 pi0；
- **LIBERO**：四子集基准（spatial / object / goal / 10），覆盖 pi0.5。

运行环境：**Ubuntu 22.04 + RTX 5090 32GB + Python 3.12（`.venv312`）+ PyTorch 2.11.0+cu128**。

```bash
cd ~/ws/11_pi_serial_models && source .venv312/bin/activate   # 一切命令的前提
```

## 目录结构

```
11_pi_serial_models/
├── data/            # 本地数据集（LeRobot v3.0 格式，视频为 AV1 编码）
│   ├── aloha_sim_transfer_cube_human/    # ACT/pi0 主数据集（50 演示 × 400 帧）
│   ├── aloha_sim_insertion_human/        # 插孔任务（换任务训练用）
│   ├── aloha_sim_*_scripted/             # 脚本生成的对照数据
│   ├── libero/                           # LIBERO 四子集（1.9G，pi0.5 用）
│   └── pusht/                            # Push-T 单臂 2D 任务（备用）
├── models/          # 基座权重软链 → ModelScope 缓存（~/.cache/modelscope/）
│   ├── pi0_base / pi0_libero_base        # pi0 通用基座 / LIBERO 预训练版
│   └── pi05_base / pi05_libero_base      # pi0.5 通用基座 / LIBERO 预训练版
├── repos/
│   └── lerobot-main/                     # lerobot v0.6.2 源码（pip install -e 已装）
├── scripts/         # 按阶段编号的配套脚本（见「脚本导航」）
├── docs/            # 编号指南 17~25（部署/训练/评估/数据流详解，见「文档索引」）
├── logs/            # 训练/评估日志（train_*_<时间戳>.log / eval_*_<时间戳>.log）
├── outputs/         # checkpoint 与 rollout 视频
└── wandb/           # wandb 离线 run（默认不用，曲线走 41_plot_train_log.py）
```

## 模型与评测链路

| 模型 | 训练方式 | 数据集 | 评测任务 | 状态与实测 |
|------|----------|--------|----------|------------|
| **ACT** | 从零训练（52M 参数，无基座权重） | aloha_sim_transfer_cube_human | AlohaTransferCube-v0 | ✅ 100k 步 ≈ 46 分钟；**500 轮成功率 79%**（官方基线 83%） |
| **pi0** | LoRA（r=16，20k 步）/ 全参（200k 步） | 同上 | AlohaTransferCube-v0 | ✅ 链路可用（见 docs/17/18/20） |
| **pi0.5** | LoRA（r=16，30k 步）/ Action Expert | libero | LIBERO 四子集（50 轮/子集） | ✅ 链路可用（见 docs/21/22/24） |

产物约定：`outputs/<任务>/checkpoints/`，评估用 `checkpoints/last/pretrained_model`；
rollout 视频在 `outputs/eval_*/videos/`。

## 学习待办：lerobot v0.6.2 还支持哪些策略（已学 pi0 / pi0.5 / ACT）

`repos/lerobot-main/src/lerobot/policies/` 下共有约 20 种策略，除已学的三个外，按学习成本与互补性分三档待办：

### A. 模仿学习经典基线（可复用现有 aloha 数据集从零训练，建议优先）

| 策略 | `--policy.type` | 一句话定位 | 学习价值 |
|------|------|-----------|----------|
| **Diffusion Policy** | `diffusion` | DDPM 去噪式动作生成（Chi et al. 2023） | ACT 最经典的对照基线，确定性 chunk vs 扩散生成的范式对比 |
| **TD-MPC** | `tdmpc` | 基于模型的隐空间控制（世界模型 + MPC） | 模仿学习之外的 model-based 路线 |
| **VQ-BeT** | `vqbet` | 行为 VQ 预训练 + 微调的两阶段范式 | 理解“先压缩行为再精调”的训练思路 |
| **Multi-Task DiT** | `multi_task_dit` | 文本+视觉条件的多任务扩散 Transformer | 多任务条件化的进阶形态 |
| **Gaussian Actor** | `gaussian_actor` | 高斯分布动作头的策略 | 随机策略表示的对照 |
| **RTC / BID** | （动作选择器） | Real Time Chunking 与双向解码 | 提升 chunk 类策略实时性的推理技巧，可直接叠加到已学 ACT 上 |

### B. VLA 扩展（需下载基座权重做微调，与已学 pi0 同源最顺）

| 策略 | 一句话定位 | 备注 |
|------|-----------|------|
| **pi0-FAST** | pi0 的 FAST 离散动作 token 变体 | 学完 pi0 后的自然延伸，同一模型家族 |
| **SmolVLA** | HuggingFace 轻量 VLA | 体量小，5090 友好，与 pi0 同赛道对照 |
| **GR00T** | NVIDIA 通用机器人基础模型 | 生态位大，依赖 Isaac 系资源 |
| **MolmoAct2** | Allen AI 的 VLM 机器人策略 | 另一家 VLM→动作方案对照 |
| **Wall-X** | Qwen2.5-VL + flow matching 跨本体控制 | 国产 VLM 底座 + 流匹配 |
| **xVLA** | Florence-2 嵌入 + 时序/动作 Transformer 头 | 轻量 VLA 的另一条路线 |
| **VLA-JEPA** | 基于 V-JEPA 表征的 VLA | 自监督表征 + 动作 |
| **EO1** | EO1 策略（robot-only 训练/评估封装） | 新接入，资料较少，需先看上游文档 |

### C. 世界模型/前沿方向（重、先调研再上机）

| 策略 | 一句话定位 | 备注 |
|------|-----------|------|
| **LingBot-VA** | Wan2.2 底座的自回归视频-动作世界模型（蚂蚁灵波） | 与 `22_lingbot/` 同源，可联动学习 |
| **FastWAM** | FastWAM 策略封装 | 新接入，先看上游文档 |
| **EVO-1** | 含推理延迟/残余块机制的动作选择 | 新接入，先看上游文档 |
| **pi-gemma** | `pi_gemma.py` 独立实现的 Gemma 底座变体 | pi 家族另一分支 |

**建议的学习顺序**：① Diffusion Policy（与 ACT 同数据集从零训练，可直接复用 `21_train_act.sh` 的脚本框架改 `--policy.type=diffusion`，一两天即可出对照实验）→ ② pi0-FAST（pi0 的自然延伸）→ ③ SmolVLA（轻量 VLA 对照）→ ④ 其余按兴趣。所有策略的训练/评估入口不变（`lerobot-train` / `lerobot-eval`），只需换 `--policy.type` 与配套数据集。

## 脚本导航

`scripts/` 下脚本**十位数字 = 阶段，个位留空可扩展**（新增按 `25_`/`34_` 顺延），
全部支持环境变量覆盖（`STEPS=`/`N_EPISODES=` 等），用法细节见各脚本头部注释。

| 脚本 | 作用 | 关键默认值 | 详细文档 |
|------|------|-----------|----------|
| `01_smoke_test_env.py` | gym-aloha 仿真环境冒烟（随机动作 + 帧率统计） | — | docs/17 |
| `02_test_dataset.py` | 验证数据集可被 LeRobotDataset 加载 | — | docs/17/18 |
| `03_test_pi0_io.py` | pi0 输入/输出接口冒烟（不依赖外部仓库代码） | `--model-name=pi0_libero_base` | docs/19 |
| `11_download_act_assets.sh` | ModelScope 下载 ACT 数据集/官方预训练模型 | 仅主数据集；`ALL_DATASETS=true` / `PRETRAINED=true` 可选 | docs/25 |
| `21_train_act.sh` | ACT 从零训练（ALOHA 搬方块） | 100k 步、batch 8、chunk 100、kl 10.0（与官方一致）；`DATASET_NAME=` 可换插入任务 | docs/25 |
| `22_train_pi0.sh` | pi0 LoRA 微调（ALOHA） | 20k 步、batch 2、LoRA r=16 | docs/17/18 |
| `23_train_pi0_full.sh` | pi0 全参微调（ALOHA） | 200k 步、batch 2 | docs/20 |
| `24_train_pi05_libero.sh` | pi0.5 LIBERO 后训练 | `TRAIN_MODE=lora`(默认)/`expert`；30k 步、LoRA r=16 | docs/21 |
| `31_eval_act.sh` | ACT rollout 评估（ALOHA） | 25 轮；`EVAL_BATCH>1` 时自动走 Sync 环境 | docs/25 |
| `32_eval_pi0.sh` | pi0 rollout 评估（ALOHA） | 25 轮、batch 1 | docs/17 |
| `33_eval_pi05_libero.sh` | pi0.5 LIBERO 评估 | `LIBERO_TASK=libero_spatial`(默认)；50 轮 | docs/21/22 |
| `41_plot_train_log.py` | 解析训练日志画 loss 曲线（loss / l1_loss / kld_loss 三子图） | 自动找最新 `logs/train_act_*.log` | docs/25 |

### 典型工作流

```bash
source .venv312/bin/activate

# —— ACT 三件套（从零训练，无需基座权重；实测 100k 步 46 分钟、500 轮验收 79%）——
python scripts/01_smoke_test_env.py && python scripts/02_test_dataset.py
bash scripts/21_train_act.sh
python scripts/41_plot_train_log.py
N_EPISODES=500 EVAL_BATCH=16 bash scripts/31_eval_act.sh

# —— pi0 LoRA ——
bash scripts/22_train_pi0.sh
bash scripts/32_eval_pi0.sh

# —— pi0.5 LIBERO ——
bash scripts/24_train_pi05_libero.sh
bash scripts/33_eval_pi05_libero.sh
```

## 文档索引（docs/，按编号递进）

| 编号 | 主题 |
|------|------|
| 17 | pi0 本地化部署与 ALOHA 仿真训练指南（py3.10 / py3.12 两版） |
| 18 | pi0 与 ALOHA 本地训练入门教程 |
| 19 | PI0 训练与评估技术数据流详解 |
| 20 | PI0 理论与实践全链路实录（含全参微调） |
| 21 | pi0.5 本地化部署与仿真训练指南 |
| 22 | PI0.5 LIBERO 评估日志全链路解析 |
| 23 | PI0.5 LIBERO 场景搭建与推理交互全链路详解 |
| 24 | PI0.5 从 pi0 演化到本地后训练 LIBERO 评估全记录 |
| 25 | ACT 模型架构、数据集、训练与评估指南（现行主文档） |

## 已知坑（实测沉淀）

- `31_eval_act.sh` 必须带 `--eval.use_async_envs=false`：gym_aloha 0.1.4 无 entry_points 注册，
  多进程子进程报 `NamespaceNotFound`（脚本内已处理）；
- 评估默认 `MUJOCO_GL=egl` 无头渲染，带显示器可 `RENDER=glfw`；
- ACT 评估不需要 `--rename_map`（pi0 系列才需要）；
- `data/` 视频为 AV1 编码：训练加载正常，本地播放器可能打不开（属预期）；
- 下载优先走 ModelScope（`11_download_act_assets.sh`），国内网络友好。
