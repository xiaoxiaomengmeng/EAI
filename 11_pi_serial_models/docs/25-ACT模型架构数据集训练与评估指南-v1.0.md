# ACT 模型架构、数据集、训练与评估指南（5090 机 · py3.12 · LeRobot main v0.6.2 · ModelScope 国内镜像）

> 版本 v1.0 · 2026-08-29 · 目标硬件：5090 台式机（RTX 5090 32GB，Ubuntu 22.04）· 技术路线：复用本项目 `.venv312` + `repos/lerobot-main`（实测 v0.6.2），**无需新建目录、无需下载基座权重**
>
> ACT（Action Chunking with Transformers）是 LeRobot 官方推荐的**入门首选策略**：训练快、参数量小（约 52M）、50 条演示即可达到高成功率。本文介绍其模型架构、数据集、训练方法与测评方法，所有资源下载均走 **ModelScope 国内镜像**（本机无法直连 huggingface.co，ModelScope 链接均于 2026-08 实测可访问）。

---

## 〇、总览

**目标**：在现有 `11_pi_serial_models` 目录与 `.venv312` 环境内，完成 ACT 策略在 `aloha_sim_transfer_cube_human` 数据集上的**从零训练**，并在 ALOHA 仿真双臂环境（gym-aloha，✅ 已装）中 rollout 评估成功率。

**与 pi0 路线的关键差异**：

| 维度 | pi0（已有文档 17/18/24） | ACT（本文） |
|------|------------------------|------------|
| 模型规模 | 约 3.5B（PaliGemma 3B + Action Expert） | 约 52M |
| 权重来源 | 需下载 14GB 基座权重（ModelScope `lerobot/pi0_base`） | **从零训练，无需任何预训练权重** |
| 训练方式 | LoRA / Action Expert 微调 | 全参数直接训练 |
| 训练耗时 | 长（小时级～天级） | 单卡数小时内（官方 80k 步 A100 约 1h45m） |
| 显存压力 | 32GB 需 LoRA + 内存自检 | 32GB 绰绰有余，可开大 batch |

**资源清单（ModelScope 镜像，均实测可用）：**

| 资源 | ModelScope 地址 | 说明 |
|------|----------------|------|
| ALOHA 搬方块数据集（主） | <https://modelscope.cn/datasets/lerobot/aloha_sim_transfer_cube_human> | ✅ 本地已有（`data/`，v3.0 格式，50 episodes / 20000 帧） |
| ALOHA 搬方块（脚本化演示） | <https://modelscope.cn/datasets/lerobot/aloha_sim_transfer_cube_scripted> | 同任务的脚本演示版 |
| ALOHA 插孔数据集（人类演示） | <https://modelscope.cn/datasets/lerobot/aloha_sim_insertion_human> | 精细装配任务 |
| ALOHA 插孔（脚本化演示） | <https://modelscope.cn/datasets/lerobot/aloha_sim_insertion_scripted> | 同上 |
| PushT 数据集 | <https://modelscope.cn/datasets/lerobot/pusht> | 单臂推块轻量验证任务 |
| 官方预训练 ACT 策略 | <https://modelscope.cn/models/lerobot/act_aloha_sim_transfer_cube_human> | 80k 步训好，可直接评估对比（约 200MB） |

> HF 原始地址（仅作存档，本机直连超时）：数据集 `huggingface.co/datasets/lerobot/aloha_sim_transfer_cube_human`、模型 `huggingface.co/lerobot/act_aloha_sim_transfer_cube_human`。

**一键下载脚本**（本指南配套）：

```bash
bash scripts/11_download_act_assets.sh                          # 仅主数据集（已存在则跳过）
ALL_DATASETS=true bash scripts/11_download_act_assets.sh        # 全部 5 个数据集
PRETRAINED=true bash scripts/11_download_act_assets.sh          # 额外下载官方预训练 ACT 模型
```

---

## 一、模型架构

**论文**：[Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/abs/2304.13705)（Zhao et al., 2023，即 ALOHA 论文）；本仓库实现见 [src/lerobot/policies/act/modeling_act.py](../repos/lerobot-main/src/lerobot/policies/act/modeling_act.py)。

### 1.1 核心思想

1. **Action Chunking（动作分块）**：不逐帧预测单个动作，而是一次性预测未来 `chunk_size` 步的动作序列（默认 100 步），执行时按 `n_action_steps` 消费后再基于新观测重新预测。这消除了逐帧预测的累积抖动，使动作连贯、适合精细接触类操作。
2. **CVAE（条件变分自编码器）风格变量**：人类演示存在多模态性（同一场景有多种同样正确的操作方式）。训练时用 **VAE 编码器**读取"整段未来动作"编码出潜变量 `z`（风格变量），让模型学会"哪种演示风格"；**推理时 `z` 置零**，模型输出平均意义上的动作。损失 = 重建损失（L1）+ `kl_weight` × KL 散度（默认 `kl_weight=10.0`）。
3. **Temporal Ensemble（时序集成，可选）**：多个时间步的重叠预测块按指数加权平均，进一步平滑动作（`temporal_ensemble_coeff`，默认关闭）。

### 1.2 结构组件（对应本仓库代码类）

```
输入：多路相机图像（如 observation.images.top）+ 机器人关节状态（observation.state, 14 维）

① 视觉骨干  torchvision ResNet-18（ImageNet 预训练权重，FrozenBatchNorm2d，取 layer4 特征图）
             —— 每路相机独立提取，类 DETR 的 IntermediateLayerGetter

② VAE 编码器（仅训练时用）
   [cls token, (关节状态), chunk_size 个未来动作] → ACTEncoder
   → 输出 μ、σ → 采样潜变量 z（latent_dim，默认 32）

③ Transformer 编码器
   输入 token：[z, (关节状态), (环境状态), 各相机特征图像素]
   —— 1D 特征用可学习 Embedding 位置编码，图像特征用二维正弦位置编码

④ Transformer 解码器
   chunk_size 个可学习查询（类 DETR object queries，位置编码为 nn.Embedding）
   对编码器输出做 cross-attention

⑤ 动作回归头  nn.Linear(dim_model → 动作维度 14)
   输出：未来 100 步 × 14 维关节动作
```

推理时跳过 ②，`z` 置零直接进入 ③→④→⑤。

### 1.3 关键默认超参数（`configuration_act.py`）

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `chunk_size` | 100 | 每次预测的动作块长度（步） |
| `n_obs_steps` | 1 | 输入观测步数（当前仅支持 1） |
| `latent_dim` | 32 | 风格潜变量维度 |
| `use_vae` | true | 启用 CVAE 训练目标 |
| `kl_weight` | 10.0 | KL 损失权重：`loss = 重建损失 + kl_weight × kld_loss` |
| `temporal_ensemble_coeff` | None | 时序集成系数（与多步执行互斥） |
| `vision_backbone` | resnet18 | 视觉骨干 |

---

## 二、数据集

### 2.1 本地已有（无需下载）

`data/aloha_sim_transfer_cube_human/`：**LeRobot v3.0 格式**（✅ 与当前 lerobot-main 兼容），ALOHA 双臂搬方块任务：

- 50 episodes × 400 帧 = 20,000 帧，fps=50
- `observation.images.top`：480×640×3 顶视相机（av1 视频编码）
- `observation.state` / `action`：14 维（左右臂各 6 关节 + 1 夹爪）
- 任务语义：一只臂抓起方块，传递给另一只臂

### 2.2 其他可选数据集

全部可从 ModelScope 镜像下载（见〇节资源表与 `scripts/11_download_act_assets.sh`）。对应仿真评估任务：

| 数据集 | 评估环境任务 |
|--------|-------------|
| aloha_sim_transfer_cube_* | `AlohaTransferCube-v0` |
| aloha_sim_insertion_* | `AlohaInsertion-v0` |
| pusht | `PushT-v0` |

### 2.3 下载后验证

```bash
source .venv312/bin/activate
python scripts/02_test_dataset.py   # 或运行 11_download_act_assets.sh 末尾的自动验证
```

---

## 三、训练方法

### 3.1 训练命令（推荐直接用配套脚本 `scripts/21_train_act.sh`，已实测）

```bash
source .venv312/bin/activate
bash scripts/21_train_act.sh                          # 默认 100000 步、batch 8（含训练前自检/日志落盘）
STEPS=200 bash scripts/21_train_act.sh                # 冒烟测试（已实测通过：5 步链路无报错）
STEPS=80000 BATCH_SIZE=16 bash scripts/21_train_act.sh  # 官方步数 + 大 batch（32GB 显存可上探）
```

手动执行时的等价命令（ACT **从零训练**，不需要基座权重，也不需要 `HF_HUB_OFFLINE` 之类的外部模型缓存）：

```bash
source .venv312/bin/activate

lerobot-train \
  --dataset.repo_id=aloha_sim_transfer_cube_human \
  --dataset.root="$HOME/ws/11_pi_serial_models/data/aloha_sim_transfer_cube_human" \
  --policy.type=act \
  --output_dir=outputs/act_aloha_cube \
  --job_name=act_aloha_cube
```

关键参数说明：

| 参数 | 说明 |
|------|------|
| `--dataset.root` | 本地数据集目录（含 `meta/info.json`），与 `22_train_pi0.sh` 一致，不触网 |
| `--policy.type=act` | 选用 ACT 策略（零预训练权重） |
| `--steps` | 训练总步数（v0.6.2 为顶层参数），默认 100,000，与官方 train_config.json 完全一致（官方发布模型取的是 80k 步 checkpoint） |
| `--batch_size` | 默认 8，与官方一致；32GB 显存可上探 16~32（会偏离官方配置） |
| `--wandb.enable=true` | 可选：loss 曲线可视化（国内用 `WANDB_MODE=offline`） |

官方模型卡参考（`lerobot/act_aloha_sim_transfer_cube_human` README 与其 `train_config.json`）：配置为 **100,000 步 / batch 8**（发布模型取 80k 步 checkpoint），A100 上约 1 小时 45 分钟。**本脚本默认参数（100k 步 / batch 8 / chunk 100 / kl 10.0 / lr 1e-5）与官方 train_config.json 逐项一致**，仅 `log_freq`（100 vs 200，只影响日志密度）与 wandb 开关不同。
本机实测（RTX 5090）：可学习参数 52M，batch 8 稳定后约 **36 step/s**（首步含编译约 3.5s），**100k 步约 46 分钟**；首次训练会自动下载 ResNet-18 ImageNet 骨干（~45MB，脚本已预下载处理，实测 pytorch.org 可达）。

### 3.2 产物位置

训练产物与 pi0 路线结构一致：

```
outputs/act_aloha_cube/
├── checkpoints/
│   ├── 000xxx/pretrained_model/     # 周期 checkpoint（config.json + model.safetensors）
│   └── last/pretrained_model/       # 最终模型，评估用这个
└── logs/
```

### 3.3 直接使用官方预训练模型（跳过训练）

若只想先验证评估链路，可下载官方 80k 步模型（见〇节 `PRETRAINED=true`），评估时把 `--policy.path` 指向 `models/act_aloha_sim_transfer_cube_human` 即可。

### 3.4 Loss 曲线（离线方案，已实测）

脚本默认不开 wandb（离线 run 在国内无法直接查看）。`21_train_act.sh` 已把指标行按 `--log_freq`（`LOG_FREQ` 环境变量，默认 100）落盘到 `logs/train_act_<时间戳>.log`，训练后运行配套画图脚本即可：

```bash
python scripts/41_plot_train_log.py        # 自动解析最新 train_act 日志 → logs/<日志名>_loss_curve.png
```

输出含 `loss` / `l1_loss`（重建）/ `kld_loss` 三条曲线。冒烟实测：200 步内 loss 41.7 → 3.7，下降正常。

三个指标的含义（ACT 损失 = `l1_loss + kl_weight × kld_loss`，`kl_weight=10.0`）：

| 指标 | 含义 | 怎么看 |
|------|------|--------|
| `l1_loss` | 重建损失：预测动作块与演示动作的 L1 距离（归一化空间） | **核心指标**，越小模仿越准；100k 步实测 0.68 → 0.046 |
| `kld_loss` | KL 散度：CVAE 潜变量 `z` 分布与标准正态的距离（正则项） | 训练初期大、后期趋近 0 属正常 |
| `loss` | 总损失 = `l1_loss + 10 × kld_loss` | 量级约为 l1 的 10 倍以上，故画图用三子图分开看 |
若需要交互式实验跟踪，仍可 `WANDB_ENABLE=true`（默认 `WANDB_MODE=offline`，后续需 `wandb sync` 才能查看，国内不便）。

---

## 四、测评方法

### 4.1 评估环境

仿真环境为 **gym-aloha**（MuJoCo）：✅ 本项目 `.venv312` 已安装 `[aloha]` extra 并做过冒烟测试（实测 351 FPS），无需再装。**推荐直接用配套脚本 `scripts/31_eval_act.sh`（已实测）**：

```bash
source .venv312/bin/activate
bash scripts/31_eval_act.sh                       # 默认 25 轮，无头 EGL 渲染
N_EPISODES=5 bash scripts/31_eval_act.sh          # 快速验证
N_EPISODES=500 EVAL_BATCH=16 bash scripts/31_eval_act.sh   # 官方模型卡配置
POLICY_DIR=models/act_aloha_sim_transfer_cube_human bash scripts/31_eval_act.sh  # 评估官方预训练模型
```

等价手动命令（评估入口 `lerobot-eval`，v0.6.2 实测参数，与 `32_eval_pi0.sh` 同源）：

```bash
source .venv312/bin/activate
export MUJOCO_GL=egl   # 无头/SSH 场景用 EGL GPU 渲染；带显示器可改 glfw

lerobot-eval \
  --policy.path="$HOME/ws/11_pi_serial_models/outputs/act_aloha_cube/checkpoints/last/pretrained_model" \
  --env.type=aloha \
  --env.task=AlohaTransferCube-v0 \
  --eval.n_episodes=25 \
  --eval.batch_size=1 \
  --output_dir=outputs/eval_act_aloha_cube \
  --job_name=eval_act_aloha_cube
```

参数要点（与 `32_eval_pi0.sh` 对齐）：

- `--policy.path` 评估本地 checkpoint，策略类型从 checkpoint 的 `config.json` 自动识别（与 `--policy.type` 互斥）；
- `--env.type=aloha` + `--env.task=AlohaTransferCube-v0`：v0.6.2 新写法（v0.4.4 的 `--env.name` 已废弃）；
- **ACT 不需要 `--rename_map`**（pi0 才需要）：ACT 从零训练，视觉特征名直接取自数据集 `observation.images.top`，与 gym-aloha 环境输出一致；
- **`--eval.use_async_envs=false` 必带**（实测坑）：gym_aloha 0.1.4 无 gymnasium entry_points 注册，环境注册仅靠主进程 `import gym_aloha`；batch>1 默认启用 AsyncVectorEnv，其 forkserver 子进程不继承该 import，报 `NamespaceNotFound: gym_aloha`。改 Sync 后在主进程建环境，实测 batch 16 约 2.6s/轮（500 轮约 22 分钟）；
- 本机实测：100k 步训练产物 **500 轮官方口径成功率 79.0%**（395/500，耗时 22.8 分钟 ≈ 2.73s/轮；小样本 5/16 轮时的 56~60% 偏差属采样噪声，官方 500 轮为 83%）。

### 4.2 成功判定与验收标准

- **单轮成功判定**：方块被一只臂抓起并成功交到另一只臂（gym-aloha 内置判定；注意其对"对侧夹爪形成对握接触"即记成功的宽松判据）；
- **验收建议**：与本项目 pi0 指南一致，先跑 **25 轮**统计成功率；官方模型卡以 **500 轮**报告，成功率 **83.0%**（LeRobot 版）对比原 ACT 仓库实现的 68.0%；
- **结果查看**：日志末尾的 rollout 汇总（`pc_success` 等指标）；每轮视频默认保存到
  `outputs/eval_act_aloha_cube/videos/aloha_*/eval_episode_N.mp4`，人工抽检方块交接动作。

### 4.3 官方参考评估命令（500 轮 · 大批量并行）

官方模型卡所用配置（5090 32GB 可尝试把 `--eval.batch_size` 从 50 降到 8~16）：

```bash
lerobot-eval \
  --policy.path=outputs/act_aloha_cube/checkpoints/last/pretrained_model \
  --env.type=aloha --env.task=AlohaTransferCube-v0 \
  --eval.n_episodes=500 --eval.batch_size=16 \
  --output_dir=outputs/eval_act_aloha_cube_500
```

### 4.4 本机 500 轮验收实测记录（2026-08-29）

对 100k 步从零训练的产物执行 `N_EPISODES=500 EVAL_BATCH=16 bash scripts/31_eval_act.sh`，
日志：`logs/eval_act_20260829_201746.log`。评估结果总览：

| 项目 | 数值 |
|---|---|
| **成功率 `pc_success`** | **79.0%**（395/500 成功） |
| 平均累计奖励 `avg_sum_reward` | 217.99（满分 400，每步最高 +1） |
| 平均单轮最高奖励 `avg_max_reward` | 3.466（>3 说明交接+放置环节大多完成） |
| 总耗时 `eval_s` | 1367.4 秒 ≈ **22.8 分钟** |
| 单轮耗时 | 2.73s/轮 |
| 配置 | `batch_size=16`、`n_episodes=500`、`use_async_envs=False`（修复后生效）、每轮上限 400 步、480×640 像素观测 |
| 视频 | 500 个 `eval_episode_N.mp4` 已保存至 `outputs/eval_act_aloha_cube/videos/aloha_0/` |

结论：

1. **与官方基线基本对齐**：官方 500 轮 83.0%，本机 100k 步从零训练产物 **79.0%**，差 4 个百分点属正常波动（单卡数值差异、随机种子、官方发布模型取 80k checkpoint），属验收通过水平；
2. **小样本是采样噪声**：5 轮 60%、16 轮 56.25% 样本太小，500 轮才是官方口径，这也印证模型卡必须用 500 轮报告；
3. **日志无异常**：全程无 ERROR/Traceback，仅一条无害警告（`The environment does not have 'task_description'`——ACT 不用语言指令，可忽略），进度条末尾成功率与最终汇总一致。

---

## 五、快速上手流程（TL;DR）

```bash
cd ~/ws/11_pi_serial_models
source .venv312/bin/activate

# ① 数据（本地已有；需要更多数据集/官方预训练模型时执行）
bash scripts/11_download_act_assets.sh                    # ALL_DATASETS=true / PRETRAINED=true 可选

# ② 训练（从零训练，默认 100k 步；推荐用配套脚本，等价手动命令见第三节）
bash scripts/21_train_act.sh

# ③ 评估（25 轮，无头 EGL；推荐用配套脚本，等价手动命令见第四节）
bash scripts/31_eval_act.sh
# 正式验收（500 轮官方口径，约 22 分钟；本机实测成功率 79%，见 4.4 节）
N_EPISODES=500 EVAL_BATCH=16 bash scripts/31_eval_act.sh
```

---

## 附：初学者术语速查

| 缩写/术语 | 全称与含义 |
|-----------|-----------|
| ACT | Action Chunking with Transformers，动作分块 Transformer 策略 |
| ALOHA | ACT 的出处论文与双臂硬件平台；ACT 就是为 ALOHA 低成本双臂精细操作设计的策略 |
| ResNet | 残差网络，带残差跳连的 CNN 图像骨干；跳连让深达上百层的卷积网络也能稳定训练 |
| ResNet-18 | ResNet 的 18 层版本，参数约 11.7M；ACT 的默认视觉骨干，加载 ImageNet 预训练权重，卷积参与训练、BN 统计量冻结 |
| DETR | Detection Transformer，目标检测模型；ACT 解码器的可学习查询 object queries 与 encoder-decoder 结构源自它 |
| CVAE | Conditional Variational Auto-Encoder，条件变分自编码器；ACT 用它编码演示风格潜变量 `z` |
| Action Chunking | 一次预测未来若干步动作（默认 100 步）并按块执行的机制 |
| z / latent | 风格潜变量（默认 32 维）：训练时由未来动作编码得到，推理时置零 |
| chunk_size | 动作块长度，ACT 默认 100 |
| kl_weight | KL 散度损失权重（默认 10.0），约束 `z` 分布接近标准正态 |
| Temporal Ensemble | 对重叠预测块做指数加权平均的平滑技巧 |
| gym-aloha | ALOHA 双臂仿真 Gym 环境包（MuJoCo），提供 AlohaTransferCube-v0 / AlohaInsertion-v0 |
| rollout | 策略在仿真环境中闭环执行一整轮并统计成败的评估方式 |
| episode | 一轮完整的任务演示/评估轨迹（本数据集每轮 400 帧 @ 50fps） |
| ModelScope | 魔搭社区，国内模型/数据集托管平台，本项目所有下载的首选镜像 |

### 200 字摘要

ACT即Action Chunking with Transformers，源自ALOHA论文，面向低成本硬件的双臂精细操作。核心是Encoder-Decoder结构的Transformer：ResNet-18提取各相机图像特征，与关节状态及风格潜变量z拼接送入编码器，解码器用chunk_size个可学习查询一次性预测未来100步动作，即动作分块，消除逐帧预测的累积抖动；训练引入CVAE把整段未来动作编码为风格变量z并加KL散度约束，建模同一场景多种正确操作的演示多模态，推理时z置零输出平均意义动作；重叠预测块可用Temporal Ensembling按指数加权平均进一步平滑。ACT从零训练无需预训练权重，可训参数约52M，本机5090训练100k步约46分钟，在gym-aloha搬方块任务500轮评估成功率79.0%，与官方83.0%基线基本对齐，是入门机器人模仿学习性价比最高的策略。和PI0相比，ACT是小模型、从零训练、无语言的专才。


ACT的本质是解决模仿学习两大痛点：人类演示天然抖动，且同一场景存在多种合理操作。它放弃逐帧闭环预测，让解码器一次并行回归未来100步动作，策略学的是轨迹的时序结构而非单步反应，误差不再累积，动作自然连贯；再用CVAE把整段动作编码成风格变量z建模演示多模态，推理时置零取平均走法。架构是DETR式encoder-decoder加ImageNet预训练的ResNet-18视觉骨干，总参仅52M。5090从零训100k步约46分钟，搬方块500轮79.0%贴近官方83.0%——50条演示加单卡一小时，就能换来流畅的双臂精细操作策略。和PI0相比，ACT是小模型、从零训练、无语言的专才。

### 视频号字幕版

ACT源自ALOHA论文，解决模仿学习两大痛点：人类演示天然抖动，同一场景又有多种合理操作。ACT放弃逐帧闭环预测，让解码器一次并行回归未来100步动作，策略学的是整段轨迹的时序结构，误差不再累积，动作自然连贯；再用CVAE风格变量z建模演示多模态，推理时z置零，输出平均走法。骨架是DETR式encoder-decoder，视觉端挂ImageNet预训练的ResNet-18，全部参数仅52M。单张5090从零训练100k步只要46分钟，搬方块任务500轮成功率79.0%，贴近官方的83.0%。50条演示，单卡一小时，就能换来流畅的双臂精细操作。和PI0相比，ACT是小模型、从零训练、无语言的专才。


### ACT 与 PI0 / PI0.5 的最大区别

| 维度 | ACT | PI0 | PI0.5 |
|------|-----|-----|-------|
| 定位 | 单任务模仿学习入门策略 | 通用 VLA 基座 | 通用 VLA 泛化升级版 |
| 总参数 | 约 52M（ResNet-18 约 11M + Transformer 约 40M） | 约 3.5B（PaliGemma 2B + Expert 300M） | 约 4.14B（PaliGemma 3B + Expert 300M） |
| 预训练依赖 | 策略部分从零训练，仅 ResNet-18 加载 ImageNet 骨干 | 需下载 14GB 大规模真机数据预训练基座 | 异构 6 类数据 co-training 底座（约 168k 条） |
| 微调方式 | 全参直接训练 | LoRA 或冻结 PaliGemma 只训 Expert | 同 PI0（LoRA / Expert-only） |
| 动作生成 | 单次前向直接回归 100 步动作块 | Flow Matching 去噪出 50 步块 | Flow Matching 出 10 步块 + deque 队列消费 |
| 多模态演示建模 | CVAE 风格变量 z，推理时置零 | Flow Matching 概率分布天然建模 | 同 PI0 |
| 语言条件 | **无**（不吃任务指令，评估时有无害警告可忽略） | 48 token 任务描述 | 200 token 长指令 + subtask 预测 |
| 本机训练成本（5090） | 100k 步约 46 分钟 | LoRA 160k 步 4.5 小时；Expert 200k 步 10 小时 | 30k 步（LoRA / Expert 均数小时级） |
| 本机成功率 | 搬方块 500 轮 79.0%（官方 83.0%） | ALOHA 搬方块 25 轮 16%（Expert 全参） | LIBERO spatial 500 集 98.4% |
| 适用场景 | 小数据单任务、快速跑通链路 | 语言跟随 + 多任务泛化 | out-of-the-box 跨本体跨环境泛化 |

一句话总结最大区别：**ACT 是「小模型 + 从零训练 + 无语言」的专才**，50 条 demo 就能在单一任务上训出可用策略；**PI0 / PI0.5 是「大模型 + 预训练底座 + 语言条件」的通才**，靠 VLA 架构和 Flow Matching 吃语言指令、做多任务泛化，PI0.5 再用异构 co-training 把泛化做到开箱即用。学原理、验链路用 ACT；要语言理解、跨任务泛化就上 PI 系列。
