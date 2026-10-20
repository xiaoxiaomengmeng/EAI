# UnifoLM-WMA-0 架构与关键技术解析（初稿 v0.1）

> 定位：[docs/01](01-UnifoLM-WMA学习笔记.md) 是学习笔记（概念入门、参数解释、使用记录），本文是**工程视角**的架构与关键技术解析——技术栈分层、参数决策逻辑、RTX 5090 适配本质、与 RL 路线对照（呼应 [docs/03](03-unitree_rl_mjlab架构与关键技术解析.md)）。
> 证据约定（重要）：本子项目的源码 / .venv / 权重 / 推理输出**已于本机清理**（2026-10-20 核实：`repos/unifolm-world-model-action/` 为空目录，子模块未 checkout）。本文基于 docs/01 与 README 的第一手运行记录整理——标「实测」= 当时的真实运行数据（可信）；标「待核」= 源码级细节，子模块恢复后需核实。

## 1. 技术栈总览

| 层 | 技术 | 在本项目中的作用 |
|---|---|---|
| 生成范式 | Latent Video Diffusion（SD 系 UNet） | 世界模型主干：图像+指令 → 未来视频 |
| 视觉条件编码 | DINO + SigLIP 双 ViT（`dinosiglip_vit.py`） | 初始观测帧 → 语义特征（待核） |
| 条件注入 | MLPProjector + Cross-Attention | 语言指令 / 机器人状态投影后注入 UNet |
| 动作头 | Conditional UNet1D（Diffusion Policy 路线） | 与视频 UNet **并行去噪**生成 16 步关节动作 |
| 采样器 | DDIM（50 步，eta 1.0） | 1000 步噪声调度的子采样加速 |
| 数据格式 | LeRobot V2.1 + dlimp 子模块 | 训练数据加载（推理时仅用 H5 归一化统计） |
| 运行栈 | Python 3.10 venv + torch 2.11.0+cu128 + xformers | 本机 fork 含 sm_120 适配五修复 |

数据流一行图：

```
初始图像 + 指令 + 机器人状态
   ├─► DINO+SigLIP / MLP 编码 ──► cross-attn 条件注入 ─┐
   ├─► VAE encode ──► z_T ─► WM UNet 去噪 ×50 ─► z_0 ─► VAE decode ─► 未来 16 帧
   └─► Diffusion Head（UNet1D）并行去噪 ─► 16 步关节动作 ─┐
                                                        │
        ┌───────────────────────────────────────────────┘
        ▼
   Sim 模式：动作 → 预测执行后视频 + 新状态 → 最后一帧反馈 → 下一轮 DM（×12）
```

## 2. 核心架构：一个 UNet、两种模式、并联动作头

**「双模式共享主干」是本架构最巧的设计**（源码视角，细节待核）：

| 模式 | 条件 | 输出 | 角色 |
|---|---|---|---|
| DM（决策） | 观测 + 指令 + 状态 | 动作序列 + 决策视角未来帧 | 「想象自己怎么动」 |
| Sim（仿真） | 观测 + **动作** + 状态 | 环境变化视频 + 新状态 | 「想象执行后世界什么样」 |

同一个 WM UNet，**换条件即换角色**——世界模型不是外挂的视频生成器，而是策略的「想象力」组件。

**并联动作头**：Conditional UNet1D 与视频 UNet 共享条件编码、各自维护噪声、DDIM 并行去噪——动作与视频在同一扩散过程中对齐，这是「视频预测辅助决策」的实现基础（区别于「先生成视频再反推动作」的两段式）。

**自回归闭环（实测跑通 12 轮）**：DM 产动作 → Sim 拿动作产未来帧+新状态 → 最后一帧+状态作为下一轮 DM 输入。长程 rollout 不崩溃（12 iter / 192 帧连贯）说明模型学到的是物理交互因果，不是帧间外推——这是当时实测最重要的观察。

## 3. 潜空间与采样参数体系

**latent 形状（修正 01 笔记的示意）**：实际推理用 `--height 320 --width 512`，VAE 8× 下采样后 z_t ≈ `[1, 4, 16, 40, 64]`（16 帧 × 40×64）；01 笔记中的 32×32 是按 256×256 的通用示意，本机参数下是 40×64。

**噪声调度**：linear beta 0.00085 → 0.012（SD 惯例）、训练调度 1000 步、DDIM 推理取 50 个子步——50 步是质量/耗时的平衡点（实测单 iter 33s 的大头）。

**本机参数的决策逻辑（工程视角读参数）**：

| 参数 | 值 | 为什么这么定 |
|---|---|---|
| `unconditional_guidance_scale` | 1.0 | **不用 CFG**——条件已通过 cross-attention 充分注入，省一半推理量（CFG 需 cond/uncond 双前向） |
| `guidance_rescale` | 0.7 | 为启用 CFG 时的过饱和防护预留 |
| `ddim_eta` | 1.0 | 保留最大随机性——世界模型要「想象」多样性，确定性输出会单调 |
| `timestep_spacing` | uniform_trailing | 尾部均匀采步，低噪声端更密（细节生成稳） |
| `frame_stride` | 2~6 按数据集 | 跨帧步长 = 时间压缩比：stride 6 时 16 帧覆盖 96 原始帧，长程任务必配大 stride |
| `perframe_ae` | 开 | VAE 逐帧编解码，显存峰值 20GB（5090 32GB 内从容） |

## 4. 动作生成与滚动执行

- **动作空间**：关节角度连续向量（Z1 臂 / G1 全身），每次生成 `n_action_steps=16` 步。
- **滚动执行**：`exe_steps=16` = 全执行再重规划——每轮把 16 步全部「想象执行」完，用预测结果重开下一轮。若 exe_steps < n_action_steps 则是 MPC 式滚动窗口（执行部分、重算剩余），本机配置未用。
- **归一化约定（对齐红线）**：H5 文件携带该数据集动作/状态的 **min/max 归一化统计**，模型输出 [0,1]（或 [-1,1]）区间值，反归一化后才真实关节角。**换数据集必须换统计文件**——这与 RL 侧的观测归一化（docs/03 §6）是同一类坑：统计量与数据必须成对。

## 5. RTX 5090 (sm_120) 适配：五修复的工程本质

官方代码基于 A100/H100 (sm_80/90) + torch 2.3.1 + xformers 开发，上 Blackwell 的五处修复可归纳为**三板斧**（细节见 README §5）：

| # | 问题 | 修复 | 归类 |
|---|---|---|---|
| 1 | `requires-python ==3.10.18` 拒装 | 放宽 `>=3.10` | 依赖链 pin |
| 2 | xformers fa2 不支持 fp32 (sm_120) | 入口检测 dtype → bf16 cast，算完还原 | dtype cast |
| 3 | xformers fa2 不支持 tensor attn_bias | 该处换 PyTorch 原生 `scaled_dot_product_attention` | SDPA 兜底 |
| 4 | `torchvision.io.write_video` 已移除 | 改 `imageio.mimwrite` | API 演进 |
| 5 | 旧 Python 缓存干扰（`.pyc`/`__pycache__`） | 清除缓存 | 环境卫生 |

另有一条**依赖链 pin**（README「依赖安装注意事项」）：tf 2.13.1 兼容 numpy 1.x + `tensorflow-graphics --no-deps` + 事后拉回 typing-extensions——属安装流程而非代码修复。

**本质**：Blackwell 适配三板斧 = **dtype cast（bf16 化）/ SDPA 兜底（绕开 xformers kernel 缺口）/ 依赖链 pin**。其中修复 2/3 有共性——xformers 对新架构的 kernel 支持总是滞后，PyTorch 原生 SDPA 是随时可用的保底。修复 2 还顺带是性能优化（bf16 本就更快）。

**实测性能（历史 run）**：12 iter × 50 DDIM 步共 6 分 40 秒（~33s/iter）、显存 ~20GB——5090 跑 16 帧 320×512 视频扩散是宽裕的。

## 6. 与 RL 路线的对照（呼应 docs/03）

同一台 5090 上跑的两条具身技术路线，本工作区各有一个实证：

| 维度 | WMA（本篇，扩散世界模型） | mjlab+PPO（docs/03，仿真 RL） |
|---|---|---|
| 动作怎么来 | 扩散去噪一次生成 16 步序列 | 高斯策略逐步采样（50Hz 闭环） |
| 训练信号 | 视频扩散损失（自监督） | reward（任务塑形） |
| 数据来源 | 真实遥操作数据集（宇树 5 集） | 仿真自生成（2048 envs × 87k steps/s） |
| 物理合理性 | 从视频先验隐式学（长程会漂） | MuJoCo 显式保证（接触/动力学精确） |
| 动作精度 | 受限（视频生成是间接通道） | 关节级精确（实测稳定行走/舞蹈） |
| 泛化来源 | 大规模视频先验 → **语义泛化**（没见过的物体也能想象） | 域随机化 → **物理泛化**（参数漂移下稳） |
| 部署形态 | ckpt 离线推理（33s/iter，规划节奏） | ONNX → C++ 50Hz 实时控制 |
| 计算特征 | 显存重（20GB）、IO 轻 | GPU 吞吐重、显存中（~15GB） |

**互补定位**（呼应 01 笔记 §11 的「大脑/小脑」）：WMA 强在「看懂场景 + 想象结果」——语义层泛化是 RL 永远学不到的（仿真里没有 Open-X 级的视觉多样性）；RL 强在「动作精确 + 物理可信」——扩散动作头的精度天花板受视频分辨率约束。两者的混合（OmniA 方向）：VLA/WMA 出目标与想象，RL 出可执行的精确控制。

## 7. 本机实测数据速查（历史 run，2026-10 记录）

| 指标 | 数值 |
|---|---|
| GPU / 显存 | RTX 5090 / ~20GB（perframe_ae 开启） |
| 单轮迭代 | ~33s（DDIM 50 步 × DM+Sim 双模式） |
| 12 轮任务总耗时 | ~6 分 40 秒 |
| 输出 | ~60MB / 35 个 mp4（wm+dm 各 12 段 + 全景拼接） |
| 任务 | G1 装相机入盒（11 iter, fs6）、Z1 双臂收铅笔（8, fs4）、Z1 叠盒（12, fs2）等 5 数据集 |

## 8. 资产现状与恢复路径（2026-10-20 核实）

| 资产 | 状态 | 恢复命令（如需重跑） |
|---|---|---|
| 源码（fork，含 5090 修复） | ❌ 目录空、子模块未 checkout | `cd ~/ws && git submodule update --init --recursive 104-unitreerobotics/repos/unifolm-world-model-action` |
| .venv（py3.10 + torch2.11+cu128） | ❌ 已删 | README §3 重建流程（约 1 小时） |
| 模型权重 unifolm_wma_dual.ckpt（16.7G） | ❌ 已删 | `modelscope download --model unitreerobotics/UnifoLM-WMA-0-Dual` |
| 推理输出（wm/dm 视频） | ❌ 已删 | 重跑 `scripts/run_wma_interaction.sh`（恢复前三样后） |

⚠️ 连带影响：START_HERE.md 的「A（世界模型推理）」条目描述的环境/权重路径当前均不存在——文档先于资产保留（记录价值），但**按 A 条目操作会失败**，需用户决定标注「已清理归档」还是择机恢复。

---

> 初稿说明：§3/§5/§6/§7/§8 基于本机第一手记录（docs/01 + README + 参数表）；§2/§4 的源码级机制描述（并联去噪的条件共享方式、UNet1D 的 global cond 来源）标「待核」，子模块恢复后可定位 `models/ddpms.py`、`models/diffusion_head/`、`scripts/evaluation/world_model_interaction.py` L552-641 核实。结构按「总览 → 架构 → 潜空间/参数 → 动作 → 适配 → 路线对照 → 数据 → 资产」组织。
