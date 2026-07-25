# UnifoLM-WMA 世界模型交互推理 — 学习笔记

> 第一次运行 UnifoLM-WMA-0-Dual 世界模型交互推理的学习记录。
> 本文解释输入数据、推理参数、输出结构，以及背后的核心概念。

---

## 1. 模型是什么？

**UnifoLM-WMA**（World Model + Action）是宇树科技发布的具身智能基础模型，核心能力：

- **世界模型（World Model）**：给定当前观测图像 + 文本指令，预测未来环境变化的视频
- **动作生成（Action）**：同时输出机器人应执行的动作序列（关节角度/末端位姿）
- **Dual 模式**：同时生成视频预测（wm）和决策视频（dm），形成"想象→行动"闭环

模型架构基于 **Latent Diffusion**（潜空间扩散），在压缩的视频 latent 空间上做去噪生成。

---

## 2. 输入数据

示例数据位于 `examples/world_model_interaction_prompts/`：

```
world_model_interaction_prompts/
├── unitree_z1_stackbox.csv          # 数据集描述文件
├── images/
│   └── unitree_z1_stackbox/
│       └── 0.png                    # 初始观测图像（第一帧）
└── transitions/
    └── unitree_z1_stackbox/
        ├── 0.h5                     # 动作/状态归一化统计量
        └── meta_data                # 元数据
```

### CSV 文件字段

| 字段 | 含义 | 示例 |
|------|------|------|
| `videoid` | 样本 ID | `0` |
| `data_dir` | 数据集子目录名 | `unitree_z1_stackbox` |
| `instruction` | 自然语言任务指令 | "Stack the blocks: red at bottom, yellow middle, green on top." |
| `embodiment` | 机器人本体类型 | `Unitree Z1 Robot Arm` |
| `fps` | 原始数据帧率 | `30` |

### 初始图像

`images/{dataset}/{videoid}.png` — 机器人摄像头的初始观测帧，作为世界模型生成的起点。

### H5 文件

`transitions/{dataset}/{videoid}.h5` — 包含该数据集的动作/状态归一化参数（min/max），
用于将模型输出的归一化动作反归一化为真实关节值。

---

## 3. 推理参数详解

```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/evaluation/world_model_interaction.py \
    --seed 123 \
    --ckpt_path <模型权重路径> \
    --config configs/inference/world_model_interaction.yaml \
    --savedir <输出目录> \
    --bs 1 --height 320 --width 512 \
    --unconditional_guidance_scale 1.0 \
    --ddim_steps 50 --ddim_eta 1.0 \
    --prompt_dir examples/world_model_interaction_prompts \
    --dataset unitree_z1_stackbox \
    --video_length 16 --frame_stride 2 \
    --n_action_steps 16 --exe_steps 16 \
    --n_iter 12 \
    --timestep_spacing uniform_trailing \
    --guidance_rescale 0.7 \
    --perframe_ae
```

### 参数分类

#### 基础配置

| 参数 | 值 | 含义 |
|------|-----|------|
| `--seed` | 123 | 随机种子，保证可复现 |
| `--ckpt_path` | `.ckpt` | 模型权重文件（~5GB） |
| `--config` | `.yaml` | 模型结构配置（UNet、VAE、CLIP 等） |
| `--savedir` | 路径 | 输出保存目录 |
| `--prompt_dir` | 路径 | 输入数据目录（图像+CSV+H5） |
| `--dataset` | 名称 | 选择哪个数据集的样本 |

#### 生成分辨率

| 参数 | 值 | 含义 |
|------|-----|------|
| `--bs` | 1 | batch size（必须为 1） |
| `--height` | 320 | 生成视频帧高度（像素） |
| `--width` | 512 | 生成视频帧宽度（像素） |

#### 扩散采样参数

| 参数 | 值 | 含义 |
|------|-----|------|
| `--ddim_steps` | 50 | DDIM 去噪步数（越多越精细，越慢） |
| `--ddim_eta` | 1.0 | 随机性系数（0=确定性，1=最大随机） |
| `--unconditional_guidance_scale` | 1.0 | CFG 引导强度（1.0=不使用分类器引导） |
| `--timestep_spacing` | uniform_trailing | 时间步采样策略（尾部均匀分布） |
| `--guidance_rescale` | 0.7 | 引导重缩放因子（防止过饱和） |

#### 视频与动作参数

| 参数 | 值 | 含义 |
|------|-----|------|
| `--video_length` | 16 | 每次生成的视频帧数 |
| `--frame_stride` | 2 | 帧间隔（越大→动作幅度越大） |
| `--n_action_steps` | 16 | 每次生成的动作步数 |
| `--exe_steps` | 16 | 实际执行（反馈回模型）的步数 |
| `--n_iter` | 12 | 自回归迭代次数（总共生成 n_iter × exe_steps 帧） |

#### 显存优化

| 参数 | 值 | 含义 |
|------|-----|------|
| `--perframe_ae` | flag | 逐帧解码 VAE（降低显存峰值，推荐开启） |

---

## 4. 推理流程（Pipeline）

```
┌─────────────────────────────────────────────────────────────┐
│                    自回归交互循环 (n_iter 次)                  │
│                                                             │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │ 初始图像  │───▶│  World Model │───▶│ 预测未来16帧视频  │  │
│  │ + 指令   │    │  (扩散生成)   │    │ + 动作序列       │  │
│  └──────────┘    └──────────────┘    └────────┬─────────┘  │
│       ▲                                       │            │
│       │         ┌──────────────┐              │            │
│       └─────────│ 取最后帧作为  │◀─────────────┘            │
│                 │ 下一轮输入    │                           │
│                 └──────────────┘                           │
└─────────────────────────────────────────────────────────────┘
```

每次迭代的详细步骤：

1. **编码**：将当前观测图像通过 VAE 编码到 latent 空间
2. **条件注入**：CLIP 编码文本指令 + 图像条件
3. **DDIM 采样**：在 latent 空间做 50 步去噪，生成 16 帧视频 latent
4. **解码**：VAE 解码 latent → RGB 视频帧
5. **动作提取**：Diffusion Policy Head 从中间特征提取 16 步动作
6. **状态预测**：同时预测机器人状态（关节角等）
7. **反馈**：取生成视频的最后帧作为下一轮的观测输入

---

## 5. 输出结构详解

```
outputs/interaction/
├── inference/
│   ├── 0_full_fs2.mp4                    # ① 完整 rollout 视频
│   └── sample_0/
│       ├── wm/2/                         # ② World Model 输出
│       │   ├── itr-0.mp4                 #    第 0 次迭代
│       │   ├── itr-1.mp4                 #    第 1 次迭代
│       │   ├── ...
│       │   └── itr-11.mp4               #    第 11 次迭代
│       └── dm/2/                         # ③ Decision Making 输出
│           ├── itr-0.mp4
│           ├── ...
│           └── itr-11.mp4
└── tensorboard/
    └── events.out.tfevents.*             # ④ TensorBoard 日志
```

### 各输出含义

| 输出 | 路径模式 | 含义 |
|------|----------|------|
| **完整视频** | `{videoid}_full_fs{stride}.mp4` | 所有迭代的 wm 视频首尾拼接，展示完整任务执行过程 |
| **wm 视频** | `sample_{id}/wm/{stride}/itr-{n}.mp4` | 第 n 次迭代中，世界模型预测的环境变化（"想象的未来"） |
| **dm 视频** | `sample_{id}/dm/{stride}/itr-{n}.mp4` | 第 n 次迭代中，决策模块生成的动作对应的视频（"决策依据"） |
| **TensorBoard** | `tensorboard/events.*` | 可用 `tensorboard --logdir` 查看视频和指标 |

### 文件命名规则

- `fs2` / `fs4` / `fs6`：frame_stride 值，表示帧间隔
- `itr-{n}`：第 n 次自回归迭代（0-indexed）
- `sample_{id}`：对应 CSV 中的 videoid

### 视频内容

每个 `itr-{n}.mp4` 包含 **16 帧**（由 `--video_length 16` 决定），
以 8fps 保存（`--save_fps` 默认值），时长约 2 秒。

完整视频 `0_full_fs2.mp4` = 12 次迭代 × 16 帧 = **192 帧**，约 24 秒。

---

## 6. 核心概念

### 6.1 Latent Diffusion（潜空间扩散）

不直接在像素空间做扩散（太贵），而是：
- 用 **VAE Encoder** 将图像压缩到 1/8 大小的 latent（320×512 → 40×64×4）
- 在 latent 空间做扩散去噪
- 用 **VAE Decoder** 将 latent 还原为像素

### 6.2 DDIM 采样

Denoising Diffusion Implicit Models — 一种加速采样方法：
- 训练时用 1000 步噪声调度
- 推理时只取 50 个时间步（`--ddim_steps 50`），跳过中间步骤
- `eta=1.0` 表示完全随机（等价于 DDPM），`eta=0` 为确定性

### 6.3 Classifier-Free Guidance (CFG)

`--unconditional_guidance_scale 1.0` 表示不使用 CFG：
- CFG > 1 时：`output = uncond + scale × (cond - uncond)`
- 本模型使用 `guidance_rescale=0.7` 做引导重缩放，防止过饱和

### 6.4 World Model vs Decision Making

| | World Model (wm) | Decision Making (dm) |
|---|---|---|
| 作用 | 预测环境未来状态 | 生成机器人动作 |
| 输入 | 当前观测 + 指令 | 当前观测 + 指令 |
| 输出 | 未来视频帧 | 动作序列 + 状态预测 |
| 关系 | 提供"想象"供决策参考 | 基于想象做出行动 |

### 6.5 自回归交互

模型不是一次性生成整段视频，而是：
1. 生成 16 帧 → 取最后帧
2. 将最后帧作为新的观测 → 再生成 16 帧
3. 重复 n_iter 次

这样每一步都基于上一步的预测结果，形成**闭环交互**。

---

## 7. 数据集对照表

| 数据集 | 机器人 | 任务 | n_iter | frame_stride |
|--------|--------|------|--------|-------------|
| unitree_z1_stackbox | Z1 单臂 | 积木堆叠（红黄绿） | 12 | 2 |
| unitree_z1_dual_arm_stackbox | Z1 双臂 | 积木堆叠 | 7 | 4 |
| unitree_z1_dual_arm_stackbox_v2 | Z1 双臂 | 积木堆叠 v2 | 11 | 4 |
| unitree_z1_dual_arm_cleanup_pencils | Z1 双臂 | 收拾铅笔 | 8 | 4 |
| unitree_g1_pack_camera | G1 人形 | 包装相机 | 11 | 6 |

---

## 8. 性能参考

| 指标 | 数值 |
|------|------|
| GPU | RTX 5090 (32GB VRAM) |
| 每次迭代耗时 | ~33 秒 |
| 总耗时（12 步） | ~6 分 40 秒 |
| 显存占用 | ~20GB（perframe_ae 开启） |
| 输出大小 | ~60MB（35 个 mp4 文件） |

---

## 9. 查看结果

```bash
# 直接播放完整视频
mpv /home/creator/ws/20_unitreerobotics/outputs/interaction/inference/0_full_fs2.mp4

# 或用 TensorBoard 查看所有迭代
tensorboard --logdir /home/creator/ws/20_unitreerobotics/outputs/interaction/tensorboard --port 6006
```

---

## 10. UnifoLM-WMA-0 与 UnifoLM-OmniA-0.3 的关系

### 时间线

| 模型 | 发布时间 | 开源状态 |
|------|----------|----------|
| UnifoLM-WMA-0-Dual | 2025 年 9 月 | ✅ 已开源（GitHub + ModelScope） |
| UnifoLM-OmniA-0.3 | 2026 年 7 月 20 日 | ❌ 未开源（无代码、无权重） |

### 定位对比

| | UnifoLM-WMA-0 | UnifoLM-OmniA-0.3 |
|---|---|---|
| **全称** | World Model + Action | Omni-Agent（全能智能体） |
| **定位** | 开源研究架构 | 产品级应用模型 |
| **核心能力** | 世界模型视频预测 + 动作生成 | 全模态交互理解 + 自主任务执行 |
| **任务范围** | 单一操作任务（积木堆叠、收拾铅笔等） | 多任务统筹（家居、康养、设备控制） |
| **交互方式** | 图像 + 文本指令 | 视觉识别、语言对话、精细操作、设备控制 |
| **运行平台** | Z1 机械臂 / G1 人形 | G1 人形机器人 |
| **架构** | Latent Diffusion + Diffusion Policy Head | 未公开（推测为 WMA 的进化版本） |

### 技术演进关系

```
UnifoLM-WMA-0 (2025.09, 开源)
│
│  • 世界模型：给定观测+指令，预测未来视频
│  • 动作头：同时输出机器人动作序列
│  • 自回归交互：多步迭代生成
│
▼
UnifoLM-OmniA-0.3 (2026.07, 未开源)
│
│  • 全模态理解：视觉 + 语言 + 触觉(?)
│  • 多任务统筹：单模型处理家居/康养多类任务
│  • 全程自主：无需人工干预，抗干扰稳定执行
│  • 闭环控制：感知 → 理解 → 决策 → 执行
│
▼
未来开源版本（待宇树发布）
```

### OmniA-0.3 展示的任务示例

- 物品搬运：将地上的抱枕放到沙发上
- 视觉识别与回答：识别抱枕颜色、药盒颜色和数量
- 精细操作：将第三层蓝色药盒拿到上面来、把衣服放进脏衣篓、将盘子放入洗碗机
- 设备控制：调节病床高度，用户说“停”时立即停止

### 当前状态

> **需要区分 UnifoLM 体系下的不同组件：**
>
> **已开源：UnifoLM-WMA（World Model Action）**
> - GitHub: [unitreerobotics/unifolm-world-model-action](https://github.com/unitreerobotics/unifolm-world-model-action)
> - README 明确写了 "released the training and inference code along with the model weights"
> - 提供 UnifoLM-WMA-0_Base 和 UnifoLM-WMA-0_Dual 两个检查点（HuggingFace / ModelScope 下载）
> - 包含训练、推理和真机部署的完整代码
>
> **未开源：UnifoLM-OmniA-0.3 整体模型**
> - 截至 2026 年 7 月，未看到模型权重开源或 HuggingFace 下载链接
> - 更多是以产品发布的形式对外公布，强调在宇树自家机器人上的落地能力
> - GitHub / ModelScope / HuggingFace 均无相关仓库
>
> 简单说：**WMA 组件已开源可下载，OmniA-0.3 整体模型目前还没有开源。**
>
> 关注后续动态可留意：
> - 宇树 GitHub 组织：https://github.com/unitreerobotics
> - 宇树官网：https://unifolm.com

### 学习建议

当前可学习的开源资源仍然是 **UnifoLM-WMA-0**：
- 它代表了宇树世界模型 + 动作生成的核心技术路线
- OmniA-0.3 很可能在其基础上增加了多模态理解、任务规划、实时控制等能力
- 理解 WMA 的架构（Latent Diffusion + Diffusion Policy + 自回归交互）是理解后续演进的基础
