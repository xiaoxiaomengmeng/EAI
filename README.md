# ws 工作区项目梳理

本目录下每个主要子目录基本都是一个独立 git 仓库，当前机器环境为 **Ubuntu 22.04 + NVIDIA RTX 5090**。本文件用于记录各仓库的大致用途、当前验证情况，以及哪些项目因为显存/算力占用较高暂未做完整验证。

## 状态说明

- **✅ 已验证/可运行**：已有本地输出、示例结果，或轻量流程可以直接跑通。
- **🟡 基础可用/部分验证**：核心代码或轻量功能可用，但完整链路依赖模型、数据、硬件或外设。
- **⚠️ 暂未完整验证**：项目本身较重，主要受显存、模型体积、多 GPU 或运行时长限制影响。

## 目录结构

```
ws/
├── 09_human_dexterous_teleop/   # 人体姿态/手势/灵巧手遥操作
├── 10_openVLA/                  # OpenVLA 视觉-语言-动作模型
├── 11_pi_serial_models/         # Pi 系列（Pi0 / Pi0.5）与 ACT 训练评估（lerobot）
├── 20_unitreerobotics/          # 宇树 UnifoLM-WMA 世界模型
├── 21_demo/                     # 演示项目（RynnBrain、RynnWorld-4D）
├── 22_lingbot/                  # Lingbot 系列（depth、video、world-v2）
├── colibri/                     # 纯 C GLM MoE 运行时
├── mujoco_menagerie/            # MuJoCo 机器人模型库
├── Wan-Dancer/                  # 音乐驱动舞蹈视频生成
├── README.md                    # 本文件
└── scripts/                     # 统一部署脚本
    ├── setup_py310_cu128_rtx5090_torch2110.sh  # py3.10 版（旧项目）
    ├── setup_py312_cu128_rtx5090_torch.sh      # py3.12 版（lerobot v0.6.2+ 等新项目）
    └── setup_py312_cu130_rtx5090_torch.sh      # py3.12 版，清华源 cu130（下载快）
```

## 仓库概览

以下目录按编号/名称顺序排列，每个项目单独说明用途、当前状态和备注。

### `09_human_dexterous_teleop/`

- **内容概述**：人体姿态估计、手势识别、MuJoCo 仿真、Aero Hand/ROS2 灵巧手遥操作方案。
- **Git 链接**：[https://gitee.com/zgrxmm/hand_gesture_recognition.git](https://gitee.com/zgrxmm/hand_gesture_recognition.git)
- **当前分支**：`master`
- **当前状态**：🟡 基础可用/部分验证。
- **备注**：MediaPipe/MuJoCo 等轻量模块可跑；完整 ROS2、真实硬件或重型动捕方案需要外设和额外环境。

### `10_openVLA/`

- **内容概述**：OpenVLA 7B 视觉-语言-动作模型，支持 MuJoCo 仿真推理（WX250S 机械臂）和 LIBERO 评测。
- **Git 链接**：`git@github.com:xiaoxiaomengmeng/openvla.git`（fork）；上游 `https://github.com/openvla/openvla`
- **当前分支**：`main`
- **当前状态**：✅ 已验证/可运行。
- **备注**：已完成单步推理验证（sdpa/flash_attention_2）、WX250S MuJoCo 仿真闭环；详见 `10_openVLA/README.md`。

### `11_pi_serial_models/`

- **内容概述**：Pi 系列（Pi0 / Pi0.5）与 ACT 策略模型的本地训练与仿真评估，基于 lerobot-main v0.6.2（PyTorch）；含 ALOHA 双臂（MuJoCo）与 LIBERO 两条评测链路。
- **Git 链接**：上游 `https://github.com/Physical-Intelligence/openpi`（pi 系列）、`https://github.com/huggingface/lerobot`（训练框架）
- **当前分支**：—
- **当前状态**：✅ 已验证/可运行（ACT 全链路；Pi 系列部分验证）。
- **备注**：ACT 从零训练（100k 步 ≈ 46 分钟，500 轮评估成功率 79%，官方基线 83%）；pi0 LoRA 微调与 pi0.5 LIBERO 后训练链路可用；脚本已按阶段编号（预检/下载/训练/评估/可视化），详见 `11_pi_serial_models/README.md` 与 `docs/17~25`。

### `20_unitreerobotics/`

- **内容概述**：宇树 UnifoLM-WMA-0 世界模型-动作架构，支持视频预测 + 动作生成的自回归交互推理。
- **Git 链接**：`git@github.com:xiaoxiaomengmeng/unifolm-world-model-action.git`（fork）；上游 `https://github.com/unitreerobotics/unifolm-world-model-action`
- **当前分支**：`main`
- **当前状态**：✅ 已验证/可运行。
- **备注**：RTX 5090 兼容性修复已完成（xformers sm_120 适配）；12 步交互推理约 6 分 40 秒；详见 `20_unitreerobotics/readme.md`。

### `21_demo/`

- **内容概述**：演示项目集合，包含 RynnBrain（具身多模态理解）和 RynnWorld-4D（4D 世界模型）。
- **当前状态**：🟡 部分验证。

| 子目录 | 说明 | 状态 |
|--------|------|------|
| `RynnBrain/` | 具身基础模型：空间理解、目标定位、OCR、规划、导航 | ✅ 已验证 |
| `RynnWorld-4D/` | 4D 世界模型：RGB+Depth+Flow 场景演化 + 操控 policy | ⚠️ 暂未验证 |

### `22_lingbot/`

- **内容概述**：Lingbot 系列模型集合，涵盖深度估计、视频生成、交互世界模型。
- **当前状态**：🟡 部分验证。

| 子目录 | 说明 | 状态 |
|--------|------|------|
| `lingbot-depth/` | Masked Depth Modeling 深度补全/精修 | ✅ 已验证 |
| `lingbot-video/` | 大规模 MoE 视频生成（Dense 1.3B / MoE 30B） | ⚠️ 暂未验证 |
| `lingbot-world-v2/` | 14B causal-fast 交互世界模型 | ⚠️ 暂未验证 |

### `colibri/`

- **内容概述**：纯 C 的 GLM-5.2 744B MoE 运行时，通过磁盘/RAM/VRAM 分层加载专家。
- **Git 链接**：`git@github.com:xiaoxiaomengmeng/colibri.git`
- **当前分支**：`main`
- **当前状态**：⚠️ 暂未验证。
- **备注**：完整运行依赖 GLM-5.2 MoE 模型/专家文件，磁盘占用较大。

### `mujoco_menagerie/`

- **内容概述**：Google DeepMind 维护的 MuJoCo 机器人模型库，包含机械臂、四足、人形、无人机等 MJCF 模型。
- **Git 链接**：`git@gitee.com:eai_2026/mujoco_menagerie.git`
- **当前分支**：`master`
- **当前状态**：✅ 已验证/可用。
- **备注**：主要是仿真模型资源库，不依赖 GPU；适合离线加载、仿真和算法原型。

### `Wan-Dancer/`

- **内容概述**：音乐驱动的长时长舞蹈视频生成，包含 global/local 两阶段视频生成脚本。
- **Git 链接**：`git@github.com:xiaoxiaomengmeng/Wan-Dancer.git`
- **当前分支**：`main`
- **当前状态**：⚠️ 暂无法在单卡 RTX 5090 验证。
- **备注**：需要 8 卡环境才能正常使用；当前单卡无法完成有效验证。

## 顶层文件

| 文件 | 说明 |
| --- | --- |
| `README.md` | 当前工作区总览说明（本文件） |
| `scripts/setup_py310_cu128_rtx5090_torch2110.sh` | RTX 5090 + PyTorch cu128 统一部署脚本（py3.10 版） |
| `scripts/setup_py312_cu128_rtx5090_torch.sh` | RTX 5090 + PyTorch cu128 统一部署脚本（py3.12 版） |
| `scripts/setup_py312_cu130_rtx5090_torch.sh` | RTX 5090 + PyTorch cu130 统一部署脚本（py3.12 版，清华 PyPI 源） |
| `embodied_ai_model_test_catalog.md` | 具身智能模型测试目录 |

## RTX 5090 PyTorch 环境提示

RTX 5090 对应 `sm_120`，部分项目如果使用旧版 PyTorch/CUDA wheel，可能会出现架构不支持或 CUDA 不可用的问题。当前推荐使用统一部署脚本安装 **CUDA 12.8 版本的 PyTorch 2.11.0**。

```bash
# 安装到当前已激活的 conda/venv 环境（按项目 Python 版本选择对应脚本）
bash scripts/setup_py310_cu128_rtx5090_torch2110.sh   # py3.10 环境
bash scripts/setup_py312_cu128_rtx5090_torch.sh       # py3.12 环境（lerobot v0.6.2+ 等）
bash scripts/setup_py312_cu130_rtx5090_torch.sh       # py3.12 环境，清华源 cu130，无需本地 wheel
```

阿里云 `pytorch-wheels` 镜像下载 cu128 wheel 常只有约 100 kB/s；cu130 脚本改从清华 PyPI 源安装默认构建 `torch 2.11.0+cu130`，需驱动 ≥ 580（支持 CUDA 13.0）。

脚本默认读取 `~/Downloads/py310-cu128-torch` 下的本地 wheel，安装完成后会自动验证 `torch.cuda.is_available()`、GPU 名称和 `sm_120` 计算能力。

验证命令：

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
# 期望: PyTorch 2.11.0+cu128, CUDA: True, GPU: NVIDIA GeForce RTX 5090
```
