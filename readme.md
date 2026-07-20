# ws 工作区项目梳理

本目录下每个主要子目录基本都是一个独立 git 仓库，当前机器环境为 **Ubuntu 22.04 + NVIDIA RTX 5090**。本文件用于记录各仓库的大致用途、当前验证情况，以及哪些项目因为显存/算力占用较高暂未做完整验证。

## 状态说明

- **✅ 已验证/可运行**：已有本地输出、示例结果，或轻量流程可以直接跑通。
- **🟡 基础可用/部分验证**：核心代码或轻量功能可用，但完整链路依赖模型、数据、硬件或外设。
- **⚠️ 暂未完整验证**：项目本身较重，主要受显存、模型体积、多 GPU 或运行时长限制影响。
- **📦 资料/依赖文件**：不是独立运行项目，主要作为安装包、压缩包或资料留存。

## 仓库概览

以下目录按名称的字母顺序排列，每个项目单独说明用途、当前状态和备注。

### `colibri/`

- **内容概述**：纯 C 的 GLM-5.2 744B MoE 运行时，通过磁盘/RAM/VRAM 分层加载专家。
- **Git 链接**：`git@github.com:xiaoxiaomengmeng/colibri.git`
- **当前分支**：`main`
- **当前状态**：⚠️ 暂未验证。
- **备注**：代码主体为纯 C 实现；但完整运行依赖 GLM-5.2 MoE 模型/专家文件，磁盘占用较大，当前环境暂未完成验证。

### `human_dexterous_teleop/`

- **内容概述**：人体姿态估计、手势识别、MuJoCo 仿真、Aero Hand/ROS2 灵巧手遥操作方案。
- **Git 链接**：[https://gitee.com/zgrxmm/hand_gesture_recognition.git](https://gitee.com/zgrxmm/hand_gesture_recognition.git)
- **当前分支**：`master`
- **当前状态**：🟡 基础可用/部分验证。
- **备注**：MediaPipe/MuJoCo 等轻量模块可跑；完整 ROS2、真实硬件或重型动捕方案需要外设和额外环境。

### `lingbot-depth/`

- **内容概述**：Masked Depth Modeling 深度补全/精修模型，用于空间感知、点云和机器人操作。
- **Git 链接**：`git@github.com:Robbyant/lingbot-depth.git`
- **当前分支**：`main`
- **当前状态**：✅ 已验证/可运行。
- **备注**：`example.py` 提供直接推理流程，`result/` 下已有输出结果；相对适合 RTX 5090 单卡验证。

### `lingbot-video/`

- **内容概述**：面向具身智能的大规模 MoE 视频生成模型，包含 Dense 1.3B 和 MoE 30B-A3B。
- **Git 链接**：`git@github.com:xiaoxiaomengmeng/lingbot-video.git`
- **当前分支**：`master`
- **当前状态**：⚠️ 暂未完整验证。
- **备注**：视频生成和 MoE 推理显存占用高，完整验证需要更充足 GPU 资源；可优先尝试 Dense/低分辨率/短视频路径。

### `lingbot-world-v2/`

- **内容概述**：14B causal-fast 交互世界模型，支持长时交互视频生成和 KV cache 推理。
- **Git 链接**：`git@github.com:xiaoxiaomengmeng/lingbot-world-v2.git`
- **当前分支**：`main`
- **当前状态**：⚠️ 暂未完整验证。
- **备注**：模型体积和推理显存较大，完整体验更偏多 GPU/大显存环境；单 RTX 5090 可作为后续低配置尝试对象。

### `mujoco_menagerie/`

- **内容概述**：Google DeepMind 维护的 MuJoCo 机器人模型库，包含机械臂、四足、人形、无人机等 MJCF 模型。
- **Git 链接**：`git@gitee.com:eai_2026/mujoco_menagerie.git`
- **当前分支**：`master`
- **当前状态**：✅ 已验证/可用。
- **备注**：主要是仿真模型资源库，通常不依赖 GPU；适合离线加载、仿真和算法原型。

### `RynnBrain/`

- **内容概述**：具身基础模型，包含空间理解、目标定位、OCR、轨迹定位、规划、导航等 cookbooks 和验证代码。
- **Git 链接**：`git@github.com:xiaoxiaomengmeng/RynnBrain.git`
- **当前分支**：`main`
- **当前状态**：✅ 已验证/可运行。
- **备注**：`outputs/` 下已有多组示例结果图，适合作为具身多模态理解与推理 demo。

### `RynnWorld-4D/`

- **内容概述**：4D embodied world model，用 RGB、Depth、Optical Flow 建模场景演化，并带机器人操控 policy。
- **Git 链接**：`git@github.com:Alibaba-DAMO-Academy/RynnWorld-4D`
- **当前分支**：`main`
- **当前状态**：⚠️ 暂未完整验证。
- **备注**：依赖 Wan2.2 5B backbone、RynnWorld 权重和可选 Depth-Anything；训练/Policy 闭环较吃 GPU。

### `Wan-Dancer/`

- **内容概述**：音乐驱动的长时长舞蹈视频生成，包含 global/local 两阶段视频生成脚本。
- **Git 链接**：`git@github.com:xiaoxiaomengmeng/Wan-Dancer.git`
- **当前分支**：`main`
- **当前状态**：⚠️ 暂无法在单卡 RTX 5090 验证。
- **备注**：该项目需要 8 卡环境才能正常使用；当前单卡 RTX 5090 环境无法完成有效验证，需等具备 8 卡 GPU 资源后再验证。

## 顶层文件/资料

| 文件 | 说明 | 状态 |
| --- | --- | --- |
| `readme.md` | 当前工作区总览说明。 | ✅ 本文件 |
| `setup_py310_cu128_rtx5090_torch2110.sh` | RTX 5090 + PyTorch 2.11.0 cu128 统一部署脚本。 | ✅ 环境脚本 |

## RTX 5090 PyTorch 环境提示

RTX 5090 对应 `sm_120`，部分项目如果使用旧版 PyTorch/CUDA wheel，可能会出现架构不支持或 CUDA 不可用的问题。当前推荐使用统一部署脚本安装 **CUDA 12.8 版本的 PyTorch 2.11.0**，也可按下方命令手动安装。

### 推荐：使用统一部署脚本

顶层已提供统一脚本，便于后续其他仓库复用：

```bash
# 安装到当前已激活的 conda/venv 环境
bash setup_py310_cu128_rtx5090_torch2110.sh
```

脚本默认读取 `~/Downloads/py310-cu128-torch` 下的本地 wheel，安装完成后会自动验证 `torch.cuda.is_available()`、GPU 名称和 `sm_120` 计算能力。

### 备用：手动安装支持 sm_120 的 PyTorch

```bash
cd ~/Downloads/py310-cu128-torch
pip install \
torch-2.11.0+cu128-cp310-cp310-manylinux_2_28_x86_64.whl \
torchvision-0.26.0+cu128-cp310-cp310-manylinux_2_28_x86_64.whl \
torchaudio-2.11.0+cu128-cp310-cp310-manylinux_2_28_x86_64.whl \
triton-3.6.0-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl \
-f https://mirrors.aliyun.com/pytorch-wheels/ \
-i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
```

### 验证 PyTorch 安装

```bash
python -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Name: {torch.cuda.get_device_name(0)}')"
```

期望输出示例：

```text
PyTorch Version: 2.11.0+cu128
CUDA Available: True
GPU Name: NVIDIA GeForce RTX 5090
```

## 后续验证建议

- **优先验证轻量/确定性项目**：`RynnBrain/`、`lingbot-depth/`、`mujoco_menagerie/`。
- **中等优先级**：`human_dexterous_teleop/`，重点先跑基础 demo 或 doctor/setup 检查。
- **资源受限项目**：`colibri/` 代码主体为纯 C，但模型/专家文件磁盘占用较大，当前暂未验证；`Wan-Dancer/` 需要 8 卡环境，单卡 RTX 5090 暂无法验证。
- **显存压力较大项目**：`lingbot-video/`、`lingbot-world-v2/`、`RynnWorld-4D/`，建议后续按低分辨率、短序列、Dense/小模型、单阶段脚本逐步验证。
- **记录方式**：每次验证后可在本表状态中追加“已跑通命令、输出目录、失败原因或显存占用”。
