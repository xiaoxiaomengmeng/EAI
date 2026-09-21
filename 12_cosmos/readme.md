# Cosmos-Predict2.5：基于扩散模型的世界基础模型套件

## 描述

Cosmos-Predict2.5：一系列高性能预训练世界基础模型，专为生成具备物理感知能力的图像、视频和世界状态以支持物理 AI 开发而设计。

Cosmos-Predict2.5 扩散模型是一组基于扩散机制的世界基础模型，可根据文本、图像或视频输入生成动态、高质量的图像和视频。该模型可作为各类世界生成相关应用或研究的基础模块。

* **模型开发者**：NVIDIA
* **许可用途**：本模型可用于商业 / 非商业用途。

## 模型下载

本项目用到两个模型，均已通过 ModelScope 下载并软链接到 `models/` 目录：

```bash
# 扩散世界基础模型（本项目主模型）
modelscope download --model nv-community/Cosmos-Predict2.5-2B

# 机器人边缘策略模型
modelscope download --model nv-community/Cosmos3-Edge-Policy-DROID

# Cosmos3 边缘多模态模型（VLM 理解 / 图像 / 视频生成）
modelscope download --model nv-community/Cosmos3-Edge
```

### 本地模型路径

| 模型 | 软链接路径 | 实际存储路径 |
| --- | --- | --- |
| Cosmos-Predict2.5-2B | `models/Cosmos-Predict2.5-2B` | `~/.cache/modelscope/models/nv-community--Cosmos-Predict2.5-2B/snapshots/master` |
| Cosmos3-Edge-Policy-DROID | `models/Cosmos3-Edge-Policy-DROID` | `~/.cache/modelscope/models/nv-community--Cosmos3-Edge-Policy-DROID/snapshots/master` |
| Cosmos3-Edge | `models/Cosmos3-Edge` | `~/.cache/modelscope/models/nv-community--Cosmos3-Edge/snapshots/master` |

> 模型以软链接方式指向 ModelScope 默认缓存目录，避免重复占用磁盘空间。

## 代码仓库

Cosmos 3 采用 **Server（加载运行模型）+ Client（仿真评测）** 架构，模型权重之外还需克隆以下仓库代码：

| 仓库 | 克隆命令 | 用途 |
| --- | --- | --- |
| NVIDIA/cosmos-framework | `git clone https://github.com/NVIDIA/cosmos-framework.git` | 核心推理框架 + policy server（`action_policy_server_robolab`），加载并运行 Edge / Edge-Policy 模型 |
| NVlabs/RoboLab | `git clone https://github.com/NVlabs/RoboLab.git` | 仿真评测客户端 / benchmark，驱动机器人跑任务（如 `BananaInBowlTask`），含 `policies/cosmos3/run.py` |
| NVIDIA/cosmos | `git clone https://github.com/NVIDIA/cosmos.git` | 主仓库：cookbooks、部署文档、评测脚本说明、蒸馏配方 |

- **仅跑 `Cosmos3-Edge` 单模型推理**（VLM 理解 / 图像 / 视频生成）：`cosmos-framework` 一个仓库即可。
- **闭环评测 `Cosmos3-Edge-Policy-DROID`**：需要 `cosmos-framework`（server）+ `RoboLab`（client），`cosmos` 提供 cookbook 步骤参考。
- 三个仓库官方均推荐 **Docker + `uv sync`** 方式装依赖（cosmos-framework 用 `cu130-torch213` 依赖组），以适配 RTX 5090 的 Blackwell（sm_120）。

## 环境配置（RTX 5090 · conda · Python 3.12）

目标硬件：RTX 5090（sm_120 / Blackwell），NVIDIA 驱动 ≥ 570。PyTorch wheel 自带 CUDA 12.8 运行库，**无需单独安装 CUDA Toolkit**。

### 1. 创建 conda 环境

```bash
conda create -n cosmos python=3.12 pip -y
conda activate cosmos
```

### 2. 安装 PyTorch 2.9.1 + cu128

5090（sm_120）需要 torch ≥ 2.7，旧版 `torch 2.4.1+cu121` 不含 Blackwell kernel，会报 `no kernel image is available`。复用本机 uv 缓存可免下载、硬链接秒装：

```bash
# uv 二进制借用 microduck 环境，会自动装入当前激活的 cosmos 环境（CONDA_PREFIX）
~/ws/100-microduck/.venv/bin/uv pip install "torch==2.9.1"
```

缓存不存在时退回普通 pip（PyPI 默认 wheel 即捆绑 cu128 运行库）：

```bash
pip install "torch==2.9.1" -i https://pypi.tuna.tsinghua.edu.cn/simple
```

验证 GPU：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 期望输出：2.9.1+cu128 True NVIDIA GeForce RTX 5090
```

### 3. 安装推理依赖

Cosmos3-Edge-Policy-DROID 是标准 diffusers 格式，`model_index.json` 要求 **diffusers ≥ 0.40.0**（提供 `Cosmos3OmniPipeline`）；Cosmos-Predict2.5-2B 为 NVIDIA 自定义 `.pt` 格式，仅需 torch。

> 版本约束：diffusers 0.40.0 要求 `huggingface-hub>=1.23,<2.0`，而 transformers 4.x 全系要求 `huggingface-hub<1.0`，二者互斥。因此 **diffusers 0.40 必须搭配 transformers 5.x**（同代发布，hub 均为 1.x）。

```bash
~/ws/100-microduck/.venv/bin/uv pip install \
    numpy "diffusers==0.40.0" "transformers>=5.0,<6" \
    accelerate safetensors sentencepiece protobuf pillow \
    -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 4. 验证模型可用性

```bash
cd ~/ws/12_cosmos
python scripts/verify_models.py            # 完整验证（含模型加载）
python scripts/verify_models.py --no-load  # 仅文件级检查
```

## 模型版本

Cosmos-Predict2.5 基于扩散机制的模型系列均以 `Cosmos-Predict2.5-2B` 为基础，包含以下 7 个变体：

| 变体                 | 输入条件                        | 预测 / 生成内容                                                       | 视频规格       |
| ------------------ | --------------------------- | --------------------------------------------------------------- | ---------- |
| 预训练版               | 文本描述、一张作为首帧的图像和 / 或一段视频     | 后续帧                                                             | 720P、16FPS |
| 后训练版               | 文本描述、一张作为首帧的图像和 / 或一段视频     | 后续帧                                                             | 720P、16FPS |
| 自动驾驶 / 多视角版        | 文本描述、一张作为首帧的图像和 / 或一段视频     | 包含 7 个摄像头视角的世界场景                                                | 720P、16FPS |
| 机器人 / 多视角版         | 文本描述、一段静态视频以及两条目标相机轨迹       | 两段重新渲染的视频                                                       | 720P、16FPS |
| 机器人 / 多视角-Agibot 版 | 文本描述、一段头部视角视频以及两条目标手部视角相机轨迹 | 两段头部视角视频                                                        | 720P、16FPS |
| 机器人 / 动作条件版        | 一张作为首帧的图像和一段机器人动作序列作为条件     | 后续帧                                                             | 256p、4FPS  |
| 机器人 / 策略版          | 当前图像、机器人本体感知数据和任务描述         | 机器人动作、未来图像、未来机器人本体感知数据以及价值（可选）；分别在 Libero 和 robocasa 数据集上进行了后训练 | —          |

## 模型架构

Cosmos-Predict2.5-2B 是一种扩散 Transformer 模型，专为在潜在空间中进行视频去噪而设计。该网络以交错排列的自注意力层、交叉注意力层和前馈层作为基本构建模块。交叉注意力层使模型能够在整个去噪过程中以输入文本为条件。在每一层之前，应用自适应层归一化以嵌入去噪所需的时间信息。当提供图像或视频作为输入时，其潜在帧会沿时间维度与生成的帧拼接在一起。同时，向条件潜在帧中添加增强噪声，以弥合训练与推理之间的差距。

* **基座模型**：本模型基于 `Cosmos-Predict2-2B` 开发。
* **模型参数数量**：**2,059,174,912**

## 输入/输出规范

### 输入

| 项目   | 规范                                           |
| ---- | -------------------------------------------- |
| 输入类型 | 文本 + 图像、文本 + 视频                              |
| 输入格式 | 文本：字符串；图像：`jpg`、`png`、`jpeg`、`webp`；视频：`mp4` |
| 输入维度 | 文本：一维（1D）；图像：二维（2D）；视频：三维（3D）                |

与输入相关的其他属性：

* 输入字符串应少于 300 个单词，并提供用于世界生成的描述性内容，例如场景描述、关键物体或角色、背景，以及在 5 秒时长内需呈现的特定动作或运动。
* 对于 720P 模型，输入图像尺寸应为 `1280×704`；对于 480P 模型，请使用 `832×480`。
* 输入视频应包含 5 帧，每帧分辨率在 720P 模型下为 `1280×704`，在 480P 模型下为 `832×480`。

### 输出

| 项目   | 规范     |
| ---- | ------ |
| 输出类型 | 视频     |
| 输出格式 | `mp4`  |
| 输出维度 | 三维（3D） |

与输出相关的其他属性：生成的视频为一段 5 秒的片段，其分辨率和帧率由所使用的模型变体决定。例如，720P 16FPS 模型生成的视频分辨率为 `1280×704`，帧率为 16 FPS。

视频内容将输入的文本描述可视化为一个简短的动画场景，在指定的时间限制内捕捉关键元素。

## 硬件要求

我们的 AI 模型专为在 **NVIDIA GPU 加速系统**上运行而设计和 / 或优化。通过利用 NVIDIA 的硬件（如 GPU 核心）和软件框架（如 CUDA 库），该模型相比仅使用 CPU 的解决方案实现了更快的训练和推理速度。
