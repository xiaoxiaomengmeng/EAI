# π₀ 本地化部署与 ALoHa 仿真训练指南（5090 机 · py3.12 · LeRobot main 专版）

> 版本 v1.0 · 2026-08-18 · 目标硬件：5090 台式机（RTX 5090 32GB，Ubuntu 22.04）· 技术路线：Python 3.12（deadsnakes）+ LeRobot main（v0.6.x）+ torch cu128 + ModelScope 国内直连
>
> 本文档从《17-pi0本地化部署py310与ALoHa仿真训练指南-v1.0.md》（4090 机 + py3.10 + lerobot v0.4.4，✅ 已实测）拆分而来，**单独记录 py3.12 + lerobot main 新栈路线**（py310 文档中的"方案三"）。py3.10 路线内容以该文档为准，本文不再重复。

---

## 〇、总览

**目标**：在 5090 台式机（RTX 5090 32GB）上，使用 Python 3.12 + LeRobot main（v0.6.x）+ torch cu128 环境，完成 π₀（pi0）VLA 模型的下载、微调训练与 ALoHa 仿真（MuJoCo）闭环验证——在 `lerobot/aloha_sim_transfer_cube_human` 数据集上微调 `lerobot/pi0_base` 基座模型，然后在 ALoHa 仿真双臂环境中 rollout 推理，统计抓取搬运成功率。

**端到端 Pipeline：**

```
① 系统与驱动准备（驱动 ≥570，nvidia-smi 实测 580.173.02）
      ↓
② Python 环境（Python 3.12 deadsnakes + venv312 + torch cu128 + LeRobot main + modelscope）
      ↓
③ 下载 π₀ 模型（ModelScope: lerobot/pi0_base，约 14GB）
      ↓
④ 下载并验证数据集（aloha_sim_transfer_cube_human，v3.0 格式）
      ↓
⑤ 微调训练（lerobot-train，32GB 显存 LoRA 首选 / Action Expert 备选）
      ↓
⑥ 安装 ALoHa 仿真环境（MuJoCo + gym_aloha）
      ↓
⑦ Rollout 推理评估（25 轮成功率 + 视频验收）
```

**软硬件需求表：**

| 项目 | 要求 | 说明 |
|------|------|------|
| 操作系统 | Ubuntu 22.04 | 已满足（实测 22.04.5 LTS） |
| GPU | RTX 5090（32GB） | Blackwell sm_120，实测 32607MiB |
| NVIDIA 驱动 | ≥ 570 | 实测 580.173.02（`nvidia-smi` 显示 CUDA Version 13.0 为驱动支持上限） |
| CUDA | ≥ 12.8 | PyTorch 自带 cu128 运行时即可，无需单独装 CUDA Toolkit |
| Python | **3.12（deadsnakes PPA）** | ✅ 已装 3.12.13；22.04 系统自带 3.10 不满足 LeRobot ≥v0.5.0 要求 |
| LeRobot | main 分支（v0.6.x） | v0.5.0 起要求 Python ≥3.12（官方 PR #3023） |
| 内存 | ≥ 32GB（建议 64GB） | 加载 14GB F32 权重峰值约 21-29GB，注意 RAM OOM（见排错表） |
| 磁盘 | ≥ 100GB 空闲 | 模型 14GB + 数据集约 2GB + 环境 + checkpoint |

**本机实测环境（2026-08-18）：**

```bash
$ nvidia-smi
NVIDIA-SMI 580.173.02    Driver Version: 580.173.02    CUDA Version: 13.0
   0  NVIDIA GeForce RTX 5090    32607MiB

$ source ~/ws/11_pi_serial_models/.venv312/bin/activate
(.venv312) $ python --version
Python 3.12.13
```

- 桌面环境（Xorg / gnome-shell / 浏览器等）会占用部分显存与内存，**训练时建议关闭图形界面程序**，或无头模式运行。
- `nvidia-smi` 显示的 CUDA 13.0 是驱动支持上限；本机 PyTorch 实测为 **cu130 构建**（torch 2.11.0+cu130，自带 CUDA 13.0 运行时，含 sm_120 kernel），与驱动 580 匹配，不冲突。


**三个硬约束（先读，避免白折腾）：**

1. **5090（sm_120）必须用 cu128 及以上构建的 PyTorch**：cu126 及以下会报 `no kernel image is available for execution on the device`。本机实测 torch 2.11.0+cu130（清华源 PyPI 默认构建）在 5090 上 `torch.cuda.is_available()=True` 且 bf16 matmul 正常（arch list 含 sm_120）；若需固定 cu128 构建，可用阿里云 `pytorch-wheels/cu128` 镜像（见 2.2）。
2. **LeRobot ≥v0.5.0 要求 Python ≥3.12**（官方 PR #3023）：22.04 系统 Python 是 3.10，故本环境用 deadsnakes PPA 装 Python 3.12 + 独立 `venv312`，**不动系统默认 python3**。lerobot main（v0.6.x）的 CLI 参数、数据集接口与 v0.4.4 有差异（如官方已移除 pi 安装说明、升级 transformers v5），本环境内一切以 `--help` 实测为准，验证后回填本文档。
3. **单机显存装不下全参数微调**：π₀ 约 3.5B 参数（PaliGemma 3B VLM + 300M Action Expert），全参数 AdamW 微调需约 50-56GB 显存。5090 机 32GB 必须走 **LoRA 微调**（首选）或**仅训练 Action Expert**（备选）路线（见第五节决策表）。
4. **数据集格式版本**：ModelScope 上该数据集当前已是 LeRobot **v3.0** 格式（py310 v0.4.4 环境已实测），预期与 lerobot main 原生匹配，无需转换；若不匹配时用官方转换脚本转换（见第四节）。

**模型与数据资源（ModelScope）：**

| 资源 | 地址 | 规模 |
|------|------|------|
| π₀ 基座模型 | https://modelscope.cn/models/lerobot/pi0_base | 约 14GB（safetensors，F32） |
| ALoHa 仿真数据集 | https://modelscope.cn/datasets/lerobot/aloha_sim_transfer_cube_human | 50 episodes / 20,000 帧 |

> ⚠️ 许可证提示：`lerobot/pi0_base` 权重遵循 **Gemma 许可证**（商用有条件限制）；本指南仅用于研究与学习场景。

**当前进度（随验证推进回填）：**

| 步骤 | 状态 |
|------|------|
| 驱动 ≥570，RTX 5090 可见 | ✅ 实测（580.173.02） |
| Python 3.12 安装（deadsnakes） | ✅ 实测（3.12.13） |
| venv312 虚拟环境 | ✅ 已创建（~/ws/11_pi_serial_models/.venv312） |
| π₀ 基座模型（pi0_base） | ✅ 已下载（models/pi0_base 软链到 ModelScope 缓存） |
| torch 安装与 GPU 验证 | ✅ 已完成（实测 2.11.0+cu130，RTX 5090 识别正常） |
| lerobot main 克隆 | ✅ 已完成（repos/lerobot-main，实测 v0.6.2） |
| lerobot-main editable 安装 | ✅ 基础安装完成（[pi]+[aloha] extra，lerobot-train --help 已通过），⏳ 待补装 wandb/accelerate/peft（[training,peft]，第二节 2.3） |
| modelscope CLI / 数据集下载 | ⏳ 待执行（第二、四节） |
| ALoHa 仿真环境安装与冒烟测试 | ✅ 已完成（[aloha] extra；冒烟 351 FPS，见第六节） |
| 训练 / rollout 评估 | ⏳ 待执行（第五、七节） |

---

## 一、系统与驱动准备

**1.1 检查系统版本**

```bash
lsb_release -a          # 确认 Ubuntu 22.04
uname -r                # 内核版本，建议 ≥ 5.15（22.04 默认满足）
```

**1.2 安装 NVIDIA 驱动（≥570，✅ 本机已就绪：580.173.02）**

本机实测已安装 `nvidia-driver-580-open`（580.173.02），**无需操作**；若在未装驱动的机器上重装，参考：

```bash
sudo apt update
sudo apt install -y ubuntu-drivers-common
ubuntu-drivers devices                     # 查看可用驱动
sudo apt install -y nvidia-driver-580-open # 本机实测包名；570 及以上系列均可（5090 硬性要求 ≥570）
sudo reboot
```

**1.3 验证 GPU 与 CUDA（✅ 已实测通过）**

```bash
nvidia-smi
```

预期输出要点（本机实测）：`NVIDIA-SMI 580.173.02`、`NVIDIA GeForce RTX 5090`、`CUDA Version 13.0`（驱动支持上限，≥ 12.8 满足要求）、显存 32607MiB。

> 说明：`nvidia-smi` 显示的是驱动支持的最高 CUDA 版本，PyTorch 自带 CUDA 运行时，**无需单独安装 CUDA Toolkit**（除非要编译自定义算子）。

**1.4 磁盘规划**

```bash
df -h ~    # 确认 ≥100GB 空闲
```

目录规划（源码/脚本/文档/数据/输出/虚拟环境统一放在本工作区 `11_pi_serial_models` 下，与本工作区 `repos/`（源码）、`models/`（模型软链）惯例一致；以下 `$PI0_ROOT` 代指 `~/ws/11_pi_serial_models`）：

```
$PI0_ROOT/models/pi0_base/                          ← π₀ 模型权重（软链到 ModelScope 缓存）✅ 已就绪
$PI0_ROOT/repos/lerobot-main/                       ← LeRobot 源码（main 分支）✅ 已克隆（实测 v0.6.2）
$PI0_ROOT/scripts/                                  ← 训练/冒烟测试等脚本（train_pi0.sh 等）
$PI0_ROOT/docs/                                     ← 学习笔记与补充文档
$PI0_ROOT/data/aloha_sim_transfer_cube_human/       ← 数据集
$PI0_ROOT/outputs/                                  ← 训练输出与 checkpoint
$PI0_ROOT/.venv312/                                 ← 本文档虚拟环境（py3.12）✅ 已创建
```

---

## 二、Python 环境搭建

**2.1 Python 3.12 与 venv312 虚拟环境（✅ 已完成）**

Ubuntu 22.04 系统源自带 Python 3.10，不满足 LeRobot ≥v0.5.0 的要求，需通过 deadsnakes PPA 安装 3.12；虚拟环境命名 `.venv312`，建在本工作区 `11_pi_serial_models` 目录下（与 `repos/`、`models/` 等工作区惯例一致），不与 py3.10 环境互相污染：

```bash
# 1. 安装 Python 3.12 ✅ 已完成（实测 3.12.13）
sudo apt update
sudo apt install -y software-properties-common   # add-apt-repository 命令依赖此包
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3.12-dev  # dev 包是后续编译类依赖的前提
python3.12 --version    # 应输出 Python 3.12.x

# 2. 建独立虚拟环境 ✅ 已完成
python3.12 -m venv ~/ws/11_pi_serial_models/.venv312
source ~/ws/11_pi_serial_models/.venv312/bin/activate   # 激活（提示符前出现 (.venv312)）
pip install -U pip                         # 升级 pip（实测升到 26.2.1），避免新格式 wheel 安装失败
python --version                           # 验证环境内为 Python 3.12.x
```

> 每次新开终端需重新执行 `source ~/ws/11_pi_serial_models/.venv312/bin/activate`；退出环境用 `deactivate`。**不要动系统默认的 `python3`**（指向 3.10，系统工具依赖它），本环境只通过 `python3.12` / `.venv312` 内的 `python` 调用。旧环境 `~/xmm_ws/pi0/venv312` 已废弃，确认无用后可删除。

**2.2 安装 PyTorch（✅ 已完成：实测 2.11.0+cu130）**

> ⚠️ 5090（sm_120）硬性要求 **cu128 及以上**构建：cu126 及以下会在训练/推理时报 `no kernel image is available for execution on the device`。本机实际安装路径：2.3 从清华源安装 lerobot 时自动带入 **torch 2.11.0+cu130**（torch 2.11 的 PyPI 默认构建即 cu130，含 sm_120 kernel），实测在 5090 上可用。若需固定 cu128 构建，可改用阿里云 `pytorch-wheels/cu128` 镜像（备选 PyTorch 官方 `whl/cu128` 源）：

```bash
# 确保已在 venv312 内（固定 cu128 构建的备选方案，当前环境未使用）：
pip install torch torchvision \
  -f https://mirrors.aliyun.com/pytorch-wheels/cu128/ \
  -i https://mirrors.aliyun.com/pypi/simple/
# 若镜像缺某个版本，回退官方源：
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

> 说明：阿里云镜像只镜像了 wheel 文件本身，没有复刻官方源的索引结构，因此用 `-f`（find-links）指向 `pytorch-wheels/cu128/` 目录找 torch/torchvision 的 +cu128 构建，同时用 `-i` 指定阿里云 PyPI 索引解析其余依赖（nvidia 库、sympy 等）；`-f` 中链接优先级高于索引，保证装到的一定是 cu128 构建。

验证 GPU 被正确识别（**这一步必须通过，否则后面全部白搭**）：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda); \
print(torch.cuda.is_available(), torch.cuda.device_count()); \
print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
```

✅ 本机实测输出：`2.11.0+cu130 13.0`、`True 1`、`['NVIDIA GeForce RTX 5090']`；另用 bf16 matmul 冒烟测试确认 sm_120 kernel 可正常执行（`torch.cuda.get_arch_list()` 含 `sm_120`）。

> 记录要点：实际装到的为 torch 2.11.0+cu130 / torchvision 0.26.0，均在 lerobot v0.6.2 约束（`torch>=2.7,<2.12.0`、`torchvision>=0.22.0,<0.27.0`）范围内；配套 nvidia 库为 cu13 系列（nvidia-cudnn-cu13 等），勿混装 cu12 系列库。
>
> ❓ 为什么 pyproject.toml 里写了 cu128（`[[tool.uv.index]]` + `[tool.uv.sources]`）却装到了 cu130？因为该配置段是 **uv 专属**，pip 只读 `[project]`/`[build-system]`（PEP 621），完全忽略 `[tool.uv.*]`；而 `dependencies` 里对 torch 只有 `>=2.7,<2.12.0` 的版本约束、无 CUDA 构建约束，pip 经清华源解析到 PyPI 默认的 torch 2.11.0（cu130 构建）。若需固定 cu128，用上面的备选方案 `--force-reinstall` 重装即可（之后再跑 `pip install -e` 不会动 torch）。

**耗时预估（提前知悉，耐心等待）**：torch wheel 约 900MB+，加上 opencv（54MB）、av（39MB）、cmake（30MB）、wandb（23MB）等，lerobot 全量依赖下载约 **1.5-2GB**，整体安装通常需 **10-30 分钟**。

**2.3 克隆并安装 LeRobot main（含 π₀ 支持）（克隆 ✅ / 基础安装 ✅ / ⏳ 待补装 [training,peft]）**

LeRobot 从 **v0.5.0** 起要求 **Python ≥3.12**（PR #3023 "require Python 3.12+ as minimum version"，见 https://github.com/huggingface/lerobot/releases/tag/v0.5.0），本环境（Python 3.12.13）满足要求，直接安装 main 分支（v0.6.x）。采用浅克隆源码 + editable 安装（比 `pip install git+...` 快一个数量级，且便于查看/修改源码排错）：

```bash
# 确保已在 venv312 内
cd ~/ws/11_pi_serial_models/repos
# 1. 克隆源码 ✅ 已完成（实测 main 分支 v0.6.2，commit 713a409）
git clone --depth 1 https://github.com/huggingface/lerobot.git lerobot-main
# 若 GitHub 克隆仍慢/失败，换镜像加速前缀（二选一）：
# git clone --depth 1 https://gh-proxy.com/https://github.com/huggingface/lerobot.git lerobot-main
# git clone --depth 1 https://gitclone.com/github.com/huggingface/lerobot.git lerobot-main

pip install ffmpeg-python                 # ✅ 已装（0.2.0）
sudo apt install -y ffmpeg                # torchcodec 视频解码依赖系统 ffmpeg

# 2. editable 安装 + PyPI 国内镜像 ✅ 已完成（下载约 1.5-2GB，耗时 10-30 分钟，见 2.2）：
pip install -e "./lerobot-main[pi]" -i https://pypi.tuna.tsinghua.edu.cn/simple
# 注：在 repos/ 目录下执行，源码目录为 ./lerobot-main

# 3. 补装训练与 LoRA 依赖（⏳ 待执行）：v0.6.2 采用细粒度 extra 设计，[pi] 只装策略依赖
#    （transformers/scipy），训练还需 [training]（datasets/wandb/accelerate/av/torchcodec 等），
#    LoRA 需 [peft]。实测只装 [pi] 时 lerobot-train 报 ImportError: 'datasets' is required：
pip install -e "./lerobot-main[training,peft]" -i https://pypi.tuna.tsinghua.edu.cn/simple
```

说明：

- `-e`（editable）安装后源码即 `~/ws/11_pi_serial_models/repos/lerobot-main/`，后续核对训练参数名、数据集接口直接看本地目录；
- ✅ 已核对 v0.6.2 的 `pyproject.toml`：`[pi]`、`[training]`、`[peft]`、`[aloha]`、`[libero]` 等 extra 均存在；v0.6.2 实测 `[pi] = [transformers-dep, scipy-dep]`、`[training] = [dataset, wandb, accelerate-dep]`、`[peft] = [transformers-dep, peft-dep]`，三者需组合安装才能训练；若将来更新的 main 版本移除了某个 extra，改为 `pip install -e "./lerobot-main"` 并确认 `python -c "import lerobot.policies.pi0"` 可导入；
- 兜底方案：clone 彻底不通时，下载 main 源码包解压后同样 `pip install -e`：
  `wget https://gh-proxy.com/https://github.com/huggingface/lerobot/archive/refs/heads/main.zip`

验证（注意：`| head` 截断长 help 会在末尾抛 `BrokenPipeError`，属正常现象，help 已正常输出即算通过）：

```bash
pip check                                    # ✅ 已通过（No broken requirements found）
python -c "from lerobot.policies.pi0.modeling_pi0 import PI0Policy; print('OK')"   # ✅ 已通过
lerobot-train --help 2>/dev/null | head -20  # ✅ 已通过（[aloha] 连带 [dataset] 后 datasets 就位；注意训练前仍需补装 wandb/accelerate/peft）
```

**2.4 安装 ModelScope CLI（⏳ 待执行）**

```bash
pip install modelscope
modelscope --version
```

---

## 三、下载 π₀ 模型（ModelScope）（✅ 已就绪）

```bash
mkdir -p ~/ws/11_pi_serial_models/models
modelscope download --model lerobot/pi0_base \
  --local_dir ~/ws/11_pi_serial_models/models/pi0_base
```

- ✅ 本机已下载：工作区已有软链 `models/pi0_base -> ~/.cache/modelscope/models/lerobot--pi0_base/snapshots/master`（含 `model.safetensors`、`config.json` 等），**直接复用，无需重复下载**（另有 `pi0_libero_base` 软链，供 LIBERO 评测场景使用）。
- 体积约 **14GB**，若需重新下载：中断后重新执行同一命令可断点续传。
- 检查目录结构：

```bash
ls -lh ~/ws/11_pi_serial_models/models/pi0_base
```

预期包含：`model.safetensors`（约 14GB，F32 精度保存）、`config.json`（策略配置）等文件。

> 说明：权重以 F32 保存，训练/推理时通过 `--policy.dtype=bfloat16` 转为 bf16 使用，实际占用约 7GB 显存。

---

## 四、下载并验证数据集

**4.1 下载数据集**

```bash
mkdir -p ~/ws/11_pi_serial_models/data
modelscope download --dataset lerobot/aloha_sim_transfer_cube_human \
  --local_dir ~/ws/11_pi_serial_models/data/aloha_sim_transfer_cube_human
```

**4.2 数据规格卡**

| 属性 | 值 |
|------|-----|
| LeRobot 格式版本 | **v3.0**（py310 v0.4.4 环境已实测；lerobot main 同为 v3.0 格式，预期**无需格式转换**，装完后以 4.3 实测为准） |
| Episode 数 | 50 |
| 总帧数 | 20,000（每集约 400 帧） |
| 采样率 | 50 fps |
| 图像观测 | `observation.images.top`，480×640×3，单目俯视相机（视频编码 av1） |
| 动作维度 | 14（ALoHa 双臂：每臂 6 关节 + 1 夹爪 × 2） |
| 状态维度 | 14（`observation.state`） |
| 任务 | 右臂抓取方块并转移到左臂（人类遥操作演示） |

**4.3 本地加载验证**

```python
# test_dataset.py（在 venv312 内运行）
from lerobot.datasets.lerobot_dataset import LeRobotDataset   # ⚠️ main 分支的导入路径以实际源码为准（v0.4.4 路径无 .common，main 可能不同）

dataset = LeRobotDataset(
    repo_id="aloha_sim_transfer_cube_human",
    root="/home/creator/ws/11_pi_serial_models/data/aloha_sim_transfer_cube_human",  # ⚠️ 必须指向数据集目录本身（含 meta/info.json），不是父目录
)
print("episodes:", dataset.num_episodes)
print("frames:", len(dataset))
sample = dataset[0]
print("keys:", list(sample.keys()))
print("action shape:", sample["action"].shape)
print("image shape:", sample["observation.images.top"].shape)
```

```bash
python test_dataset.py
```

预期输出（参照 py310 v0.4.4 环境实测，main 分支待回填）：`episodes: 50`、`frames: 20000`、`action shape: torch.Size([14])`、`image shape: torch.Size([3, 480, 640])`（CHW、float32，已归一化）。

**4.4 格式版本兼容说明**

ModelScope 上该数据集当前版本已是 **v3.0**，预期与 lerobot main 原生匹配，**无需任何转换**。若拿到 v2.0 格式的旧数据集（加载时报 `codebase_version` 不匹配），再用 LeRobot 仓库自带的转换脚本逐级升级（v2.0 → v2.1 → v3.0，脚本位于仓库 `scripts/` 目录，如 `convert_dataset_v20_to_v21`），转换后重新运行 4.3 验证。

> 加载路径注意事项（py310 环境实测踩坑，main 分支预期同理）：`LeRobotDataset` 的 `root` 若指向父目录（`.../data`），会报 `FileNotFoundError: .../data/meta/info.json` 并尝试联网回退；必须指到数据集目录本身。同理，第五节训练命令中 `--dataset.root` 也要写数据集目录全路径。

---

## 五、微调训练

**5.1 显存预算决策表（RTX 5090 = 32GB，必读）**

π₀ 共约 3.5B 参数。不同训练模式的显存估算：

| 训练模式 | 可训练参数 | 显存估算 | 5090（32GB） |
|----------|-----------|----------|--------------|
| 全参数微调（AdamW fp32 状态） | 3.5B | ≈ 50-56GB | ❌ 不可行 |
| 全参数 + 8-bit 优化器 + 梯度检查点 + batch 1-2 | 3.5B | ≈ 25-30GB | ⚠️ 勉强，易 OOM |
| **仅训练 Action Expert**（冻结 VLM） | 300M | ≈ 15-20GB | ✅ 备选 |
| **LoRA 微调**（peft） | ~1-5% | ≈ 12-18GB | ✅ 推荐首选 |

**推荐策略**：首选 LoRA；若 lerobot main 的训练入口未内置 LoRA 开关（以 `lerobot-train --help` 实测为准），可退而选"冻结 VLM、只训练 Action Expert"。两者在 50 条演示的小数据集上效果差异通常不大。

> 训练前先关闭占用显存的桌面程序（桌面环境约占用 2GB 显存）。

**5.2 训练命令（lerobot-train）**

以下命令为 py310 v0.4.4 环境的实测模板（ModelScope 官方微调模板 + 针对本数据集的必要适配），**lerobot main 的参数名可能有差异（如 v0.5.0 起官方接口有调整、升级 transformers v5），执行前先对照 `lerobot-train --help` 逐项核对并回填本节**：

```bash
lerobot-train \
  --dataset.repo_id=aloha_sim_transfer_cube_human \
  --dataset.root=~/ws/11_pi_serial_models/data/aloha_sim_transfer_cube_human \
  --output_dir=~/ws/11_pi_serial_models/outputs/pi0_aloha_cube \
  --job_name=pi0_aloha_cube \
  --policy.type=pi0 \
  --policy.pretrained_path=~/ws/11_pi_serial_models/models/pi0_base \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --rename_map='{"observation.images.top":"observation.images.base_0_rgb"}' \
  --peft.method_type=LORA \
  --peft.r=16 \
  --steps=20000 \
  --batch_size=4        # 5090 机 32GB 可用 4，OOM 降 2
```

> 建议封装为启动脚本 [`scripts/train_pi0.sh`](./scripts/train_pi0.sh)：含训练前自检（权重/数据集/CLI/GPU）、日志落盘 `logs/`、支持 `STEPS`/`BATCH_SIZE` 环境变量覆盖（如 `STEPS=200 bash scripts/train_pi0.sh` 先做冒烟测试）。

参数说明（语义以 py310 v0.4.4 实测为基准，main 分支待核对）：

| 参数 | 含义 |
|------|------|
| `--dataset.repo_id` + `--dataset.root` | 本地数据集（**root 必须指向数据集目录本身**，即含 `meta/info.json` 的目录，见 4.3 踩坑说明） |
| `--policy.type=pi0` | 选择 π₀ 策略（v0.4.4 支持 pi0/pi0_fast/pi05/act/smolvla 等 14 种，必填） |
| `--policy.pretrained_path` | 本地 π₀ 基座权重目录（即第三节软链位置；不传则从头训练） |
| `--policy.dtype=bfloat16` | F32 权重加载后转 bf16，显存减半 |
| `--policy.push_to_hub=false` | ⚠️ **必加**：基座 config.json 里 `push_to_hub=true`，不关会在保存 checkpoint 时尝试上传 HF Hub（未登录会报错） |
| `--rename_map='{"observation.images.top":"observation.images.base_0_rgb"}'` | ⚠️ **必加**：基座期望相机键名 `base_0_rgb`，本数据集只有 `top`；v0.4.4 通过 `rename_observations_processor` 做映射（JSON 字符串语法实测可用；基座声明的其余相机键缺失时自动用空图适配）。state/action 14 维也会自动 pad 到基座的 32 维，无需处理；main 分支的映射机制待实测确认 |
| `--peft.method_type=LORA` + `--peft.r=16` | 启用 LoRA（按 5.1 决策表的首选路线）。π₀ 自带官方默认 target_modules（Action Expert 的 q/v 投影 + state/action 投影层），无需手填；不传 `--peft.*` 则走全参数（本机显存不可行） |
| `--steps` / `--batch_size` | 训练步数与批大小 |

> 以上参数名与语法均已按 py310 v0.4.4 源码/`lerobot-train --help`/draccus 解析实测核对（注意是 `pretrained_path`；旧资料常见的 `--policy.path` 在该版本不存在）。**main 分支以本环境 `lerobot-train --help` 实测为准，核对后回填**。`--policy.chunk_size`、`--policy.gradient_checkpointing=true`（显存再紧时开启）等策略参数可按需追加。

**5.3 超参数建议**

| 超参数 | 建议值 | 说明 |
|--------|--------|------|
| steps | 10,000-20,000 | 50 条演示数据，20k 步 × batch4 ≈ 4 个 epoch |
| batch_size | 4（OOM 降 2；LoRA 模式下可上探 4-8） | 32GB 显存余量较足 |
| dtype | bfloat16 | Blackwell 原生支持 bf16 |
| chunk_size | 默认（50） | π₀ 动作块长度，小数据集不必改 |
| 学习率 | 默认（约 1e-5 量级） | 微调不必调大 |

**5.4 训练监控与产物**

- 训练日志与 checkpoint 写入 `~/ws/11_pi_serial_models/outputs/pi0_aloha_cube/checkpoints/`；
- loss 应在前几百步明显下降后进入平台期；
- 预计耗时：20k 步约 **3-8 小时**（取决于数据加载与 chunk 配置，暂未实测）；
- 最终用于推理的是 `checkpoints/last/` 下的策略权重目录。

---

## 六、安装 ALoHa 仿真环境

ALoHa 仿真双臂环境源自 ACT 论文（arXiv 2304.13705）的仿真实现，基于 MuJoCo / dm_control。

**6.1 安装依赖（✅ 已完成）**

```bash
pip install mujoco dm_control
# LeRobot 的 aloha 仿真 extra：已核对 v0.6.2 pyproject.toml 声明为
# aloha = ["lerobot[dataset]", "gym-aloha>=0.1.4,<0.2.0", "lerobot[scipy-dep]"]，
# 会从清华源自动安装独立的 PyPI 包 gym-aloha
cd ~/ws/11_pi_serial_models/repos
pip install -e "./lerobot-main[aloha]" -i https://pypi.tuna.tsinghua.edu.cn/simple
```

✅ 本机实测：gym-aloha 0.1.4、gymnasium 1.3.0、mujoco 3.8.1、datasets 4.8.5、av 15.1.0、torchcodec 0.11.1 均已就位（`[aloha]` 会连带 `[dataset]`）；冒烟测试实测未装 dm_control 也能跑通。

> ✅ 已核对 v0.6.2 中 `lerobot-main[aloha]` extra 存在；若将来更新的 main 版本 extra 命名有变化，查阅 LeRobot 仓库文档中 ALOHA 仿真环境的安装说明。

**6.2 任务说明**

| 属性 | 值 |
|------|-----|
| 环境名 | gym_aloha |
| 任务 ID | `AlohaTransferCube-v0` |
| 内容 | 左臂区域出现方块，双臂协作将其搬运至右臂托盘 |
| 单轮步数 | rollout 上限 400 步：gym-aloha 注册默认 `max_episode_steps=300`，但 lerobot 的 `AlohaEnv` 配置用 `episode_length=400` 覆盖（✅ v0.6.2 源码实测确认：`configs.py` 中 `AlohaEnv.episode_length=400`，并经 `gym_kwargs.max_episode_steps` 生效） |
| 观测 | 裸 `gym.make` 默认只有 `top` 图像（480×640×3，✅ 本机实测）；lerobot 创建环境时传 `obs_type="pixels_agent_pos"`（✅ v0.6.2 源码实测确认），环境同时返回 14 维 `agent_pos`，再经 `features_map` 改名为 `observation.state`/`observation.images.top`（与数据集键名对齐） |

**6.3 渲染方式**

```bash
# 台式机带显示器：默认 glfw 渲染即可
# 无头/SSH 场景：使用 EGL GPU 渲染
export MUJOCO_GL=egl
```

**6.4 冒烟测试（训练前必做）**

先用随机动作验证环境能正常跑通，排除渲染/依赖问题。已封装为 [`scripts/smoke_test_env.py`](./scripts/smoke_test_env.py)（✅ 已创建；含观测/动作空间断言与帧率统计，支持 `--env-id`/`--steps`/`--render` 参数）：

```bash
python scripts/smoke_test_env.py                      # 台式机带显示器：默认 glfw 渲染
MUJOCO_GL=egl python scripts/smoke_test_env.py        # 无头/SSH：EGL GPU 渲染
```

参照 py310 环境实测输出（EGL）：`obs keys: ['top']`（480×640×3 uint8）、`action 空间: Box(-1.0, 1.0, (14,))`、`冒烟测试 OK | 步数: 100 | 帧率: 241 FPS`。

✅ **本机（5090 机）实测已通过**（EGL）：`action 空间: Box(-1.0, 1.0, (14,), float32)`、`obs keys: ['top']`（480×640×3 uint8）、`冒烟测试 OK | 步数: 100 | 帧率: 351 FPS`（实测依赖：gym-aloha 0.1.4、mujoco 3.8.1；gym-aloha 0.1.4 直接依赖 mujoco，未装 dm_control 也能跑通）。

<details>
<summary>脚本核心逻辑（展开查看；完整实现见 scripts/smoke_test_env.py）</summary>

```python
# smoke_test_env.py
import gymnasium as gym
import gym_aloha  # noqa: F401（注册环境）

env = gym.make("gym_aloha/AlohaTransferCube-v0")
obs, _ = env.reset()
print("obs keys:", obs.keys())
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
print("smoke test OK")
env.close()
```

</details>

---

## 七、推理与 Rollout 评估

**7.1 用 lerobot 评估命令（推荐）**

```bash
lerobot-eval \
  --policy.type=pi0 \
  --policy.pretrained_path=~/ws/11_pi_serial_models/outputs/pi0_aloha_cube/checkpoints/last/pretrained_model \
  --env.name=gym_aloha \
  --env.task=AlohaTransferCube-v0 \
  --eval.n_episodes=25
```

> 参数名已按 py310 v0.4.4 的 `lerobot-eval --help` 实测核对；**main 分支以本环境 `lerobot-eval --help` 为准，核对后回填**；可加视频保存参数（如 `--eval.save_video=true`，视版本而定）。

**7.2 独立推理代码（理解原理用）**

ModelScope 模型页给出的推理接口：

```python
from lerobot.policies.pi0.modeling_pi0 import PI0Policy   # ⚠️ main 分支导入路径以实际源码为准
# from lerobot.policies.pi0.processing_pi0 import preprocess, postprocess

policy = PI0Policy.from_pretrained(
    "/home/creator/ws/11_pi_serial_models/outputs/pi0_aloha_cube/checkpoints/last/pretrained_model"
)
policy.eval()

# frame: 包含 observation.images.top 与 observation.state 的字典
# pred_action = policy.select_action(preprocess(frame))
# pred_action = postprocess(pred_action)
```

**7.3 验收标准**

| 指标 | 判定 |
|------|------|
| 成功率 | 25 轮 rollout 中方块被成功搬到右盘并保持稳定的比例；微调充分时预期可达到较高水平 |
| 过程观察 | 视频中左臂定位方块 → 抓取 → 双臂交接 → 右臂放置的动作连贯性 |
| 失败模式 | 抓空、交接掉落、放置偏移——分别指向定位、协调、精调不足 |

> 若成功率明显偏低：优先检查训练步数是否足够、数据格式转换是否完整、图像输入尺寸与预处理是否与训练一致。

---

## 八、排错速查表

| 症状 | 原因 | 解决 |
|------|------|------|
| `no kernel image is available for execution on the device` | 5090（sm_120）上用了 torch ≤2.9 的默认构建（cu126 及以下） | 按本文 2.2 节从官方 `whl/cu128` 源重装 torch（或升级到 torch 2.10.0+，其 PyPI 默认构建即 cu128） |
| 装 lerobot main 时 pip 更换了预装的 torch | main 对 torch 的版本约束与预装版本不同（v0.6.2 实测约束为 `torch>=2.7,<2.12.0`） | 本机实际为 lerobot 安装时自动带入 torch 2.11.0+cu130；若发生更换，确认新构建含 sm_120 kernel（`python -c "import torch; print(torch.__version__, torch.cuda.get_arch_list())"`），否则固定 cu128/cu130 版本重装 |
| `lerobot-train` 在 `Creating policy` 阶段被"已杀死" | **系统内存（RAM）OOM**：加载 14GB F32 权重峰值约 21-29GB，超过 32GB 物理内存减桌面占用后的余量（py310 环境实测，内核日志 `Out of memory: Killed process ... lerobot-train`） | ① 关闭浏览器/编辑器释放 RAM；② 扩容 swap 兜底：`sudo fallocate -l 24G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile`（仅加载阶段变慢几分钟，训练不受影响） |
| `nvidia-smi` 看不到 GPU / CUDA Version 过低 | 驱动过旧 | 升级驱动 ≥570 并重启 |
| 训练启动即 CUDA OOM | 全参数微调超显存 | 改 LoRA 或仅训 Action Expert；batch 降到 1-2；开梯度检查点；关闭桌面程序释放约 2GB |
| 模型加载阶段 OOM | F32 权重直接载入 | 确认 `--policy.dtype=bfloat16` |
| `lerobot-train` 报 `ImportError: 'datasets' is required` | v0.6.2 细粒度 extra 设计，`[pi]` 不含训练依赖（实测踩坑） | 补装 `pip install -e "./lerobot-main[training,peft]"`（见 2.3） |
| 安装 lerobot 报 `./lerobot-main[pi] is not a valid editable requirement` | 未在 `repos/` 目录下执行，`./lerobot-main` 路径不存在（实测踩坑） | 先 `cd ~/ws/11_pi_serial_models/repos` 再执行 `pip install -e`，或改用完整路径 |
| 安装 lerobot 报 `requires a different Python: 3.10.x not in '>=3.12'` | 当前终端用的是系统 Python 3.10（未激活 .venv312）；LeRobot ≥v0.5.0 要求 Python ≥3.12（官方 PR #3023） | 先 `source ~/ws/11_pi_serial_models/.venv312/bin/activate` 并用 `python --version` 确认为 3.12.x 后重装（第二节 2.1） |
| `codebase_version` 不匹配报错 | 数据集 v2.0 旧格式 vs LeRobot 期望 v3.0（当前 ModelScope 数据集已是 v3.0，正常不会遇到） | 用官方转换脚本升级数据集（第四节 4.4） |
| modelscope 下载中断 | 网络波动 | 重跑同一命令，自动续传 |
| MuJoCo 渲染报 `libGL` / GLFW 错误 | 图形库缺失或无头环境 | `sudo apt install libgl1 libglfw3`；无头改用 `MUJOCO_GL=egl` |
| 训练极慢 / dataloader 打满 CPU | 视频帧解码瓶颈 | 确认系统 ffmpeg 已装；适当提高 dataloader worker 数 |
| rollout 动作抖动 | 动作块边界未平滑 | 确认使用 π₀ 默认 chunk 执行逻辑；必要时启用 temporal ensemble |

---

## 九、附录

**9.1 一键流程清单**

```
✅ 驱动 ≥570，nvidia-smi 可见 RTX 5090（32GB），CUDA ≥ 12.8
✅ Python 3.12（deadsnakes，实测 3.12.13）+ venv312（~/ws/11_pi_serial_models/.venv312）
✅ π₀ 基座模型就绪（models/pi0_base 软链，约 14GB）
□ venv312 内 torch 安装与 GPU 验证通过 → ✅ 已完成（实测 2.11.0+cu130；识别到 NVIDIA GeForce RTX 5090，bf16 matmul 冒烟通过）
✅ lerobot main 源码克隆完成（repos/lerobot-main，v0.6.2）
□ lerobot-main[training,peft] + modelscope 安装完成 → 部分完成：[pi]+[aloha] 已装、pip check 通过、lerobot-train --help 通过；⏳ 还差 wandb/accelerate（[training]）与 peft（LoRA 必需）
□ 数据集下载完成并通过 LeRobotDataset 加载验证
□ 数据集格式确认（v3.0，预期与 lerobot main 匹配，无需转换）
□ 冒烟测试：仿真环境随机动作跑通 → ✅ 已完成（EGL，351 FPS，见第六节 6.4）
□ 训练完成，checkpoint 生成
□ 25 轮 rollout 成功率统计 + 视频验收
```

**9.2 目录结构总览**

```
~/ws/11_pi_serial_models/
├── .venv312/                      ← 本文档虚拟环境（py3.12）✅
├── repos/lerobot-main/            ← LeRobot 源码（main 分支，实测 v0.6.2）✅
├── models/pi0_base/               ← 软链到 ModelScope 缓存 ✅
│   ├── model.safetensors          (~14GB, F32)
│   └── config.json
├── data/aloha_sim_transfer_cube_human/
│   ├── meta/                      (info.json, episodes.jsonl, tasks.jsonl)
│   └── data/ | videos/            (chunk 数据)
├── scripts/                       ← train_pi0.sh、smoke_test_env.py 等
├── docs/                          ← 学习笔记与补充文档
└── outputs/pi0_aloha_cube/
    └── checkpoints/
        ├── 010000/
        └── last/pretrained_model/   ← 推理用
```

**9.3 参考资料**

- π₀ 模型（ModelScope）：https://modelscope.cn/models/lerobot/pi0_base
- ALoHa 仿真数据集（ModelScope）：https://modelscope.cn/datasets/lerobot/aloha_sim_transfer_cube_human
- LeRobot 仓库：https://github.com/huggingface/lerobot
- LeRobot v0.5.0 Release Notes（Python ≥3.12 要求的官方依据，PR #3023）：https://github.com/huggingface/lerobot/releases/tag/v0.5.0
- π₀ 论文：arXiv 2410.24164（本工作空间 papers/ 已有 PDF 与阅读报告）
- ACT 论文（ALoHa 仿真环境来源）：arXiv 2304.13705
- π₀.5 / π₀.7 对比见本工作空间 03-π系列专题与各论文阅读报告

---

*文档版本 v1.0 · 2026-08-18：从 py310 文档拆分出专版，专门记录 5090 机 + Python 3.12（deadsnakes，实测 3.12.13）+ lerobot main（v0.6.x）+ torch cu128 新栈路线（⏳ 验证中）；venv312（~/ws/11_pi_serial_models/.venv312）与 pi0_base 权重已就绪 · 配套 py310 版：17-pi0本地化部署py310与ALoHa仿真训练指南-v1.0.md*
