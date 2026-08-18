# π₀ 本地化部署与 ALoHa 仿真训练指南

> 版本 v1.2 · 2026-08-18 · 目标硬件：两台台式机（均 Ubuntu 22.04）——5090 机（RTX 5090 32GB）与 4090 机（RTX 4090 24GB），本指南同一份适用于两台机器（三种部署方案见 2.2 矩阵）· 技术路线：LeRobot（PyTorch）+ ModelScope 国内直连

---

## 〇、总览

**目标**：在本地台式机完成 π₀（pi0）VLA 模型的下载、微调训练与仿真闭环验证——使用 ModelScope 上的 `lerobot/pi0_base` 基座模型，在 `lerobot/aloha_sim_transfer_cube_human` 数据集上微调，然后在 ALoHa 仿真双臂环境（MuJoCo）中 rollout 推理，统计抓取搬运成功率。

**端到端 Pipeline：**

```
① 系统与驱动准备（驱动 ≥570 / CUDA 12.8，两台机器流程相同）
      ↓
② Python 环境（venv + PyTorch，三种部署方案见 2.2 + LeRobot + modelscope）
      ↓
③ 下载 π₀ 模型（ModelScope: lerobot/pi0_base，约 14GB）
      ↓
④ 下载并验证数据集（aloha_sim_transfer_cube_human，✅ 实测 v3.0 格式，无需转换）
      ↓
⑤ 微调训练（lerobot-train，按显存选 LoRA / Action Expert：方案三 32GB 灵活，方案一/二 24GB 首选 LoRA）
      ↓
⑥ 安装 ALoHa 仿真环境（MuJoCo + gym_aloha）
      ↓
⑦ Rollout 推理评估（25 轮成功率 + 视频验收）
```

**软硬件需求表：**

| 项目 | 要求 | 说明 |
|------|------|------|
| 操作系统 | Ubuntu 22.04 | 已满足（实测 22.04.5 LTS） |
| GPU | RTX 5090（32GB）**或** RTX 4090（24GB） | 两台独立台式机；5090 机：Blackwell sm_120，4090 机：Ada sm_89 |
| NVIDIA 驱动 | ≥ 570 | 5090 机硬性要求；4090 机 ≥520 即可（实测 580.173.02），同一安装流程两机通用 |
| CUDA | ≥ 12.8 | 驱动自带运行时即可；实测三种方案均为 cu128 运行时（torch 2.10.0 默认构建即 cu128，见 2.2） |
| Python | 方案一：3.10；方案二/三：3.12（deadsnakes） | 22.04 自带 3.10 满足方案一；LeRobot 需 pin v0.4.4（v0.5.0 起要求 ≥3.12，见 2.3/2.5 节） |
| 内存 | ≥ 32GB | 实测 32GB + 桌面环境加载 π₀ 权重（峰值约 21-29GB）可能被 OOM 杀死，需关浏览器/编辑器并扩 swap 兜底（见排错表）；建议 64GB |
| 磁盘 | ≥ 100GB 空闲 | 模型 14GB + 数据集约 2GB + 环境 + checkpoint |

**本机实测环境（2026-08-18，于 4090 台式机实测；5090 台式机要求同上）：**

```bash
$ lsb_release -a
Description:    Ubuntu 22.04.5 LTS

$ nvidia-smi
NVIDIA-SMI 580.173.02    Driver Version: 580.173.02    CUDA Version: 13.0
   0  NVIDIA GeForce RTX 4090    2541MiB / 24564MiB
```

- 以上为 4090 机实测；5090 机驱动同样要求 ≥570，安装流程完全一致。
- 桌面环境（Xorg / gnome-shell / 浏览器等）已占用约 2GB 显存，**训练时建议关闭图形界面程序**，或无头模式运行。
- `nvidia-smi` 显示的 CUDA 13.0 是驱动支持上限，PyTorch 自带 cu128 运行时，两者不冲突。


**三个硬约束（先读，避免白折腾）：**

1. **分三种部署方案（详见 2.2 矩阵）**：Blackwell（5090，sm_120）必须用 cu128 构建的 PyTorch，cu126 及以下会报 `no kernel image is available for execution on the device`；Ada（4090，sm_89）cu118+ 即可。结合 lerobot 版本约束，本指南规划三种部署方案：
   - **方案一（✅ 实测）**：4090 机 + Python 3.10 + lerobot v0.4.4 + torch 2.10.0+cu128（因 v0.4.4 限制 `torch<2.11.0` 自动降级；2.10 默认构建即 cu128）——本指南正文默认按此方案；
   - **方案二（⏳ 待验证）**：4090 机 + Python 3.12（deadsnakes）+ lerobot main（v0.6.x）+ torch cu128；
   - **方案三（⏳ 待验证）**：5090 机 + Python 3.12（deadsnakes）+ lerobot main（v0.6.x）+ torch cu128。
   方案二/三的安装步骤见 2.5 节。
2. **单机显存装不下全参数微调**：π₀ 约 3.5B 参数（PaliGemma 3B VLM + 300M Action Expert），全参数 AdamW 微调需要约 50-56GB 显存。两台机器都必须走 **LoRA 微调**或**仅训练 Action Expert**路线；4090 机显存更紧，首选 LoRA + batch 1-2（见第五节决策表）。
3. **数据集格式版本**：ModelScope 上该数据集当前已是 LeRobot **v3.0** 格式（✅ 实测），与 lerobot v0.4.4 原生匹配，无需转换；若将来拿到 v2.0 旧格式，加载前需校验版本，不匹配时用官方转换脚本转换（见第四节）。

**模型与数据资源（ModelScope）：**

| 资源 | 地址 | 规模 |
|------|------|------|
| π₀ 基座模型 | https://modelscope.cn/models/lerobot/pi0_base | 约 14GB（safetensors，F32） |
| ALoHa 仿真数据集 | https://modelscope.cn/datasets/lerobot/aloha_sim_transfer_cube_human | 50 episodes / 20,000 帧 |

> ⚠️ 许可证提示：`lerobot/pi0_base` 权重遵循 **Gemma 许可证**（商用有条件限制）；本指南仅用于研究与学习场景。

---

## 一、系统与驱动准备

**1.1 检查系统版本**

```bash
lsb_release -a          # 确认 Ubuntu 22.04
uname -r                # 内核版本，建议 ≥ 5.15（22.04 默认满足）
```

**1.2 安装 NVIDIA 驱动（≥570）**

```bash
sudo apt update
sudo apt install -y ubuntu-drivers-common
ubuntu-drivers devices                 # 查看可用驱动
sudo apt install -y nvidia-driver-570  # 或更高的 575/580 系列
sudo reboot
```

**1.3 验证 GPU 与 CUDA**

```bash
nvidia-smi
```

预期输出要点：

- **5090 环境**：`NVIDIA GeForce RTX 5090`、`CUDA Version ≥ 12.8`、显存 32GB；
- **4090 环境**：`NVIDIA GeForce RTX 4090`、`CUDA Version ≥ 12.1`、显存 24GB（实测驱动 580.173.02 下显示 CUDA 13.0）。

驱动 ≥570 对两代卡同时兼容，**两台机器使用同一套驱动安装流程**，在任一机器上照本节操作即可。

> 说明：`nvidia-smi` 显示的是驱动支持的最高 CUDA 版本，PyTorch 自带 CUDA 运行时，**无需单独安装 CUDA Toolkit**（除非你要编译自定义算子）。

**1.4 磁盘规划**

```bash
df -h ~    # 确认 ≥100GB 空闲
```

建议目录规划（统一放在当前 `pi0` 目录下，后文命令默认使用此结构）：

```
~/xmm_ws/pi0/models/pi0_base/                          ← π₀ 模型权重
~/xmm_ws/pi0/data/aloha_sim_transfer_cube_human/       ← 数据集
~/xmm_ws/pi0/outputs/                                  ← 训练输出与 checkpoint
~/xmm_ws/pi0/venv/                                     ← 方案一虚拟环境（py3.10）
~/xmm_ws/pi0/venv312/                                  ← 方案二/三虚拟环境（py3.12，见 2.5）
~/xmm_ws/pi0/lerobot/                                  ← 方案一 LeRobot 源码（v0.4.4）
~/xmm_ws/pi0/lerobot-main/                             ← 方案二/三 LeRobot 源码（main，见 2.5）
```

---

## 二、Python 环境搭建

**2.1 创建 Python 虚拟环境（venv，方案一专用；方案二/三的 venv312 见 2.5）**

Ubuntu 22.04 系统自带 Python 3.10，满足 LeRobot 要求，直接用标准库 `venv` 创建即可（无需 conda）：

```bash
python3 --version                     # 确认 3.10.x
sudo apt install -y python3.10-venv   # 首次使用 venv 需安装该包
python3 -m venv ~/xmm_ws/pi0/venv     # 虚拟环境建在 pi0 目录下
source ~/xmm_ws/pi0/venv/bin/activate # 激活（提示符前出现 (venv)）
pip install -U pip                    # 升级 pip，避免新格式 wheel 安装失败
```

> 每次新开终端需重新执行 `source ~/xmm_ws/pi0/venv/bin/activate`；退出环境用 `deactivate`。

**2.2 安装 PyTorch（三种部署方案与版本约束链，必读）**

> ⚠️ 实测版本约束链：lerobot v0.4.4 要求 `torch>=2.2.1,<2.11.0`，**torch 2.11.x 不被接受**，安装 lerobot 时会自动（从清华源）降级安装 torch 2.10.0。**实测好消息：torch 2.10.0 的 PyPI/清华源默认构建已是 cu128**（4090 机实测 `2.10.0+cu128 12.8`），cu128 同时支持 sm_89 与 sm_120，两机均可用。注意区分：torch ≤2.9 的 PyPI 默认构建为 cu126 及以下，不支持 5090；PyTorch 官方 `whl/cu128` 源的 cu128 构建则两机均可用。

**三种部署方案矩阵：**

| 方案 | 机器 | Python | lerobot | torch | CUDA 运行时 | 状态 |
|------|------|--------|---------|-------|------------|------|
| 方案一 | 4090 机 | 3.10 | v0.4.4 | 2.10.0（清华源默认构建，实测为 cu128） | cu128 | ✅ 实测（装 lerobot 时自动安装） |
| 方案二 | 4090 机 | 3.12（deadsnakes） | main（v0.6.x） | 最新 +cu128（官方源） | cu128 | ⏳ 待验证（见 2.5） |
| 方案三 | 5090 机 | 3.12（deadsnakes） | main（v0.6.x） | 最新 +cu128（官方源） | cu128 | ⏳ 待验证（见 2.5） |

**方案一（4090 机 + py3.10，当前默认路线）**：无需单独安装 torch——执行 2.3 节安装 lerobot 时，pip 会自动从清华源拉取 torch 2.10.0（实测为 cu128 构建），两机均可用。本节其余内容可跳过。

> 注：torch 2.10.0 默认构建即 cu128，意味着"5090 机 + py3.10 + v0.4.4"技术上也可行；但按既定规划 5090 机仍走方案三（py3.12 + main），保持与官方新栈同步。

验证当前 GPU 被正确识别（这一步必须通过，否则后面全部白搭）：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda); \
print(torch.cuda.is_available(), torch.cuda.device_count()); \
print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
```

预期输出：方案一实测为 `2.10.0+cu128 12.8`、`True 1`、`['NVIDIA GeForce RTX 4090']`；方案二/三同为 cu128 构建（`x.y.z+cu128 12.8`），卡名对应各自机器。

**耗时预估（提前知悉，耐心等待）**：torch wheel 约 915MB（4090 机清华源实测约 2.2MB/s ≈ 7 分钟），加上 opencv（54MB）、av（39MB）、cmake（30MB）、wandb（23MB）等，lerobot 全量依赖下载约 **1.5-2GB**，整体安装通常需 **10-30 分钟**。

> 安装完成后建议执行 `pip check`：若报 torchvision 版本冲突（此前按旧文档装过配 torch 2.11 的 torchvision 0.26），执行 `pip install torchvision==0.25.0` 对齐即可（lerobot 本身不强依赖 torchvision，直接卸载亦可）。

**2.3 安装 LeRobot（含 π₀ 支持）**

ModelScope 模型页官方推荐 `pip install "lerobot[pi] @ git+https://github.com/huggingface/lerobot.git"`，但该命令会**完整克隆 GitHub 仓库（含全部 git 历史）+ 从国外 PyPI 拉依赖，国内很慢**。改用浅克隆源码 + editable 安装（快一个数量级，且便于查看/修改源码排错）：

> ⚠️ 版本注意（官方 Release Notes 证实）：LeRobot 从 **v0.5.0** 起要求 **Python ≥3.12**（PR #3023 "require Python 3.12+ as minimum version"，见 https://github.com/huggingface/lerobot/releases/tag/v0.5.0），而 Ubuntu 22.04 系统 Python 是 3.10，装 v0.5.0+ 会报 `Package 'lerobot' requires a different Python`。**v0.4.4 是支持 Python 3.10 的最后一个正式 release**（远程 tag 中 0.4.x 最高即到 v0.4.4），且已采用 v3.0 数据集格式、`[pi]` extra 文档完整（v0.5.0 起官方删掉了 pi 安装说明），与本指南第四节的数据集（v3.0，无需转换）匹配，故必须 pin 到 v0.4.4。

```bash
cd ~/xmm_ws/pi0
git clone -b v0.4.4 --depth 1 https://github.com/huggingface/lerobot.git
# 若 GitHub 克隆仍慢/失败，换镜像加速前缀（二选一）：
# git clone -b v0.4.4 --depth 1 https://gh-proxy.com/https://github.com/huggingface/lerobot.git
# git clone -b v0.4.4 --depth 1 https://gitclone.com/github.com/huggingface/lerobot.git

pip install ffmpeg-python
sudo apt install -y ffmpeg        # torchcodec 视频解码依赖系统 ffmpeg

# editable 安装 + PyPI 国内镜像（会自动降级安装 torch 2.10.0，下载约 1.5-2GB，耗时 10-30 分钟，详见 2.2）：
pip install -e "./lerobot[pi]" -i https://pypi.tuna.tsinghua.edu.cn/simple
```

说明：

- `-e`（editable）安装后源码即 `~/xmm_ws/pi0/lerobot/`，后续核对训练参数名、数据集转换脚本等直接看本地目录；
- 已克隆 main 分支的情况下可就地切换，无需重新克隆：
  `cd ~/xmm_ws/pi0/lerobot && git fetch --depth 1 origin tag v0.4.4 && git checkout v0.4.4`；
- 若希望使用 main 分支（v0.6.x），见 **2.5 节方案二/三**（py3.12 + cu128，待验证）；
- 兜底方案：clone 彻底不通时，直接下载 tag 源码包解压后同样 `pip install -e`：
  `wget https://gh-proxy.com/https://github.com/huggingface/lerobot/archive/refs/tags/v0.4.4.zip`

验证（注意：`| head` 截断长 help 会在末尾抛 `BrokenPipeError`，属正常现象，help 已正常输出即算通过）：

```bash
lerobot-train --help 2>/dev/null | head -20
```

**2.4 安装 ModelScope CLI**

```bash
pip install modelscope
modelscope --version
```

**2.5 方案二/三（待验证）：Python 3.12 + lerobot main + cu128**

方案二（4090 机）与方案三（5090 机）步骤完全相同，仅机器不同。目标：验证 py3.12 + lerobot main（v0.6.x）+ torch cu128 的新栈路线（5090 机最终将落地方案三）。

```bash
# 1. 安装 Python 3.12（22.04 系统源自带 3.10，需 deadsnakes PPA）✅ 已完成
sudo apt update
sudo apt install -y software-properties-common   # add-apt-repository 命令依赖此包
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3.12-dev  # dev 包是后续编译类依赖的前提
python3.12 --version    # 应输出 Python 3.12.x

# 2. 建独立虚拟环境（不覆盖方案一的 venv，便于对照回退）✅ 已完成
python3.12 -m venv ~/xmm_ws/pi0/venv312
source ~/xmm_ws/pi0/venv312/bin/activate   # 激活（提示符前出现 (venv312)）
pip install -U pip                         # 升级 pip，避免新格式 wheel 安装失败
python --version                           # 验证环境内为 Python 3.12.x

# 3. 预装 torch cu128（5090 机必需；4090 机用 cu128 同样可行）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 4. 克隆 main 分支并安装（与方案一的 v0.4.4 目录分开，避免互相污染）
cd ~/xmm_ws/pi0
git clone --depth 1 https://github.com/huggingface/lerobot.git lerobot-main
pip install ffmpeg-python
pip install -e "./lerobot-main[pi]" -i https://pypi.tuna.tsinghua.edu.cn/simple
```

验证与记录要点（验证通过后回填本节状态为 ✅）：

- 与方案一相同的三件套：`pip check`、`python -c "import torch; ..."`（预期 `+cu128`）、`lerobot-train --help`；
- 记录 main 对 torch 的版本约束（安装日志中 `Collecting torch ...` 一行）：若 pip 试图更换第 3 步预装的 torch，确认换入的仍是 cu128 构建，否则按 2.2 节思路固定版本；
- main 分支的 CLI 参数名、数据集接口可能与 v0.4.4 有差异（如 v0.5.0 起官方已移除 pi 安装说明、升级 transformers v5），后续第四至七节流程在此环境下执行时以 `--help` 为准；
- 两方案的 `[pi]` extra 若不存在（新版可能默认内置 π₀），改为 `pip install -e "./lerobot-main"` 并确认 `lerobot.policies.pi0` 可导入。

---

## 三、下载 π₀ 模型（ModelScope）

```bash
mkdir -p ~/xmm_ws/pi0/models
modelscope download --model lerobot/pi0_base \
  --local_dir ~/xmm_ws/pi0/models/pi0_base
```

- 体积约 **14GB**，中断后重新执行同一命令可断点续传。
- 下载完成后检查目录结构：

```bash
ls -lh ~/xmm_ws/pi0/models/pi0_base
```

预期包含：`model.safetensors`（约 14GB，F32 精度保存）、`config.json`（策略配置）等文件。

> 说明：权重以 F32 保存，训练/推理时通过 `--policy.dtype=bfloat16` 转为 bf16 使用，实际占用约 7GB 显存。

---

## 四、下载并验证数据集

**4.1 下载数据集**

```bash
mkdir -p ~/xmm_ws/pi0/data
modelscope download --dataset lerobot/aloha_sim_transfer_cube_human \
  --local_dir ~/xmm_ws/pi0/data/aloha_sim_transfer_cube_human
```

**4.2 数据规格卡**

| 属性 | 值 |
|------|-----|
| LeRobot 格式版本 | **v3.0**（✅ 实测；与 v0.4.4 期望一致，**无需格式转换**） |
| Episode 数 | 50 |
| 总帧数 | 20,000（每集约 400 帧） |
| 采样率 | 50 fps |
| 图像观测 | `observation.images.top`，480×640×3，单目俯视相机（视频编码 av1） |
| 动作维度 | 14（ALoHa 双臂：每臂 6 关节 + 1 夹爪 × 2） |
| 状态维度 | 14（`observation.state`） |
| 任务 | 右臂抓取方块并转移到左臂（人类遥操作演示） |

**4.3 本地加载验证（✅ 已实测通过）**

```python
# test_dataset.py
from lerobot.datasets.lerobot_dataset import LeRobotDataset   # 注意：v0.4.4 路径无 .common

dataset = LeRobotDataset(
    repo_id="aloha_sim_transfer_cube_human",
    root="/home/xy/xmm_ws/pi0/data/aloha_sim_transfer_cube_human",  # ⚠️ 必须指向数据集目录本身（含 meta/info.json），不是父目录
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

实测输出：`episodes: 50`、`frames: 20000`、`action shape: torch.Size([14])`、`image shape: torch.Size([3, 480, 640])`（CHW、float32，已归一化）。

**4.4 格式版本兼容说明**

ModelScope 上该数据集当前版本已是 **v3.0**，与本指南所用的 lerobot v0.4.4 原生匹配，**无需任何转换**。若将来拿到 v2.0 格式的旧数据集（加载时报 `codebase_version` 不匹配），再用 LeRobot 仓库自带的转换脚本逐级升级（v2.0 → v2.1 → v3.0，脚本位于仓库 `scripts/` 目录，如 `convert_dataset_v20_to_v21`），转换后重新运行 4.3 验证。

> 加载路径注意事项（实测踩坑）：`LeRobotDataset` 的 `root` 若指向父目录（`.../data`），会报 `FileNotFoundError: .../data/meta/info.json` 并尝试联网回退；必须指到数据集目录本身。同理，第五节训练命令中 `--dataset.root` 也要写数据集目录全路径。

---

## 五、微调训练

**5.1 显存预算决策表（RTX 5090 = 32GB / RTX 4090 = 24GB，必读）**

π₀ 共约 3.5B 参数。不同训练模式的显存估算：

| 训练模式 | 可训练参数 | 显存估算 | 方案三·5090（32GB） | 方案一/二·4090（24GB） |
|----------|-----------|----------|--------------|--------------|
| 全参数微调（AdamW fp32 状态） | 3.5B | ≈ 50-56GB | ❌ 不可行 | ❌ 不可行 |
| 全参数 + 8-bit 优化器 + 梯度检查点 + batch 1-2 | 3.5B | ≈ 25-30GB | ⚠️ 勉强，易 OOM | ❌ 不可行 |
| **仅训练 Action Expert**（冻结 VLM） | 300M | ≈ 15-20GB | ✅ 可行 | ⚠️ 偏紧（桌面占用后易 OOM，batch 1） |
| **LoRA 微调**（peft） | ~1-5% | ≈ 12-18GB | ✅ 推荐首选 | ✅ 推荐首选（batch 1-2） |

**推荐策略**：均优先 LoRA；若所用 LeRobot 版本的训练脚本未内置 LoRA 开关，方案三（5090 机 32GB）可退而求其次选"冻结 VLM、只训练 Action Expert"，方案一/二（4090 机 24GB）则不建议走该路线。两者在 50 条演示的小数据集上效果差异通常不大。

> 4090 机训练前先关闭占用显存的桌面程序（本机实测桌面占用约 2GB），5090 机余量更足但同理建议。

**5.2 训练命令（lerobot-train）**

ModelScope 模型页给出的官方微调模板 + 针对本数据集的**三项必要适配**（均由源码/实测确认）：

```bash
lerobot-train \
  --dataset.repo_id=aloha_sim_transfer_cube_human \
  --dataset.root=~/xmm_ws/pi0/data/aloha_sim_transfer_cube_human \
  --output_dir=~/xmm_ws/pi0/outputs/pi0_aloha_cube \
  --job_name=pi0_aloha_cube \
  --policy.type=pi0 \
  --policy.pretrained_path=~/xmm_ws/pi0/models/pi0_base \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --rename_map='{"observation.images.top":"observation.images.base_0_rgb"}' \
  --peft.method_type=LORA \
  --peft.r=16 \
  --steps=20000 \
  --batch_size=2        # 方案一（4090 机 24GB）建议 2，OOM 降 1；方案三（5090 机 32GB）可用 4
```

> 已封装为启动脚本 [`train_pi0.sh`](./train_pi0.sh)：含训练前自检（权重/数据集/CLI/GPU）、日志落盘 `logs/`、支持 `STEPS`/`BATCH_SIZE` 环境变量覆盖（如 `STEPS=200 bash train_pi0.sh` 先做冒烟测试）。

参数说明：

| 参数 | 含义 |
|------|------|
| `--dataset.repo_id` + `--dataset.root` | 本地数据集（**root 必须指向数据集目录本身**，即含 `meta/info.json` 的目录，见 4.3 踩坑说明） |
| `--policy.type=pi0` | 选择 π₀ 策略（v0.4.4 支持 pi0/pi0_fast/pi05/act/smolvla 等 14 种，必填） |
| `--policy.pretrained_path` | 本地 π₀ 基座权重目录（即第三节下载位置；不传则从头训练） |
| `--policy.dtype=bfloat16` | F32 权重加载后转 bf16，显存减半 |
| `--policy.push_to_hub=false` | ⚠️ **必加**：基座 config.json 里 `push_to_hub=true`，不关会在保存 checkpoint 时尝试上传 HF Hub（未登录会报错） |
| `--rename_map='{"observation.images.top":"observation.images.base_0_rgb"}'` | ⚠️ **必加**：基座期望相机键名 `base_0_rgb`，本数据集只有 `top`；v0.4.4 通过 `rename_observations_processor` 做映射（JSON 字符串语法实测可用；基座声明的其余相机键缺失时自动用空图适配）。state/action 14 维也会自动 pad 到基座的 32 维，无需处理 |
| `--peft.method_type=LORA` + `--peft.r=16` | 启用 LoRA（按 5.1 决策表的首选路线）。π₀ 自带官方默认 target_modules（Action Expert 的 q/v 投影 + state/action 投影层），无需手填；不传 `--peft.*` 则走全参数（本机显存不可行） |
| `--steps` / `--batch_size` | 训练步数与批大小 |

> 以上参数名与语法均已按本机 v0.4.4 源码/`lerobot-train --help`/draccus 解析实测核对（注意是 `pretrained_path`；旧资料常见的 `--policy.path` 在本版本不存在）。`--policy.chunk_size`、`--policy.gradient_checkpointing=true`（显存再紧时开启）等策略参数可按需追加。

**5.3 超参数建议**

| 超参数 | 建议值 | 说明 |
|--------|--------|------|
| steps | 10,000-20,000 | 50 条演示数据，20k 步 × batch4 ≈ 4 个 epoch |
| batch_size | 方案一/二（4090）：2（OOM 降 1）；方案三（5090）：4（OOM 降 2） | LoRA 模式下方案三可上 4-8，4090 一般 2-4 |
| dtype | bfloat16 | 5090/4090 均支持（Ada 上 bf16 原生支持） |
| chunk_size | 默认（50） | π₀ 动作块长度，小数据集不必改 |
| 学习率 | 默认（约 1e-5 量级） | 微调不必调大 |

**5.4 训练监控与产物**

- 训练日志与 checkpoint 写入 `~/xmm_ws/pi0/outputs/pi0_aloha_cube/checkpoints/`；
- loss 应在前几百步明显下降后进入平台期；
- 预计耗时：20k 步约 **3-8 小时**（取决于数据加载与 chunk 配置；方案一/二在 4090 上因显存限制 batch 更小，耗时预计比方案三多 1.5-2 倍，暂未实测对照）；
- 最终用于推理的是 `checkpoints/last/` 下的策略权重目录。

---

## 六、安装 ALoHa 仿真环境

ALoHa 仿真双臂环境源自 ACT 论文（arXiv 2304.13705）的仿真实现，基于 MuJoCo / dm_control。

**6.1 安装依赖**

```bash
pip install mujoco dm_control
# LeRobot 的 aloha 仿真 extra：会按 pyproject.toml 声明（gym-aloha>=0.1.2,<0.2.0）
# 从清华源自动安装独立的 PyPI 包 gym-aloha（实测 0.1.4，位于 venv site-packages，非 lerobot 源码）
cd ~/xmm_ws/pi0
pip install -e "./lerobot[aloha]" -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> 若 `lerobot[aloha]` extra 在你安装的版本中不存在，查阅 LeRobot 仓库文档中 ALOHA 仿真环境的安装说明（不同版本 extra 命名可能有变化）。

**6.2 任务说明**

| 属性 | 值 |
|------|-----|
| 环境名 | gym_aloha |
| 任务 ID | `AlohaTransferCube-v0` |
| 内容 | 左臂区域出现方块，双臂协作将其搬运至右臂托盘 |
| 单轮步数 | rollout 上限 400 步：gym-aloha 注册默认 `max_episode_steps=300`，但 lerobot 的 `AlohaEnv` 配置用 `episode_length=400` 覆盖（实测 gym_kwargs） |
| 观测 | 裸 `gym.make` 默认只有 `top` 图像（480×640×3，实测）；lerobot 创建环境时传 `obs_type="pixels_agent_pos"`，环境同时返回 14 维 `agent_pos`，再经 `features_map` 改名为 `observation.state`/`observation.images.top`（与数据集键名对齐） |

**6.3 渲染方式**

```bash
# 台式机带显示器：默认 glfw 渲染即可
# 无头/SSH 场景：使用 EGL GPU 渲染
export MUJOCO_GL=egl
```

**6.4 冒烟测试（训练前必做）**

先用随机动作验证环境能正常跑通，排除渲染/依赖问题。已封装为 [`smoke_test_env.py`](./smoke_test_env.py)（含观测/动作空间断言与帧率统计）：

```bash
python smoke_test_env.py                      # 台式机带显示器：默认 glfw 渲染
MUJOCO_GL=egl python smoke_test_env.py        # 无头/SSH：EGL GPU 渲染
```

实测输出（EGL，4090 机）：`obs keys: ['top']`（480×640×3 uint8）、`action 空间: Box(-1.0, 1.0, (14,))`、`冒烟测试 OK | 步数: 100 | 帧率: 241 FPS`。

<details>
<summary>脚本核心逻辑（展开查看）</summary>

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
  --policy.pretrained_path=~/xmm_ws/pi0/outputs/pi0_aloha_cube/checkpoints/last/pretrained_model \
  --env.name=gym_aloha \
  --env.task=AlohaTransferCube-v0 \
  --eval.n_episodes=25
```

> 参数名已按本机 v0.4.4 的 `lerobot-eval --help` 实测核对；可加视频保存参数（如 `--eval.save_video=true`，视版本而定）。

**7.2 独立推理代码（理解原理用）**

ModelScope 模型页给出的推理接口：

```python
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
# from lerobot.policies.pi0.processing_pi0 import preprocess, postprocess

policy = PI0Policy.from_pretrained(
    "/home/xy/xmm_ws/pi0/outputs/pi0_aloha_cube/checkpoints/last/pretrained_model"
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
| `no kernel image is available for execution on the device` | 5090（sm_120）上用了 torch ≤2.9 的默认构建（cu126 及以下） | 升级到 torch 2.10.0（PyPI 默认构建即 cu128），或直接按 2.5 节方案二/三部署 |
| 装 lerobot 时自动下载 915MB 的 torch | v0.4.4 要求 `torch<2.11.0`，自动降级到 2.10.0 | 属正常现象；实测默认构建即 cu128，两机均可用 |
| `lerobot-train` 在 `Creating policy` 阶段被"已杀死" | **系统内存（RAM）OOM**：加载 14GB F32 权重峰值约 21-29GB，超过 32GB 物理内存减桌面占用后的余量（实测 4090 机，内核日志 `Out of memory: Killed process ... lerobot-train`） | ① 关闭浏览器/编辑器释放 RAM；② 扩容 swap 兜底：`sudo fallocate -l 24G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile`（仅加载阶段变慢几分钟，训练不受影响） |
| `nvidia-smi` 看不到 GPU / CUDA Version 过低 | 驱动过旧 | 升级驱动 ≥570 并重启（两机通用） |
| 训练启动即 CUDA OOM | 全参数微调超显存（4090 尤甚） | 改 LoRA 或仅训 Action Expert；batch 降到 1-2；开梯度检查点；关闭桌面程序释放约 2GB |
| 模型加载阶段 OOM | F32 权重直接载入 | 确认 `--policy.dtype=bfloat16` |
| 安装 lerobot 报 `requires a different Python: 3.10.x not in '>=3.12'` | LeRobot 从 v0.5.0 起要求 Python ≥3.12（官方 PR #3023） | pin 到 v0.4.4 tag 重新安装（第二节 2.3） |
| `codebase_version` 不匹配报错 | 数据集 v2.0 旧格式 vs LeRobot 期望 v3.0（当前 ModelScope 数据集已是 v3.0，正常不会遇到） | 用官方转换脚本升级数据集（第四节 4.4） |
| modelscope 下载中断 | 网络波动 | 重跑同一命令，自动续传 |
| MuJoCo 渲染报 `libGL` / GLFW 错误 | 图形库缺失或无头环境 | `sudo apt install libgl1 libglfw3`；无头改用 `MUJOCO_GL=egl` |
| 训练极慢 / dataloader 打满 CPU | 视频帧解码瓶颈 | 确认系统 ffmpeg 已装；适当提高 dataloader worker 数 |
| rollout 动作抖动 | 动作块边界未平滑 | 确认使用 π₀ 默认 chunk 执行逻辑；必要时启用 temporal ensemble |

---

## 九、附录

**9.1 一键流程清单**

```
□ 驱动 ≥570，nvidia-smi 可见 RTX 5090（32GB）或 RTX 4090（24GB），CUDA ≥ 12.8
□ venv 虚拟环境 + torch 验证通过（实测均为 cu128；识别到当前机器的卡型号）
□ lerobot[pi] + modelscope 安装完成
□ 模型下载完成（~/xmm_ws/pi0/models/pi0_base，约 14GB）
□ 数据集下载完成并通过 LeRobotDataset 加载验证
□ 数据集格式确认（实测 v3.0，与 lerobot v0.4.4 匹配，无需转换）
□ 冒烟测试：仿真环境随机动作跑通
□ 训练完成，checkpoint 生成
□ 25 轮 rollout 成功率统计 + 视频验收
```

**9.2 目录结构总览**

```
~/xmm_ws/pi0/
├── venv/                          ← 方案一虚拟环境（py3.10）
├── venv312/                       ← 方案二/三虚拟环境（py3.12）
├── lerobot/                       ← 方案一 LeRobot 源码（v0.4.4）
├── lerobot-main/                  ← 方案二/三 LeRobot 源码（main）
├── models/pi0_base/
│   ├── model.safetensors          (~14GB, F32)
│   └── config.json
├── data/aloha_sim_transfer_cube_human/
│   ├── meta/                      (info.json, episodes.jsonl, tasks.jsonl)
│   └── data/ | videos/            (chunk 数据)
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

*文档版本 v1.2 · 2026-08-18 更新：确立三种部署方案——方案一（4090 机 + py3.10 + lerobot v0.4.4 + torch 2.10.0+cu128，✅ 实测）；方案二/三（py3.12 + lerobot main + torch cu128，⏳ 待验证，见 2.2/2.5）· 配套 HTML 版：17-pi0本地化部署与ALoHa仿真训练指南-v1.0.html*
