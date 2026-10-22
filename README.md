# ws 工作区索引

本工作区按「四大块 + 公共底座」组织，每个顶层目录基本是一个独立项目（多为 git 子模块）。当前机器环境：**Ubuntu 22.04 + NVIDIA RTX 5090（sm_120）**。

本文件是**纯索引**：按编号段列出每个目录的用途、环境、验证状态与入口文档。详细使用说明见各项目的 `START_HERE.md` → `README.md` → `docs/`；目录重组的完整方案与执行记录见 [docs/ws-restructure-plan.md](docs/ws-restructure-plan.md)。

> **约定**：顶层目录命名 `NNN-kebab-case`（专有名词保留原大小写）；子模块需 `git submodule update --init` 拉取；模型权重统一软链到 `models/`，用本地源码离线加载。

## 编号规范

| 段 | 用途 |
|---|---|
| **1xx** | 具身智能模型系列（VLA、世界模型、具身基础模型） |
| **2xx** | 非具身模型（LLM 基模、生成类） |
| **3xx** | 真机部署测试、人形机器人、趣味本地复现 |
| **4xx** | 预留，暂时留空 |
| **5xx** | 具身数据 infra（数据采集 / 重定向 / 遥操作） |
| 无编号 | 公共底座：`models/`、`datasets/`、`assets/`、`scripts/`、`docs/` |

## 状态说明

- **✅ 已验证/可运行**：已有本地实跑输出，或轻量流程可直接跑通。
- **🟡 基础可用/部分验证**：核心或轻量功能可用，完整链路依赖模型、数据、硬件或外设。
- **⚠️ 暂未验证/环境待定**：项目较重或环境尚未就绪，受显存、模型体积、多卡、外设或运行时长限制。

> 状态反映**项目本身的验证成熟度**。换新机器（如新克隆的工作区）后，需先拉取子模块、按各项目文档重建 venv/conda 环境，方能复现。

## 项目索引

### 1xx · 具身智能模型系列

| 目录 | 用途 | 环境 | 状态 | 入口 |
|---|---|---|---|---|
| [`100-openVLA`](100-openVLA/) | OpenVLA 7B 视觉-语言-动作模型；MuJoCo（WX250S/Franka）仿真推理 + LIBERO 评测 | `.venv`（3.10） | ✅ | [START_HERE](100-openVLA/START_HERE.md) |
| [`101-pi-serial-models`](101-pi-serial-models/) | π0 / π0.5 与 ACT 策略训练与评估（lerobot）；ALOHA 双臂 + LIBERO 两条评测链路 | `.venv312`（3.12） | ✅ | [START_HERE](101-pi-serial-models/START_HERE.md) |
| [`102-cosmos`](102-cosmos/) | NVIDIA Cosmos 世界模型（RoboLab + cosmos-framework）；视频预测 + 动作策略 | conda `cosmos`（3.12）+ `3rd/cosmos-framework/.venv`（3.13） | ✅ | [START_HERE](102-cosmos/START_HERE.md) |
| [`103-lingbot`](103-lingbot/) | Lingbot 系列：depth（深度补全）/ video（MoE 视频生成）/ world-v2（交互世界模型）/ vla-v2 | 多套；depth 用 conda `lingbot-depth`，其余待定 | 🟡 | [START_HERE](103-lingbot/START_HERE.md) |
| [`104-unitreerobotics`](104-unitreerobotics/) | 宇树 UnifoLM-WMA 世界模型-动作；视频预测 + 动作生成的自回归交互推理 | `.venv`（3.10） | ✅ | [START_HERE](104-unitreerobotics/START_HERE.md) |
| [`105-damo`](105-damo/) | 达摩院：RynnBrain（具身多模态理解）+ RynnWorld-4D（4D 世界模型） | 待定 | 🟡 | [START_HERE](105-damo/START_HERE.md) |

### 2xx · 非具身模型

| 目录 | 用途 | 环境 | 状态 | 入口 |
|---|---|---|---|---|
| [`200-video`](200-video/) | ComfyUI + MiniMax 视频生成工作流 | `ComfyUI/.venv`（3.10） | ✅ | [START_HERE](200-video/START_HERE.md) |
| [`201-colibri`](201-colibri/) | 纯 C 的 GLM MoE 运行时（磁盘 / RAM / VRAM 分层加载专家） | 无（`make` 编译） | ⚠️ | [START_HERE](201-colibri/START_HERE.md) |

### 3xx · 真机部署 / 人形机器人 / 趣味复现

| 目录 | 用途 | 环境 | 状态 | 入口 |
|---|---|---|---|---|
| [`300-robots`](300-robots/) | RynnMotion 机器人运动框架（异构硬件统一抽象：9 家厂商真机驱动 + 12 款 MuJoCo 机械臂）+ `sim_robot` 仿真验证 | 待定（`sim_robot` 用系统 `python3` + mujoco） | 🟡 | [START_HERE](300-robots/START_HERE.md) |
| [`301-microduck`](301-microduck/) | Microduck 微型机械臂：`microduck`（Rust 板载运行时）+ `microduck_rl`（mjlab + PPO 训练） | 待定（根 `.venv` 与 `microduck_rl/.venv`，均 3.12） | ⚠️ | [START_HERE](301-microduck/START_HERE.md) |
| [`302-kingkongrobotics`](302-kingkongrobotics/) | 金刚机器人 `jumper`（跳跃足式机器人，mjlab + rsl_rl） | `jumper/.venv`（3.12） | ⚠️ | [START_HERE](302-kingkongrobotics/START_HERE.md) |

### 5xx · 具身数据 infra

| 目录 | 用途 | 环境 | 状态 | 入口 |
|---|---|---|---|---|
| [`500-umr`](500-umr/) | UMR：人体动作 → 点云对应 → 机器人 qpos 重定向（实测 LAFAN1 跳舞 → Unitree G1） | conda `umr` | ⚠️ | [START_HERE](500-umr/START_HERE.md) |
| [`501-human-dexterous-teleop`](501-human-dexterous-teleop/) | 人体姿态 / 手势识别、灵巧手遥操作、动捕数据采集（MediaPipe / MuJoCo / ROS2） | 多套：3 个 `.venv`（3.10）+ conda `4D-humans` | 🟡 | [START_HERE](501-human-dexterous-teleop/START_HERE.md) |
| [`502-DexAgent`](502-DexAgent/) | DexAgent agentic Human2Sim2Robot 框架复现：人类操作视频 → 物理验证灵巧手轨迹 → 随机化多视角训练视频 | conda `dexagent`（3.11，待建） | ⚠️ | [START_HERE](502-DexAgent/START_HERE.md) |
| [`503-do-as-i-do`](503-do-as-i-do/) | Do as I Do 固定管线复现：手物视频重建 6DoF 跟踪 → Sharpa Wave 重定向（MuJoCo Warp MPC）→ dual UR3e 回放 | conda：doasido_retargeting / sam3 / tapnet 已建（cu128 实测），sam3d / hawor / deployment 待建 | 🟡 | [START_HERE](503-do-as-i-do/START_HERE.md) |
| [`504-retargeting`](504-retargeting/) | 穿戴设备重定向系列：PICO 动捕 + 手套 → G1 全身（IK / SONIC）+ Wuji / Dex3 手部；MuJoCo 回放、LeRobot v2.1 录制 | 三链路 replay 全通（data/ 3 session 实测：GUI 起窗 / ZMQ 实收帧；三链路均模块化重构、tests 全绿），live 待设备 | 🟡 | [START_HERE](504-retargeting/START_HERE.md) |

### 公共底座（无编号）

| 目录 | 说明 | 入口 |
|---|---|---|
| [`models/`](models/) | 模型权重软链统一目录（指向 `~/.cache/modelscope`、`~/.cache/huggingface`） | [README](models/README.md) |
| [`datasets/`](datasets/) | 数据集位置索引（实体数据留各项目内：aloha / libero / pusht 等 lerobot v2） | [README](datasets/README.md) |
| [`assets/`](assets/) | 仿真资产（`mujoco/mujoco_menagerie` 机器人模型库） | [README](assets/README.md) |
| [`scripts/`](scripts/) | 统一部署与环境脚本（PyTorch / RTX 5090、venv editable 检查、git 配置核查） | — |
| [`docs/`](docs/) | 工作区文档：目录重组方案、模型测试目录、具身智能资料 | — |

## 顶层文件

| 文件 | 说明 |
|---|---|
| `README.md` | 工作区总览索引（本文件） |
| `LICENSE` | 许可证 |
| `.gitmodules` | 子模块登记（21 个 gitlink） |
| `.gitignore` | git 忽略规则 |
| `.qoder/rules/user-profile.md` | 用户身份设定（Qoder 规则，始终生效） |

## RTX 5090 PyTorch 环境提示

RTX 5090 对应 `sm_120`，旧版 PyTorch/CUDA wheel 可能出现架构不支持或 CUDA 不可用。`scripts/` 下提供统一部署脚本，安装 **CUDA 12.8 / 13.0 版 PyTorch 2.11.0**：

| 脚本 | 适用 |
|---|---|
| `setup_py310_cu128_rtx5090_torch2110.sh` | py3.10 环境（旧项目），默认读 `~/Downloads` 下本地 cu128 wheel |
| `setup_py312_cu128_rtx5090_torch.sh` | py3.12 环境（lerobot v0.6.2+ 等新项目） |
| `setup_py312_cu130_rtx5090_torch.sh` | py3.12，清华 PyPI 源装 `torch 2.11.0+cu130`，需驱动 ≥ 580（无需本地 wheel） |

辅助脚本：`check_venv_editable.py`（检查各 venv 的 editable 包 import 状态，改名后排查用）、`check_git_config.sh`（git 配置核查）。

安装到当前已激活的 conda/venv 后验证：

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
# 期望: PyTorch 2.11.0+cu1xx, CUDA: True, GPU: NVIDIA GeForce RTX 5090
```

## 子模块按需拉取

21 个子模块默认不随 `git clone` 下载，按需拉取；完整路径清单见 [`.gitmodules`](.gitmodules)。

```bash
cd /home/damorobot/ws

# 拉取全部子模块
git submodule update --init --recursive

# 只拉取指定的某一个（示例：101 的 lerobot）
git submodule update --init 101-pi-serial-models/repos/lerobot-main
```
