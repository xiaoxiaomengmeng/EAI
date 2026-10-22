# pico2g1 部署复现实测记录

> **证据等级**：实测记录型。本文记录 2026-10-20/21 两日在本机（Ubuntu 22.04 + RTX 5090，damorobot-Z890-UD）从零搭建到回放/录制验证的实际命令、输出与判据，可直接照抄复现。
> 与解析系列的分工（[02-g1_ik](02-g1_ik架构与关键技术解析.md)、[03-g1_sonic](03-g1_sonic架构与关键技术解析.md)、[04-g1_sonic_bridge](04-g1_sonic_bridge架构与关键技术解析.md)）：解析篇讲「为什么这么设计」、逐模块原理；本文讲「怎么搭、怎么跑、怎么判断成功」。当前状态见 [START_HERE.md](../START_HERE.md)。

## 目录

- [0. 复现总路线](#0-复现总路线)
- [1. 公共底座（一次管三条链路）](#1-公共底座一次管三条链路)
  - [1.1 系统依赖](#11-系统依赖) ｜ [1.2 数据准备](#12-数据准备)
- [2. 链路一：g1_ik（完整验证阶梯）](#2-链路一g1_ik轻量-ik完整验证阶梯)
  - [2.1 搭环境](#21-搭环境) ｜ [2.2 阶梯 1：模型装配](#22-验证阶梯-1模型装配不起-gui)
  - [2.3 阶梯 2：GUI 回放](#23-验证阶梯-2gui-回放) ｜ [2.4 阶梯 3：GUI 录制 + 产物校验](#24-验证阶梯-3gui-录制--产物校验)
- [3. 链路二：g1_sonic（环境已就位）](#3-链路二g1_sonicsonic-策略环境已就位)
  - [3.1 搭环境](#31-搭环境) ｜ [3.2 单独下权重（镜像）](#32-单独下权重本机必须走镜像)
  - [3.3 依赖冒烟 + 权重加载](#33-验证依赖冒烟--权重加载不起-gui) ｜ [3.4 回放](#34-回放)
- [4. 链路三：g1_sonic_bridge（数据桥）](#4-链路三g1_sonic_bridge数据桥)
- [5. 实测坑汇总](#5-实测坑汇总)
- [6. 状态时间线](#6-状态时间线)

## 0. 复现总路线

三条链路互相独立（代码互不 import、各自 `.venv`），按依赖从浅到深排序复现：

```
第 1 步  公共底座       系统依赖（libxcb-cursor0 / ffprobe）+ session 数据解压
第 2 步  g1_ik          环境最轻（无权重下载）→ 验证阶梯完整走通（装配→回放→录制）
第 3 步  g1_sonic       多一步 onnx 权重下载（本机 HF 直连不通，需镜像）
第 4 步  g1_sonic_bridge 无 GUI 依赖最少，最后收尾
```

## 1. 公共底座（一次管三条链路）

### 1.1 系统依赖

```bash
# PySide6 6.5+ 的 xcb 插件硬依赖（两条 GUI 链路共用；2026-10-20 实测缺它起不了窗）
sudo apt install libxcb-cursor0

# 录制产物校验（scripts/04）用系统 ffprobe；实测本机已就位 /usr/bin/ffprobe
command -v ffprobe
```

Python 用系统 `python3 -m venv`（实测 3.12），不装 conda 也能跑；pip 走默认源或清华源均可。

### 1.2 数据准备

session 是 `umi_multimodal_capture` schema v1.1.1 录制包，放到 `504-retargeting/data/`（根 `.gitignore` 整体忽略 `data/`，zip 与解压目录都不入库）：

```bash
unzip session_20260828_161538.zip -d /home/damorobot/ws/504-retargeting/data/
```

回放必需 `data/chunk-000/` 下四类 parquet：`vr_full_body` / `vr` / `glove_left` / `glove_right`（部分 session 还带 `vr_left_hand` / `vr_right_hand` 与 `.jsonl`，属采集器原生共存文件，本链路不读）。概览与齐全性判据：

```bash
cd /home/damorobot/ws/504-retargeting
scripts/00_list_sessions.sh
```

**实测输出**（2026-10-20）：3 个 session（161120 / 161459 / 161538，2026-08-28 采集）四类 parquet 全部齐全（`parquet: 四类齐全 ✅`）。判据：每个 session 显示四类齐全；有缺项时 01 脚本会在回放中因文件替换规则找不到配套数据而报错。

## 2. 链路一：g1_ik（轻量 IK，完整验证阶梯）

### 2.1 搭环境

```bash
cd /home/damorobot/ws/504-retargeting/pico2g1/g1_ik
bash create_env.sh
```

`create_env.sh` 五步（[1/5] 拉 wuji → [2/5] 建 .venv → [3/5] 升 pip → [4/5] 装运行依赖 → [5/5] wuji editable `--no-deps`），`set -euo pipefail` 全程短路，末尾自动冒烟 import 10 个关键包，任一失败即中止。

`fetch_wuji.sh` 可单独预下载（慢网窗口提前拉），三个防御：① **幂等**——目录已存在则跳过 clone，但始终补跑 `git submodule update --init --recursive` 修复缺失的 `wuji-description` 子模块（手部 MJCF/URDF 全在里面，缺了 build_model 直接 FileNotFoundError）；② **残缺检出**——目录存在但无 `.git`（clone 中断残留）时报错并给出清理命令，不在坏目录上继续；③ **位置对齐**——目标路径写死 `$REPO_ROOT/3rd/wuji-retargeting`，与 create_env.sh 的 editable 安装位置、`g1_config.py` 的 `WUJI_ROOT` 默认值三处一致，不会拉两份。

wuji editable 为什么 `--no-deps`：wuji-retargeting 声明的依赖 `wuji_sdk` 是只随真实手套硬件分发的私有 wheel，PyPI 上没有，普通安装会直接中止；而本链路只用 `wuji_retargeting.Retargeter`（源码核实：它从不 import wuji_sdk），所以 [4/5] 先装齐它真正需要的求解器 nlopt、pinocchio，[5/5] 再 `--no-deps` 装本体。

**实测版本**（2026-10-21 复核，环境 2026-10-20 建）：

| 包 | 版本 |
|---|---|
| mujoco | 3.16.0 |
| numpy | 2.5.4 |
| pyarrow | 26.0.0 |
| PySide6 | 6.12.0 |
| scipy | 1.18.1 |
| websockets | 17.3 |
| pyyaml | 6.0.3 |
| nlopt | 2.11.0 |
| pinocchio | 4.1.0 |
| wuji_retargeting | editable 安装（无版本号） |

### 2.2 验证阶梯 1：模型装配（不起 GUI）

```bash
cd pico2g1/g1_ik
.venv/bin/python -c "from g1_model import build_model; m = build_model(); print('nq =', m.nq)"
```

**实测输出**：`nq = 76`（7 基座 + 29 本体 + 40 手部）。判据：nq=76 即 G1+Wuji Hand 装配正确、wuji-description 子模块完整（缺失会 FileNotFoundError）。

### 2.3 验证阶梯 2：GUI 回放

```bash
cd /home/damorobot/ws/504-retargeting
DRY_RUN=1 scripts/01_replay_g1_ik.sh session_20260828_161538   # 先干跑看命令
scripts/01_replay_g1_ik.sh session_20260828_161538             # 真跑
# 可透传：--scene pulse --render-shadows
```

脚本自动定位 `data/` 下 session、校验 parquet 与 `.venv`、远程会话探测 DISPLAY（窗口出在物理机屏幕，远程看不到，需 AnyDesk 或坐到机器前）。

**实测**（2026-10-20）：GUI 起窗、50 Hz 回放推进、状态栏实时显示 IK 残差；终端首行打印 `MuJoCo model ready: 76 qpos (robot 76, environment 0)`。判据：起窗 + 人体线框（正视/侧视/双手）与 G1 同步运动；`裸跑 python launch_g1_ik.py` 打印 usage 是正常行为（必须带 replay/live 子命令），不是报错。

### 2.4 验证阶梯 3：GUI 录制 + 产物校验

界面录制面板开始/停止后，后台把录制导出为 LeRobot v2.1 数据集（三路 1280×720/30fps H.264 + 69 维关节 + 基座位姿）到 `pico2g1/recordings/`（gitignore）：

```bash
# 校验/诊断录制产物（状态机：完成 / 导出中断可恢复 / 未导出）
pico2g1/g1_ik/.venv/bin/python scripts/04_validate_recording.py <录制名> --list
# 中断恢复（必须 EGL，无 DISPLAY 进程里 mujoco.Renderer 需要 EGL 上下文）
cd pico2g1/g1_ik && MUJOCO_GL=egl .venv/bin/python -m sim_scenes.lerobot_recording <录制目录>
```

**实测**（2026-10-20）：录制实跑产出 `g1_20261020_181011_a63011`，后台导出被中断（数据本体完好）。**2026-10-21 复核：`recordings/` 目录已不在本机**（产物被清理），录制链路结论仍为「已实跑、可复录」；如需数据集，重新走一遍本节即可。导出中断的根因与三状态判定详见 [02 §3.8](02-g1_ik架构与关键技术解析.md#38-录制链路与校验恢复)。

## 3. 链路二：g1_sonic（SONIC 策略，环境已就位）

### 3.1 搭环境

```bash
cd /home/damorobot/ws/504-retargeting/pico2g1/g1_sonic
bash setup.sh
```

与 g1_ik 同构（独立 .venv + wuji 复用 `3rd/` 同一份检出），额外装 onnxruntime，并把权重下载委托给 `download_models.sh`。**坑（实测）**：`setup.sh` 全流程跑完后 `models/` 可能仍是空的——本机直连 huggingface.co 探不到 Content-Length，下载段静默失败，脚本不报错。

### 3.2 单独下权重（本机必须走镜像）

```bash
cd pico2g1/g1_sonic
HF_BASE=https://hf-mirror.com/nvidia/GEAR-SONIC/resolve/main/sonic_v1_1 bash download_models.sh
```

**实测输出**（2026-10-21，三件套字节数精确匹配后落位）：

```
model_encoder.onnx OK (50117796 bytes)
model_decoder.onnx OK (149778638 bytes)
observation_config.yaml OK (2475 bytes)
```

### 3.3 验证：依赖冒烟 + 权重加载（不起 GUI）

```bash
# 11 包冒烟（与 g1_ik 的 10 包相比多 onnxruntime）
.venv/bin/python -c "import onnxruntime, PySide6.QtCore, pinocchio; print('ok')"
# onnx 加载冒烟：能建 InferenceSession 且 IO shape 合理
.venv/bin/python - <<'EOF'
import onnxruntime as ort, pathlib
d = pathlib.Path("models")
for name in ("model_encoder.onnx", "model_decoder.onnx"):
    s = ort.InferenceSession(str(d / name), providers=["CPUExecutionProvider"])
    print(name, [(i.name, i.shape) for i in s.get_inputs()], "->", [(o.name, o.shape) for o in s.get_outputs()])
EOF
```

**实测输出**（2026-10-21）：11 包全通；encoder `obs_dict [1,1751] → encoded_tokens [1,64]`，decoder `obs_dict [1,994] → action [1,29]`（29 = G1 本体 DoF）。判据：两个 session 可建、decoder 输出维度为 29。

### 3.4 回放

```bash
cd /home/damorobot/ws/504-retargeting
DRY_RUN=1 scripts/02_replay_g1_sonic.sh session_20260828_161538
scripts/02_replay_g1_sonic.sh session_20260828_161538
```

注意与 g1_ik 的入参差异：本链路收 **session 目录**（脚本自动定位），不是单个 parquet 文件。

**实测**（2026-10-21）：dry-run 正确打印命令（入参为 session 目录）；真跑 GUI 起窗、物理显示屏可见画面、SONIC v1.1 ONNX 策略驱动 G1；终端打印 `MuJoCo ready: 76 qpos (robot 76, environment 0)`，DISPLAY 自动探测 `:1` 生效。判据达成：起窗 + 策略驱动（物理步进会真的推道具，区别于 IK 链路的运动学锁定）。

## 4. 链路三：g1_sonic_bridge（数据桥）

```bash
cd pico2g1/g1_sonic_bridge && bash setup.sh        # 建 .venv：torch(CPU) / pyzmq / dex-retargeting
cd /home/damorobot/ws/504-retargeting
DRY_RUN=1 scripts/03_replay_g1_sonic_bridge.sh session_20260828_161538
SONIC_ENDPOINT=tcp://*:5557 scripts/03_replay_g1_sonic_bridge.sh session_20260828_161538
```

无界面、依赖最少：重采样 50 Hz + 5 帧滑窗打包，ZMQ PUB 发布（topic `pose`，单帧 = topic 4B + 1280B 头 + 5 帧二进制；`SNDHWM=2` 无订阅者时丢弃）。入参与 g1_ik 相同（vr_full_body parquet 文件）；手部走 Dex3 重定向（腕 + 拇指/食指/中指三指尖共 4 个关键点，`dex-retargeting` 向量法），不需要 G1 模型和 wuji。

**坑（实测）**：默认端口 5556 被钉钉（com.alibabainc.dingtalk）占用，`socket.bind` 报 `ZMQError: Address already in use`——注意链路本体无恙（1368 帧转换、窗口打包回环校验都在崩溃前完成，勿误判为数据/依赖问题）。处置：源码 `SONIC_ENDPOINT` 已改为环境变量可覆盖（对齐同文件 `ASSETS_DIR` 惯例，默认值仍为 5556），5556 被占时如上例前置 `SONIC_ENDPOINT=tcp://*:5557`。

**实测**（2026-10-21）：venv 依赖全通（pyzmq 27.2.0、torch 2.14.1+cpu、dex_retargeting）；桥以 5557 发布，SUB 订阅者实收 300/300 帧、首帧 5320 字节、topic 前缀 `pose`。判据达成：转换 + 发布 + 外部订阅收帧全链路通。注：仓库内无消费端，下游 SONIC 进程需另行准备（消息字段与二进制布局见 [g1_sonic_bridge/README.md](../pico2g1/g1_sonic_bridge/README.md)）。

## 5. 实测坑汇总

| 坑 | 现象/根因 | 处置（实测验证） |
|---|---|---|
| libxcb-cursor0 缺失 | PySide6 6.5+ xcb 插件报 "Could not load the Qt platform plugin xcb" | `apt install libxcb-cursor0`；报错文本固定含这句，即使已装库、DISPLAY 连不上时仍报同一句，需分层排查 |
| 远程会话无 DISPLAY | IDE/SSH 会话环境变量为空，物理桌面 Xorg 在 `:1` | 01/02 脚本自动探测（`/tmp/.X11-unix/X*` + xset 试连）；手动等价 `export DISPLAY=:1 XAUTHORITY=/run/user/$(id -u)/gdm/Xauthority` |
| 离线恢复导出挂 GL | `mujoco.Renderer` 报 "an OpenGL platform library has not been loaded" | 恢复命令前加 `MUJOCO_GL=egl`（GUI 进程内导出走进程内 GL 不受影响） |
| HF 直连不通 | setup.sh 跑完 `models/` 仍空（curl 探不到 Content-Length，静默失败） | `HF_BASE=https://hf-mirror.com/...` 镜像下载（§3.2） |
| 5556 被钉钉占用 | ZMQ 桥 `socket.bind` 报 `Address already in use`（数据转换在崩溃前已完成） | 前置 `SONIC_ENDPOINT=tcp://*:5557` 换端口（源码已支持，默认仍 5556；§4） |
| 裸跑入口打 usage | `python launch_g1_ik.py` 报 `error: the following arguments are required: mode` | 正常行为：必须带 `replay`/`live` 子命令，非环境问题 |
| 导出中断 | 强杀进程无保护（正常关窗的 `closeEvent` 会等录制/导出 settle 后才关），停在中间态 | `scripts/04` 判状态；raw 完好即 `MUJOCO_GL=egl` 离线恢复，幂等可重试；录制导出期间别杀进程 |
| venv 混用 | 三链路依赖互不重叠是刻意设计（IK venv 无 onnxruntime，sonic venv 有） | 各用各的 `.venv/bin/python`；校验脚本 04 明确用 g1_ik 的 venv 跑 |

## 6. 状态时间线

| 日期 | 事项 | 结果 |
|---|---|---|
| 2026-10-20 | g1_ik 环境搭建（create_env.sh + 10 包冒烟） | ✅ |
| 2026-10-20 | 六模块重构五级回归（build_model nq=76、IK 残差 0.0498/0.0503、100 帧 finite、GUI 起窗） | ✅ |
| 2026-10-20 | replay 经 scripts/01 实跑（session_20260828_161538） | ✅ GUI 起窗回放推进 |
| 2026-10-20 | GUI 录制实跑（g1_20261020_181011_a63011），导出中断 | ⚠️ 链路已验证；产物 2026-10-21 核实已清理 |
| 2026-10-21 | g1_sonic venv 11 包冒烟 + hf-mirror 权重下载 + onnx 加载 | ✅ 部署就位 |
| 2026-10-21 | g1_sonic replay 经 scripts/02 实跑（session_20260828_161538） | ✅ GUI 起窗、物理屏可见画面、策略驱动 G1 |
| 2026-10-21 | g1_sonic_bridge replay 实测（5557 端口，SUB 实收 300 帧） | ✅ 三链路 replay 全通 |
| 2026-10-21 | 三链路目录、主文件与 scripts/01-03 统一 g1_ 系命名 | ✅ 全链路引用同步（改动前文本备份于 tmp/backup-before-rename-20261021.tar.gz） |
| 2026-10-21 | g1_sonic（1646 行单文件 → `sonic_` 7 模块 + 65 行入口）与 g1_sonic_bridge（754 行单文件 → `bridge_` 6 模块 + 26 行入口，入口改名 launch_g1_sonic_bridge.py）按 g1_ik 模板模块化重构 | ✅ pyflakes/py_compile 15 文件、tests 2 套全绿、GUI 起窗 76 qpos、桥 SUB 实收 300 帧（改动前面文本备份于 tmp/backup-before-modularize-sonic-20261021.tar.gz） |
| 待办 | live 模式（需设备接入） | ⏳ |
