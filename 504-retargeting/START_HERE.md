# 504-retargeting

> **状态**：🟡 三链路 replay 全通（2026-10-20/21 实测：g1_ik 与 g1_sonic GUI 起窗、g1_sonic_bridge SUB 实收 300 帧；录制链路已实跑可复录；三链路代码均已按 g1_ik 模板模块化：g1_ik 6 模块、g1_sonic 7 模块、g1_sonic_bridge 6 模块 + 瘦入口，tests 全绿）；live 模式待设备

穿戴设备（PICO 头显 + 数据手套）动捕数据 → 人形机器人重定向与遥操作验证系列。后续 ego 系重定向 / 遥操项目均以独立子目录放本目录下。

## 目录结构

| 子目录 | 说明 |
|---|---|
| [`pico2g1/`](pico2g1/) | PICO 动捕 + 手套 → Unitree G1 全身（29 DoF）+ Wuji Hand / Dex3 手部重定向；MuJoCo 离线回放与 live 验证、LeRobot v2.1 录制、八路实时推流 |
| [`docs/`](docs/) | 系列级专题文档（按 `NN-主题.md` 命名）：[01-pico2g1 部署复现实测记录](docs/01-pico2g1部署复现实测记录.md)、[02-g1_ik 架构与关键技术解析](docs/02-g1_ik架构与关键技术解析.md)、[03-g1_sonic 架构与关键技术解析](docs/03-g1_sonic架构与关键技术解析.md)、[04-g1_sonic_bridge 架构与关键技术解析](docs/04-g1_sonic_bridge架构与关键技术解析.md) |
| `tmp/` | 原始交付 zip 等临时文件（不入库） |

## 怎么跑（pico2g1）

三个互相独立的方法：代码互不 import、依赖互不重叠、各自环境脚本建独立 `.venv`。详细命令、参数、界面说明见 [pico2g1/README.md](pico2g1/README.md)。

```bash
# 1) 轻量 IK：不跑策略网络，MuJoCo 雅可比 + 带限位最小二乘直接解 G1 29 DoF，界面滑杆实时调参
cd pico2g1/g1_ik && bash create_env.sh
.venv/bin/python launch_g1_ik.py replay ../session_20260828_161538/data/chunk-000/observation.data.vr_full_body_000000.parquet
.venv/bin/python launch_g1_ik.py live <设备IP>

# 2) SONIC（Mac 侧）：SONIC v1.1 ONNX 策略驱动 G1，wuji-retargeting 解手部，PySide6 双画面
cd pico2g1/g1_sonic && bash setup.sh        # 额外从 HF 下载 ~191 MB SONIC onnx
.venv/bin/python launch_g1_sonic.py replay ../session_20260828_161538

# 3) SONIC 数据桥（Linux 侧）：重采样 50 Hz、5 帧滑窗打包，ZMQ PUB tcp://*:5556（topic: pose），无界面
cd pico2g1/g1_sonic_bridge && bash setup.sh
.venv/bin/python launch_g1_sonic_bridge.py replay ../session_20260828_161538/data/chunk-000/observation.data.vr_full_body_000000.parquet
```

`session_*` 录制目录不入库、随用随备（本机 `data/` 下已有 3 个 session，2026-08-28 采集），无数据时走 `live <设备IP>` 模式；排查数据可用 `pico2g1/tools/inspect_*.py`（在线抓流 / 看 parquet schema）。

上述命令已包装成 `scripts/` 下四个启动脚本（推荐入口：自动定位 `data/` 下 session、校验 parquet 齐全性与 `.venv`/onnx 就位、远程会话自动探测 DISPLAY）：

```bash
scripts/00_list_sessions.sh                                     # 概览：任务/时长/采集时间/四类 parquet 齐全性
scripts/01_replay_g1_ik.sh session_20260828_161538          # 轻量 IK 回放（venv 已建，即跑）
scripts/02_replay_g1_sonic.sh session_20260828_161538  # SONIC 策略回放（先 bash setup.sh）
scripts/03_replay_g1_sonic_bridge.sh session_20260828_161538   # ZMQ 数据桥（venv 已建；5556 被钉钉占用时前置 SONIC_ENDPOINT=tcp://*:5557）
scripts/04_validate_recording.py <录制名>                        # 校验录制产物是否有效 LeRobot v2.1（可诊断导出中断，--deep 走 lerobot 库）
# 皆支持 DRY_RUN=1 干跑看完整命令；session 后可透传 --scene pulse --render-shadows 等
```

> 坑：PySide6 6.5+ 的 xcb 插件需系统库 `libxcb-cursor0`（apt 补装，一次管两条 GUI 链路）；IDE/SSH 远程会话无 `DISPLAY`，01/02 脚本自动探测物理桌面（`:1`）——窗口出在物理机屏幕上，远程看不到，看画面需 AnyDesk / 坐到机器前。另：`python -m sim_scenes.lerobot_recording` 离线恢复导出必须 `MUJOCO_GL=egl`（无 DISPLAY 时 `mujoco.Renderer` 需 EGL 上下文，录制时的导出走 GUI 进程内 GL 故不受影响）。另：ZMQ 桥默认端口 5556 常被钉钉占用（bind 报 Address already in use），前置 `SONIC_ENDPOINT=tcp://*:5557` 换端口（详见 docs/01 §4）。

## 输入 / 输出 / 控制方式

- **输入**（二选一）：live 模式订阅设备端 WebSocket `ws://<设备IP>:9990/subscribe?token=rynn_dev`（需 `vr` + `glove` 两个 channel）；replay 模式读 `session_*/` 目录或其中的 `observation.data.vr_full_body_*.parquet`
- **输出**：MuJoCo 可视化（主画面 + 头部/左右腕三相机同步视角）；数据桥输出 ZMQ pose 流；录制输出 LeRobot v2.1 数据集（三路 1280×720 / 30 fps / H.264 + 69 维关节实际/目标 + 浮动基座位姿）；八路 MJPEG 推流（默认端口 8876）
- **控制**：PySide6 界面滑杆实时调参（写 `g1_ik/tuning.json`，本地不入库）；录制面板（任务描述 / 保存位置 / 开始停止）；`--scene empty|xixi|pulse|strike|orbit|switchyard` 互动场景，可加 `--render-shadows`
- 三条链路均只做本机 MuJoCo / 数据转换验证，**不直接控制真机**

## 复现对齐红线（改代码前先核对）

- 跨流对齐必须按时间戳，不能按帧号（手套 ~90.9 Hz、VR 手部 ~76.6 Hz、视频 30 Hz）
- 手套 parquet 的 `wrist` 列是局部坐标占位，真实腕部朝向在 `imu` 列
- 回放需配套 `vr_full_body` / `vr` / `glove_left` / `glove_right` 四类 parquet

## 与相邻项目的边界

| 目录 | 分工 |
|---|---|
| 本目录（504） | 穿戴设备路线：全身 + 手部重定向、遥操与录制验证 |
| [`500-umr`](../500-umr/) | 离线动作库重定向（LAFAN1 → G1 qpos），无设备 |
| [`501-human-dexterous-teleop`](../501-human-dexterous-teleop/) | 视觉路线人手遥操作（MediaPipe / ROS2） |
| [`503-do-as-i-do`](../503-do-as-i-do/) | 视频路线手物交互重建 + 灵巧手重定向回放 |

详细说明见 [README.md](README.md)；专题深读见 [docs/01-pico2g1 部署复现实测记录](docs/01-pico2g1部署复现实测记录.md)（环境搭建命令、验证阶梯与实测输出、坑汇总）与解析系列：[02-g1_ik](docs/02-g1_ik架构与关键技术解析.md)、[03-g1_sonic](docs/03-g1_sonic架构与关键技术解析.md)、[04-g1_sonic_bridge](docs/04-g1_sonic_bridge架构与关键技术解析.md)（三条链路的架构与关键技术原理）。
