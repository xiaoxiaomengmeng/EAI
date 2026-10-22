# 504-retargeting

具身数据 infra（5xx 段）下的重定向与遥操系列：**穿戴设备路线**。收 PICO 头显 + 数据手套动捕数据到人形机器人重定向、遥操验证及 LeRobot 数据集产出的各项目。

## 定位与边界

- 数据源固定为穿戴设备（PICO 动捕 `vr` channel + 手套 `glove` channel），区别于：
  - [500-umr](../500-umr/)：离线动作库（LAFAN1 等）重定向
  - [501-human-dexterous-teleop](../501-human-dexterous-teleop/)：视觉（MediaPipe）人手遥操作
  - [503-do-as-i-do](../503-do-as-i-do/)：操作视频重建后的手物交互重定向
- 验证方式为本机 MuJoCo 仿真回放 / live 转发，不直接控制真机；产出为重定向结果验证与 LeRobot v2.1 训练数据

## 子项目

### pico2g1

PICO 动捕（24 关节 body tracking）+ 手套（左右各 21 关键点）→ Unitree G1 全身（29 DoF，`assets/g1_description/`）+ 手部（Wuji Hand 40 关节 / Dex3）重定向。仓库内三个互相独立的方法：

| 方法 | 目录 | 路线 | 关键依赖 |
|---|---|---|---|
| 轻量 IK | `g1_ik/` | MuJoCo 雅可比 + `lsq_linear` 带限位最小二乘，界面实时调参 | MuJoCo · PySide6 · wuji-retargeting |
| SONIC（Mac） | `g1_sonic/` | SONIC v1.1 ONNX 策略（CPU）驱动 G1 + wuji 手部，PySide6 双画面 | MuJoCo · PySide6 · onnxruntime |
| SONIC 数据桥（Linux） | `g1_sonic_bridge/` | 50 Hz 重采样 + 5 帧滑窗，ZMQ PUB 转发给下游 SONIC 进程 | pyzmq · dex-retargeting · torch(CPU) |

**外部资产**：G1 模型网格、人体骨架、Dex3 URDF 随仓库入库（`pico2g1/assets/`）；wuji-retargeting 由各工程环境脚本 clone 到 `3rd/`（不入库）；SONIC v1.1 onnx（~191 MB）由 `setup.sh` 从 `nvidia/GEAR-SONIC` 下载（不入库）。路径可用 `G1_DESCRIPTION` / `WUJI_ROOT` 覆盖，默认全在仓库内。

## 文档路由

| 内容 | 位置 |
|---|---|
| 入口、运行速查、状态 | [START_HERE.md](START_HERE.md) |
| pico2g1 完整使用说明 | [pico2g1/README.md](pico2g1/README.md) |
| LeRobot v2.1 录制 | [pico2g1/docs/lerobot_recording.md](pico2g1/docs/lerobot_recording.md) |
| 八路实时推流 | [pico2g1/docs/live_streaming.md](pico2g1/docs/live_streaming.md) |
| 互动场景（PULSE / 游戏场） | [pico2g1/docs/pulse_scene.md](pico2g1/docs/pulse_scene.md) · [play_worlds.md](pico2g1/docs/play_worlds.md) |
