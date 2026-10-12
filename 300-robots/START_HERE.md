# 300-robots

一句话：机器人整机与运动框架复现合集——主体为 RynnMotion（快速原型验证、原语运动控制、异构遥操作、数据采集、多平台部署框架），另有 sim_robot（π0.5 → UR5e MuJoCo 仿真桥脚本集）。

## 内容

- `RynnMotion/`：gitlink 子模块；`robots/` 下 10 个厂商硬件模块（Franka FR3、非夕 Flexiv Rizon 4s 等），CMake 构建，附 MuJoCo 仿真与 Python 绑定
- `sim_robot/`：本地脚本——π0.5 UR5e 仿真部署桥（`pi05_ur5e_bridge.py`）、相机渲染、模型查看与验证；中文指南见其中《PI0-UR5E仿真部署指南.md》
- `RynnMotion.zip`：上游压缩包存档（空间清理候选，见重组计划第 6 步）

## 从这里开始

- 目录总览与硬件模块表：[README.md](README.md)
- RynnMotion 上游说明：[RynnMotion/README.md](RynnMotion/README.md)（框架定位、各机器人模块、部署流程）
- 仿真玩法：`sim_robot/` 下各脚本与部署指南
