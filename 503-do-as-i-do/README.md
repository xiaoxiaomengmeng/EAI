# 503-do-as-i-do

Do as I Do（arXiv 2606.19333，malik-group/do-as-i-do）本地复现：**Do as I Do: Dexterous Manipulation Data from Everyday Human Videos**。固定三段式管线（无 LLM、零 API 成本）：单视频手物重建与 6DoF 跟踪 → 重定向到 Sharpa Wave 机器人手（MuJoCo Warp 采样式 MPC）→ dual UR3e 真机回放。

快速上手见 [START_HERE.md](START_HERE.md)，架构与关键技术解析见 [docs/01-do-as-i-do架构与关键技术解析.md](docs/01-do-as-i-do架构与关键技术解析.md)。

## 三段管线

```
人类手物视频
  └─ reconstruction ──→ 物体 mesh + 逐帧 6DoF 位姿 + MANO 手轨迹 + 重力方向
       └─ retargeting ──→ scene.xml + IK 轨迹 + MuJoCo Warp MPC 物理优化轨迹
            └─ deployment ──→ dual UR3e 关节轨迹 → 50Hz 真机流式
```

| 模块 | 内容 | 关键技术 | 环境 |
|---|---|---|---|
| `reconstruction/` | 视频分割 → 物体 mesh → 逐帧位姿 → 手重建 → 平移/尺度优化 | SAM3 → SAM3D → MoGe pointmaps → HaWoR → GeoCalib → TAPIR 速度跟踪 → Fast-SAM3D 引导式位姿预测 | sam3 / sam3d / hawor / tapnet |
| `retargeting/` | 轨迹清洗 + 重力对齐 → CoACD 凸分解 → MJCF 场景 → mink IK → 物理优化 | 采样式 MPC on **MuJoCo Warp**（GPU 并行物理，基于 SPIDER） | doasido_retargeting（py3.12） |
| `deployment/` | sim 轨迹 → dual UR3e 碰撞感知 IK → 真机回放 | mink IK + viser GUI 调工作空间位姿 + 50Hz 流式 | deployment |

## 核心机制

- **reconstruction**（`run_pipeline.sh VIDEO [FRAME_N] [OBJECT] [ANCHOR_HAND]` 驱动 10 步）：分割（物体点击 + 手文本提示）→ SAM3D 生成物体 mesh → MoGe 点图（参考帧+全帧）→ HaWoR 手重建 → GeoCalib 重力 → TAPIR 速度跟踪 → 引导扩散式位姿预测 → 相机系转换 → 平移/尺度优化。产物：`layout.json → layout_camera_frame.json → layout_camera_frame_optimized.json`（retargeting 的输入）。
- **retargeting**（`launch.py` 单入口 5 阶段）：数据清洗对齐 → 凸分解 → MJCF 场景生成（含 UR3 腕圆柱）→ IK → 底座求解 → MPC 物理优化。产物 `trajectory_mjwp.npz` 内含**逐步跟踪误差指标**；`replay_viser.py` 三层叠加回放（MANO 参考 / IK 幽灵 / 优化实体），自带 whisking demo 开箱即播。
- **deployment**：`mujoco_replay`（viser GUI 调 x/y/z + yaw，避碰 IK，`--solve-only` 可无头验证残差）→ `robot_replay`（`--dry-run` 校验 → 低速 → 实时，急停在手边）。

## 与 502-DexAgent 的对照

| 维度 | 503-do-as-i-do | 502-DexAgent |
|---|---|---|
| 架构 | 确定性固定管线，零 API | LLM agent + 自进化工具库 |
| 重建 | SAM3D mesh + TAPIR + 引导扩散（显式 3D） | Stage 2 参数化 MJCF 重建 + IoU 验证 |
| 物理优化 | 采样式 MPC（MuJoCo Warp，基于 SPIDER） | agent 搜索 + SPIDER/MPC/RL 三级兜底链 |
| 部署 | dual UR3e + Sharpa Wave 真机回放 | 随机化渲染训练数据（不落真机） |
| 交集 | retargeting 底座即 SPIDER；同为 Sharpa Wave 平台 | |

## 环境矩阵（6 套）

| 环境 | 用途 |
|---|---|
| `sam3` / `sam3d` / `hawor` / `tapnet` | reconstruction 各感知段（cu128 配方见 `reconstruction/env/README.md`） |
| `doasido_retargeting`（py3.12，`pip install -e .`；env 名自定防撞名，模块无 env 名硬编码） | 5 阶段重定向 + viser 回放 |
| `deployment` | dual UR3e IK 回放 + 真机流式（本机仅 sim 侧） |

权重与门槛：HF gated（sam3 / sam-3d-objects）+ MANO 注册下载 + LFS 按需拉取，详见 START_HERE「已知坑」。
