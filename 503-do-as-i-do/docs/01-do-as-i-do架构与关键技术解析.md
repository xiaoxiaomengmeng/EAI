# 01 · do-as-i-do 架构与关键技术解析

> 来源：malik-group/do-as-i-do（arXiv 2606.19333，"Do as I Do: Dexterous Manipulation Data from Everyday Human Videos"）官方 README 及 reconstruction / retargeting / deployment 三模块 README 通读笔记。2026-10 记录，未实跑验证。

## 一句话定位

从日常人类操作视频中提取灵巧操作数据的固定管线：单视频 → 手物重建与 6DoF 跟踪 → 重定向到机器人手（物理一致）→ 真机回放。与 502-DexAgent 同题（人类视频→机器人灵巧数据），但走确定性管线而非 agentic 路线。

## 仓库结构

```
do-as-i-do/
├── reconstruction/   # 管线 A：手物重建与跟踪
│   ├── run_pipeline.sh          # 驱动器
│   ├── config/paths.sh          # 唯一需要改的路径配置
│   ├── scripts/                 # 7 个一方阶段脚本
│   ├── modules/                 # 5 个带改动的 fork 子模块（sam3 / sam-3d-objects / Fast-SAM3D / HaWoR / tapnet）
│   ├── weights/                 # 模型权重
│   ├── env/                     # 各 conda env 文档（cu128 配方）
│   └── setup/                   # 00 子模块 · 01 环境 · 02 权重
├── retargeting/      # 管线 B：重定向（launch.py 单入口，pip 包）
└── deployment/       # 管线 C：真机回放（mujoco_replay + robot_replay）
```

外部代码全部以 git submodule 引入且改动已 baked in（fork 均 malik-group 名下、锁上游 commit）。

## 管线 A：reconstruction

**输入**：视频 + 参考帧号 + 物体名 + 锚定手。**输出**（落视频同目录）：物体 `.obj` mesh、`*_pointmap.npy` + 内参、HaWoR `all_hand_meshes.npz`、`gravity.json`（GeoCalib 相机系向上方向）、`obj_tracking_out/<object>/combined_visualization/` 下的 `layout.json → layout_camera_frame.json → layout_camera_frame_optimized.json`（管线 B 的输入）。

⚠️ 输出落视频同目录意味着：对官方 `whisking/whisking.mp4` 原地复跑会覆盖子模块内官方产物（masks、obj_tracking_out、all_hand_meshes.npz 等）。复现时先拷视频到 `503-do-as-i-do/data/` 再跑；官方 whisking 目录（含其全套中间产物）是复现对照的标准答案，保持只读。

| # | 阶段 | 脚本 | env |
|---|---|---|---|
| 0 | 抽帧 | ffmpeg | sam3 |
| 1 | SAM3 分割（物体点击 + 手文本） | `scripts/run_sam3_video.py` | sam3 |
| 2 | mask→3D mesh | `modules/sam-3d-objects/generate_mesh_sam3d.py` | sam3d |
| 2 | MoGe 点图（参考帧+全帧） | `scripts/get_pointmap_dir.py` | sam3d |
| 2 | 手重建 | `modules/HaWoR/demo.py` | hawor |
| 2 | 重力估计（GeoCalib） | `scripts/predict_video_gravity.py` | sam3d |
| 2.5 | 速度跟踪 | `scripts/tapir_velocity_tracking.py` | tapnet |
| 3 | 引导式位姿预测 | `modules/Fast-SAM3D/track_object.py` | sam3d |
| 3 | 投影 mesh / 相机系 | `run_project_mesh_combined.py`、`convert_layout_to_camera_frame.py` | sam3d |
| 4 | 平移/尺度优化 | `scripts/optimize_translation_scale.py` | sam3d |
| （可） | viser 3D 可视化 | `scripts/visualize_3d.py` | sam3d |

要点：
- 门槛三件套：GPU **≥32GB VRAM**（5090 压线过）、HF gated 权重（sam3 / sam-3d-objects）、**MANO 手模型注册下载**（HaWoR 用，mano.is.tue.mpg.de）；
- SAM3 分割是**点击式 GUI、需要 X display**（`SAM3_DISPLAY=:1`），headless 需 X 转发或试 text prompting；
- `sam3d` env 建好后两步手动收尾：`pip uninstall -y notebook`（仓库自带 `notebook/` 包被 pypi notebook 遮蔽）+ 编译 Mip-Splatting `diff_gaussian_rasterization`（渲染器 inria 后端）；
- 建议各 fork 自带 setup 分别建 env（`01_create_envs.sh` 仅兜底）；`env/README.md` 有 **cu128** 配方——与 RTX 5090/CUDA 12.8 对口；
- 克隆用 `GIT_LFS_SKIP_SMUDGE=1`，LFS 权重按需拉。

## 管线 B：retargeting

**输入**：管线 A 输出目录。**输出**（`outputs/{robot}/{hand}/{task}/{id}/`）：`scene.xml`（阶段 3–4.5）、`trajectory_kinematic.npz`（IK，阶段 4）、`trajectory_mjwp.npz` + `config.yaml`（MPC 优化 + **逐步跟踪误差指标**，阶段 5）。

| # | 阶段 | 模块 |
|---|---|---|
| 1 | 数据清洗 + 重力对齐 → 关键点轨迹 | `pipeline/process_dataset.py` |
| 2 | 物体凸分解（CoACD） | `pipeline/decompose_mesh.py` |
| 3 | MJCF 场景生成（机器人 + 物体 + UR3 腕圆柱） | `pipeline/generate_scene.py` |
| 4 | IK（mink） | `pipeline/solve_ik.py` |
| 4.5 | 底座求解（scene_ik.xml → scene.xml） | `pipeline/resolve_pedestal.py` |
| 5 | 物理优化（采样式 MPC，MuJoCo Warp） | `pipeline/optimize_physics.py` |

- 入口：`python launch.py --task whisking --raw-dir ../reconstruction/whisking`；`--robot-type sharpa`（默认）、`--no-show-viewer` headless、`--max-sim-steps N` 限定优化长度；
- 环境：`conda create -n doasido_retargeting python=3.12 && pip install -e .`（本地 env 名自定防撞名；模块是纯 pip 包入口、无 env 名硬编码——reconstruction 的 4 个 env 名则被 `config/paths.sh` 锁定）；MuJoCo Warp 在 GPU 上跑物理优化；
- **底座是 SPIDER（facebookresearch/spider）**——与 DexAgent Stage 3 兜底方法 `physics_informed_retargeting` 同源（其 yaml 里 `models.spider_dir` + `environments.spider`）；
- `replay_viser.py` 免重跑回放：三层叠加（MANO 参考·橙 / IK 幽灵·蓝透明 / 优化实体·solid），图层按文件存在自动启用；**whisking demo 全资产随仓库分发，开箱即播**（scene.xml + 两套轨迹 + MANO 关键点 + 物体/手 meshes，含 `outputs/assets/`），也是复跑对照的标准答案；
- 机器人资产：`retargeting/assets/robots/`（sharpa、mano）。

## 管线 C：deployment

两段：`mujoco_replay/`（sim→轨迹：加载 `trajectory_mjwp.npz`，放到 dual-UR3e 场景、mink 碰撞感知 IK，viser GUI 调 x/y/z+yaw/pitch/roll 与起始帧，`Recompute IK` → `Save retarget` 出 `trajectory_dual_ur3e.npz`；`--solve-only` 无头验证 IK 残差/碰撞间隙）→ `robot_replay/`（50Hz 流式到 UR3e + Sharpa Wave，`--dry-run` 先校验）。

- 工作空间调节口诀：只动 x/y/z 和 yaw；pitch/roll 仅当 reconstruction 的重力对齐明显错了才动；
- **Sharpa Wave SDK 私有不随仓库发布**：需自行放 `robot_replay/Sharpa/`，且 `config.example.yaml → config.yaml` 填机械臂 IP / 手序列号；
- 安全：核对 home 位、急停在手边、`--dry-run` → 低 `--speed` → 实时。

## 与本工作区的关联

| 关联点 | 说明 |
|---|---|
| 502-DexAgent | SPIDER 同源（两边物理优化底座）；同用 Sharpa Wave（资产与关节约定可对齐互喂）；固定管线 vs agentic 的对照实验组合 |
| 500-umr | 同为人类动作→机器人重定向家族（UMR 是全身动作，本管线是手物操作） |
| 501 | 手部姿态估计链路（HaWoR 环节）与遥操作感知栈重叠 |
| 104-unitreerobotics | MuJoCo Warp GPU 并行物理的使用范式与 microduck_rl/jumper 同源 |
