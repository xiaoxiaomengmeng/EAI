# 01 · DexAgent 架构与关键技术解析

> 来源：dexagent1/DexAgent（arXiv 2609.35318，Wang Youhui / Yunzhu Li / Li Fei-Fei / Jiajun Wu / Huang Huang）官方 README 及主入口 `run_dexagent.py` 通读笔记。2026-10 记录，未实跑验证。

## 一句话定位

输入固定相机的人类第一人称操作视频 + 任务 prompt，经 LLM agent 驱动的四阶段「生成-验证」管线，输出物理验证的机器人关节轨迹与随机化照片级多视角训练视频——「人类演示 → 机器人策略训练数据」自动化合成，面向灵巧手操作（rigid / articulated / deformable）。

## 核心思想

1. **Agent 干活**：LLM 经 function calling 调工具库的 skill（产出资产/位姿/动作）与 verifier（判决），循环「生成 → 验证 → 反馈修正」直到通过或预算耗尽；
2. **工具库自进化**：任务缺工具时 agent 现场写新工具（Python 文件 + 测试），≥3 个变体过测试才入库、失败即移除——跑得越多库越强；
3. **每步有物理/几何验证**：verifier 重跑物理仿真、算 IoU、做穿透检查，LLM 说了不算。

## 四阶段数据流

| 阶段 | 输入 | 输出 | agentic | 算力 |
|---|---|---|---|---|
| 1 语义理解 | 视频 + prompt | `scene_semantics.json`：物体（类型 rigid/articulated/deformable、属性、初始/终态）、有序子目标（acting_hand、motion 词表、双手角色、可观测 goal） | 是 | CPU + 可选 GPU 感知 |
| 2 属性化场景重建 | 视频 + Stage 1 | `scene.xml`（每物体 MJCF+mesh+参数，按视频位姿放置）+ `frame0_compare.png` + IoU 报告；可变形走 Isaac Sim 5.0 | 是 | GPU（SAM 3 分割） |
| 3 轨迹生成 | 子目标 + 场景 | 物理验证 `trajectory.npz`（qpos[T,nq]+fps+labels）、`sim_video.mp4`、打包 `stage4_task/` | 是 | CPU（MuJoCo） |
| 4 数据生成 | `stage4_task/` | 随机化多视角渲染视频 + 轨迹 | 否（固定五步） | GPU ≥40GB（plate 步） |

关键设计：**Stage 1 全自然语言、零数字**——位姿尺寸等几何量全部后置到 Stage 2 由感知工具产生，语义与测量解耦。

## Agent Harness（`agent/`）

- **循环**（`runner.py`）：发 stage prompt + 工具清单 → 执行工具调用（结果含图像）→ `finish` 时跑该 stage verifiers → 判决回喂模型，不通过继续修；
- **内置工具**（`devtools.py`）：`look`（看帧/contact sheet——模型只经图像看视频）、`read/write_text`、`list_dir`、`run_python`、`library_search/read/create`、`finish`、`report_blocked`；
- **跨阶段修复**：`report_blocked(upstream='stage1'|'stage2')` 重开上游、下游重跑，`repair.max_loops_per_stage`（默认 1）限次；
- **三重预算**：每 stage `max_rounds` + 全 run `budget_usd`（必填，90% 提醒、花完即停、`budget.json` 记账）+ `max_wall_minutes`；
- **可观测**：`trace.jsonl` 全量记录模型轮次/工具调用/verifier 判决；失败留 `failure_report.json`；
- **LLM 侧**：Anthropic Messages API 或任意 OpenAI 兼容端点，须支持图像 + tool use；推荐 Claude Opus 5 / GPT-5.6 档；另有 `provider: external`（让 Claude Code/Codex fork subagent 经 `toolcall.py` CLI 手工执行，不烧 API）与 scripted mock 干跑。

## 工具库（`tool_library/`）

- skills 产出、verifiers 判决（返回带 `ok` 的 report）；一文件一工具，文件头 `TOOL = {...}` 纯字面量元数据（name/type/category/inputs/outputs/entrypoint/status），`build_index.py` 不 import 直接解析生成 `index.json` → [library visualizer](https://dexagent1.github.io/library/)；
- `library_create` 唯一入库通道，强制：永不改现有工具（fork 新名）、测试 ≥3 变体通过（verifier 须含应过/应挂两用例）、禁硬编码路径/帧号/物体名；
- 手动使用三途径：Python 函数、`python -m agent.toolcall <tool> '<json>'`、工具自带 CLI；
- **多解释器路由**：神经模型工具在函数内 import、缺包自报换路；`environments:`（env 名→解释器路径）+ `tool_environments:`（工具名→env 名）把 SAM3/SAM2/HaMeR/SAM3D/深度/Isaac/BODex/DexGraspNet/SPIDER 隔离在各自环境，主 harness 保持 MuJoCo 3.10 + NumPy 1.x 轻量锁定。

## Stage 4 五步（固定流水线）

| 步 | 内容 | 资源 |
|---|---|---|
| randomize | 抽新初始状态→轨迹适配→运动学检查+物理回放（按 Stage 3 manifest 的 subgoal expectation 逐段复检） | CPU |
| plate | SAM 3 分割 + VOID（5B 视频修复）抹掉人手与被操作物体，得干净背景板 | GPU ≥40GB（峰值 38GB） |
| export | 每 episode × 每视角写 Blender 场景 | CPU |
| render | Blender Cycles 渲染 ego + 双腕相机 | GPU ~8GB |
| composite | 渲染贴回背景板编码视频 | CPU |

输出每 episode 带 `trajectory.npz`（qpos/joint_names/labels/fps）+ ego/wrist 视频——离 lerobot 数据集只差一层转换。渲染帧级缓存。

## 默认 setup 与可迁移性

- 平台：I2RT YAM Box + 双 6-DOF YAM 臂 + 双 Sharpa Wave 手（22 指关节）；ego 相机 640×480 / fovy 38°（焦距 697px）、桌面 z=0、物理 timestep 2ms `implicitfast`；
- **手根 body 放世界原点，6 腕关节（3 slide + 3 hinge，intrinsic XYZ）即手的世界位姿**——单手场景解出的位姿原样迁移整机；
- 臂-手 position-only weld，方向由腕关节保持；
- 换机器人只改 `setup/`（scene.xml + setup.yaml，key 不变改值）；相机未知时 `camera.source: estimate` 由桌面平面反估。

## 其他工程要点

- **复用**（`examples/`）：同视频复跑经复验后直接复用，Stage 3 轨迹按资产位置差平移 + IK 重解、不重执行物理搜索；`save_assets: true` 可存全通过 run 为可复用资产；
- **Stage 3 停滞检测**：连续 10 次执行无改进提示换策略、25 次终止；轮次预算按子目标数缩放（`rounds_per_subgoal`，上限 `max_rounds_cap` 300）；
- **Stage 3 兜底方法链**：agent 搜索未完成时依次跑 physics_informed_retargeting（SPIDER 优化器）/ sampling_mpc_refinement / reinforcement_learning_tracking，首个过验证者发布；
- **阶段间确认**：每 stage 结束打印产物位置并询问是否继续，非 TTY 自动停（`--yes` 跳过）；`--from/--to` 断点续跑。

## 与本工作区的关联

| 关联点 | 说明 |
|---|---|
| 数据管线 | 产出即 VLA 训练数据形态，可转 lerobot 后接 `101-pi-serial-models` 微调 π0/π0.5 |
| 感知串联 | SAM3 → Video Depth Anything → HaMeR → GeoCalib，与 501 的 SAM→深度→光流→姿态链路同源 |
| 约定对齐 | MuJoCo 版本锁定理由（接触仿真不同）与 100-openVLA 的 LIBERO 经验同频 |
| 对照路线 | `503-do-as-i-do`：固定管线（SAM3→SAM3D→MoGe→HaWoR→TAPIR→diffusion tracking；MuJoCo Warp MPC；UR3e+Sharpa Wave 真机），共享感知 env 与 Sharpa Wave 资产，结合方案待 503 启动后另立文档 |
