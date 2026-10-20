# 502-DexAgent

DexAgent（arXiv 2609.35318，dexagent1/DexAgent）本地复现：**An Agentic Human2Sim2Robot Framework for Dexterous Manipulation with Self-Evolving Tool Library**。输入一段固定相机的人类操作视频 + 任务描述，输出物理验证过的机器人灵巧手关节轨迹与随机化、照片级真实的多视角训练视频——「人类演示 → 机器人策略训练数据」的自动化合成管线，覆盖 rigid / articulated / deformable 物体。

快速上手见 [START_HERE.md](START_HERE.md)，架构与关键技术解析见 [docs/01-DexAgent架构与关键技术解析.md](docs/01-DexAgent架构与关键技术解析.md)。

## 四阶段数据流

```
人类视频 + prompt
  └─ Stage 1 语义理解 ──→ scene_semantics.json（纯自然语言：物体/属性/有序子目标/双手角色）
       └─ Stage 2 属性化场景重建 ──→ scene.xml（MuJoCo MJCF + mesh，按视频位姿放置，相机对齐）
            └─ Stage 3 机器人轨迹生成 ──→ trajectory.npz + sim_video.mp4（物理验证 episode）
                 └─ Stage 4 数据生成 ──→ 随机化 + 重纹理 + Blender 渲染多视角视频（喂策略训练）
```

| 阶段 | 输入 | 输出 | agentic | 算力 |
|---|---|---|---|---|
| 1 语义理解 | 视频 + prompt | `scene_semantics.json` | 是 | CPU + 可选 GPU 感知 |
| 2 场景重建 | 视频 + Stage 1 | `scene.xml` + 资产 + IoU 报告（可变形走 Isaac Sim 5.0） | 是 | GPU（SAM 3） |
| 3 轨迹生成 | 子目标 + 场景 | 物理验证 `trajectory.npz`、打包 `stage4_task/` | 是 | CPU（MuJoCo） |
| 4 数据生成 | `stage4_task/` | 随机化多视角渲染视频 + 轨迹 | 否（固定五步） | GPU ≥40GB（plate） |

## 核心机制

- **生成-验证循环**（`agent/runner.py`）：LLM 工具调用 → 结果（含图像）回喂 → `finish` 时跑 verifiers → 判决不通过继续修，直到通过或预算（`max_rounds` / `budget_usd` / 墙钟）耗尽。
- **自进化工具库**（`tool_library/`）：一文件一工具（skill 产出 / verifier 判决），`TOOL` 元数据纯字面量；`library_create` 是唯一入库通道，新工具须在 ≥3 个变体上过测试否则回滚；永不改现有工具，fork 新名。
- **跨阶段修复**：`report_blocked(upstream='stage1'|'stage2')` 可重开上游阶段，`repair.max_loops_per_stage` 限次。
- **复用机制**（`examples/`）：同一视频复跑复用已验证结果；全通过的 run 可存为可复用资产。
- **Stage 3 兜底方法链**：agent 搜索失败后依次试 physics_informed_retargeting（SPIDER）→ sampling_mpc_refinement → reinforcement_learning_tracking。

## 默认硬件 setup

I2RT YAM Box + 双 6-DOF YAM 臂 + 双 Sharpa Wave 灵巧手（22 指关节），固定 ego 相机 640×480 / fovy 38°。换机器人只改 `setup/`（scene.xml + setup.yaml），所有工具以 body/joint 名为参数。手根 body 放世界原点、6 腕关节即世界位姿的设计使单手场景解出的位姿可原样迁移到整机。

## 环境矩阵

| 环境 | 用途 |
|---|---|
| conda `dexagent`（py3.11，MuJoCo 3.10 + NumPy 1.x） | 主 harness：Stage 1/3 + CPU 工具 |
| 独立 env（`environments:` 路由） | SAM3 / SAM2 / HaMeR / SAM3D / 深度 / Isaac Sim / BODex / DexGraspNet / SPIDER |
| Stage 4 env（py3.11 + bpy + VOID，`setup.sh`） | randomize / plate / export / render / composite |

## 与工作区的衔接

- 产出的 `trajectory.npz` + 多视角视频是标准 VLA 训练数据形态，可接 `101-pi-serial-models` 的 lerobot 管线微调 π0/π0.5。
- 感知串联（SAM→深度→手部）与 `501-human-dexterous-teleop` 的感知链路同源。
- 对照路线：`503-do-as-i-do`（固定管线 + MuJoCo Warp MPC + UR3e/Sharpa Wave 真机），见 START_HERE「关联」。
