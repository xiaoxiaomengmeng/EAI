# Microduck 文档中心 · 总索引

> 本目录（`docs/`）是对 [`microduck/`](../microduck/)（Rust 运行时）与 [`microduck_rl/`](../microduck_rl/)（RL 训练）两个仓库的**学习与梳理笔记**——中文书写、关键术语保留英文，**非仓库自带的权威文档**。
>
> 组织方式：**按用途两类分治**——**端侧部署**（[`deploy/`](deploy/)）与**强化学习**（[`rl/`](rl/)）。**动手复现代码**独立在与 `docs/` 同级的 [`../practice/`](../practice/)；架构图 / 海报提示词在同级的 [`../diagrams/`](../diagrams/)。

---

## 目录结构

```text
301-microduck/
├── docs/
│   ├── README.md            ← 本文：两类总索引（唯一顶层入口）
│   │
│   ├── deploy/              ← 类①：端侧部署（Rust 运行时，机器人板载"大脑"）——只做总结文档
│   │   ├── README.md          整体架构 · IPC 契约 · 代码布局 · 设计不变量 · 路线图 · 构建测试
│   │   ├── modules.md         模块详解骨架（守护进程 / 库 / 工具，逐个索引）
│   │   └── modules/
│   │       ├── robotd.md      控制守护进程：50Hz 循环 / 电机总线 / 策略 / 安全层
│   │       └── updater.md     更新引擎：签名验证 / 原子切换 / 健康门 / 三层回滚网
│   │
│   └── rl/                  ← 类②：强化学习（GPU 仿真 + PPO 训练场）——只做总结文档
│       ├── README.md          ⭐ 唯一入口：认知阶梯导航（00→07）+ 路线图
│       ├── learning_roadmap.md  从入门到大师：约 12 个月周计划 + 每阶段硬检查点
│       ├── mdp_principles.md  MDP 纯原理笔记（+ 同名 poster png/svg）
│       └── 00-primer/ … 07-reference/   【总结文档】按学习顺序编号的认知阶梯
│
├── practice/                ← 与 docs 同级：【本地复现】动手侧（可运行代码 + 复现记录）
│   ├── README.md              复现导航
│   ├── fundamentals/          通用 RL 基础课（session_01 PyTorch/MLP · session_02 MDP/Q-learning/DQN）
│   └── tasks/ball_kick/       microduck 任务复现：踢球看能踢多高（高踢 reward 变体，不改子模块）
│
└── diagrams/                ← 与 docs 同级：架构图 / 海报提示词 + 生成物 + 素材
    ├── 00-prompts.md          microduck 软件架构图 + 硬件爆炸图提示词
    ├── 00-norm-prompts.md     归一化（Norm）专题海报系列提示词
    ├── 01-software-architecture.svg
    └── assets/                参考图 / 素材
```

---

## 两类导航

### 类① [`deploy/`](deploy/) — 端侧部署（Rust 运行时）

鸭子板载的一整套 Rust 守护进程软件栈：50 Hz 控制循环驱动 15 个舵机，多个 systemd 守护进程经 Unix socket 上的 JSON-RPC 2.0 通信。**本地无硬件，这一类只做总结文档**（读懂架构 / IPC 契约 / 模块职责），不追求复现。

| 想了解 | 去读 |
|---|---|
| 整体架构、IPC 契约、设计不变量、路线图 | [`deploy/README.md`](deploy/README.md) |
| 每个模块的职责 / 关键源文件 / 官方文档指引 | [`deploy/modules.md`](deploy/modules.md) |
| 控制守护进程 `robotd` 详解 | [`deploy/modules/robotd.md`](deploy/modules/robotd.md) |
| 更新引擎 `updater` 详解 | [`deploy/modules/updater.md`](deploy/modules/updater.md) |

### 类② [`rl/`](rl/) — 强化学习（训练场）

用 MuJoCo Warp + PPO + BAM 执行器模型训练运动策略，导出 ONNX 交给类①在真机执行。**这一类既要总结文档、又要本地复现**，两者物理分开：

| 你想做的事 | 去读 | 属哪半 |
|---|---|---|
| RL 初学者，想知道**几个月怎么安排** | [`rl/learning_roadmap.md`](rl/learning_roadmap.md) | 总结 |
| 系统理解"训练一只鸭子"的全流程 | [`rl/03-walkthrough/training_walkthrough.md`](rl/03-walkthrough/training_walkthrough.md) | 总结 |
| 架构总览 + 概念词典（PPO/BAM/61-D…） | [`rl/02-architecture/architecture_and_glossary.md`](rl/02-architecture/architecture_and_glossary.md) | 总结 |
| 深入某个源码模块（robot/tasks/export…） | [`rl/04-modules/`](rl/04-modules/README.md) | 总结 |
| 代码入口速查 | [`rl/07-reference/entry_points_map.md`](rl/07-reference/entry_points_map.md) | 总结 |
| **在本机把鸭子跑起来 / 复现训练** | [`rl/01-quickstart/desktop_quickstart.md`](rl/01-quickstart/desktop_quickstart.md) · [`sim_deploy_rl_reproduce_guide.md`](rl/01-quickstart/sim_deploy_rl_reproduce_guide.md) | 复现 |
| **动手练通用 RL 基础**（PyTorch/DQN） | [`../practice/fundamentals/`](../practice/fundamentals/) | 复现 |

> **从 [`rl/README.md`](rl/README.md) 进入**——它是该类的唯一入口，详解「总结文档（00→07 认知阶梯）」与「本地复现（项目顶层 [`../practice/`](../practice/)）」两半如何配合。

### 附：[`../diagrams/`](../diagrams/) — 图与提示词（与 docs 同级）

架构图 / 专题海报的提示词、风格规范与生成物。生成图存该目录，并在相关文档中引用。

---

## 两个仓库的关系（单向数据流）

类①（Rust 运行时）与类②（RL 训练）靠一个 **61 维观察 / 14 维动作**的契约对齐，是**单向数据流**：

```text
microduck_rl (类②, Python/GPU)              microduck (类①, Rust/RK3566)
MuJoCo Warp 仿真 + PPO 训练                    robotd 50Hz 控制循环
  → checkpoint (model_*.pt)                    → 加载 policy.onnx
  → export ONNX (烘焙 obs normalizer)  ───►     → obs[1,61] → policy → action[1,14]
  → publish 到 HF Hub (schema-2 manifest)       → safety 层 clamp → 15 舵机
```

---

## 维护约定

- **两类分治**：新增端侧部署笔记进 [`deploy/`](deploy/)；RL 笔记进 [`rl/`](rl/)——理解型文档按 00→07 编号对号入座，动手复现进**项目顶层** [`../practice/`](../practice/)（与 docs 同级，不放 docs 内）。图与提示词进**与 docs 同级**的 [`../diagrams/`](../diagrams/)。顶层只放本索引。
- **相对路径**（本文 = `docs/README.md`）：指向 Rust 仓库 `../microduck/…`、RL 仓库 `../microduck_rl/…`；指向类内文档 `deploy/…`、`rl/…`；指向图目录 `../diagrams/…`。子块内文档的路径约定见各自 README。
- **每个主题只有一个权威文档**，其余链接过去，不复制。
- **改完验证死链**：移动/新增文档后跑一次相对链接扫描，确保 `dead=0`。

> 官方一手资料：[`microduck/docs/`](../microduck/docs/)（`design/` 讲机制、`project/` 讲进度、`robot/` 讲使用）；[`microduck_rl/README.md`](../microduck_rl/README.md) 与 [`microduck_rl/AGENTS.md`](../microduck_rl/AGENTS.md)。
