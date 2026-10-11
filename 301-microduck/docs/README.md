# Microduck 文档中心 · 总索引

> 本目录（`docs/`）是对 [`microduck/`](../microduck/)（Rust 运行时）与 [`microduck_rl/`](../microduck_rl/)（RL 训练）两个仓库的**学习与梳理笔记**——中文书写、关键术语保留英文，**非仓库自带的权威文档**。
>
> 组织方式：**三块分治**。顶层只保留本索引，其余内容按仓库/用途归入三个子块。

---

## 目录结构

```text
docs/
├── README.md                ← 本文：三块总索引（唯一顶层入口）
│
├── microduck/               ← 块①：Rust 运行时（机器人板载"大脑"）
│   ├── README.md              整体架构 · IPC 契约 · 代码布局 · 设计不变量 · 路线图 · 构建测试
│   ├── modules.md             模块详解骨架（守护进程 / 库 / 工具，逐个索引）
│   └── modules/
│       ├── robotd.md          控制守护进程：50Hz 循环 / 电机总线 / 策略 / 安全层
│       └── updater.md         更新引擎：签名验证 / 原子切换 / 健康门 / 三层回滚网
│
├── microduck_rl/            ← 块②：RL 训练场（GPU 仿真 + PPO，00→07 编号认知阶梯）
│   ├── README.md              ⭐ 唯一入口：导航 + 认知阶梯 + 学习路线图
│   ├── learning_roadmap.md    从入门到大师：约 12 个月周计划 + 每阶段硬检查点
│   ├── 00-primer/ … 07-reference/   按学习顺序编号的文档树
│
└── diagrams/                ← 块③：架构图 / 海报提示词 + 生成物 + 素材
    ├── 00-prompts.md          microduck 软件架构图 + 硬件爆炸图提示词
    ├── 00-norm-prompts.md     归一化（Norm）专题海报系列提示词
    ├── 01-software-architecture.svg
    └── assets/                参考图 / 素材
```

---

## 三块导航

### 块① [`microduck/`](microduck/README.md) — Rust 运行时

鸭子板载的一整套 Rust 守护进程软件栈：50 Hz 控制循环驱动 15 个舵机，多个 systemd 守护进程经 Unix socket 上的 JSON-RPC 2.0 通信。

| 想了解 | 去读 |
|---|---|
| 整体架构、IPC 契约、设计不变量、路线图 | [`microduck/README.md`](microduck/README.md) |
| 每个模块的职责 / 关键源文件 / 官方文档指引 | [`microduck/modules.md`](microduck/modules.md) |
| 控制守护进程 `robotd` 详解 | [`microduck/modules/robotd.md`](microduck/modules/robotd.md) |
| 更新引擎 `updater` 详解 | [`microduck/modules/updater.md`](microduck/modules/updater.md) |

### 块② [`microduck_rl/`](microduck_rl/README.md) — RL 训练场

用 MuJoCo Warp + PPO + BAM 执行器模型训练运动策略，导出 ONNX 交给块①在真机执行。**从 [`microduck_rl/README.md`](microduck_rl/README.md) 进入**（它是该块的唯一入口 + 认知阶梯导航）。

| 你想做的事 | 去读 |
|---|---|
| RL 初学者，想知道**几个月怎么安排** | [`microduck_rl/learning_roadmap.md`](microduck_rl/learning_roadmap.md) |
| 尽快把鸭子在仿真里跑起来 | [`01-quickstart/desktop_quickstart.md`](microduck_rl/01-quickstart/desktop_quickstart.md) |
| 无真机时的仿真部署 + RL 训练复现（5090 / py3.12） | [`01-quickstart/sim_deploy_rl_reproduce_guide.md`](microduck_rl/01-quickstart/sim_deploy_rl_reproduce_guide.md) |
| 系统理解"训练一只鸭子"的全流程 | [`03-walkthrough/training_walkthrough.md`](microduck_rl/03-walkthrough/training_walkthrough.md) |
| 架构总览 + 概念词典（PPO/BAM/61-D…） | [`02-architecture/architecture_and_glossary.md`](microduck_rl/02-architecture/architecture_and_glossary.md) |
| 代码入口速查 | [`07-reference/entry_points_map.md`](microduck_rl/07-reference/entry_points_map.md) |

### 块③ [`diagrams/`](diagrams/) — 图与提示词

架构图 / 专题海报的提示词、风格规范与生成物。生成图存本目录，并在相关文档中引用。

---

## 两个仓库的关系（单向数据流）

块①（Rust 运行时）与块②（RL 训练）靠一个 **61 维观察 / 14 维动作**的契约对齐，是**单向数据流**：

```text
microduck_rl (块②, Python/GPU)              microduck (块①, Rust/RK3566)
MuJoCo Warp 仿真 + PPO 训练                    robotd 50Hz 控制循环
  → checkpoint (model_*.pt)                    → 加载 policy.onnx
  → export ONNX (烘焙 obs normalizer)  ───►     → obs[1,61] → policy → action[1,14]
  → publish 到 HF Hub (schema-2 manifest)       → safety 层 clamp → 15 舵机
```

---

## 维护约定

- **三块分治**：新增 Rust 运行时笔记进 [`microduck/`](microduck/README.md)；RL 训练笔记进 [`microduck_rl/`](microduck_rl/README.md)（按其 00→07 编号对号入座）；图与提示词进 [`diagrams/`](diagrams/)。顶层只放本索引。
- **相对路径**（本文 = `docs/README.md`）：指向 Rust 仓库 `../microduck/…`、RL 仓库 `../microduck_rl/…`；指向块内文档 `microduck/…`、`microduck_rl/…`、`diagrams/…`。子块内文档的路径约定见各自 README。
- **每个主题只有一个权威文档**，其余链接过去，不复制。
- **改完验证死链**：移动/新增文档后跑一次相对链接扫描，确保 `dead=0`。

> 官方一手资料：[`microduck/docs/`](../microduck/docs/)（`design/` 讲机制、`project/` 讲进度、`robot/` 讲使用）；[`microduck_rl/README.md`](../microduck_rl/README.md) 与 [`microduck_rl/AGENTS.md`](../microduck_rl/AGENTS.md)。
