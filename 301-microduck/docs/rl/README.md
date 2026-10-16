# microduck_rl 学习文档 · 导航与路线图

> 这里是为**强化学习（RL）初学者**准备的 [`microduck_rl`](../../microduck_rl/) 学习笔记体系。
>
> `microduck_rl` 是 microduck 双足鸭子机器人的**策略训练场**——用 GPU 并行物理仿真 + PPO 算法，训练出"会走、会滚、会翻身、会踢球"的运动策略，导出成 ONNX，再交给隔壁 Rust 仓库 [`microduck`](../../microduck/)（机器人运行时）在真机板子上加载执行。
>
> **这个 README 是整个体系的唯一入口。** 先花 5 分钟读完它，你就会知道：该按什么顺序读、每份文档解决什么问题、目录为什么这么分、以及未来新文档往哪里长。读完其他代码库时，第六节的「方法论」也能直接套用。
>
> **本类分两半（物理分开）**：【总结文档】= 理解型 markdown（`00→07` 认知阶梯 + [`learning_roadmap.md`](learning_roadmap.md) + [`mdp_principles.md`](mdp_principles.md)）；【本地复现】= 动手型内容（[`practice/`](practice/)：可运行的通用 RL 基础课 + 未来复现 microduck 已有任务的脚本与记录）。

---

## 一、30 秒决策：我现在该读哪份？

| 你想做的事 | 直接去读 | 说明 |
|---|---|---|
| 我是 RL 初学者，想知道**几个月怎么安排** | [`learning_roadmap.md`](learning_roadmap.md) | ⭐ **约 12 个月的周计划 + 每阶段硬检查点**，把认知阶梯拉成可执行时间表 |
| 尽快在机器上把鸭子跑起来 | [`01-quickstart/desktop_quickstart.md`](01-quickstart/desktop_quickstart.md) | 环境搭建 + 首次训练 + 故障速查，**命令的唯一权威** |
| 无真机，想在本机复现仿真部署 + 训练（5090/py3.12） | [`01-quickstart/sim_deploy_rl_reproduce_guide.md`](01-quickstart/sim_deploy_rl_reproduce_guide.md) | 仿真查看 / 官方策略回放 / 本地 PPO 训练 + ONNX 导出的复现指南 |
| 系统理解"训练一只鸭子"的全流程 | [`03-walkthrough/training_walkthrough.md`](03-walkthrough/training_walkthrough.md) | ⭐ **学习主线**，跟着 velocity 实例走 8 个阶段 |
| 搞懂某个概念/原理（PPO、BAM、61 维…） | [`02-architecture/architecture_and_glossary.md`](02-architecture/architecture_and_glossary.md) | 第一部分是架构，第二部分是 A–F 概念词典 |
| 查"某个命令的代码入口在哪个文件" | [`07-reference/entry_points_map.md`](07-reference/entry_points_map.md) | 三类入口（插件/命令/脚本）速查 |
| 深入某个源码模块（robot/tasks/export…） | `04-modules/`（**#5 velocity 精讲已上线**，余见第五节） | 以 velocity 为脊柱逐个击破 |
| 我是纯 RL 小白，术语都不懂 | 先读 walkthrough 的「第 0 部分 · 预备知识」 | 未来会独立成 `00-primer/` |
| 只想快速搞懂 MDP 的数学原理 | [`mdp_principles.md`](mdp_principles.md) | 【总结】纯原理笔记（+ 同名海报），不绑定代码 |
| **动手练通用 RL 基础**（PyTorch/MLP、DQN/CartPole） | [`practice/fundamentals/`](practice/fundamentals/) | 【复现】可运行的 session_01 / session_02 |

---

## 二、文档地图（现有 + 规划）

分两半：【总结文档】按编号 **00 → 07 由浅入深、从整体到局部、最后到专题**；【本地复现】统一收在 `practice/`。

| 目录 | 文档 | 定位（一句话职责） | 属哪半 | 状态 |
|---|---|---|---|---|
| （本目录） | [`learning_roadmap.md`](learning_roadmap.md) | **几个月怎么学**（约 12 个月周计划 + 每阶段硬检查点，把编号阶梯拉成时间表） | 总结 | ✅ 现有 |
| （本目录） | [`mdp_principles.md`](mdp_principles.md) | **MDP 纯原理**（五元组 / 马尔可夫性 / Bellman）+ 同名海报 png/svg | 总结 | ✅ 现有 |
| `00-primer/` | RL 零基础预备 | 强化学习 / 仿真 / sim2real 扫盲 | 总结 | 🚧 规划（内容暂并入 walkthrough 第 0 部分） |
| `01-quickstart/` | [`desktop_quickstart.md`](01-quickstart/desktop_quickstart.md) · [`sim_deploy_rl_reproduce_guide.md`](01-quickstart/sim_deploy_rl_reproduce_guide.md) | **怎么跑起来 + 坏了怎么办**（命令与故障权威）；后者是无真机的仿真部署+训练复现指南 | 总结 | ✅ 现有 |
| `02-architecture/` | [`architecture_and_glossary.md`](02-architecture/architecture_and_glossary.md) | **是什么 / 为什么**（架构总览 §1–11 + 概念词典 A–F） | 总结 | ✅ 现有 |
| `03-walkthrough/` | [`training_walkthrough.md`](03-walkthrough/training_walkthrough.md) | **跟着实例走一遍全链路**（阶段 0–8，velocity 贯穿） | 总结 | ✅ 现有 ⭐主线 |
| `04-modules/` | 13 份模块导读（#5 已完成） | **逐个吃透源码模块**（以 velocity 为脊柱） | 总结 | 🚧 1/13 · [#5 velocity 精讲](04-modules/05-velocity-deep-dive.md) ✅ |
| `05-source-walkthrough/` | velocity 端到端源码走读（规划） | **一条命令串起所有模块**的源码级走读 | 总结 | 🚧 规划 |
| `06-topics/` | sim2real / reward 设计 / 案例复盘（规划） | **专题深挖难点**，串联仓库自带的 specs/plans | 总结 | 🚧 规划 |
| `07-reference/` | [`entry_points_map.md`](07-reference/entry_points_map.md) | **代码入口速查**（想跑某功能从哪进） | 总结 | ✅ 现有 |
| `practice/` | [`README.md`](practice/README.md) + `fundamentals/`（session_01 / session_02） | **本地复现**：动手跑通——通用 RL 基础课 + 未来 microduck 已有任务的复现脚本/记录 | 复现 | ✅ 现有（fundamentals 已就位；任务复现待补） |

> **仓库自带的一手资料**（不是本笔记，但学习时要配合看）：
> [`README.md`](../../microduck_rl/README.md)（项目全景）、[`AGENTS.md`](../../microduck_rl/AGENTS.md)（改 env 前必读的规则）、[`docs/superpowers/specs/`](../../microduck_rl/docs/superpowers/specs/)（每个任务的设计文档）、[`docs/superpowers/plans/`](../../microduck_rl/docs/superpowers/plans/)（实现计划含验收标准）、[`docs/roller_standup_policy_summary.md`](../../microduck_rl/docs/roller_standup_policy_summary.md)（一个策略从设计到调试的完整复盘）。

---

## 三、学习思路：从 0 到 1 的认知阶梯

学一个陌生的 RL 机器人代码库，**不该是"从第一行读到最后一行"**，而是沿着下面这条螺旋阶梯，每一层都比上一层更具体、更接近代码：

```text
 ①地图      ②直觉        ③骨架        ④概念        ⑤模块击破        ⑥源码串联        ⑦专题        ⑧创造
 知道有啥 → 跑起来看 → 架构分层 → RL/术语 → 逐个吃透模块 → 一个实例串起来 → 深挖难点 → 自己改
 本README   quickstart  architecture  architecture  04-modules       05-source        06-topics   练习/加任务
                                  第二部分       (velocity脊柱)    -walkthrough
 ────────────────────────────────────────────────────────────────────────────────────────────────
 抽象、整体 ←──────────────────────────────────────────────────────────────────→ 具体、动手
```

**每一层"读完你应该能回答"：**

| 层 | 目标 | 对应文档 | 检验：你能回答…… |
|---|---|---|---|
| ① 地图 | 建立全局索引 | 本 README | 这个项目由哪些部分组成？我该先读什么？ |
| ② 直觉 | 亲手跑通一次 | quickstart | 怎么装环境、跑 smoke test、看到鸭子动？ |
| ③ 骨架 | 理解架构分层 | architecture §1–11 | 训练→导出→部署三条链路各经过哪些模块？ |
| ④ 概念 | 扫清术语障碍 | architecture 第二部分 A–F | PPO / obs normalizer / BAM / 61-D 各是什么？ |
| ⑤ 模块击破 | 吃透每个模块 | 04-modules（规划） | robot/tasks/export 每个模块内部怎么组织？ |
| ⑥ 源码串联 | 打通端到端 | 05-source-walkthrough（规划） | 一条 `uv run train` 从入口到 checkpoint 经过哪些函数？ |
| ⑦ 专题 | 深挖难点 | 06-topics（规划） | sim2real 为什么难？reward 符号约定是什么？ |
| ⑧ 创造 | 能自己改 | walkthrough 第 3 部分练习 | 我能改一个参数 / 加一个新任务吗？ |

### 🔑 核心原则：以 velocity 实例为"脊柱"

第 ⑤⑥ 层**不是孤立地逐个模块读**，而是**跟着一个主实例去读每个模块**——这个实例就是 `Mjlab-Velocity-Flat-MicroDuck`（训练鸭子走路）。

**为什么选它？** 它是仓库 README 明说的 "The main task"，也是其余 26 个任务的**共享基类**： StandUp / SitStand / Roulade / BallKick 全都是它的变体（换奖励、换地形、换命令，骨架不变）。学透 velocity，其他任务都是"改了几个 term"。

**脊柱怎么串起模块？**

```text
       uv run train Mjlab-Velocity-Flat-MicroDuck
                        │
   ┌────────────────────┼─────────────────────────────┐
   ▼                    ▼                              ▼
 robot 身体         tasks 任务定义                   export 导出
 (鸭子长啥样)   (velocity_env_cfg + mdp.py 词库)    (checkpoint→ONNX)
   │                    │                              │
 actuator 怎么动    symmetry 61维布局               publish 发布
 (BAM 舵机模型)     (obs/action 契约)               (gate→HF Hub)
   │                    │                              │
   └──────────→ 训练循环 (rsl_rl PPO) ←─────────────→ 部署 (infer_policy / duck-body)
```

每击破一个模块，就回到 velocity 看"它在训练一只鸭子走路时扮演什么角色"——这样你永远不迷失在孤立的代码里。

---

## 四、文档布局思路：为什么是"编号目录 + 增量填"

这套目录结构的设计理念，也是你未来整理任何学习笔记都可复用的：

1. **编号 = 学习顺序**：`00→07` 就是认知阶梯的层号，新人不用猜从哪读，跟着编号走即可。
2. **一层一目录，职责单一**：每个目录只回答一类问题（跑起来 / 原理 / 教学 / 模块 / 专题 / 速查），**互不重叠**——避免"同一件事在两份文档里各讲一遍"的混乱。
3. **骨架先立，文档增量填**：目录结构先建好（哪怕暂时空着放一份说明），新文档往对应层塞即可，**增长不破坏结构**。
4. **每个主题只有一个"权威文档"**：精确命令 → quickstart；概念原理 → architecture；全链路教学 → walkthrough；代码入口 → entry_points_map。其他地方需要时**链接过去**，不复制。

> 现状：`01/02/03/07` 已归位现有 4 份文档；`04` 已开始填充（**#5 velocity 精讲 ✅**）；`00/05/06` 仍是待填目录（各有一份占位 README）。

---

## 五、模块击破清单（`04-modules/` 的填充计划）

按"依赖顺序 + 重要性"排。★ = 最高杠杆，建议优先写。**每份都沿 velocity 脊柱，讲"这个模块在训练走路时干什么"，并保留代码里的真实调参案例与踩坑推理。**

| # | 模块 | 核心文件（行数） | 导读要回答 | 状态 |
|---|---|---|---|---|
| 1 | **robot** 身体 | `robot/microduck_constants.py`(275) + XML + `add_backlash.py`(139) | 鸭子长什么样？HOME 姿态、8 套 XML 变体、`passive_` 约定、BAM 参数从哪来 | 🚧 |
| 2 | **actuator** 怎么动 | `actuator/friction_dr_bam.py`(112) | BAM 如何模拟真实 XL330 舵机？摩擦/电压/延迟的域随机化怎么实现 | 🚧 |
| 3 | **tasks 总览** | `tasks/__init__.py`(278) | 27 个任务怎么注册？env_cfg 的组装模式、runner 继承 | 🚧 |
| 4 ★ | **mdp 词库** | `tasks/mdp.py`(**7188**) | 这么大怎么查？term 分类、两种惩罚风格、符号约定（双重否定 bug） | 🚧 |
| 5 ★ | **[velocity 精讲](04-modules/05-velocity-deep-dive.md)** | `tasks/microduck_velocity_env_cfg.py`(950) | 逐段读一个完整 env_cfg（脊柱本体） | ✅ |
| 6 | **任务变体** | `symmetry.py`(169) + `backlash.py`(92) + `slope_terrain.py`(115) | 对称性 / 齿隙 / 地形三种变体机制 | 🚧 |
| 7 | **其余任务对照** | 13 个 `*_env_cfg.py` | 差异地图：每个任务相对 velocity 改了什么（不逐个精讲） | 🚧 |
| 8 | **export** | `export.py`(311) | ONNX 导出、烘焙 obs normalizer、monkey-patch 过滤 passive 关节 | 🚧 |
| 9 | **publish** | `publish/cli.py`(227) + `manifest.py`(386) | shape gate、smoke run、manifest schema-2 的拒绝规则 | 🚧 |
| 10 | **训练入口** | `train_cli.py`(36) + `train_hook.py`(64) + `hf_jobs.py`(483) | train 命令链路、`--hf-jobs` 拦截、远程提交 | 🚧 |
| 11 | **sim** | `sim/body_server.py`(545) + `tof.py`(92) | 反向 sim2real、TCP JSON 协议、15→14 关节映射 | 🚧 |
| 12 | **scripts** | `scripts/infer_policy.py`(1524) + `testbench_sim2real.py`(611) 等 | 部署演练、sim2real 对比、可视化工具 | 🚧 |
| 13 | **tests** | `tests/` 19 个 CPU-only 测试 | 不变量如何锁住契约（读测试 = 读设计意图） | 🚧 |

---

## 六、🌍 方法论迁移：如何用这套体系学习**任何**陌生代码库

你要的"方便后续阅读其他代码作为指导"——把上面的阶梯抽象成通用步骤，换成任何项目都适用：

1. **建地图（半天）**：读 README、看目录树、理清依赖和入口。产出一张"这个项目由哪几块组成"的图。**不要急着读代码细节。**
2. **跑起来获直觉（半天）**：找到项目的 "smoke test / hello world"，亲手跑通一次，看到输入输出。跑不起来就先解决环境——**能跑，才有直觉**。
3. **画架构（1 天）**：搞清楚数据怎么流（输入→处理→输出）、分几层、层与层怎么调。画一张数据流图。
4. **补概念（按需）**：把领域术语（这里是 RL/仿真术语）列成词典，遇到就查，别想一次记全。
5. **选一个"主实例"当脊柱（关键！）**：找那个"最典型、其他都是它变体"的例子（这里是 velocity）。**以它为线索去读，而不是孤立读模块。**
6. **模块击破（主体）**：沿脊柱，逐个模块搞懂"它在这个实例里干什么"。大文件（如 7000 行的 mdp.py）抓主线 + 按需查，**不通读**。
7. **源码串联（打通任督二脉）**：选一条端到端路径（如一条 `train` 命令），从入口逐函数跟到底，把散落的模块串成一条线。
8. **专题深挖 + 动手改（内化）**：挑几个难点专题（如 sim2real），然后**真的改一个参数、跑一次、看结果**——能改懂，才是真懂。

> **贯穿始终的两个习惯**：① 每读一块，问"它在主实例里扮演什么角色"；② 遇到代码注释里的**真实案例、踩坑、调参推理**（如某参数从 ±0.5 改成 ±0.3 的原因、某次 NaN 崩溃的复盘），一定记下来——**这些"为什么"比"是什么"更值钱**。

---

## 七、维护约定（写给未来的自己 / 新增文档时遵守）

- **新文档放哪**：理解型文档按认知阶梯对号入座——模块导读进 `04-modules/`、专题进 `06-topics/`、速查进 `07-reference/`；**动手复现型（可运行代码 / 复现记录）进 `practice/`**。文件名用小写连字符（如 `04-modules/05-velocity-deep-dive.md`）。
- **相对路径约定**（本目录 = `docs/rl/`）：
  - 指向 **RL 仓库代码**：`../../microduck_rl/...`（子目录里的文档要多一层 `../../../microduck_rl/...`）
  - 指向 **Rust 仓库**：`../../microduck/...`（子目录里 `../../../microduck/...`）
  - 指向 **兄弟文档**：`../NN-目录名/文件.md`
  - 指向 **上层 docs / 端侧部署类**：`../README.md`、`../deploy/modules/robotd.md`（子目录里 `../../`）
- **保留案例与推理**：梳理原始资料时，保留原作者的技术例子、问题背景、推理过程、run 编号、踩坑复盘；**不要抽象成空泛的架构提纲**。
- **改完验证死链**：移动/新增文档后，跑一次死链扫描（提取所有 markdown 相对链接的目标路径，逐个 `test -e` 确认文件存在），确保 `dead=0` 再收工。

---

> 一句话总结这个体系：**先建地图，再跑起来，然后以 velocity 为脊柱逐个击破模块、串起源码、深挖专题，最后动手改。** 顺着编号读，你不会迷路。
