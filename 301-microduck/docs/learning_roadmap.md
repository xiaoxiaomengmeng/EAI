# 强化学习学习路线图 · 从入门到大师（带周计划与检查点）

> 这是一份**时间维度**的学习编排表：把 [`README.md`](README.md) 描述的「认知阶梯（空间维度：00→07 该读什么）」拉成一条**约 12 个月、按周推进、每阶段带硬检查点**的可执行计划。
>
> 它面向 **RL 初学者**，终点是「能对 [`microduck_rl`](../../microduck_rl/) 这类真实机器人 RL 项目做研究级贡献」。前半程（阶段 0–2）补通用 RL 基础，后半程（阶段 3–5）以 `microduck_rl` 的 **velocity 任务为脊柱**做专项深入。
>
> **怎么用这份表**：
> - **时间预算**：按每周投入 **10–15 小时**（业余）估算；全职学习可把总时长压到约一半。
> - **检查点 = 硬门槛**：每个阶段末的「✅ 检查点」不通过**不要进阶**——RL 的知识是叠加的，前面虚了后面必崩。
> - **允许跳级**：若你已具备某阶段能力，直接做该阶段的检查点自测，过了就跳到下一阶段。
> - **动手 >> 看理论**：每个算法都要亲手实现一遍。计划里「产出」一栏就是你的验收物，缺一个都算没学完。
> - **记踩坑**：从阶段 1 起维护一份学习日志（模板见[文末](#附录学习日志与踩坑记录模板)）。RL 里「为什么错」比「是什么」值钱得多——这正是 [`AGENTS.md`](../../microduck_rl/AGENTS.md) 的写法。

---

## 阶段总览

| 阶段 | 周次 | 目标 | 主线资源 | 与本地文档的关系 | 检查点（硬门槛） |
|---|---|---|---|---|---|
| **0 · 前置** | 第 1–2 周 | Python/PyTorch + 够用的数学 | 任意 DL 入门课 | 暂不涉及 | 手写 MLP 跑通 MNIST |
| **1 · 入门** | 第 3–8 周 | 建立 RL 直觉 + 核心概念 | Sutton&Barto + HF Deep RL Course | 可读 [`00-primer/`](00-primer/README.md) | 实现 DQN 打通 CartPole |
| **2 · 深度 RL** | 第 9–20 周 | 掌握 PPO/SAC 算法体系 | OpenAI Spinning Up + CS285 + CleanRL | 配套 [`02-architecture`](02-architecture/architecture_and_glossary.md) 第二部分词典 | **手写 PPO** 打通连续控制 |
| **3 · 专项** | 第 21–36 周 | 吃透 microduck_rl（locomotion + sim2real） | 本仓库 + [`AGENTS.md`](../../microduck_rl/AGENTS.md) | 沿 [`03-walkthrough`](03-walkthrough/training_walkthrough.md) + [`04-modules`](04-modules/README.md) | 能改 reward / 加新 task 训练成功 |
| **4 · 精通** | 第 37–52 周 | 复现论文 + RL 调试工程能力 | CoRL/RSS/NeurIPS 论文 | 深挖 [`06-topics`](06-topics/README.md) 专题 | 独立复现一篇论文结果 |
| **5 · 大师** | 第 53 周起（持续） | 创造知识 | 自己的研究 | 反哺本地文档 | 顶会论文 / 有影响力的开源 |

> 认知阶梯（[`README.md` 第三节](README.md)）与本路线图的对应：阶段 1 ≈ 阶梯①②（地图/直觉），阶段 2 ≈ 阶梯③④（骨架/概念），阶段 3 ≈ 阶梯⑤⑥⑧（模块击破/源码串联/动手改），阶段 4–5 ≈ 阶梯⑦（专题深挖）+ 创造。

---

## 阶段 0 · 前置基础（第 1–2 周）

**目标**：扫清编程与数学障碍，让后面的 RL 学习不被「PyTorch 不会用」「梯度是什么」卡住。

| 周 | 主题 | 具体任务（动手） | 产出 |
|---|---|---|---|
| 1 | Python + PyTorch | 张量操作、autograd、`nn.Module`、优化器；跟着任一 DL 教程写一个 MLP | 一个能跑的 MLP 脚本 |
| 2 | 够用数学 + 训练循环 | 复习矩阵乘法、期望、梯度下降链式法则；理解 overfitting / normalization；训练 MLP 打通 MNIST | MNIST 分类准确率 > 97% |

**核心概念清单**（不必精通，见过即可）：向量/矩阵、期望 `E[·]`、梯度 `∇`、反向传播、loss、learning rate、batch。

> ✅ **阶段 0 检查点**：能独立用 PyTorch 从零写一个训练 MNIST 的 MLP，跑通并解释「loss 为什么下降」。做不到就先补，别往下走。

---

## 阶段 1 · 入门：建立直觉与核心概念（第 3–8 周）

**目标**：真正理解 RL 的问题建模（MDP），亲手实现第一个深度 RL 算法（DQN）。

**主线资源（选一条，别贪多）**：
- 教材：Sutton & Barto《Reinforcement Learning: An Introduction》——**前 6 章是根基**
- 视频课：**Hugging Face Deep RL Course**（免费、有实战、有证书，最适合初学者）或 David Silver 的 UCL RL Course
- 动手环境：**Gymnasium**（原 OpenAI Gym）

| 周 | 主题 | 具体任务（动手） | 产出 |
|---|---|---|---|
| 3 | MDP 与核心概念 | 学 state/action/reward/policy/return/discount γ；读 Sutton 前 3 章 | 一页纸概念笔记 |
| 4 | Value function & Bellman | 理解 V(s)、Q(s,a)、Bellman 方程；手算一个 3 状态小 MDP | Bellman 手算示例 |
| 5 | Tabular 方法 | 实现 **Q-learning** 打通 `FrozenLake`；理解 exploration（ε-greedy） | 可运行的 Q-learning |
| 6 | 从表格到神经网络 | 学 DQN 原理（experience replay、target network）；读 CleanRL 的 `dqn.py` | DQN 流程图解 |
| 7 | 实现 DQN（上） | 在 `CartPole-v1` 上从零实现 DQN，调通训练循环 | DQN 训练脚本 |
| 8 | 实现 DQN（下） | 调参让 CartPole 稳定撑到 500 步；记录调参过程 | **DQN 打通 CartPole** |

**这一阶段可以在本地文档里预热**：读 [`00-primer/`](00-primer/README.md) 的 RL/仿真扫盲，对 microduck_rl 是干什么的先有个印象。

> ✅ **阶段 1 检查点**（入门毕业考）：
> 1. 能白板推导 Bellman 期望方程；
> 2. **独立实现 DQN 让 `CartPole-v1` 稳定跑满 500 步**；
> 3. 能口头解释：为什么要 experience replay？为什么需要 target network？

---

## 阶段 2 · 深度强化学习：算法体系（第 9–20 周）

**目标**：系统掌握三大算法家族，**手写一遍 PPO**——这是读懂 `microduck_rl`（它用的就是 rsl_rl 的 PPO）的**绝对前提**。

**主线资源**：
- **OpenAI Spinning Up in Deep RL**——最好的实战教材（文档 + 可读代码）
- **Berkeley CS285**（Sergey Levine）——深度 RL 理论最强的公开课
- **CleanRL**（GitHub）——每个算法单文件实现，读源码进阶的最佳材料

| 周 | 主题 | 具体任务（动手） | 产出 |
|---|---|---|---|
| 9 | Policy Gradient 基础 | 学 REINFORCE、policy gradient theorem；读 Spinning Up「Intro to Policy Optimization」 | 手推 policy gradient |
| 10 | Actor-Critic | 理解 advantage、value baseline、A2C/A3C；跑通 CleanRL 的 `a2c` | A2C 在 CartPole 跑通 |
| 11 | **GAE 与 return 估计** | 吃透 **Generalized Advantage Estimation**、λ-return；这是 PPO 的核心零件 | GAE 从零实现（numpy） |
| 12 | PPO 原理 | 学 importance ratio、clip、value loss、entropy bonus；读 Spinning Up PPO 章 | PPO 流程图解 |
| 13 | **手写 PPO（上）** | 在 `CartPole` 从零实现 PPO；理解 rollout buffer、minibatch update | PPO 打通 CartPole |
| 14 | **手写 PPO（下）** | 加 **obs/reward normalization、KL 早停、advantage normalization**；理解每个 trick 为什么需要 | 带 normalization 的 PPO |
| 15 | PPO 连续控制 | 把 PPO 迁到连续动作环境（如 `Pendulum` / BipedalWalker 简化版）；理解高斯策略 | PPO 连续控制跑通 |
| 16 | Value-based 进阶 | DQN → Double/Dueling/PER；读 CleanRL `dqn_atari.py` 选段 | 进阶 DQN 笔记 |
| 17 | 连续控制 Actor-Critic | 学 DDPG → TD3 → **SAC**；跑通 CleanRL 的 `sac` | SAC 在 Pendulum 跑通 |
| 18 | Normalization / 稳定性 | 深入 obs normalizer、running mean/std、reward scaling——**直接对应 microduck_rl 的「烘焙 normalizer」不变量** | 稳定性 trick 清单 |
| 19 | 框架对比 | 读 **rsl_rl** 源码（microduck 用的框架），对照自己手写的 PPO | rsl_rl PPO 走读笔记 |
| 20 | 阶段复盘 | 用 rsl_rl 跑一个官方 PPO 示例；把「手写 PPO」与「rsl_rl PPO」逐模块对齐 | 复盘文档 |

**配套本地文档**：这一阶段边学边翻 [`02-architecture/architecture_and_glossary.md`](02-architecture/architecture_and_glossary.md) 的**第二部分概念词典（A–F）**——PPO、obs normalizer、BAM、61-D 契约这些术语都能查到，且都贴着 microduck_rl 的代码讲。

> ✅ **阶段 2 检查点**（最关键的一道门槛）：
> 1. **能不看参考、从零手写 PPO 打通一个连续控制环境**；
> 2. 能解释 GAE 里 λ 的作用、为什么要 clip、obs normalizer 为什么必须一致；
> 3. 能说出 rsl_rl 的 PPO 与你手写版本的 3 处主要差异。
>
> ⚠️ 这一关不过，**不要进阶段 3**——`microduck_rl` 的 [`mdp.py`](../../microduck_rl/src/mjlab_microduck/tasks/mdp.py)（7188 行）会把你劝退。

---

## 阶段 3 · 专项深入：吃透 microduck_rl（第 21–36 周）

**目标**：以 **velocity 任务为脊柱**，把 microduck_rl 从环境搭建走到能改代码。这是「机器人 locomotion + sim2real」方向的完整实战教材。

**主线资源**：本仓库 [`microduck_rl/`](../../microduck_rl/) + [`README`](../../microduck_rl/README.md) + [`AGENTS.md`](../../microduck_rl/AGENTS.md)；本地文档 [`01-quickstart`](01-quickstart/desktop_quickstart.md) → [`03-walkthrough`](03-walkthrough/training_walkthrough.md) → [`04-modules`](04-modules/README.md)。

### 3A · 跑起来 + 建全局（第 21–24 周）

| 周 | 主题 | 具体任务（动手） | 产出 |
|---|---|---|---|
| 21 | 环境搭建 | 装 uv + CUDA；跟着 [`01-quickstart`](01-quickstart/desktop_quickstart.md) `uv sync`（ARM 机记得 `UV_HTTP_TIMEOUT=600`） | 可用的开发环境 |
| 22 | **Smoke test** | `uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 64 --agent.max_iterations 5`（先冒烟再长跑，[入口见此](07-reference/entry_points_map.md)） | 冒烟测试跑通 |
| 23 | 看鸭子动 | `uv run play` 回放；`uv run scripts/infer_policy.py --walking out.onnx` 键盘驱动 | 亲眼看到策略行为 |
| 24 | 架构全景 | 读 [`02-architecture`](02-architecture/architecture_and_glossary.md) §1–11 + 三条技术链路（训练/部署/反向 sim2real） | 一张手绘数据流图 |

### 3B · 以 velocity 为脊柱逐个击破模块（第 25–32 周）

> 沿 [`03-walkthrough/training_walkthrough.md`](03-walkthrough/training_walkthrough.md)（8 阶段主线）+ [`04-modules`](04-modules/README.md) 的 13 份模块导读推进。**每读一个模块，回到 velocity 问「它在训练走路时扮演什么角色」**。

| 周 | 模块 | 核心文件 | 产出 |
|---|---|---|---|
| 25 | velocity 精讲（脊柱本体） | [`microduck_velocity_env_cfg.py`](../../microduck_rl/src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py)(949) + [本地精讲](04-modules/05-velocity-deep-dive.md) | 逐段注释笔记 |
| 26 | robot 身体 | [`microduck_constants.py`](../../microduck_rl/src/mjlab_microduck/robot/microduck_constants.py)(275) + 8 套 XML + `passive_` 约定 + HOME 姿态 | 机器人模型笔记 |
| 27 | actuator 执行器 | [`friction_dr_bam.py`](../../microduck_rl/src/mjlab_microduck/actuator/friction_dr_bam.py)(112) + BAM 三层继承 + 摩擦/电压/延迟 DR | BAM 机制笔记 |
| 28 | tasks 注册总览 | [`tasks/__init__.py`](../../microduck_rl/src/mjlab_microduck/tasks/__init__.py)(278) + `MicroduckOnPolicyRunner` | 27 任务注册表理解 |
| 29 | **mdp 词库（重点）** | [`mdp.py`](../../microduck_rl/src/mjlab_microduck/tasks/mdp.py)(7188)——抓主线+按需查，**不通读**；两种惩罚风格 + 符号约定 | mdp term 分类图 |
| 30 | 61-D 契约 & symmetry | [`symmetry.py`](../../microduck_rl/src/mjlab_microduck/tasks/symmetry.py)(169) + obs/action 布局 | 61 维逐维表 |
| 31 | export → ONNX | [`export.py`](../../microduck_rl/src/mjlab_microduck/export.py)(311)：烘焙 obs normalizer、monkey-patch 过滤 passive 关节 | 导出链路笔记 |
| 32 | publish + sim2real | [`publish/`](../../microduck_rl/src/mjlab_microduck/publish/) 发布 gate + [`sim/body_server.py`](../../microduck_rl/src/mjlab_microduck/sim/body_server.py)(545) 反向 sim2real | 部署链路笔记 |

### 3C · 真正动手改（第 33–36 周）

| 周 | 主题 | 具体任务（动手） | 产出 |
|---|---|---|---|
| 33 | 读规则 | 精读 [`AGENTS.md`](../../microduck_rl/AGENTS.md) 的 invariants / reward 设计规则 / sim2real footguns——**这些是踩坑换来的分水岭知识** | 不变量清单 |
| 34 | 改参数 | 改一个 reward 权重 / DR 开关，跑 4096 envs 训练，对比行为变化；每个 penalty term 必须 ≤ 0 | 一次 A/B 对比记录 |
| 35 | 读测试 | 跑 `uv run --with pytest pytest tests/`；读几个 cfg-invariant 测试，理解「读测试=读设计意图」 | 测试如何锁契约 |
| 36 | **加新 task** | 照最接近的模板（如 velocity）建一个新 env cfg，注册、smoke test、短训 | **一个新 task 训练成功** |

> ✅ **阶段 3 检查点**：
> 1. 能独立把 smoke test → 正式训练 → export ONNX → infer 演练整条链路走一遍；
> 2. 能画出 61-D obs 布局并解释「命令块零填充约定」为什么支撑运行时策略热切换；
> 3. **改一个 reward 权重并预测+验证行为变化**，且每个 `Episode_Reward/<penalty>` ≤ 0；
> 4. 照模板加一个新 task 并 smoke test 通过（obs 61-D、每 reward term 能算、ONNX 能导出）。

---

## 阶段 4 · 精通：研究与工程能力（第 37–52 周）

**目标**：从「会改」到「会研究」——能复现论文、能设计新任务、能调试真实 RL 故障。

| 周 | 主题 | 具体任务（动手） | 产出 |
|---|---|---|---|
| 37–38 | 读论文入门 | 选一篇 legged locomotion 经典（如 ANYmal / ETH 的 sim2real 工作），精读 + 复述方法 | 论文阅读报告 |
| 39–41 | **复现论文** | 挑一篇有开源代码的论文，独立复现其核心结果（先跑通再改） | 复现结果对齐原论文 |
| 42–44 | RL 调试工程 | 系统练 reward hacking 识别、curriculum 设计、reward shaping；参考 [`AGENTS.md`](../../microduck_rl/AGENTS.md) 的 reward 规则与 [`docs/roller_standup_policy_summary.md`](../../microduck_rl/docs/) 的完整复盘 | 一份调试 case study |
| 45–47 | sim2real 深潜 | actuator system identification、domain randomization 设计、real-vs-sim obs 对比（用 [`plot_observations`](07-reference/entry_points_map.md) 类工具） | sim2real 差距分析报告 |
| 48–50 | 大规模训练 | 理解 GPU 并行（为何 4096 envs）、分布式 PPO、`--hf-jobs` 远程训练；跑一次长训并全程读 wandb | 一次完整长训 + 日志解读 |
| 51–52 | 设计新任务 | 在 microduck_rl 里从零设计一个全新技能（reward + curriculum + gate），走完训练→评估 | **一个新技能训练成功并能评估** |

**本地文档**：这一阶段深挖 [`06-topics/`](06-topics/README.md) 专题（sim2real / reward 设计 / 案例复盘），并配合仓库自带的 `docs/superpowers/specs/`（任务设计文档）与 `docs/superpowers/plans/`（含验收标准的实现计划）。

> ✅ **阶段 4 检查点**：
> 1. 能独立复现一篇 RL 论文的核心结果并解释差距；
> 2. 面对一个「total reward 涨了但技能没学会」的 run，能用 headless eval + 读 wandb 定位根因（而非瞎调 reward）；
> 3. 能从零设计一个新任务，reward + curriculum + 硬 gate 齐备，训练出可用策略。

---

## 阶段 5 · 大师：创造知识（第 53 周起，持续）

「大师」不是学完的，而是**做出原创贡献并沉淀成可复用方法论**。没有固定周计划，改为持续的能力目标：

| 能力维度 | 具体标志 |
|---|---|
| 原创研究 | 提出新方法/新洞见，在 CoRL/RSS/NeurIPS/ICLR 发表论文 |
| 系统洞见 | 对某子领域（如 legged locomotion sim2real）有成体系的判断力 |
| 工程影响 | 开源有影响力的项目，或被他人采用的工具/框架 |
| 方法论沉淀 | 像 [`AGENTS.md`](../../microduck_rl/AGENTS.md) 那样把踩坑固化成不变量；像本地 `docs/` 那样写出让后人少踩坑的文档 |
| 传承 | 带团队/指导学生，评审论文，推动社区 |

> ✅ **阶段 5 无「通过」一说**——它是持续产出与影响的过程。当你开始反哺文档、指导他人、发表原创工作时，你已经在这条路上了。

---

## 检查点总表（速查）

| 阶段 | 一句话门槛 | 关键产出物 |
|---|---|---|
| 0 | 手写 MLP 跑通 MNIST | MLP 脚本 |
| 1 | **DQN 打通 CartPole（500 步）** | DQN 实现 |
| 2 | **从零手写 PPO 打通连续控制** | 手写 PPO + GAE |
| 3 | **改 reward 验证行为 + 加新 task 成功** | 新 task cfg |
| 4 | **复现一篇论文 + 从零设计新技能** | 复现报告 + 新技能 |
| 5 | 持续原创贡献 | 论文 / 开源 / 方法论 |

---

## 资源清单（去重汇总）

**课程 / 教材**
- Sutton & Barto《Reinforcement Learning: An Introduction》（圣经，前 6 章）
- Hugging Face Deep RL Course（入门首选，免费+实战）
- David Silver UCL RL Course（经典视频）
- OpenAI **Spinning Up in Deep RL**（阶段 2 实战主线）
- Berkeley **CS285**（Sergey Levine，深度 RL 理论最强）

**代码 / 框架**
- **CleanRL**（单文件实现，读源码进阶）
- **Gymnasium**（练手环境）
- **rsl_rl**（microduck_rl 用的 PPO 框架）
- MuJoCo / MuJoCo Warp（mjlab）、Isaac Lab、Stable-Baselines3

**本地（microduck_rl）**
- [`README.md`](../../microduck_rl/README.md)（项目全景）· [`AGENTS.md`](../../microduck_rl/AGENTS.md)（改 env 前必读规则）
- [`01-quickstart`](01-quickstart/desktop_quickstart.md) · [`02-architecture`](02-architecture/architecture_and_glossary.md) · [`03-walkthrough`](03-walkthrough/training_walkthrough.md) · [`04-modules`](04-modules/README.md) · [`07-reference`](07-reference/entry_points_map.md)

---

## 贯穿始终的 5 条心法

1. **动手 >> 看理论**：每个算法都亲手实现一遍，哪怕照抄也要敲一遍。
2. **读源码是最快的进阶**：CleanRL（学算法）→ rsl_rl（学框架）→ microduck_rl（学真实工程）。
3. **建立「reward 是设计出来的」直觉**：RL 优化的是 reward 的字面意思，任何没约束的自由度都会被策略钻空子。
4. **记录踩坑**：把每次失败的原因写下来——「为什么」比「是什么」值钱得多。
5. **别孤立学**：始终有一个真实项目当主线（对你就是 microduck_rl），所有理论都挂到它上面理解。

---

## 附录：学习日志与踩坑记录模板

从阶段 1 起，每周维护一份。格式刻意模仿 [`AGENTS.md`](../../microduck_rl/AGENTS.md) 的「现象 → 根因 → 修法 → 可复用教训」四段式：

```markdown
## 第 N 周（阶段 X）

### 本周完成
- [ ] 任务 1（产出：xxx）

### 卡住的点 / 踩坑
- **现象**：训练时 reward 一直涨但鸭子不走
- **根因**：penalty 项符号写反，双重否定变成了奖励违规
- **修法**：把自否定惩罚函数的权重改为正
- **可复用教训**：每次 run 都要检查 wandb 里 Episode_Reward/<penalty> ≤ 0

### 下周计划
- ...
```

---

> 本路线图为学习梳理笔记，结合通用 RL 学习路径与 [`microduck_rl`](../../microduck_rl/) 项目实际编写。配套导航见 [`README.md`](README.md)；具体命令以 [`01-quickstart`](01-quickstart/desktop_quickstart.md) 为准；概念查询以 [`02-architecture`](02-architecture/architecture_and_glossary.md) 词典为准。
