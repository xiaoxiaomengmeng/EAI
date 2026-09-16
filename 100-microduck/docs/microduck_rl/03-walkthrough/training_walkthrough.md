# microduck_rl 全链路实战教程（初学者版）

> **这篇文档是什么**：跟着**一个真实的训练实例**（`Mjlab-Velocity-Flat-MicroDuck`，也就是"让鸭子走路"），从零开始走完整条链路——**训练的是什么 → 怎么用代码定义 → 怎么配参数 → 怎么训练 → 怎么看曲线 → 怎么导出 → 怎么部署到真机 → 怎么验证**。
>
> **写给谁**：第一次接触本仓库、甚至第一次接触强化学习（RL）的人。遇到关键名词都会停下来讲解，尽量用白话和类比。
>
> **配套文档**（建议的阅读顺序）：
> 1. [`desktop_quickstart.md`](../01-quickstart/desktop_quickstart.md)——先把环境跑起来（装 uv、CUDA、第一次训练）
> 2. **本篇 `training_walkthrough.md`**——跟着实例走全链路（你在这里）
> 3. [`architecture_and_glossary.md`](../02-architecture/architecture_and_glossary.md)——想深挖原理时的"词典"（分层架构、61-D 契约、BAM、monkey-patch 等）
> 4. [`entry_points_map.md`](../07-reference/entry_points_map.md)——想搞清楚"每个命令的入口在哪"时查
>
> **贯穿实例**：全篇用 `Mjlab-Velocity-Flat-MicroDuck` 举例。它是 README 说的 "The main task"，也是其他 26 个任务的**共享基类**——学透它，其余任务都是它的变体（换奖励、换地形、换命令）。

---

## 目录

- [第 0 部分 · 预备知识（零基础必读）](#第-0-部分--预备知识零基础必读)
- [第 1 部分 · 全链路总览](#第-1-部分--全链路总览)
- [第 2 部分 · 分阶段详解](#第-2-部分--分阶段详解)
  - [阶段 0 · 准备环境](#阶段-0--准备环境)
  - [阶段 1 · 训练的是什么（任务 = 一个 MDP）](#阶段-1--训练的是什么任务--一个-mdp)
  - [阶段 2 · 如何定义（代码怎么组织）](#阶段-2--如何定义代码怎么组织)
  - [阶段 3 · 如何定义参数、配置参数（核心）](#阶段-3--如何定义参数配置参数核心)
  - [阶段 4 · 训练](#阶段-4--训练)
  - [阶段 5 · 查看曲线（wandb + play）](#阶段-5--查看曲线wandb--play)
  - [阶段 6 · 导出 ONNX](#阶段-6--导出-onnx)
  - [阶段 7 · 部署（两条路）](#阶段-7--部署两条路)
  - [阶段 8 · 验证（反向 sim2real）](#阶段-8--验证反向-sim2real)
- [第 3 部分 · 边读边改（练习点）](#第-3-部分--边读边改练习点)
- [第 4 部分 · 代码阅读顺序](#第-4-部分--代码阅读顺序)
- [第 5 部分 · 术语速查表](#第-5-部分--术语速查表)

---

## 第 0 部分 · 预备知识（零基础必读）

如果你已经懂 RL，可以跳到[第 1 部分](#第-1-部分--全链路总览)。如果是第一次，这一部分会帮你建立正确的心智模型——**后面所有代码都是为这套模型服务的**。

### 0.1 强化学习到底在做什么

用一个类比：**训练一只小狗**。

- 小狗（我们叫它 **agent / 策略 policy**）做一个动作（**action**，比如"坐下"）。
- 做对了给零食（**reward / 奖励**，一个分数，越高越好），做错了不给甚至扣分。
- 反复试错很多很多次，小狗慢慢学会："听到'坐下'口令（**observation / 观测**，它能感知到的信息）就坐下，能拿到零食"。

强化学习（Reinforcement Learning, **RL**）就是把这个过程写成程序：

> **一个"大脑"（神经网络）在"环境"里反复试错，靠"奖励"信号，学会"看到什么状态就该做什么动作"。**

放到本项目：

- **大脑** = 一个神经网络，输入是鸭子的身体状态（关节角度、姿态、想要的速度……共 61 个数字），输出是 14 个关节各自"该动到哪"。
- **环境** = MuJoCo 物理仿真里的鸭子 + 地面。
- **奖励** = "跟上命令速度给分""走得稳给分""跌倒扣分""脚打滑扣分"……很多项加起来。
- **训练** = 让 **4096 只虚拟鸭子同时**试错几百万步，把"怎么走路"这件事学进神经网络的权重里。
- **部署** = 把训练好的网络导出成一个文件（ONNX），塞进真机鸭子的小电脑，让它每 20 毫秒（50 Hz）跑一次：读传感器 → 网络算一下 → 给 14 个舵机发指令。

**关键洞察**：训练时我们**不写"怎么走路"的代码**（那太难了，涉及平衡、步态、落脚点……）。我们只定义**"什么样算走得好"（奖励）**，然后让算法自己摸索出走法。这就是 RL 和传统编程最大的区别。

### 0.2 关键名词速成

后面会反复遇到这些词，先混个眼熟（不用背，用到再回来查）：

| 名词 | 白话解释 | 本项目的值 |
|---|---|---|
| **policy / 策略**（也叫 agent） | 鸭子的"大脑"，一个神经网络。输入状态 → 输出动作 | 3 层 MLP `(512,256,128)` |
| **environment / env（环境）** | 仿真世界：物理 + 机器人 + 地面。策略在里面活动 | MuJoCo Warp，4096 个并行 |
| **step（步）** | 环境往前推进一个瞬间 | 物理步 0.005s |
| **episode（回合）** | 从"重置"到"结束"的一整段，像"一局游戏" | 跌倒或超时就结束 |
| **observation / obs（观测）** | 策略能"看到"的输入向量 | **61 维**（见阶段 1） |
| **action（动作）** | 策略输出的控制向量 | **14 维**（14 个关节目标） |
| **reward（奖励）** | 每步打的分数，策略的目标是让它长期最大 | 十几个加权项 |
| **termination（终止）** | 什么情况下一回合结束 | 跌倒 / NaN / 超时 |
| **command（命令）** | 外部给策略的目标 | 速度 twist + 头部姿态 + 身体姿态 |
| **MLP / 神经网络** | 多层感知机，策略的数学形式。一堆可调"权重" | 输入 61 → 512 → 256 → 128 → 输出 14 |
| **checkpoint（检查点）** | 训练中途保存的权重快照，文件名 `model_<迭代数>.pt` | 每 250 迭代存一个 |
| **ONNX** | 一种跨语言的神经网络文件格式，是"训练(Python)→部署(Rust)"的桥梁 | `policy.onnx` |
| **simulation / MuJoCo** | 物理仿真引擎，负责算"动作之后身体怎么变" | MuJoCo Warp（GPU 版） |
| **PPO** | 一种 RL 算法，决定"怎么根据奖励更新网络权重" | 见阶段 3 |
| **iteration（迭代）** | 训练的一轮："采样一批数据 → 更新一次网络" | 每迭代 = 24 步 × 4096 envs |
| **domain randomization（DR，域随机化）** | 训练时**故意随机化**物理参数（重心、摩擦、电压……），逼策略学得鲁棒，这样到真机（参数不完全一样）也能用 | 十几个开关 |
| **sim2real** | "从仿真到真机"的迁移。DR 就是为它服务的 | 全项目的核心难题 |

### 0.3 一张心智图：训练侧 vs 部署侧

整个项目横跨两个世界，中间靠 **ONNX** 这个文件格式握手：

```text
        训练侧（microduck_rl，本仓库）              │  部署侧（microduck，Rust 真机）
                                                    │
  Python + GPU + MuJoCo Warp 仿真                    │  Rust + RK3566 小电脑 + 真舵机
  4096 只虚拟鸭子并行试错几百万步                     │  1 只真鸭子，每 20ms 跑一次
                                                    │
  产出：checkpoint（model_*.pt，PyTorch 权重）        │
        │                                           │
        │  ① export：把 obs 归一化"烘焙"进网络         │
        ▼                                           │
     policy.onnx  ────────────────────────────────► 加载 onnx，50Hz 推理
     （跨语言握手：[1,61] 输入 → [1,14] 输出）         │
                                                    │
  契约：obs 必须是同样的 61 维、同样顺序；action 同样 14 维、同样关节顺序
        —— 训练和部署两侧字节级对齐，错一位鸭子就乱动
```

**记住这张图**：后面 8 个阶段，本质就是把左边这条流水线一步步走通，最后跨过 ONNX 这座桥到右边。

---

## 第 1 部分 · 全链路总览

把整条链路拆成 **8 个阶段**。你现在不用懂细节，先看清"从定义到部署"的骨架，后面每个阶段再展开：

```text
 ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐
 │ 阶段 0  │→ │ 阶段 1  │→ │ 阶段 2  │→ │ 阶段 3  │→ │ 阶段 4  │→ │ 阶段 5  │→ │ 阶段 6  │→ │ 阶段 7/8│
 │ 准备环境 │  │任务是什么│  │ 如何定义 │  │ 配参数  │  │  训练   │  │ 看曲线  │  │ 导出ONNX│  │ 部署验证 │
 └─────────┘  └─────────┘  └─────────┘  └─────────┘  └─────────┘  └─────────┘  └─────────┘  └─────────┘
   uv sync      MDP 五要素    3 个文件      env_cfg       uv run       wandb        export.py     publish
   CUDA GPU     obs/act/      协作定义      + rl_cfg      train        + play       烘焙归一化     /infer_policy
                reward/       一个任务      两大块参数     4096 envs    看奖励曲线    → policy.onnx  /duck-body
                term/command                             存 ckpt      看是否健康
 ────────────────────────────────────────────────────────────────────────────────────────────────────────
   环境          概念           代码          配置          执行         观测          转换          落地
```

每个阶段我都会用同一套结构讲，方便你对照：

- 🎯 **这一步在干什么**（白话）
- 📖 **关键名词**（本阶段遇到的术语，逐个讲）
- 💻 **对应代码**（文件 + 真实片段 + 逐行解释）
- ⌨️ **动手**（可以照着敲的命令）
- 🔗 **深入**（想深挖时去读哪份文档 / 哪个文件）

> **重要提示**：本项目的铁律是 **"先冒烟，再长跑"**（AGENTS.md 原话：*"A 5-iteration smoke test at 64 envs catches ~95% of config errors for cents. Never launch a long run without one."*）——先用 64 个环境跑 5 次迭代（几分钟、几分钱），能抓住 95% 的配置错误，然后才敢开 4096 环境的长训练。阶段 4 会展开。

---

## 第 2 部分 · 分阶段详解

### 阶段 0 · 准备环境

🎯 **这一步在干什么**

在能训练之前，你要把"工具链"和"依赖"准备好：装一个叫 **uv** 的工具，用它把项目需要的所有 Python 库（MuJoCo、PyTorch、mjlab……）下载到一个隔离的**虚拟环境**里。这一步只做一次。

📖 **关键名词**

- **uv**：一个用 Rust 写的、超快的 Python 包管理器，由 Astral（也是 `ruff` 的作者）开发。它一个工具就替代了过去一堆工具（`pip` 装包 + `venv` 建虚拟环境 + `poetry`/`pip-tools` 管依赖 + `pyenv` 管 Python 版本）。你可以把它类比成 Rust 的 `cargo` 或 Node 的 `npm`。
- **虚拟环境（`.venv`）**：一个项目专属的、隔离的 Python + 库的目录。这样本项目的库版本不会和你电脑上别的项目打架。
- **CUDA / GPU**：训练要在 NVIDIA 显卡上跑（4096 只鸭子并行仿真 + 神经网络反向传播，CPU 太慢）。CUDA 是 NVIDIA 的 GPU 计算平台。
- **warp-lang**：NVIDIA 的 GPU 物理仿真库，MuJoCo Warp 就建在它上面。`pyproject.toml` 里钉死 `warp-lang==1.12.0`。
- **`pyproject.toml` / `uv.lock`**：前者是"依赖清单"（声明要哪些库、什么版本），后者是"依赖快照"（把每个库的精确版本和哈希锁死，保证任何人任何机器装出来的环境完全一致）。

💻 **对应代码**

- [`pyproject.toml`](../../../microduck_rl/pyproject.toml)——依赖都在这里声明。几个关键点：
  - `requires-python = ">=3.12,<3.13"`——**必须 Python 3.12**（因为 BAM 执行器库钉了 `<3.13`）。
  - `dependencies` 里有 `mjlab==1.3.0`、`warp-lang==1.12.0`、`torch==2.9.1`、`onnxruntime`、`scipy>=1.16`（这条是 mjlab 忘了声明、本项目补上的）。
  - `[tool.uv.sources]` 把 `better-actuator-models`（BAM）指向 Rhoban 的 git 仓库 `mjlab_frictionloss` 分支。
- [`uv.lock`](../../../microduck_rl/uv.lock)——锁文件，别手改。

⌨️ **动手**

```bash
cd microduck_rl
uv sync          # 按 uv.lock 建 .venv 并装齐依赖（第一次会比较久，要下 ~2GB 的 CUDA 轮子）
```

> ⚠️ **ARM 机器（如 Apple Silicon / Jetson）注意**：先 `export UV_HTTP_TIMEOUT=600`。因为要下约 2GB 的 CUDA 轮子，uv 默认 30 秒超时会中断下载（README 明确写了这条）。

🔗 **深入**：[`desktop_quickstart.md`](../01-quickstart/desktop_quickstart.md)（完整的环境搭建）、[`entry_points_map.md`](../07-reference/entry_points_map.md) 的"前置知识：uv 与 uv run"节。

---

### 阶段 1 · 训练的是什么（任务 = 一个 MDP）

🎯 **这一步在干什么**

先想清楚：**我们到底要训练鸭子做什么？** 对 `Mjlab-Velocity-Flat-MicroDuck` 来说，答案是——**让鸭子跟着"速度命令"在平地上走路，同时把头/身体摆成"命令要求的姿态"**。

在 RL 里，任何"要训练的任务"都被形式化成一个叫 **MDP** 的数学框架。搞懂这个任务的 MDP，就搞懂了它的全部。

📖 **关键名词**

- **MDP（马尔可夫决策过程，Markov Decision Process）**：RL 描述"一个任务"的标准框架，由**五要素**组成。别被名字吓到，它就是把我们 0.1 节的小狗类比形式化：

  | MDP 五要素 | 含义 | velocity 任务里具体是什么 |
  |---|---|---|
  | **状态 / 观测（obs）** | 策略能看到的输入 | **61 维向量**（下方详解） |
  | **动作（action）** | 策略输出的控制 | **14 维**：14 个舵机关节各自的目标位置 |
  | **奖励（reward）** | 每步的打分 | 跟上速度 +、走得稳 +、跌倒 −、脚打滑 −… |
  | **终止（termination）** | 一回合何时结束 | 跌倒 / 状态出现 NaN / 超时 |
  | **命令（command）** | 外部给的目标 | 速度 twist + 头部姿态 + 身体姿态 |

- **twist（速度命令）**：机器人学里表示"线速度 + 角速度"的三元组。本项目是 `lin_vel_x`（前进/后退）、`lin_vel_y`（左右平移）、`ang_vel_z`（原地转向）。
- **proprioception（本体感受）**：机器人对"自己身体状态"的感知，比如每个关节转了多少度、身体朝哪、在往哪个方向倒。这是 61 维里最大的一块。
- **61 维 obs 的构成**（这是训练侧和部署侧必须字节级对齐的**核心契约**）：

  ```text
  [0:3]   base_ang_vel      躯干角速度（roll/pitch/yaw，来自 IMU）        ┐
  [3:6]   projected_gravity 重力在身体坐标系的投影（判断"有没有倾斜/倒")   │
  [6:20]  joint_pos_rel     14 个关节当前位置（相对 HOME 姿态）           │ 48 维本体感受
  [20:34] joint_vel_rel     14 个关节当前速度                            │
  [34:48] last_action       上一步输出的 14 维动作（给策略"记忆")         ┘
  ─────────────────────────────────────────────────────────────────
  [48:51] twist             速度命令（lin_x, lin_y, ang_z）              ┐
  [51:55] head              头部姿态命令（neck_pitch/head_pitch/yaw/roll)│ 13 维命令块
  [55:61] body              身体姿态命令（x,y,z,roll,pitch,yaw）         ┘
  ```

  即 **48 维本体感受 + 13 维命令 = 61 维**。（注意：训练时 critic 还能额外看到一些"特权信息"，比如真实接触力，但导出给真机的 actor 只看这 61 维。）

💻 **对应代码**

- [`microduck_velocity_env_cfg.py`](../../../microduck_rl/src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py) 开头的 docstring（L1-17）就是这个任务的"需求说明书"，白纸黑字写着它要鸭子做什么。
- 61 维布局的权威定义在 [`symmetry.py`](../../../microduck_rl/src/mjlab_microduck/tasks/symmetry.py) 的模块注释里（连每个关节的顺序都列了出来）。

⌨️ **动手**

```bash
uv run list-envs    # 列出所有已注册的任务 id，你会看到 Mjlab-Velocity-Flat-MicroDuck 等 27 个
```

🔗 **深入**：[`architecture_and_glossary.md`](../02-architecture/architecture_and_glossary.md) §B.5（MDP）、§D.1（61-D 逐维解读）。

---

### 阶段 2 · 如何定义（代码怎么组织）

🎯 **这一步在干什么**

知道了"要训练什么"（阶段 1 的 MDP），接下来看**这套 MDP 是用哪些代码文件、按什么结构写出来的**。这是学代码的关键一步——理解了组织方式，任何一个任务你都能快速定位。

📖 **关键名词**

- **manager-based 架构（管理器架构）**：这是 mjlab 框架的核心设计哲学。**你不写"环境循环"（读状态→算奖励→更新→判断终止……那套主循环框架已经写好了）**，你只写一个个**被框架回调的"零件"**，再把零件塞进对应的"管理器"。mjlab 有 7 个管理器：`observation`（观测）、`action`（动作）、`reward`（奖励）、`termination`（终止）、`event`（事件，如 DR）、`command`（命令）、`curriculum`（课程，随训练进度调难度）。
- **term（项）**：塞进管理器里的**一个零件** = 一个函数 + 权重 + 参数。比如"脚打滑惩罚"就是一个 reward term：`func=foot_slip函数, weight=-0.1, params={...}`。一个任务就是几十上百个 term 的组合。
- **env_cfg（环境配置）**：一个描述"这个任务的环境长什么样"的大对象——把上面 7 个管理器的所有 term 都装进去。由 `make_microduck_velocity_env_cfg()` 函数构造并返回。
- **rl_cfg（训练配置）**：一个描述"怎么训练"的大对象——神经网络结构、PPO 超参、训练多少迭代、存 wandb 哪里。本项目里叫 `MicroduckRlCfg`。
- **register_mjlab_task（注册任务）**：把 `task_id → (env_cfg, rl_cfg, runner)` 登记到框架的表里，这样 `uv run train <task_id>` 才能找到它。

💻 **对应代码**

**一个任务由三个文件协作定义**：

| 文件 | 角色 | 类比 |
|---|---|---|
| [`microduck_velocity_env_cfg.py`](../../../microduck_rl/src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py) | 组装这个任务：`make_microduck_velocity_env_cfg()` 返回 env_cfg，文件末尾定义 `MicroduckRlCfg` | "这道菜的菜谱" |
| [`mdp.py`](../../../microduck_rl/src/mjlab_microduck/tasks/mdp.py) | term 函数库：reward/obs/event/command 的**具体实现**（几千行，被所有任务共享） | "食材和调料" |
| [`tasks/__init__.py`](../../../microduck_rl/src/mjlab_microduck/tasks/__init__.py) | 用 `register_mjlab_task()` 把 task_id 登记进框架 | "把菜端上桌、写进菜单" |

**注册的真实代码**（[`tasks/__init__.py`](../../../microduck_rl/src/mjlab_microduck/tasks/__init__.py)，velocity 任务）：

```python
register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-MicroDuck",                  # ← uv run train 用的名字
    env_cfg=make_microduck_velocity_env_cfg(),                # ← 训练用环境配置（阶段 3 详解）
    play_env_cfg=make_microduck_velocity_env_cfg(play=True),  # ← 回放/演练用配置（关掉 DR、少环境）
    rl_cfg=MicroduckRlCfg,                                    # ← 训练超参（阶段 3 详解）
    runner_cls=MicroduckOnPolicyRunner,                       # ← 训练循环类（继承 mjlab 的 runner）
)
# Rough 版只是多传 rough=True（换成崎岖地形），其余完全一样。
```

**机器人本体也在代码里定义**——[`microduck_constants.py`](../../../microduck_rl/src/mjlab_microduck/robot/microduck_constants.py) 是"鸭子长什么样"的唯一真相源：

```python
# ① HOME 姿态：鸭子开机/重置时的默认站姿（各关节的弧度值）
HOME_FRAME = EntityCfg.InitialStateCfg(
    joint_pos={
        r".*left_hip_pitch.*": -0.4579,   # 用正则匹配关节名
        r".*left_ankle.*":     0.4530,
        r".*neck_pitch.*":     0.3491,    # 头也摆到默认位
        # …共 14 个驱动关节
    },
    joint_vel={".*": 0.0},                # 初始速度全 0（静止站姿）
)

# ② BAM 执行器：模拟真实 XL330 舵机的物理特性（这是 sim2real 的关键，见阶段 3 的名词讲解）
_BAM_ACTUATOR_KWARGS = dict(
    motor_name="xl330", model="m6",
    target_names_expr=(r"^(?!passive_).*",),  # 只驱动"非 passive_"的关节
    kp_fw=200.0,                               # 固件里的位置环刚度
    vin_range=(6.5, 8.2),                      # 电池电压随机化范围（DR！）
    vin_drop_gain_range=(0.0, 0.2),            # 负载越大电压越跌
    vin_min=6.0,                               # 电压下限
    delay_min_lag=3, delay_max_lag=6,          # 舵机响应延迟 3~6 拍（真实舵机有延迟！）
)
actuators = FrictionDRBamActuatorCfg(**_BAM_ACTUATOR_KWARGS)

# ③ 把 XML 模型 + HOME 姿态 + 执行器 打包成一个"机器人实体配置"
MICRODUCK_WALK_ROBOT_CFG = EntityCfg(
    spec_fn=get_walk_spec,          # 从 robot_walk.xml 读几何/质量/关节
    init_state=HOME_FRAME,          # 用上面的默认站姿
    collisions=(FULL_COLLISION,),   # 碰撞设置（脚底摩擦等）
    articulation=EntityArticulationInfoCfg(
        actuators=(actuators,),                 # 用 BAM 执行器
        soft_joint_pos_limit_factor=0.9,        # 关节限位留 10% 余量
    ),
)
```

> 📖 **补充名词：BAM（better-actuator-models）**——Rhoban（法国机器人团队）做的高保真舵机模型。普通仿真把舵机理想化成"给个目标角度就瞬间到位"，但真实的 XL330 舵机有**反向电动势、摩擦、电压依赖、响应延迟**。BAM 把这些都建模进去，这样仿真里训出来的策略到真机才不会"水土不服"。上面 `vin_range`、`delay_min_lag` 就是在模拟这些真实特性。

⌨️ **动手**

打开 [`microduck_velocity_env_cfg.py`](../../../microduck_rl/src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py)，找到 `make_microduck_velocity_env_cfg()` 函数，从上往下扫一遍——你会看到它在往 `cfg.observations[...]`、`cfg.rewards[...]`、`cfg.events[...]` 里一项项塞东西。这就是 manager-based 架构的样子（阶段 3 细讲）。

🔗 **深入**：[`architecture_and_glossary.md`](../02-architecture/architecture_and_glossary.md) §A.7（manager-based 哲学）、§A.3（BAM）。

---

### 阶段 3 · 如何定义参数、配置参数（核心）

🎯 **这一步在干什么**

一个任务所有的"可调旋钮"都集中在**两大块配置**里。学会配这两块，就等于学会"调教鸭子的行为"：

| | **env_cfg（环境配置）** | **rl_cfg（训练配置）** |
|---|---|---|
| 回答的问题 | "环境长什么样、什么算好" | "怎么学、学多久、网络多大" |
| 由谁构造 | `make_microduck_velocity_env_cfg()` | `MicroduckRlCfg`（同文件末尾） |
| 命令行前缀 | `--env.*` | `--agent.*` |
| 典型旋钮 | 奖励权重、DR 开关、命令范围、地形 | 网络层数、学习率、迭代数、PPO 参数 |

> **一个关键认知**：`make_microduck_velocity_env_cfg()` **不是从零写起**。它先调用 mjlab 基类的 `make_velocity_env_cfg()` 拿到"一套通用行走任务的默认配置"，然后**在默认基础上逐项微调**（改权重、加/删 term、开关 DR）。这就是 velocity 被称作"基类"的原因——其他任务也这样继承并微调它。

#### 3.1 env_cfg：定义"环境和任务"

下面按"子块"拆开看，每块都配真实代码。

**(a) 域随机化（DR）开关与范围**

📖 **名词**：**域随机化（domain randomization）** 是 sim2real 的核心手段——训练时**故意把物理参数随机打乱**（每只鸭子重心不同、摩擦不同、电压不同、还会被随机推一把），逼策略学得足够鲁棒，这样到了参数"某个固定值"的真机上也能扛住。

```python
# 开关（L30-42 附近）：每个 ENABLE_* 是一个布尔，决定要不要开这项随机化
ENABLE_COM_RANDOMIZATION = True      # 随机化质心位置
ENABLE_ARMATURE_RANDOMIZATION = True # 随机化关节转子惯量
# …共十几个开关

# 范围（L55-87 附近）：开了之后，随机化的幅度
COM_RANDOMIZATION_RANGE = 0.003      # 质心 ±3mm（训练中还 ramp 到 ±8mm）
ARMATURE_RANDOMIZATION_RANGE = (0.9, 1.1)   # 转子惯量在 0.9~1.1 倍之间随机
VELOCITY_PUSH_RANGE = (-0.3, 0.3)    # 随机"推搡"的强度（曾经是 ±0.5，太猛，调小到 ±0.3）
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 6.0   # IMU 安装角度误差 ±6°
ENCODER_BIAS_RANGE = (-0.015, 0.015) # 编码器零点偏差
```

> 💡 **案例**：`VELOCITY_PUSH_RANGE` 从 `±0.5` 调到 `±0.3`。代码注释（2026-07 audit）给了真实原因：±0.5 是个**加性踢力，比最大行走速度 0.4 还大**，每 3–6 秒来一下，会训出一种"永久紧张的跌倒恢复步态"；±0.3 既保留抗推鲁棒性，又让"平静的步态"成为最优解。这就是"配参数"的真实日常：调一个数、看训练效果、再调。

**(b) 奖励（rewards）——最体现"调教行为"的地方**

📖 **名词**：每个 **reward term** = 一个函数 + 一个 **weight（权重）** + 参数。**权重为正 = 鼓励**这个行为，**为负 = 惩罚**。所有 term 加权求和就是这一步的总奖励，策略的目标就是让它长期最大。

配置奖励有**两种写法**，都在真实代码里：

```python
# 写法一：改基类已有 term 的权重/参数（最常见）
cfg.rewards["upright"].weight = 2.0               # "躯干保持直立"，故意给强（防前倾跌倒）
cfg.rewards["upright"].params["std"] = math.sqrt(0.05)
cfg.rewards["foot_slip"].weight = -0.1            # "脚打滑"惩罚，故意给弱（这机器人转向靠 pivot，太严会绑住脚）
cfg.rewards["air_time"].weight = 3.0             # "每步腾空时间落在合理窗口"鼓励
cfg.rewards["air_time"].params["threshold_min"] = 0.125   # 窗口下限 0.125s
cfg.rewards["air_time"].params["threshold_max"] = 0.300   # 窗口上限 0.300s

# 写法二：新增一个基类没有的 term
cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,   # 用 mdp.py 里的函数
    weight=-1.0,                    # 惩罚
    params={"sensor_name": self_collision_cfg.name},
)

# 也可以删掉一个基类默认的 term
cfg.rewards.pop("soft_landing", None)
```

> ⚠️ **符号约定（踩坑点）**：惩罚项的 weight **必须为负**。sitstand 任务曾出过一个著名 bug——把几个惩罚项的 weight 写成了"负负得正"，结果它们变成了**奖励暴力行为**（wandb 上 `Episode_Reward/descent_speed` 冲到 +4.6），训出一只"用屁股猛砸地面"的鸭子。所以 AGENTS.md 规定：**改完奖励，必查 wandb 上 `Episode_Reward/<惩罚项>` 是否 ≤ 0**（阶段 5 会讲怎么看）。

**(c) 命令（commands）**

velocity 任务有三组命令 term，对应 61 维 obs 里的 13 维命令块：

- **twist**：速度命令，`lin_x ∈ (-0.4, 0.4)`、`lin_y`、`ang_z`——每回合随机采样一个目标速度，鸭子要跟上。
- **head_pose**：4 维头部姿态命令（脖子/头的俯仰偏航翻滚）。
- **body_pose**：6 维身体姿态命令（walking 任务里权重常设 0，即不强制，留给 sitstand 等任务用）。

**(d) 课程学习（curriculum）**

📖 **名词**：**课程学习（curriculum learning）** = 随训练进度**逐步调难度或调权重**，像学校从易到难。本项目用它在训练后期才慢慢加重"头部姿态跟踪"的要求（早期先学会走，别被头的要求带偏）。

```python
# 真实代码（L896-907）：head_pose_bias 这个奖励的权重分 4 阶段爬升
cfg.curriculum["head_pose_bias_weight"] = CurriculumTermCfg(
    func=microduck_mdp.reward_weight,
    params={
        "reward_name": "head_pose_bias",
        "weight_stages": [
            {"step": 0,                        "weight": 0.0},  # 一开始不管头
            {"step": 600  * NUM_STEPS_PER_ENV, "weight": 1.0},  # 600 迭代后开始要求
            {"step": 1000 * NUM_STEPS_PER_ENV, "weight": 2.0},
            {"step": 1500 * NUM_STEPS_PER_ENV, "weight": 3.0},  # 1500 迭代后要求最严
        ],
    },
)
```

（`NUM_STEPS_PER_ENV=24`，所以 `600 * 24` 是把"迭代数"换算成"环境步数"。）

**(e) 观测与终止（简述）**

- **observations**：actor 的 61 维（阶段 1 已详解）+ critic 的特权信息，每项都可加噪声（模拟真实传感器）。
- **terminations**：跌倒（躯干倾角过大）、状态出现 NaN、超时——满足任一条，这一回合结束并重置。

🔗 **深入**：奖励/观测/事件函数的实现全在 [`mdp.py`](../../../microduck_rl/src/mjlab_microduck/tasks/mdp.py)；DR 的物理参数含义见 [`architecture_and_glossary.md`](../02-architecture/architecture_and_glossary.md) §A.6（timestep / solref / armature）。

---

#### 3.2 rl_cfg：定义"怎么训练"

这是 [`microduck_velocity_env_cfg.py`](../../../microduck_rl/src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py) **文件末尾的 `MicroduckRlCfg`（L912-949）全文**，我逐块讲解：

```python
MicroduckRlCfg = RslRlOnPolicyRunnerCfg(
    # ── ① actor：策略网络（就是"鸭子的大脑"，最终会导出成 ONNX）──
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),   # 3 个隐藏层，宽度依次 512→256→128
        activation="elu",              # 激活函数（给网络加"非线性"，否则学不了复杂行为）
        obs_normalization=True,        # ★关键★ 对 61 维输入做归一化；导出时这一步会"烘焙"进 ONNX
        distribution_cfg={             # 输出一个高斯分布（均值+方差），靠方差来"探索"
            "class_name": "GaussianDistribution",
            "init_std": 1.0, "std_type": "scalar",
        },
    ),
    # ── ② critic：价值网络（只在训练时用，估计"当前状态有多好"，不导出到真机）──
    critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True,
    ),
    # ── ③ algorithm：PPO 算法超参（决定"怎么根据奖励更新网络"）──
    algorithm=PpoWithSymmetryCfg(
        clip_param=0.2,            # PPO 的 ε：限制每次更新的幅度，防止一步走太狠把策略搞崩
        entropy_coef=0.01,         # 熵系数：鼓励"探索"。太小 → 过早收敛（见下方案例）
        learning_rate=1.0e-3,      # 学习率
        schedule="adaptive",       # 自适应：按 KL 散度动态调学习率
        desired_kl=0.01,           # adaptive 模式的目标 KL（新旧策略的差异度）
        gamma=0.99,                # 折扣因子：未来奖励打多少折（越接近 1 越看重长远）
        lam=0.95,                  # GAE 的 λ：优势估计的"偏差 vs 方差"权衡
        num_learning_epochs=5,     # 每批采到的数据，反复训练 5 遍
        num_mini_batches=4,        # 每遍把数据切成 4 个小批
        max_grad_norm=1.0,         # 梯度裁剪：防止梯度爆炸
        value_loss_coef=1.0, use_clipped_value_loss=True,
        symmetry_cfg=SYMMETRY_CFG if ENABLE_SYMMETRY else None,  # 左右对称约束（可选）
    ),
    # ── ④ 日志与训练时长 ──
    wandb_project="mjlab_microduck",  # 曲线存到这个 wandb 项目（阶段 5）
    experiment_name="velocity",       # 输出目录名
    run_name="velocity",              # wandb run 名：<日期时间>_velocity
    save_interval=250,                # 每 250 迭代存一个 checkpoint（model_250.pt, model_500.pt…）
    num_steps_per_env=24,             # 每迭代、每个环境采 24 步 → 一批数据
    max_iterations=50_000,            # 最多训 5 万迭代
)
```

📖 **关键名词讲解**

- **actor / critic（演员-评论家）**：现代 RL 常用两个网络。**actor（演员）** = 策略，决定动作，**这是要部署的东西**。**critic（评论家）** = 价值函数，评估"当前局面多好"，只用来帮 actor 学得更稳，**训练完就扔掉**。本项目两者结构相同（都是 512/256/128），但 critic 训练时能看到 actor 看不到的"特权信息"（这叫**非对称 actor-critic**）。
- **obs_normalization（观测归一化）**：把 61 维输入各自缩放到差不多的量纲（有的量是角速度、有的是角度，数值范围差很多）。**为什么标了 ★**：这个归一化的均值/方差是训练中学出来的，导出 ONNX 时必须把它"烘焙"进网络图里，否则真机喂进来的原始数据分布对不上，鸭子直接乱动。
- **PPO（Proximal Policy Optimization，近端策略优化）**：目前机器人 RL 最主流的算法，出自 OpenAI。核心思想是"**小步快跑**"——每次更新都别改太狠（`clip_param=0.2` 就是这个"刹车"），这样训练稳定。
- **entropy（熵）/ 探索**：熵衡量策略的"随机性/探索欲"。`entropy_coef` 给熵一点奖励，防止策略过早"想当然"。**案例**：velocity 的 run `5yay13u4` 里熵从 10.9 崩到 1.9，策略过早收敛，最后**干脆不迈步了**（air_time 从 1.01 掉到 0.02）——这就是探索不足的典型病症。
- **gamma（γ，折扣因子）/ lam（λ）**：`gamma=0.99` 表示"未来第 n 步的奖励按 0.99ⁿ 打折"，越看重长远越接近 1；`lam=0.95` 是 GAE（广义优势估计）的平滑系数。这两个是 RL 的经典超参，先用默认值即可。

#### 3.3 两种改参数的方式

配好参数后，你有**两种方式**改它们：

| 方式 | 怎么做 | 何时用 | 特点 |
|---|---|---|---|
| **① 改代码** | 直接编辑 `make_microduck_velocity_env_cfg()` 或 `MicroduckRlCfg` 里的数值 | 确定了要长期生效的改动 | 持久、进 git、团队共享 |
| **② 命令行覆盖** | `uv run train <task> --env.<字段> <值>` 或 `--agent.<字段> <值>` | 做实验、临时试不同值 | 不改代码、一次性 |

命令行覆盖的例子：

```bash
# 改 env_cfg 的字段：把并行环境数设成 4096
uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096

# 改 rl_cfg 的字段：只训 5 迭代（冒烟测试）
uv run train Mjlab-Velocity-Flat-MicroDuck --agent.max_iterations 5
```

📖 **名词：tyro**——命令行参数**不是手写的**。mjlab 用一个叫 `tyro` 的库，**自动**从 env_cfg / rl_cfg 这些 dataclass 的字段生成 `--env.*` / `--agent.*` 参数。所以能覆盖哪些字段、完整列表是什么，用 `--help` 现场看最准：

```bash
uv run train Mjlab-Velocity-Flat-MicroDuck --help   # tyro 自动列出所有可覆盖字段
```

🔗 **深入**：`uv run train` 从命令到训练循环的完整调用链路，见 [`entry_points_map.md`](../07-reference/entry_points_map.md) §五；PPO / obs normalizer / 对称性的原理见 [`architecture_and_glossary.md`](../02-architecture/architecture_and_glossary.md) §B.2、§B.6。

---

### 阶段 4 · 训练

🎯 **这一步在干什么**

参数都配好了，现在**真正把 4096 只虚拟鸭子放出去试错**。这是最"省事"的一步（一条命令），但也是最需要"盯着"的一步（阶段 5 看曲线）。

📖 **关键名词**

- **smoke test（冒烟测试）**：正式长跑前，先用极小规模（64 环境、5 迭代）跑一遍，确认"能跑起来、不报错"。名字来自硬件——通电看会不会冒烟。
- **num_envs（并行环境数）**：同时仿真多少只鸭子。越多越快（GPU 并行），但越吃显存。正式训练用 4096，冒烟用 64。
- **iteration（迭代）**：训练的基本轮次，下面详解一轮里发生了什么。
- **resume（续训）**：从某个 checkpoint 接着训，而不是从头开始。

💻 **训练循环里，一个 iteration 发生了什么**

```text
每一轮 iteration：
 ① 采样 rollout：让 4096 只鸭子各走 num_steps_per_env=24 步
                 → 收集 4096 × 24 ≈ 9.8 万条 (obs, action, reward, ...) 经验
 ② 算优势：用 critic 网络 + GAE(λ=0.95) 估计"每个动作比平均好多少"
 ③ PPO 更新：拿这批经验更新 actor/critic，反复 num_learning_epochs=5 遍，
             每遍切成 num_mini_batches=4 个小批；每步都受 clip_param=0.2 约束
 ④ 记录曲线：把这轮的奖励、熵、loss、学习率等写到 wandb（阶段 5）
 ⑤ 存档：每逢 save_interval=250 的整数倍迭代，存一个 model_<iter>.pt
 然后进入下一轮，直到 max_iterations=50_000 或你手动停。
```

> **算笔账**：50000 迭代 × 每迭代约 9.8 万步 ≈ **49 亿步**经验。这就是为什么需要 GPU 上 4096 只鸭子并行——单只鸭子串行跑要跑到天荒地老。

⌨️ **动手**（务必按这个顺序）

```bash
# 第 1 步：冒烟测试（几分钟、几分钱，抓 95% 的配置错误）——永远先跑这个！
uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 64 --agent.max_iterations 5

# 第 2 步：正式训练（GPU 上约 1-2 小时出可用步态，训满 5 万迭代更久）
uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096

# （可选）从某个 checkpoint 续训
uv run train Mjlab-Velocity-Flat-MicroDuck \
    --agent.run-name resume --agent.load-checkpoint model_29999.pt --agent.resume True

# （可选）提交到 HuggingFace Jobs 云端跑（本项目特有 flag）
uv run train Mjlab-Velocity-Flat-MicroDuck --hf-jobs
```

**产出**：checkpoint（`model_250.pt`、`model_500.pt`……）存在 `output/velocity/<日期时间>_velocity/` 目录，并同步到对应的 wandb run。后面阶段 6 导出时，就用 `--wandb-run-path` + `--checkpoint` 来定位要哪个。

🔗 **深入**：入口 `train` 的调用链（uv → mjlab → runner）见 [`entry_points_map.md`](../07-reference/entry_points_map.md)；`--hf-jobs` 的远程提交机制见 [`architecture_and_glossary.md`](../02-architecture/architecture_and_glossary.md) §C.2。

---

### 阶段 5 · 查看曲线（wandb + play）

🎯 **这一步在干什么**

训练**不是"点下去等结果"**。你要**盯着曲线**判断它学得健不健康——奖励在涨吗？熵有没有崩？有没有鸭子因 NaN 挂掉？以及用 `play` 直接把策略跑起来**肉眼看鸭子走得怎么样**。这一步是 RL 的"仪表盘"，也是最需要经验的地方。

📖 **关键名词**

- **wandb（Weights & Biases）**：一个训练可视化网站/服务。训练时它自动把每轮的指标画成曲线，你在浏览器里看。项目名在 rl_cfg 里写死为 `wandb_project="mjlab_microduck"`，run 名是 `<日期时间>_velocity`。
- **play（回放）**：加载训好的策略，在仿真里让鸭子实际走给你看（关掉 DR、通常单只或少数几只，带可视化窗口）。

💻 **该盯哪些曲线**（这是本项目经验的结晶，每条都有真实病例）

| wandb 曲线 | 含义 | 健康信号 | 真实病例 |
|---|---|---|---|
| `Episode_Reward/<term>` | **每个**奖励项的累计值 | 惩罚项应 **≤ 0** | sitstand 符号 bug：`descent_speed` 冲到 **+4.6**，说明惩罚写反成了奖励 |
| `Episode_Termination/<reason>` | 各类终止原因的占比 | `nan_state` 应 ≈ 0 | 2026-08-21 的 NaN 崩溃就是从这里发现的 |
| entropy（熵；wandb 里通常在 `Loss/` 前缀下） | 策略的探索度 | 缓慢下降 | run `5yay13u4`：10.9 → 1.9 **崩塌**，过早收敛 |
| air_time / foot_height（通常在 `Episode_Reward/` 下） | 步态腾空时间/抬脚高 | 稳定在合理窗口 | 同上 run：air_time 1.01 → 0.02，鸭子**不迈步了** |
| `Train/mean_reward`、`Loss/learning_rate`、KL | 总奖励 + PPO 优化器状态 | 总奖励上升、KL 平稳 | KL 长期偏离 `desired_kl=0.01` 说明学习率不合适 |

> 💡 **怎么用这张表**：训练时开着 wandb 页面，重点看**奖励总曲线在涨**、**entropy 不崩**、**nan_state 为 0**、**air_time 稳定**。任何一条异常，就回到阶段 3 调对应参数（比如 entropy 崩 → 调大 `entropy_coef`；惩罚项变正 → 检查 weight 符号）。

⌨️ **动手**

```bash
# 回放最新/指定 run 的策略，肉眼看鸭子走路
uv run play Mjlab-Velocity-Flat-MicroDuck --wandb-run-path <entity/project/run_id>
```

`--wandb-run-path` 的格式是 `实体/项目/run_id`。不想手抄 run_id 的话，[`wandb_utils.py`](../../../microduck_rl/scripts/wandb_utils.py) 里有 `find_latest_run` / `find_latest_runs` 帮你自动找最新的 run（项目常量 `WANDB_PROJECT="pollen-robotics/mjlab_microduck"`）。

🔗 **深入**：这些曲线背后的奖励/终止实现见 [`mdp.py`](../../../microduck_rl/src/mjlab_microduck/tasks/mdp.py)（注释里记录了大量 run 编号和调参推理）；obs normalizer 为什么关键见 [`architecture_and_glossary.md`](../02-architecture/architecture_and_glossary.md) §B.6。

---

### 阶段 6 · 导出 ONNX

🎯 **这一步在干什么**

训完得到的是 **checkpoint（`model_*.pt`）**——PyTorch 格式，**只有 Python 训练侧认得**。真机鸭子跑的是 Rust，它需要一个跨语言、跨框架的格式。这一步就是把 `.pt` **转换成 `policy.onnx`**（输入 `[1,61]` → 输出 `[1,14]`）。这是"跨过 ONNX 这座桥"的第一步（回顾 0.3 的心智图）。

📖 **关键名词**

- **ONNX（Open Neural Network Exchange）**：开放的神经网络格式。训练侧（Python/PyTorch）导出成 `.onnx`，部署侧（Rust）用 `onnxruntime` 加载——两种语言都能读，这就是它能当"桥梁"的原因。
- **烘焙 normalizer（bake in the normalizer）**：阶段 3 讲过 `obs_normalization=True`。那套归一化的均值/方差是训练中学出来的，**导出时必须把它固化进 ONNX 图里**，让真机喂进来的原始 61 维数据先被同样地归一化。漏了它，鸭子会乱动。
- **never convert by hand（绝不手工转换）**：`export.py` docstring 的原话铁律。为什么？因为 `play` 回放时会**自己**应用 normalizer，所以一个"忘了烘焙 normalizer"的手转 ONNX 在 play 里看着正常，一上真机就废。**唯一安全路径**是用官方脚本，它走 `runner.export_policy_to_onnx`，导出的就是 `actor(normalizer(obs))`。
- **monkey-patch 过滤 passive 关节**：导出时项目打了个"补丁"，自动把仿真里的 `passive_*` 关节（如轮子、backlash 铰链、下巴连杆）从动作空间剔除，保证输出严格是 14 维（真机只有 14 个舵机）。

💻 **对应代码**

- [`scripts/export.py`](../../../microduck_rl/scripts/export.py)——只是个 12 行的命令行壳，真正逻辑在包里。
- [`export.py`](../../../microduck_rl/src/mjlab_microduck/export.py)——`main()` 先把**第一个位置参数**解析成 task id，再把剩余参数按 `ExportConfig` 解析；核心是 `run_export()`。`publish`（阶段 7）也直接调它，保证不会跳过烘焙这步。

⌨️ **动手**

```bash
# 注意：task id 是【位置参数】，直接跟在脚本后面，不要写成 --registry-name
uv run scripts/export.py Mjlab-Velocity-Flat-MicroDuck \
    --wandb-run-path <entity/project/run_id> \
    --checkpoint 3000 \
    --onnx-file policy.onnx
```

- `<TASK_ID>`（位置参数）：从哪个任务导出，`main()` 用 tyro 从所有已注册任务里匹配它。
- `--wandb-run-path`：从哪个 wandb run 取 checkpoint（`实体/项目/run_id`）。
- `--checkpoint 3000`：取第 3000 迭代的 `model_3000.pt`（省略则用该 run 最新的）。
- `--onnx-file`：输出文件名（默认 `output.onnx`）。完整选项 `--help` 看。

🔗 **深入**：ONNX 契约与 monkey-patch 细节见 [`architecture_and_glossary.md`](../02-architecture/architecture_and_glossary.md) §B.4、§C.1。

---

### 阶段 7 · 部署（两条路）

🎯 **这一步在干什么**

有了 `policy.onnx`，你有两条路：**路 A** 把它发布到 HuggingFace Hub，让真机鸭子下载加载；**路 B** 先在自己电脑的仿真里"演练"，肉眼确认策略没问题再上真机。新手建议先走路 B。

#### 路 A：发布到真机（`publish`）

`publish` 会**强制先走一遍 `run_export`**（所以 normalizer 一定被烘焙，跳过不了），再做几道关卡，最后上传：

```bash
uv run publish \
    --task Mjlab-Velocity-Flat-MicroDuck \
    --wandb-run-path <entity/project/run_id> \
    --checkpoint 3000 \
    --repo <user>/microduck-my-walk \
    --kind perpetual --slot walk
```

📖 **关键名词**

- **shape gate（形状门）**：上传前校验 ONNX 的输入/输出形状是不是 `[1,61]→[1,14]`，不对就拒绝——把"61/14 维契约"卡在上传前。
- **smoke run**：默认 `--smoke` 开启，用"看似合理的输入"跑一遍网络，**出现 NaN 就拒绝上传**（`cli.py` L82-83）。
- **manifest**：随 ONNX 一起发布的元数据（这个策略是什么、从哪个 run 来、怎么加载），schema-2 格式。
- **kind**：`perpetual`（持续，比如走路，保持到被告知停）或 `episodic`（一次性，跑 `--duration-s` 秒后自己回来，比如鞠躬、翻滚）。
- **slot**：perpetual 步态装到真机哪个槽位——`walk`/`stand`/`sitstand`/`ground_pick`/`kick_left`/`kick_right`/`roulade`（`cli.py` L64）。

发布成功后，**真机侧**（在 `microduck` 仓库，Rust 运行时）用 `robotctl` 加载该策略。确切子命令以 microduck 仓库文档为准，形如：

```bash
sudo robotctl policy load walk <user>/microduck-my-walk   # 示意，详见 microduck 仓库
```

#### 路 B：本地演练（`infer_policy.py`）

不上真机，直接在自己电脑的 **CPU 版 MuJoCo** 里把 ONNX 跑起来看效果：

```bash
uv run scripts/infer_policy.py --walking policy.onnx
```

- **键盘驱动**：方向键给速度命令，`G` 捡东西 / `Y` 坐站 / `R` 翻滚 / `K`·`L` 左右踢球（多策略热切换，模拟真机运行时在共享 61 维契约下换策略）。
- **同款执行器**：用和训练**完全相同的 BAM M6 XL330** 模型（`bam.mujoco.MujocoController`），所以演练结果可信。
- `--vin` / `--vin-drop-gain` / `--kp-fw`：把训练的 DR 范围钉到某个固定值；`--no-bam`：退回普通 XML PD 执行器（对比用）。

🔗 **深入**：[`publish/cli.py`](../../../microduck_rl/src/mjlab_microduck/publish/cli.py)（发布关卡）、[`scripts/infer_policy.py`](../../../microduck_rl/scripts/infer_policy.py)（演练）；整体部署链路见 [`architecture_and_glossary.md`](../02-architecture/architecture_and_glossary.md) §4.2。

---

### 阶段 8 · 验证（反向 sim2real，进阶）

🎯 **这一步在干什么**

前面是"仿真训练 → 真机部署"（正向 sim2real）。这一步玩个**镜像操作**：把**仿真的鸭子身体**，喂给**真机的完整软件栈**（`microduck` 仓库里的 `robotd`）。这样你**手上没有真机**，也能验证"真机那套 50Hz 循环 + ONNX 推理 + 安全 + 跌倒检测"整条链路跑得通。

📖 **关键名词**

- **反向 sim2real**：正常是"仿真里训好的策略装到真机"；反过来是"真机的软件栈去驱动一个仿真的身体"。`duck-body` 进程是**唯一知道'其实没有真机器人'的那部分**——它上面的所有代码（`duck_control::io::RobotIo` 以上）和真机跑的一模一样。
- **协议**：换行分帧的 JSON over TCP，握手时校验 `protocol` 版本号（两半分属两个仓库，必须能区分"是你的仿真器旧了"还是"守护进程旧了"）。

💻 **对应代码**：[`sim/body_server.py`](../../../microduck_rl/src/mjlab_microduck/sim/body_server.py)。它还负责两个"故意的映射"：① 真机侧 15 个关节（含 `mouth`，在索引 9），模型只有 14 个，这里插入/丢弃对齐；② 策略观测的是"重力在躯干坐标系的投影"，MuJoCo 给的是四元数，这里做旋转——和真机 IMU 的 SFLP 滤波是同一套算术。

⌨️ **动手**（两个终端）

```bash
# 终端 1（本仓库）：起 4 只仿真鸭子，共享一个世界一个窗口，监听 7801 起
uv run duck-body --ducks 4 --port 7801

# 终端 2（microduck 仓库，真机栈）：让 robotd 连到仿真身体
robotd --sim 127.0.0.1:7801
```

🔗 **深入**：[`architecture_and_glossary.md`](../02-architecture/architecture_and_glossary.md) §4.3；真机栈的 `robotd` 见 [`docs/modules/robotd.md`](../../modules/robotd.md)。

---

## 第 3 部分 · 边读边改（练习点）

学代码最好的方式是**改一个参数、跑冒烟、看曲线怎么变**。下面 5 个练习由浅入深，每个都基于前文的真实参数：

| # | 改什么 | 怎么改 | 预测 → 去 wandb 验证 |
|---|---|---|---|
| 1 | `foot_slip` 权重 | `-0.1` → `-0.5`（阶段 3.1b） | 脚打滑更少，但转向可能变"笨"（这机器人靠 pivot 转身） |
| 2 | `entropy_coef` | `0.01` → `0.001`（阶段 3.2） | 探索不足，可能重演 run `5yay13u4` 的熵崩塌、不迈步 |
| 3 | `VELOCITY_PUSH_RANGE` | `(-0.3,0.3)` → `(-0.5,0.5)`（阶段 3.1a） | 抗推更鲁棒，但步态更"紧张"（正是当初调小的原因） |
| 4 | `air_time` 窗口 | `threshold_max` `0.300` → `0.5`（阶段 3.1b） | 鼓励迈更大步，观察 foot_height 曲线 |
| 5 | 关掉一个 DR 开关 | `ENABLE_COM_RANDOMIZATION` → `False`（阶段 3.1a） | 训练更易收敛，但 sim2real 鲁棒性下降 |

> **方法**：每次只改**一个**参数，先跑冒烟（`--env.scene.num-envs 64 --agent.max_iterations 5`）确认不报错，再看要不要长跑。改完对照阶段 5 的曲线表，验证你的预测——**预测对了，说明你真的懂了**。

---

## 第 4 部分 · 代码阅读顺序

如果你想真正读透代码，按这个顺序（对应上面的阶段）：

```text
1. README.md + AGENTS.md            → 全局约定、命令清单、5 条不变量（先建立地图）
2. architecture_and_glossary.md     → 我沉淀的整体理解（分层/链路/契约/概念词典）
3. microduck_velocity_env_cfg.py    → ★核心★ 一个完整 env 怎么组装（阶段 1-3 全在这）
4. mdp.py 里 velocity 用到的 term    → reward/obs/event 的具体实现（按需查，别通读）
5. robot/microduck_constants.py     → 机器人本体 + HOME 姿态 + BAM 执行器参数
6. tasks/__init__.py                → 任务怎么注册进框架（register_mjlab_task）
7. 动手跑 smoke test                 → 阶段 4，看它真的动起来
8. export.py + publish/cli.py       → 部署链路（阶段 6-7）
```

> **提示**：`mdp.py` 有几千行，**不要从头读**。用到哪个 term（比如 `upright`、`foot_slip`）再搜哪个。它的注释里有大量调参推理和 wandb run 编号，是这个项目最宝贵的"经验库"。

---

## 第 5 部分 · 术语速查表

按"从训练到部署"的顺序排列，方便回查（详细讲解见对应阶段）：

| 术语 | 一句话 | 详见 |
|---|---|---|
| RL / policy | 靠奖励试错学出的"大脑"（神经网络） | 0.1 |
| MDP | 描述任务的五要素：obs/action/reward/termination/command | 阶段 1 |
| obs（61 维） | 策略输入 = 48 本体感受 + 13 命令 | 阶段 1 |
| action（14 维） | 策略输出 = 14 个关节目标 | 阶段 1 |
| manager-based | 只写"被回调的 term"，不写环境循环 | 阶段 2 |
| env_cfg / rl_cfg | 环境配置 / 训练配置，一个任务的两大块参数 | 阶段 3 |
| DR（域随机化） | 训练时随机化物理参数，逼出鲁棒性 | 3.1a |
| reward term / weight | 一个奖励项 + 权重（正鼓励、负惩罚） | 3.1b |
| curriculum | 随训练进度调难度/权重 | 3.1d |
| BAM | Rhoban 的高保真 XL330 舵机模型 | 阶段 2 |
| actor / critic | 策略网络（部署）/ 价值网络（训练辅助） | 3.2 |
| obs normalizer | 输入归一化，导出时烘焙进 ONNX | 3.2 / 阶段 6 |
| PPO / clip_param | 主流 RL 算法 / 它的"更新幅度刹车" | 3.2 |
| entropy | 探索度，崩了会过早收敛 | 3.2 / 阶段 5 |
| gamma / lam | 折扣因子 / GAE 平滑系数 | 3.2 |
| tyro | 从 dataclass 自动生成命令行参数的库 | 3.3 |
| smoke test | 64 环境 5 迭代，抓 95% 配置错误 | 阶段 4 |
| iteration | 一轮：采样 24×N 步 → PPO 更新 → 存档 | 阶段 4 |
| wandb | 训练曲线可视化服务 | 阶段 5 |
| checkpoint | `model_<iter>.pt`，每 250 迭代存一个 | 阶段 4 |
| ONNX | 跨语言网络格式，训练→部署的桥 | 阶段 6 |
| shape gate | 发布前校验 [1,61]→[1,14] | 阶段 7 |
| publish / slot / kind | 发布到 HF Hub / 步态槽位 / 持续或一次性 | 阶段 7 |
| sim2real（正/反） | 仿真↔真机的双向桥 | 阶段 8 |

---

## 结语

这份教程把"训练一只鸭子走路"从**定义 → 配置 → 训练 → 观测 → 导出 → 部署 → 验证**完整走了一遍。回头看，全链路的精髓其实是一句话：

> **一个任务 = env_cfg（定义"什么算好"）+ rl_cfg（定义"怎么学"）；改这两份配置 → 冒烟 → 看 wandb 曲线 → 迭代。**

其余 26 个任务都是 velocity 的变体（换奖励、换地形、换命令），骨架完全一样。你现在已经有了这张地图，接下来最好的学习就是——**挑一个练习点（第 3 部分），真的动手改一次、跑一次、看一次曲线**。

> 有原理层面的疑问（为什么是 61 维、monkey-patch 具体怎么打、BAM 的摩擦模型……），去 [`architecture_and_glossary.md`](../02-architecture/architecture_and_glossary.md) 查；有"这个命令的入口在哪"的疑问，去 [`entry_points_map.md`](../07-reference/entry_points_map.md) 查。
