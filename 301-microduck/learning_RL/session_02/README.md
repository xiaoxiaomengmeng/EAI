# session_02 · 阶段 1：MDP 建模 + 从零 DQN

> 对应 [`learning_roadmap.md`](../../docs/learning_roadmap.md) **阶段 1（第 3–8 周）** 的动手产出。
> 目标：**真正理解 RL 的问题建模（MDP），亲手实现第一个深度 RL 算法（DQN）**，并让 `CartPole-v1` 稳定撑到 500 步——通过阶段 1 的硬检查点。

这是整条 RL 学习路线的**第二节课**。session_01 练的是「深度学习零件」（张量 / autograd / 四步循环 / MLP），
本目录把这些零件**组装成第一个会自己学策略的 RL 智能体**：从「MDP 是什么」到「表格 Q-learning」再到「用神经网络逼近 Q 的 DQN」，
一步步把 roadmap 阶段 1 的三道硬门槛全部跑通。

认知路径：**先懂问题建模（MDP/Bellman）→ 再写最简单的学习算法（表格 Q-learning）→ 最后上深度 RL（DQN）**。

---

## 文件地图

| 文件 | 对应 roadmap | 讲什么 | 怎么跑 |
|---|---|---|---|
| [`01_mdp_and_bellman.py`](01_mdp_and_bellman.py) | 第 3 周：state/action/reward/policy/return/γ；第 4 周：V(s)/Q(s,a)/Bellman 方程 + 手算 3 状态 MDP | 把「手算 Bellman」变成**可运行、可验证**的代码：4 个小节，用两种 Bellman 方程把 3 状态 MDP 的价值**迭代到收敛**，并与解析解对照 | `python 01_mdp_and_bellman.py` |
| [`02_q_learning_frozenlake.py`](02_q_learning_frozenlake.py) | 第 5 周：实现 Q-learning 打通 FrozenLake，理解 ε-greedy 探索 | 第一个「会学习」的 RL 算法：表格 Q-learning（Bellman 最优方程的采样增量版）+ ε-greedy，在 FrozenLake-v1 上学出最优策略地图 | `python 02_q_learning_frozenlake.py` |
| [`03_dqn_cartpole.py`](03_dqn_cartpole.py) | 第 6 周：DQN 原理（replay / target net）；第 7 周：从零实现调通；第 8 周：调参稳定到 500 步 | **主产出 / 硬检查点**：从零 DQN（Q 网络 + 经验回放 + 目标网络 + ε-greedy + 四步循环），打通 CartPole-v1；带两个消融开关做对照 | `python 03_dqn_cartpole.py --smoke` 然后 `python 03_dqn_cartpole.py` |

建议顺序：`01` 懂建模 → `02` 写最简算法 → `03` 上深度 RL 攻检查点。三个脚本**各自独立可运行**（不跨文件 import，各自复制所需小工具）。

---

## 环境搭建

代码**设备无关**：CUDA（RTX 5090 机器）/ Apple MPS（本机 Mac）/ CPU 都能跑，会自动挑可用后端。
PyTorch 目前最高支持到 Python 3.12，**不要用系统 3.14**（无 torch wheel），用下面的 3.11 环境。

本课只用两个纯 Python 环境：`CartPole-v1`（classic_control）与 `FrozenLake-v1`（toy_text），
**无需 pygame / 不做 GUI 渲染**（headless 友好，评估用数值回报）。训练曲线打印 + 存 CSV，不引入 matplotlib。

### 方式 A · uv（推荐，本目录已用它建好 `.venv`）

```bash
cd 301-microduck/learning_RL/session_02
uv venv --python 3.11            # 建虚拟环境（若已存在可跳过）
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 方式 B · pip

```bash
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

验证环境就绪：

```bash
python -c "import torch, gymnasium; print(torch.__version__, gymnasium.__version__)"
```

---

## 快速开始

```bash
source .venv/bin/activate

# ① MDP + Bellman（秒级跑完，无需联网）
python 01_mdp_and_bellman.py                 # 4 个小节全跑
python 01_mdp_and_bellman.py --section 4     # 只看 Bellman 最优方程那节
python 01_mdp_and_bellman.py --gamma 0.99    # 换折扣因子看 V* 如何变化

# ② 表格 Q-learning × FrozenLake（CPU 数秒~数十秒）
python 02_q_learning_frozenlake.py --smoke   # 冒烟：少量 episode 验证流程
python 02_q_learning_frozenlake.py           # 默认 3000 episode，学出最优策略

# ③ 从零 DQN × CartPole（主产出）
python 03_dqn_cartpole.py --smoke            # 冒烟：极少步数验证 replay/target/训练循环无报错
python 03_dqn_cartpole.py                    # 正式训练：默认 12 万步，冲 500（CPU 约 1–2 分钟）

# ④ 消融实验（体会 replay / target 为何重要，见下方「消融」）
python 03_dqn_cartpole.py --no-replay        # 关掉经验回放，看它多不稳
python 03_dqn_cartpole.py --no-target-net    # 关掉目标网络，看自举震荡
```

常用参数：
`01` — `--section {1..4}`、`--gamma`；
`02` — `--episodes`、`--alpha`、`--gamma`、`--epsilon-decay`、`--eval-episodes`、`--seed`、`--smoke`；
`03` — `--total-timesteps`、`--seed`、`--device {auto,cpu,cuda,mps}`、`--eval-episodes`、`--smoke`、`--no-replay`、`--no-target-net`、`--out`、`--curve`。

---

## ✅ 阶段 1 检查点（roadmap 三道硬门槛）

| # | 硬门槛 | 怎么验 |
|---|---|---|
| 1 | **白板上推导 Bellman 期望方程** | 跑 `01`，对照 §3 打印的迭代过程能讲清 `V(s)=Σ_a π(a\|s) Σ_s' P(s'\|s,a)[r+γV(s')]`；§4 的迭代 V\* 与解析解一致即数值真值 |
| 2 | **DQN 让 CartPole-v1 稳定撑到 500 步** | 跑 `03`（完整），看到 `✅ 阶段 1 硬检查点达成…（≥475 即视为 solved）`，`dqn_cartpole_best.pt` 里 `eval_return=500.0` |
| 3 | **能口述为什么需要 experience replay 和 target network** | 跑 `03` 的两个消融开关，对比默认版曲线，讲清各自被关掉后为何不稳（见「消融」） |

**本机实测（seed=1, CPU, 12 万步）**：

```
step  65000 | 评估平均回报=500.0   ← 首次撑满，存 best checkpoint
step  75000 | 500.0    step 80000 | 500.0    step 90000 | 500.0 …（多次达 500）
step 120000 | 107.3                ← 末期又崩了（DQN 典型不稳定）
最高评估回报 = 500.0 / 500  →  ✅ 阶段 1 硬检查点达成
```

> ⚠️ **教学点：为什么存 best 而不是 last？** 上面曲线里 DQN 反复冲到 500，末期却崩回 107.3——这是 value-based RL 的经典不稳定现象。
> 所以我们**只在评估回报刷新最高时**才 `torch.save`（`03` 的 `if ev > best_eval`），存的是历史最佳权重，不是最后一版。这一点与 session_01「存 best checkpoint」一脉相承，在 RL 里更是**必须**。

**自测**（能答上来说明真懂了，否则回看对应文件的注释）：

1. MDP 五元组是哪五个？「马尔可夫性」是什么意思，为何是 RL 的理论前提？（见 `01` §1）
2. 折扣因子 γ 调大 / 调小分别让智能体更「远视」还是「短视」？为何实践中 γ<1？（见 `01` §2）
3. Bellman **期望**方程与**最优**方程的唯一区别是什么？各自解决什么问题（评估 vs 求最优）？（见 `01` §3/§4）
4. Q-learning 更新式 `Q←Q+α[r+γ·max Q(s')−Q]` 里，哪部分是 TD target、哪部分是 TD error？为何说它是 Bellman 最优方程的「采样版」？（见 `02`）
5. DQN 为何要用**神经网络**代替 Q 表？`terminated` 与 `truncated` 在算 TD 目标时为何要区别对待？（见 `03` 文件头 + `dqn_update` 注释）

---

## 🧪 消融实验：亲眼看到 replay / target 为何重要（检查点 #3）

`03` 内置两个开关，直接服务「能口述为什么需要 experience replay 和 target network」。三条命令对照跑：

```bash
python 03_dqn_cartpole.py                                             # ① 默认（两者都在）→ 能到 500
python 03_dqn_cartpole.py --no-replay    --curve /tmp/noreplay.csv    # ② 关经验回放
python 03_dqn_cartpole.py --no-target-net --curve /tmp/notarget.csv   # ③ 关目标网络
```

| 消融 | 关掉了什么 | 预期现象 | 印证的道理 |
|---|---|---|---|
| `--no-replay` | 只用**当前这一条** transition 训练，不随机采样 buffer | 回报剧烈抖动、难以稳定爬升，常卡在低位 | 相邻样本高度相关，破坏 SGD 的 i.i.d. 假设；replay **打破相关性** + 让一条经验被反复利用（提高样本效率） |
| `--no-target-net` | TD 目标 `r+γ·maxQ(s')` 用**正在更新的在线网络自己**算 | 自举震荡：追一个「自己在动的靶子」，loss 与回报发散 / 反复崩 | target network **冻结目标一阵子**（每 `target_update_interval` 步才硬同步），让回归目标稳定 |

> 消融结果对随机种子 / 步数敏感，重点是**对比默认版**看趋势差异，而非某个具体数字。

**本机实测（6 万步，各跑一次）**：

- `--no-replay`：eval_return 剧烈抖动、撑不住——`…→500(step30k)→184→149→266→204→**19.7**(step55k)→116`，刚冲到 500 立刻崩回 19.7，始终无法稳定。
- `--no-target-net`：**loss 直接爆炸** 3~4 个数量级——`70 → 4073 → 31595 → 34025 → …`（默认版 loss 仅 0.1~9），正是「追自己在动的靶子」的自举发散。
- 默认版（两者都在）：loss 稳定在小量级，eval 反复冲到 500（见上方检查点实测）。

---

## 📚 本课知识点汇总

> 按「问题建模 → 最简算法 → 深度 RL」的顺序编排。带 🔗 的是「这个零件未来在 RL / microduck 里对应什么」。

### 1. MDP 五元组 (S, A, P, R, γ)——RL 的标准问题建模（`01` §1）

- **S** 状态集合、**A** 动作集合、**P(s'|s,a)** 转移概率、**R(s,a)** 奖励函数、**γ** 折扣因子。
- **马尔可夫性**：下一个状态只取决于 (当前状态, 动作)，与「怎么走到这里」无关——这是 RL「只盯当前 obs 做决策」的理论前提。
- 🔗 microduck 的 velocity 任务本质就是一个 MDP：**S**=61 维观测（关节 / 角速度 / 指令），**A**=关节目标，**R**=`mdp.py` 里设计的奖励项，**P**=MuJoCo 物理仿真的状态转移，**γ**=训练配置里的 discount。

### 2. return 与折扣 γ——未来奖励为什么要打折（`01` §2）

- **return** `G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + … = Σ γ^k·r_{t+k}`：把未来奖励按 γ 逐期打折后求和。
- γ 越小越**短视**（只看眼前），越大越**远视**（重视长期）；γ=1.0 不打折，无限长任务里 return 可能发散，所以实践中 γ<1（如 0.99）。

### 3. V(s) / Q(s,a) 与两种 Bellman 方程（`01` §3/§4）——几乎所有 RL 算法的内核

- **V(s)**：状态价值（在这个状态、按策略走，未来能拿多少折扣回报）；**Q(s,a)**：动作价值（在 s 先做 a，之后按策略走）。
- **Bellman 期望方程**（policy evaluation，评估给定策略 π）：
  `V(s) = Σ_a π(a|s) · Σ_s' P(s'|s,a) · [R(s,a) + γ·V(s')]`
- **Bellman 最优方程**（value iteration，求最优 V\*）：把上式的「按 π 求期望 Σ_a π(a|s)」换成「**取 max_a**」——`max` 体现了「每个状态都挑当下最好的动作」。
- **迭代式 backup**：从全 0 的 V 出发反复套用方程，V 会**收敛**到真值（`01` §4 迭代结果与解析解 `V*=[9,10,10]`（γ=0.9）完全一致）。
- **policy evaluation → policy improvement** 的循环，就是几乎所有 RL 算法的内核。

### 4. 表格 Q-learning——第一个「会学习」的 RL 算法（`02`）

- 真实 RL 里 **P / R 未知**，智能体只能**试错**：一边走一边用采样到的 `(s,a,r,s')` 逼近 Bellman 最优方程。
- **更新式（off-policy TD control）**：`Q[s,a] ← Q[s,a] + α·[ r + γ·max_a' Q[s',a'] − Q[s,a] ]`
  - `r + γ·max Q[s']` = **TD target**（时序差分目标）；`TD target − Q[s,a]` = **TD error**；α = 学习率（步长）。
  - 它就是 Bellman 最优方程的「**采样增量版**」：不遍历 P，而是用一条真实经验去挪一小步。
- **terminated 时** bootstrap 项置 0（真终态无未来价值）；`is_slippery=False` 关掉冰面滑动 → 确定性转移，表格法可靠收敛。
- **结果**：本机 3000 episode 贪心成功率 **100%**，并打印出「每格该往哪走」的最优策略箭头地图。

### 5. ε-greedy：探索 vs 利用（`02` / `03` 共用）

- **利用(exploitation)**：选 Q 最大的动作 → 用已知最好的知识拿奖励；**探索(exploration)**：以 ε 概率随机试别的动作 → 万一有更好 / 还没发现的路径。
- 训练早期 ε 大（多探索），后期**衰减**到很小（多利用）：`02` 用指数衰减（每 episode ×decay），`03` 用线性衰减（前 `exploration_fraction` 步数内从 1.0 降到 0.05）。
- 🔗 阶段 2 的 PPO 用**策略网络采样动作**来探索（`Categorical.sample()`），是「随机性内建在策略里」的另一条路线。

### 6. DQN 三大件：为什么需要它们（`03`）——本课核心

- **为什么要神经网络？** FrozenLake 只有 16 个离散状态，一张 16×4 表就装得下；但 CartPole 状态是**连续**的 4 维向量（小车位置/速度、杆角度/角速度），状态无穷多，表格存不下 → 用 **Q 网络 `Q(s,a;θ)` 逼近**那张存不下的表。
- **Q 网络**：MLP `4→128→128→2`，ReLU，输出各动作的 Q 值（**回归**价值，末尾**不加 softmax**——这点和 session_01 分类网络不同）。
- **① Experience Replay（经验回放）**：把经历存进 buffer，训练时随机抽样。
  - **为什么**：相邻样本高度相关，直接训练破坏 SGD 的 i.i.d. 假设 → 震荡/发散；随机抽样**打破相关性**，还能让一条经验被反复利用（提高样本效率）。
- **② Target Network（目标网络）**：算 TD 目标时用一份「冻结的旧网络」，每隔若干步才硬同步。
  - **为什么**：TD 目标 `r+γ·maxQ(s')` 里也含网络参数，若用正在更新的同一网络，就成了「**追一个自己在动的靶子**」→ 自举震荡。冻结目标 = 把靶子固定一阵子。
- **③ terminated vs truncated（bootstrap mask）**：算 TD 目标时用 `(1 - terminated)` 而非 `(1 - done)`。
  - CartPole 撑满 500 步是 **truncated（超时截断）**，并非真终态，仍应对它 bootstrap；只有杆倒了 / 出界才是 **terminated（真终态）**，无未来价值。初学者极易搞错。
- **四步循环**：`optimizer.zero_grad() → loss=F.mse_loss(pred_q, td_target) → loss.backward() → optimizer.step()`——**原样复用 session_01 的训练骨架**，只是 loss 从「分类交叉熵」换成「回归 TD 目标的 MSE」。

### 7. value-based vs policy-based（承前启后）

- **DQN 是 value-based**：学的是「每个动作的价值 Q」，策略是**间接**的（贪心选 Q 最大的动作）。
- 阶段 2 的 **PPO 是 policy-based**：直接学「策略网络输出动作分布」，用策略梯度更新。
- 但**稳定性思想一脉相承**：四步循环、replay buffer → rollout buffer、target net 的「固定目标」思路，最终对应 microduck_rl 的 `rsl_rl` PPO。

### 8. 🔗 零件 → RL 映射总表

| session_02 学的 | 未来 RL / microduck 里变成 |
|---|---|
| MDP 五元组 (S,A,P,R,γ) | velocity 任务：61 维 obs / 关节目标 / `mdp.py` 奖励 / MuJoCo 转移 / discount |
| return `Σ γ^k r` | 折扣累计回报（GAE 里的 return 基础） |
| Bellman 期望 / 最优方程 | 所有 value-based 算法的方程内核；PPO 的 critic 也在估 V |
| Q-table | ★ **Q-network**（状态多到存不下时的逼近） |
| ε-greedy 探索 | 策略采样探索（PPO 的 `Categorical.sample()`） |
| Experience Replay buffer | ★ **Rollout buffer**（PPO 按轨迹收集，而非随机采样） |
| Target network（固定目标） | 稳定性 trick：PPO 的 old policy / value clip 是同类思想 |
| 四步循环 zero/fwd/bwd/step | ★ **PPO 训练循环（原样）** |
| best checkpoint（存最佳非最后） | RL 曲线会震荡甚至崩 → 保存最佳策略 `model_*.pt` 尤其重要 |

---

## 下一站

阶段 1 通过后，进入 [`learning_roadmap.md`](../../docs/learning_roadmap.md) **阶段 2**：
**手写 PPO**（policy gradient → actor-critic → GAE → PPO-clip），把本课的 value-based 经验迁移到 policy-based，
最终对齐 microduck_rl 的 `rsl_rl` PPO 实现。届时会新建 `session_03/`。

---

## 目录约定

- `qtable_frozenlake.npy`：`02` 学出的 Q 表（成功率 ≥90% 时保存，已在 `.gitignore` 忽略）。
- `dqn_cartpole_best.pt`：`03` 评估回报最高时保存的权重（含 `model_state / arch / eval_return / step`，忽略）。
- `dqn_cartpole_curve.csv`：`03` 的训练曲线（step / epsilon / eval_return / loss，忽略）。
- `.venv/`：虚拟环境（忽略）。
- 代码风格：中文讲解、保留英文术语、每节带 ✏️ 试一试、处处标 🔗 RL 联系、设备无关、脚本各自独立可运行——与 session_01、`docs/` 学习笔记一致。
