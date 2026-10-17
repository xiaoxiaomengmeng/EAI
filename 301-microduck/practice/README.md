# practice/ · 本地复现（动手侧）

> RL 类文档的**两半之一**：另一半是[「总结文档」](../docs/rl/README.md)（`00→07` 认知阶梯 + 原理笔记），负责「读懂」；本目录负责「**跑通**」——在本机（RTX 5090 / py3.12，或任意 CPU/Mac）亲手把代码运行起来。
>
> 上级入口：[`docs/rl/README.md`](../docs/rl/README.md)。

---

## 这里放什么

| 子目录 / 内容 | 定位 | 状态 |
|---|---|---|
| [`fundamentals/`](fundamentals/) | **通用 RL 基础课**（与 microduck 无关的前置动手）：从零实现，逐个打通硬检查点 | ✅ session_01 / session_02 已就位 |
| [`tasks/ball_kick/`](tasks/ball_kick/README.md) | **复现 microduck 已有任务**：以「踢球看能踢多高」为脊柱，在 practice 侧新建高踢变体（不改子模块），走通 注册 → reward 设计 → PPO 训练 → 回放 全链路 | ✅ 链路已通（阶梯 1）；训练调参进行中 |

> 「复现 microduck 已有任务」的**命令与流程权威**在总结侧的 [`01-quickstart/desktop_quickstart.md`](../docs/rl/01-quickstart/desktop_quickstart.md) 与 [`01-quickstart/sim_deploy_rl_reproduce_guide.md`](../docs/rl/01-quickstart/sim_deploy_rl_reproduce_guide.md)；本目录存放你**照着跑之后的产物、笔记与踩坑记录**（按 `NN-主题` 或按任务名归档）。

---

## fundamentals/ · 通用 RL 基础课

对应 [`learning_roadmap.md`](../docs/rl/learning_roadmap.md) 的**阶段 0–1**，是读懂 microduck_rl（阶段 2 起）之前的必备地基。每个 session 各自独立可运行、自带 README + `requirements.txt`。

| session | 对应 roadmap | 内容 | 硬检查点 |
|---|---|---|---|
| [`fundamentals/session_01/`](fundamentals/session_01/README.md) | 阶段 0（第 1–2 周） | PyTorch 地基：张量 / autograd / 四步循环 / MLP 训 MNIST | MLP 把 MNIST 训到 >97% |
| [`fundamentals/session_02/`](fundamentals/session_02/README.md) | 阶段 1（第 3–8 周） | MDP + Bellman → 表格 Q-learning(FrozenLake) → 从零 DQN(CartPole) | DQN 让 CartPole-v1 稳定撑到 500 步 |

### 环境（每个 session 独立，Python 3.11）

```bash
cd practice/fundamentals/session_02   # 或 session_01
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -r requirements.txt
python 01_mdp_and_bellman.py                   # 各脚本从上到下直接运行，详见该 session 的 README
```

> 代码**设备无关**：CUDA / Apple MPS / CPU 都能跑，自动挑后端。不要用系统 3.14（无 torch wheel）。

---

## 约定

- **总结 vs 复现物理分开**：理解型 markdown 一律放[上级 `docs/rl/`](../docs/rl/README.md) 的编号阶梯，本目录只放**动手型**内容（可运行代码 + 复现产物/记录）。
- 新文档命名 `NN-主题.md`；已有文档不改名以免断链。
- 各 session 的训练产物（`.pt` / `.npy` / `.csv` / `.venv/`）已被其自带 `.gitignore` 忽略，不入库。
