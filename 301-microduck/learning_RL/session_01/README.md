# session_01 · 前置基础：PyTorch 地基 + MLP 训 MNIST

> 对应 [`learning_roadmap.md`](../../docs/learning_roadmap.md) **阶段 0（第 1–2 周）** 的动手产出。
> 目标：扫清「PyTorch 不会用 / 梯度是什么」的障碍，并把一个 MLP 在 MNIST 上训到 **>97%**——通过阶段 0 的硬检查点。

这是整条 RL 学习路线的**第一节课**。它不讲 RL，只补 RL 之前必须先会的深度学习地基：
roadmap 的心法第 1 条是「每学一个概念立刻实现一遍」，本目录就是阶段 0 那两个概念的**可运行实现**。

---

## 文件地图

| 文件 | 对应 roadmap | 讲什么 | 怎么跑 |
|---|---|---|---|
| [`01_tensors_autograd.py`](01_tensors_autograd.py) | 第 1 周：张量操作 / autograd / `nn.Module` / 优化器；第 2 周：矩阵乘法 / 期望 / 链式法则 | 9 个小节，从张量创建到手写梯度下降，**从上到下直接运行**，每节带讲解 + ✏️ 练习 | `python 01_tensors_autograd.py` |
| [`02_mlp_mnist.py`](02_mlp_mnist.py) | 第 1 周「一个能跑的 MLP 脚本」+ 第 2 周「MNIST >97%」 | 完整训练脚本：数据→模型→损失→四步循环→评估→存最优权重 | `python 02_mlp_mnist.py --epochs 5` |
| [`03_overfit_normalization.py`](03_overfit_normalization.py) | 第 2 周「理解 overfitting / normalization」 | 两个**对照实验**：小样本过拟合（看 train/test gap）、有无归一化（看收敛速度） | `python 03_overfit_normalization.py` |
| [`04_load_and_predict.py`](04_load_and_predict.py) | 第 2 周延伸：把「训出来 → 用起来」闭环 | 加载 `02` 存的 `mlp_mnist_best.pt` → 重建网络灌权重 → `eval()`+`no_grad` 推理（ASCII 可视化 + 复核准确率）；预演 RL 策略部署 | `python 04_load_and_predict.py`（需先跑过 `02`） |

建议顺序：`01` 打地基 → `02` 出主产出 → `03` 补两个概念的直觉 → `04` 学会加载复用权重。

---

## 环境搭建

代码**设备无关**：CUDA（另一台 RTX 5090 机器）/ Apple MPS（本机 Mac）/ CPU 都能跑，
会自动挑可用后端。PyTorch 目前最高支持到 Python 3.12，**不要用系统 3.14**，用下面的 3.11 环境。

### 方式 A · uv（推荐，本目录已用它建好 `.venv`）

```bash
cd 301-microduck/learning_RL/session_01
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
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"
```

---

## 快速开始

```bash
source .venv/bin/activate

# ① 地基演示（秒级跑完，无需联网/下载）
python 01_tensors_autograd.py
python 01_tensors_autograd.py --section 7     # 只看 autograd 链式法则那节

# ② 主产出：MLP 训 MNIST（首次会自动下载 MNIST 到 ./data，约 10MB）
python 02_mlp_mnist.py --smoke                # 冒烟：小数据 1 轮，先验证流程通不通
python 02_mlp_mnist.py --epochs 5             # 正式训练，CPU/MPS 约 1–3 分钟

# ③ 概念对照实验
python 03_overfit_normalization.py --smoke    # 快速版
python 03_overfit_normalization.py            # 完整版

# ④ 加载 02 训出的权重做推理（需先跑过 02，生成 mlp_mnist_best.pt）
python 04_load_and_predict.py --n 5           # 展示前 5 张：ASCII 画 + 预测 + 复核准确率
```

常用参数：`02` — `--epochs`、`--batch-size`、`--lr`、`--device {auto,cpu,cuda,mps}`、`--data-dir`、`--out`；`04` — `--ckpt`、`--n`、`--device`。

---

## ✅ 阶段 0 检查点（roadmap 硬门槛）

跑完 `python 02_mlp_mnist.py --epochs 5`，看到：

```
最优测试准确率 = 9x.xx%
✅ 检查点达成：测试准确率 > 97%（roadmap 阶段 0 的硬门槛已通过）
```

即通过。若没到 97%：多为训练不足——`--epochs 8`，或确认 `lr=1e-3`、数据已归一化。
最优权重存于 `mlp_mnist_best.pt`。

**自测**（能答上来说明真懂了，否则回看对应文件的注释）：

1. `optimizer.zero_grad()` 为什么必须在每个 batch 开头调用？（见 `01` §7 梯度累积）
2. `nn.Linear` 的 `y = x @ W.T + b` 里，哪一步是「矩阵乘法」？（见 `01` §5）
3. MLP 输出为什么是 logit 而不是概率？谁负责做 softmax？（见 `02` 的 `MLP.forward` 注释）
4. 过拟合时 train_acc 和 test_acc 分别什么表现？gap 说明了什么？（见 `03` 实验一）
5. 归一化 `(x-mean)/std` 为什么能加快收敛？（见 `03` 实验二）

---

## 📚 本课知识点汇总（问答沉淀）

> 本节把学习 session_01 时的关键问答沉淀成可复习的知识点，按**数据流顺序**编排：
> 数据 → 训练 → 评估 → 模式开关 → checkpoint → 推理 → softmax 本质。
> 带 🔗 的是「这个零件未来在 RL 里对应什么」。

### 0. 一句话定位：session_01 是 RL 的「引擎室」

**深度强化学习 = 深度学习（用神经网络拟合策略/价值）+ 强化学习（用环境奖励来学习）**。
session_01 练的全是**前者**——它是 RL 的引擎室。认知路径是「先零件 → 再整机 → 再懂原理 → 再会复用」：

```
01 拆零件(张量/autograd/nn.Module/优化器) → 02 装整机(完整训练跑出>97%)
   → 03 懂原理(过拟合/归一化对照实验) → 04 会复用(加载checkpoint做推理)
```

之所以先在 MNIST 上练，是因为 roadmap 心法「**能跑 ≠ 正确**」：RL 没有「loss 一路降就对了」的反馈、调试极难，
所以要先在**有标准答案、能快速看到结果**的环境里，把训练循环练成肌肉记忆，将来才能把脑力集中在 RL 特有的难点上。

### 1. 数据加载 `build_loaders`（`02`）——Dataset ≠ DataLoader

| 角色 | 类比 | 职责 |
|---|---|---|
| **Dataset** | 一整本书 | 「**存**」样本：给编号 `i` 返回第 `i` 个 |
| **DataLoader** | 读书助手 | 「**取**」：按 batch 切好、打乱、并行地递给你 |

- **`transforms.Compose([ToTensor(), Normalize()])`**：每张图的预处理流水线。`ToTensor` 把 PIL 图转张量并把像素从 `[0,255]` 缩放到 `[0,1]`；`Normalize` 做 `(x-0.1307)/0.3081`。参数是**元组** `(0.1307,)`（为多通道设计，单通道也不能省逗号）。
- **`shuffle`**：训练集 `True`（每 epoch 打乱，提升泛化、让每批梯度更有代表性）；测试集 `False`（考试不打乱，且便于把预测和原样本对齐）。
- **`num_workers=0`**：不开子进程加载数据，最稳、最好调试（大数据集会调大以喂饱 GPU）。
- **`smoke` + `Subset`**：冒烟测试只取小子集，用最小代价验证「流程通不通」，而非追求准确率。
- 🔗 `Compose` → RL 的 **obs 预处理管线**；`Dataset` → **RolloutStorage**；`shuffle` 分批 → PPO 的 **minibatch 打乱**；「先 smoke 再 full」是 RL 必备工程纪律（RL 训练动辄数小时，先小规模验证再上大规模）。
- ⚠️ 深层差异：MNIST 数据是**静态下载好的**；RL 数据是**策略边跑边产的**（每走一步才产生一条经验），届时 DataLoader 的角色会变成 **Rollout/Replay Buffer**，但「按 batch 取、可能打乱、喂四步循环」的机制一样。

### 2. 训练四步循环 `train_one_epoch`（`02`）——所有 PyTorch 训练的骨架

```
① optimizer.zero_grad()  清零梯度   ② logits = model(xb)   前向
③ loss.backward()        反向求梯度 ④ optimizer.step()     更新参数(参数←参数-lr×梯度)
```

关键参数/易错点：

| 代码 | 要点 |
|---|---|
| `zero_grad()` | 梯度默认**累加**，不清零会叠加上一 batch（`01` §7 的坑） |
| `F.cross_entropy(logits, yb)` | 内部自带 softmax，默认 `reduction='mean'` 返回本批平均 loss |
| `loss.item()` | 把标量张量取成 Python float，**脱离计算图**（否则一直占显存） |
| `* xb.size(0)` | mean 乘回本批样本数 B → 还原成总和，最后按总样本求**无偏**平均（末批不满也不会偏） |
| `logits.argmax(dim=1)` | 沿「10 类」这一维取最高分**下标** = 预测数字 |
| 🔗 | 这四步**原样**就是 PPO 的训练循环骨架；`Adam` 也是 PPO 默认优化器 |

### 3. 评估逻辑 `evaluate`（`02`）——只「考试」不「学习」

| | `train_one_epoch`（上课） | `evaluate`（考试） |
|---|---|---|
| 目的 | **改**权重 | **测**水平，权重不动 |
| 梯度 | 需要 backward | 不需要 → `@torch.no_grad()` |
| 模式 | `model.train()` | `model.eval()` |
| 循环 | 四步全做 | **只有前向** |
| optimizer | 要用 | 参数里根本没有它 |

- `reduction="sum"`：直接要本批 loss 总和再累加，最后 `/seen`——与训练里「mean×B 累加」殊途同归，都是**按样本无偏加权平均**。
- 🔗 `no_grad`+`eval()` = **策略部署到真机时的纯推理状态**；「训练一轮→评估→创新高就存」= PPO 的「跑评估局统计平均回报/成功率→存最佳策略」。

### 4. `train()`/`eval()` 是「模式开关」，**不是**训练！

| 代码 | 是什么 | 调用频率 |
|---|---|---|
| `model.train()` / `model.eval()` | **设标志位**（切模式），不算数据不改参数 | 进 batch 循环**前**各一次 |
| `model(xb)`（触发 `forward`） | **真正的前向计算** | 循环**内**每个 batch 一次 |

- 开关只影响 **dropout / BatchNorm** 这类层在 forward 里的行为（dropout 训练时随机丢弃、评估时全保留；BN 训练用当前批统计、评估用累计统计）。
- 本 MLP 没有这两种层，故 `train()`/`eval()` 看似无效——但**必须养成成对切换的习惯**，否则将来网络带 BN/dropout 时会出现「准确率莫名偏低、每次结果都不一样」的隐蔽 bug。

### 5. checkpoint `mlp_mnist_best.pt`（`02` 存 / `04` 读）

存的内容（一个 dict）：

| key | 值示例 | 含义 |
|---|---|---|
| `model_state` | 6 个张量 | `state_dict()`：学到的全部权重 |
| `test_acc` | 0.9747 | 这份权重对应的测试准确率 |
| `epoch` | 4 | 在**第 4 轮**达到（不是最后一轮） |
| `arch` | MLP-784-128-64-10 | 网络结构，加载时要照着它重建 |

- **写入时机**：训练循环里 `if te_acc > best_acc:` 才 `torch.save` → 存的是**历史最佳**，不是最后一版（RL 曲线会震荡甚至崩，这点尤其重要）。
- **加载四步**（`04`）：`torch.load(weights_only=True)` 读档 → 按 `arch` 重建同构空网络 → `load_state_dict`（`strict=True` 防呆）灌权重 → `eval()`+`no_grad` 推理。
- `weights_only=True` 是安全写法：只反序列化张量与基础类型，拒绝任意 Python 对象（防止加载来路不明的 `.pt` 执行恶意代码）。
- 🔗 它就是 microduck_rl 里 **`model_*.pt`** 的极简预演；差别只在 RL 最后还会**导出 ONNX** 交给板载 Rust runtime 推理。它已在 `.gitignore` 忽略（产物，可重新生成）。

### 6. 推理三步 `logits → softmax → argmax`（`04`）

以测试集第 0 张「7」为例（真实数字）：

```
下标(类):     0      1      2      3      4      5       6      7      8      9
logits = [-3.04, -9.35, 3.02, 3.59, -9.34, -4.31, -17.81, 10.13, -2.39, -3.78]  ← 原始分数(可正可负,和≠1)
probs  = [ 0.0,   0.0,  0.0008,0.0014, 0.0,  0.0,   0.0,  0.9977, 0.0,  0.0 ]  ← softmax后概率(和=1)
pred   = 7                                                                       ← argmax取最大值的下标
shape:  (B,1,28,28) →(B,10)          →(B,10)                                  →(B,)
```

- **logits**：每类一个「原始打分」，未归一化、可正可负、和≠1，只有相对大小有意义。
- **softmax**：`exp(logit_i)/Σexp(logit_j)`，把分数翻译成「都在 0~1、和为 1」的概率；`dim=1` 表示沿类别维逐样本归一化。
- **argmax**：返回最大值的**下标**（不是值）。`max` 给值、`argmax` 给位置，分类要「哪一类」所以用 argmax。

### 7. 为什么要用 softmax（本质）——训练时就确立了「概率」语义

- **训练侧**：`F.cross_entropy` 内部 = `log_softmax` + NLL（已实测：`cross_entropy(logits,y)` 与 `-log_softmax(logits)[y]` 与 `-log(softmax(logits)[y])` 三者完全相等）。所以模型被优化成「让 `softmax(logits)` 在正确类上概率趋近 1」——**`softmax(logits)` 从训练起就是模型学会的那个概率分布**。推理要拿概率，就必须用同一个 softmax。
- **何时需要 softmax**：

| 你要什么 | 需要 softmax？ | 原因 |
|---|---|---|
| 只要预测类别（argmax） | ❌ | softmax 单调，不改变谁最大 |
| 要概率/置信度（99.8%） | ✅ | 只有 softmax 能得到和为 1 的概率 |
| 要按概率采样（RL 选动作） | ✅ | 采样必须基于真实分布 |

- **双重 softmax 的坑**（已实测）：模型末尾若加 softmax，再喂 `cross_entropy`（内部又 softmax）→ loss 从 `0.417` 变成 `0.802`，**梯度就错了**。
- **PyTorch 约定（务必记住）**：**模型输出 raw logits（不加 softmax），损失函数内部负责 softmax**；只有「推理且需要概率」时才手动补 `F.softmax`。这样设计是因为 `cross_entropy` 把 `log_softmax` 与 NLL 融合、用 log-sum-exp 技巧，**数值更稳定**（避免先 `exp` 溢出）。
- 🔗 RL 的 categorical policy 完全一样：网络输出 **action logits** 不加 softmax，交给 `Categorical(logits=...)`；只有采样动作/打印概率时才 softmax。且 RL 训练用 `Categorical(probs).sample()` **随机采样**（探索），而非 MNIST 推理的 `argmax`（贪心）。

### 8. 🔗 零件 → RL 映射总表

| session_01 学的 | 未来 RL 里变成 |
|---|---|
| 张量切片 `obs[6:20]` | 从 61 维观测取状态 |
| 矩阵乘法 `x@W.T+b` | Actor/Critic 网络的一层 |
| 期望 `.mean()` | 期望回报 E[return] |
| autograd + 链式法则 | 策略梯度反传 |
| 四步循环 zero/fwd/bwd/step | ★ PPO 训练循环（原样） |
| `Normalize` / 归一化 | obs normalizer（对抗 sim2real gap） |
| overfitting / 看 test_acc | sim2real gap / 策略泛化评估 |
| best checkpoint | 保存最佳策略 `model_*.pt` |
| `eval()`+`no_grad` 推理 | 策略部署到真机（导出 ONNX 纯推理） |
| `logits→softmax→argmax` | categorical policy `logits→softmax→sample()` |

**唯一本质区别**：MNIST 的 loss 有标准答案（标签）；RL 没有标准答案，loss 来自稀疏、带噪、延迟的**奖励**。但训练循环的机械结构完全一样——这正是阶段 0 要先练熟它的原因。

---

## 下一站

阶段 0 通过后，进入 [`learning_roadmap.md`](../../docs/learning_roadmap.md) **阶段 1（入门）**：
MDP → tabular Q-learning → 实现 DQN。届时会新建 `session_02/`。

---

## 目录约定

- `data/`：MNIST 数据集缓存（自动下载，已在 `.gitignore` 中忽略，不进版本库）。
- `mlp_mnist_best.pt`：`02` 训练出的最优权重（同样忽略）。
- `.venv/`：虚拟环境（忽略）。
- 代码风格：中文讲解、保留英文术语、每节带 ✏️ 练习、设备无关——与 `docs/` 学习笔记一致。
