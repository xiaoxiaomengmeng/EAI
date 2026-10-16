"""
session_01 · 第 1 周：PyTorch 张量操作 + autograd + nn.Module + 优化器
=====================================================================

对应 learning_roadmap.md 阶段 0 的「产出：一个能跑的 MLP 脚本」的前半程——
先把 PyTorch 的四块地基打牢，下一节 (02_mlp_mnist.py) 就能专心搭 MLP 训 MNIST。

本文件**从上到下直接运行即可**，每一节都会打印结果 + 讲解：

    python 01_tensors_autograd.py            # 跑全部小节
    python 01_tensors_autograd.py --section 7  # 只看第 7 节（autograd）

学习建议：不要只「看输出」，动手改数字再跑一遍——roadmap 的心法第 1 条：
「动手 >> 看理论」。每节末尾的「✏️ 试一试」就是给你的练习。

四块地基与本文件的对应：
  ① 张量操作      → §1–§5（创建 / 索引 / 变形 / 广播 / 矩阵乘法·期望）
  ② autograd      → §6–§7（自动求导 / 链式法则）
  ③ nn.Module     → §8（用类搭一个网络）
  ④ 优化器        → §9（手动梯度下降 vs torch.optim）
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# 工具：设备选择（cuda / Apple mps / cpu 三选一，代码设备无关）
# --------------------------------------------------------------------------- #
def get_device() -> torch.device:
    """挑一个可用的加速后端。

    - NVIDIA GPU（如仓库另一台 RTX 5090 机器）→ 'cuda'
    - Apple Silicon（本机 Mac）→ 'mps'
    - 都没有 → 'cpu'（本文件所有例子都足够小，CPU 秒级跑完）
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def header(n: int, title: str) -> None:
    print("\n" + "=" * 72)
    print(f"§{n}  {title}")
    print("=" * 72)


# --------------------------------------------------------------------------- #
# §1 张量创建：PyTorch 里一切数据都是 tensor（带 shape / dtype / device 三属性）
# --------------------------------------------------------------------------- #
def section1_create() -> None:
    header(1, "张量创建")

    a = torch.tensor([[1, 2, 3], [4, 5, 6]])   # 从 Python 嵌套列表创建
    print("从列表创建 a =\n", a)
    print("a.shape =", tuple(a.shape), "| a.dtype =", a.dtype, "| a.device =", a.device)

    print("\nzeros(2,3) =\n", torch.zeros(2, 3))          # 全 0
    print("ones(2,3) =\n", torch.ones(2, 3))              # 全 1
    print("eye(3)（单位阵）=\n", torch.eye(3))            # 对角 1
    print("arange(0,10,2) =", torch.arange(0, 10, 2))     # 等差序列 [0,2,4,6,8]
    print("linspace(0,1,5) =", torch.linspace(0, 1, 5))   # 等间隔 5 个点

    torch.manual_seed(0)                                   # 固定随机种子 → 结果可复现
    print("\nrand(2,3)（均匀分布 [0,1)）=\n", torch.rand(2, 3))
    print("randn(2,3)（标准正态 N(0,1)）=\n", torch.randn(2, 3))
    print("randint(0,10,(2,3))（整数 [0,10)）=\n", torch.randint(0, 10, (2, 3)))

    # ✏️ 试一试：把 randn(2,3) 改成 randn(4,5)，观察 shape 变化；
    #            注释掉 manual_seed(0) 再跑两次，看随机数是否每次不同。


# --------------------------------------------------------------------------- #
# §2 索引与切片：和 NumPy 几乎一致（RL 里取 obs 的某几维全靠它）
# --------------------------------------------------------------------------- #
def section2_indexing() -> None:
    header(2, "索引与切片")

    x = torch.arange(12).reshape(3, 4)   # 3 行 4 列，元素 0..11
    print("x =\n", x)
    print("x[1]      （第 1 行）        =", x[1])
    print("x[1, 2]   （第1行第2列标量） =", x[1, 2].item())
    print("x[:, 0]   （所有行的第 0 列）=", x[:, 0])
    print("x[0:2, 1:3]（子块）=\n", x[0:2, 1:3])
    print("x[x > 5]  （布尔掩码筛选）  =", x[x > 5])

    # 联系 roadmap：61 维观测里取「关节位置 [6:20]」就是这种切片
    obs = torch.randn(61)
    print("\nobs[6:20]（模拟取 14 维关节位置）shape =", tuple(obs[6:20].shape))
    # ✏️ 试一试：把布尔掩码改成 x[x % 2 == 0]，取出所有偶数。


# --------------------------------------------------------------------------- #
# §3 形状变换：view/reshape、transpose、unsqueeze——RL 里对齐 batch 维天天用
# --------------------------------------------------------------------------- #
def section3_reshape() -> None:
    header(3, "形状变换")

    x = torch.arange(6)                       # shape (6,)
    print("原始 x.shape =", tuple(x.shape))
    print("x.view(2,3) =\n", x.view(2, 3))      # 变形成 2×3（要求内存连续）
    print("x.reshape(3,2) =\n", x.reshape(3, 2))  # reshape 更宽容（必要时自动拷贝）

    m = torch.arange(6).reshape(2, 3)
    print("\nm =\n", m)
    print("m.transpose(0,1)（转置 2×3 → 3×2）=\n", m.transpose(0, 1))
    print("m.T（二维转置的简写）=\n", m.T)

    b = torch.randn(4)                         # 一个长度 4 的向量
    print("\nb.shape =", tuple(b.shape))
    print("b.unsqueeze(0).shape =", tuple(b.unsqueeze(0).shape), "  # 前面加一维 → (1,4)，常用来补 batch 维")
    print("b.unsqueeze(1).shape =", tuple(b.unsqueeze(1).shape), "  # 中间加一维 → (4,1)")
    c = torch.randn(1, 4)
    print("c.squeeze(0).shape   =", tuple(c.squeeze(0).shape), "  # 去掉大小为 1 的维 → (4,)")

    # -1 是「占位符」：让 PyTorch 自己算这一维多大
    print("\nx.view(3, -1).shape =", tuple(x.view(3, -1).shape), "  # 3×? → ? 自动算成 2")
    # ✏️ 试一试：把 28×28 的图片展平成 784 维向量（下一节的 MLP 就靠它）：
    #            img = torch.randn(1, 28, 28); print(img.flatten().shape)


# --------------------------------------------------------------------------- #
# §4 广播 broadcasting：shape 不同也能运算的规则（省去大量手写循环）
# --------------------------------------------------------------------------- #
def section4_broadcasting() -> None:
    header(4, "广播 broadcasting")

    m = torch.ones(3, 4)          # (3,4)
    v = torch.arange(4)           # (4,)  →  广播成 (3,4)：每一行都加 [0,1,2,3]
    print("m (3,4) + v (4,) =\n", m + v)

    col = torch.tensor([[10], [20], [30]])   # (3,1) → 广播成 (3,4)：每一列都加
    print("\n(3,4) + (3,1) =\n", m + col)

    # 这正是「图像归一化」的底层机制：一张 (3,H,W) 图减一个 (3,1,1) 的均值向量
    img = torch.rand(3, 2, 2)               # 3 通道的小图
    mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    print("\n图像归一化 (img-mean).shape =", tuple((img - mean).shape), "（通道各自减 0.5）")
    # ✏️ 试一试：把 v 改成 torch.arange(3).view(3,1)，看广播如何按列进行。


# --------------------------------------------------------------------------- #
# §5 矩阵乘法 · 期望 · 归约：对应 roadmap 第 2 周「够用数学」
# --------------------------------------------------------------------------- #
def section5_math() -> None:
    header(5, "矩阵乘法 / 期望 / 归约")

    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])   # (2,2)
    B = torch.tensor([[5.0, 6.0], [7.0, 8.0]])   # (2,2)
    print("A @ B（矩阵乘法，等价 torch.matmul）=\n", A @ B)
    print("A * B（逐元素相乘，完全不同的东西！）=\n", A * B)

    # 神经网络的一层本质就是 y = x @ W.T + b —— 一次矩阵乘法
    x = torch.randn(5, 3)        # batch=5，每个样本 3 维
    W = torch.randn(4, 3)        # 把 3 维映射到 4 维
    b = torch.randn(4)
    y = x @ W.T + b              # (5,3)@(3,4) + (4,) → (5,4)
    print("\n线性层 x@W.T+b：", tuple(x.shape), "→", tuple(y.shape))

    # 期望 = 均值。RL 里「期望回报」就是对 return 求均值
    r = torch.tensor([1.0, 2.0, 3.0, 4.0])
    print("\n期望 E[r] = r.mean() =", r.mean().item())
    print("求和 r.sum() =", r.sum().item(), "| 标准差 r.std() ≈", round(r.std().item(), 4))

    t = torch.arange(6).reshape(2, 3).float()
    print("\nt =\n", t)
    print("t.mean()      （全局均值）   =", t.mean().item())
    print("t.mean(dim=0) （按列/沿第0维）=", t.mean(dim=0))   # 结果 shape (3,)
    print("t.mean(dim=1) （按行/沿第1维）=", t.mean(dim=1))   # 结果 shape (2,)
    # ✏️ 试一试：dim=0 和 dim=1 的结果 shape 为什么不同？动手改 t 的尺寸验证。


# --------------------------------------------------------------------------- #
# §6 autograd（一）：自动求导——PyTorch 帮你算梯度，这是深度学习的引擎
# --------------------------------------------------------------------------- #
def section6_autograd_basic() -> None:
    header(6, "autograd 自动求导")

    # y = x^2 + 3x + 1，导数 dy/dx = 2x + 3；在 x=2 处应为 2*2+3 = 7
    x = torch.tensor(2.0, requires_grad=True)   # requires_grad=True → 开始追踪它
    y = x**2 + 3 * x + 1
    y.backward()                                 # 反向传播：把 dy/dx 算进 x.grad
    print(f"y = x^2 + 3x + 1 在 x=2 处：")
    print(f"  手算导数 2x+3 = {2 * 2 + 3}")
    print(f"  autograd x.grad = {x.grad.item()}   ← 两者一致，说明求导正确")

    # 多变量：f = u*v + u^2，∂f/∂u = v + 2u，∂f/∂v = u
    u = torch.tensor(1.0, requires_grad=True)
    v = torch.tensor(3.0, requires_grad=True)
    f = u * v + u**2
    f.backward()
    print(f"\nf = u*v + u^2 在 (u=1,v=3)：")
    print(f"  ∂f/∂u = v+2u = {3 + 2 * 1}，autograd u.grad = {u.grad.item()}")
    print(f"  ∂f/∂v = u    = {1}，autograd v.grad = {v.grad.item()}")
    # ✏️ 试一试：把 x 改成 5.0，先心算 2x+3，再跑验证 autograd 是否吻合。


# --------------------------------------------------------------------------- #
# §7 autograd（二）：链式法则 + 梯度会累积（两个初学者最容易踩的点）
# --------------------------------------------------------------------------- #
def section7_chain_rule() -> None:
    header(7, "autograd 链式法则 & 梯度累积")

    # 链式法则：z = (2x)^3，dz/dx = 3*(2x)^2 * 2 = 24x^2；x=1 → 24
    x = torch.tensor(1.0, requires_grad=True)
    t = 2 * x            # 中间量 t
    z = t**3             # z = t^3
    z.backward()
    print("z = (2x)^3 在 x=1：手算 24x^2 =", 24 * 1**2, "| autograd x.grad =", x.grad.item())

    # 坑点：grad 是「累加」的，不是「覆盖」的！每次 backward 前要清零
    w = torch.tensor(1.0, requires_grad=True)
    for i in range(3):
        loss = w * 2          # dloss/dw = 2
        loss.backward()       # 每次都会把 2 累加进 w.grad
        print(f"  第 {i} 次 backward 后 w.grad = {w.grad.item()}（未清零 → 2,4,6 累加）")
        # 正确做法：w.grad.zero_()  —— 训练循环里 optimizer.zero_grad() 就是干这个
    w.grad.zero_()
    (w * 2).backward()
    print("  清零后 w.grad =", w.grad.item(), "← 回到干净的 2")

    # detach / no_grad：推理时不需要梯度，省显存也更快（部署 ONNX 前常用）
    a = torch.tensor(3.0, requires_grad=True)
    b = a * 2
    with torch.no_grad():
        c = a * 2
    print("\nno_grad 下算出的 c.requires_grad =", c.requires_grad, "（False：不建计算图）")
    print("b.detach() 会切断与计算图的联系，常用于取中间结果做别的计算")
    # ✏️ 试一试：删掉循环里的累加演示，改成每次 loss.backward() 前 w.grad.zero_()，
    #            观察 w.grad 是否恒为 2。


# --------------------------------------------------------------------------- #
# §8 nn.Module：用「类」搭一个网络（MLP = 多层感知机）
# --------------------------------------------------------------------------- #
class TinyMLP(nn.Module):
    """一个最小的多层感知机：input_dim → hidden → output_dim。

    nn.Module 是 PyTorch 所有网络的基类。你只需：
      1) 在 __init__ 里声明有哪些层（参数在这里被创建和注册）；
      2) 在 forward 里写数据怎么流过这些层。
    反向传播、梯度管理由 PyTorch 自动完成——你从不手写 backward 里的链式法则。
    """

    def __init__(self, input_dim: int = 1, hidden_dim: int = 16, output_dim: int = 1):
        super().__init__()                       # 必须先调用父类构造
        self.fc1 = nn.Linear(input_dim, hidden_dim)   # 全连接层 1
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # 全连接层 2
        self.act = nn.ReLU()                     # 激活函数：引入非线性

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.fc1(x))                # 第一层 + 非线性
        x = self.fc2(x)                          # 输出层（回归任务不加激活）
        return x


def section8_module() -> None:
    header(8, "nn.Module 搭网络")

    model = TinyMLP(input_dim=1, hidden_dim=16, output_dim=1)
    print("模型结构：\n", model)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n可训练参数总数 = {n_params}")
    for name, p in model.named_parameters():
        print(f"  {name:12s} shape = {tuple(p.shape)}")

    x = torch.tensor([[0.5]])
    y = model(x)                                  # 前向：直接「调用」model 即可
    print("\n前向 model([[0.5]]) =", y.detach().numpy().ravel())
    print("（等价于 model.forward(x)，但推荐写 model(x)，会走 hook 等机制）")
    # ✏️ 试一试：把 hidden_dim 从 16 改成 64，看参数总数如何变化。


# --------------------------------------------------------------------------- #
# §9 优化器：手动梯度下降 vs torch.optim（对应 roadmap「优化器」）
# --------------------------------------------------------------------------- #
def section9_optimizer(device: torch.device) -> None:
    header(9, "优化器：手动 GD vs torch.optim")

    # 任务：拟合 y = 2x + 1（学出 w≈2, b≈1）
    torch.manual_seed(0)
    X = torch.linspace(-1, 1, 50).unsqueeze(1)    # (50,1)
    Y = 2 * X + 1

    # --- (a) 手动梯度下降：亲手更新参数，理解优化器到底在做什么 ---
    w = torch.randn(1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    lr = 0.1
    for step in range(100):
        pred = w * X + b
        loss = ((pred - Y) ** 2).mean()           # 均方误差 MSE
        loss.backward()                           # 算梯度（累加进 w.grad/b.grad）
        with torch.no_grad():                     # 更新参数时不要建计算图
            w -= lr * w.grad                       # 核心：参数 ← 参数 - 学习率×梯度
            b -= lr * b.grad
            w.grad.zero_()                         # 清零，否则下次 backward 会累加
            b.grad.zero_()
    print(f"(a) 手动 GD 100 步：w = {w.item():.3f}（目标 2），b = {b.item():.3f}（目标 1）")

    # --- (b) 用 torch.optim：把上面手写的更新逻辑交给优化器 ---
    model = TinyMLP(input_dim=1, hidden_dim=16, output_dim=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)   # Adam：自适应学习率
    loss_fn = nn.MSELoss()
    Xd, Yd = X.to(device), Y.to(device)
    for step in range(300):
        optimizer.zero_grad()          # ① 清零梯度（= 手动的 w.grad.zero_()）
        loss = loss_fn(model(Xd), Yd)  # ② 前向 + 算 loss
        loss.backward()                # ③ 反向求梯度
        optimizer.step()               # ④ 更新参数（= 手动的 w -= lr*grad）
    print(f"(b) torch.optim.Adam 300 步：最终 loss = {loss.item():.5f}")
    print("    四步循环 zero_grad→forward→backward→step 就是所有 PyTorch 训练的骨架，")
    print("    下一节 02_mlp_mnist.py 训练 MNIST 用的是完全一样的循环。")
    # ✏️ 试一试：把 Adam 换成 torch.optim.SGD(model.parameters(), lr=0.05)，
    #            对比收敛速度；再把 lr 调成 10，观察 loss 是否发散（NaN）。


# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="session_01 §1 PyTorch 地基演示")
    parser.add_argument("--section", type=int, default=None,
                        help="只运行指定小节编号（1–9），默认全跑")
    args = parser.parse_args()

    device = get_device()
    print(f"PyTorch {torch.__version__} | 使用设备：{device}")

    sections = {
        1: section1_create,
        2: section2_indexing,
        3: section3_reshape,
        4: section4_broadcasting,
        5: section5_math,
        6: section6_autograd_basic,
        7: section7_chain_rule,
        8: section8_module,
        9: lambda: section9_optimizer(device),
    }
    targets = [args.section] if args.section else list(sections)
    for n in targets:
        sections[n]()

    print("\n" + "=" * 72)
    print("✅ §1 全部跑完。四块地基（张量/autograd/nn.Module/优化器）已就位。")
    print("   下一步：python 02_mlp_mnist.py —— 用这些搭 MLP 训练 MNIST，冲击 >97%。")
    print("=" * 72)


if __name__ == "__main__":
    main()
