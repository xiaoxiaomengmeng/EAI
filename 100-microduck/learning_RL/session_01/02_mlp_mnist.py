"""
session_01 · 第 1–2 周主产出：用 MLP 训练 MNIST，冲击 >97% 准确率
==================================================================

对应 learning_roadmap.md 阶段 0 的两个产出：
  第 1 周「一个能跑的 MLP 脚本」+ 第 2 周「MNIST 分类准确率 > 97%」。

这是你的第一个**完整深度学习训练脚本**。它把 01_tensors_autograd.py 里练过的
四块地基拼成一个真实训练流程：

    数据(DataLoader) → 模型(nn.Module MLP) → 损失(cross_entropy)
       → 四步循环(zero_grad/forward/backward/step) → 评估 → 存最优权重

怎么跑：
    python 02_mlp_mnist.py                     # 默认 5 epoch，CPU/MPS 约 1–3 分钟
    python 02_mlp_mnist.py --epochs 8          # 多训几轮，准确率更高
    python 02_mlp_mnist.py --smoke             # 冒烟测试：小数据 1 轮，验证流程通不通
    python 02_mlp_mnist.py --device cpu        # 强制用 CPU

第 2 周的数学概念在本文件里的落点：
  - 矩阵乘法    → MLP 每个 nn.Linear 就是 y = x @ W.T + b
  - 期望        → loss 是一个 batch 上误差的均值（期望的蒙特卡洛估计）
  - 链式法则    → loss.backward() 自动沿网络逐层反传梯度
  - normalization → transforms.Normalize，见下方 §为什么归一化
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# MNIST 全数据集的均值/标准差（社区公认值）。Normalize 会做 (x-mean)/std，
# 把像素从 [0,1] 拉到零均值、单位方差附近。
MNIST_MEAN, MNIST_STD = 0.1307, 0.3081


# --------------------------------------------------------------------------- #
# 为什么归一化（normalization）？—— roadmap 第 2 周要理解的概念
# --------------------------------------------------------------------------- #
# 原始像素在 [0,1]，且大部分接近 0（背景是黑的）。若直接喂进网络：
#   ① 各维尺度/分布不均 → 梯度方向偏斜 → 收敛慢甚至震荡；
#   ② 输入恒为正 → 某些激活下更新方向受限。
# Normalize((0.1307,),(0.3081,)) 把数据变成零均值、单位方差，让优化「好走」得多。
# 这正是 01 文件 §4 广播的应用：一张 (1,28,28) 图减去标量均值、除以标量标准差。
# 记住这个直觉——RL 里的 obs normalizer（microduck_rl 的 61 维观测归一化）是同一件事。


# --------------------------------------------------------------------------- #
# 模型：一个三层 MLP（多层感知机）
# --------------------------------------------------------------------------- #
class MLP(nn.Module):
    """784 → 128 → 64 → 10 的全连接网络。

    输入是展平的 28×28=784 维手写数字像素，输出 10 个 logit（对应数字 0–9）。
    为什么它能到 >97%？因为 MNIST 数字大致线性可分，两层非线性隐藏层足够。
    """

    def __init__(self, input_dim: int = 784, hidden1: int = 128, hidden2: int = 64,
                 num_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)          # (B,1,28,28) → (B,784)：展平，见 01 §3
        x = F.relu(self.fc1(x))            # 第一隐层 + ReLU 非线性
        x = F.relu(self.fc2(x))            # 第二隐层 + ReLU
        x = self.fc3(x)                    # 输出层：10 个 logit（不加激活）
        return x
    # 注意：这里输出的是「logit」而非概率。nn.cross_entropy 内部会自己做 softmax，
    # 所以模型末尾不要再加 softmax/log_softmax，否则等于做了两次，梯度会不对。


def get_device(pref: str) -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_loaders(data_dir: str, batch_size: int, smoke: bool):
    """下载 MNIST 并包成 DataLoader。smoke=True 时只取小子集快速验证流程。"""
    tf = transforms.Compose([
        transforms.ToTensor(),                                  # PIL → (1,28,28) float [0,1]
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),      # 归一化，见上方说明
    ])
    train_ds = datasets.MNIST(data_dir, train=True, download=True, transform=tf)
    test_ds = datasets.MNIST(data_dir, train=False, download=True, transform=tf)

    if smoke:                       # 冒烟：2048 训练 + 1024 测试，1 轮也能看到准确率上升
        train_ds = Subset(train_ds, range(2048))
        test_ds = Subset(test_ds, range(1024))

    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_ld = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_ld, test_ld


def train_one_epoch(model, loader, optimizer, device) -> tuple[float, float]:
    """训练一个 epoch（把整个训练集完整过一遍），返回 (平均 loss, 准确率)。

    这就是 PyTorch 训练的「四步循环」，也是未来 PPO 训练循环的骨架。

    参数：
      model     : nn.Module 模型本身（02 里的 MLP）。函数内会就地更新它的权重。
      loader    : 训练集的 DataLoader（见 build_loaders），迭代时每次吐一个 batch。
      optimizer : torch.optim 优化器（这里是 Adam），负责按梯度更新 model 的参数。
      device    : torch.device，cuda / mps / cpu；数据与模型都要搬到同一设备上算。
    返回：
      (total_loss/seen, correct/seen) —— 本 epoch 的「按样本加权平均 loss」和「准确率」。
    """
    model.train()                                  # 切到「训练模式」：开启 dropout/BatchNorm 的训练行为
                                                   #（本 MLP 没有这两种层，但养成习惯：训练前 train()、评估前 eval()）
    total_loss, correct, seen = 0.0, 0, 0          # 三个累加器：累计 loss、累计答对数、累计已见样本数
    for xb, yb in loader:                          # 逐 batch 迭代：xb 输入图 (B,1,28,28)，yb 标签 (B,) 整数 0~9
        xb, yb = xb.to(device), yb.to(device)      # 把这一批数据搬到 device；必须和 model 同设备，否则报错
        optimizer.zero_grad()                      # ① 清零上一步残留的梯度（grad 默认累加，见 01 §7 的坑）
        logits = model(xb)                         # ② 前向传播：算出每个样本对 10 个类别的原始分数 (B,10)
        loss = F.cross_entropy(logits, yb)         #    交叉熵损失：内部自动对 logits 做 softmax 再取负对数似然，
                                                   #    默认 reduction='mean'，即返回这一批的平均 loss（标量）
        loss.backward()                            # ③ 反向传播：autograd 沿网络自动用链式法则算出每个参数的梯度
        optimizer.step()                           # ④ 更新参数：本质是 参数 ← 参数 - 学习率 × 梯度（见 01 §9）

        total_loss += loss.item() * xb.size(0)     # loss.item() 把标量张量取成 Python float（脱离计算图）；
                                                   # × xb.size(0)（本批样本数 B）→ 还原成「本批 loss 总和」，便于最后按样本数求平均
        correct += (logits.argmax(dim=1) == yb).sum().item()  # argmax(dim=1)：沿第 1 维(10 个类别)取分数最高的下标=预测数字；
                                                   # == yb 得到布尔张量，.sum() 数出答对个数，.item() 转成 Python int 累加
        seen += xb.size(0)                         # 累计已处理样本数（末批可能不足 batch_size，所以要逐批累加而非乘批数）
    return total_loss / seen, correct / seen       # 平均 loss = 总 loss / 总样本；准确率 = 答对数 / 总样本


@torch.no_grad()                                   # 装饰器：整个函数内不建计算图、不追踪梯度 → 省显存、算得更快
                                                   #（评估只做前向、从不 backward，不需要梯度；等价于函数体裹在 with torch.no_grad():）
def evaluate(model, loader, device) -> tuple[float, float]:
    """在给定数据集上「考试」一遍，返回 (平均 loss, 准确率)，且【绝不修改 model 的权重】。

    参数：
      model  : 要评估的模型（评估中保持不变，只读）。
      loader : 数据集的 DataLoader（通常传测试集，也可传训练集算 train 指标）。
      device : torch.device，与 model 保持一致。
    返回：
      (total_loss/seen, correct/seen) —— 与 train_one_epoch 同构，便于两边对照打印。
    """
    model.eval()                                   # 切到「评估模式」：dropout 停止丢弃、BatchNorm 改用累计统计量
                                                   #（本 MLP 没有这两种层，故行为与 train() 相同，但仍要养成成对切换的习惯）
    total_loss, correct, seen = 0.0, 0, 0          # 三个累加器：累计 loss、累计答对数、累计已见样本数
    for xb, yb in loader:                          # 逐 batch 迭代（注意：这里没有 zero_grad/backward/step 三件套）
        xb, yb = xb.to(device), yb.to(device)      # 数据搬到 device，与 model 同设备
        logits = model(xb)                         # 只做前向：算出每个样本对 10 类的分数 (B,10)
        total_loss += F.cross_entropy(logits, yb, reduction="sum").item()  # reduction="sum"：返回这一批的 loss【总和】而非均值，
                                                   # 直接累加即可；最后统一除以 seen 得到全局按样本平均（比逐批 mean 再平均更准，末批不满也不怕偏）
        correct += (logits.argmax(dim=1) == yb).sum().item()  # argmax(dim=1) 取预测数字，与 yb 比对，数出本批答对个数
        seen += xb.size(0)                         # 累计已处理样本数（末批可能不足 batch_size）
    return total_loss / seen, correct / seen       # 平均 loss = 总 loss / 总样本；准确率 = 答对数 / 总样本


def main() -> None:
    ap = argparse.ArgumentParser(description="session_01 MLP × MNIST")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="auto", help="auto / cpu / cuda / mps")
    ap.add_argument("--data-dir", default=str(Path(__file__).parent / "data"))
    ap.add_argument("--smoke", action="store_true", help="小数据 1 轮冒烟测试")
    ap.add_argument("--out", default=str(Path(__file__).parent / "mlp_mnist_best.pt"))
    args = ap.parse_args()

    if args.smoke:
        args.epochs = 1

    device = get_device(args.device)
    print(f"PyTorch {torch.__version__} | 设备：{device} | epochs={args.epochs} "
          f"batch={args.batch_size} lr={args.lr}{' [SMOKE]' if args.smoke else ''}")

    torch.manual_seed(0)
    train_ld, test_ld = build_loaders(args.data_dir, args.batch_size, args.smoke)
    print(f"训练样本 {len(train_ld.dataset)} | 测试样本 {len(test_ld.dataset)}")

    model = MLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)   # Adam：初学者的稳妥默认

    n_params = sum(p.numel() for p in model.parameters())
    print(f"model 结构 = {model}")
    print(f"model.parameters = {model.parameters}")
    print(f"模型参数量 = {n_params}\n")

    best_acc, t0 = 0.0, time.time()
    for ep in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_ld, optimizer, device)
        te_loss, te_acc = evaluate(model, test_ld, device)
        if te_acc > best_acc:                       # 只保存「目前最好」的权重
            best_acc = te_acc
            torch.save({"model_state": model.state_dict(), "test_acc": te_acc,
                        "epoch": ep, "arch": "MLP-784-128-64-10"}, args.out)
        print(f"epoch {ep}/{args.epochs} | "
              f"train loss {tr_loss:.4f} acc {tr_acc*100:5.2f}% | "
              f"test loss {te_loss:.4f} acc {te_acc*100:5.2f}% "
              f"{'★' if te_acc >= best_acc else ''}")

    dt = time.time() - t0
    print("\n" + "=" * 72)
    print(f"训练完成，用时 {dt:.1f}s")
    print(f"最优测试准确率 = {best_acc*100:.2f}%   （权重已存 {args.out}）")
    if args.smoke:
        print("ℹ️  这是冒烟测试（小数据 1 轮），只验证流程；跑完整训练看真实准确率：")
        print("     python 02_mlp_mnist.py --epochs 5")
    elif best_acc > 0.97:
        print("✅ 检查点达成：测试准确率 > 97%（roadmap 阶段 0 的硬门槛已通过）")
    else:
        print("❌ 还没到 97%。多为训练不足：加大 --epochs 到 8，或确认 lr=1e-3、已归一化。")
    print("=" * 72)


if __name__ == "__main__":
    main()
