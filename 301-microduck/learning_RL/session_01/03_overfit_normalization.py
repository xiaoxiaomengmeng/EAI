"""
session_01 · 第 2 周概念实验：overfitting（过拟合） & normalization（归一化）
============================================================================

对应 learning_roadmap.md 阶段 0 第 2 周「理解 overfitting / normalization」。
02_mlp_mnist.py 是「把 MNIST 训到 >97%」的正面产出；本文件是配套的**对照实验**，
用两个可运行的小实验把这两个抽象概念变成能亲眼看到的现象。

    python 03_overfit_normalization.py                 # 依次跑两个实验
    python 03_overfit_normalization.py --exp overfit   # 只看过拟合
    python 03_overfit_normalization.py --exp norm      # 只看归一化
    python 03_overfit_normalization.py --smoke         # 更快：更少 epoch/样本

实验一（overfitting）：同一个 MLP，分别用「极小子集」和「全量」训练，
    对比 train_acc 与 test_acc 的差距（generalization gap）。
    → 你会看到：小子集把 train_acc 顶到接近 100%，test_acc 却卡住 = 过拟合。

实验二（normalization）：同一个 MLP，分别在「不归一化」和「归一化」的数据上训练，
    对比前几个 epoch 的收敛速度与最终 test_acc。
    → 你会看到：归一化后 loss 下降更快、更稳，这正是 02 里 Normalize 的价值。

两个实验都复用 MNIST（会缓存到 ./data，02 跑过就无需重新下载）。
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

# 与 02 保持一致的 MNIST 均值/标准差（社区公认值）
MNIST_MEAN, MNIST_STD = 0.1307, 0.3081


# --------------------------------------------------------------------------- #
# 模型：直接复用 02 里的 MLP 结构（784 → 128 → 64 → 10）
# --------------------------------------------------------------------------- #
class MLP(nn.Module):
    """和 02_mlp_mnist.py 里完全相同的三层 MLP。

    之所以复制一份而不是 import，是为了让每个教学文件都能**独立运行**、
    不产生跨文件依赖（初学者直接 python 03_...py 就能跑，不必管包结构）。
    """

    def __init__(self, input_dim: int = 784, hidden1: int = 128, hidden2: int = 64,
                 num_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)          # (B,1,28,28) → (B,784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)                    # 输出 logit，交给 cross_entropy 内部做 softmax
        return x


def get_device(pref: str) -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_mnist(data_dir: str, normalize: bool):
    """按是否归一化，返回 (train_dataset, test_dataset)。"""
    steps = [transforms.ToTensor()]
    if normalize:
        steps.append(transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)))
    tf = transforms.Compose(steps)
    train_ds = datasets.MNIST(data_dir, train=True, download=True, transform=tf)
    test_ds = datasets.MNIST(data_dir, train=False, download=True, transform=tf)
    return train_ds, test_ds


def run_epoch(model, loader, optimizer, device, train: bool) -> tuple[float, float]:
    """train=True 走四步循环训练；train=False 只评估。返回 (loss, acc)。"""
    model.train() if train else model.eval()
    total_loss, correct, seen = 0.0, 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            if train:
                optimizer.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb, reduction="sum")
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            correct += (logits.argmax(dim=1) == yb).sum().item()
            seen += xb.size(0)
    return total_loss / seen, correct / seen


# --------------------------------------------------------------------------- #
# 实验一：overfitting —— 小样本把训练集背下来，测试集却上不去
# --------------------------------------------------------------------------- #
def exp_overfit(device: torch.device, data_dir: str, smoke: bool) -> None:
    print("\n" + "=" * 72)
    print("实验一 · overfitting（过拟合）")
    print("=" * 72)
    print("同一个 MLP，同一份测试集，只改「训练样本量」：")
    print("  · 小样本（过拟合组）：只用极少训练数据，还训很多轮 → 把答案背下来")
    print("  · 全量  （正常组）  ：用完整 6 万张训练 → 学到可泛化的规律")
    print("看两组的 train_acc 与 test_acc 差距（generalization gap）。\n")

    train_ds, test_ds = make_mnist(data_dir, normalize=True)
    n_small = 128 if not smoke else 64
    small_ds = Subset(train_ds, range(n_small))
    epochs_small = 30 if not smoke else 15
    epochs_full = 3 if not smoke else 1

    cases = [
        ("过拟合组", small_ds, epochs_small),
        ("正常组  ", train_ds, epochs_full),
    ]
    test_ld = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    for name, ds, epochs in cases:
        torch.manual_seed(0)
        model = MLP().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train_ld = DataLoader(ds, batch_size=128, shuffle=True, num_workers=0)
        t0 = time.time()
        for ep in range(epochs):
            tr_loss, tr_acc = run_epoch(model, train_ld, optimizer, device, train=True)
        te_loss, te_acc = run_epoch(model, test_ld, None, device, train=False)
        gap = tr_acc - te_acc
        print(f"[{name}] 训练样本 {len(ds):>6d} · {epochs:>2d} epoch · {time.time()-t0:4.1f}s")
        print(f"           train_acc = {tr_acc*100:6.2f}%  |  test_acc = {te_acc*100:6.2f}%  "
              f"|  gap = {gap*100:5.2f}%")

    print("\n结论：过拟合组 train_acc 逼近 100% 但 gap 巨大——它「背题」而非「学会」。")
    print("      这就是为什么 RL/深度学习都要看验证/测试指标，而不是只盯训练 loss。")
    print("      解药：更多数据、正则化（dropout/weight_decay）、早停、数据增强。")
    # ✏️ 试一试：把过拟合组的 n_small 从 128 加到 2000，观察 gap 如何缩小。


# --------------------------------------------------------------------------- #
# 实验二：normalization —— 归一化让优化「好走」，收敛更快更稳
# --------------------------------------------------------------------------- #
def exp_norm(device: torch.device, data_dir: str, smoke: bool) -> None:
    print("\n" + "=" * 72)
    print("实验二 · normalization（归一化）")
    print("=" * 72)
    print("同一个 MLP，同一份数据，只改「是否做 (x-mean)/std 归一化」：")
    print("  · 不归一化：像素停留在 [0,1]，且大量接近 0（背景黑）")
    print("  · 归一化  ：零均值、单位方差，梯度地形更规整")
    print("逐 epoch 打印 test_acc，对比收敛速度与最终水平。\n")

    epochs = 5 if not smoke else 2
    n_train = 6000 if smoke else None   # smoke 时用小子集加速
    results = {}

    for normalize in (False, True):
        train_ds, test_ds = make_mnist(data_dir, normalize=normalize)
        if n_train is not None:
            train_ds = Subset(train_ds, range(n_train))
        torch.manual_seed(0)
        model = MLP().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train_ld = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
        test_ld = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

        tag = "归一化  " if normalize else "不归一化"
        accs = []
        t0 = time.time()
        for ep in range(1, epochs + 1):
            tr_loss, _ = run_epoch(model, train_ld, optimizer, device, train=True)
            _, te_acc = run_epoch(model, test_ld, None, device, train=False)
            accs.append(te_acc)
            print(f"[{tag}] epoch {ep}/{epochs} | train_loss {tr_loss:.4f} "
                  f"| test_acc {te_acc*100:6.2f}%")
        results[normalize] = accs
        print(f"[{tag}] 用时 {time.time()-t0:4.1f}s | 首个 epoch 就到 "
              f"{accs[0]*100:.2f}%\n")

    fast, slow = max(results[True]), max(results[False])
    print("结论：归一化组通常第一个 epoch 就明显更高、整体更稳。")
    print(f"      最终 test_acc：归一化 {fast*100:.2f}% vs 不归一化 {slow*100:.2f}%。")
    print("      直觉：把输入拉到零均值单位方差后，各维权重地位平等，")
    print("      梯度不再被「大尺度维度」主导 → 学习率更好选、收敛更快。")
    print("      联系 RL：microduck_rl 对 61 维观测做 obs normalizer，是同一招。")
    # ✏️ 试一试：把 lr 调大到 0.05，对比不归一化组是否开始震荡/发散。


def main() -> None:
    ap = argparse.ArgumentParser(description="session_01 过拟合 & 归一化对照实验")
    ap.add_argument("--exp", choices=["overfit", "norm", "both"], default="both")
    ap.add_argument("--device", default="auto", help="auto / cpu / cuda / mps")
    ap.add_argument("--data-dir", default=str(Path(__file__).parent / "data"))
    ap.add_argument("--smoke", action="store_true", help="更少样本/epoch，快速验证流程")
    args = ap.parse_args()

    device = get_device(args.device)
    print(f"PyTorch {torch.__version__} | 设备：{device}"
          f"{' | [SMOKE]' if args.smoke else ''}")

    if args.exp in ("overfit", "both"):
        exp_overfit(device, args.data_dir, args.smoke)
    if args.exp in ("norm", "both"):
        exp_norm(device, args.data_dir, args.smoke)

    print("\n" + "=" * 72)
    print("✅ 两个概念实验跑完。回到 02_mlp_mnist.py：现在你应能理解它为何要")
    print("   Normalize、为何用 test_acc 判定 >97% 检查点，而不是看 train_acc。")
    print("=" * 72)


if __name__ == "__main__":
    main()
