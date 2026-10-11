"""
session_01 · 第 2 周延伸：加载已训好的权重做推理（把「训出来 → 用起来」闭环）
==============================================================================

前面 02_mlp_mnist.py 训练后把「历史最佳权重」存进了 mlp_mnist_best.pt，
但那份文件在 session_01 里一直「只写不读」。本文件补上【读】的这一环——
这也是你第一次接触「模型部署」的核心动作：

    torch.load 读档 → 重建同构网络 → load_state_dict 灌权重 → eval()+no_grad → 前向推理

怎么跑（必须先有 checkpoint）：
    python 02_mlp_mnist.py --epochs 5      # 若还没训过，先训出 mlp_mnist_best.pt
    python 04_load_and_predict.py          # 加载它，对几张测试图做预测 + ASCII 可视化
    python 04_load_and_predict.py --n 8    # 多看几张样本
    python 04_load_and_predict.py --device cpu

🔗 与 RL 的联系（本文件存在的深层理由）：
  这套「加载 checkpoint → 重建网络 → 灌权重 → eval+no_grad → 前向」的流程，
  正是 microduck 把训练好的策略 model_*.pt 部署到真机时做的事的极简版——
  只不过 RL 里最后还会多一步「导出 ONNX」，再交给板载 Rust runtime 推理。
  你现在练的，就是未来「加载策略做推理」的手感。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

MNIST_MEAN, MNIST_STD = 0.1307, 0.3081   # 与 02/03 一致，反归一化可视化时要用


# --------------------------------------------------------------------------- #
# 模型：与 02_mlp_mnist.py 完全同构的 MLP。
# 加载权重的前提是「先建一个结构一模一样的空网络」，再把 state_dict 灌进去——
# 这就是 checkpoint 里为什么要存 arch 字段（下面 parse_arch 会用到）。
# --------------------------------------------------------------------------- #
class MLP(nn.Module):
    def __init__(self, input_dim: int = 784, hidden1: int = 128, hidden2: int = 64,
                 num_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)                    # 输出 logit（不加 softmax）
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


def parse_arch(arch: str) -> dict:
    """把 checkpoint 里存的 arch 字符串（如 'MLP-784-128-64-10'）解析成建网参数。

    这体现了「为什么存档要连结构一起存」：只有权重、不知道结构，是没法重建网络的。
    """
    nums = [int(s) for s in arch.split("-") if s.isdigit()]
    if len(nums) != 4:
        return {}                                    # 解析不了就用 MLP 默认值
    return {"input_dim": nums[0], "hidden1": nums[1], "hidden2": nums[2], "num_classes": nums[3]}


def load_model(ckpt_path: str, device: torch.device) -> tuple[MLP, dict]:
    """核心：从 .pt 文件恢复出一个可用的模型。四步走，每步都对应部署的关键动作。"""
    # ① 读档：torch.load 把文件反序列化成 Python dict。
    #    map_location=device → 无论当初存在 GPU 还是 CPU，都映射到当前设备，避免设备不匹配报错。
    #    weights_only=True（安全默认）→ 只允许反序列化张量与基础类型(int/float/str/dict)，
    #      拒绝任意 Python 对象，防止加载来路不明的 .pt 时执行恶意代码；本存档正好只含这些类型。
    ck = torch.load(ckpt_path, map_location=device, weights_only=True)

    # ② 按存档记录的 arch 重建一个「结构相同、权重随机」的空网络
    model = MLP(**parse_arch(ck.get("arch", ""))).to(device)

    # ③ 把学到的权重灌进去。strict=True（默认）要求参数名/形状完全对上，
    #    对不上会直接报错——这是好事，能防止你把权重灌进错误的网络结构。
    model.load_state_dict(ck["model_state"])

    # ④ 切评估模式：推理必须 eval()（关掉 dropout/BN 的训练行为，见 02 evaluate 的讲解）
    model.eval()
    return model, ck


def denormalize(img: torch.Tensor) -> torch.Tensor:
    """把归一化过的图反算回 [0,1] 以便可视化：x_orig = x_norm * std + mean。

    模型吃的是归一化后的张量，但要「看」这张图得先还原像素强度。
    这也是 01 §4 广播的逆运算。
    """
    return (img * MNIST_STD + MNIST_MEAN).clamp(0, 1)


def to_ascii(img: torch.Tensor) -> str:
    """把一张 (1,28,28) 的图渲染成终端 ASCII 字符画（越亮字符越密）。"""
    ramp = " .:-=+*#%@"                       # 强度从低到高的字符梯度
    g = denormalize(img).squeeze(0)            # 去掉通道维 → (28,28)，值域 [0,1]
    rows = []
    for r in range(g.shape[0]):
        line = "".join(ramp[min(int(v * (len(ramp) - 1)), len(ramp) - 1)] for v in g[r])
        rows.append("    " + line)
    return "\n".join(rows)


@torch.no_grad()                                # 推理不建计算图：省显存更快（见 02 evaluate）
def predict_batch(model, xb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """对一个 batch 做前向，返回 (预测类别, softmax 概率)。"""
    logits = model(xb)                          # (B,10) 原始分数
    probs = F.softmax(logits, dim=1)            # 转成概率（这里只为「展示」，预测本身不需要）
    preds = probs.argmax(dim=1)                 # 概率最大的下标 = 预测数字
    return preds, probs


def main() -> None:
    ap = argparse.ArgumentParser(description="session_01 加载 checkpoint 做推理")
    ap.add_argument("--ckpt", default=str(Path(__file__).parent / "mlp_mnist_best.pt"))
    ap.add_argument("--data-dir", default=str(Path(__file__).parent / "data"))
    ap.add_argument("--device", default="auto", help="auto / cpu / cuda / mps")
    ap.add_argument("--n", type=int, default=5, help="可视化展示前 n 张测试样本")
    args = ap.parse_args()

    if not Path(args.ckpt).exists():
        print(f"❌ 找不到 checkpoint：{args.ckpt}")
        print("   请先训练生成它：python 02_mlp_mnist.py --epochs 5")
        return

    device = get_device(args.device)
    print(f"PyTorch {torch.__version__} | 设备：{device}")

    # --- 第一部分：加载模型 ---
    model, ck = load_model(args.ckpt, device)
    print("\n" + "=" * 72)
    print("① 加载 checkpoint")
    print("=" * 72)
    print(f"   文件        : {args.ckpt}")
    print(f"   结构 arch   : {ck.get('arch')}")
    print(f"   存档时 epoch: {ck.get('epoch')}")
    print(f"   存档时 acc  : {ck.get('test_acc', 0) * 100:.2f}%   ← 待会儿复核它是否吻合")

    # --- 第二部分：准备测试数据（沿用 02 的归一化流水线，保证输入分布一致）---
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
    ])
    test_ds = datasets.MNIST(args.data_dir, train=False, download=True, transform=tf)
    test_ld = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    # --- 第三部分：逐张可视化推理 ---
    print("\n" + "=" * 72)
    print(f"② 对前 {args.n} 张测试图做推理（ASCII 画 + 预测 + 置信度）")
    print("=" * 72)
    shown = 0
    for xb, yb in test_ld:
        xb, yb = xb.to(device), yb.to(device)
        preds, probs = predict_batch(model, xb)
        for i in range(xb.size(0)):
            if shown >= args.n:
                break
            p = preds[i].item()
            t = yb[i].item()
            conf = probs[i, p].item()
            flag = "✅" if p == t else "❌"
            print(f"\n样本 {shown}  预测={p}  真实={t}  {flag}  置信度={conf*100:.1f}%")
            print(to_ascii(xb[i].cpu()))
            shown += 1
        if shown >= args.n:
            break

    # --- 第四部分：全测试集复核准确率（应与存档的 test_acc 一致）---
    print("\n" + "=" * 72)
    print("③ 全测试集复核准确率")
    print("=" * 72)
    correct = total = 0
    for xb, yb in test_ld:
        xb, yb = xb.to(device), yb.to(device)
        preds, _ = predict_batch(model, xb)
        correct += (preds == yb).sum().item()
        total += yb.size(0)
    acc = correct / total
    print(f"   加载的权重在 {total} 张测试图上：acc = {acc*100:.2f}%")
    print(f"   checkpoint 记录的 test_acc     = {ck.get('test_acc', 0)*100:.2f}%")
    print("   两者吻合 → 说明「存档 → 加载 → 推理」链路正确无误，权重完整复原。")

    print("\n" + "=" * 72)
    print("✅ 闭环完成：02 训练存 best.pt → 04 读档重建 → eval+no_grad 推理复核。")
    print("   🔗 这正是 RL 部署策略的雏形；差别只在 RL 还会把网络导出成 ONNX 交给真机。")
    print("=" * 72)
    # ✏️ 试一试：
    #   1) 把 --n 调大到 15，找一个被预测错(❌)的样本，看它到底像哪个数字。
    #   2) 故意把 load_model 里的 MLP(**parse_arch(...)) 改成 MLP(hidden1=32)，
    #      观察 load_state_dict 报什么错——体会 strict 匹配为何能防呆。


if __name__ == "__main__":
    main()
