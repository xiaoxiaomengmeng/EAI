#!/usr/bin/env python3
"""
从 lerobot-train 日志中提取指标并绘制 loss 曲线（离线版，不依赖 wandb）。

背景：v0.6.2 训练日志中每 --log_freq 步输出一行
  `step:200 smpl:2K ep:4 epch:0.08 loss:7.468 grdn:159.210 lr:1.0e-05 ... l1_loss:0.621 kld_loss:0.685`
注意 step 从 1000 起会被缩写为 `step:1K`（100000 为 `step:100K`），解析时需还原。

三个指标的含义（ACT，见 configuration_act.py）：
  loss     = 总损失 = l1_loss + kl_weight * kld_loss（kl_weight 默认 10.0）
  l1_loss  = 重建损失：预测动作块与演示动作的 L1 距离（越小 = 模仿越准，核心指标）
  kld_loss = KL 散度：CVAE 潜变量 z 分布与标准正态的距离（正则项）
因 loss 量级约为 l1_loss 的 10 倍以上，单轴画会把 l1/kld 压扁，故用三子图分别展示。

用法：
  python scripts/plot_train_log.py                                   # 自动找最新 train_act_*.log
  python scripts/plot_train_log.py logs/train_act_20260829_172156.log
  python scripts/plot_train_log.py <日志路径> -o out.png             # 指定输出文件

注意：曲线密度取决于训练时的 --log_freq（train_act.sh 用 LOG_FREQ 环境变量控制，默认 100）。
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无头环境（SSH/无显示器）直接出图
import matplotlib.pyplot as plt

# step 可能被缩写：step:900 / step:1K / step:100K / step:1.5M
STEP_RE = re.compile(r"step:([\d.]+)([KkMm]?)\b")
KV_RE = re.compile(r"([a-z][a-z0-9_]*):(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\b")
CURVES = {
    "loss": "total loss = l1 + 10*kld",
    "l1_loss": "l1 reconstruction (action L1)",
    "kld_loss": "KL divergence (z ~ N(0,1))",
}
SUFFIX = {"": 1, "k": 1_000, "m": 1_000_000}


def parse_step(token: str, suffix: str) -> int:
    return int(float(token) * SUFFIX[suffix.lower()])


def parse_log(log_path: Path) -> dict[str, list[tuple[int, float]]]:
    """解析日志，返回 {指标名: [(step, value), ...]}（按 step 升序）"""
    series: dict[str, list[tuple[int, float]]] = {k: [] for k in CURVES}
    for line in log_path.read_text(errors="ignore").splitlines():
        m = STEP_RE.search(line)
        if not m:
            continue
        step = parse_step(m.group(1), m.group(2))
        kvs = dict(KV_RE.findall(line))
        for key in CURVES:
            if key in kvs:
                series[key].append((step, float(kvs[key])))
    for key in series:
        series[key].sort(key=lambda p: p[0])
    return series


def main() -> None:
    parser = argparse.ArgumentParser(description="从 lerobot 训练日志绘制 loss 曲线")
    parser.add_argument("log", nargs="?", help="训练日志路径（默认取 logs/ 下最新 train_act_*.log）")
    parser.add_argument("-o", "--out", help="输出 PNG 路径（默认：日志同目录下 <日志名>_loss_curve.png）")
    args = parser.parse_args()

    home = Path(__file__).resolve().parent.parent
    if args.log:
        log_path = Path(args.log)
    else:
        candidates = sorted((home / "logs").glob("train_act_*.log"))
        if not candidates:
            sys.exit("✗ 未找到 logs/train_act_*.log，请指定日志路径")
        log_path = candidates[-1]
    if not log_path.exists():
        sys.exit(f"✗ 日志不存在：{log_path}")

    series = parse_log(log_path)
    available = {k: v for k, v in series.items() if v}
    if not available:
        sys.exit(f"✗ 日志中没有解析到任何指标行（检查训练时 --log_freq 是否过大）：{log_path}")
    n_points = max(len(v) for v in available.values())
    last_step = max(p[0] for v in available.values() for p in v)
    print(f"✓ 解析 {log_path}：{n_points} 个数据点，step 范围 0~{last_step}")
    for key, points in available.items():
        print(f"  {key:9s} 首值={points[0][1]:.4f}  末值={points[-1][1]:.4f}")

    out_path = Path(args.out) if args.out else log_path.with_name(log_path.stem + "_loss_curve.png")
    n = len(available)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3.2 * n), sharex=True, squeeze=False)
    for ax, (key, points) in zip((a[0] for a in axes), available.items()):
        steps, values = zip(*points)
        ax.plot(steps, values, color="tab:blue", linewidth=1.2)
        ax.set_ylabel(key)
        ax.set_title(f"{key} — {CURVES[key]}", fontsize=10)
        ax.grid(alpha=0.3)
    axes[-1][0].set_xlabel("step")
    fig.suptitle(f"ACT training curves ({log_path.stem})", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"✓ 曲线已保存：{out_path}")


if __name__ == "__main__":
    main()
