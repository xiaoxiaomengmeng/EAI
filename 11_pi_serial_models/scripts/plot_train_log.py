#!/usr/bin/env python3
"""
从 lerobot-train 日志中提取指标并绘制 loss 曲线（离线版，不依赖 wandb）。

背景：v0.6.2 训练日志中每 --log_freq 步输出一行
  `step:200 smpl:2K ep:4 epch:0.08 loss:7.468 grdn:159.210 lr:1.0e-05 ... l1_loss:0.621 kld_loss:0.685`
本脚本解析这些行，把 loss / l1_loss / kld_loss（存在即画）对 step 画成 PNG。
日志路径默认为 logs/ 下最新的 train_act_*.log。

用法：
  python scripts/plot_train_log.py                                   # 自动找最新 train_act 日志
  python scripts/plot_train_log.py logs/train_act_20260829_171537.log
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

# 匹配指标行：INFO ... step:200 smpl:2K ... loss:7.468 ... l1_loss:0.621 kld_loss:0.685
STEP_RE = re.compile(r"step:(\d+)")
KV_RE = re.compile(r"([a-z][a-z0-9_]*):(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\b")
CURVES = ["loss", "l1_loss", "kld_loss"]  # 存在的才画


def parse_log(log_path: Path) -> dict[str, list[tuple[int, float]]]:
    """解析日志，返回 {指标名: [(step, value), ...]}"""
    series: dict[str, list[tuple[int, float]]] = {k: [] for k in CURVES}
    for line in log_path.read_text(errors="ignore").splitlines():
        m = STEP_RE.search(line)
        if not m:
            continue
        step = int(m.group(1))
        kvs = dict(KV_RE.findall(line))
        for key in CURVES:
            if key in kvs:
                series[key].append((step, float(kvs[key])))
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
    print(f"✓ 解析 {log_path}：{n_points} 个数据点，指标 {list(available)}")

    out_path = Path(args.out) if args.out else log_path.with_name(log_path.stem + "_loss_curve.png")
    fig, ax = plt.subplots(figsize=(10, 6))
    for key, points in available.items():
        steps, values = zip(*points)
        ax.plot(steps, values, label=key, linewidth=1.2)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title(f"ACT training loss ({log_path.stem})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"✓ 曲线已保存：{out_path}")


if __name__ == "__main__":
    main()
