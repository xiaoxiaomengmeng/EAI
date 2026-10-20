#!/usr/bin/env python
"""画出 g1_velocity 正式训练的 mean_reward 曲线，复现 docs/05 口播文稿所述「塌陷」证据。

目的：一张图看懂 velocity run 的曲线故事——峰值 39.37 @ iter 4965、5000→6000 断崖
跌 14 分、终值 27.45 仅恢复到峰值 70%，直接对应 docs/05-口播文稿-G1走路与跳舞复现.md
的「数字来源对照」表。也可用于查看 g1_tracking 或其他 scalar tag。

用法（unitree_rl_mjlab 环境）：
    ~/miniconda3/envs/unitree_rl_mjlab/bin/python scripts/14_plot_velocity_curve.py
    ~/miniconda3/envs/unitree_rl_mjlab/bin/python scripts/14_plot_velocity_curve.py --task g1_tracking
    ~/miniconda3/envs/unitree_rl_mjlab/bin/python scripts/14_plot_velocity_curve.py --tag error_vel_xy --run 2026-10-19_17-23-46

判据：
    终端摘要应显示 peak ≈39.37 @ iter 4965、谷底 ≈24.6（5000-6000 窗口内）、
    终值 ≈27.45（≈峰值 70%）；PNG 输出到 pics/，含峰值/塌陷区间/checkpoint 标注。
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 纯落盘，无显示环境依赖
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

WS = Path(__file__).resolve().parent.parent  # 104-unitreerobotics/
REPO = WS / "repos" / "unitree_rl_mjlab"
LOGS = REPO / "logs" / "rsl_rl"


def find_run(task: str, run: str | None) -> Path:
    """定位 run 目录：默认取该任务下最近修改且含 tfevents 的一个。"""
    task_dir = LOGS / task
    if not task_dir.is_dir():
        sys.exit(f"[x] 任务目录不存在: {task_dir}")
    candidates = [d for d in task_dir.iterdir() if d.is_dir() and list(d.glob("events.out.tfevents.*"))]
    if not candidates:
        sys.exit(f"[x] {task_dir} 下没有含 tfevents 的 run")
    if run:
        hit = next((d for d in candidates if d.name == run), None)
        if hit is None:
            sys.exit(f"[x] 未找到 run '{run}'，可选: {sorted(d.name for d in candidates)}")
        return hit
    return max(candidates, key=lambda d: d.stat().st_mtime)


def pick_tag(acc: EventAccumulator, want: str | None) -> str:
    """模糊选 scalar tag：默认优先名字含 reward 的。"""
    tags = acc.Tags()["scalars"]
    if not tags:
        sys.exit("[x] tfevents 里没有任何 scalar tag")
    if want:
        hit = next((t for t in tags if want in t), None)
        if hit is None:
            sys.exit(f"[x] 无 tag 含 '{want}'，全部 tags:\n  " + "\n  ".join(tags))
        return hit
    for t in tags:  # 名字含 reward 的优先，取最长者（mean_reward > reward）
        if "reward" in t:
            return t
    sys.exit(f"[x] 未找到名字含 reward 的 tag，全部 tags:\n  " + "\n  ".join(tags))


def ema(values: list[float], alpha: float = 0.05) -> list[float]:
    out, s = [], values[0]
    for v in values:
        s = alpha * v + (1 - alpha) * s
        out.append(s)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", default="g1_velocity", help="任务名（logs/rsl_rl 下的目录，默认 g1_velocity）")
    ap.add_argument("--run", default=None, help="run 目录名（默认最近修改的一个）")
    ap.add_argument("--tag", default=None, help="scalar tag（默认自动选名字含 reward 的）")
    ap.add_argument("--out", default=None, help="PNG 输出路径（默认 pics/<task>_reward_curve.png）")
    args = ap.parse_args()

    run_dir = find_run(args.task, args.run)
    acc = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})  # 0 = 全量加载
    acc.Reload()
    tag = pick_tag(acc, args.tag)
    events = acc.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    if len(steps) < 2:
        sys.exit("[x] scalar 点太少，画不了曲线")

    # ---- 关键指标（与 docs/05 对照表口径一致）----
    peak_i = max(range(len(values)), key=lambda i: values[i])
    peak_step, peak_val = steps[peak_i], values[peak_i]
    trough_i = max(range(peak_i, len(values)), key=lambda i: -values[i])  # 峰值之后的谷底
    trough_step, trough_val = steps[trough_i], values[trough_i]
    final_val = values[-1]
    print(f"run     : {run_dir.name}")
    print(f"tag     : {tag}（{len(steps)} 点）")
    print(f"peak    : {peak_val:.2f} @ iter {peak_step}")
    print(f"trough  : {trough_val:.2f} @ iter {trough_step}（较峰值跌 {peak_val - trough_val:.2f}）")
    print(f"final   : {final_val:.2f}（峰值的 {final_val / peak_val * 100:.0f}%）")

    ckpts = sorted(int(m.group(1)) for p in run_dir.glob("model_*.pt") if (m := re.match(r"model_(\d+)\.pt", p.name)))
    print(f"ckpts   : {['model_%d' % c for c in ckpts]}")

    # ---- 画图 ----
    out = Path(args.out) if args.out else WS / "pics" / f"{args.task}_reward_curve.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(steps, values, lw=0.8, alpha=0.4, color="#1f77b4", label="raw")
    ax.plot(steps, ema(values), lw=1.8, color="#1f77b4", label="EMA(0.05)")

    ax.scatter([peak_step], [peak_val], color="red", zorder=5)
    ax.annotate(f"peak {peak_val:.2f} @ {peak_step}", (peak_step, peak_val), textcoords="offset points",
                xytext=(-10, 10), ha="right", color="red")
    if trough_val < peak_val * 0.9:  # 明显塌陷才画塌陷窗口
        win0, win1 = peak_step, trough_step
        ax.axvspan(win0, win1, color="red", alpha=0.08)
        ax.annotate(f"collapse {peak_val - trough_val:.1f}", ((win0 + win1) / 2, trough_val),
                    textcoords="offset points", xytext=(0, -14), ha="center", color="darkred")
    ax.axhline(final_val, color="gray", ls="--", lw=0.8)
    ax.annotate(f"final {final_val:.2f} ({final_val / peak_val * 100:.0f}% of peak)", (steps[-1], final_val),
                textcoords="offset points", xytext=(-5, 6), ha="right", color="gray")
    for c in ckpts:  # 已保留的 checkpoint 竖虚线——直观看到「最佳是不是末位」
        ax.axvline(c, color="green", ls=":", lw=0.9, alpha=0.7)
        ax.annotate(f"model_{c}", (c, ax.get_ylim()[0]), textcoords="offset points",
                    xytext=(3, 4), rotation=90, fontsize=8, color="green")

    ax.set_title(f"{args.task} / {tag}")  # 不带 run 日期，便于对外分享
    ax.set_xlabel("iteration")
    ax.set_ylabel(tag.split("/")[-1])
    ax.legend(loc="lower right")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved   : {out}")


if __name__ == "__main__":
    main()
