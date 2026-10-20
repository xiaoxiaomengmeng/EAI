#!/usr/bin/env bash
# 12_play_g1_velocity_rough.sh —— G1 粗糙地形速度跟随（velocity rough）策略回放【待验证】
#
# 【这是什么】
#   加载训练好的 velocity rough 策略，用 viser 在浏览器里看 G1 在粗糙地形上走路。
#   日志目录是 logs/rsl_rl/g1_velocity_rough/（11 训练脚本已做 experiment 隔离），
#   与 flat 的 g1_velocity/ 天然分开，不会拿错 checkpoint。
#
# 【状态：待验证】
#   本任务未实测。checkpoint「末位 ≠ 最佳」的经验来自 flat/tracking，rough 未必同款——
#   训完先看曲线再挑：./scripts/14_plot_velocity_curve.py --task g1_velocity_rough
#
# 【用法】
#   ./scripts/12_play_g1_velocity_rough.sh 10000          # 最新 run 的 model_10000.pt
#   ./scripts/12_play_g1_velocity_rough.sh                # 不给 iter = 最新 run 末位
#   ./scripts/12_play_g1_velocity_rough.sh path/model.pt  # 直接给 checkpoint 路径
#   DRY_RUN=1 ./scripts/12_play_g1_velocity_rough.sh      # 只打印命令不执行
#
# 【坑（源码核实过）】
#   rough 与 flat 的 checkpoint 不通用：rough 观测多 height_scan（地形高度射线扫描），
#   policy 输入维度不同，拿 flat 的 checkpoint 加载会维度不匹配报错，反之亦然
#   --checkpoint-file 必须给完整相对路径（只给文件名会 FileNotFoundError）；play 没有 --agent.* 参数组
#   浏览器打开 http://localhost:8080；有显示器想用原生窗口，把脚本尾部 --viewer viser 改 native

set -euo pipefail

ARG="${1:-}"
TASK_NAME="Unitree-G1-Rough"    # 任务注册名（list_envs.py 可查）
EXPERIMENT="g1_velocity_rough"  # 与 11 训练脚本保持一致
WS_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$WS_DIR/repos/unitree_rl_mjlab"
LOGS="$REPO/logs/rsl_rl/$EXPERIMENT"
PY=~/miniconda3/envs/unitree_rl_mjlab/bin/python

# ---- 定位 checkpoint：给了 .pt 路径直接用；给了数字取最新 run 的 model_<n>.pt；不给取末位 ----
if [[ "$ARG" == *.pt ]]; then
  CKPT="$ARG"
else
  RUN_DIR=$(ls -dt "$LOGS"/*/ 2>/dev/null | head -1) || true
  [[ -n "${RUN_DIR:-}" ]] || { echo "[x] $LOGS 下没有任何 run，先跑 11_train_g1_velocity_rough.sh"; exit 1; }
  if [[ -n "$ARG" ]]; then
    CKPT="$RUN_DIR/model_${ARG}.pt"
  else
    CKPT=$(ls "$RUN_DIR"model_*.pt 2>/dev/null | sort -t_ -k2 -n | tail -1) || true
    [[ -n "${CKPT:-}" ]] || { echo "[x] $RUN_DIR 下没有 model_*.pt"; exit 1; }
    echo "[w] 未指定 iter，自动取末位。rough 尚无实测最佳 checkpoint 数据，建议看曲线挑（14 脚本 --task $EXPERIMENT）"
  fi
fi
[[ -f "$CKPT" ]] || { echo "[x] checkpoint 不存在: $CKPT"; exit 1; }
CKPT_REL="${CKPT#"$REPO"/}"     # --checkpoint-file 需相对 repo 根的完整路径
CKPT_REL="${CKPT_REL//\/\//\/}" # 收敛 RUN_DIR 尾斜杠产生的 //

echo "[i] run     : $(dirname "$CKPT_REL")"
echo "[i] ckpt    : $(basename "$CKPT_REL")"
echo "[i] 浏览器打开 http://localhost:8080（viser；Ctrl+C 停止）"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[i] DRY_RUN，将执行:"
  (cd "$REPO" && echo "$PY scripts/play.py $TASK_NAME --checkpoint-file $CKPT_REL --viewer viser")
  exit 0
fi

cd "$REPO"
exec "$PY" scripts/play.py "$TASK_NAME" \
  --checkpoint-file "$CKPT_REL" \
  --viewer viser
