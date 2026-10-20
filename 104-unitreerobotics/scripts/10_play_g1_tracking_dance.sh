#!/usr/bin/env bash
# 10_play_g1_tracking_dance.sh —— G1 动作模仿（tracking / 跳舞）策略回放
#
# 【这是什么】
#   加载训练好的 tracking 策略，用 viser 在浏览器里看 G1 跟着动捕跳一整段舞。
#   实测最佳 checkpoint 是 model_29000（峰值 26.22 @ iter 29079，健康收敛）——末位 ≠ 最佳，建议传 29000。
#
# 【用法】
#   ./scripts/10_play_g1_tracking_dance.sh 29000            # 最新 run 的 model_29000.pt（实测最佳）
#   ./scripts/10_play_g1_tracking_dance.sh                  # 不给 iter = 最新 run 末位（会提示末位≠最佳）
#   ./scripts/10_play_g1_tracking_dance.sh path/model.pt    # 直接给 checkpoint 路径
#   DRY_RUN=1 ./scripts/10_play_g1_tracking_dance.sh 29000  # 只打印命令不执行
#
# 【坑（源码核实过）】
#   play 必须同时给 --motion-file（策略只学了 dance1_subject2 这一段动捕，脚本已自动带上）
#   --checkpoint-file 必须给完整相对路径（只给文件名会 FileNotFoundError）；play 没有 --agent.* 参数组
#   浏览器打开 http://localhost:8080；有显示器想用原生窗口，把脚本尾部 --viewer viser 改 native

set -euo pipefail

ARG="${1:-}"
TASK_NAME="Unitree-G1-Tracking-No-State-Estimation"   # 任务注册名（list_envs.py 可查）
WS_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$WS_DIR/repos/unitree_rl_mjlab"
LOGS="$REPO/logs/rsl_rl/g1_tracking"
PY=~/miniconda3/envs/unitree_rl_mjlab/bin/python
MOTION="src/assets/motions/g1/dance1_subject2.npz"

# ---- 定位 checkpoint：给了 .pt 路径直接用；给了数字取最新 run 的 model_<n>.pt；不给取末位 ----
if [[ "$ARG" == *.pt ]]; then
  CKPT="$ARG"
else
  RUN_DIR=$(ls -dt "$LOGS"/*/ 2>/dev/null | head -1) || true
  [[ -n "${RUN_DIR:-}" ]] || { echo "[x] $LOGS 下没有任何 run，先跑 09_train_g1_tracking_dance.sh"; exit 1; }
  if [[ -n "$ARG" ]]; then
    CKPT="$RUN_DIR/model_${ARG}.pt"
  else
    CKPT=$(ls "$RUN_DIR"model_*.pt 2>/dev/null | sort -t_ -k2 -n | tail -1) || true
    [[ -n "${CKPT:-}" ]] || { echo "[x] $RUN_DIR 下没有 model_*.pt"; exit 1; }
    echo "[w] 未指定 iter，自动取末位。实测末位 ≠ 最佳（最佳 model_29000），看曲线挑（11/12 脚本）"
  fi
fi
[[ -f "$CKPT" ]] || { echo "[x] checkpoint 不存在: $CKPT"; exit 1; }
[[ -f "$REPO/$MOTION" ]] || { echo "[x] 动捕文件不存在: $REPO/$MOTION"; exit 1; }
CKPT_REL="${CKPT#"$REPO"/}"     # --checkpoint-file 需相对 repo 根的完整路径
CKPT_REL="${CKPT_REL//\/\//\/}" # 收敛 RUN_DIR 尾斜杠产生的 //

echo "[i] run     : $(dirname "$CKPT_REL")"
echo "[i] ckpt    : $(basename "$CKPT_REL")"
echo "[i] motion  : $MOTION"
echo "[i] 浏览器打开 http://localhost:8080（viser；Ctrl+C 停止）"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[i] DRY_RUN，将执行:"
  (cd "$REPO" && echo "$PY scripts/play.py $TASK_NAME --motion-file $MOTION --checkpoint-file $CKPT_REL --viewer viser")
  exit 0
fi

cd "$REPO"
exec "$PY" scripts/play.py "$TASK_NAME" \
  --motion-file "$MOTION" \
  --checkpoint-file "$CKPT_REL" \
  --viewer viser
