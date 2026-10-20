#!/usr/bin/env bash
# 02_play_ball_kick_high.sh —— microduck 高踢球（ball_kick high kick）策略回放
#
# 【这是什么】
#   加载训练好的高踢 checkpoint 看效果。本机 headless（无 DISPLAY），默认走
#   play_to_video.py：rollout 逐帧 EGL 渲染成 mp4（相机水平跟球 + 球高/峰值标注）；
#   有显示器时自动切 play_high_kick.py 原生交互 viewer（可拖拽）。
#
# 【用法】
#   ./scripts/02_play_ball_kick_high.sh                       # v5 最新 run 末位 checkpoint
#   ./scripts/02_play_ball_kick_high.sh v5 1000               # v5 最新 run 的 model_1000.pt（实测最佳）
#   ./scripts/02_play_ball_kick_high.sh v4 23750 my.mp4       # v4 指定 iter + 输出名
#   ./scripts/02_play_ball_kick_high.sh v5 path/model.pt      # 直接给 checkpoint 路径
#   DRY_RUN=1 ./scripts/02_play_ball_kick_high.sh v5 1000     # 只打印命令不执行
#
# 【实测经验与现状】
#   - v5 实测最佳 = model_1000（mean ~70cm / max 96cm / 16/16 脚腿）；v4 model_23750 是
#     「脖子夹球」负教材（12cm 假挑高），回放对照用
#   - 2026-10-20 核实：logs/ 已整体清理，历史 checkpoint 不在——重训（01 脚本）后
#     本脚本自动定位最新 run；「末位 ≠ 最佳」经验来自 v4（23775 iter 末位远超 plateau），
#     建议传实测最佳 iter
#
# 【坑（practice README §7.4 源码核实）】
#   headless 下 play_high_kick.py 必崩（native 要 DISPLAY、viser 有 slider assert bug）——
#   无屏一律走 play_to_video.py；viser 在本任务不可用（区别于 104 的 G1 任务）
#   输出 mp4 已 gitignore；640×480@50fps 约 6s，含一次完整挑球

set -euo pipefail

VER="${1:-v5}"
ARG="${2:-}"
OUT="${3:-play_ball_kick.mp4}"
WS_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TASK_DIR="$WS_DIR/practice/tasks/ball_kick"
LOGS="$TASK_DIR/logs/rsl_rl"
PY="$WS_DIR/microduck_rl/.venv/bin/python"

case "$VER" in
  v4) EXP="ball_kick_aggr_right" ;;
  v5) EXP="ball_kick_clean_right" ;;
  *)  echo "用法: $0 [v4|v5] [iter 或 .pt 路径] [输出.mp4]"; exit 1 ;;
esac

# ---- 定位 checkpoint：给了 .pt 路径直接用；给了数字取最新 run 的 model_<n>.pt；不给取末位 ----
if [[ "$ARG" == *.pt ]]; then
  CKPT="$ARG"
else
  RUN_DIR=$(ls -dt "$LOGS/$EXP"/*/ 2>/dev/null | head -1) || true
  [[ -n "${RUN_DIR:-}" ]] || { echo "[x] $LOGS/$EXP 下没有任何 run，先跑 01_train_ball_kick_high.sh $VER"; exit 1; }
  if [[ -n "$ARG" ]]; then
    CKPT="$RUN_DIR/model_${ARG}.pt"
  else
    CKPT=$(ls "$RUN_DIR"model_*.pt 2>/dev/null | sort -t_ -k2 -n | tail -1) || true
    [[ -n "${CKPT:-}" ]] || { echo "[x] $RUN_DIR 下没有 model_*.pt"; exit 1; }
    echo "[w] 未指定 iter，自动取末位。实测末位 ≠ 最佳（v5 最佳 model_1000），看曲线挑（03 脚本）"
  fi
fi
[[ -f "$CKPT" ]] || { echo "[x] checkpoint 不存在: $CKPT"; exit 1; }
CKPT_REL="${CKPT#"$TASK_DIR"/}"     # play_to_video 相对 CWD 解析
CKPT_REL="${CKPT_REL//\/\//\/}"

MODE="video"   # headless 默认：离屏渲染 mp4
[[ -n "${DISPLAY:-}" ]] && MODE="native"
[[ "$CKPT" != "$TASK_DIR"/* ]] && OUT="$CKPT.mp4" && echo "[w] 外部 checkpoint，输出名忽略自定义（$OUT）"

echo "[i] 版本    : $VER（实验目录: $EXP）"
echo "[i] run     : $(dirname "$CKPT_REL")"
echo "[i] ckpt    : $(basename "$CKPT_REL")"
echo "[i] 模式    : $MODE（$([[ "$MODE" == "video" ]] && echo 'headless 离屏渲染 mp4' || echo 'DISPLAY 已接，原生交互 viewer')）"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[i] DRY_RUN，将执行:"
  if [[ "$MODE" == "video" ]]; then
    (cd "$TASK_DIR" && echo "MUJOCO_GL=egl $PY play_to_video.py $CKPT_REL $OUT")
  else
    (cd "$TASK_DIR" && echo "$PY play_high_kick.py --checkpoint-file $CKPT_REL --viewer native --num-envs 1")
  fi
  exit 0
fi

cd "$TASK_DIR"
export MUJOCO_GL=egl
if [[ "$MODE" == "video" ]]; then
  exec "$PY" play_to_video.py "$CKPT_REL" "$OUT"
else
  exec "$PY" play_high_kick.py --checkpoint-file "$CKPT_REL" --viewer native --num-envs 1
fi
