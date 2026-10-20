#!/usr/bin/env bash
# 01_train_ball_kick_high.sh —— microduck 高踢球（ball_kick high kick）训练
#
# 【这是什么】
#   practice/tasks/ball_kick 的自研高踢任务（Mjlab-BallKickHigh-Flat-MicroDuck）：
#   在官方 ball_kick 配方上做 shim 覆盖（垫高细桩 + 挑高 reward），不改 microduck_rl 子模块。
#   reward 版本由环境变量 HIGH_KICK_REWARD_MODE 切换，本脚本参数化：
#     v5（默认）= 机制感知门控（废掉脖子夹球 exploit）→ 实验目录 ball_kick_clean_right
#                实测终局：4096 envs × 1000 iter → 干净踢射 mean ~70cm / max 96cm，机制 16/16 脚腿
#     v4       = 机制盲（复现「12cm 脖子夹球」对照线）  → 实验目录 ball_kick_aggr_right
#                实测终局：4096 envs × 23875 iter → 12cm 但 it3000 即 plateau，且为 reward hacking
#
# 【用法】
#   ./scripts/01_train_ball_kick_high.sh                # v5，默认 1000 iters（实测已足够达标）
#   ./scripts/01_train_ball_kick_high.sh v5 2000        # v5 指定迭代数
#   ./scripts/01_train_ball_kick_high.sh v4             # v4，默认 23875 iters（对照实验，很久）
#   ./scripts/01_train_ball_kick_high.sh v5 1 64        # 第 3 参数 num_envs：64 envs × 1 iter 冒烟
#   DRY_RUN=1 ./scripts/01_train_ball_kick_high.sh      # 只打印命令不执行
#
# 【环境说明（2026-10-20 核实）】
#   practice/README 写的根 .venv（../../../.venv）已不存在——唯一可用环境是
#   microduck_rl/.venv（uv 管理，py3.12，import 链已实测：注册 + cfg 组装通过）。
#   本脚本固定用该环境；README 里的旧路径属文档漂移。
#
# 【产物与观察】
#   落盘：practice/tasks/ball_kick/logs/rsl_rl/<实验名>/<时间戳>/
#         （model_N.pt + *.onnx + tfevents + params/）+ ./wandb/offline-run-*
#   看曲线：./scripts/03_launch_tensorboard.sh（v4/v5 两条线同时看）
#   回放：./scripts/02_play_ball_kick_high.sh
#   坑：日志控制台字段逐项解读见 practice/tasks/ball_kick/README.md §7.5；
#       mean_reward 比 v3 低是预期（站姿被主动砍了），主指标是 Episode_Reward/ball_lift

set -euo pipefail

VER="${1:-v5}"
WS_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TASK_DIR="$WS_DIR/practice/tasks/ball_kick"
PY="$WS_DIR/microduck_rl/.venv/bin/python"
NUM_ENVS=4096    # 并行环境数（实测口径）

case "$VER" in
  v4) DEFAULT_ITERS=23875; EXP="ball_kick_aggr_right" ;;
  v5) DEFAULT_ITERS=1000;  EXP="ball_kick_clean_right" ;;
  *)  echo "用法: $0 [v4|v5] [iters] [num_envs]"; exit 1 ;;
esac
ITERS="${2:-$DEFAULT_ITERS}"
NUM_ENVS="${3:-$NUM_ENVS}"

echo "[i] 版本    : $VER（实验目录: $EXP）"
echo "[i] 工作目录: $TASK_DIR（入口依赖 CWD，日志落 ./logs）"
echo "[i] 规模    : $NUM_ENVS envs × $ITERS iters"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[i] DRY_RUN，将执行:"
  (cd "$TASK_DIR" && echo "WANDB_MODE=offline MUJOCO_GL=egl HIGH_KICK_REWARD_MODE=$VER $PY train_high_kick.py --env.scene.num-envs $NUM_ENVS --agent.max-iterations $ITERS")
  exit 0
fi

cd "$TASK_DIR"
export WANDB_MODE=offline MUJOCO_GL=egl HIGH_KICK_REWARD_MODE="$VER"
exec "$PY" train_high_kick.py \
  --env.scene.num-envs "$NUM_ENVS" \
  --agent.max-iterations "$ITERS"
