#!/usr/bin/env bash
# 07_train_g1_velocity_flat.sh —— G1 速度跟随（velocity flat）正式训练
#
# 【这是什么】
#   训练 G1 平地速度跟随策略：任务 Unitree-G1-Flat，2048 并行环境，PPO，logger=tensorboard。
#   命令与 docs/02 §6 阶段 3 已实测 run（2026-10-19_17-23-46）完全一致。
#
# 【用法】
#   ./scripts/07_train_g1_velocity_flat.sh            # 默认 10001 iters，实测约 94 分钟
#   ./scripts/07_train_g1_velocity_flat.sh 5500       # 实测充分收敛点约 5000 iters，传 5500 省近一半时间
#   DRY_RUN=1 ./scripts/07_train_g1_velocity_flat.sh  # 只打印命令不执行
#
# 【产物】
#   repos/unitree_rl_mjlab/logs/rsl_rl/g1_velocity/<时间戳>/
#   （model_*.pt 每 100 iter + policy.onnx + params/{env,agent}.yaml）
#   看曲线：./scripts/13_launch_tensorboard.sh；出标注结论图：./scripts/14_plot_velocity_curve.py
#   中断续训：命令尾手动加 --agent.resume True（自动取最新 checkpoint，写新目录不覆盖）
#   首次运行 mujoco-warp JIT 编译约 30s，之后走 ~/.cache/warp 缓存
#
# 【本任务实测经验】
#   - 无课程（env_cfgs.py: cfg.curriculum = {}，指令范围全程固定）：reward 断崖下跌≠任务变难
#   - 峰值 39.37 @ iter 4965，5000→6000 策略塌陷（跌 18 分）后仅恢复七成 → 最佳 checkpoint = model_5000
#   - play 效果 ./scripts/08_play_g1_velocity_flat.sh 5000

set -euo pipefail

ITERS="${1:-10001}"
TASK_NAME="Unitree-G1-Flat"   # 任务注册名（list_envs.py 可查）
NUM_ENVS=2048                  # 并行环境数
WS_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$WS_DIR/repos/unitree_rl_mjlab"
PY=~/miniconda3/envs/unitree_rl_mjlab/bin/python

echo "[i] 任务    : $TASK_NAME（G1 速度跟随，平地）"
echo "[i] 工作目录: $REPO（import src 与 logs/ 均依赖当前目录）"
echo "[i] 迭代数  : $ITERS"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[i] DRY_RUN，将执行:"
  (cd "$REPO" && echo "$PY scripts/train.py $TASK_NAME --env.scene.num-envs $NUM_ENVS --agent.max-iterations $ITERS --agent.logger tensorboard")
  exit 0
fi

cd "$REPO"
exec "$PY" scripts/train.py "$TASK_NAME" \
  --env.scene.num-envs "$NUM_ENVS" \
  --agent.max-iterations "$ITERS" \
  --agent.logger tensorboard
