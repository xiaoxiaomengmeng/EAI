#!/usr/bin/env bash
# 09_train_g1_tracking_dance.sh —— G1 动作模仿（tracking / 跳舞）正式训练
#
# 【这是什么】
#   训练 G1 模仿真人动捕舞蹈 dance1_subject2：任务 Unitree-G1-Tracking-No-State-Estimation，
#   2048 并行环境，PPO，logger=tensorboard。命令与 docs/02 §6 阶段 4 已实测 run 完全一致。
#
# 【用法】
#   ./scripts/09_train_g1_tracking_dance.sh            # 默认 30000 iters，实测约 4.35 小时
#   ./scripts/09_train_g1_tracking_dance.sh 10000      # 指定迭代数
#   DRY_RUN=1 ./scripts/09_train_g1_tracking_dance.sh  # 只打印命令不执行
#
# 【动作数据】
#   仓库自带动捕 CSV，已转换为 NPZ（src/assets/motions/g1/dance1_subject2.npz，30fps → 50fps）。
#   若要换新动作：python scripts/csv_to_npz.py --input-file <csv> --output-name <名.npz> \
#     --input-fps 30 --output-fps 50 --robot g1，再改本脚本 MOTION 变量。
#
# 【产物】
#   repos/unitree_rl_mjlab/logs/rsl_rl/g1_tracking/<时间戳>/
#   看曲线：./scripts/13_launch_tensorboard.sh；出标注结论图：./scripts/14_plot_velocity_curve.py --task g1_tracking
#   中断续训：命令尾手动加 --agent.resume True
#
# 【本任务实测经验】
#   - 收敛健康：mean_reward -2.0 → 峰值 26.22 @ iter 29079，单调爬升无塌陷 → 最佳 checkpoint = model_29000
#   - Episode_Termination/ee_body_pos 400 → 0.29（超差强制终止基本消失）、episode 长度 462/500
#   - 与 velocity 对比：tracking 带 motion 采样 curriculum，收敛慢但曲线稳，后半程仍有增益
#   - play 效果 ./scripts/10_play_g1_tracking_dance.sh 29000

set -euo pipefail

ITERS="${1:-30000}"
TASK_NAME="Unitree-G1-Tracking-No-State-Estimation"   # 任务注册名（list_envs.py 可查）
NUM_ENVS=2048                  # 并行环境数
WS_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$WS_DIR/repos/unitree_rl_mjlab"
PY=~/miniconda3/envs/unitree_rl_mjlab/bin/python
MOTION="src/assets/motions/g1/dance1_subject2.npz"

echo "[i] 任务    : $TASK_NAME（G1 跳舞）"
echo "[i] 动捕   : $MOTION"
echo "[i] 工作目录: $REPO（import src 与 logs/ 均依赖当前目录）"
echo "[i] 迭代数  : $ITERS"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[i] DRY_RUN，将执行:"
  (cd "$REPO" && echo "$PY scripts/train.py $TASK_NAME --motion-file $MOTION --env.scene.num-envs $NUM_ENVS --agent.max-iterations $ITERS --agent.logger tensorboard")
  exit 0
fi

cd "$REPO"
exec "$PY" scripts/train.py "$TASK_NAME" \
  --motion-file "$MOTION" \
  --env.scene.num-envs "$NUM_ENVS" \
  --agent.max-iterations "$ITERS" \
  --agent.logger tensorboard
