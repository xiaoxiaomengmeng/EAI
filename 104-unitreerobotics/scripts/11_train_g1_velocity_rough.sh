#!/usr/bin/env bash
# 11_train_g1_velocity_rough.sh —— G1 粗糙地形速度跟随（velocity rough）正式训练【带课程】
#
# 【这是什么】
#   训练 G1 在粗糙地形上的速度跟随策略：任务 Unitree-G1-Rough，2048 并行环境，PPO。
#   与 07（velocity flat）同源，实质差异两点（源码 env_cfgs.py / docs/03）：
#     ① 地形课程开启：terrain_generator.curriculum = True——机器人走得稳就「升级」到更难的地形砖块
#        （平地→粗糙→台阶），摔倒降级。这是本目录里第一个真正带课程的 velocity 任务
#     ② 观测多 height_scan：pelvis 周边地形高度射线扫描进观测（rough 特有，flat 已删）
#   注意：命令课程同样是空的（cfg.curriculum = {} 只清命令课程，地形课程保留）
#
# 【状态：待验证】
#   本任务未实测（docs/02 只实测过 Flat/Tracking）。首跑建议先 smoke 验证链路：
#     DRY_RUN=1 ./scripts/11_train_g1_velocity_rough.sh 50   # 干跑看命令
#     ./scripts/11_train_g1_velocity_rough.sh 50             # 50 iters 链路通后再放全量
#   时长未实测：同规模 flat 为 94 分钟，rough 地形生成 + height_scan 观测更重，预计略长（待核）
#
# 【用法】
#   ./scripts/11_train_g1_velocity_rough.sh            # 默认 10001 iters
#   ./scripts/11_train_g1_velocity_rough.sh 50         # smoke
#   DRY_RUN=1 ./scripts/11_train_g1_velocity_rough.sh  # 只打印命令不执行
#
# 【experiment 隔离（重要）】
#   上游 rough/flat 共用 experiment_name=g1_velocity，run 混在同一个目录会串台——
#   本脚本显式加 --agent.experiment-name g1_velocity_rough（CLI 覆盖，不改上游），
#   日志独立落 logs/rsl_rl/g1_velocity_rough/，play（12 脚本）按此目录定位，互不干扰。
#
# 【产物与观察】
#   落盘：repos/unitree_rl_mjlab/logs/rsl_rl/g1_velocity_rough/<时间戳>/
#   看曲线：./scripts/13_launch_tensorboard.sh；出图：./scripts/14_plot_velocity_curve.py --task g1_velocity_rough
#   【课程判读（与 flat 相反！）】曲线台阶式下行 = 课程在切换（合法难度递增，会爬回来）；
#   断崖式下跌 + 部分恢复才是策略失稳。rough 有真课程，别看到 reward 下降就当塌陷。

set -euo pipefail

ITERS="${1:-10001}"
TASK_NAME="Unitree-G1-Rough"   # 任务注册名（list_envs.py 可查）
NUM_ENVS=2048                  # 并行环境数
EXPERIMENT="g1_velocity_rough" # 日志目录名（覆盖上游默认 g1_velocity，与 flat 隔离）
WS_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$WS_DIR/repos/unitree_rl_mjlab"
PY=~/miniconda3/envs/unitree_rl_mjlab/bin/python

echo "[i] 任务    : $TASK_NAME（G1 粗糙地形速度跟随，带地形课程）"
echo "[i] 工作目录: $REPO（import src 与 logs/ 均依赖当前目录）"
echo "[i] 迭代数  : $ITERS（实验目录: $EXPERIMENT）"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[i] DRY_RUN，将执行:"
  (cd "$REPO" && echo "$PY scripts/train.py $TASK_NAME --env.scene.num-envs $NUM_ENVS --agent.max-iterations $ITERS --agent.logger tensorboard --agent.experiment-name $EXPERIMENT")
  exit 0
fi

cd "$REPO"
exec "$PY" scripts/train.py "$TASK_NAME" \
  --env.scene.num-envs "$NUM_ENVS" \
  --agent.max-iterations "$ITERS" \
  --agent.logger tensorboard \
  --agent.experiment-name "$EXPERIMENT"
