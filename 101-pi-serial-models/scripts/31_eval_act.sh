#!/usr/bin/env bash
# =============================================================================
# ACT Rollout 评估脚本（5090 机 + py3.12 + lerobot main v0.6.2 + gym-aloha）
# 在 ALOHA 仿真双臂环境（AlohaTransferCube-v0）中闭环 rollout，统计搬方块交接成功率。
# CLI 参数已按 v0.6.2 实测校验（与 32_eval_pi0.sh 同源；ACT 无需 rename_map——
# 策略从零训练，特征名直接来自数据集 observation.images.top，与 gym-aloha 环境输出一致）。
# 用法：
#   bash scripts/31_eval_act.sh                          # 默认 25 轮，无头 EGL 渲染
#   N_EPISODES=5 bash scripts/31_eval_act.sh             # 快速验证：先跑 5 轮
#   N_EPISODES=500 EVAL_BATCH=16 bash scripts/31_eval_act.sh   # 官方模型卡配置（500 轮，参考成功率 83%；本机实测约 22 分钟）
#   RENDER=glfw bash scripts/31_eval_act.sh              # 台式机带显示器：glfw 窗口渲染
#   POLICY_DIR=~/ws/models/act_aloha_sim_transfer_cube_human bash scripts/31_eval_act.sh
#                                                     # 评估 ModelScope 下载的官方预训练模型（需先 PRETRAINED=true 下载）
#
# 支持测评的任务共 3 个（ENV_TYPE x ENV_TASK 组合，策略必须与该任务数据集的训练产物匹配）：
# ① 搬方块交接（默认，本脚本即为此配置）
#   bash scripts/31_eval_act.sh
#   # 等价：ENV_TYPE=aloha ENV_TASK=AlohaTransferCube-v0 POLICY_DIR=outputs/act_aloha_cube/checkpoints/last/pretrained_model \
#   #       OUTPUT_DIR=outputs/eval_act_aloha_cube bash scripts/31_eval_act.sh
#   # 策略来源：21_train_act.sh 默认产物，或官方预训练模型 ~/ws/models/act_aloha_sim_transfer_cube_human
# ② 插孔装配（aloha 套件）
#   ENV_TASK=AlohaInsertion-v0 POLICY_DIR=outputs/act_aloha_insertion/checkpoints/last/pretrained_model \
#     OUTPUT_DIR=outputs/eval_act_aloha_insertion bash scripts/31_eval_act.sh
#   # 策略来源：需先换任务训练，见本注释块末行
# ③ 推块 PushT（单臂轻量任务，pusht 套件）
#   ENV_TYPE=pusht ENV_TASK=PushT-v0 POLICY_DIR=outputs/act_pusht/checkpoints/last/pretrained_model \
#     OUTPUT_DIR=outputs/eval_act_pusht bash scripts/31_eval_act.sh
#   # 策略来源：需先训练 DATASET_NAME=pusht OUTPUT_DIR=outputs/act_pusht bash scripts/21_train_act.sh
#
#   换任务训练：DATASET_NAME=aloha_sim_insertion_human OUTPUT_DIR=outputs/act_aloha_insertion bash scripts/21_train_act.sh
# =============================================================================
set -euo pipefail

# ---------- 渲染方式：无头/SSH 用 EGL GPU 渲染（默认）；带显示器可 RENDER=glfw ----------
if [[ "${RENDER:-egl}" == "egl" ]]; then
  export MUJOCO_GL=egl
fi

# ---------- 路径与可调参数（可用环境变量覆盖） ----------
ACT_HOME="$HOME/ws/101-pi-serial-models"
POLICY_DIR="${POLICY_DIR:-$ACT_HOME/outputs/act_aloha_cube/checkpoints/last/pretrained_model}"
ENV_TYPE="${ENV_TYPE:-aloha}"                 # gym 环境套件：aloha / pusht
ENV_TASK="${ENV_TASK:-AlohaTransferCube-v0}"  # aloha 套件可选：AlohaTransferCube-v0（搬方块交接）/ AlohaInsertion-v0（插孔）
OUTPUT_DIR="${OUTPUT_DIR:-$ACT_HOME/outputs/eval_act_aloha_cube}"
N_EPISODES="${N_EPISODES:-25}"      # 验收标准：25 轮 rollout 统计成功率（官方模型卡为 500 轮）
EVAL_BATCH="${EVAL_BATCH:-1}"       # 并行环境数；显存/CPU 富余可上探（官方用 50）

# ---------- 评估前自检 ----------
echo "===== 评估前自检 ====="
[[ -f "$POLICY_DIR/config.json" ]] || { echo "✗ 缺少 checkpoint 配置 $POLICY_DIR/config.json，请先完成训练（scripts/21_train_act.sh）或下载官方模型（PRETRAINED=true scripts/11_download_act_assets.sh）"; exit 1; }
command -v lerobot-eval >/dev/null || { echo "✗ 未找到 lerobot-eval，请先 source $ACT_HOME/.venv312/bin/activate"; exit 1; }
nvidia-smi --query-gpu=name,memory.used --format=csv || { echo "✗ nvidia-smi 失败，检查驱动"; exit 1; }
echo "✓ 自检通过（MUJOCO_GL=${MUJOCO_GL:-默认 glfw}，N_EPISODES=$N_EPISODES，POLICY_DIR=$POLICY_DIR）"
echo

# ---------- 启动评估 ----------
LOG_FILE="$ACT_HOME/logs/eval_act_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$ACT_HOME/logs"

echo "===== 启动 Rollout 评估（日志: $LOG_FILE） ====="
lerobot-eval \
  --policy.path="$POLICY_DIR" \
  --env.type="$ENV_TYPE" \
  --env.task="$ENV_TASK" \
  --eval.n_episodes="$N_EPISODES" \
  --eval.batch_size="$EVAL_BATCH" \
  --eval.use_async_envs=false \
  --output_dir="$OUTPUT_DIR" \
  --job_name=eval_act_aloha_cube \
  2>&1 | tee "$LOG_FILE"

# 注：
# - --eval.use_async_envs=false 必带（实测）：gym_aloha 0.1.4 无 gymnasium entry_points 注册，
#   环境注册仅靠 import gym_aloha（lerobot 只在主进程导入）；batch>1 默认启用 AsyncVectorEnv，
#   其 forkserver 子进程不继承该 import，报 NamespaceNotFound: gym_aloha。改 Sync 后在主进程建环境
# - 成功率统计：查看日志末尾的 rollout 汇总（pc_success 等指标）；
#   单轮成功判定：方块被一只臂抓起并交到另一只臂（gym-aloha 内置判定）
# - 每轮视频默认自动保存到 $OUTPUT_DIR/videos/aloha_*/eval_episode_N.mp4，人工抽检交接动作
# - 官方参考成绩：LeRobot 版 ACT 500 轮 83% 成功率（原 ACT 仓库实现 68%）
# - 与 32_eval_pi0.sh 的差异：ACT 不需要 --rename_map（pi0 基座期望 base_0_rgb，ACT 直接用 top）
