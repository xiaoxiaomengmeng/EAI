#!/usr/bin/env bash
# =============================================================================
# π₀.₅ LIBERO checkpoint Rollout 评估脚本（5090 机 + py3.12 + lerobot main v0.6.2）
# 配套 scripts/train_pi05_libero.sh 训练产物：
#   - LoRA 模式：outputs/pi05_libero_lora/checkpoints/last/pretrained_model
#   - Expert 模式：outputs/pi05_libero_expert/checkpoints/last/pretrained_model
#
# LIBERO 4 个子集（通过 LIBERO_TASK 切换，对标官方 97.5% benchmark）：
#   - libero_spatial  (max 280 steps / episode，最简单，建议从这里开始)
#   - libero_object   (max 280 steps / episode)
#   - libero_goal     (max 300 steps / episode)
#   - libero_10       (max 520 steps / episode，最难，官方默认)
#
# 用法：
#   bash scripts/eval_pi05_libero.sh                            # 默认：LoRA checkpoint + libero_spatial + 50 轮
#   TRAIN_MODE=expert bash scripts/eval_pi05_libero.sh          # 切到 expert checkpoint
#   LIBERO_TASK=libero_10 bash scripts/eval_pi05_libero.sh      # 换 LIBERO-10 子集（最难）
#   N_EPISODES=10 bash scripts/eval_pi05_libero.sh              # 快速验证：先跑 10 轮
#   RENDER=glfw bash scripts/eval_pi05_libero.sh                # 台式机带显示器：glfw 窗口渲染
#   TRAIN_MODE=expert LIBERO_TASK=libero_goal bash scripts/eval_pi05_libero.sh   # 组合用
# =============================================================================
set -euo pipefail

# ---------- 离线模式：与训练脚本一致，paligemma tokenizer 已注入 HF 缓存 ----------
export HF_HUB_OFFLINE=1

# ---------- 渲染方式：无头/SSH 用 EGL GPU 渲染（默认）；带显示器可 RENDER=glfw ----------
if [[ "${RENDER:-egl}" == "egl" ]]; then
  export MUJOCO_GL=egl
fi

# ---------- 路径与可调参数 ----------
PI05_HOME="$HOME/ws/11_pi_serial_models"
TRAIN_MODE="${TRAIN_MODE:-lora}"       # lora（默认）| expert
LIBERO_TASK="${LIBERO_TASK:-libero_spatial}"  # libero_spatial（默认）/ libero_object / libero_goal / libero_10

# 根据 TRAIN_MODE 推导 POLICY_DIR 和 OUTPUT_DIR（与训练脚本对称）
case "$TRAIN_MODE" in
  lora)
    POLICY_DIR="$PI05_HOME/outputs/pi05_libero_lora/checkpoints/last/pretrained_model"
    OUTPUT_DIR="$PI05_HOME/outputs/eval_pi05_libero_${LIBERO_TASK}_lora"
    JOB_NAME="eval_pi05_libero_${LIBERO_TASK}_lora"
    ;;
  expert)
    POLICY_DIR="$PI05_HOME/outputs/pi05_libero_expert/checkpoints/last/pretrained_model"
    OUTPUT_DIR="$PI05_HOME/outputs/eval_pi05_libero_${LIBERO_TASK}_expert"
    JOB_NAME="eval_pi05_libero_${LIBERO_TASK}_expert"
    ;;
  *)
    echo "✗ 未知 TRAIN_MODE=$TRAIN_MODE，支持: lora | expert"; exit 1 ;;
esac

N_EPISODES="${N_EPISODES:-50}"     # 官方 benchmark：每子集 50 轮
BATCH_SIZE="${BATCH_SIZE:-1}"      # 单环境串行，显存/CPU 压力最小，结果可复现

# ---------- 评估前自检 ----------
echo "===== 评估前自检（π₀.₅ LIBERO） ====="
[[ -f "$POLICY_DIR/config.json" ]] || {
  echo "✗ 缺少 checkpoint 配置 $POLICY_DIR/config.json"
  echo "  请先完成训练：TRAIN_MODE=$TRAIN_MODE bash scripts/train_pi05_libero.sh"
  exit 1
}
command -v lerobot-eval >/dev/null || { echo "✗ 未找到 lerobot-eval，请先 source $PI05_HOME/.venv312/bin/activate"; exit 1; }
nvidia-smi --query-gpu=name,memory.used --format=csv || { echo "✗ nvidia-smi 失败，检查驱动"; exit 1; }
echo "✓ 自检通过"
echo "  训练模式: $TRAIN_MODE"
echo "  Checkpoint: $POLICY_DIR"
echo "  LIBERO 子集: $LIBERO_TASK"
echo "  渲染: MUJOCO_GL=${MUJOCO_GL:-默认 glfw}"
echo "  N_EPISODES: $N_EPISODES"
echo

# ---------- 启动评估 ----------
LOG_FILE="$PI05_HOME/logs/eval_pi05_libero_${LIBERO_TASK}_${TRAIN_MODE}_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$PI05_HOME/logs"

echo "===== 启动 Rollout 评估（日志: $LOG_FILE） ====="
lerobot-eval \
  --policy.path="$POLICY_DIR" \
  --env.type=libero \
  --env.task="$LIBERO_TASK" \
  --eval.n_episodes="$N_EPISODES" \
  --eval.batch_size="$BATCH_SIZE" \
  --output_dir="$OUTPUT_DIR" \
  --job_name="$JOB_NAME" \
  2>&1 | tee "$LOG_FILE"

# 注：
# - --policy.path：从 checkpoint 自动加载 policy.type=pi05 + normalization_mapping + n_action_steps
#   + empty_cameras 等所有训练时配置，无需重复传（与 pi0 eval 同款机制）
# - --env.type=libero：lerobot 注册的 LIBERO env（基于 robosuite + MuJoCo）
# - --env.task：4 个子集任选 1 个（libero_spatial / libero_object / libero_goal / libero_10）
#   官方 benchmark 4 子集各跑 50 轮取平均，本脚本单次跑 1 个子集，4 次跑完用不同 LIBERO_TASK
# - 不传 --rename_map：LIBERO env 产出的 observation.images.image / image2 已与 pi05 基座期望
#   匹配（与训练时一致，不需要重命名）
# - MUJOCO_GL=egl：无头 GPU 渲染（默认，SSH/无显示器场景）；RENDER=glfw 切到窗口渲染
# - 验收标准（对标官方 97.5%）：libero_spatial 97% / libero_object 99% / libero_goal 98% / libero_10 96%
# - 4 子集跑全流程示例：
#   for task in libero_spatial libero_object libero_goal libero_10; do
#     LIBERO_TASK=$task bash scripts/eval_pi05_libero.sh
#   done
