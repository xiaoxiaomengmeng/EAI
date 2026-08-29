#!/usr/bin/env bash
# =============================================================================
# π₀ LoRA checkpoint Rollout 评估脚本（5090 机 + py3.12 + lerobot main v0.6.2）
# 配套 scripts/22_train_pi0.sh 训练产物：outputs/pi0_aloha_cube/checkpoints/last/pretrained_model
# CLI 参数已按 v0.6.2 实测校验：
#   - --policy.path 评估本地 checkpoint（与 --policy.type 互斥，类型从 checkpoint config 自动识别）
#   - --env.type=aloha + --env.task=AlohaTransferCube-v0（v0.4.4 的 --env.name 已不存在）
# 用法：
#   bash scripts/32_eval_pi0.sh                    # 默认 25 轮，无头 EGL 渲染
#   N_EPISODES=5 bash scripts/32_eval_pi0.sh       # 快速验证：先跑 5 轮
#   RENDER=glfw bash scripts/32_eval_pi0.sh        # 台式机带显示器：glfw 窗口渲染
# =============================================================================
set -euo pipefail

# ---------- 离线模式：与训练脚本一致，paligemma tokenizer 已注入 HF 缓存 ----------
export HF_HUB_OFFLINE=1

# ---------- 渲染方式：无头/SSH 用 EGL GPU 渲染（默认）；带显示器可 RENDER=glfw ----------
if [[ "${RENDER:-egl}" == "egl" ]]; then
  export MUJOCO_GL=egl
fi

# ---------- 路径与可调参数（可用环境变量覆盖） ----------
PI0_HOME="$HOME/ws/11_pi_serial_models"
POLICY_DIR="$PI0_HOME/outputs/pi0_aloha_cube/checkpoints/last/pretrained_model"
OUTPUT_DIR="$PI0_HOME/outputs/eval_pi0_aloha_cube"
N_EPISODES="${N_EPISODES:-25}"    # 指南验收标准：25 轮 rollout 统计成功率
BATCH_SIZE="${BATCH_SIZE:-1}"     # 单环境串行，显存/CPU 压力最小，结果可复现

# ---------- 评估前自检 ----------
echo "===== 评估前自检 ====="
[[ -f "$POLICY_DIR/config.json" ]] || { echo "✗ 缺少 checkpoint 配置 $POLICY_DIR/config.json，请先完成训练（scripts/22_train_pi0.sh）"; exit 1; }
command -v lerobot-eval >/dev/null || { echo "✗ 未找到 lerobot-eval，请先 source $PI0_HOME/.venv312/bin/activate"; exit 1; }
nvidia-smi --query-gpu=name,memory.used --format=csv || { echo "✗ nvidia-smi 失败，检查驱动"; exit 1; }
echo "✓ 自检通过（MUJOCO_GL=${MUJOCO_GL:-默认 glfw}，N_EPISODES=$N_EPISODES）"
echo

# ---------- 启动评估 ----------
LOG_FILE="$PI0_HOME/logs/eval_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$PI0_HOME/logs"

echo "===== 启动 Rollout 评估（日志: $LOG_FILE） ====="
lerobot-eval \
  --policy.path="$POLICY_DIR" \
  --env.type=aloha \
  --env.task=AlohaTransferCube-v0 \
  --eval.n_episodes="$N_EPISODES" \
  --eval.batch_size="$BATCH_SIZE" \
  --output_dir="$OUTPUT_DIR" \
  --job_name=eval_pi0_aloha_cube \
  --rename_map='{"observation.images.top":"observation.images.base_0_rgb"}' \
  2>&1 | tee "$LOG_FILE"

# 注：
# - rename_map 必带（实测）：环境产出 observation.images.top，而基座策略 config 期望
#   observation.images.base_0_rgb，不加会在 validate_visual_features_consistency 报
#   Feature mismatch（与训练时的 rename_map 同一思路）
# - 成功率统计：查看日志末尾的 rollout 汇总（pc_success 等指标）；
#   每轮视频默认自动保存到 $OUTPUT_DIR/videos/aloha_*/eval_episode_N.mp4；
#   --eval.recording=true 是另把 rollout 录成 LeRobot 数据集落盘
# - 验收标准（指南 7.3）：25 轮中方块被成功搬到右盘并保持稳定；
#   失败模式：抓空（定位）/交接掉落（协调）/放置偏移（精调不足）
