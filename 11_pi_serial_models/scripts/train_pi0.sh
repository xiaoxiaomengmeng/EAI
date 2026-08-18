#!/usr/bin/env bash
# =============================================================================
# π₀ LoRA 微调启动脚本（方案一：4090 机 + py3.10 + lerobot v0.4.4 + torch cu128）
# 用法：
#   bash train_pi0.sh                 # 默认 20000 步、batch 2
#   STEPS=200 BATCH_SIZE=1 bash train_pi0.sh   # 冒烟测试：先跑 200 步验证流程
#   BATCH_SIZE=4 bash train_pi0.sh    # 方案三（5090 机 32GB）可用 4
# =============================================================================
set -euo pipefail

# ---------- 路径与可调参数（可用环境变量覆盖） ----------
PI0_HOME="$HOME/xmm_ws/pi0"
DATASET_DIR="$PI0_HOME/data/aloha_sim_transfer_cube_human"   # 数据集目录本身（含 meta/info.json），不是父目录
MODEL_DIR="$PI0_HOME/models/pi0_base"                        # π₀ 基座权重（14GB，F32 保存）
OUTPUT_DIR="$PI0_HOME/outputs/pi0_aloha_cube"                # checkpoint 与日志输出目录
STEPS="${STEPS:-20000}"          # 总训练步数：50 条演示 × 400 帧，20k 步约 4 个 epoch
BATCH_SIZE="${BATCH_SIZE:-2}"    # 4090(24GB) 建议 2，OOM 降 1；5090(32GB) 可用 4

# ---------- 训练前自检 ----------
echo "===== 训练前自检 ====="
[[ -f "$MODEL_DIR/model.safetensors" ]] || { echo "✗ 缺少基座权重 $MODEL_DIR/model.safetensors"; exit 1; }
[[ -f "$DATASET_DIR/meta/info.json"   ]] || { echo "✗ 缺少数据集元信息 $DATASET_DIR/meta/info.json"; exit 1; }
command -v lerobot-train >/dev/null     || { echo "✗ 未找到 lerobot-train，请先 source ~/xmm_ws/pi0/venv/bin/activate"; exit 1; }
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv || { echo "✗ nvidia-smi 失败，检查驱动"; exit 1; }
AVAIL_MEM_MB=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
echo "可用内存: ${AVAIL_MEM_MB}MB"
if [[ $AVAIL_MEM_MB -lt 18000 ]]; then
  echo "⚠ 可用内存不足 18GB：加载 14GB F32 权重峰值约 21-29GB，会被 OOM killer 杀死（实测）。"
  echo "  请先关闭浏览器/编辑器释放内存，并扩容 swap（如 sudo fallocate -l 24G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile）"
  read -r -p "仍要继续吗？[y/N] " ans; [[ "$ans" == "y" || "$ans" == "Y" ]] || exit 1
fi
echo "✓ 自检通过"
echo

# ---------- 启动训练 ----------
LOG_FILE="$PI0_HOME/logs/train_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$PI0_HOME/logs"

echo "===== 启动训练（日志: $LOG_FILE） ====="
lerobot-train \
  --dataset.repo_id=aloha_sim_transfer_cube_human \
  --dataset.root="$DATASET_DIR" \
  --output_dir="$OUTPUT_DIR" \
  --job_name=pi0_aloha_cube \
  --policy.type=pi0 \
  --policy.pretrained_path="$MODEL_DIR" \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --rename_map='{"observation.images.top":"observation.images.base_0_rgb"}' \
  --peft.method_type=LORA \
  --peft.r=16 \
  --steps="$STEPS" \
  --batch_size="$BATCH_SIZE" \
  2>&1 | tee "$LOG_FILE"

# 注：中断后续训用 lerobot-train --output_dir=$OUTPUT_DIR --resume=true（其余参数从 checkpoint 恢复）
