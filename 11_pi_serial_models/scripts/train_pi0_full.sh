#!/usr/bin/env bash
# =============================================================================
# π₀ 全参微调脚本（5090 32GB + py3.12 + lerobot main v0.6.2 + torch cu130）
# 与 train_pi0.sh（LoRA）的核心区别：
#   1. 不用 PEFT/LoRA，训练全部 2.3B 参数（PaliGemma 2B + Gemma Expert 300M）
#   2. 开启 gradient_checkpointing 节省显存（用计算换显存）
#   3. 默认 batch_size=2（LoRA 脚本默认 4），显存峰值 ~30GB 刚好不 OOM
#   4. 默认 steps=200000（batch 2 × 200k = 400k 样本 = 20 epoch）
#
# 显存预估（5090 32GB）：
#   模型权重 bf16:        ~4.6 GB
#   梯度 bf16:            ~4.6 GB
#   AdamW 优化器 (m+v):   ~18.4 GB  ← 占大头
#   激活值 (grad ckpt):   ~2-4 GB
#   合计:                 ~30-32 GB（紧凑但可行）
#
# 如果 OOM：
#   1. 降 batch: BATCH_SIZE=1 bash scripts/train_pi0_full.sh
#   2. 冻结视觉编码器: FREEZE_VISION=true bash scripts/train_pi0_full.sh（省 ~1GB）
#   3. 只训 Expert: TRAIN_EXPERT_ONLY=true（冻结 PaliGemma，只训 300M Expert，省 ~16GB）
#
# 用法：
#   bash scripts/train_pi0_full.sh                            # 默认 200k 步、batch 2
#   BATCH_SIZE=1 bash scripts/train_pi0_full.sh              # OOM 时降 batch
#   STEPS=100000 bash scripts/train_pi0_full.sh              # 快速试跑
#   WANDB_ENABLE=true bash scripts/train_pi0_full.sh         # 开启 wandb 可视化
# =============================================================================
set -euo pipefail

# ---------- 离线模式：与 train_pi0.sh 一致 ----------
export HF_HUB_OFFLINE=1

# ---------- 路径与可调参数 ----------
PI0_HOME="$HOME/ws/11_pi_serial_models"
DATASET_DIR="$PI0_HOME/data/aloha_sim_transfer_cube_human"
MODEL_DIR="$PI0_HOME/models/pi0_base"
OUTPUT_DIR="$PI0_HOME/outputs/pi0_aloha_cube_full"    # ← 与 LoRA 脚本不同，避免覆盖
STEPS="${STEPS:-200000}"          # 200k 步 × batch 2 = 400k 样本 = 20 epoch
BATCH_SIZE="${BATCH_SIZE:-2}"     # 5090 32GB 全参微调建议 2，OOM 降到 1
FREEZE_VISION="${FREEZE_VISION:-false}"   # true 冻结 SigLIP 视觉编码器（省 ~1GB 显存）
TRAIN_EXPERT_ONLY="${TRAIN_EXPERT_ONLY:-false}"  # true 只训 Gemma Expert（省 ~16GB 显存）
WANDB_ENABLE="${WANDB_ENABLE:-false}"      # true 开启 wandb 离线可视化

# ---------- 训练前自检 ----------
echo "===== 训练前自检（全参微调） ====="
[[ -f "$MODEL_DIR/model.safetensors" ]] || { echo "✗ 缺少基座权重 $MODEL_DIR/model.safetensors"; exit 1; }
[[ -f "$DATASET_DIR/meta/info.json"   ]] || { echo "✗ 缺少数据集元信息 $DATASET_DIR/meta/info.json"; exit 1; }
command -v lerobot-train >/dev/null     || { echo "✗ 未找到 lerobot-train，请先 source $PI0_HOME/.venv312/bin/activate"; exit 1; }
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv || { echo "✗ nvidia-smi 失败，检查驱动"; exit 1; }
AVAIL_MEM_MB=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
echo "可用内存: ${AVAIL_MEM_MB}MB"
PALIGEMMA_TOK="$HOME/.cache/huggingface/hub/models--google--paligemma-3b-pt-224/snapshots/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/tokenizer.json"
[[ -f "$PALIGEMMA_TOK" ]] || { echo "✗ 缺少 paligemma tokenizer 缓存，请先从 ModelScope 下载注入"; exit 1; }
if [[ $AVAIL_MEM_MB -lt 18000 ]]; then
  echo "⚠ 可用内存不足 18GB：加载 14GB F32 权重峰值约 21-29GB，可能被 OOM killer 杀死。"
  echo "  请先关闭浏览器/编辑器释放内存，或扩容 swap。"
  read -r -p "仍要继续吗？[y/N] " ans; [[ "$ans" == "y" || "$ans" == "Y" ]] || exit 1
fi

# 全参微调显存警告
GPU_MEM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
if [[ $GPU_MEM_MB -lt 30000 ]]; then
  echo "⚠ GPU 显存 ${GPU_MEM_MB}MB < 30GB，全参微调可能 OOM。"
  echo "  建议：BATCH_SIZE=1 或 TRAIN_EXPERT_ONLY=true"
  read -r -p "仍要继续吗？[y/N] " ans; [[ "$ans" == "y" || "$ans" == "Y" ]] || exit 1
fi

echo "✓ 自检通过"
echo "  训练模式: 全参微调（无 LoRA）"
echo "  Steps: $STEPS, Batch: $BATCH_SIZE"
echo "  Freeze Vision: $FREEZE_VISION, Train Expert Only: $TRAIN_EXPERT_ONLY"
echo "  WandB: $WANDB_ENABLE"
echo

# ---------- 构建 PEFT 参数（全参微调时不传 --peft.*） ----------
# 注意：不传 --peft.method_type 即不做 PEFT，训练全部参数

# ---------- 构建 WandB 参数 ----------
WANDB_ARGS=""
if [[ "$WANDB_ENABLE" == "true" ]]; then
  export WANDB_MODE=offline
  export WANDB_DIR="$PI0_HOME/wandb"
  mkdir -p "$WANDB_DIR"
  WANDB_ARGS="--wandb.enable=true --wandb.mode=offline --wandb.project=pi0_aloha_full"
fi

# ---------- 启动训练 ----------
LOG_FILE="$PI0_HOME/logs/train_full_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$PI0_HOME/logs"

echo "===== 启动全参微调（日志: $LOG_FILE） ====="
lerobot-train \
  --dataset.repo_id=aloha_sim_transfer_cube_human \
  --dataset.root="$DATASET_DIR" \
  --output_dir="$OUTPUT_DIR" \
  --job_name=pi0_aloha_cube_full \
  --policy.type=pi0 \
  --policy.pretrained_path="$MODEL_DIR" \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.use_peft=false \
  --policy.gradient_checkpointing=true \
  --policy.freeze_vision_encoder="$FREEZE_VISION" \
  --policy.train_expert_only="$TRAIN_EXPERT_ONLY" \
  $WANDB_ARGS \
  --rename_map='{"observation.images.top":"observation.images.base_0_rgb"}' \
  --steps="$STEPS" \
  --batch_size="$BATCH_SIZE" \
  2>&1 | tee "$LOG_FILE"

# 注：
# - --policy.use_peft=false: 关闭 PEFT，训练全部 2.3B 参数
# - --policy.gradient_checkpointing=true: 开启梯度检查点，用计算换显存（必带）
# - --policy.freeze_vision_encoder: 可选冻结 SigLIP（~400M 参数），省 ~1GB 显存
# - --policy.train_expert_only: 可选只训 Gemma Expert（~300M 参数），省 ~16GB 显存
#   （如果开启，等效于"只训 Expert 全参"而非 LoRA，比 LoRA r=32 多 ~100 倍可训参数）
# - 不传 --peft.method_type: 不做 PEFT，与 --policy.use_peft=false 双保险
# - 全参 checkpoint 体积 ~4.6GB（vs LoRA 5.5MB），save_freq 默认 20000 不变
# - 评估时 POLICY_DIR 指向 outputs/pi0_aloha_cube_full/checkpoints/last/pretrained_model
#   示例: POLICY_DIR=outputs/pi0_aloha_cube_full/checkpoints/last/pretrained_model bash scripts/eval_pi0.sh
