#!/usr/bin/env bash
# =============================================================================
# π₀ LoRA 微调启动脚本（5090 机 32GB + py3.12 + lerobot main v0.6.2 + torch cu130）
# 全部 CLI 参数已经 v0.6.2 实测校验（--policy.type + --policy.pretrained_path 组合有效；
# 注意与 --policy.path 互斥）；模型 checkpoint 为 v0.4.4 格式，v0.6.2 加载器实测向后兼容
# 用法：
#   bash scripts/train_pi0.sh                 # 默认 20000 步、batch 2
#   STEPS=200 BATCH_SIZE=1 bash scripts/train_pi0.sh   # 冒烟测试：先跑 200 步验证流程
#   BATCH_SIZE=4 bash scripts/train_pi0.sh    # 5090(32GB) LoRA 可用 4，OOM 降 2/1
#   LORA_R=32 STEPS=160000 BATCH_SIZE=4 bash scripts/train_pi0.sh  # 160k 步、batch 4、LoRA R=32
#   WANDB_ENABLE=true LORA_R=32 STEPS=160000 BATCH_SIZE=4 bash scripts/train_pi0.sh  # 同上 + wandb 可视化
# =============================================================================
set -euo pipefail

# ---------- 离线模式：paligemma tokenizer 已注入 HF 缓存（ModelScope 下载），
# 国内无法访问 huggingface.co，不设此变量会在加载 tokenizer 时挂起超时 ----------
export HF_HUB_OFFLINE=1

# ---------- 路径与可调参数（可用环境变量覆盖） ----------
PI0_HOME="$HOME/ws/11_pi_serial_models"
DATASET_DIR="$PI0_HOME/data/aloha_sim_transfer_cube_human"   # 数据集目录本身（含 meta/info.json），不是父目录
MODEL_DIR="$PI0_HOME/models/pi0_base"                        # π₀ 基座权重（14GB，F32 保存）
OUTPUT_DIR="$PI0_HOME/outputs/pi0_aloha_cube"                # checkpoint 与日志输出目录（非 resume 时若已存在会报 FileExistsError，需先删或改名）
STEPS="${STEPS:-20000}"          # 总训练步数：50 条演示 × 400 帧，20k 步约 4 个 epoch
BATCH_SIZE="${BATCH_SIZE:-2}"    # 5090(32GB) LoRA 建议 2 起步，富余可升 4

# ---------- wandb 实验跟踪（loss 曲线可视化） ----------
# wandb 是机器学习实验跟踪平台，开启后自动画 loss/学习率/梯度等曲线图。
# 国内连不上 wandb.ai 时用离线模式：设 WANDB_MODE=offline，生成本地 HTML 报告，不联网。
# 开启方式：WANDB_ENABLE=true bash scripts/train_pi0.sh
WANDB_ENABLE="${WANDB_ENABLE:-false}"
if [[ "$WANDB_ENABLE" == "true" ]]; then
  export WANDB_MODE="${WANDB_MODE:-offline}"   # 默认离线模式，不需要联网
  export WANDB_DIR="${WANDB_DIR:-$PI0_HOME/wandb}"
  mkdir -p "$WANDB_DIR"
  echo "✓ wandb 已开启（mode=${WANDB_MODE}, dir=${WANDB_DIR}）"
  WANDB_ARGS="--wandb.enable=true --wandb.project=pi0_aloha --wandb.mode=offline"
else
  WANDB_ARGS="--wandb.enable=false"
fi

# ---------- 训练前自检 ----------
echo "===== 训练前自检 ====="
[[ -f "$MODEL_DIR/model.safetensors" ]] || { echo "✗ 缺少基座权重 $MODEL_DIR/model.safetensors"; exit 1; }
[[ -f "$DATASET_DIR/meta/info.json"   ]] || { echo "✗ 缺少数据集元信息 $DATASET_DIR/meta/info.json"; exit 1; }
command -v lerobot-train >/dev/null     || { echo "✗ 未找到 lerobot-train，请先 source $PI0_HOME/.venv312/bin/activate"; exit 1; }
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv || { echo "✗ nvidia-smi 失败，检查驱动"; exit 1; }
AVAIL_MEM_MB=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
echo "可用内存: ${AVAIL_MEM_MB}MB"
PALIGEMMA_TOK="$HOME/.cache/huggingface/hub/models--google--paligemma-3b-pt-224/snapshots/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/tokenizer.json"
[[ -f "$PALIGEMMA_TOK" ]] || { echo "✗ 缺少 paligemma tokenizer 缓存（$PALIGEMMA_TOK），请先从 ModelScope 下载注入"; exit 1; }
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
  $WANDB_ARGS \
  --rename_map='{"observation.images.top":"observation.images.base_0_rgb"}' \
  --peft.method_type=LORA \
  --peft.r="${LORA_R:-16}" \
  --steps="$STEPS" \
  --batch_size="$BATCH_SIZE" \
  2>&1 | tee "$LOG_FILE"

# 注：
# - --policy.push_to_hub=false 必带：v0.6.2 默认推 Hub，缺失 repo_id 时 validate 阶段报
#   "'repo_id' argument missing"（实测）；wandb 默认关闭，用 WANDB_ENABLE=true 开启
# - rename_map 把数据集的 observation.images.top 改名对齐基座模型的 base_0_rgb 输入名，
#   v0.6.2 要求 rename_map 必须配合 pretrained_path 使用（本脚本已满足）
# - 中断后续训：lerobot-train --resume=true --config_path=$OUTPUT_DIR/checkpoints/<step>/train_config.json
#   （v0.6.2 注释：resume 默认沿用 checkpoint 内的配置）
# - 显存吃紧时可追加 --policy.gradient_checkpointing=true（官方 pi0.mdx 推荐）
# - HF_HUB_OFFLINE=1：pi0 的 processor 需从 google/paligemma-3b-pt-224 加载 tokenizer，
#   该模型在 HF 上为 Gemma 许可 gated 仓库（hf-mirror 也需授权），已从 ModelScope 下载
#   tokenizer 小文件注入 HF 缓存（refs/main 指向伪造 commit hash 目录），离线可用
