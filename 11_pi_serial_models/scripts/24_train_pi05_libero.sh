#!/usr/bin/env bash
# =============================================================================
# π₀.₅ LIBERO 微调启动脚本（5090 机 32GB + py3.12 + lerobot main v0.6.2 + torch cu130）
#
# 支持两种训练方式（用 TRAIN_MODE 切换）：
#   1. TRAIN_MODE=lora    （默认）LoRA 参数高效微调（~2.8M 可训参数，r=16 默认）
#   2. TRAIN_MODE=expert         仅训练 Action Expert（冻结 PaliGemma，训 300M Expert + proj）
#
# 与 22_train_pi0.sh / 23_train_pi0_full.sh 的关键区别：
#   - --policy.type=pi05（不是 pi0）
#   - --policy.normalization_mapping：LIBERO 数据集无 q01/q99，强制 MEAN_STD 绕开 quantile 缺失
#   - --policy.n_action_steps=10：pi05 默认 action chunk 长度
#   - --policy.empty_cameras=1：LIBERO 2 相机，pi05_base 期望 3 个，补 1 个空槽
#   - 不传 --rename_map：LIBERO 数据集键名 observation.images.image / image2 已与 pi05 匹配
#   - --policy.gradient_checkpointing=true：LIBERO 数据量大（273k 帧），建议开
#
# 用法：
#   bash scripts/24_train_pi05_libero.sh                            # 默认 LoRA r=16、30k 步、batch 4
#   TRAIN_MODE=expert bash scripts/24_train_pi05_libero.sh          # 切到 Action Expert 全参微调
#   STEPS=200 BATCH_SIZE=1 bash scripts/24_train_pi05_libero.sh     # 冒烟测试：200 步验证链路
#   LORA_R=32 STEPS=60000 bash scripts/24_train_pi05_libero.sh      # LoRA r=32、60k 步
#   BASE_MODEL=pi05_base bash scripts/24_train_pi05_libero.sh       # 改用通用基座（非 LIBERO 预训练版）
#   WANDB_ENABLE=true bash scripts/24_train_pi05_libero.sh          # 开启 wandb 离线可视化
# =============================================================================
set -euo pipefail

# ---------- 路径（最先定义，供 LIBERO assets 自检使用）----------
PI05_HOME="$HOME/ws/11_pi_serial_models"

# ---------- 离线模式：paligemma tokenizer 已注入 HF 缓存（与 pi0 同款 trick）
# pi05 用同一个 google/paligemma-3b-pt-224 gated tokenizer，国内必须离线
# 注意：虽然训练流程不直接依赖仿真 env，但 LIBERO assets 缺省时，
# lerobot 数据加载流程可能隐式触发 libero/__init__.py 资产查找，
# 为保险起见训练脚本也做同款自检与软链 ----------
export HF_HUB_OFFLINE=1

# ---------- LIBERO assets 自检（与 33_eval_pi05_libero.sh 同款逻辑）----------
LIBERO_VENV_PATH="$PI05_HOME/.venv312/lib/python3.12/site-packages/libero/libero"
LIBERO_USER_ASSETS="$HOME/.libero/assets"
LIBERO_ASSETS_KEY_FILE="scenes/libero_tabletop_base_style.xml"
ensure_libero_assets() {
  if [[ -f "$LIBERO_VENV_PATH/assets/$LIBERO_ASSETS_KEY_FILE" ]]; then return 0; fi
  if [[ -f "$LIBERO_USER_ASSETS/$LIBERO_ASSETS_KEY_FILE" ]]; then
    ln -sfn "$LIBERO_USER_ASSETS" "$LIBERO_VENV_PATH/assets"
    echo "✓ 建立 venv libero assets 软链 → $LIBERO_USER_ASSETS"
    return 0
  fi
  echo "✗ 缺少 LIBERO 仿真 assets，一键修复："
  echo "    source $PI05_HOME/.venv312/bin/activate"
  echo "    mkdir -p ~/.libero"
  echo "    modelscope download --dataset lerobot/libero-assets --local_dir ~/.libero/assets"
  exit 1
}
ensure_libero_assets

# ---------- 路径与可调参数（PI05_HOME 已在首部定义）----------
DATASET_DIR="$PI05_HOME/data/libero"                          # LIBERO 数据集（v3.0，1693 episodes）
BASE_MODEL="${BASE_MODEL:-pi05_libero_base}"                  # 默认 LIBERO 微调基座（benchmark 路线），可选 pi05_base
MODEL_DIR="$PI05_HOME/models/$BASE_MODEL"                     # π₀.₅ 基座权重（~14GB F32）
TRAIN_MODE="${TRAIN_MODE:-lora}"                              # lora（默认）| expert
STEPS="${STEPS:-30000}"            # 30k 步 × batch 4 ÷ 273465 帧 ≈ 0.44 epoch（LIBERO 数据大）
BATCH_SIZE="${BATCH_SIZE:-}"       # 不设默认值，由 TRAIN_MODE 分支决定（lora=4 / expert=2）
LORA_R="${LORA_R:-16}"             # LoRA 秩，默认 16，可调 32/64
WANDB_ENABLE="${WANDB_ENABLE:-false}"

# ---------- 根据 TRAIN_MODE 推导 OUTPUT_DIR、JOB_NAME、BATCH_SIZE（避免覆盖） ----------
case "$TRAIN_MODE" in
  lora)
    OUTPUT_DIR="$PI05_HOME/outputs/pi05_libero_lora"
    JOB_NAME="pi05_libero_lora"
    BATCH_SIZE="${BATCH_SIZE:-4}"   # LoRA 只训 2.8M，5090(32GB) 可用 batch 4
    ;;
  expert)
    OUTPUT_DIR="$PI05_HOME/outputs/pi05_libero_expert"
    JOB_NAME="pi05_libero_expert"
    # expert 模式训 300M 全参 + AdamW，显存吃紧，默认 batch 2
    BATCH_SIZE="${BATCH_SIZE:-2}"
    ;;
  *)
    echo "✗ 未知 TRAIN_MODE=$TRAIN_MODE，支持: lora | expert"; exit 1 ;;
esac

# ---------- wandb 实验跟踪 ----------
WANDB_ARGS="--wandb.enable=false"
if [[ "$WANDB_ENABLE" == "true" ]]; then
  export WANDB_MODE="${WANDB_MODE:-offline}"   # 默认离线，国内不联网
  export WANDB_DIR="${WANDB_DIR:-$PI05_HOME/wandb}"
  mkdir -p "$WANDB_DIR"
  echo "✓ wandb 已开启（mode=${WANDB_MODE}, dir=${WANDB_DIR}）"
  WANDB_ARGS="--wandb.enable=true --wandb.project=pi05_libero --wandb.mode=offline"
fi

# ---------- 训练前自检 ----------
echo "===== 训练前自检（π₀.₅ LIBERO） ====="
[[ -f "$MODEL_DIR/model.safetensors" ]] || { echo "✗ 缺少基座权重 $MODEL_DIR/model.safetensors"; exit 1; }
[[ -f "$DATASET_DIR/meta/info.json"   ]] || { echo "✗ 缺少数据集元信息 $DATASET_DIR/meta/info.json"; exit 1; }
command -v lerobot-train >/dev/null     || { echo "✗ 未找到 lerobot-train，请先 source $PI05_HOME/.venv312/bin/activate"; exit 1; }
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

# expert 模式显存警告（与 23_train_pi0_full.sh 同款）
GPU_MEM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
if [[ "$TRAIN_MODE" == "expert" && $GPU_MEM_MB -lt 30000 ]]; then
  echo "⚠ expert 模式（Action Expert 全参 ~300M）+ GPU 显存 ${GPU_MEM_MB}MB < 30GB，可能 OOM。"
  echo "  建议：BATCH_SIZE=1 或改用 TRAIN_MODE=lora（默认，仅训 ~2.8M 参数）"
  read -r -p "仍要继续吗？[y/N] " ans; [[ "$ans" == "y" || "$ans" == "Y" ]] || exit 1
fi

echo "✓ 自检通过"
echo "  训练模式: $TRAIN_MODE"
echo "  基座: $BASE_MODEL ($MODEL_DIR)"
echo "  数据集: LIBERO ($DATASET_DIR)"
echo "  Steps: $STEPS, Batch: $BATCH_SIZE"
if [[ "$TRAIN_MODE" == "lora" ]]; then
  echo "  LoRA r: $LORA_R"
fi
echo "  WandB: $WANDB_ENABLE"
echo

# ---------- 构建 PEFT / Expert 参数 ----------
POLICY_EXTRA_ARGS=""
case "$TRAIN_MODE" in
  lora)
    # LoRA 模式：传 --peft.method_type=LORA --peft.r，不开 gradient_checkpointing（显存充裕）
    POLICY_EXTRA_ARGS="--peft.method_type=LORA --peft.r=$LORA_R"
    ;;
  expert)
    # Expert 模式：不传 --peft.*（不做 PEFT），开 train_expert_only + gradient_checkpointing
    # 等效于"只训 Gemma Expert 300M 全参"，比 LoRA r=16 多 ~100 倍可训参数
    POLICY_EXTRA_ARGS="--policy.use_peft=false --policy.train_expert_only=true --policy.gradient_checkpointing=true"
    ;;
esac

# ---------- 启动训练 ----------
LOG_FILE="$PI05_HOME/logs/train_pi05_libero_${TRAIN_MODE}_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$PI05_HOME/logs"

echo "===== 启动 π₀.₅ LIBERO 训练（日志: $LOG_FILE） ====="
lerobot-train \
  --dataset.repo_id=lerobot/libero \
  --dataset.root="$DATASET_DIR" \
  --output_dir="$OUTPUT_DIR" \
  --job_name="$JOB_NAME" \
  --policy.type=pi05 \
  --policy.pretrained_path="$MODEL_DIR" \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}' \
  --policy.n_action_steps=10 \
  --policy.empty_cameras=1 \
  $POLICY_EXTRA_ARGS \
  $WANDB_ARGS \
  --steps="$STEPS" \
  --batch_size="$BATCH_SIZE" \
  2>&1 | tee "$LOG_FILE"

# 注：
# - --policy.type=pi05：选择 π₀.₅ 模型类（PI05Policy），不是 pi0
# - --policy.pretrained_path：加载 models/pi05_libero_base 权重（benchmark 路线）
#   可用 BASE_MODEL=pi05_base 切到通用基座做链路验证（不必下 pi05_libero_base）
# - --policy.normalization_mapping：LIBERO 数据集实测无 q01/q99，强制用 MEAN_STD 绕开
#   quantile 缺失（默认 QUANTILES 会在 dataloader 初始化时报 KeyError）
# - --policy.n_action_steps=10：pi05 默认 action chunk 长度，与 pi05.mdx 官方示例对齐
# - --policy.empty_cameras=1：LIBERO 只有 2 相机（image + image2），pi05_base 期望 3 个
#   相机槽位（base_0_rgb + left_wrist_0_rgb + right_wrist_0_rgb），补 1 个空槽填零
# - 不传 --rename_map：LIBERO 数据集自带键名 observation.images.image / image2 已与
#   pi05 基座期望匹配（pi05.mdx 原文：No --rename_map is needed here — the keys already match）
# - LoRA 模式不开 gradient_checkpointing：5090 32GB LoRA 显存充裕，开了反而训练变慢
# - expert 模式必开 gradient_checkpointing：300M 全参 + AdamW 优化器显存吃紧
# - 评估：POLICY_DIR=$OUTPUT_DIR/checkpoints/last/pretrained_model bash scripts/32_eval_pi0.sh
#   （eval 脚本 pi05 复用同款，只需改 POLICY_DIR 指向 pi05 训练输出）
