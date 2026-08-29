#!/usr/bin/env bash
# =============================================================================
# ACT 训练启动脚本（5090 机 32GB + py3.12 + lerobot main v0.6.2）
# ACT（Action Chunking with Transformers）从零训练，无需基座权重（与 pi0 路线不同），
# 不需要 rename_map / pretrained_path（策略结构直接从数据集特征构建）。
# 全部 CLI 参数已按 v0.6.2 --help 实测校验（--policy.type=act + --steps/--batch_size 顶层参数）。
# 用法：
#   bash scripts/train_act.sh                          # 默认 100000 步、batch 8
#   STEPS=200 bash scripts/train_act.sh                # 冒烟测试：先跑 200 步验证流程
#   STEPS=80000 bash scripts/train_act.sh              # 官方发布模型对应的 checkpoint 步数（官方实际训练为 100k 步，见官方 train_config.json）
#   BATCH_SIZE=16 bash scripts/train_act.sh            # 32GB 显存富余可上探（80M 参数模型压力很小）
#   WANDB_ENABLE=true bash scripts/train_act.sh        # 开启 wandb 离线记录（国内查看不便，推荐用离线画曲线方案）
# 训练后画 loss 曲线（离线方案，不依赖 wandb）：
#   python scripts/plot_train_log.py                   # 自动解析最新 train_act 日志 → PNG
# =============================================================================
set -euo pipefail

# ---------- ResNet-18 ImageNet 预训练骨干预下载（首次训练需要，约 45MB） ----------
# ACT 视觉骨干默认加载 ImageNet 预训练权重，torchvision 会从 download.pytorch.org 下载
# 到 TORCH_HOME（默认 ~/.cache/torch/hub/checkpoints/）。提前下载并给出失败提示，
# 避免训练中途卡在下载上；缓存命中后不再联网（实测本机直连可达，26MB/s）。
RESNET_PTH="$HOME/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth"
if [[ ! -f "$RESNET_PTH" ]]; then
  mkdir -p "$(dirname "$RESNET_PTH")"
  echo "↓ 首次训练：预下载 ResNet-18 ImageNet 骨干 → $RESNET_PTH"
  curl -fL --connect-timeout 15 -o "$RESNET_PTH" \
    https://download.pytorch.org/models/resnet18-f37072fd.pth \
    || { echo "✗ 骨干下载失败（网络不通）。可手动下载该文件放入 $RESNET_PTH，"; \
         echo "  或改用不加载预训练权重：追加 --policy.pretrained_backbone_weights=null"; rm -f "$RESNET_PTH"; exit 1; }
fi

# ---------- 路径与可调参数（可用环境变量覆盖） ----------
ACT_HOME="$HOME/ws/11_pi_serial_models"
DATASET_DIR="$ACT_HOME/data/aloha_sim_transfer_cube_human"   # 数据集目录本身（含 meta/info.json），不是父目录
OUTPUT_DIR="$ACT_HOME/outputs/act_aloha_cube"                # checkpoint 与日志输出目录（非 resume 时若已存在会报 FileExistsError，需先删或改名）
STEPS="${STEPS:-100000}"          # 总训练步数：默认 100k，与官方 train_config.json 一致（官方发布的模型取 80k 步 checkpoint）；50 episodes × 400 帧，100k 步 × batch8 ≈ 40 epoch
BATCH_SIZE="${BATCH_SIZE:-8}"     # lerobot 默认 8；ACT 仅 ~52M 可学习参数，32GB 显存可上探 16~32
CHUNK_SIZE="${CHUNK_SIZE:-100}"   # 动作块长度（默认 100，与官方一致）
KL_WEIGHT="${KL_WEIGHT:-10.0}"    # KL 损失权重（默认 10.0，与官方一致）
LOG_FREQ="${LOG_FREQ:-100}"       # 日志指标行频率（lerobot 默认 200 太稀疏，画曲线建议 ≤100）

# ---------- wandb 实验跟踪（loss 曲线可视化） ----------
# 开启方式：WANDB_ENABLE=true bash scripts/train_act.sh
# 国内连不上 wandb.ai 时默认离线模式：生成本地报告，不联网。
WANDB_ENABLE="${WANDB_ENABLE:-false}"
if [[ "$WANDB_ENABLE" == "true" ]]; then
  export WANDB_MODE="${WANDB_MODE:-offline}"   # 默认离线模式，不需要联网
  export WANDB_DIR="${WANDB_DIR:-$ACT_HOME/wandb}"
  mkdir -p "$WANDB_DIR"
  echo "✓ wandb 已开启（mode=${WANDB_MODE}, dir=${WANDB_DIR}）"
  WANDB_ARGS="--wandb.enable=true --wandb.project=act_aloha --wandb.mode=offline"
else
  WANDB_ARGS="--wandb.enable=false"
fi

# ---------- 训练前自检 ----------
echo "===== 训练前自检 ====="
[[ -f "$DATASET_DIR/meta/info.json" ]] || { echo "✗ 缺少数据集元信息 $DATASET_DIR/meta/info.json（可先运行 scripts/download_act_assets.sh）"; exit 1; }
command -v lerobot-train >/dev/null     || { echo "✗ 未找到 lerobot-train，请先 source $ACT_HOME/.venv312/bin/activate"; exit 1; }
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv || { echo "✗ nvidia-smi 失败，检查驱动"; exit 1; }
if [[ -d "$OUTPUT_DIR" ]]; then
  echo "⚠ 输出目录已存在：$OUTPUT_DIR"
  echo "  续训请用 --resume（见文末注释）；全新训练请先删除或改 OUTPUT_DIR，否则报 FileExistsError"
  read -r -p "仍要继续吗（大概率报错）？[y/N] " ans; [[ "$ans" == "y" || "$ans" == "Y" ]] || exit 1
fi
echo "✓ 自检通过（STEPS=$STEPS, BATCH_SIZE=$BATCH_SIZE, CHUNK_SIZE=$CHUNK_SIZE）"
echo

# ---------- 启动训练 ----------
LOG_FILE="$ACT_HOME/logs/train_act_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$ACT_HOME/logs"

echo "===== 启动训练（日志: $LOG_FILE） ====="
lerobot-train \
  --dataset.repo_id=aloha_sim_transfer_cube_human \
  --dataset.root="$DATASET_DIR" \
  --output_dir="$OUTPUT_DIR" \
  --job_name=act_aloha_cube \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.chunk_size="$CHUNK_SIZE" \
  --policy.kl_weight="$KL_WEIGHT" \
  $WANDB_ARGS \
  --steps="$STEPS" \
  --batch_size="$BATCH_SIZE" \
  --log_freq="$LOG_FREQ" \
  2>&1 | tee "$LOG_FILE"

# 注：
# - loss 曲线：本脚本不默认开 wandb（离线 run 国内无法直接查看）；曲线走离线方案：
#   指标行已按 --log_freq 落盘到 $LOG_FILE，训练后运行 python scripts/plot_train_log.py 出 PNG
# - --policy.push_to_hub=false 必带：v0.6.2 默认推 Hub，缺失 repo_id 时 validate 阶段报
#   "'repo_id' argument missing"（与 train_pi0.sh 同一实测结论）
# - ACT 从零训练：不需要 --policy.pretrained_path 与 --rename_map（pi0 才需要基座对齐）
# - 首次训练会下载 ResNet-18 ImageNet 骨干（~45MB，脚本开头已预下载；实测 pytorch.org 可达）；
#   网络不通时可改 --policy.pretrained_backbone_weights=null 从随机初始化训练
# - 实测速度参考（RTX 5090）：batch 8 稳定后约 36 step/s，100k 步约 46 分钟，可学习参数 52M
# - checkpoint 每 20000 步 + 最后一步自动保存（默认 save_freq=20000, save_checkpoint=true），
#   最终模型位于 $OUTPUT_DIR/checkpoints/last/pretrained_model（评估用）
# - 中断后续训：lerobot-train --resume=true --config_path=$OUTPUT_DIR/checkpoints/<step>/train_config.json
# - 评估（训练完成后）：
#     export MUJOCO_GL=egl
#     lerobot-eval \
#       --policy.path="$OUTPUT_DIR/checkpoints/last/pretrained_model" \
#       --env.type=aloha --env.task=AlohaTransferCube-v0 \
#       --eval.n_episodes=25 --eval.batch_size=1 \
#       --output_dir="$ACT_HOME/outputs/eval_act_aloha_cube" --job_name=eval_act_aloha_cube
#   详见 docs/25-ACT模型架构数据集训练与评估指南-v1.0.md 第四节
