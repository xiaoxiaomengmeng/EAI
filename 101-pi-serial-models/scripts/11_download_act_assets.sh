#!/usr/bin/env bash
# =============================================================================
# ACT 资源下载脚本（5090 机 · ModelScope 国内直连，优先于 huggingface.co）
# 下载对象：
#   1. ALOHA 仿真数据集（LeRobot v3.0 格式，落到 data/<数据集名>/）
#   2. 可选：官方预训练 ACT 策略权重（lerobot/act_aloha_sim_transfer_cube_human）
# 依赖：modelscope CLI（实测 1.38.1，位于 ~/.local/bin/modelscope）；
#       若未安装：pip install modelscope
# 用法：
#   bash scripts/11_download_act_assets.sh
#       默认仅下载 aloha_sim_transfer_cube_human（若本地已存在则跳过）
#   ALL_DATASETS=true bash scripts/11_download_act_assets.sh
#       下载全部 5 个数据集（transfer_cube human/scripted + insertion human/scripted + pusht）
#   PRETRAINED=true bash scripts/11_download_act_assets.sh
#       额外下载官方预训练 ACT 模型到 models/act_aloha_sim_transfer_cube_human
#   ALL_DATASETS=true PRETRAINED=true bash scripts/11_download_act_assets.sh
# =============================================================================
set -euo pipefail

ACT_HOME="$HOME/ws/101-pi-serial-models"
DATA_DIR="$ACT_HOME/data"
MODEL_DIR="$HOME/ws/models"                  # 工作区统一权重目录
mkdir -p "$DATA_DIR" "$MODEL_DIR"

# ---------- modelscope CLI 自检 ----------
if ! command -v modelscope >/dev/null 2>&1; then
  if [[ -x "$HOME/.local/bin/modelscope" ]]; then
    export PATH="$HOME/.local/bin:$PATH"
  else
    echo "✗ 未找到 modelscope CLI，请先执行：pip install modelscope"
    exit 1
  fi
fi
echo "✓ modelscope CLI: $(command -v modelscope)（$(modelscope --version 2>/dev/null | tail -1)）"

# ---------- 数据集下载函数（已存在则跳过） ----------
download_dataset() {
  local name="$1"
  local target="$DATA_DIR/$name"
  if [[ -f "$target/meta/info.json" ]]; then
    echo "○ 数据集 $name 已存在（$target），跳过"
    return
  fi
  echo "↓ 下载数据集 lerobot/$name → $target"
  modelscope download --dataset "lerobot/$name" --local_dir "$target"
  echo "✓ $name 下载完成"
}

# ---------- 数据集清单（ModelScope 镜像，2026-08 实测可访问） ----------
# 主数据集：ACT 论文标准训练集（双臂搬方块，50 episodes / 20000 帧，本项目已存在）
download_dataset "aloha_sim_transfer_cube_human"

if [[ "${ALL_DATASETS:-false}" == "true" ]]; then
  download_dataset "aloha_sim_transfer_cube_scripted"   # 脚本化演示版（同任务）
  download_dataset "aloha_sim_insertion_human"           # 插孔装配任务（人类演示）
  download_dataset "aloha_sim_insertion_scripted"        # 插孔装配任务（脚本化演示）
  download_dataset "pusht"                               # PushT 轻量验证任务
fi

# ---------- 可选：官方预训练 ACT 策略（直接评估/对比用，非训练必需） ----------
if [[ "${PRETRAINED:-false}" == "true" ]]; then
  ACT_MODEL_TARGET="$MODEL_DIR/act_aloha_sim_transfer_cube_human"
  if [[ -f "$ACT_MODEL_TARGET/config.json" ]]; then
    echo "○ 预训练 ACT 模型已存在（$ACT_MODEL_TARGET），跳过"
  else
    echo "↓ 下载预训练模型 lerobot/act_aloha_sim_transfer_cube_human → $ACT_MODEL_TARGET"
    modelscope download --model "lerobot/act_aloha_sim_transfer_cube_human" --local_dir "$ACT_MODEL_TARGET"
    echo "✓ 预训练 ACT 模型下载完成（约 200MB，含 model.safetensors）"
  fi
fi

# ---------- 下载后用 LeRobotDataset 做一次加载验证 ----------
VENV_PY="$ACT_HOME/.venv312/bin/python"
if [[ -x "$VENV_PY" ]]; then
  echo "===== LeRobotDataset 加载验证 ====="
  "$VENV_PY" - <<'EOF'
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset(
    "aloha_sim_transfer_cube_human",
    root="/home/creator/ws/101-pi-serial-models/data/aloha_sim_transfer_cube_human",
)
print(f"✓ 数据集加载成功：{len(ds)} 帧，{ds.num_episodes} episodes，fps={ds.fps}")
print(f"  特征：{list(ds.features.keys())}")
EOF
else
  echo "⚠ 未找到 $VENV_PY，跳过加载验证（可后续手动运行 scripts/02_test_dataset.py）"
fi

echo "===== 全部完成 ====="
