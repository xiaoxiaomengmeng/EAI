#!/bin/bash
# UnifoLM-WMA 世界模型交互推理脚本
# 用法: bash run_interaction.sh [dataset_name]
# 示例: bash run_interaction.sh unitree_z1_stackbox

set -e

# ============ 路径配置 ============
VENV_DIR="/home/creator/ws/20_unitreerobotics/.venv"
REPO_DIR="/home/creator/ws/20_unitreerobotics/repos/unifolm-world-model-action"
CKPT_PATH="$HOME/.cache/modelscope/models/unitreerobotics--UnifoLM-WMA-0-Dual/snapshots/master/unifolm_wma_dual.ckpt"

# ============ 数据集参数 ============
DATASET="${1:-unitree_z1_stackbox}"
SAVE_DIR="/home/creator/ws/20_unitreerobotics/outputs/interaction/${DATASET}"

case "$DATASET" in
    unitree_z1_stackbox)
        N_ITER=12; FRAME_STRIDE=2 ;;
    unitree_z1_dual_arm_stackbox)
        N_ITER=7; FRAME_STRIDE=4 ;;
    unitree_z1_dual_arm_stackbox_v2)
        N_ITER=11; FRAME_STRIDE=4 ;;
    unitree_z1_dual_arm_cleanup_pencils)
        N_ITER=8; FRAME_STRIDE=4 ;;
    unitree_g1_pack_camera)
        N_ITER=11; FRAME_STRIDE=6 ;;
    *)
        echo "未知数据集: $DATASET"
        echo "可选: unitree_z1_stackbox | unitree_z1_dual_arm_stackbox | unitree_z1_dual_arm_stackbox_v2 | unitree_z1_dual_arm_cleanup_pencils | unitree_g1_pack_camera"
        exit 1 ;;
esac

# ============ 激活环境 ============
source "$VENV_DIR/bin/activate"
cd "$REPO_DIR"

# 清除旧缓存，确保加载最新代码
find src -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# HuggingFace 镜像（国内加速 CLIP 权重下载）
export HF_ENDPOINT=https://hf-mirror.com

# ============ 运行推理 ============
echo "=========================================="
echo " 数据集: $DATASET"
echo " 迭代步数: $N_ITER"
echo " 帧步长: $FRAME_STRIDE"
echo " 输出目录: $SAVE_DIR"
echo "=========================================="

CUDA_VISIBLE_DEVICES=0 python3 scripts/evaluation/world_model_interaction.py \
    --seed 123 \
    --ckpt_path "$CKPT_PATH" \
    --config configs/inference/world_model_interaction.yaml \
    --savedir "$SAVE_DIR" \
    --bs 1 --height 320 --width 512 \
    --unconditional_guidance_scale 1.0 \
    --ddim_steps 50 --ddim_eta 1.0 \
    --prompt_dir examples/world_model_interaction_prompts \
    --dataset "$DATASET" \
    --video_length 16 --frame_stride "$FRAME_STRIDE" \
    --n_action_steps 16 --exe_steps 16 \
    --n_iter "$N_ITER" \
    --timestep_spacing uniform_trailing \
    --guidance_rescale 0.7 \
    --perframe_ae

echo ""
echo "推理完成！结果保存在: $SAVE_DIR/inference/"
