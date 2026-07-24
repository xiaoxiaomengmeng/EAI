# 20_unitreerobotics — UnifoLM-WMA-0 本地推理

## 1. 项目结构

```
20_unitreerobotics/
├── repos/unifolm-world-model-action/   # 官方源码（含 external/dlimp 子模块）
└── .venv/                              # 独立虚拟环境
```

## 2. 模型权重（已下载）

```bash
modelscope download --model unitreerobotics/UnifoLM-WMA-0-Dual
```

本地路径：

```
~/.cache/modelscope/models/unitreerobotics--UnifoLM-WMA-0-Dual/snapshots/master/unifolm_wma_dual.ckpt
```

## 3. 从零初始化环境

### 3.1 创建 Python 3.10 虚拟环境

```bash
sudo apt update
sudo apt install nvidia-driver-580-open

cd /home/creator/ws/20_unitreerobotics
python3.10 -m venv --clear .venv
source .venv/bin/activate
python -m pip install -U pip
```

### 3.2 安装 RTX 5090 适配版 PyTorch

复用工作区顶层部署脚本（PyTorch 2.11.0 + cu128）：

```bash
cd /home/creator/ws/20_unitreerobotics
source .venv/bin/activate
bash /home/creator/ws/setup_py310_cu128_rtx5090_torch2110.sh
```

### 3.3 安装项目依赖

> **注意**：pyproject.toml 中 pin 了 `torch==2.3.1`，直接使用 `pip install -e .` 会覆盖已安装的 cu128 版 torch。
> 必须使用 `--no-deps` 安装主包，再手动安装其余依赖。

```bash
source /home/creator/ws/20_unitreerobotics/.venv/bin/activate
cd /home/creator/ws/20_unitreerobotics/repos/unifolm-world-model-action

# 初始化子模块（external/dlimp）
git submodule update --init --recursive

# 以 --no-deps 方式安装主包，避免覆盖 torch
python -m pip install -e . --no-deps

# 安装 external/dlimp
cd external/dlimp
python -m pip install -e . --no-deps
cd ../..

# 手动安装运行时依赖（跳过 torch/torchvision/xformers）
python -m pip install \
    decord==0.6.0 einops==0.8.0 imageio==2.35.1 "numpy==1.24.2" \
    omegaconf==2.3.0 opencv-python==4.10.0.84 pandas==2.0.0 \
    pillow==9.5.0 pytorch-lightning==1.9.3 pyyaml==6.0 tqdm==4.66.5 \
    transformers==4.40.1 moviepy==1.0.3 av==12.3.0 gradio==4.39.0 \
    timm==0.9.10 scikit-learn==1.5.1 open-clip-torch==2.22.0 \
    kornia==0.7.3 diffusers==0.30.2 termcolor==2.4.0 draccus==0.11.5 \
    accelerate==1.7.0 tensorflow-metadata==1.16.1 "protobuf==3.20.3" \
    datasets==3.6.0 tensorflow-graphics==2021.12.3 fairscale==0.4.13
```

### 3.4 安装 conda 依赖（pinocchio / ffmpeg）

如果系统已有 conda，可在 base 或单独环境中安装：

```bash
conda install pinocchio=3.2.0 -c conda-forge -y
conda install ffmpeg=7.1.1 -c conda-forge -y
```

或在 venv 中通过 pip 安装 ffmpeg-python（若仅需视频编解码）：

```bash
source /home/creator/ws/20_unitreerobotics/.venv/bin/activate
python -m pip install ffmpeg-python
```

## 4. 运行推理（世界模型交互模式）

```bash
source /home/creator/ws/20_unitreerobotics/.venv/bin/activate
cd /home/creator/ws/20_unitreerobotics/repos/unifolm-world-model-action

CUDA_VISIBLE_DEVICES=0 python3 scripts/evaluation/world_model_interaction.py \
    --seed 123 \
    --ckpt_path ~/.cache/modelscope/models/unitreerobotics--UnifoLM-WMA-0-Dual/snapshots/master/unifolm_wma_dual.ckpt \
    --config configs/inference/world_model_interaction.yaml \
    --savedir /home/creator/ws/20_unitreerobotics/outputs/interaction \
    --bs 1 --height 320 --width 512 \
    --unconditional_guidance_scale 1.0 \
    --ddim_steps 50 --ddim_eta 1.0 \
    --prompt_dir examples/world_model_interaction_prompts \
    --dataset unitree_z1_stackbox \
    --video_length 16 --frame_stride 2 \
    --n_action_steps 16 --exe_steps 16 \
    --n_iter 12 \
    --timestep_spacing uniform_trailing \
    --guidance_rescale 0.7 \
    --perframe_ae
```

## 5. 备注

- xformers==0.0.27 仅兼容 torch 2.3.x；在 RTX 5090 + torch 2.11.0 环境下如遇到 xformers 导入错误，可尝试 `pip install xformers` 安装最新兼容版，或设置环境变量跳过 xformers。
- 官方 README 推荐 conda 环境，本工作区统一使用 venv + 顶层部署脚本管理 PyTorch 版本。
