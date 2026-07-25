# 20_unitreerobotics — UnifoLM-WMA-0 本地推理

> 相关文档：
> - [WMA 推理脚本](scripts/run_wma_interaction.sh) — 世界模型交互推理（支持切换数据集）
> - [学习笔记](STUDY_NOTES.md) — 参数解读、输入输出、核心概念
> - [兼容性修复](fix_readme.md) — RTX 5090 (sm_120) 适配记录

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

### 3.1 安装 GPU 驱动 + 创建 Python 3.10 虚拟环境 ✅

```bash
sudo apt update
sudo apt install nvidia-driver-580-open nvidia-utils-580
# 重启后验证：nvidia-smi 应显示 RTX 5090

cd /home/creator/ws/20_unitreerobotics
python3.10 -m venv --clear .venv
source .venv/bin/activate
python -m pip install -U pip
```

### 3.2 安装 RTX 5090 适配版 PyTorch ✅

复用工作区顶层部署脚本（PyTorch 2.11.0 + cu128）：

```bash
cd /home/creator/ws/20_unitreerobotics
source .venv/bin/activate
bash /home/creator/ws/setup_py310_cu128_rtx5090_torch2110.sh
```

验证：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 期望输出: 2.11.0+cu128 True NVIDIA GeForce RTX 5090
```

### 3.3 安装项目依赖 ✅

> **注意**：pyproject.toml 中 pin 了 `torch==2.3.1`，直接使用 `pip install -e .` 会覆盖已安装的 cu128 版 torch。
> 必须使用 `--no-deps` 安装主包，再手动安装其余依赖。

```bash
source /home/creator/ws/20_unitreerobotics/.venv/bin/activate

cd /home/creator/ws/20_unitreerobotics/repos/

# 官方上游仓库：https://github.com/unitreerobotics/unifolm-world-model-action
# 本地使用 fork（已包含 RTX 5090 兼容性修复）：
#   仓库: git@github.com:xiaoxiaomengmeng/unifolm-world-model-action.git
#   分支: main
#   基础 commit: c7e4d9d (update Readme.md - add G1_Dex1_DiverseManip datasets)
#   修复 commit: 9e82896 (feat: adapt UnifoLM-WMA inference for RTX 5090)
git clone --recurse-submodules git@github.com:xiaoxiaomengmeng/unifolm-world-model-action.git

cd /home/creator/ws/20_unitreerobotics/repos/unifolm-world-model-action

# ① 初始化子模块（external/dlimp）
#    如果 GitHub 访问慢，可先配置代理或手动 clone
git submodule update --init --recursive

# ② 以 --no-deps 方式安装主包，避免覆盖 torch
# pyproject.toml 中写死了 requires-python = "==3.10.18"，而你的 Python 是 3.10.12。直接放宽这个限制即可：requires-python 从 ==3.10.18 放宽为 >=3.10。
python -m pip install -e . --no-deps

# ③ 安装 external/dlimp
cd external/dlimp
python -m pip install -e . --no-deps
cd ../..

# ④ （不太好安装）手动安装运行时依赖（跳过 torch/torchvision/xformers）
python -m pip install \
    decord==0.6.0 einops==0.8.0 imageio==2.35.1 "numpy==1.24.2" \
    omegaconf==2.3.0 opencv-python==4.10.0.84 pandas==2.0.0 \
    pillow==9.5.0 pytorch-lightning==1.9.3 pyyaml==6.0 tqdm==4.66.5 \
    transformers==4.40.1 moviepy==1.0.3 av==12.3.0 gradio==4.39.0 \
    timm==0.9.10 scikit-learn==1.5.1 open-clip-torch==2.22.0 \
    kornia==0.7.3 diffusers==0.30.2 termcolor==2.4.0 draccus==0.11.5 \
    accelerate==1.7.0 tensorflow-metadata==1.16.1 "protobuf==3.20.3" \
    datasets==3.6.0 tensorflow-graphics==2021.12.3 fairscale==0.4.13


# 第一步：核心依赖（无 tensorflow，快速完成）
python -m pip install \
    decord==0.6.0 einops==0.8.0 imageio==2.35.1 "numpy==1.24.2" \
    omegaconf==2.3.0 opencv-python==4.10.0.84 pandas==2.0.0 \
    pillow==9.5.0 pytorch-lightning==1.9.3 pyyaml==6.0 tqdm==4.66.5 \
    transformers==4.40.1 moviepy==1.0.3 av==12.3.0 gradio==4.39.0 \
    timm==0.9.10 scikit-learn==1.5.1 open-clip-torch==2.22.0 \
    kornia==0.7.3 diffusers==0.30.2 termcolor==2.4.0 draccus==0.11.5 \
    accelerate==1.7.0 fairscale==0.4.13

# 第二步：tensorflow 相关（pin 2.13.1 兼容 numpy 1.x + protobuf 放宽）
python -m pip install "tensorflow==2.13.1" "protobuf>=3.20.3,<5" tensorflow-metadata==1.16.1
python -m pip install tensorflow-graphics==2021.12.3 --no-deps

python -m pip install "typing-extensions>=4.16.0"

python -m pip install pyarrow
python -m pip install datasets==3.6.0

python -m pip install xformers

# ⑤ 验证核心 import
python -c "import unifolm_wma; import einops; import omegaconf; import transformers; print('All imports OK')"
```

### 3.4 安装系统级依赖（pinocchio / ffmpeg）

pinocchio 用于机器人运动学，ffmpeg 用于视频编解码：

```bash
# 方式一：conda（推荐，官方要求 pinocchio=3.2.0）
conda install pinocchio=3.2.0 -c conda-forge -y
conda install ffmpeg=7.1.1 -c conda-forge -y

# 方式二：若推理时不依赖 pinocchio（交互模式可能不需要），可先跳过
# ffmpeg 可通过系统包安装：
sudo apt install ffmpeg
```

## 4. 运行推理（世界模型交互模式，无需真机） ✅

使用仓库自带的 `examples/world_model_interaction_prompts` 示例数据即可测试，无需 G1/Z1 机器人本体。
模型会根据输入图像 + 文本指令，生成未来交互视频预测 + 动作序列。

### 快捷方式（推荐）

```bash
# 默认跑 unitree_z1_stackbox
bash /home/creator/ws/20_unitreerobotics/scripts/run_wma_interaction.sh

# 指定其他数据集
bash /home/creator/ws/20_unitreerobotics/scripts/run_wma_interaction.sh unitree_g1_pack_camera

# 或用独立脚本
bash /home/creator/ws/20_unitreerobotics/scripts/run_g1_pack_camera.sh
bash /home/creator/ws/20_unitreerobotics/scripts/run_z1_dual_arm_stackbox.sh
bash /home/creator/ws/20_unitreerobotics/scripts/run_z1_dual_arm_stackbox_v2.sh
bash /home/creator/ws/20_unitreerobotics/scripts/run_z1_dual_arm_cleanup_pencils.sh
```

### 手动执行

```bash
source /home/creator/ws/20_unitreerobotics/.venv/bin/activate
cd /home/creator/ws/20_unitreerobotics/repos/unifolm-world-model-action
mkdir -p /home/creator/ws/20_unitreerobotics/outputs

export HF_ENDPOINT=https://hf-mirror.com

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

输出结构（`outputs/interaction/inference/`）：

```
inference/
├── 0_full_fs2.mp4              # 完整 rollout 视频
└── sample_0/
    ├── wm/2/itr-{0..11}.mp4    # 每步 world model 生成
    └── dm/2/itr-{0..11}.mp4    # 每步 dynamics model 生成
```

推理耗时参考：RTX 5090，12 步迭代 × 50 DDIM steps，总计约 6 分 40 秒。

如需测试其他数据集，替换 `--dataset` 和 `--n_iter`/`--frame_stride` 参数：

| dataset | n_iter | frame_stride |
|---------|--------|-------------|
| unitree_z1_stackbox | 12 | 4 |
| unitree_z1_dual_arm_stackbox | 7 | 4 |
| unitree_z1_dual_arm_stackbox_v2 | 11 | 4 |
| unitree_z1_dual_arm_cleanup_pencils | 8 | 4 |
| unitree_g1_pack_camera | 11 | 6 |

## 5. RTX 5090 (sm_120) 兼容性修复记录

详见 [fix_readme.md](fix_readme.md)，包含 5 项修复：

1. Python 版本约束放宽
2. xformers fa2 不支持 float32 → 自动 cast bfloat16
3. xformers fa2 不支持 tensor attn_bias → 改用 PyTorch SDPA
4. torchvision.io.write_video 移除 → 改用 imageio
5. 清除 __pycache__ 确保新代码生效

## 6. 备注

- 官方 README 推荐 conda 环境，本工作区统一使用 venv + 顶层部署脚本管理 PyTorch 版本。
- CLIP ViT-H-14 权重通过 `HF_ENDPOINT=https://hf-mirror.com` 自动下载缓存。
- tensorflow 仅用于数据加载（rlu_tfds），推理时不使用 GPU。
