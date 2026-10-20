# 104-unitreerobotics — UnifoLM-WMA-0 本地推理

> 定位：本文档仅覆盖子项目 A（UnifoLM-WMA-0 世界模型推理）。RL 训练子项目（unitree_rl_mjlab）见 [docs/02-unitree_rl_mjlab环境搭建与验证.md](docs/02-unitree_rl_mjlab环境搭建与验证.md)，总入口见 [START_HERE.md](START_HERE.md)。
>
> 相关文档：
> - [WMA 推理脚本](scripts/run_wma_interaction.sh) — 世界模型交互推理（支持切换数据集）
> - [学习笔记](docs/01-UnifoLM-WMA学习笔记.md) — 原理架构、输入输出、核心概念、WMA/VLA/OmniA 关系
> - RTX 5090 (sm_120) 兼容性修复记录已并入本文第 5 节

## 1. 项目结构

```
104-unitreerobotics/
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

cd ~/ws/104-unitreerobotics
python3.10 -m venv --clear .venv
source .venv/bin/activate
python -m pip install -U pip
```

### 3.2 安装 RTX 5090 适配版 PyTorch ✅

复用工作区顶层部署脚本（PyTorch 2.11.0 + cu128）：

```bash
cd ~/ws/104-unitreerobotics
source .venv/bin/activate
bash ~/ws/scripts/setup_py310_cu128_rtx5090_torch2110.sh
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
source ~/ws/104-unitreerobotics/.venv/bin/activate

cd ~/ws/104-unitreerobotics/repos/

# 官方上游仓库：https://github.com/unitreerobotics/unifolm-world-model-action
# 本地使用 fork（已包含 RTX 5090 兼容性修复）：
#   仓库: git@github.com:xiaoxiaomengmeng/unifolm-world-model-action.git
#   分支: main
#   基础 commit: c7e4d9d (update Readme.md - add G1_Dex1_DiverseManip datasets)
#   修复 commit: 9e82896 (feat: adapt UnifoLM-WMA inference for RTX 5090)
git clone --recurse-submodules git@github.com:xiaoxiaomengmeng/unifolm-world-model-action.git

cd ~/ws/104-unitreerobotics/repos/unifolm-world-model-action

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

# ④ 手动安装运行时依赖（跳过 torch/torchvision/xformers，分两步走）
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
bash ~/ws/104-unitreerobotics/scripts/run_wma_interaction.sh

# 指定其他数据集
bash ~/ws/104-unitreerobotics/scripts/run_wma_interaction.sh unitree_g1_pack_camera

# 或用独立脚本
bash ~/ws/104-unitreerobotics/scripts/run_g1_pack_camera.sh
bash ~/ws/104-unitreerobotics/scripts/run_z1_dual_arm_stackbox.sh
bash ~/ws/104-unitreerobotics/scripts/run_z1_dual_arm_stackbox_v2.sh
bash ~/ws/104-unitreerobotics/scripts/run_z1_dual_arm_cleanup_pencils.sh
```

### 手动执行

```bash
source ~/ws/104-unitreerobotics/.venv/bin/activate
cd ~/ws/104-unitreerobotics/repos/unifolm-world-model-action
mkdir -p ~/ws/104-unitreerobotics/outputs

export HF_ENDPOINT=https://hf-mirror.com

CUDA_VISIBLE_DEVICES=0 python3 scripts/evaluation/world_model_interaction.py \
    --seed 123 \
    --ckpt_path ~/.cache/modelscope/models/unitreerobotics--UnifoLM-WMA-0-Dual/snapshots/master/unifolm_wma_dual.ckpt \
    --config configs/inference/world_model_interaction.yaml \
    --savedir ~/ws/104-unitreerobotics/outputs/interaction \
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

> 环境：RTX 5090 (compute capability 12.0) + PyTorch 2.11.0+cu128 + Python 3.10.12
>
> 官方代码基于 torch 2.3.1 + xformers 0.0.27 + A100/H100 (sm_80/sm_90) 开发，
> 在 RTX 5090 上需要以下适配修改。修复已包含在本地 fork 中。

### 修复 1：Python 版本约束放宽

**文件**：`pyproject.toml`

**问题**：`requires-python = "==3.10.18"`，而系统 Python 为 3.10.12，pip 拒绝安装。

**修复**：

```toml
# 原
requires-python = "==3.10.18"
# 改为
requires-python = ">=3.10"
```

### 修复 2：xformers fa2 不支持 float32 (sm_120)

**文件**：`src/unifolm_wma/modules/attention.py` — `efficient_forward` 方法

**问题**：xformers 的 flash attention 2 后端在 sm_120 上仅支持 fp16/bf16，
当输入为 float32 时抛出：

```
NotImplementedError: No operator found
`memory_efficient_attention` is not supported on the current GPU (float32, sm_120)
```

**修复**：在 `efficient_forward` 入口处检测 dtype，自动 cast 到 bfloat16：

```python
# RTX 5090 (sm_120): xformers fa2 only supports fp16/bf16
_attn_dtype = q.dtype
_use_cast = _attn_dtype not in (torch.float16, torch.bfloat16)
if _use_cast:
    q = q.to(torch.bfloat16)

# 每个 xformers 调用前 cast k/v，调用后 cast 回原 dtype
if _use_cast:
    k, v = k.to(torch.bfloat16), v.to(torch.bfloat16)
out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=None)
out = out.to(_attn_dtype)
```

### 修复 3：xformers fa2 不支持 tensor attn_bias (sm_120)

**文件**：`src/unifolm_wma/modules/attention.py` — `efficient_forward` 方法中 `out_aa` 部分

**问题**：agent-action cross attention 使用 tensor 类型的 `attn_bias`（mask），
xformers fa2 在 sm_120 上不支持：

```
NotImplementedError: No operator found
`memory_efficient_attention` is not supported (attn_bias type: Tensor, sm_120)
```

**修复**：将该处 xformers 调用替换为 PyTorch 原生 `scaled_dot_product_attention`：

```python
# Use PyTorch native SDPA for masked attention
_q_4d = q.reshape(b, self.heads, q.shape[1], self.dim_head)
_k_4d = k_aa.reshape(b, self.heads, k_aa.shape[1], self.dim_head).to(q.dtype)
_v_4d = v_aa.reshape(b, self.heads, v_aa.shape[1], self.dim_head).to(q.dtype)
_m_4d = attn_mask_aa.reshape(b, self.heads, attn_mask_aa.shape[1], attn_mask_aa.shape[2])
out_aa = torch.nn.functional.scaled_dot_product_attention(
    _q_4d, _k_4d, _v_4d, attn_mask=_m_4d)
out_aa = out_aa.permute(0, 2, 1, 3).reshape(b, q.shape[1], self.heads * self.dim_head)
out_aa = out_aa.to(_attn_dtype)
```

> **注意**：`k_aa`/`v_aa` 必须 `.to(q.dtype)` 确保与已 cast 为 bfloat16 的 `q` dtype 一致。

### 修复 4：torchvision.io.write_video 已移除

**文件**：`scripts/evaluation/world_model_interaction.py` — `save_results` 函数

**问题**：新版 torchvision（≥0.20）移除了 `torchvision.io.write_video`：

```
AttributeError: module 'torchvision.io' has no attribute 'write_video'
```

**修复**：改用已导入的 `imageio`：

```python
# 原
torchvision.io.write_video(filename, grid, fps=fps, video_codec='h264', options={'crf': '10'})
# 改为
imageio.mimwrite(filename, grid.numpy(), fps=fps, quality=9)
```

### 修复 5：清除 Python 缓存

修改 `.py` 文件后，如果存在旧的 `__pycache__`，Python 可能仍加载旧字节码：

```bash
find ~/ws/104-unitreerobotics/repos/unifolm-world-model-action/src \
    -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
```

### 依赖安装注意事项

| 问题 | 原因 | 解决 |
|------|------|------|
| `pip install -e .` 覆盖 torch | pyproject.toml pin 了 torch==2.3.1 | 使用 `--no-deps` 安装主包 |
| tensorflow 依赖回溯 | tf-graphics 依赖 tf>=2.2 与 numpy 1.x 冲突 | pin `tensorflow==2.13.1`，tf-graphics 用 `--no-deps` |
| `typing-extensions` 被降级 | tensorflow 安装时拉低版本 | 事后 `pip install "typing-extensions>=4.16.0"` |
| 缺少 pyarrow / datasets | 数据加载需要 | 单独 `pip install pyarrow datasets==3.6.0` |

## 6. 备注

- 官方 README 推荐 conda 环境，本工作区统一使用 venv + 顶层部署脚本管理 PyTorch 版本。
- CLIP ViT-H-14 权重通过 `HF_ENDPOINT=https://hf-mirror.com` 自动下载缓存。
- tensorflow 仅用于数据加载（rlu_tfds），推理时不使用 GPU。
