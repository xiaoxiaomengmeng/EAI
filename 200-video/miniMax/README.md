# MiniMax H3 音视频生成模型本地部署指南（RTX 5090 32G）

> 本文档针对本机环境：**Ubuntu 22.04 + RTX 5090 32G (sm_120) + Python 3.10 + PyTorch 2.11.0 cu128 + CUDA Toolkit 12.8**。
> ⚠️ 状态（2026-10-12）：MiniMax-H3 权重已清理（modelscope 缓存 97G + ComfyUI 侧 6 条软链同步删除，`--quick-test-for-ci` 不依赖它）；需要时按 §4 重新下载 ModelScope INT8 量化版并重建软链。

---

## 1. 模型区分：MiniMax-M3 vs MiniMax-H3

| 项目 | MiniMax-M3 | MiniMax-H3 |
|------|-----------|-----------|
| 模型类型 | 多模态大语言模型（理解） | 全模态 Omni-Transformer 音视频生成 |
| 用途 | 文本/图像/视频对话、理解、推理 | 文字/参考图 → 带同步音频的短视频 |
| 参数规模 | ~428B（~23B 激活） | 33.1B |
| 本机已下载 | ✅ `~/.cache/modelscope/models/MiniMax--MiniMax-M3-MXFP8/`（~433GB，MXFP8 量化） | ❌ 尚未下载，需要额外下载 INT8 量化版 |
| RTX 5090 单卡 32G | ❌ 完全跑不动（需要 450GB+ VRAM） | ✅ **INT8 量化 + CPU 内存卸载可跑** |
| 输出格式 | 文本 / 图像 / 视频描述 | **768P MP4 视频 + 原生双声道立体声**（一体化生成） |
| 最长时长 | — | 4~15 秒原生；2K 需 H3-Regenerate-2K API（闭源） |

> **重点**：你已经下载的 M3 是 LLM（理解模型），测试视频生成还需要下载 **H3 的 INT8 量化权重**。二者分工不同，不冲突。

---

## 2. 硬件门槛（RTX 5090 32G）

### 2.1 量化方案对比

| 精度方案 | 显存占用 | 整机内存 | 单卡 5090 32G |
|---------|---------|---------|-------------|
| BF16 全精度 | 135GB | 384GB | ❌ 跑不动 |
| 标准 INT8 | 85GB | 123GB | ❌ 超出 |
| **Pruned INT8 ConvRot（推荐）** | **21~30GB** | **≥48GB** | ✅ **唯一可行方案** |

### 2.2 本机建议

- 初期只测试 **5~10 秒 768P 短视频**
- FL2VA 和 Ref2VA **不同时加载**
- 系统内存 ≥64GB（48GB 最低，易爆交换区）
- 模型文件存 **NVMe SSD**，机械盘加载过慢

---

## 3. 从零初始化环境

### 3.1 创建 Python 3.10 虚拟环境

虚拟环境放在 ComfyUI 仓库根（结构如下）：

```bash
cd /home/creator/ws/200-video/ComfyUI
python3.10 -m venv --clear .venv
source .venv/bin/activate
python -m pip install -U pip
```

### 3.2 安装 RTX 5090 适配版 PyTorch（cu128，sm_120）

**不要用教程里的 CUDA 12.4**。本机 RTX 5090 Blackwell sm_120 需要 cu128，直接复用顶层部署脚本：

```bash
cd /home/creator/ws/200-video/ComfyUI
source .venv/bin/activate
bash /home/creator/ws/scripts/setup_py310_cu128_rtx5090_torch2110.sh
```

验证：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
PY
```

期望：

```text
torch: 2.11.0+cu128
torch cuda: 12.8
gpu: NVIDIA GeForce RTX 5090
capability: (12, 0)
```

### 3.3 安装 ComfyUI 基础依赖

```bash
source /home/creator/ws/200-video/ComfyUI/.venv/bin/activate
cd /home/creator/ws/200-video/ComfyUI
pip install -r requirements.txt
```

> 说明：本仓库 ComfyUI **已内置** MiniMax H3 节点（见 [comfy_extras/nodes_minimax_h3.py](file:///home/creator/ws/200-video/ComfyUI/comfy_extras/nodes_minimax_h3.py)），无需额外安装 `comfyui-minimax-h3` custom_nodes 插件。

---

## 4. 下载量化权重（关键！）

### 4.1 下载地址

| 镜像 | 链接 | 说明 |
|-----|------|------|
| 🇨🇳 **ModelScope（国内推荐，快）** | https://www.modelscope.cn/models/MiniMax/MiniMax-H3 | 官方仓，含 INT8 / FP8 / NVFP4 多种量化 |
| 🌍 HuggingFace | https://huggingface.co/MiniMaxAI/MiniMax-H3 | 同上，海外镜像 |
| ComfyUI 打包版（ModelScope） | https://www.modelscope.cn/models/Comfy-Org/MiniMax-H3 | Comfy-Org 重新打包，文件名略有不同 |
| ComfyUI 打包版（HuggingFace） | https://huggingface.co/Comfy-Org/MiniMax-H3 | 同上 |

### 4.2 RTX 5090 32G 推荐下载清单（已下载 ✅）

已从 ModelScope `MiniMax/MiniMax-H3` 下载到 `~/.cache/modelscope/models/MiniMax--MiniMax-H3/snapshots/master/`：

| 文件名 | 实际大小 | 选用 | 用途 |
|-------|---------|------|------|
| `minimax_h3_fl2va_pruned_fp8_scaled.safetensors` | 20 GB | ✅ **选用** | FL2VA 扩散模型（pruned FP8，最轻） |
| `minimax_h3_fl2va_int8_convrot.safetensors` | 32 GB | 备选 | FL2VA 标准 INT8（更重，质量略好） |
| `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | 15 GB | ✅ **选用** | 文本编码器（NVFP4，最轻） |
| `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` | 26 GB | 备选 | 文本编码器 INT8（更重，质量略好） |
| `minimax_h3_video_vae_fp16.safetensors` | 4.9 GB | ✅ | 视频 VAE |
| `minimax_h3_audio_vae_fp32.safetensors` | 578 MB | ✅ | 音频 VAE |

> **选用理由**：pruned_fp8_scaled (20G) + nvfp4_awq (15G) = 35GB，比标准 INT8 组合 (32G+26G=58G) 轻 23GB，是 5090 32G 最优组合。

### 4.3 ModelScope SDK 下载方式（备查）

```bash
source /home/creator/ws/200-video/ComfyUI/.venv/bin/activate
pip install modelscope

# 一次性下载所有量化变体
python - <<'PY'
from modelscope import snapshot_download
path = snapshot_download(
    "MiniMax/MiniMax-H3",
    allow_file_pattern=[
        "*fl2va_pruned_fp8_scaled.safetensors",
        "*fl2va_int8_convrot.safetensors",
        "*nvfp4_awq.safetensors",
        "*int8_convrot.safetensors",
        "*video_vae_fp16.safetensors",
        "*audio_vae_fp32.safetensors",
    ],
    cache_dir="/home/creator/.cache/modelscope/hub"
)
print("下载完成:", path)
PY
```

### 4.4 软链接到 ComfyUI models 目录（已完成 ✅）

下载缓存路径：`~/.cache/modelscope/models/MiniMax--MiniMax-H3/snapshots/master/`
（注意不是 `Comfy-Org/MiniMax-H3`，而是 `MiniMax--MiniMax-H3`）

```bash
CACHE_DIR=~/.cache/modelscope/models/MiniMax--MiniMax-H3/snapshots/master
COMFYUI_DIR=/home/creator/ws/200-video/ComfyUI

mkdir -p "$COMFYUI_DIR/models/diffusion_models"
mkdir -p "$COMFYUI_DIR/models/text_encoders"
mkdir -p "$COMFYUI_DIR/models/vae"

# FL2VA: 选用 pruned_fp8_scaled（20G）
ln -sf "$CACHE_DIR/diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors" \
  "$COMFYUI_DIR/models/diffusion_models/"

# 文本编码器: 选用 nvfp4_awq（15G）
ln -sf "$CACHE_DIR/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors" \
  "$COMFYUI_DIR/models/text_encoders/"

# VAE
ln -sf "$CACHE_DIR/vae/minimax_h3_video_vae_fp16.safetensors" \
  "$COMFYUI_DIR/models/vae/"
ln -sf "$CACHE_DIR/vae/minimax_h3_audio_vae_fp32.safetensors" \
  "$COMFYUI_DIR/models/vae/"

# 验证
ls -lh "$COMFYUI_DIR/models/diffusion_models/"
ls -lh "$COMFYUI_DIR/models/text_encoders/"
ls -lh "$COMFYUI_DIR/models/vae/"
```

---

## 5. 启动 ComfyUI

### 5.1 环境变量（显存优化必设）

```bash
source /home/creator/ws/200-video/ComfyUI/.venv/bin/activate
export HF_XET_HIGH_PERFORMANCE=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### 5.2 启动服务

```bash
cd /home/creator/ws/200-video/ComfyUI
python main.py --listen 0.0.0.0 --port 8188
```

浏览器打开：`http://<本机IP>:8188`

---

## 6. 运行工作流

### 6.1 官方工作流模板 JSON

加载模板：ComfyUI 界面 → 菜单 **Load** → 选择 JSON 文件。

| 生成模式 | 工作流 JSON | 说明 |
|---------|------------|------|
| T2V 文生视频 | [video_minimax_h3_t2v.json](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_t2v.json) | 纯文字生成视频+音频 |
| I2V 首尾帧 | [video_minimax_h3_i2v.json](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_i2v.json) | 输入第一张/最后一张图插值 |
| R2V 参考驱动 | [video_minimax_h3_r2v.json](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_r2v.json) | 参考图/视频/音频 → 生成同风格视频 |

也可以在 ComfyUI 界面内：**Template Library → Video → MiniMax H3 T2V / I2V / R2V**。

### 6.2 关键节点参数（防止 OOM）

- **ModelLoader（加载扩散模型）**：选择 `minimax_h3_fl2va_pruned_fp8_scaled.safetensors`，开启 `model_cpu_offload`（CPU 内存卸载，慢一点但能装下）
- **CLIPLoader（文本编码器）**：选择 `minimax` 类型，加载 `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`
- **VAE Loader**：视频 VAE 选择 `vae-use-tiling` 分块解码，大幅降低显存
- **MiniMaxH3SigmaShift**：
  - `shift_video = 12.0`（默认）
  - `shift_audio = 3.0`（默认）
- **EmptyMiniMaxH3LatentAV**：初期测试时 `length=124`（≈5 秒），跑通后再拉长

### 6.3 Prompt 建议

描述完整场景 → 分镜/镜头运动 → 音频音效：

```
A cinematic shot of a red panda eating bamboo in a lush bamboo forest at sunrise,
soft golden light filtering through leaves, camera slowly pans left following the panda.
Accompanied by gentle forest ambience: distant bird chirps, bamboo leaves rustling in
light wind, and soft crunching sounds.
```

---

## 7. RTX 5090 32G 显存优化清单（防止 OOM）

| # | 优化项 | 说明 |
|---|-------|------|
| 1 | 开启 **model_cpu_offload** | 模型权重卸载到系统内存，推理时按需切回，速度慢 ~20% 但不 OOM |
| 2 | **VAE 启用 tiling** | 大幅降低 VAE 解码阶段的峰值显存 |
| 3 | 初期 length=124（5 秒） | 先跑通，再逐步加到 244（10 秒） / 362（15 秒） |
| 4 | 不同时加载 FL2VA 和 Ref2VA | 跑 T2V 只加载 FL2VA；跑 R2V 只加载 Ref2VA |
| 5 | 关闭浏览器/IDE 等大型软件 | 系统内存紧张时会被 Linux OOM Killer 杀死 |
| 6 | 文本编码器用 **NVFP4 AWQ** 版 | 14.6GB，比 INT8 ConvRot 版轻 40%，5090 首选 |
| 7 | 不尝试原生 2K 输出 | 2K 需要 H3-Regenerate-2K 闭源 API，本地只到 768P |

---

## 8. 常见踩坑

| 问题 | 原因 | 解决 |
|------|------|------|
| `flash-attn` 编译失败 | CUDA toolkit 与 PyTorch 版本不匹配 | 本机已用 cu128，一般不需要 flash-attn（用 sdpa），如要编译需 `TORCH_CUDA_ARCH_LIST="12.0"` |
| 加载权重直接 OOM | 下了 BF16 原版，不是 INT8 版 | 确认文件名含 `pruned_int8_convrot`；或开启 CPU offload |
| 生成视频没音频 | ffmpeg 版本老或缺少音频编码器 | `sudo apt install -y ffmpeg` 并验证 `ffmpeg -encoders | grep aac` |
| 文本编码器加载就 OOM | 选了 bf16/int8_convrot 版 48GB/25GB | 改用 `nvfp4_awq` 版（14.6GB） |
| `ImportError: ... not found` | 没激活 venv 或依赖没装全 | `source .venv/bin/activate` + `pip install -r requirements.txt` |
| `minimax_token_tags` 报错 | 文本编码器文件名不对 / CLIP 类型选错了 | CLIPLoader 的 `type` 要选 `minimax`（MiniMax-H3），不是通用 clip |

---

## 9. 进阶与加速

- **MiniMax-H3-Turbo-LoRA**：加载加速 LoRA，4 步出视频，大幅缩短推理时间
- **H3-Regenerate-2K API**：本地生成 768P 初稿，再调用 MiniMax 官方云端 API 超分到 2K
- **Sage Attention**：在 ComfyUI Setting 中启用 Sage Attention，可进一步降低 H3 的推理显存

---

## 10. 目录结构

```
/home/creator/ws/200-video/miniMax/
├── README.md                     # 本文件
├── refer.md                      # 原参考教程（海螺 H3 本地化部署）
├── ComfyUI/
│   ├── .venv/                    # Python 3.10 虚拟环境
│   ├── main.py                   # ComfyUI 入口
│   ├── requirements.txt          # 依赖清单
│   ├── comfy_extras/
│   │   └── nodes_minimax_h3.py   # H3 原生节点（EmptyLatent / I2V / R2V / SigmaShift）
│   ├── comfy/ldm/minimax/        # H3 模型实现（model.py + vae.py + audio_vae.py）
│   ├── comfy/text_encoders/minimax.py  # MiniMax CLIP 文本编码器
│   └── models/
│       ├── diffusion_models/     # 放：minimax_h3_fl2va_pruned_fp8_scaled.safetensors（软链接）
│       ├── text_encoders/        # 放：qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors（软链接）
│       └── vae/                  # 放：minimax_h3_video_vae_fp16 / audio_vae_fp32（软链接）
```

---

## 11. 参考文档

| 资源 | 链接 |
|------|------|
| ComfyUI 官方 H3 教程 | https://docs.comfy.org/tutorials/video/minimax/minimax-h3 |
| Comfy-Org/MiniMax-H3（权重） | https://www.modelscope.cn/models/Comfy-Org/MiniMax-H3 |
| MiniMaxAI/MiniMax-H3（官方原版） | https://huggingface.co/MiniMaxAI/MiniMax-H3 |
| H3 本地部署指南（minimaxh3.run） | https://minimaxh3.run/zh-cn/minimax-h3-local-install |
