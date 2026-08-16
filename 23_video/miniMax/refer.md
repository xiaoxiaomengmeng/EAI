MiniMax‑H3 完整介绍 + Ubuntu22.04 + RTX5090‑32G 本地化部署教程
一、MiniMax H3 模型简介
基础参数
MiniMax‑H3 是 33B 参数全模态 Omni‑Transformer 音视频生成大模型（不是文本大模型），可以文字一键生成带同步音频、口型对齐的短视频

    双核心权重组件（必须全部准备）
        FL2VA：文本→视频音频、首尾帧驱动生成，磁盘占用约 135GB
        Ref2VA：参考图 / 参考视频驱动生成，磁盘占用约 168GB
    文本编码器：Qwen3‑VL‑32B
    原生输出分辨率 768P；搭配官方上云增强管线最高输出 2K 视频
    特性：视频 + 音频联合输出、人物口型和语音自动对齐、支持图生视频、视频续剪

显存 / 内存硬件门槛（你的 RTX5090 32G）
表格
精度方案	显存占用	整机内存最低	适配你的单卡 5090‑32G
BF16 全精度	135GB	384GB	❌ 完全跑不动
INT8 全量权重	85GB	123GB	❌ 超出 32G 显存
INT8 裁剪量化版（社区 ComfyUI 优化）	21‑30GB	≥48GB	✅ 唯一可行方案

    你的硬件：Ubuntu22.04 + RTX5090 32G，只能采用 INT8 量化 + ComfyUI 可视化工作流部署；官方 SGLang/vLLM 原生 BF16 推理最少需要 4 张 H100，单卡无法运行原版权重

二、前置环境配置 Ubuntu 22.04
1、安装 NVIDIA 显卡驱动、CUDA‑12.4（推荐）
bash

# 1.禁用nouveau开源驱动
sudo bash -c "echo blacklist nouveau > /etc/modprobe.d/blacklist-nouveau.conf"
sudo bash -c "echo options nouveau modeset=0 >> /etc/modprobe.d/blacklist-nouveau.conf"
sudo update-initramfs -u
sudo reboot

# 重启后安装驱动
sudo apt update
sudo apt install nvidia-driver-565
sudo reboot

# 验证
nvidia-smi

    RTX5090 为 Blackwell 架构，必须 CUDA≥12.2

2、安装系统依赖
bash

sudo apt install -y python3 python3-pip python3-venv git ffmpeg build-essential

3、创建独立 Python 虚拟环境
bash

mkdir ~/minimax-h3 && cd ~/minimax-h3
python3 -m venv venv
source venv/bin/activate

# 安装适配cuda124的pytorch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install flash-attn --no-build-isolation

三、部署路线选择

    首选：ComfyUI + INT8 预量化权重（适合你的单卡 32G，可视化界面，门槛最低）
    SGLang/vLLM：仅多卡 80G + 显卡可用，直接放弃
    diffusers：适合二次开发，单卡显存压力巨大

下面只讲解你硬件适配的 ComfyUI 完整部署流程
步骤 1：下载指定版本 ComfyUI（H3 经过验证的 commit）
bash

git clone https://github.com/comfyanonymous/ComfyUI
cd ComfyUI
git checkout 14b05228cef127ce529bc0c08660770d4af3e9a8
pip install -r requirements.txt

步骤 2：安装 MiniMax‑H3 ComfyUI 自定义节点
bash

cd custom_nodes
git clone https://github.com/Comfy‑Org/comfyui‑minimax‑h3
cd comfyui‑minimax‑h3
pip install -r requirements.txt

步骤 3：下载 INT8 量化权重（关键！不要下载原版 BF16）
权重目录结构，把文件放置到 ComfyUI/models/
plaintext

models/
├─diffusion_models/
│   ├─minimax_h3_fl2va_int8.safetensors
│   └─minimax_h3_ref2va_int8.safetensors
├─text_encoders/
│   └─qwen3vl_32b_minimax_h3_int8.safetensors
└─vae/
   ├─h3_video_vae.safetensors
   └─h3_audio_vae.safetensors

    原版 HuggingFace 官方 BF16 权重需要权限申请，单卡 32G 无法加载；直接下载社区已经做好的 INT8 裁剪量化版权重

步骤 4：配置环境变量优化显存释放
bash

# 写入环境变量，开启高性能XET加速
echo 'export HF_XET_HIGH_PERFORMANCE=1' >> ~/.bashrc
echo 'export PYTHONUNBUFFERED=1' >> ~/.bashrc
source ~/.bashrc

步骤 5：启动 ComfyUI 服务
bash

# 回到ComfyUI根目录
cd ~/minimax-h3/ComfyUI
source ../venv/bin/activate

# 启动，允许局域网访问
python main.py --listen 0.0.0.0 --port 8188

浏览器打开 http://本机IP:8188

    加载 H3 官方工作流 json
    加载 FL2VA 节点，开启 8bit 量化加载、CPU 内存卸载
    输入提示词，即可生成 768P 带音频短视频

四、关键优化，防止 RTX5090‑32G 爆显存 OOM

    必须开启模型 CPU 内存卸载：ComfyUI 节点内开启 model_cpu_offload
    视频时长：初期只生成 2‑4s 短视频，长视频显存压力暴涨
    关闭不必要后台程序，整机系统内存建议≥64G（最低 48G）
    启用 VAE 分块 vae‑use‑tiling，大幅降低 VAE 显存占用
    不要同时加载 FL2VA 和 Ref2VA，推理哪个就加载哪一个权重

五、常见踩坑清单

    flash‑attn 编译失败：安装 CUDA‑12.4 环境、gcc 版本≥11
    权重加载 OOM：确认你下载INT8 量化版，不要原版 BF16；开启 CPU 卸载
    生成视频无音频：系统 ffmpeg 版本不能过低，必须支持音频编码
    Qwen3‑VL‑32B 文本编码器显存爆炸：文本编码器强制使用 INT8 量化加载

六、进阶方案

    如果需要输出 2K 高清视频：本地 H3 生成 768P 初稿，调用 MiniMax 官方云端 API 做超分升级到 2K 分辨率
    想要更快出片：加载 MiniMax‑H3‑Turbo‑LoRA 加速 LoRA，4 步即可生成视频，缩短推理耗时

