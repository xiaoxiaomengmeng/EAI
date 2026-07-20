# OpenVLA 环境与 Demo 运行说明

本文记录在当前工作区基于 **Ubuntu 22.04 + RTX 5090（sm_120）+ Python 3.10 + PyTorch 2.11.0 cu128 + CUDA Toolkit 12.8** 准备 OpenVLA demo 环境的步骤。

> 当前机器是 RTX 5090，不要直接照搬 OpenVLA 官方 README 中的 PyTorch 2.2.0 / CUDA 12.4 相关命令。CUDA 扩展（如 flash-attn）需要使用 CUDA 12.8 的 `nvcc`，并显式设置 `TORCH_CUDA_ARCH_LIST="12.0"`。

## 1. 目录结构

当前 OpenVLA 相关目录位于：

```text
/home/creator/ws/openVLA
├── README.md                 # 本说明文档
├── demo_predict_action.py    # 最小动作预测 demo
└── openvla/                  # OpenVLA 源码目录
```

OpenVLA 源码可通过以下方式准备：

```bash
conda deactivate
cd /home/creator/ws/openVLA
git clone https://github.com/xiaoxiaomengmeng/openvla.git
```

如果 `openvla/` 目录已存在，则无需重复 clone。

## 2. 创建 Python 3.10 虚拟环境

```bash
cd /home/creator/ws/openVLA
python3.10 -m venv --clear .venv
source .venv/bin/activate
python -m pip install -U pip
```

后续命令默认在该虚拟环境内执行：

```bash
source /home/creator/ws/openVLA/.venv/bin/activate
```

## 3. 安装 RTX 5090 适配版 PyTorch

当前环境使用 **PyTorch 2.11.0+cu128**。优先复用工作区顶层的部署脚本：

```bash
cd /home/creator/ws/openVLA
source .venv/bin/activate
bash /home/creator/ws/setup_py310_cu128_rtx5090_torch2110.sh
```

验证 PyTorch 与 GPU：

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

期望关键信息：

```text
torch: 2.11.0+cu128
torch cuda: 12.8
gpu: NVIDIA GeForce RTX 5090
capability: (12, 0)
```

## 4. 安装基础依赖

```bash
source /home/creator/ws/openVLA/.venv/bin/activate
python -m pip install modelscope huggingface_hub pillow accelerate sentencepiece

cd /home/creator/ws/openVLA/openvla
python -m pip install -r requirements-min.txt
```

如需以源码包方式导入 OpenVLA，可执行：

```bash
cd /home/creator/ws/openVLA/openvla
python -m pip install -e . --no-deps
```

## 5. 安装 CUDA Toolkit 12.8

PyTorch 自带的 CUDA runtime 不能替代 `nvcc`。为 RTX 5090 编译 flash-attn 等 CUDA 扩展时，需要系统安装 **CUDA Toolkit 12.8**。

### 5.1 添加 NVIDIA CUDA apt 源

```bash
cd /tmp
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
```

### 5.2 安装 CUDA Toolkit

```bash
sudo apt install -y cuda-toolkit-12-8
```

如果遇到 `You might want to run 'apt --fix-broken install'`，先修复系统 APT 依赖：

```bash
sudo apt --fix-broken install
sudo apt install -y cuda-toolkit-12-8
```

如果 broken package 来自无关软件（例如 `baidunetdisk`），可先移除阻塞包后再安装：

```bash
sudo apt remove baidunetdisk
sudo apt --fix-broken install
sudo apt install -y cuda-toolkit-12-8
```

### 5.3 验证 CUDA 12.8

```bash
ls -d /usr/local/cuda*
/usr/local/cuda-12.8/bin/nvcc --version
```

期望看到：

```text
/usr/local/cuda-12.8
Cuda compilation tools, release 12.8
```

临时切换当前 shell 到 CUDA 12.8：

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

which nvcc
nvcc --version
```

确认 `which nvcc` 输出：

```text
/usr/local/cuda-12.8/bin/nvcc
```

## 6. 编译安装 flash-attn

RTX 5090 是 `sm_120` 架构，不能直接使用不含 `sm_120` 的预编译 wheel。必须使用 CUDA 12.8 从源码编译，并设置 `TORCH_CUDA_ARCH_LIST="12.0"`。

```bash
source /home/creator/ws/openVLA/.venv/bin/activate

pip uninstall -y flash-attn flash_attn
rm -rf ~/.cache/pip ~/.cache/torch_extensions

python -m pip install -U pip setuptools wheel packaging ninja psutil

export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="12.0"
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export MAX_JOBS=2

pip install flash-attn --no-build-isolation --no-cache-dir --no-binary flash-attn -v
```

如果编译过程内存不足，可降低并行数：

```bash
export MAX_JOBS=2
pip install flash-attn --no-build-isolation --no-cache-dir --no-binary flash-attn -v
```

验证 flash-attn：

```bash
python - <<'PY'
import torch
import flash_attn
from flash_attn import flash_attn_func

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("flash_attn:", flash_attn.__version__)
print("gpu:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))

q = torch.randn(1, 128, 8, 64, device="cuda", dtype=torch.bfloat16)
k = torch.randn(1, 128, 8, 64, device="cuda", dtype=torch.bfloat16)
v = torch.randn(1, 128, 8, 64, device="cuda", dtype=torch.bfloat16)

out = flash_attn_func(q, k, v, causal=True)
torch.cuda.synchronize()
print("ok:", out.shape, out.dtype, out.device)
PY
```

如果出现 `CUDA error: no kernel image is available for execution on the device`，通常表示当前 flash-attn 没有正确编译进 `sm_120`，需确认：

```bash
which nvcc
nvcc --version
echo $TORCH_CUDA_ARCH_LIST
```

应满足：

```text
/usr/local/cuda-12.8/bin/nvcc
Cuda compilation tools, release 12.8
12.0
```

## 7. OpenVLA 基座模型

`openvla-7b` 使用 ModelScope 下载：

```bash
source /home/creator/ws/openVLA/.venv/bin/activate
modelscope download --model zixiaoBios/openvla-7b
```

ModelScope 默认缓存路径：

```text
~/.cache/modelscope/models/zixiaoBios--openvla-7b/snapshots/master/
```

当前完整路径：

```text
/home/creator/.cache/modelscope/models/zixiaoBios--openvla-7b/snapshots/master/
```

代码中优先直接使用该路径：

```python
model_path = "/home/creator/.cache/modelscope/models/zixiaoBios--openvla-7b/snapshots/master"
```

关键文件包括：

```text
config.json
configuration_prismatic.py
model-00001-of-00003.safetensors
model-00002-of-00003.safetensors
model-00003-of-00003.safetensors
model.safetensors.index.json
modeling_prismatic.py
processing_prismatic.py
tokenizer.json
tokenizer.model
```

如需在项目目录内固定一份模型副本，可手动复制或同步到：

```text
/home/creator/ws/openVLA/models/openvla-7b
```

## 8. 运行最小动作预测 demo

`demo_predict_action.py` 用于验证 OpenVLA 模型加载与动作预测链路。

### 8.1 使用默认占位图运行

```bash
cd /home/creator/ws/openVLA
source .venv/bin/activate
python demo_predict_action.py
```

当前 demo 默认使用 `sdpa` attention。也可以显式指定：

```bash
python demo_predict_action.py --attn-implementation sdpa
```

如果 flash-attn 已通过验证，可切换到：

```bash
python demo_predict_action.py --attn-implementation flash_attention_2
```

### 8.2 使用真实图片与指令

```bash
python demo_predict_action.py \
  --image /path/to/image.jpg \
  --instruction "pick up the object" \
  --attn-implementation sdpa
```

脚本会打印运行环境、模型加载状态、输入张量形状和最终预测动作。

> 注意：基座 `openvla-7b` 输出的是 BridgeData/WidowX 风格的 7DoF action。该 demo 主要用于验证推理链路，不等同于可直接控制 Franka Panda 或 LIBERO 的闭环策略。

## 9. LIBERO / MuJoCo 后续 demo

如果要跑 LIBERO/MuJoCo 仿真闭环，需要额外安装 LIBERO，并使用 OpenVLA 官方 LIBERO 微调权重，而不是只使用 base `openvla-7b`。

安装 LIBERO 相关依赖：

```bash
cd /home/creator/ws/openVLA/openvla
source /home/creator/ws/openVLA/.venv/bin/activate
python -m pip install -r experiments/robot/libero/libero_requirements.txt
```

安装 LIBERO 源码：

```bash
cd /home/creator/ws/openVLA
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
python -m pip install -e .
```

运行一个最小 LIBERO-Spatial 评测示例：

```bash
cd /home/creator/ws/openVLA/openvla
source /home/creator/ws/openVLA/.venv/bin/activate

python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --center_crop True \
  --num_trials_per_task 1 \
  --use_wandb False
```

常用 LIBERO 微调权重：

```text
openvla/openvla-7b-finetuned-libero-spatial
openvla/openvla-7b-finetuned-libero-object
openvla/openvla-7b-finetuned-libero-goal
openvla/openvla-7b-finetuned-libero-10
```

## 10. 常见问题

### 10.1 `demo.jpg` 不存在

当前 demo 支持不传 `--image`，会自动生成一张 224x224 占位图用于验证推理链路：

```bash
python demo_predict_action.py
```

如需真实图片，显式传入：

```bash
python demo_predict_action.py --image /path/to/image.jpg
```

### 10.2 `eager` attention 报 causal mask 长度不一致

如果使用：

```bash
python demo_predict_action.py --attn-implementation eager
```

可能在 `model.predict_action(...)` 阶段出现类似错误：

```text
RuntimeError: The size of tensor a (275) must match the size of tensor b (274) at non-singleton dimension 3
```

这是 OpenVLA 在插入视觉 token 后，Transformers LLaMA `eager` attention 路径里的 causal mask 长度与实际序列长度不一致导致的。当前 RTX 5090 环境下已验证 `sdpa` 可以正常完成推理，demo 默认也已改为 `sdpa`：

```bash
python demo_predict_action.py
# 或显式指定
python demo_predict_action.py --attn-implementation sdpa
```

### 10.3 flash-attn 报 `no kernel image is available`

这是 flash-attn 没有为 RTX 5090 的 `sm_120` 架构正确编译。确认：

```bash
which nvcc
nvcc --version
echo $TORCH_CUDA_ARCH_LIST
```

必须使用 CUDA 12.8 的 `nvcc`，并设置：

```bash
export TORCH_CUDA_ARCH_LIST="12.0"
```

### 10.4 `apt install cuda-toolkit-12-8` 提示 broken packages

先修复系统 APT 状态：

```bash
sudo apt --fix-broken install
```

如果破损来自无关软件包，可先移除该软件包后重试 CUDA Toolkit 安装。此前遇到过 `baidunetdisk` 依赖 `libgtkmm-2.4-1v5` 阻塞全局安装的情况。

### 10.5 暂时不使用 flash-attn

如果 flash-attn 尚未编译成功，可先使用 PyTorch attention 后端：

```bash
python demo_predict_action.py --attn-implementation sdpa
```

`sdpa` 可用于临时验证模型加载和推理流程；待 flash-attn 编译成功后再切换到 `flash_attention_2`。
