# OpenVLA 环境与 Demo 运行说明

本文记录在当前工作区运行 **OpenVLA-7B** 最小动作预测 demo 的环境、模型路径、运行命令和常见问题。

> 当前机器为 **Ubuntu 22.04 + RTX 5090（sm_120）+ Python 3.10 + PyTorch 2.11.0 cu128 + CUDA Toolkit 12.8**。不要直接照搬 OpenVLA 官方 README 中的 PyTorch 2.2.0 / CUDA 12.4 命令；如需编译 flash-attn 等 CUDA 扩展，必须使用 CUDA 12.8 的 `nvcc`，并设置 `TORCH_CUDA_ARCH_LIST="12.0"`。

## 1. 当前结论

### 1.1 已验证可运行配置

| 项目 | 当前配置 |
|---|---|
| GPU | NVIDIA GeForce RTX 5090 |
| GPU 架构 | `sm_120` / capability `(12, 0)` |
| Python | 3.10 |
| PyTorch | `2.11.0+cu128` |
| CUDA runtime | 12.8 |
| 推理 dtype | `torch.bfloat16` |
| 默认 attention 后端 | `sdpa` |
| 已验证高性能后端 | `flash_attention_2` |
| 模型来源 | ModelScope：`zixiaoBios/openvla-7b` |
| 模型缓存 | `~/.cache/modelscope/models/zixiaoBios--openvla-7b/snapshots/master/` |

### 1.2 已验证 demo 命令

```bash
cd /home/creator/ws/openVLA
source .venv/bin/activate
python demo_predict_action.py
```

成功时会看到类似输出：

```text
[OpenVLA Demo] Attention implementation: sdpa
[OpenVLA Demo] Model loaded: OpenVLAForActionPrediction
[OpenVLA Demo] Running model.predict_action(...)
[OpenVLA Demo] Prediction finished.
Predicted action: [ 0.00337 ... 0. ]
```

> 不传 `--image` 时，脚本会自动生成一张 224x224 灰色占位图。这个 demo 主要用于验证 **模型加载、图像预处理、文本 prompt、GPU 推理、动作解码** 整条链路是否可用。

## 2. 项目目录

```text
/home/creator/ws/openVLA
├── README.md                 # 本说明文档
├── demo_predict_action.py    # 最小动作预测 demo
├── .venv/                    # Python 3.10 虚拟环境
└── openvla/                  # OpenVLA 源码目录
```

如果 `openvla/` 源码目录不存在，可执行：

```bash
cd /home/creator/ws/openVLA
git clone https://github.com/xiaoxiaomengmeng/openvla.git
```

## 3. 快速开始

### 3.1 激活虚拟环境

```bash
cd /home/creator/ws/openVLA
source .venv/bin/activate
```

### 3.2 验证 PyTorch 与 GPU

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

### 3.3 运行默认占位图 demo

```bash
python demo_predict_action.py
```

当前脚本默认使用 `sdpa` attention。如需显式指定：

```bash
python demo_predict_action.py --attn-implementation sdpa
```

### 3.4 使用真实图片与指令

```bash
python demo_predict_action.py \
  --image /path/to/image.jpg \
  --instruction "pick up the object"
```

脚本会打印运行环境、模型加载状态、输入张量形状和最终预测动作。

## 4. 从零初始化环境

如果需要重新创建环境，可按本节执行。

### 4.1 创建 Python 3.10 虚拟环境

```bash
cd /home/creator/ws/openVLA
python3.10 -m venv --clear .venv
source .venv/bin/activate
python -m pip install -U pip
```

### 4.2 安装 RTX 5090 适配版 PyTorch

优先复用工作区顶层部署脚本：

```bash
cd /home/creator/ws/openVLA
source .venv/bin/activate
bash /home/creator/ws/setup_py310_cu128_rtx5090_torch2110.sh
```

### 4.3 安装基础依赖

```bash
source /home/creator/ws/openVLA/.venv/bin/activate
python -m pip install modelscope huggingface_hub pillow accelerate sentencepiece

cd /home/creator/ws/openVLA/repos/openvla
python -m pip install -r requirements-min.txt
```

如需以源码包方式导入 OpenVLA，可执行：

```bash
cd /home/creator/ws/openVLA/repos/openvla
python -m pip install -e . --no-deps
```

## 5. OpenVLA 模型下载与路径

### 5.1 使用 ModelScope 下载

```bash
source /home/creator/ws/openVLA/.venv/bin/activate
modelscope download --model zixiaoBios/openvla-7b
```

ModelScope 默认缓存路径：

```text
~/.cache/modelscope/models/zixiaoBios--openvla-7b/snapshots/master/
```

当前机器实际路径：

```text
/home/creator/.cache/modelscope/models/zixiaoBios--openvla-7b/snapshots/master/
```

`demo_predict_action.py` 默认模型路径：

```python
DEFAULT_MODEL_PATH = "/home/creator/.cache/modelscope/models/zixiaoBios--openvla-7b/snapshots/master"
```

### 5.2 关键模型文件

模型目录中应包含：

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

然后运行时显式指定：

```bash
python demo_predict_action.py --model-path /home/creator/ws/openVLA/models/openvla-7b
```

## 6. Demo 脚本说明

### 6.1 脚本入口

```text
/home/creator/ws/openVLA/demo_predict_action.py
```

### 6.2 常用参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--model-path` | ModelScope 默认缓存路径 | 本地模型目录或 Hugging Face repo id |
| `--image` | `None` | 输入 RGB 图片；不传则生成占位图 |
| `--instruction` | `pick up the object` | 机器人自然语言指令 |
| `--unnorm-key` | `bridge_orig` | 动作反归一化 key |
| `--attn-implementation` | `sdpa` | attention 后端，可选 `eager` / `sdpa` / `flash_attention_2` |

### 6.3 动作输出含义

`openvla-7b` 输出的是 **BridgeData / WidowX 风格 7DoF action**，通常可理解为：

```text
[x, y, z, roll, pitch, yaw, gripper]
```

注意：base `openvla-7b` 的最小 demo 只验证动作预测链路，不等同于能直接闭环控制 Franka Panda、LIBERO 或其他机械臂。

### 6.4 `unnorm_key` 为什么重要

OpenVLA 推理时并不是主要靠自然语言 prompt 里的机器人名字来选择动作尺度，而是通过 `unnorm_key` 选择模型内对应的数据集动作统计。

简化流程如下：

```text
7 个 action token
  -> 根据 unnorm_key 查找 norm_stats
  -> 反归一化为连续 7DoF action
  -> 交给具体机器人控制器 / IK / retargeting
```

当前 base demo 默认：

```text
unnorm_key = bridge_orig
```

这表示使用 BridgeData / WidowX 风格的动作分布来解释模型输出。换成 LIBERO 或其他机器人任务时，不能只在 instruction 里写机器人名字；需要使用对应微调 checkpoint、对应 `unnorm_key` / action statistics，以及对应的机器人控制适配层。

`bridge_orig` 不是随意命名的字符串。在 OpenVLA 官方 OXE 数据转换代码中，它对应：

```text
Original Version of Bridge V2 from Project Website
```

因此可以把 `bridge_orig` 理解为：**BridgeData V2 原始版数据集的动作统计 key**。BridgeData V2 主要来自 WidowX 系列机器人操作数据，所以它的动作尺度、夹爪语义和相机分布更接近 BridgeData / WidowX，而不是 Franka、UR5 或 LIBERO。

当前本地 ModelScope `openvla-7b` 的 `config.json` 中，`norm_stats` 支持的 key 如下：

```text
austin_buds_dataset_converted_externally_to_rlds
austin_sailor_dataset_converted_externally_to_rlds
austin_sirius_dataset_converted_externally_to_rlds
bc_z
berkeley_autolab_ur5
berkeley_cable_routing
berkeley_fanuc_manipulation
bridge_orig
cmu_stretch
dlr_edan_shared_control_converted_externally_to_rlds
dobbe
fmb_dataset
fractal20220817_data
furniture_bench_dataset_converted_externally_to_rlds
iamlab_cmu_pickup_insert_converted_externally_to_rlds
jaco_play
kuka
nyu_franka_play_dataset_converted_externally_to_rlds
roboturk
stanford_hydra_dataset_converted_externally_to_rlds
taco_play
toto
ucsd_kitchen_dataset_converted_externally_to_rlds
utaustin_mutex
viola
```

注意：这些 key 表示当前 checkpoint 内置了对应数据集的动作统计，**不等于模型已经在你的当前机器人/场景上达到可用成功率**。例如虽然列表里有 `nyu_franka_play_dataset_converted_externally_to_rlds`，但它对应的是 NYU Franka Play 数据统计，不等同于当前自定义 MuJoCo Franka 场景；如果要追求稳定抓取，仍需要控制适配、场景对齐，甚至微调。

## 7. Attention 后端选择

### 7.1 推荐默认使用 `sdpa`

当前 RTX 5090 环境下已验证：

```bash
python demo_predict_action.py --attn-implementation sdpa
```

可以正常完成 `model.predict_action(...)` 并输出 7DoF action。

### 7.2 不建议使用 `eager`

如果运行：

```bash
python demo_predict_action.py --attn-implementation eager
```

可能在 `model.predict_action(...)` 阶段出现：

```text
RuntimeError: The size of tensor a (275) must match the size of tensor b (274) at non-singleton dimension 3
```

原因是 OpenVLA 在插入视觉 token 后，Transformers LLaMA `eager` attention 路径里的 causal mask 长度与实际序列长度不一致。当前已将 demo 默认后端改为 `sdpa`。

### 7.3 已验证可用的 `flash_attention_2`

flash-attn 已在当前 RTX 5090 / `sm_120` 环境下完成验证，可以正常运行：

```bash
python demo_predict_action.py --attn-implementation flash_attention_2
```

成功时会看到：

```text
[OpenVLA Demo] Attention implementation: flash_attention_2
[OpenVLA Demo] Model loaded: OpenVLAForActionPrediction
[OpenVLA Demo] Prediction finished.
Predicted action: [ 0.00404102  0.01058371 ... 0. ]
```

如果加载时出现：

```text
You are attempting to use Flash Attention 2.0 without specifying a torch dtype.
```

但后续日志显示：

```text
[OpenVLA Demo] Model dtype: torch.bfloat16
```

并且 `model.predict_action(...)` 正常结束，则说明当前 demo 已按 `torch.bfloat16` 推理完成；该提示不影响本次验证结果。

日常稳定验证仍可直接使用默认 `sdpa`；需要性能测试或更省显存时，可切换到 `flash_attention_2`。

## 8. CUDA Toolkit 12.8 与 flash-attn

### 8.1 什么时候需要安装 CUDA Toolkit 12.8

仅运行当前 demo 时，**不一定需要编译 flash-attn**；`sdpa` 已可完成验证。当前环境也已验证 `flash_attention_2` 可正常完成 OpenVLA 推理，可作为性能优化后端使用。

如果要编译 flash-attn、xformers 或其他 CUDA 扩展，则需要系统安装 **CUDA Toolkit 12.8**，因为 PyTorch 自带的 CUDA runtime 不能替代 `nvcc`。

### 8.2 安装 CUDA Toolkit 12.8

添加 NVIDIA CUDA apt 源：

```bash
cd /tmp
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
```

安装：

```bash
sudo apt install -y cuda-toolkit-12-8
```

验证：

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

### 8.3 编译安装 flash-attn

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

## 9. LIBERO / MuJoCo 后续 demo

如果要跑 LIBERO / MuJoCo 仿真闭环，需要额外安装 LIBERO，并使用 OpenVLA 官方 LIBERO 微调权重，而不是只使用 base `openvla-7b`。

### 9.1 安装 LIBERO 相关依赖

```bash
cd /home/creator/ws/openVLA/repos/openvla
source /home/creator/ws/openVLA/.venv/bin/activate
python -m pip install -r experiments/robot/libero/libero_requirements.txt
```

### 9.2 安装 LIBERO 源码

```bash
cd /home/creator/ws/openVLA
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
python -m pip install -e .
```

### 9.3 运行最小 LIBERO-Spatial 评测

```bash
cd /home/creator/ws/openVLA/repos/openvla
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

不要使用 `eager` 作为默认后端。使用：

```bash
python demo_predict_action.py
# 或
python demo_predict_action.py --attn-implementation sdpa
```

### 10.3 flash-attn 报 `no kernel image is available`

这是 flash-attn 没有为 RTX 5090 的 `sm_120` 架构正确编译。确认：

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

如果 flash-attn 尚未编译成功，可先使用 `sdpa` 验证模型加载和推理流程；当前环境已验证 flash-attn 可用，编译成功后可以切换到 `flash_attention_2`。

### 10.4 `apt install cuda-toolkit-12-8` 提示 broken packages

先修复系统 APT 状态：

```bash
sudo apt --fix-broken install
```

如果破损来自无关软件包，可先移除该软件包后重试 CUDA Toolkit 安装。此前遇到过 `baidunetdisk` 依赖 `libgtkmm-2.4-1v5` 阻塞全局安装的情况。
