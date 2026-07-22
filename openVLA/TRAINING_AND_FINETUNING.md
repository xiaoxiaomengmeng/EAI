# OpenVLA 训练、数据格式与微调总结

本文面向当前 `/home/creator/ws/openVLA` 工作区，整理 `openvla-7b` 的训练逻辑、数据格式、动作 token 化、从头训练、LoRA 微调、全量微调和当前环境注意事项。

它不是官方 README 的完整翻译，而是把本项目后续接入自定义机器人、理解 `unnorm_key`、准备微调数据时最关键的概念集中说明。

## 0. 初学者术语与关键参数速查

这一节先把常见缩写、关键名称和训练参数解释清楚。后面章节会继续展开它们在代码中的具体位置。

### 0.1 模型与任务相关名称

```text
OpenVLA = Open Vision-Language-Action
```

OpenVLA 是“视觉-语言-动作”模型。它输入图像和语言指令，输出机器人动作。

```text
VLA = Vision-Language-Action
```

VLA 指把视觉、语言和动作统一建模的机器人策略模型。OpenVLA 就是一类 VLA。

```text
VLM = Vision-Language Model
```

VLM 是视觉语言模型，通常只做图像理解、问答或描述。OpenVLA 是在 VLM 基础上继续训练，让它输出机器人动作。

```text
LLM = Large Language Model
```

LLM 是大语言模型。OpenVLA 里的语言骨干来自 Llama-2 系列，所以它可以用自回归生成 token 的方式生成 action token。

```text
Prismatic
```

Prismatic 是 OpenVLA 依赖的 VLM 训练框架。全量训练和从头训练主要使用 Prismatic 格式的模型与训练脚本。

```text
HF = Hugging Face
```

Hugging Face 是模型加载和保存生态。当前推理常用 `AutoProcessor` 和 `AutoModelForVision2Seq` 加载 HF 格式 checkpoint。

### 0.2 数据相关名称

```text
RLDS = Reinforcement Learning Datasets
```

RLDS 是强化学习/机器人数据集格式，用 episode、step、observation、action、reward 等结构存储机器人轨迹。OpenVLA 官方训练管道默认支持 RLDS。

```text
TFDS = TensorFlow Datasets
```

TFDS 是 TensorFlow 的数据集加载格式。OpenVLA 的 RLDS 数据通常通过 TFDS 管道读取，再包装成 PyTorch `IterableDataset`。

```text
OXE = Open-X Embodiment
```

Open-X Embodiment 是多机器人、多任务的大规模机器人数据集合。`openvla-7b` 预训练使用了其中的大规模混合数据。

```text
BridgeData V2
```

BridgeData V2 是 Berkeley/Bridge 系列 WidowX 机器人操作数据集。当前 base `openvla-7b` 推理常用的 `bridge_orig` 就来自它。

```text
DROID
```

DROID 是大规模真实机器人操作数据集。OpenVLA 微调配置里有 `droid_wipe` 这类示例任务。

```text
LIBERO
```

LIBERO 是基于 MuJoCo / robosuite 的机器人操作 benchmark。LIBERO 高成功率评测需要使用 LIBERO 微调 checkpoint，而不是直接用 base `openvla-7b`。

### 0.3 动作相关名称

```text
7DoF = 7 Degrees of Freedom
```

在 OpenVLA 里通常指 7 维末端动作：

```text
[dx, dy, dz, droll, dpitch, dyaw, gripper]
```

其中前三维是末端位置增量，中间三维是末端姿态增量，最后一维是夹爪开合信号。

```text
EEF = End Effector
```

EEF 是末端执行器，例如机械臂夹爪或手爪。OpenVLA 输出的是末端相对动作，不是每个关节角。

```text
action token
```

action token 是把连续动作离散化后的 token。OpenVLA 像语言模型生成文字 token 一样生成动作 token。

```text
ActionTokenizer
```

ActionTokenizer 把连续动作映射到 tokenizer 词表末尾的 256 个动作 token，也能把动作 token 解码回归一化后的连续动作。

```text
norm_stats = normalization statistics
```

`norm_stats` 是动作归一化统计，例如每个动作维度的 `q01`、`q99`。训练时用它把真实动作归一化，推理时用它把模型输出反归一化。

```text
unnorm_key = un-normalization key
```

`unnorm_key` 是选择哪套 `norm_stats` 的 key。例如 `bridge_orig` 表示使用 BridgeData V2 原始版动作统计。

```text
bridge_orig = Original Version of Bridge V2 from Project Website
```

`bridge_orig` 可以理解为 BridgeData V2 原始版动作统计 key。它不是机器人名字，而是动作统计名称。

### 0.4 训练方法相关名称

```text
LoRA = Low-Rank Adaptation
```

LoRA 是低秩适配微调方法。它不直接训练全部 7B 参数，而是在模型线性层旁边加小的低秩矩阵，只训练这些新增参数。

```text
PEFT = Parameter-Efficient Fine-Tuning
```

PEFT 是参数高效微调。LoRA 是 PEFT 的一种。OpenVLA 的 `finetune.py` 使用 Hugging Face PEFT 库实现 LoRA。

```text
FSDP = Fully Sharded Data Parallel
```

FSDP 是 PyTorch 的分布式训练策略，会把模型参数、梯度和优化器状态切分到多张 GPU 上，适合全量训练 7B 模型。

```text
DDP = Distributed Data Parallel
```

DDP 是 PyTorch 常见多 GPU 训练方式。OpenVLA 的 LoRA 微调脚本用 DDP 包装模型。

```text
BF16 = Brain Floating Point 16
```

BF16 是 16 位浮点格式，动态范围比 FP16 更稳。OpenVLA 推理和训练默认常用 `torch.bfloat16`。

```text
FP32 = Floating Point 32
```

FP32 是 32 位浮点格式。全量训练加载基础模型时通常先验证参数是 FP32，再进入混合精度训练。

```text
4bit / NF4 quantization
```

4bit 量化把模型权重压到更低位宽，减少显存占用。NF4 是常见 4bit 量化格式。它能省显存，但可能影响性能。

```text
checkpoint
```

checkpoint 是训练中保存的模型状态。它可以用于恢复训练，也可以用于后续推理或评测。

```text
adapter
```

在 LoRA 语境下，adapter 是新增的低秩参数模块。训练后可以单独保存，也可以 merge 到 base 模型里。

### 0.5 LoRA 微调关键参数怎么读

```text
--vla_path
```

要加载的 base VLA 模型路径或 Hugging Face repo，例如 `openvla/openvla-7b`。本项目如果用本地模型，也可以换成本地 checkpoint 路径。

```text
--data_root_dir
```

RLDS / Open-X 数据集根目录。它下面应包含类似 `bridge_orig/` 这样的数据集子目录。

```text
--dataset_name
```

本次微调使用哪个数据集。它必须能被 OpenVLA 数据管道识别，例如 `bridge_orig` 或你自己注册的新数据集名。

```text
--run_root_dir
```

训练日志和 checkpoint 保存根目录。

```text
--adapter_tmp_dir
```

LoRA adapter 临时保存目录。脚本保存 checkpoint 时会先保存 adapter，再合并到完整模型。

```text
--batch_size
```

每张 GPU 每次前向/反向处理的样本数。越大越吃显存，太小可能训练不稳定。

```text
--grad_accumulation_steps
```

梯度累积步数。显存不够时，可以降低 `batch_size`，增加这个参数来保持有效 batch size。

```text
effective batch size = batch_size × grad_accumulation_steps × GPU 数量
```

```text
--learning_rate
```

学习率，控制参数更新幅度。LoRA 默认示例是 `5e-4`；太大容易发散，太小训练慢或学不到。

```text
--max_steps
```

最大训练步数。数据越复杂、任务越难，通常需要更多 step，但也更容易过拟合。

```text
--save_steps
```

每隔多少个 gradient step 保存一次 checkpoint。

```text
--lora_rank
```

LoRA 低秩矩阵的 rank。rank 越大，可训练容量越强，但显存和计算也更多。OpenVLA 默认示例常用 `32`。

```text
--lora_dropout
```

LoRA 分支上的 dropout。用于抑制过拟合。官方默认示例为 `0.0`。

```text
--use_quantization
```

是否启用 4bit 量化训练。它能降低显存，但可能影响最终效果。当前应优先不量化，显存不够时再考虑。

```text
--image_aug
```

是否启用图像增强。相机视角、光照、背景变化较大时建议开启；如果只是复现已见数据，关闭后训练指标可能很高但泛化较弱。

### 0.6 全量训练关键参数怎么读

```text
--vla.type
```

选择 `prismatic/conf/vla.py` 中注册的训练配置，例如 `prism-dinosiglip-224px+mx-bridge`。它决定基础 VLM、数据 mixture、batch size、学习率和训练策略。

```text
--pretrained_checkpoint
```

要继续训练或全量微调的 Prismatic 格式 checkpoint。如果不传，脚本会从配置中的 `base_vlm` 开始训练。

```text
--is_resume
```

是否是恢复中断训练。微调已有 checkpoint 时通常设为 `False`；从中断点继续训练时设为 `True`，并提供 `resume_step` / `resume_epoch`。

```text
--save_interval
```

FSDP 训练中保存 checkpoint 的间隔。

```text
expected_world_size
```

配置期望的 GPU 数量。`train.py` 会检查实际 GPU 数是否匹配；不匹配会直接报错。

```text
global_batch_size / per_device_batch_size
```

`global_batch_size` 是全局总 batch，`per_device_batch_size` 是每张 GPU 上的 batch。两者和 GPU 数量共同决定是否需要梯度累积。

```text
freeze_vision_backbone
freeze_llm_backbone
unfreeze_last_llm_layer
```

这三个布尔参数决定训练哪些模块：视觉骨干、语言模型骨干、最后一层。它们控制全量训练、冻结视觉、sandwich train、last-layer train 等不同策略。

### 0.7 初学者最容易混淆的几组概念

```text
checkpoint vs norm_stats
```

checkpoint 是模型权重；`norm_stats` 是动作统计。模型能加载不代表动作尺度一定正确，推理时还要选对 `unnorm_key`。

```text
prompt 里的机器人名字 vs robot adapter
```

prompt 主要告诉模型任务目标；真正把 7D action 转成 Franka/WX250S 控制命令的是 robot adapter、IK 和 gripper controller。

```text
LoRA 微调 vs 全量微调
```

LoRA 只训练少量 adapter 参数，省显存、适合先试；全量微调训练大量甚至全部参数，资源重但适配能力更强。

```text
训练成功 vs 机器人执行成功
```

训练 loss 下降或 action accuracy 高，不等于真实机器人一定成功。还要看相机视角、动作尺度、控制器、夹爪逻辑和环境物理是否匹配。

```text
base openvla-7b vs LIBERO 微调 checkpoint
```

base `openvla-7b` 适合 BridgeData / OXE 风格验证；LIBERO benchmark 应使用 `openvla-7b-finetuned-libero-*`，否则可能缺少对应 `libero_*` 动作统计。

## 1. 一句话理解 OpenVLA 训练

OpenVLA 把机器人控制问题建模成“看图 + 读指令 -> 生成动作 token”：

```text
RGB 图像 + 语言指令
  -> VLM / LLM 自回归生成 7 个 action token
  -> action tokenizer 解码为归一化动作
  -> 结合 norm_stats / unnorm_key 反归一化
  -> 得到连续 7DoF 末端动作
```

训练时，动作被当成语言模型要预测的特殊 token；推理时，再把这些 token 还原成连续动作。

## 2. 关键代码位置

### 官方训练入口

```text
openvla/vla-scripts/train.py
```

作用：使用 Prismatic 训练栈和 PyTorch FSDP 训练 VLA。适合从头训练、继续训练、全量微调或不同冻结策略的微调。

### LoRA 微调入口

```text
openvla/vla-scripts/finetune.py
```

作用：基于 Hugging Face `AutoModelForVision2Seq` 加载 `openvla-7b`，使用 PEFT / LoRA 做参数高效微调，可选 4bit 量化。

### 数据加载与样本转换

```text
openvla/prismatic/vla/datasets/datasets.py
openvla/prismatic/vla/materialize.py
openvla/prismatic/vla/datasets/rlds/oxe/
```

作用：加载 RLDS / Open-X Embodiment 格式数据，统一图像、语言、动作字段，并转成 OpenVLA 训练批次。

### 动作 token 化

```text
openvla/prismatic/vla/action_tokenizer.py
```

作用：把连续动作离散成 LLM tokenizer 词表末尾的 action token，也可以把 action token 解码回连续归一化动作。

### 训练配置注册

```text
openvla/prismatic/conf/vla.py
```

作用：定义不同训练实验配置，包括数据混合、基础 VLM、冻结策略、batch size、学习率、FSDP 策略等。

## 3. 训练数据格式

OpenVLA 官方训练默认使用 RLDS / TensorFlow Datasets 管道，数据来源主要是 Open-X Embodiment、BridgeData V2、DROID、LIBERO 转换数据等。

单步训练样本核心字段可以理解为：

```text
observation.image_primary        # 主相机 RGB 图像
task.language_instruction        # 自然语言任务指令
action                           # 7D 连续动作
dataset_name                     # 数据集名称，用于统计和追踪
```

在 `RLDSBatchTransform` 中，它会被转成：

```text
pixel_values     # 图像张量，输入视觉编码器
input_ids        # 文本 prompt + action token 的 token id
labels           # 只对 action token 部分计算 loss
dataset_name     # 当前样本来自哪个数据集
```

训练 prompt 形式大致是：

```text
Human: What action should the robot take to <language_instruction>?
Assistant: <action_tokens>
```

关键点：OpenVLA 不是训练成普通聊天模型，而是训练成在该 prompt 后面生成动作 token 的模型。

## 4. Action Tokenizer 如何工作

OpenVLA 的动作是 7DoF：

```text
[dx, dy, dz, droll, dpitch, dyaw, gripper]
```

训练前，每一维动作会先被归一化到近似 `[-1, 1]`。`ActionTokenizer` 再把连续值离散成 256 个 bin，并映射到 tokenizer 词表末尾的 256 个 token。

简化流程：

```text
连续 7D action
  -> clip 到 [-1, 1]
  -> 每一维离散成 256 个 bin
  -> 映射成 7 个 action token
  -> 作为语言模型 labels 训练
```

推理时反过来：

```text
模型生成 7 个 action token
  -> decode_token_ids_to_actions
  -> 得到归一化动作
  -> 根据 unnorm_key 查 norm_stats
  -> 反归一化成真实连续动作
```

## 5. norm_stats 与 unnorm_key

`norm_stats` 是每个数据集的动作统计，通常包含动作每一维的分位数或范围，例如 `q01`、`q99`。

训练时：

```text
真实动作 -> 用数据集统计归一化 -> action tokenizer 离散成 token
```

推理时：

```text
action token -> 解码为归一化动作 -> 用 unnorm_key 对应统计反归一化
```

所以 `unnorm_key` 决定输出动作的物理尺度。当前 base `openvla-7b` 常用：

```text
unnorm_key = bridge_orig
```

`bridge_orig` 表示 `Original Version of Bridge V2 from Project Website`，也就是 BridgeData V2 原始版动作统计 key。完整支持列表见 `DAMO_README.md` 中的 `norm_stats` 章节。

## 6. LoRA 微调方法

LoRA 微调是当前最实用的路线，入口是：

```text
openvla/vla-scripts/finetune.py
```

它的特点：

```text
加载 openvla-7b HF checkpoint
  -> 给线性层挂 LoRA adapter
  -> 只训练少量 LoRA 参数
  -> 保存 adapter 或合并后的模型
```

默认关键参数来自 `FinetuneConfig`：

```text
vla_path = openvla/openvla-7b
data_root_dir = datasets/open-x-embodiment
dataset_name = droid_wipe
batch_size = 16
max_steps = 200000
learning_rate = 5e-4
use_lora = True
lora_rank = 32
lora_dropout = 0.0
use_quantization = False
```

官方注释里的显存经验：

```text
LoRA rank=32, target_modules=all-linear
48GB GPU: batch size 约 12
80GB GPU: batch size 约 24
```

典型命令模板：

```bash
cd /home/creator/ws/openVLA/openvla
source /home/creator/ws/openVLA/.venv/bin/activate

torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir <PATH_TO_RLDS_DATASETS> \
  --dataset_name bridge_orig \
  --run_root_dir <PATH_TO_LOG_AND_CHECKPOINT_DIR> \
  --adapter_tmp_dir <PATH_TO_TEMP_ADAPTER_DIR> \
  --lora_rank 32 \
  --batch_size 12 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug True \
  --save_steps 5000
```

多 GPU 时，把 `--nproc-per-node` 改成 GPU 数量即可。小显存时降低 `batch_size`，再提高 `grad_accumulation_steps` 保持有效 batch size。

## 7. 全量微调方法

全量微调使用：

```text
openvla/vla-scripts/train.py
```

它不是直接用 HF `openvla-7b` checkpoint，而是使用 Prismatic 兼容格式 checkpoint，例如官方提到的：

```text
openvla/openvla-7b-prismatic
```

全量微调适合：

```text
你的机器人/相机/动作分布和预训练数据差异很大
LoRA 适配效果不足
你有足够 GPU 资源和数据量
```

官方推荐量级是 8 张 A100 80GB 级别资源；当前 RTX 5090 单卡更适合先做 LoRA 或小规模验证。

典型命令模板：

```bash
cd /home/creator/ws/openVLA/openvla
source /home/creator/ws/openVLA/.venv/bin/activate

torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/train.py \
  --pretrained_checkpoint <PATH_TO_PRISMATIC_CHECKPOINT_PT> \
  --vla.type prism-dinosiglip-224px+mx-bridge \
  --data_root_dir <PATH_TO_RLDS_DATASETS> \
  --run_root_dir <PATH_TO_LOG_AND_CHECKPOINT_DIR> \
  --image_aug True \
  --save_interval 2500 \
  --is_resume False
```

训练完成后，如果要用 Hugging Face `AutoModelForVision2Seq` 推理，需要转换格式。

## 8. 从头训练方法

从头训练同样走 `vla-scripts/train.py`，但不传 `pretrained_checkpoint`，而是从 `vla.py` 配置里的 `base_vlm` 加载基础 VLM。

OpenVLA 官方 `openvla-7b` 是从 Prismatic VLM 出发训练的：

```text
base_vlm: prism-dinosiglip-224px+7b
数据: Open-X Magic Soup++ / OXE 大规模混合数据
规模: 约 970K trajectories
```

典型配置在：

```text
openvla/prismatic/conf/vla.py
```

例如：

```text
prism-dinosiglip-224px+mx-bridge
prism-dinosiglip-224px+mx-oxe-magic-soup-plus
```

从头训练是资源最重的路线，适合复现实验或大规模新模型训练，不适合当前只想让自定义 Franka 仿真抓红方块的短期目标。

## 9. 冻结策略与训练阶段

`train.py` 根据 `VLAConfig` 中的三个开关决定训练阶段：

```text
freeze_vision_backbone
freeze_llm_backbone
unfreeze_last_llm_layer
```

对应阶段：

```text
vla-full-train
  视觉 backbone、投影层、LLM 都训练

vla-train
  冻结视觉 backbone，训练投影层和 LLM

vla-sandwich-train
  训练视觉端、投影层和 LLM 最后一层

vla-last-layer-train
  冻结视觉和 LLM 主体，只训练 LLM 最后一层
```

选择原则：

```text
数据少、怕过拟合: LoRA 或 last-layer
相机差异明显: 至少考虑图像增强，必要时动视觉端
动作分布差异大: LoRA 不够时再考虑全量微调
资源充足且数据大: full-train / FSDP
```

## 10. 自定义数据集如何接入

推荐路径是把数据转成 RLDS 格式，再接入 OpenVLA 数据管道。

需要补的注册位置：

```text
prismatic/vla/datasets/rlds/oxe/configs.py
prismatic/vla/datasets/rlds/oxe/transforms.py
prismatic/vla/datasets/rlds/oxe/mixtures.py
prismatic/conf/vla.py
```

最小需要定义：

```text
数据集名称
图像字段映射
语言指令字段映射
动作字段映射
动作归一化统计
训练 mixture
VLAConfig 实验配置
```

如果不是 RLDS，也可以参考 `finetune.py` 中注释的 `DummyDataset` / 自定义 PyTorch Dataset 路线，但官方更推荐 RLDS，因为预处理、shuffle、统计保存和多数据集混合都已经打通。

## 11. 微调后的模型如何用于推理

LoRA 微调脚本保存 checkpoint 时，会保存 processor、adapter 或合并后的模型，并保存 dataset statistics。

推理时需要确保：

```text
模型 checkpoint 包含正确 norm_stats
predict_action 使用对应 unnorm_key
机器人控制器理解该 7D action 的尺度和语义
```

全量训练 / Prismatic 格式模型需要转换为 HF 格式：

```bash
cd /home/creator/ws/openVLA/openvla
source /home/creator/ws/openVLA/.venv/bin/activate

python vla-scripts/extern/convert_openvla_weights_to_hf.py \
  --openvla_model_path_or_id <PRISMATIC_RUN_DIR> \
  --output_hf_model_local_path <OUTPUT_HF_MODEL_DIR>
```

转换后才能像当前 `demo_predict_action.py` 或 `openvla_sim/openvla_policy.py` 一样用 `AutoProcessor` 和 `AutoModelForVision2Seq` 加载。

## 12. LIBERO 微调与当前 base 模型的区别

LIBERO 的官方高成功率评测不是直接用 base `openvla-7b`，而是使用 LIBERO 任务集微调后的 checkpoint：

```text
openvla/openvla-7b-finetuned-libero-spatial
openvla/openvla-7b-finetuned-libero-object
openvla/openvla-7b-finetuned-libero-goal
openvla/openvla-7b-finetuned-libero-10
```

这些 checkpoint 通常包含对应 `libero_*` 动作统计；base `openvla-7b` 的 `norm_stats` 中没有这些 LIBERO key，因此不能把 base 模型和 LIBERO 评测脚本强行混用。

## 13. 当前 RTX 5090 工作区注意事项

当前工作区环境和官方 README 有差异：

```text
当前环境: RTX 5090 / sm_120 / CUDA 12.8 / PyTorch 2.11.0+cu128
官方 README: 常以 PyTorch 2.2.0 / CUDA 12.4 / A100 环境为基准
```

因此本工作区不要直接执行会覆盖 PyTorch 的安装命令。安装源码包时优先：

```bash
cd /home/creator/ws/openVLA/openvla
source /home/creator/ws/openVLA/.venv/bin/activate
python -m pip install -e . --no-deps
```

原因：`openvla/pyproject.toml` 中固定了 `torch==2.2.0`、`torchvision==0.17.0`、`torchaudio==2.2.0`，这与 RTX 5090 的 sm_120 支持不匹配。

训练常用后端：

```text
推理: sdpa 是当前安全默认值，flash_attention_2 已验证可用但需要 CUDA 12.8 编译环境
训练: BF16 mixed precision，必要时使用 gradient checkpointing
```

## 14. 对当前 Franka 自定义仿真的启发

如果目标只是让当前 MuJoCo Franka 红方块场景“看起来完成抓取”，优先级不是从头训练，而是：

```text
1. 修正机器人控制适配层，例如 gripper actuator index、IK、抓取状态机
2. 确认相机视角、方块尺寸、动作尺度和 unnorm_key 的匹配
3. 如果 base openvla-7b + bridge_orig 行为仍不稳定，再收集当前场景数据做 LoRA 微调
4. 只有 LoRA 明显不够，才考虑全量微调或从头训练
```

对新 Franka 数据微调时，最关键的数据闭环是：

```text
当前相机图像
当前任务指令
专家动作 7D action
动作统计 norm_stats
训练后的 checkpoint
对应 unnorm_key
Franka adapter / IK / gripper controller
```

也就是说，微调不是只给模型看“Franka”这个词，而是要让图像分布、动作分布和控制器一起对齐。

## 15. 推荐学习顺序

```text
1. 先理解 demo_predict_action.py 的单步推理
2. 再理解 DAMO_README.md 中 unnorm_key / norm_stats
3. 再看本文件的数据格式和 action tokenizer
4. 再读 finetune.py，优先掌握 LoRA 微调
5. 最后再看 train.py 和 vla.py 的 FSDP 全量训练配置
```

如果后续要真正训练当前 Franka 场景，建议先不要直接从头训练，而是先设计数据采集格式和专家策略，确保每条样本能稳定提供：图像、指令、7D expert action、gripper 标签和成功/失败评估。
