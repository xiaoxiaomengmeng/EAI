# OpenVLA 训练、数据格式与微调总结

本文整理当前工作区中 OpenVLA 训练、数据格式、动作 token 化、LoRA 微调、全量微调和当前环境注意事项。根目录旧版 `TRAINING_AND_FINETUNING.md` 保留为历史详细记录；后续推荐优先维护本文。

## 1. 一句话理解 OpenVLA 训练

OpenVLA 把机器人控制建模成“看图 + 读指令 -> 生成动作 token”：

```text
RGB 图像 + 语言指令
  -> VLM / LLM 自回归生成 7 个 action token
  -> ActionTokenizer 解码为归一化动作
  -> norm_stats / unnorm_key 反归一化
  -> 连续 7D 末端动作
```

训练时，动作被当成语言模型要预测的特殊 token；推理时，这些 token 再被还原成连续动作。

## 2. 关键代码位置

官方训练入口：

```text
repos/openvla/vla-scripts/train.py
```

作用：使用 Prismatic 训练栈和 PyTorch FSDP 训练 VLA，适合从头训练、继续训练、全量微调或不同冻结策略。

LoRA 微调入口：

```text
repos/openvla/vla-scripts/finetune.py
```

作用：基于 Hugging Face `AutoModelForVision2Seq` 加载 `openvla-7b`，使用 PEFT / LoRA 做参数高效微调。

数据加载与样本转换：

```text
repos/openvla/prismatic/vla/datasets/datasets.py
repos/openvla/prismatic/vla/materialize.py
repos/openvla/prismatic/vla/datasets/rlds/oxe/
```

动作 token 化：

```text
repos/openvla/prismatic/vla/action_tokenizer.py
```

训练配置注册：

```text
repos/openvla/prismatic/conf/vla.py
```

## 3. 训练数据格式

OpenVLA 官方训练默认使用 RLDS / TensorFlow Datasets 管道，数据来源包括 Open-X Embodiment、BridgeData V2、DROID、LIBERO 转换数据等。

单步训练样本核心字段：

```text
observation.image_primary        # 主相机 RGB 图像
task.language_instruction        # 自然语言任务指令
action                           # 7D 连续动作
dataset_name                     # 数据集名称
```

在 `RLDSBatchTransform` 中会变成：

```text
pixel_values     # 图像张量
input_ids        # 文本 prompt + action token
labels           # 只对 action token 部分计算 loss
dataset_name     # 样本来源
```

训练 prompt 大致是：

```text
Human: What action should the robot take to <language_instruction>?
Assistant: <action_tokens>
```

### 3.1 常见数据集

OpenVLA 训练和评测里常见的数据集可以按来源分成几类：

| 数据集 | 机器人/环境 | 常见用途 | 关键点 |
|---|---|---|---|
| `bridge_orig` / BridgeData V2 | WidowX / Bridge 系列真实机器人 | `openvla-7b` base 推理、Bridge/WidowX 风格动作适配 | 当前 base 模型常用 `unnorm_key=bridge_orig` |
| Open-X Embodiment / OXE mixture | 多机器人、多任务真实数据 | OpenVLA base 预训练 | 包含多个机器人数据集，每个数据集有独立动作统计 |
| LIBERO | MuJoCo / robosuite 仿真 | 标准仿真评测、LIBERO 微调 checkpoint | 需要 `libero_spatial` 等 LIBERO 专用 `unnorm_key` |
| DROID | 大规模真实机器人数据 | 微调示例、真实机器人泛化研究 | 常见于 OpenVLA 微调配置示例 |
| 自定义 MuJoCo 数据 | Franka / WX250S / 其他本地机器人 | 后续自定义场景 LoRA 微调 | 必须明确图像、指令、7D action 和 `norm_stats` |

注意：数据集名称不仅是“来源标签”，还会影响动作统计。模型推理时选择的 `unnorm_key` 必须和 checkpoint 中保存的 `norm_stats` 对齐。

### 3.2 数据格式示例

#### 示例 1：BridgeData / WidowX 风格单步样本

BridgeData / WidowX 风格样本可以抽象为：

```python
sample = {
    "observation": {
        "image_primary": "RGB image, shape=(H, W, 3)",
        "proprio": [x, y, z, roll, pitch, yaw, gripper],
    },
    "task": {
        "language_instruction": "pick up the red cube",
    },
    "action": [dx, dy, dz, droll, dpitch, dyaw, gripper],
    "dataset_name": "bridge_orig",
}
```

其中 `action` 是末端相对动作，不是直接的关节角。推理时如果使用 `bridge_orig`，模型会用 BridgeData V2 原始版动作统计把 action token 反归一化。

#### 示例 2：LIBERO 仿真单步样本

LIBERO 样本来自 MuJoCo / robosuite 仿真环境，抽象结构类似：

```python
sample = {
    "observation": {
        "image_primary": "agentview RGB image, shape=(256, 256, 3)",
        "state": [eef_x, eef_y, eef_z, eef_axis_angle, gripper_qpos],
    },
    "task": {
        "language_instruction": "put the bowl on the plate",
    },
    "action": [dx, dy, dz, droll, dpitch, dyaw, gripper],
    "dataset_name": "libero_spatial",
}
```

LIBERO 官方评测脚本会根据 `task_suite_name` 设置 `unnorm_key`，例如 `libero_spatial`。因此 LIBERO 标准评测应使用 `openvla-7b-finetuned-libero-*`，不能直接混用 base `openvla-7b`。

#### 示例 3：自定义 Franka / WX250S MuJoCo 微调样本

如果后续要把当前 `openvla_sim` 场景采成微调数据，最小样本可以设计成：

```python
sample = {
    "observation": {
        "image_primary": "policy camera RGB image, shape=(224, 224, 3)",
        "proprio": {
            "qpos": [q1, q2, q3, q4, q5, q6, q7],
            "ee_site_pos": [x, y, z],
            "gripper_state": "open",
        },
    },
    "task": {
        "language_instruction": "pick up the red cube and lift it",
    },
    "action": [dx, dy, dz, droll, dpitch, dyaw, gripper],
    "dataset_name": "franka_red_cube_sim",
}
```

这里最关键的是：专家动作最好仍统一成 OpenVLA 可学习的 7D 末端动作。如果原始控制是关节位置或 MuJoCo actuator ctrl，需要先转换成一致的动作定义，再计算该数据集自己的 `norm_stats`。

#### 示例 4：RLDS episode 结构

OpenVLA 训练管道通常不是只读单个 step，而是按 episode 存储多步轨迹。一个 episode 可以抽象为：

```python
episode = {
    "steps": [
        {
            "observation": {
                "image_primary": "RGB image at t0",
            },
            "task": {
                "language_instruction": "open the drawer",
            },
            "action": [dx, dy, dz, droll, dpitch, dyaw, gripper],
        },
        {
            "observation": {
                "image_primary": "RGB image at t1",
            },
            "task": {
                "language_instruction": "open the drawer",
            },
            "action": [dx, dy, dz, droll, dpitch, dyaw, gripper],
        },
    ]
}
```

训练时，每个 step 会被转换成 “图像 + 指令 -> action token” 的监督样本。episode 结构主要用于保留轨迹顺序、任务上下文和数据集统计。

## 4. ActionTokenizer

OpenVLA 的动作通常是 7D：

```text
[dx, dy, dz, droll, dpitch, dyaw, gripper]
```

训练前，每维动作被归一化到近似 `[-1, 1]`，再被离散成 256 个 bin，并映射到 tokenizer 词表末尾的 action token。

训练：

```text
连续 7D action
  -> clip 到 [-1, 1]
  -> 每维离散成 256 个 bin
  -> 7 个 action token
  -> 作为语言模型 labels
```

推理：

```text
模型生成 7 个 action token
  -> decode_token_ids_to_actions
  -> 得到归一化动作
  -> 根据 unnorm_key 查 norm_stats
  -> 反归一化成真实连续动作
```

## 5. norm_stats 与 unnorm_key

`norm_stats` 是每个数据集的动作统计，通常包含动作每一维的分位数或范围。

训练时：

```text
真实动作 -> 用数据集统计归一化 -> action tokenizer 离散成 token
```

推理时：

```text
action token -> 解码为归一化动作 -> 用 unnorm_key 对应统计反归一化
```

当前 base `openvla-7b` 常用：

```text
unnorm_key = bridge_orig
```

LIBERO 评测应使用对应 `libero_*` key 和微调 checkpoint。

## 6. LoRA 微调

LoRA 是当前更实用的低成本微调路线。

特点：

```text
加载 openvla-7b HF checkpoint
  -> 给线性层挂 LoRA adapter
  -> 只训练少量 LoRA 参数
  -> 保存 adapter 或合并后的模型
```

典型命令模板：

```bash
cd /home/creator/ws/openVLA/repos/openvla
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

显存不够时，优先降低 `batch_size`，再增加 `grad_accumulation_steps` 维持有效 batch size。

## 7. 全量微调

全量微调使用：

```text
repos/openvla/vla-scripts/train.py
```

它通常使用 Prismatic 兼容格式 checkpoint，例如：

```text
openvla/openvla-7b-prismatic
```

适合场景：

```text
机器人、相机、动作分布与预训练数据差异很大
LoRA 适配效果不足
有足够 GPU 资源和数据量
```

当前 RTX 5090 单卡更适合先做 LoRA 或小规模验证，不建议一开始全量训练 7B。

## 8. 自定义数据集接入

推荐把数据转成 RLDS 格式，再接入 OpenVLA 数据管道。

需要关注：

```text
数据集名称
图像字段映射
语言指令字段映射
动作字段映射
动作归一化统计
训练 mixture
VLAConfig 实验配置
```

常见注册位置：

```text
prismatic/vla/datasets/rlds/oxe/configs.py
prismatic/vla/datasets/rlds/oxe/transforms.py
prismatic/vla/datasets/rlds/oxe/mixtures.py
prismatic/conf/vla.py
```

## 9. 微调后的模型如何推理

推理时必须确保：

```text
checkpoint 包含正确 norm_stats
predict_action 使用对应 unnorm_key
机器人控制器理解该 7D action 的尺度和语义
```

Prismatic 格式模型若要用 Hugging Face AutoClasses 推理，需要转换：

```bash
cd /home/creator/ws/openVLA/repos/openvla
source /home/creator/ws/openVLA/.venv/bin/activate

python vla-scripts/extern/convert_openvla_weights_to_hf.py \
  --openvla_model_path_or_id <PRISMATIC_RUN_DIR> \
  --output_hf_model_local_path <OUTPUT_HF_MODEL_DIR>
```

## 10. 当前环境注意事项

当前工作区环境：

```text
RTX 5090 / sm_120 / CUDA 12.8 / PyTorch 2.11.0+cu128
```

官方 README 常以 PyTorch 2.2.0 / CUDA 12.4 / A100 为基准，不要直接执行会覆盖 PyTorch 的安装命令。

安装源码包时优先：

```bash
cd /home/creator/ws/openVLA/repos/openvla
source /home/creator/ws/openVLA/.venv/bin/activate
python -m pip install -e . --no-deps
```

## 11. 对当前仿真的启发

如果目标是让当前 MuJoCo Franka / WX250S 红方块场景稳定执行，优先级通常是：

```text
1. 修正控制适配层，例如 gripper actuator、IK、抓取逻辑
2. 确认相机视角、方块尺寸、动作尺度和 unnorm_key 匹配
3. 如果 base openvla-7b + bridge_orig 行为仍不稳定，再收集当前场景数据做 LoRA
4. 只有 LoRA 明显不够，才考虑全量微调或从头训练
```

微调不是只给模型看“Franka”这个词，而是要让图像分布、动作分布和控制器一起对齐。
