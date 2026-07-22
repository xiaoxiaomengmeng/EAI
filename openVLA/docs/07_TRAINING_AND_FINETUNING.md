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
openvla/vla-scripts/train.py
```

作用：使用 Prismatic 训练栈和 PyTorch FSDP 训练 VLA，适合从头训练、继续训练、全量微调或不同冻结策略。

LoRA 微调入口：

```text
openvla/vla-scripts/finetune.py
```

作用：基于 Hugging Face `AutoModelForVision2Seq` 加载 `openvla-7b`，使用 PEFT / LoRA 做参数高效微调。

数据加载与样本转换：

```text
openvla/prismatic/vla/datasets/datasets.py
openvla/prismatic/vla/materialize.py
openvla/prismatic/vla/datasets/rlds/oxe/
```

动作 token 化：

```text
openvla/prismatic/vla/action_tokenizer.py
```

训练配置注册：

```text
openvla/prismatic/conf/vla.py
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

显存不够时，优先降低 `batch_size`，再增加 `grad_accumulation_steps` 维持有效 batch size。

## 7. 全量微调

全量微调使用：

```text
openvla/vla-scripts/train.py
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
cd /home/creator/ws/openVLA/openvla
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
cd /home/creator/ws/openVLA/openvla
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
