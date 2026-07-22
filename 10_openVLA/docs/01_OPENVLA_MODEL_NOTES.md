# OpenVLA 模型理解

本文记录当前工作区中 OpenVLA 模型相关的核心概念、模型路径、加载方式和推理流程。它面向学习与工程验证，不替代官方 `openvla/README.md`。

## 1. OpenVLA 是什么

OpenVLA 是 Open Vision-Language-Action 模型。它输入机器人相机图像和自然语言指令，输出机器人动作。和普通 VLM 不同，OpenVLA 的目标不是回答图像问题，而是在 prompt 后生成动作 token。

简化流程：

```text
RGB 图像 + instruction
  -> processor 构造 prompt 和图像张量
  -> VLA 模型自回归生成 action token
  -> ActionTokenizer 解码为归一化动作
  -> norm_stats / unnorm_key 反归一化
  -> 连续 7D action
```

## 2. 当前已验证的 base 模型

当前工作区已验证的 base 模型来源为：

```text
zixiaoBios/openvla-7b
```

ModelScope 默认缓存路径：

```text
/home/creator/.cache/modelscope/models/zixiaoBios--openvla-7b/snapshots/master/
```

该模型适合：

```text
单步动作预测验证
OpenVLA 加载链路验证
BridgeData / WidowX 风格 action 验证
openvla_sim 自定义 MuJoCo demo
```

不建议直接用于：

```text
LIBERO 标准闭环评测
未做控制适配的任意机器人成功率测试
```

## 3. LIBERO 微调 checkpoint

LIBERO 标准仿真评测应使用官方 LIBERO 微调模型，而不是 base `openvla-7b`。

常用 checkpoint：

```text
openvla/openvla-7b-finetuned-libero-spatial
openvla/openvla-7b-finetuned-libero-object
openvla/openvla-7b-finetuned-libero-goal
openvla/openvla-7b-finetuned-libero-10
```

原因是 LIBERO 评测脚本会把 `unnorm_key` 设置为任务集名称，例如 `libero_spatial`。base 模型通常没有这些 LIBERO 专用 `norm_stats`，会触发断言或动作尺度不匹配。

## 4. Hugging Face 加载方式

当前推理常用 Hugging Face AutoClasses：

```python
from transformers import AutoModelForVision2Seq, AutoProcessor

processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    attn_implementation="sdpa",
)
```

当前项目中的封装位置：

```text
demo_predict_action.py
openvla_sim/openvla_policy.py
repos/openvla/experiments/robot/openvla_utils.py
repos/openvla/experiments/robot/robot_utils.py
```

## 5. prompt 与 predict_action

OpenVLA 推理 prompt 通常类似：

```text
In: What action should the robot take to <instruction>?
Out:
```

推理调用：

```python
action = model.predict_action(
    **inputs,
    unnorm_key="bridge_orig",
    do_sample=False,
)
```

`do_sample=False` 表示确定性生成，适合评测和工程调试。

## 6. ActionTokenizer 与动作 token

OpenVLA 把连续动作当成语言 token 预测。训练时，7D 连续动作会被归一化并离散成 token；推理时，模型生成 token 后再解码回连续动作。

简化理解：

```text
训练:
连续 7D action -> 归一化 -> 256-bin 离散化 -> action token

推理:
action token -> 解码 -> 归一化 action -> unnorm_key 反归一化 -> 真实尺度 action
```

关键代码参考：

```text
repos/openvla/prismatic/vla/action_tokenizer.py
repos/openvla/prismatic/extern/hf/modeling_prismatic.py
```

## 7. attention 后端

当前 RTX 5090 环境推荐默认使用：

```text
sdpa
```

已验证可用但依赖 CUDA 12.8 编译环境的高性能后端：

```text
flash_attention_2
```

不建议默认使用：

```text
eager
```

原因是 OpenVLA 插入视觉 token 后，`eager` attention 路径可能出现 causal mask 长度不匹配问题。

## 8. 模型选择原则

```text
单步推理 / 自定义 MuJoCo demo -> openvla-7b base + bridge_orig
LIBERO 标准评测 -> openvla-7b-finetuned-libero-* + 对应 task_suite_name
Bridge/WidowX 实机 -> openvla-7b base 或对应实机微调模型 + bridge_orig
自定义机器人高成功率 -> 先控制适配，再考虑 LoRA / 全量微调
```

更多训练与微调细节见 [07_TRAINING_AND_FINETUNING.md](07_TRAINING_AND_FINETUNING.md)。
