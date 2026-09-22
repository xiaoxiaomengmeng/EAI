# OpenVLA 术语与参数速查

本文提供 OpenVLA、MuJoCo、LIBERO、Bridge/WidowX 和当前 `sim` 常用术语与参数速查。

## 1. 模型术语

```text
OpenVLA = Open Vision-Language-Action
```

输入图像和自然语言指令，输出机器人动作的 VLA 模型。

```text
VLA = Vision-Language-Action
```

视觉、语言和动作联合建模的机器人策略模型。

```text
VLM = Vision-Language Model
```

视觉语言模型，通常用于图像理解、问答或描述。OpenVLA 在 VLM 基础上继续训练，让它生成动作。

```text
LLM = Large Language Model
```

大语言模型。OpenVLA 使用 LLM 的自回归 token 生成能力来生成 action token。

```text
Prismatic
```

OpenVLA 依赖的 VLM / VLA 训练框架。全量训练和从头训练主要使用 Prismatic 栈。

```text
HF = Hugging Face
```

当前推理常用 `AutoProcessor` 和 `AutoModelForVision2Seq` 加载 HF 格式 checkpoint。

## 2. 数据集术语

```text
RLDS = Reinforcement Learning Datasets
```

机器人轨迹数据格式，用 episode、step、observation、action、reward 等结构存储数据。

```text
TFDS = TensorFlow Datasets
```

TensorFlow 数据集加载格式。OpenVLA 的 RLDS 数据通常通过 TFDS 管道读取。

```text
OXE = Open-X Embodiment
```

多机器人、多任务的大规模机器人数据集合。`openvla-7b` 预训练使用了大规模 OXE mixture。

```text
BridgeData V2
```

Berkeley / Bridge 系列 WidowX 真实机器人操作数据集。`bridge_orig` 来自该数据集。

```text
LIBERO
```

基于 MuJoCo / robosuite 的机器人操作 benchmark。LIBERO 高成功率评测需要 LIBERO 微调 checkpoint。

## 3. 动作术语

```text
7DoF = 7 Degrees of Freedom
```

OpenVLA 常见 7D 末端动作：

```text
[dx, dy, dz, droll, dpitch, dyaw, gripper]
```

```text
EEF = End Effector
```

末端执行器，例如机械臂夹爪。

```text
action token
```

把连续动作离散化后的 token。OpenVLA 像语言模型生成文字一样生成动作 token。

```text
ActionTokenizer
```

把连续动作映射到 tokenizer 词表末尾动作 token 的组件，也能将动作 token 解码回连续动作。

```text
norm_stats
```

动作归一化统计，例如每个动作维度的分位数或范围。

```text
unnorm_key
```

选择哪套 `norm_stats` 做反归一化的 key。

```text
bridge_orig
```

BridgeData V2 原始版动作统计 key，可理解为 `Original Version of Bridge V2 from Project Website`。

## 4. 仿真术语

```text
MuJoCo
```

物理仿真引擎，用于机器人动力学、渲染和控制。

```text
robosuite
```

机器人仿真任务框架，LIBERO 底层依赖它。

```text
mujoco_menagerie
```

MuJoCo 官方/社区机器人模型集合，当前使用 Franka 和 WX250S 模型。

```text
body
```

MuJoCo XML 中的刚体节点。

```text
joint
```

关节自由度。

```text
actuator
```

控制器，例如 position actuator。

```text
site
```

标记点。当前 `ee_site` 用于表示末端位置并计算 IK。

```text
camera
```

仿真相机，Renderer 根据相机生成 RGB 图像。

```text
mj_jacSite
```

MuJoCo API，用于计算 site 的雅可比矩阵。

```text
MUJOCO_GL=egl
```

无显示器或服务器环境中进行 headless 渲染时常用的环境变量。

## 5. 机器人术语

```text
Franka / Panda
```

Franka Emika Panda 机械臂，常用于仿真和机器人学习任务。

```text
WidowX
```

BridgeData / OpenVLA 实机评测中常见的真实机器人平台。

```text
WX250S
```

Trossen WX250S 机械臂。当前 `sim` 支持该平台，用于接近 WidowX 风格的本地 MuJoCo 验证。

```text
robot adapter
```

把模型输出动作转换为具体机器人控制命令的适配层。

```text
motion_mode
```

当前 `sim` 中的运动模式抽象，支持 `ik`、`demo`、`direct`。

## 6. 训练术语

```text
LoRA = Low-Rank Adaptation
```

低秩适配微调方法，只训练少量 adapter 参数。

```text
PEFT = Parameter-Efficient Fine-Tuning
```

参数高效微调，LoRA 是其中一种。

```text
FSDP = Fully Sharded Data Parallel
```

PyTorch 分布式训练策略，适合大模型全量训练。

```text
DDP = Distributed Data Parallel
```

常见多 GPU 分布式训练方式。

```text
BF16 = Brain Floating Point 16
```

OpenVLA 推理和训练常用的 16 位浮点格式。

```text
checkpoint
```

模型权重或训练状态保存结果。

```text
adapter
```

在 LoRA 语境下，指新增的低秩参数模块。

## 7. 常用 CLI 参数

`scripts/demo_predict_action.py`：

```text
--model-path              本地模型目录或 Hugging Face repo id
--image                   输入 RGB 图片，不传则使用占位图
--instruction             自然语言任务指令
--unnorm-key              动作反归一化 key，base 常用 bridge_orig
--attn-implementation     eager / sdpa / flash_attention_2
```

`sim`：

```text
--robot                   franka / wx250s
--motion-mode             ik / demo / direct
--policy-steps            OpenVLA policy 查询次数
--sim-steps-per-policy    每次 policy action 后的 MuJoCo step 数
--unnorm-key              默认 bridge_orig
--dry-run                 不加载模型，使用确定性测试动作
--cube-x / --cube-y       调整红方块位置
--gripper-close-threshold 夹爪允许闭合的距离阈值
--grip-down-offset        夹爪闭合后的下压偏移
```

LIBERO：

```text
--model_family            openvla
--pretrained_checkpoint   LIBERO 微调 checkpoint
--task_suite_name         libero_spatial / libero_object / libero_goal / libero_10 / libero_90
--center_crop             LIBERO 微调模型建议 True
--num_trials_per_task     每个任务 rollout 数，smoke test 建议 1
--use_wandb               本地测试建议 False
```

## 8. 常见混淆

```text
checkpoint vs norm_stats
```

checkpoint 是模型权重，`norm_stats` 是动作统计。模型能加载不代表动作尺度一定正确。

```text
prompt 里的机器人名字 vs robot adapter
```

prompt 主要描述任务目标，真正让机器人动起来的是 `unnorm_key`、adapter、IK 和 controller。

```text
sim vs LIBERO
```

`sim` 是自定义 MuJoCo demo；LIBERO 是标准 benchmark。

```text
Bridge/WidowX vs MuJoCo 仿真
```

官方 Bridge/WidowX 脚本是实机评测，不是离线仿真。

```text
policy_steps vs sim_steps_per_policy
```

`policy_steps` 是最多查询模型多少次，`sim_steps_per_policy` 是每次动作后 MuJoCo 推进多少步。
