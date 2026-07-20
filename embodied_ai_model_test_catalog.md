# 具身智能开源模型测试项目清单（2021-2026）

本文档记录适合在当前环境中测试的具身智能开源模型、仿真平台和机器人资源，并补充代码地址、模型下载地址、机械臂/机器人资源地址与建议测试优先级。

## 1. 当前测试环境

| 项目 | 配置 |
|---|---|
| 操作系统 | Ubuntu 22.04 |
| GPU | NVIDIA RTX 5090，计算能力 sm_120 |
| Python | 3.10 |
| PyTorch | 2.11.0 cu128 |
| CUDA Toolkit | 12.8 |
| 推荐精度 | bfloat16 / float16，优先 bfloat16 |
| 注意事项 | flash-attn 等 CUDA 扩展需要源码编译，并设置 `TORCH_CUDA_ARCH_LIST="12.0"` |

## 2. 总体推荐测试顺序

| 优先级 | 项目 | 推荐原因 |
|---:|---|---|
| 1 | OpenVLA | 当前工作区已有 demo，最适合先验证 VLA 推理链路 |
| 2 | Diffusion Policy | 环境依赖相对清晰，适合机械臂模仿学习和动作生成 |
| 3 | ACT / ALOHA | 双臂模仿学习经典路线，可结合 LeRobot / ALOHA / SO-ARM 测试 |
| 4 | π0 / π0.5 / openpi | 2024-2025 VLA/flow matching 代表路线，适合进一步测试泛化策略 |
| 5 | Isaac GR00T | NVIDIA 2025-2026 人形机器人 VLA 路线，适合与 Isaac Lab / LeRobot 联动 |
| 6 | RynnBrain / RynnWorld-4D / LingBot 系列 | 2026 新一代具身基础模型、世界模型和视频/深度模型，适合重点跟踪 |
| 7 | Isaac Lab / MuJoCo Menagerie / RoboTwin | 作为仿真、评测、机器人模型资产底座 |

## 3. 按年份排序的项目清单

### 2021 年

| 项目 | 类型 | 代码地址 | 模型/数据地址 | 机械臂/仿真资源 | 测试建议 |
|---|---|---|---|---|---|
| RoboMimic | 模仿学习框架 / benchmark | https://github.com/ARISE-Initiative/robomimic | https://robomimic.github.io/ | robosuite / MuJoCo | 适合先跑 BC、BC-RNN、IQL 等离线模仿学习 baseline |
| ManiSkill | 机器人操作仿真 benchmark | https://github.com/mani-skill/ManiSkill | https://www.maniskill.ai/ | SAPIEN 仿真平台 | 适合操作技能、抓取、开柜、移动物体等任务 |

### 2022 年

| 项目 | 类型 | 代码地址 | 模型/数据地址 | 机械臂/仿真资源 | 测试建议 |
|---|---|---|---|---|---|
| R3M | 机器人视觉表征 | https://github.com/facebookresearch/r3m | https://sites.google.com/view/robot-r3m/ | Franka Panda 等真实机械臂任务 | 可作为视觉 encoder，对接 BC / RL policy |
| PerAct | 语言条件 3D 操作策略 | https://github.com/peract/peract | https://peract.github.io/ | RLBench / CoppeliaSim | 适合测试语言条件 6DoF manipulation |
| VIMA | 多模态 prompt 机器人操作 | https://github.com/vimalabs/VIMA | https://vimalabs.github.io/ | VIMA-Bench | 适合测试文字 + 图像 prompt 的操作规划 |
| MVP | Masked Visual Pretraining for Robotics | https://github.com/ir413/mvp | https://tetexiao.com/projects/real-mvp/ | 真实机械臂与视觉控制任务 | 可作为机器人视觉预训练 backbone 测试 |
| MuJoCo Menagerie | MuJoCo 高质量机器人模型库 | https://github.com/google-deepmind/mujoco_menagerie | 无单独模型权重 | Panda、ALOHA、Unitree、Shadow Hand、Allegro、Go2、G1 等 | 工作区已有 `/home/creator/ws/mujoco_menagerie`，适合加载机器人 XML 做仿真 |

### 2023 年

| 项目 | 类型 | 代码地址 | 模型/数据地址 | 机械臂/仿真资源 | 测试建议 |
|---|---|---|---|---|---|
| ManiSkill2 | 操作技能 benchmark | https://github.com/mani-skill/ManiSkill | https://maniskill2.github.io/ | SAPIEN | 比 ManiSkill 任务更丰富，适合做 policy 评测 |
| Diffusion Policy | 机器人动作扩散策略 | https://github.com/real-stanford/diffusion_policy | https://diffusion-policy.cs.columbia.edu/ | PushT、RoboMimic、robosuite、真实机械臂 | RTX 5090 上很适合测试，优先级高 |
| VC-1 | Embodied AI 视觉基础模型 | https://eai-vc.github.io/ | https://eai-vc.github.io/ | 多类 embodied AI 视觉任务 | 适合做视觉表征替换实验 |
| ALOHA / ACT | 双臂模仿学习 / Action Chunking Transformer | https://github.com/tonyzhaozh/act | https://tonyzhaozh.github.io/aloha/ | ALOHA 双臂、Trossen ViperX / WidowX | 适合遥操作数据训练、双臂插入、搬运等任务 |
| RoboFlamingo | VLM 到机器人操作策略 | https://github.com/RoboFlamingo/RoboFlamingo | https://roboflamingo.github.io/ | CALVIN / 语言条件 manipulation | 适合测试 VLM 迁移到机器人控制 |

### 2024 年

| 项目 | 类型 | 代码地址 | 模型/数据地址 | 机械臂/仿真资源 | 测试建议 |
|---|---|---|---|---|---|
| Octo | 通用机器人策略 | https://github.com/octo-models/octo | https://octo-models.github.io/ | Open X-Embodiment 多机器人数据 | 适合测试 generalist robot policy，资源需求中等 |
| OpenVLA | 开源 Vision-Language-Action 模型 | https://github.com/openvla/openvla | https://huggingface.co/openvla/openvla-7b；ModelScope 可用 `zixiaoBios/openvla-7b` | LIBERO、BridgeData、WidowX 风格动作 | 工作区已有 `/home/creator/ws/openVLA`，建议先跑 `demo_predict_action.py` |
| RDT-1B | Robotics Diffusion Transformer | https://github.com/thu-ml/RoboticsDiffusionTransformer | https://huggingface.co/robotics-diffusion-transformer/rdt-1b | 双臂 / 多机器人操作数据 | 适合测试 1B 级扩散机器人基础模型 |
| π0 / openpi | Physical Intelligence VLA / flow model | https://github.com/Physical-Intelligence/openpi | https://huggingface.co/lerobot/pi0_base | LeRobot、LIBERO、真实机器人数据 | π0 论文为 2024 路线，实际开源测试多在 2025 后更方便 |

### 2025 年

| 项目 | 类型 | 代码地址 | 模型/数据地址 | 机械臂/仿真资源 | 测试建议 |
|---|---|---|---|---|---|
| π0.5 / pi05 | Physical Intelligence VLA，强调 open-world generalization | https://github.com/Physical-Intelligence/openpi | https://huggingface.co/lerobot/pi05_base；https://huggingface.co/lerobot/pi05_libero_base | LeRobot、LIBERO、真实机械臂 | 当前非常值得测，适合和 OpenVLA、ACT 横向比较 |
| Isaac GR00T N1 | NVIDIA 人形机器人 foundation model / VLA | https://github.com/NVIDIA/Isaac-GR00T | https://huggingface.co/nvidia/GR00T-N1-2B | 人形机器人、Isaac Sim / Isaac Lab | 适合测试人形机器人 reasoning + skill policy |
| Isaac Lab | Isaac Sim 上的机器人学习框架 | https://github.com/isaac-sim/IsaacLab | 无单独模型权重 | Isaac Sim、强化学习、模仿学习、人形/四足/机械臂 | 适合做 GPU 加速仿真训练，注意 Isaac Sim 依赖较重 |
| Isaac Lab Arena | Isaac Lab 任务和 policy 评测扩展 | https://github.com/isaac-sim/IsaacLab-Arena | 无单独模型权重 | 多任务机器人评测 | 适合做大规模 policy benchmark |
| NVIDIA Cosmos | World Foundation Model 平台，面向 Physical AI | https://github.com/nvidia-cosmos | https://huggingface.co/nvidia-cosmos | 自动驾驶、机器人、仿真视频预测 | 可作为物理世界视频预测/合成数据底座 |
| RoboTwin / RoboTwin 2.0 | 双臂机器人 benchmark / 数据生成 | https://github.com/robotwin-Platform/robotwin | https://robotwin-platform.github.io/ | SAPIEN、双臂 manipulation | 适合测试双臂任务、合成数据和 domain randomization |
| LeRobot + SO-ARM100 / SO-101 | 低成本机器人学习工具链和机械臂 | https://github.com/huggingface/lerobot | https://huggingface.co/docs/lerobot | https://github.com/TheRobotStudio/SO-ARM100 | 适合实际硬件采集、ACT、Diffusion Policy、π0/π0.5 测试 |

### 2026 年

| 项目 | 类型 | 代码地址 | 模型/数据地址 | 机械臂/仿真资源 | 测试建议 |
|---|---|---|---|---|---|
| RynnBrain | 具身基础模型，空间理解、定位、规划、导航 | https://github.com/alibaba-damo-academy/RynnBrain | https://huggingface.co/collections/Alibaba-DAMO-Academy/rynnbrain；https://www.modelscope.cn/collections/DAMO_Academy/RynnBrain | AI2-THOR、导航、下游 VLA | 工作区已有 `/home/creator/ws/RynnBrain`，可跑 Gradio demo 和 AI2-THOR 验证 |
| RynnWorld-4D | 4D embodied world model，RGB + Depth + Flow | https://github.com/alibaba-damo-academy/RynnWorld-4D | https://huggingface.co/Alibaba-DAMO-Academy/RynnWorld-4D；https://www.modelscope.cn/models/DAMO_Academy/RynnWorld-4D | 双臂 manipulation、OpenPI websocket protocol | 工作区已有 `/home/creator/ws/RynnWorld-4D`，适合测试世界模型推理和 policy head |
| LingBot-Depth | Masked Depth Modeling，深度补全与几何感知 | https://github.com/robbyant/lingbot-depth | https://huggingface.co/collections/robbyant/lingbot-depth；https://www.modelscope.cn/collections/Robbyant/LingBot-Depth | RGB-D 相机、点云、抓取 | 工作区已有 `/home/creator/ws/lingbot-depth`，可先跑 `example.py` |
| LingBot-Video | 面向具身智能的大规模 MoE 视频生成模型 | https://github.com/robbyant/lingbot-video | https://huggingface.co/collections/robbyant/lingbot-video；https://www.modelscope.cn/collections/Robbyant/LingBot-Video | 合成具身视频、物理合理性评测 | 工作区已有 `/home/creator/ws/lingbot-video`，MoE 版本显存/内存压力较大 |
| LingBot-World v2 | 交互式世界模型 / infinite world | https://github.com/robbyant/lingbot-world-v2 | https://huggingface.co/robbyant/lingbot-world-v2-14b-causal-fast；https://modelscope.cn/models/Robbyant/lingbot-world-v2-14b-causal-fast | 交互式世界建模、action-conditioned video | 工作区已有 `/home/creator/ws/lingbot-world-v2`，官方示例偏多 GPU |
| LingBot-VLA | 蚂蚁灵波 VLA foundation model | https://github.com/robbyant/lingbot-vla | https://huggingface.co/robbyant/lingbot-vla-4b-depth-posttrain-robotwin | 双臂机器人、RoboTwin、真实机器人数据 | 重点关注 2026 VLA 路线，适合和 OpenVLA、π0.5、GR00T 对比 |
| LingBot-VA | Video-Action / causal world modeling for robot control | https://github.com/robbyant/lingbot-va | https://huggingface.co/robbyant/lingbot-va-posttrain-libero-long | LIBERO、机器人控制 | 适合测试视频预测与动作推理统一建模 |
| Isaac GR00T N1.5 / N1.6 / N1.7 | NVIDIA 新版 GR00T 系列 | https://github.com/NVIDIA/Isaac-GR00T | https://huggingface.co/nvidia/GR00T-N1.5-3B；https://huggingface.co/nvidia/GR00T-N1.6-3B；https://huggingface.co/nvidia/GR00T-N1.7-3B | 人形机器人、Isaac Sim、LeRobot | 适合测试人形 VLA、reasoning VLA 与 LeRobot 集成 |
| SAM 3D Body | 单图/视频人体 3D 网格 | https://github.com/facebookresearch/sam-3d-body | 见官方仓库 release | 人体遥操作、数字人、全身 motion capture | 可与工作区 `/home/creator/ws/human_dexterous_teleop` 结合 |

## 4. 本地工作区已有项目入口

| 本地项目 | 路径 | 主要入口 | 说明 |
|---|---|---|---|
| OpenVLA | `/home/creator/ws/openVLA` | `demo_predict_action.py` | 最小 VLA 动作预测 demo，已记录 RTX 5090 环境说明 |
| RynnBrain | `/home/creator/ws/RynnBrain` | `demo.py`，`ai2thor_validation/main.py` | 具身理解、空间定位、规划、导航 |
| RynnWorld-4D | `/home/creator/ws/RynnWorld-4D` | `inference-sft.py`，`rynnworld4d_policy/serve_rynnworld4d_policy.py` | 4D 世界模型与双臂 policy head |
| LingBot-Depth | `/home/creator/ws/lingbot-depth` | `example.py` | 深度补全、点云、空间感知 |
| LingBot-Video | `/home/creator/ws/lingbot-video` | `scripts/inference.py` | 具身视频生成，支持 Dense / MoE |
| LingBot-World v2 | `/home/creator/ws/lingbot-world-v2` | `generate.py`，`run_fast.sh` | 交互世界模型，偏大显存/多 GPU |
| MuJoCo Menagerie | `/home/creator/ws/mujoco_menagerie` | `python -m mujoco.viewer --mjcf ...` | 高质量 MuJoCo 机器人模型库 |
| Human Dexterous Teleop | `/home/creator/ws/human_dexterous_teleop` | `hand_gesture_recognition/scripts/full_pipeline_demo.py` | 人体姿态、手势识别、Aero Hand 遥操作 |

## 5. 常用机械臂、机器人和仿真资源地址

| 资源 | 类型 | 地址 | 适合对接项目 |
|---|---|---|---|
| LIBERO | MuJoCo / robosuite 机器人任务 benchmark | https://github.com/Lifelong-Robot-Learning/LIBERO | OpenVLA、π0.5、LingBot-VA |
| robosuite | MuJoCo manipulation framework | https://github.com/ARISE-Initiative/robosuite | RoboMimic、Diffusion Policy |
| MuJoCo | 物理仿真引擎 | https://github.com/google-deepmind/mujoco | Menagerie、RoboMimic、LIBERO |
| MuJoCo Menagerie | 机器人模型资产 | https://github.com/google-deepmind/mujoco_menagerie | Panda、ALOHA、Unitree、Shadow Hand 等 |
| ALOHA | 双臂遥操作系统 | https://tonyzhaozh.github.io/aloha/ | ACT、Diffusion Policy、双臂模仿学习 |
| LeRobot | 机器人学习统一工具链 | https://github.com/huggingface/lerobot | ACT、Diffusion Policy、π0/π0.5、GR00T |
| SO-ARM100 / SO-101 | 低成本机械臂 | https://github.com/TheRobotStudio/SO-ARM100 | LeRobot、ACT、π0.5 实机测试 |
| Franka Panda / FR3 | 7DoF 机械臂 | https://franka.de/ | Diffusion Policy、RoboMimic、MuJoCo |
| Trossen WidowX / ViperX | 低成本机械臂 / ALOHA 组件 | https://www.trossenrobotics.com/ | ACT、ALOHA、OpenVLA BridgeData 风格 |
| Unitree G1 / H1 / Go2 | 人形 / 四足机器人 | https://www.unitree.com/ | GR00T、Isaac Lab、MuJoCo Menagerie |
| Shadow Hand / Allegro Hand / LEAP Hand | 灵巧手 | https://github.com/google-deepmind/mujoco_menagerie | Dexterous manipulation、遥操作 |
| Aero Hand Open | 灵巧手硬件/ROS2/SDK | 本地 `/home/creator/ws/human_dexterous_teleop/hand_gesture_recognition/aero-hand-open` | 人体手势遥操作、ROS2 控制 |

## 6. 模型下载与测试优先建议

### 6.1 最容易先跑通

| 项目 | 建议测试方式 |
|---|---|
| OpenVLA | 使用本地 `demo_predict_action.py`，先用 `--attn-implementation sdpa`，确认图像 + 指令 → 7DoF action |
| LingBot-Depth | 运行 `python example.py`，检查深度补全、点云导出 |
| MuJoCo Menagerie | 用 `python -m mujoco.viewer --mjcf mujoco_menagerie/unitree_go2/scene.xml` 验证 MuJoCo 模型加载 |
| RynnBrain | 先下载 2B/4B，再改 `demo.py` 中的 `MODEL_PATH` 跑 Gradio |

### 6.2 中等难度

| 项目 | 建议测试方式 |
|---|---|
| Diffusion Policy | 先跑 PushT 或 RoboMimic 小任务，再考虑真实机械臂数据 |
| ACT / ALOHA | 先跑官方模拟任务 Transfer Cube / Bimanual Insertion，再接 LeRobot 或 ALOHA 数据 |
| π0 / π0.5 | 通过 LeRobot policy 集成测试，优先用 `pi05_base` 或 `pi05_libero_base` |
| OpenVLA + LIBERO | 使用 OpenVLA 官方 LIBERO 微调 checkpoint 跑 `libero_spatial` 单任务 smoke test |

### 6.3 高资源或多 GPU 优先

| 项目 | 原因 |
|---|---|
| LingBot-Video-MoE | 30B-A3B + Refiner，内存和显存压力较大 |
| LingBot-World v2 14B causal-fast | 官方示例偏 8 GPU FSDP，单 RTX 5090 需降分辨率/帧数或改推理策略 |
| RynnWorld-4D | 依赖 Wan2.2-TI2V-5B、Depth-Anything-3 等组件，建议先跑 sample |
| Isaac Sim / Isaac Lab | 依赖 Omniverse/Isaac Sim 生态，安装包和显卡驱动匹配要单独确认 |
| GR00T N1.7 | 人形 VLA，建议先走 LeRobot 或官方 notebook，再接 Isaac Lab |

## 7. RTX 5090 环境注意事项

1. 不要直接照搬旧项目里的 `torch==2.0/2.1/2.2 + cu118/cu121` 命令。
2. 当前机器应优先使用 `torch==2.11.0+cu128`，确保 `torch.cuda.get_device_capability()` 返回 `(12, 0)`。
3. 编译 flash-attn、xformers、自定义 CUDA op 时，优先确认：
   - `nvcc --version` 是 CUDA 12.8；
   - `TORCH_CUDA_ARCH_LIST="12.0"`；
   - 不使用旧的预编译 wheel。
4. 若某项目强依赖旧 CUDA 扩展，优先尝试：
   - 关闭 flash-attn，改用 `sdpa` / `eager`；
   - 跳过 xformers；
   - 只跑 CPU/MuJoCo/SAPIEN 仿真部分；
   - 使用 Docker 单独隔离旧环境。

## 8. 建议的阶段性测试计划

### 阶段 A：验证基础环境

1. 验证 PyTorch / CUDA / RTX 5090：`torch.cuda.is_available()`、GPU 名称、计算能力。
2. 跑 OpenVLA 最小动作预测 demo。
3. 跑 LingBot-Depth 单张 RGB-D 示例。
4. 跑 MuJoCo Menagerie 任意机器人 viewer。

### 阶段 B：验证经典机器人策略

1. 跑 Diffusion Policy 的 PushT 或 RoboMimic 小任务。
2. 跑 ACT 官方模拟任务。
3. 尝试 LeRobot + SO-ARM100/SO-101 或 ALOHA 数据格式。

### 阶段 C：验证 VLA / 世界模型路线

1. OpenVLA + LIBERO 微调权重。
2. π0.5 + LeRobot 或 LIBERO。
3. GR00T + LeRobot / Isaac Lab。
4. LingBot-VLA / LingBot-VA 与 RoboTwin / LIBERO。
5. RynnBrain / RynnWorld-4D / LingBot-World 作为 2026 新模型重点跟踪。
