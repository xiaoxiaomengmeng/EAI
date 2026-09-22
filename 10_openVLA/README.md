# OpenVLA Workspace

本目录用于学习和验证 OpenVLA：从 `openvla-7b` 单步动作预测，到自定义 MuJoCo 闭环仿真，再到 LIBERO 标准仿真评测和 Bridge/WidowX 实机评测能力边界。

## 1. 当前定位

当前工作区主要包含三条路线：

```text
sim      自定义 MuJoCo demo，支持 Franka / WX250S，适合工程链路验证
LIBERO           官方标准 MuJoCo / robosuite benchmark，适合成功率评测
Bridge/WidowX    官方真实机器人评测入口，不是离线仿真
```

核心区别：

```text
sim 跑通不等于 LIBERO 成功率
LIBERO 仿真不等于 Bridge/WidowX 实机
openvla-7b base 不应直接当作 LIBERO 微调 checkpoint 使用
```

## 2. 目录结构

```text
/home/creator/ws/10_openVLA
├── README.md                     # 本文：项目导航和常用命令
├── docs/                         # 结构化主题文档（01-09 主题 + reports/ 报告资产）
├── sim/                          # OpenVLA base + MuJoCo 闭环仿真包（Franka / WX250S）
├── scripts/
│   └── demo_predict_action.py    # OpenVLA 单步动作预测 demo
├── repos/
│   ├── openvla/                  # 官方 OpenVLA 源码（git submodule）
│   └── LIBERO/                   # LIBERO 源码（git submodule）
└── outputs/                      # 仿真产物（已 gitignore）
```

## 3. 文档导航

建议按下面顺序阅读：

| 文档 | 主题 |
|---|---|
| [01_OPENVLA_MODEL_NOTES.md](docs/01_OPENVLA_MODEL_NOTES.md) | OpenVLA 模型、base checkpoint、LIBERO 微调 checkpoint、推理流程 |
| [02_ACTION_SPACE_AND_CONTROL.md](docs/02_ACTION_SPACE_AND_CONTROL.md) | 7D action、`unnorm_key`、`bridge_orig`、action 到机器人控制 |
| [03_MUJOCO_SIM_NOTES.md](docs/03_MUJOCO_SIM_NOTES.md) | MuJoCo XML、camera/site/joint/actuator、headless 渲染、IK |
| [04_SIMULATION_PLAN.md](docs/04_SIMULATION_PLAN.md) | 单步推理、dry-run、base + MuJoCo、LIBERO、实机评测路线 |
| [05_LIBERO_EVAL_NOTES.md](docs/05_LIBERO_EVAL_NOTES.md) | LIBERO 任务套件、checkpoint 对应关系、运行命令、常见问题 |
| [06_BRIDGE_WIDOWX_NOTES.md](docs/06_BRIDGE_WIDOWX_NOTES.md) | BridgeData V2、WidowX/WX250S、实机依赖、`bridge_orig` |
| [07_TRAINING_AND_FINETUNING.md](docs/07_TRAINING_AND_FINETUNING.md) | 训练数据格式、ActionTokenizer、LoRA、全量微调 |
| [08_TERMS_AND_PARAMS.md](docs/08_TERMS_AND_PARAMS.md) | 术语和 CLI 参数速查 |
| [09_ENVIRONMENT_RTX5090.md](docs/09_ENVIRONMENT_RTX5090.md) | 当前 RTX 5090 环境、base demo 与 norm_stats 记录 |
| [sim/README.md](sim/README.md) | 自定义 MuJoCo 仿真模块详细说明 |

## 4. 环境前提

激活虚拟环境：

```bash
cd /home/creator/ws/10_openVLA
source .venv/bin/activate
```

无显示器或服务器环境运行 MuJoCo 时，建议设置：

```bash
export MUJOCO_GL=egl
```

LIBERO 评测还需要：

```bash
export LIBERO_CONFIG_PATH=/home/creator/ws/10_openVLA/.libero
```

当前工作区环境与官方 README 常见环境不同：

```text
当前环境: RTX 5090 / sm_120 / CUDA 12.8 / PyTorch 2.11.0+cu128
官方 README: 常以 PyTorch 2.2.0 / CUDA 12.4 / A100 为基准
```

安装 `openvla` 源码包时优先使用：

```bash
cd /home/creator/ws/10_openVLA/repos/openvla
source /home/creator/ws/10_openVLA/.venv/bin/activate
python -m pip install -e . --no-deps
```

避免自动依赖覆盖当前已验证的 PyTorch / CUDA 环境。

## 5. 最小单步推理

用于验证 `openvla-7b` base 模型加载、processor、prompt、GPU 推理和 action 解码。

```bash
cd /home/creator/ws/10_openVLA
source .venv/bin/activate
python scripts/demo_predict_action.py
```

显式使用当前推荐后端：

```bash
python scripts/demo_predict_action.py --attn-implementation sdpa
```

使用真实图片和指令：

```bash
python scripts/demo_predict_action.py \
  --image /path/to/image.jpg \
  --instruction "pick up the object" \
  --unnorm-key bridge_orig \
  --attn-implementation sdpa
```

## 6. 自定义 MuJoCo dry-run

不加载 7B 模型，只验证 MuJoCo、渲染、控制、视频和 JSON 输出。

```bash
cd /home/creator/ws/10_openVLA
source .venv/bin/activate
export MUJOCO_GL=egl

python -m sim \
  --dry-run \
  --robot wx250s \
  --policy-steps 2 \
  --sim-steps-per-policy 2
```

Franka：

```bash
python -m sim \
  --dry-run \
  --robot franka \
  --policy-steps 2 \
  --sim-steps-per-policy 2
```

## 7. openvla-7b base + MuJoCo

使用 `openvla-7b` base 和 `bridge_orig` 跑自定义 MuJoCo 闭环。

WX250S：

```bash
cd /home/creator/ws/10_openVLA
source .venv/bin/activate
export MUJOCO_GL=egl

python -m sim \
  --robot wx250s \
  --motion-mode ik \
  --policy-steps 20 \
  --sim-steps-per-policy 30 \
  --unnorm-key bridge_orig \
  --attn-implementation sdpa
```

Franka：

```bash
python -m sim \
  --robot franka \
  --motion-mode ik \
  --policy-steps 20 \
  --sim-steps-per-policy 30 \
  --unnorm-key bridge_orig \
  --attn-implementation sdpa
```

`motion_mode` 可选：

```text
ik      ee_site + MuJoCo Jacobian IK，默认推荐
demo    演示友好映射，机械臂运动更明显
direct  action[:6] 直接映射为关节增量，用于对照
```

当前仿真命令统一使用 `--motion-mode`。

## 8. LIBERO smoke test

LIBERO 是标准 MuJoCo / robosuite benchmark。不要直接用 `openvla-7b` base 跑标准 LIBERO，推荐使用 LIBERO 微调 checkpoint。

```bash
cd /home/creator/ws/10_openVLA/repos/openvla
source /home/creator/ws/10_openVLA/.venv/bin/activate
export LIBERO_CONFIG_PATH=/home/creator/ws/10_openVLA/.libero
export MUJOCO_GL=egl

python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --center_crop True \
  --num_trials_per_task 1 \
  --use_wandb False \
  --run_id_note smoke-test
```

其他常用对应关系：

```text
libero_spatial -> openvla/openvla-7b-finetuned-libero-spatial
libero_object  -> openvla/openvla-7b-finetuned-libero-object
libero_goal    -> openvla/openvla-7b-finetuned-libero-goal
libero_10      -> openvla/openvla-7b-finetuned-libero-10
```

详细说明见 [05_LIBERO_EVAL_NOTES.md](docs/05_LIBERO_EVAL_NOTES.md)。

## 9. Bridge/WidowX 实机评测

官方入口：

```text
repos/openvla/experiments/robot/bridge/run_bridgev2_eval.py
```

该脚本面向真实 Bridge V2 / WidowX 环境，需要：

```text
widowx_envs
WidowXClient 服务
相机 topic
host_ip / port
真实或兼容机器人控制端
```

它不是本地离线仿真，也不能仅靠 `repos/openvla/` 目录完整跑通。详细说明见 [06_BRIDGE_WIDOWX_NOTES.md](docs/06_BRIDGE_WIDOWX_NOTES.md)。

## 10. 输出位置

`sim` 默认输出：

```text
/home/creator/ws/10_openVLA/outputs/<robot>_openvla/
```

通常包含：

```text
<robot>_openvla_<timestamp>.mp4
<robot>_openvla_<timestamp>.json
```

JSON 重点字段：

```text
timing
records[*].timing
action
arm_delta
ik_info
target_ctrl
ee_site_pos
gripper_pos
cube_pos
gripper_cube_distance
gripper_state
```

## 11. 推荐学习顺序

```text
1. scripts/demo_predict_action.py 单步推理
2. docs/01_OPENVLA_MODEL_NOTES.md
3. docs/02_ACTION_SPACE_AND_CONTROL.md
4. sim dry-run
5. sim + openvla-7b base
6. docs/03_MUJOCO_SIM_NOTES.md
7. docs/05_LIBERO_EVAL_NOTES.md
8. docs/07_TRAINING_AND_FINETUNING.md
9. docs/06_BRIDGE_WIDOWX_NOTES.md
```



路线a:
cd /home/creator/ws/10_openVLA && source .venv/bin/activate && export MUJOCO_GL=egl
# ① 单步推理：图像+指令 → 7D action
python scripts/demo_predict_action.py --attn-implementation sdpa
# ② sim dry-run：验证 MuJoCo 链路（不加载模型）
python -m sim --dry-run --robot franka --policy-steps 2 --sim-steps-per-policy 2
# ③ base+MuJoCo 闭环：真实模型驱动仿真
python -m sim --robot franka --motion-mode ik --policy-steps 20 --sim-steps-per-policy 30 --unnorm-key bridge_orig --attn-implementation sdpa

路线b：

cd /home/creator/ws/10_openVLA && source .venv/bin/activate
# ① 重装失效的 editable 包（--no-deps 保护已验证的 torch/cuda）
(cd repos/LIBERO  && pip install -e . --no-deps)
(cd repos/openvla && pip install -e . --no-deps)
python -c "import libero, prismatic; print('import OK')"
# ② 下载微调 checkpoint（~14GB，需 HF 网络）：openvla/openvla-7b-finetuned-libero-spatial
# ③ smoke test
export LIBERO_CONFIG_PATH=/home/creator/ws/10_openVLA/.libero MUJOCO_GL=egl
cd repos/openvla && python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial --center_crop True --num_trials_per_task 1 --use_wandb False

  modelscope download --model superpeach/openvla-7b-finetuned-libero-spatial