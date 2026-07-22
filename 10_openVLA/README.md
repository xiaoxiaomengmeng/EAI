# OpenVLA Workspace

本目录用于学习和验证 OpenVLA：从 `openvla-7b` 单步动作预测，到自定义 MuJoCo 闭环仿真，再到 LIBERO 标准仿真评测和 Bridge/WidowX 实机评测能力边界。

## 1. 当前定位

当前工作区主要包含三条路线：

```text
openvla_sim      自定义 MuJoCo demo，支持 Franka / WX250S，适合工程链路验证
LIBERO           官方标准 MuJoCo / robosuite benchmark，适合成功率评测
Bridge/WidowX    官方真实机器人评测入口，不是离线仿真
```

核心区别：

```text
openvla_sim 跑通不等于 LIBERO 成功率
LIBERO 仿真不等于 Bridge/WidowX 实机
openvla-7b base 不应直接当作 LIBERO 微调 checkpoint 使用
```

## 2. 目录结构

```text
/home/creator/ws/openVLA
├── README.md                         # 本文：项目导航和常用命令
├── DAMO_README.md                    # RTX 5090 / base demo / flash-attn 环境记录
├── TRAINING_AND_FINETUNING.md        # 旧版训练微调详细记录，后续优先维护 docs 版本
├── demo_predict_action.py            # OpenVLA 单步动作预测 demo
├── docs/                             # 结构化主题文档
├── openvla_sim/                      # 自定义 MuJoCo 闭环仿真 demo
├── repos/
│   ├── openvla/                    # 官方 OpenVLA 源码
│   └── LIBERO/                     # LIBERO 源码，如已安装
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
| [openvla_sim/README.md](openvla_sim/README.md) | 自定义 MuJoCo 仿真模块详细说明 |
| [DAMO_README.md](DAMO_README.md) | 当前 RTX 5090 环境和 base demo 历史记录 |

## 4. 环境前提

激活虚拟环境：

```bash
cd /home/creator/ws/openVLA
source .venv/bin/activate
```

无显示器或服务器环境运行 MuJoCo 时，建议设置：

```bash
export MUJOCO_GL=egl
```

LIBERO 评测还需要：

```bash
export LIBERO_CONFIG_PATH=/home/creator/ws/openVLA/.libero
```

当前工作区环境与官方 README 常见环境不同：

```text
当前环境: RTX 5090 / sm_120 / CUDA 12.8 / PyTorch 2.11.0+cu128
官方 README: 常以 PyTorch 2.2.0 / CUDA 12.4 / A100 为基准
```

安装 `openvla` 源码包时优先使用：

```bash
cd /home/creator/ws/openVLA/repos/openvla
source /home/creator/ws/openVLA/.venv/bin/activate
python -m pip install -e . --no-deps
```

避免自动依赖覆盖当前已验证的 PyTorch / CUDA 环境。

## 5. 最小单步推理

用于验证 `openvla-7b` base 模型加载、processor、prompt、GPU 推理和 action 解码。

```bash
cd /home/creator/ws/openVLA
source .venv/bin/activate
python demo_predict_action.py
```

显式使用当前推荐后端：

```bash
python demo_predict_action.py --attn-implementation sdpa
```

使用真实图片和指令：

```bash
python demo_predict_action.py \
  --image /path/to/image.jpg \
  --instruction "pick up the object" \
  --unnorm-key bridge_orig \
  --attn-implementation sdpa
```

## 6. 自定义 MuJoCo dry-run

不加载 7B 模型，只验证 MuJoCo、渲染、控制、视频和 JSON 输出。

```bash
cd /home/creator/ws/openVLA
source .venv/bin/activate
export MUJOCO_GL=egl

python -m openvla_sim \
  --dry-run \
  --robot wx250s \
  --policy-steps 2 \
  --sim-steps-per-policy 2
```

Franka：

```bash
python -m openvla_sim \
  --dry-run \
  --robot franka \
  --policy-steps 2 \
  --sim-steps-per-policy 2
```

## 7. openvla-7b base + MuJoCo

使用 `openvla-7b` base 和 `bridge_orig` 跑自定义 MuJoCo 闭环。

WX250S：

```bash
cd /home/creator/ws/openVLA
source .venv/bin/activate
export MUJOCO_GL=egl

python -m openvla_sim \
  --robot wx250s \
  --motion-mode ik \
  --policy-steps 20 \
  --sim-steps-per-policy 30 \
  --unnorm-key bridge_orig \
  --attn-implementation sdpa
```

Franka：

```bash
python -m openvla_sim \
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
cd /home/creator/ws/openVLA/repos/openvla
source /home/creator/ws/openVLA/.venv/bin/activate
export LIBERO_CONFIG_PATH=/home/creator/ws/openVLA/.libero
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

`openvla_sim` 默认输出：

```text
/home/creator/ws/openVLA/outputs/<robot>_openvla/
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
1. demo_predict_action.py 单步推理
2. docs/01_OPENVLA_MODEL_NOTES.md
3. docs/02_ACTION_SPACE_AND_CONTROL.md
4. openvla_sim dry-run
5. openvla_sim + openvla-7b base
6. docs/03_MUJOCO_SIM_NOTES.md
7. docs/05_LIBERO_EVAL_NOTES.md
8. docs/07_TRAINING_AND_FINETUNING.md
9. docs/06_BRIDGE_WIDOWX_NOTES.md
```
