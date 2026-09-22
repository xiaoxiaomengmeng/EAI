# OpenVLA Workspace

本目录用于学习和验证 OpenVLA：从 `openvla-7b` 单步动作预测，到自定义 MuJoCo 闭环仿真，再到 LIBERO 标准仿真评测和 Bridge/WidowX 实机评测能力边界。

## 1. 当前定位

当前工作区主要包含三条路线：

```text
sim      自定义 MuJoCo demo，支持 Franka / WX250S，适合工程链路验证
LIBERO           官方标准 MuJoCo / robosuite benchmark，适合成功率评测
Bridge/WidowX    官方真实机器人评测入口，不是离线仿真
```


https://github.com/openvla/openvla

https://github.com/Lifelong-Robot-Learning/LIBERO



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
│   ├── demo_predict_action.py    # OpenVLA 单步动作预测 demo（裸 transformers + trust_remote_code）
│   └── 01_openvla_verify.py      # 仅依赖 repos/openvla 源码包的单步验证（不依赖 sim）
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

## 5. 最小单步推理（repos/openvla 源码验证）

本节两个脚本都做同一件事：`图像 + 指令 → 7D action` 的单步推理。区别只在依赖来源：

```text
demo_predict_action.py   裸 transformers + trust_remote_code，默认路径来自 sim.config
01_openvla_verify.py     只依赖 repos/openvla 的 prismatic 源码包，不依赖 sim
```

若目的是「验证 repos/openvla 源码本身可用」，用 `01_openvla_verify.py`。

### 5.1 demo_predict_action.py

用于验证 `openvla-7b` base 模型加载、processor、prompt、GPU 推理和 action 解码。

```bash
cd /home/creator/ws/10_openVLA
source .venv/bin/activate
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

### 5.2 01_openvla_verify.py（仅用 repos/openvla 源码）

刻意只使用 `repos/openvla` 的 `prismatic` 源码包：把 `repos/openvla` 注入 `sys.path`，
**无需先 `pip install -e .`** 即可运行；用于确认源码本身能加载 base 和 LIBERO 微调 checkpoint 并出 action。

```bash
cd /home/creator/ws/10_openVLA
source .venv/bin/activate

# base 模型（默认，unnorm_key=bridge_orig）
python scripts/01_openvla_verify.py

# 已下载的 LIBERO 微调 checkpoint（ModelScope superpeach，unnorm_key=libero_spatial）
python scripts/01_openvla_verify.py --preset libero-spatial

# 真实图片 + 指令
python scripts/01_openvla_verify.py --preset libero-spatial \
  --image /path/to/image.jpg --instruction "pick up the red cube"
```

**输入 / 输出**

```text
输入：--image        RGB 图（省略则用 224x224 灰色占位图，仅验证链路）
      --instruction  英文指令，拼进 "In: What action should the robot take to {...}?\nOut:"
      --preset / --checkpoint  选模型；--unnorm-key  选反归一化 key
输出：numpy 7D action（已反归一化，物理量纲，无需再处理）
      [dx, dy, dz, droll, dpitch, dyaw, gripper]
      前 6 维：末端 EE 位姿增量（相对当前位姿；dxyz 单位米、drpy 单位弧度），由 q01/q99 反归一化
      第 7 维：夹爪命令（mask=False，保持归一化值，约 0=开 / 1=合），非增量
      脚本断言 shape==(7,)
```

> 注意：OpenVLA 是**末端 EE（笛卡尔空间）位姿增量控制**，不是关节角控制。
> 要驱动具体机械臂，必须把前 6 维 EE 增量经 **IK 解算**转成关节命令（第 7 维夹爪单独控制）。
> 本项目 `sim` 包的 `--motion-mode ik`（ee_site + MuJoCo Jacobian 阻尼最小二乘 IK）即负责这一步。

**调用链路**

```text
sys.path.insert(repos/openvla)
  → import prismatic.extern.hf.{configuration,modeling,processing}_prismatic
  → 注册到 HF Auto 类（AutoConfig / AutoImageProcessor / AutoProcessor / AutoModelForVision2Seq）
  → AutoProcessor.from_pretrained(本地路径, trust_remote_code=False, local_files_only=True)
  → AutoModelForVision2Seq.from_pretrained(...) 得到 OpenVLAForActionPrediction，放到 cuda:0
  → 读取本地 dataset_statistics.json 覆盖 model.norm_stats（微调 checkpoint 才有）
  → processor(prompt, image) → input_ids / attention_mask / pixel_values
  → model.predict_action(unnorm_key=...) → 反归一化得到 7D action
```

**关键参数**

```text
--preset base            → zixiaoBios--openvla-7b，       unnorm_key=bridge_orig
--preset libero-spatial  → superpeach--...-libero-spatial，unnorm_key=libero_spatial
--unnorm-key             必须存在于该 checkpoint 的 norm_stats，否则脚本报错并打印可用 key
--attn-implementation    默认 sdpa；eager 在 OpenVLA 生成时会触发 causal-mask 长度不匹配
trust_remote_code=False + local_files_only=True  已在脚本内固定，见下方坑 1
```

**实测参考（RTX 5090）**

```text
base           加载 ~2.1s，推理 ~0.47s，占位图 action 幅度极小、gripper=0.0
libero-spatial 加载 ~2.5s，推理 ~0.47s，action 幅度明显、gripper≈1.0（符合微调分布）
```

### 5.3 坑总结

```text
坑 1（本节最关键）：ModelScope 的 OpenVLA 微调 checkpoint 目录内没有 .py 文件，
  且 config.json / preprocessor_config.json 的 auto_map 指向跨仓库远程代码
  "openvla/openvla-7b--modeling_prismatic.OpenVLAForActionPrediction"。
  若用 trust_remote_code=True 加载，会去 huggingface.co 拉该远程代码，国内网络下卡死在
  "加载 processor"。
  解法：先用 prismatic 源码把 OpenVLA 注册到 HF Auto 类，再以
  trust_remote_code=False + local_files_only=True 加载 → 走本地源码、完全不联网。
  （base 模型 zixiaoBios 目录自带 .py、auto_map 为本地相对引用，所以不受影响。）

坑 2：attn-implementation 用 eager 会触发 causal-mask 长度不匹配，统一用 sdpa。

坑 3：import prismatic 会顺带加载 tensorflow（约 6s + 一堆 TF/CUDA 注册日志），
  属正常现象，不影响验证，可用 grep 过滤噪音。

坑 4：unnorm_key 必须与 checkpoint 匹配（base=bridge_orig，libero 微调=libero_spatial），
  用错 key 会报错或得到错误分布的 action。

坑 5：占位灰图只验证链路能出合法 action，不代表任务语义正确；
  成功率评测须走第 8 节 LIBERO smoke test。
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