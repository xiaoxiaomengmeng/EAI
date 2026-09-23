# OpenVLA Workspace

本目录用于**按阶梯循序验证 OpenVLA**：先做 `openvla-7b` 单步动作预测，再到 LIBERO 标准成功率评测，
最后到基于 Franka Panda 的自定义 MuJoCo 闭环采集。每一阶都有明确的
「目的 / 命令 / 推理使用的仿真场景 / 验证判据」，前一阶跑通才进入下一阶。

- OpenVLA：<https://github.com/openvla/openvla>
- LIBERO：<https://github.com/Lifelong-Robot-Learning/LIBERO>

两个概念不能混淆：

```text
LIBERO 成功率        ≠  sim 自定义闭环跑通
openvla-7b base      ≠  LIBERO 微调 checkpoint（不可混用）
```

## 1. 验证阶梯总览

学习与验证按下面三阶推进（均已跑通 ✅），全部使用 Franka Panda / LIBERO 对齐的机器人平台。

| 阶段 | 脚本 / 入口 | 目的 | 推理使用的仿真场景 | 状态 |
|---|---|---|---|---|
| ① 单步推理 | `scripts/01_openvla_verify.py` | 证明 repos/openvla 源码 + checkpoint 能加载并出合法 7D action | 无真实仿真：默认 224×224 灰色占位图（或 `--image` 传真实图） | ✅ |
| ② LIBERO 官方评测 | `scripts/02_libero_eval.py` | 标准 benchmark 成功率评测（微调 checkpoint） | LIBERO `libero_spatial` robosuite 桌面场景，`agentview` 256×256 | ✅ |
| ③ 自定义 MuJoCo 闭环 | `python -m sim` | 采集「模型→action→IK→MuJoCo」闭环 rollout（Franka Panda） | 自定义 `tabletop_scene.xml`：桌面 + 红方块，`agentview` 相机 | ✅ |

> ①→②：先证明模型能出合法 action，再放进 LIBERO 标准场景测成功率；
> ②→③：从官方标准 benchmark 迁到自定义 Franka Panda 工程链路，采集闭环视频与 action 数据。

## 2. 目录结构

```text
/home/creator/ws/10_openVLA
├── README.md                     # 本文：验证阶梯导航与常用命令
├── docs/                         # 结构化主题文档（01-09 主题 + reports/ 报告资产）
├── sim/                          # OpenVLA base + MuJoCo 闭环仿真包（Franka Panda）
├── scripts/
│   ├── demo_predict_action.py    # OpenVLA 单步动作预测 demo（裸 transformers + trust_remote_code）
│   ├── 01_openvla_verify.py      # 仅依赖 repos/openvla 源码包的单步验证（不依赖 sim）✅ 已验证
│   └── 02_libero_eval.py         # LIBERO 官方评测薄封装（复用 run_libero_eval）✅ 已验证
├── repos/
│   ├── openvla/                  # 官方 OpenVLA 源码（git submodule）
│   └── LIBERO/                   # LIBERO 源码（git submodule）
└── outputs/                      # sim 仿真产物（已 gitignore）
```

## 3. 文档导航

建议按下面顺序阅读：

| 文档 | 主题 |
|---|---|
| [01_OPENVLA_MODEL_NOTES.md](docs/01_OPENVLA_MODEL_NOTES.md) | OpenVLA 模型、base checkpoint、LIBERO 微调 checkpoint、推理流程 |
| [02_ACTION_SPACE_AND_CONTROL.md](docs/02_ACTION_SPACE_AND_CONTROL.md) | 7D action、`unnorm_key`、`bridge_orig`、action 到机器人控制 |
| [03_MUJOCO_SIM_NOTES.md](docs/03_MUJOCO_SIM_NOTES.md) | MuJoCo XML、camera/site/joint/actuator、headless 渲染、IK |
| [04_SIMULATION_PLAN.md](docs/04_SIMULATION_PLAN.md) | 单步推理、dry-run、base + MuJoCo、LIBERO 路线 |
| [05_LIBERO_EVAL_NOTES.md](docs/05_LIBERO_EVAL_NOTES.md) | LIBERO 任务套件、checkpoint 对应关系、运行命令、常见问题 |
| [07_TRAINING_AND_FINETUNING.md](docs/07_TRAINING_AND_FINETUNING.md) | 训练数据格式、ActionTokenizer、LoRA、全量微调 |
| [08_TERMS_AND_PARAMS.md](docs/08_TERMS_AND_PARAMS.md) | 术语和 CLI 参数速查 |
| [09_ENVIRONMENT_RTX5090.md](docs/09_ENVIRONMENT_RTX5090.md) | 当前 RTX 5090 环境、base demo 与 norm_stats 记录 |
| [reports/libero_eval/README.md](docs/reports/libero_eval/README.md) | LIBERO 官方评测记录（结果 90%、兼容修复、可视化回放） |
| [sim/README.md](sim/README.md) | 自定义 MuJoCo 仿真模块详细说明 |

## 4. 环境前提

激活虚拟环境：

```bash
cd /home/creator/ws/10_openVLA
source .venv/bin/activate
```

无显示器 / 服务器环境跑 MuJoCo 时（必须在 `import mujoco` 之前设置）：

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

安装 `openvla` / `LIBERO` 源码包时优先用 `--no-deps`，避免自动依赖覆盖已验证的 PyTorch / CUDA：

```bash
cd /home/creator/ws/10_openVLA/repos/openvla && pip install -e . --no-deps
cd /home/creator/ws/10_openVLA/repos/LIBERO  && pip install -e . --no-deps
python -c "import libero, prismatic; print('import OK')"
```

模型权重统一经工作区 `modules/` 软链引用（去 org 前缀的仓库名）：

```text
modules/openvla-7b                          → base 模型（unnorm_key=bridge_orig）
modules/openvla-7b-finetuned-libero-spatial → LIBERO 微调（unnorm_key=libero_spatial）
```

## 5. 仿真环境说明

阶段②③用到**两套彼此独立的 MuJoCo 仿真环境**，定位不同，不要混淆：

| | LIBERO（阶段②） | 自定义 `sim` 包（阶段③） |
|---|---|---|
| 引擎 | robosuite 1.4.1 + MuJoCo 3.10 | MuJoCo 3.10 原生 API |
| 机器人 | Franka Panda（robosuite） | Franka Panda |
| 模型 | LIBERO 微调 + `libero_spatial` | `openvla-7b` base + `bridge_orig` |
| 场景 | `libero_spatial` 桌面（黑碗+盘/酥皮模/饼干盒/木柜/灶台） | `tabletop_scene.xml`（桌面 + 红方块） |
| 推理输入相机 | `agentview` 离屏 256×256 | `agentview`（另有 overview/wristview） |
| 控制 | robosuite OSC（官方内置） | IK / demo / direct（见 §8） |
| 用途 | 标准 benchmark **成功率评测** | 采集闭环 rollout，**不代表成功率** |
| 入口 | `scripts/02_libero_eval.py` | `python -m sim` |

关键版本（同一个 `.venv`，多框架共存）：

```text
mujoco 3.10.0   robosuite 1.4.1   torch 2.11.0+cu128   gym 0.26.2
```

**渲染**：无显示器用 `MUJOCO_GL=egl`（离屏，须在 import mujoco 前设置）；两套环境都把送入模型的
`agentview` 画面存成 rollout MP4。有显示屏时在桌面终端用 `ffplay/mpv` 回放这些 MP4，详见
[docs/reports/libero_eval/README.md](docs/reports/libero_eval/README.md)。

**版本兼容坑（阶段②已由 `02_libero_eval.py` 自动处理，无需手改子模块）**：

```text
1. libero_source_path.pth 残留旧路径 → import libero 失败（改指 repos/LIBERO）
2. wandb / protobuf 版本不匹配        → 关闭 wandb 时注入空 stub 跳过顶层 import
3. PyTorch≥2.6 torch.load 默认 weights_only=True → LIBERO init_states 含 numpy 对象，改回 False
4. robosuite 1.4.x × mujoco≥3 的 mj_fullM 签名变更 → 注入 numpy 兼容 shim
```

> 注意：`mujoco 3.10` 被 `mink`（sim 包 IK）与 `robosuite` 共同依赖，**不可降级**，因此阶段②选择在
> 封装层做兼容 shim，而非降 mujoco 版本。

## 6. 阶段① · 最小单步推理（repos/openvla 源码验证）

**推理使用的仿真场景**：本阶不涉及真实仿真环境——默认用 224×224 灰色占位图，只验证
「图像 + 指令 → 7D action」链路能否跑通；要用真实画面可用 `--image` 传入照片或某帧渲染图。

两个脚本都做同一件事，区别只在依赖来源：

```text
demo_predict_action.py   裸 transformers + trust_remote_code，默认路径来自 sim.config
01_openvla_verify.py     只依赖 repos/openvla 的 prismatic 源码包，不依赖 sim（推荐）
```

### 6.1 demo_predict_action.py

验证 `openvla-7b` base 的加载、processor、prompt、GPU 推理与 action 解码：

```bash
python scripts/demo_predict_action.py --attn-implementation sdpa
# 真实图片 + 指令：
python scripts/demo_predict_action.py \
  --image /path/to/image.jpg --instruction "pick up the object" \
  --unnorm-key bridge_orig --attn-implementation sdpa
```

### 6.2 01_openvla_verify.py（仅用 repos/openvla 源码，推荐）

刻意只用 `repos/openvla` 的 `prismatic` 源码包：把 `repos/openvla` 注入 `sys.path`，
**无需先 `pip install -e .`** 即可运行；用于确认源码本身能加载 base 与 LIBERO 微调 checkpoint 并出 action。

```bash
# base 模型（默认，unnorm_key=bridge_orig）
python scripts/01_openvla_verify.py
# LIBERO 微调 checkpoint（unnorm_key=libero_spatial）
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

> OpenVLA 是**末端 EE（笛卡尔空间）位姿增量控制**，不是关节角控制。要驱动具体机械臂，必须把前 6 维
> EE 增量经 **IK 解算**转成关节命令（第 7 维夹爪单独控制）——这正是阶段③ `sim` 包 `--motion-mode ik` 做的事。

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

**实测参考（RTX 5090）**

```text
base           加载 ~2.1s，推理 ~0.47s，占位图 action 幅度极小、gripper=0.0
libero-spatial 加载 ~2.5s，推理 ~0.47s，action 幅度明显、gripper≈1.0（符合微调分布）
```

### 6.3 坑总结

```text
坑 1（最关键）：ModelScope 微调 checkpoint 目录内无 .py，且 auto_map 指向跨仓库远程代码
  "openvla/openvla-7b--modeling_prismatic.*"。trust_remote_code=True 会去 huggingface.co 拉远程代码，
  国内网络下卡死在"加载 processor"。解法：先用 prismatic 源码注册 HF Auto 类，再以
  trust_remote_code=False + local_files_only=True 加载 → 走本地源码、完全不联网。
坑 2：attn-implementation 用 eager 触发 causal-mask 长度不匹配，统一用 sdpa。
坑 3：import prismatic 顺带加载 tensorflow（约 6s + TF/CUDA 注册日志），正常现象。
坑 4：unnorm_key 必须与 checkpoint 匹配（base=bridge_orig，微调=libero_spatial），用错会得到错误分布。
坑 5：占位灰图只验证链路能出合法 action，不代表任务语义正确；成功率评测须走阶段②。
```

## 7. 阶段② · LIBERO 官方评测（scripts/02_libero_eval.py，✅ 已跑通）

**推理使用的仿真场景**：LIBERO `libero_spatial` 的 robosuite 桌面场景——固定一张桌面，含黑碗、
盘子、酥皮模、饼干盒、木柜（含抽屉）、灶台等参照物；每步从 `OffScreenRenderEnv` 取 `agentview`
相机 256×256 图像喂给模型（也即 rollout MP4 的画面）。10 个任务都是"把黑碗从某空间位置放到盘子上"，
专门考察空间定位与指代理解。

LIBERO 是标准 MuJoCo / robosuite benchmark，需用 LIBERO 微调 checkpoint（勿用 base）。
**推荐用薄封装 `scripts/02_libero_eval.py`**：它不重写评测循环，只复用官方 `run_libero_eval.py`
的 `eval_libero`，额外自动处理了 §5 的 4 个兼容坑，checkpoint 默认指向本地软链 `modules/`，**全程不联网**。

```bash
cd /home/creator/ws/10_openVLA && source .venv/bin/activate

# smoke test：libero_spatial 全 10 任务×1 trial（checkpoint 自动取 modules/ 软链）
python scripts/02_libero_eval.py --num-trials-per-task 1 --run-id-note smoke-test

# 正式评测：每任务 50 trials，贴近官方口径
python scripts/02_libero_eval.py --num-trials-per-task 50
```

实测结果（RTX 5090，libero_spatial 10 任务×1 trial）：**成功率 9/10 = 90%**，与官方水平量级一致。
评测日志、兼容修复清单、可视化回放方法已归档至 [docs/reports/libero_eval/](docs/reports/libero_eval/README.md)。

> 当前 checkpoint 为 `finetuned-libero-spatial`，其 `dataset_statistics.json` 仅含 `libero_spatial`
> 一个 `unnorm_key`，而脚本自动令 `unnorm_key = task_suite_name`——因此**现有权重只能测
> `libero_spatial`**；其余套件需先下载对应微调权重（套件与 checkpoint 必须一一对应）：

```text
libero_spatial -> modules/openvla-7b-finetuned-libero-spatial（已下载）
libero_object  -> modules/openvla-7b-finetuned-libero-object
libero_goal    -> modules/openvla-7b-finetuned-libero-goal
libero_10      -> modules/openvla-7b-finetuned-libero-10
```

详细说明见 [05_LIBERO_EVAL_NOTES.md](docs/05_LIBERO_EVAL_NOTES.md)。

## 8. 阶段③ · 自定义 MuJoCo 闭环采集（sim 包）

**推理使用的仿真场景**：自定义 `sim/scenes/tabletop_scene.xml`——一张桌面加一个红方块 `red_cube`，
机器人为 Franka Panda；送入模型的是 `agentview` 相机渲染图（另有 overview 出视频、
wristview 可选调试）。用 `openvla-7b` base + `bridge_orig`，采集「模型→action→IK→MuJoCo」闭环 rollout。

不加载 7B 模型，先 **dry-run** 验证 MuJoCo、渲染、控制、视频与 JSON 输出：

```bash
export MUJOCO_GL=egl
python -m sim --dry-run --policy-steps 2 --sim-steps-per-policy 2
```

再用 `openvla-7b` base + `bridge_orig` 跑**真实模型驱动的闭环**：

```bash
python -m sim \
  --motion-mode ik \
  --policy-steps 20 \
  --sim-steps-per-policy 30 \
  --unnorm-key bridge_orig \
  --attn-implementation sdpa
```

`motion_mode` 可选（统一用 `--motion-mode`）：

```text
ik      ee_site + MuJoCo Jacobian 阻尼最小二乘 IK，默认推荐
demo    演示友好映射，机械臂运动更明显
direct  action[:6] 直接映射为关节增量，用于对照
```

模块级详细说明见 [sim/README.md](sim/README.md)。

## 9. 输出位置

`sim`（阶段③）默认输出：

```text
/home/creator/ws/10_openVLA/outputs/franka_openvla/franka_openvla_<timestamp>.mp4
/home/creator/ws/10_openVLA/outputs/franka_openvla/franka_openvla_<timestamp>.json
```

JSON 重点字段：

```text
timing / records[*].timing / action / arm_delta / ik_info / target_ctrl
ee_site_pos / gripper_pos / cube_pos / gripper_cube_distance / gripper_state
```

LIBERO（阶段②）产物：官方写入 `repos/openvla/experiments/logs/`（日志）与 `repos/openvla/rollouts/`（MP4），
精选副本已归档至 `docs/reports/libero_eval/`。

## 10. 推荐学习顺序

```text
① 单步推理    scripts/01_openvla_verify.py  →  docs/01, docs/02
② LIBERO 评测 scripts/02_libero_eval.py     →  docs/05, docs/reports/libero_eval/
③ MuJoCo 闭环 python -m sim（先 dry-run 再 base）  →  docs/03, sim/README.md
· 训练微调    docs/07（数据格式、ActionTokenizer、LoRA、全量微调）
```
