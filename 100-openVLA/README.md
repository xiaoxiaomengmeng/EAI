# OpenVLA Workspace

本目录用于**按阶梯循序验证 OpenVLA**（7B 视觉-语言-动作模型）：先做单步动作预测，再跑 LIBERO 标准
成功率评测，最后接到 Franka Panda 的自定义 MuJoCo 闭环。前一阶跑通才进入下一阶。

- OpenVLA：<https://github.com/openvla/openvla>（论文 arXiv:2406.09246）
- LIBERO：<https://github.com/Lifelong-Robot-Learning/LIBERO>

三组概念不要混淆：

```text
openvla-7b base      ≠  LIBERO 微调 checkpoint      （权重、unnorm_key、动作量纲都不同，不可混用）
末端 EE 位姿增量      ≠  关节角 / 关节增量             （OpenVLA 输出的是前者，见 §3.4）
LIBERO 成功率        ≠  sim 自定义闭环跑通           （后者只证明工程链路，不代表成功率）
```

## 1. 快速开始

```bash
cd ~/ws/100-openVLA
source .venv/bin/activate                    # Python 3.10
export LIBERO_CONFIG_PATH=$PWD/.libero       # 用项目自己的 LIBERO 配置，不用 ~/.libero
export MUJOCO_GL=egl                         # 无显示器跑 MuJoCo 时需要，须在 import mujoco 之前设置

python scripts/01_openvla_verify.py                              # ① 单步推理（约 20s）
python scripts/02_libero_eval.py --num-trials-per-task 1         # ② LIBERO 冒烟评测
python -m sim --dry-run --policy-steps 2 --sim-steps-per-policy 2  # ③ 闭环链路 dry-run（不加载模型）
```

模型权重统一从工作区的 `models/` 软链引用：

```text
models/openvla-7b                          → base 模型，unnorm_key=bridge_orig
models/openvla-7b-finetuned-libero-spatial → LIBERO 微调，unnorm_key=libero_spatial
```

## 2. 验证阶梯总览

| 阶段 | 入口 | 目的 | 仿真场景 | 状态 |
|---|---|---|---|---|
| ① 单步推理 | `scripts/01_openvla_verify.py` | 证明 repos/openvla 源码 + checkpoint 能加载，并输出合法的 7D action | 无仿真：224×224 灰色占位图，或用 `--image` 传入真实图片 | ✅ |
| ② LIBERO 官方评测 | `scripts/02_libero_eval.py` | 用标准 benchmark 测成功率（微调 checkpoint） | LIBERO `libero_spatial` robosuite 桌面，`agentview` 256×256 | ✅ 9/10 |
| ③ 自定义 MuJoCo 闭环 | `python -m sim` / `scripts/03_sim_rollout.py` | 采集「模型→action→IK→MuJoCo」闭环 rollout | 自建 `tabletop_scene.xml`：桌面 + 红方块 | ✅ 链路通 |
| ④ 自建 LIBERO 场景复刻 | `scripts/04~06_*`、`calibrate_libero_scene.py` | 不依赖 robosuite，复刻 LIBERO 观测与 OSC 控制，跑微调模型闭环 | 自建 `libero_spatial_scene.xml` | 🚧 进行中 |

> ①→②：先确认模型能输出合法 action，再放到标准场景里测成功率；
> ②→③：从官方 benchmark 迁到自建 Franka 工程链路；
> ③→④：把自建链路的观测与控制对齐到 LIBERO，让微调模型在自建场景中真正可用。

## 3. 模型介绍：输入、输出与控制方式

### 3.1 结构

```text
视觉编码  DINOv2 + SigLIP 双塔融合（dinosiglip-vit-so-224px），输入 224×224
投影      MLP projector，把视觉特征映射为 LLM token
语言模型  Llama-2 7B（llama2-7b-pure），自回归生成 7 个动作 token
动作离散  每维 256 bins，复用 Llama 词表末尾 256 个 token 作为动作 token
```

base 与 LIBERO 微调 checkpoint 结构完全一致，区别只在权重和 `norm_stats`（动作统计）。

### 3.2 输入

| 输入 | 内容 | 说明 |
|---|---|---|
| 图像 | 单张第三人称 RGB（不用腕部相机、深度或本体状态） | processor resize 到 224×224，DINOv2/SigLIP 各做一次归一化后在通道维拼接，所以 `pixel_values` 的 shape 是 `(1, 6, 224, 224)` |
| 指令 | 英文自然语言，例如 `pick up the object` | 拼进固定模板 `In: What action should the robot take to {instruction}?\nOut:` |
| `unnorm_key` | 选择哪套动作统计做反归一化 | 必须与 checkpoint 匹配：base→`bridge_orig`，微调→`libero_spatial` |

LIBERO 微调模型对图像预处理还有额外要求（官方评测与阶段④都按此复刻）：

```text
agentview 256×256 → 旋转 180°（补偿 robosuite 帧缓冲倒置）→ lanczos 缩到 224
                  → center_crop 0.9（训练时做过随机裁剪增强）→ 再缩回 224
```

**模型是单帧、无状态的**：每一步只看当前图像和指令，不看历史帧和关节状态。

### 3.3 输出与解码

`model.predict_action(...)` 一次生成 7 个 token，返回的已经是**反归一化后的 7D numpy 数组**，调用方不需要再处理：

```text
7 个动作 token
  → 离散 bin（vocab_size - token_id）
  → bin 中心值，落在 [-1, 1]                 # 归一化域
  → 按 unnorm_key 的 q01/q99 反归一化：
      前 6 维 mask=True ：0.5*(x+1)*(q99-q01)+q01
      第 7 维 mask=False：保持原值（夹爪）
  → [dx, dy, dz, droll, dpitch, dyaw, gripper]
```

### 3.4 末端 EE 控制还是关节控制？—— 末端 EE

**OpenVLA 输出的是末端执行器（EE）在笛卡尔空间的位姿增量，不是关节角，也不是关节增量。**

```text
前 6 维  末端「往哪个方向挪多少、转多少」，相对当前位姿的增量（delta），与机械臂自由度、臂长无关
第 7 维  夹爪开合的绝对命令（不是增量），1 = 张开，0 = 闭合
```

动作的具体量纲取决于训练数据，也就是 `unnorm_key`：

| | base + `bridge_orig` | LIBERO 微调 + `libero_spatial` |
|---|---|---|
| 训练数据 | BridgeData V2（WidowX 实机） | LIBERO 演示（robosuite Franka） |
| `dx,dy,dz` | 物理位移，单位米，单步约 ±0.03~0.04 | **OSC 控制器的归一化输入，范围 [-1,1]**，×0.05 才是米 |
| `droll..dyaw` | 欧拉角增量，单位弧度，单步约 ±0.08~0.2 | 轴角增量，归一化到 [-1,1]，×0.5 才是弧度 |
| gripper | 约 0~1，>0.5 表示张开 | 约 0~1；喂给 LIBERO 前先映射到 ±1 并二值化，再**取反**（LIBERO 中 -1=开、+1=合） |

由此得出的工程结论：

```text
1. 驱动任何机械臂，都要在模型和电机之间加一层「EE 增量 → 关节命令」的转换：
   用 IK 求解，或交给笛卡尔空间控制器（OSC）。第 7 维夹爪单独控制，不进 IK。
2. 换机械臂不用改 prompt，但必须保证动作约定一致：坐标系、单位、控制频率、夹爪极性。
   与训练数据差异太大时，需要在目标机器人数据上微调。
3. unnorm_key 用错，q01/q99 也会跟着用错，EE 增量的尺度就会整体错乱。
```

### 3.5 本项目中 action 如何落到执行器

| 场景 | 转换方式 | 用到的维度 |
|---|---|---|
| ② LIBERO 官方 | robosuite `OSC_POSE` 操作空间控制器（20 Hz，`control_delta=true`），由它算关节力矩 | 全部 6 维 + 夹爪 |
| ③ sim `--motion-mode ik`（默认） | `ee_site` + `mj_jacSite` 位置雅可比 + 阻尼最小二乘 IK → 7 个关节位置 ctrl | 只用 `dx,dy,dz` + 夹爪（姿态维度只记录，还没参与控制） |
| ③ sim `--motion-mode demo` | 手写线性映射，把 EE 增量混合成关节增量，让动作更明显 | 6 维 + 夹爪 |
| ③ sim `--motion-mode direct` | 把 `action[:6]` 直接当成前 6 个关节的增量 | 6 维 + 夹爪；**语义上不正确，只作对照基线** |
| ④ 自建 LIBERO 场景 | `sim/libero_controller.py`：OSC 等价控制器（复刻 osc_pose.json 的参数） | 全部 6 维 + 夹爪 |

详细推导见 [docs/02_ACTION_SPACE_AND_CONTROL.md](docs/02_ACTION_SPACE_AND_CONTROL.md)。

## 4. 阶段① · 最小单步推理

**场景**：不涉及仿真。默认使用 224×224 灰色占位图，只验证「图像 + 指令 → 7D action」链路能否跑通。

```bash
python scripts/01_openvla_verify.py                              # base，unnorm_key=bridge_orig
python scripts/01_openvla_verify.py --preset libero-spatial      # LIBERO 微调，unnorm_key=libero_spatial
python scripts/01_openvla_verify.py --preset libero-spatial \
  --image /path/to/image.jpg --instruction "pick up the red cube"
```

两个单步脚本的区别：

```text
01_openvla_verify.py     只依赖 repos/openvla 的 prismatic 源码，不依赖 sim，离线可用（推荐）
demo_predict_action.py   裸 transformers + trust_remote_code，默认路径来自 sim.config / config/*.yaml
```

**调用链路**（01_openvla_verify.py）

```text
sys.path.insert(repos/openvla)
  → import prismatic.extern.hf.{configuration,modeling,processing}_prismatic
  → 注册到 HF Auto 类（AutoConfig / AutoImageProcessor / AutoProcessor / AutoModelForVision2Seq）
  → from_pretrained(本地路径, trust_remote_code=False, local_files_only=True) → cuda:0, bf16, sdpa
  → 若存在 dataset_statistics.json，用它覆盖 model.norm_stats（只有微调 checkpoint 才有）
  → processor(prompt, image) → input_ids / attention_mask / pixel_values
  → model.predict_action(unnorm_key=...) → 7D action，脚本断言 shape==(7,)
```

**实测参考（RTX 5090）**：加载约 2~12s（取决于页缓存冷热），单步推理约 0.5~0.7s。base 在占位图上输出的
action 幅度很小；libero-spatial 输出幅度明显，gripper≈1.0（张开）。

**坑总结**

```text
坑 1（最关键）：ModelScope 下载的微调 checkpoint 目录里没有 .py，auto_map 指向跨仓库的远程代码。
  trust_remote_code=True 会去 huggingface.co 拉代码，国内网络下卡在「加载 processor」。
  解法：先用 prismatic 源码注册 HF Auto 类，再用 trust_remote_code=False + local_files_only=True 加载。
坑 2：attn-implementation 用 eager 会触发 causal-mask 长度不匹配，统一用 sdpa。
坑 3：import prismatic 会顺带加载 tensorflow（约 6s，并打印一堆 TF/CUDA 注册日志），属于正常现象。
坑 4：unnorm_key 必须与 checkpoint 匹配，用错得到的动作分布就是错的。
坑 5：占位灰图只能验证链路，不代表任务语义正确；成功率要看阶段②。
```

## 5. 阶段② · LIBERO 官方评测

**场景**：LIBERO `libero_spatial` 的 robosuite 桌面，有黑碗、盘子、酥皮模、饼干盒、木柜、灶台等物体。
10 个任务都是「把某个位置的黑碗放到盘子上」，专门考察空间定位与指代理解。

`scripts/02_libero_eval.py` 是一层薄封装：不重写评测循环，直接复用官方 `run_libero_eval.py` 的
`eval_libero`，额外自动处理 §8 的 4 个兼容坑。checkpoint 默认取 `models/` 软链，全程不联网。

```bash
python scripts/02_libero_eval.py --num-trials-per-task 1 --run-id-note smoke-test   # 冒烟：10 任务 × 1 次
python scripts/02_libero_eval.py --num-trials-per-task 50                           # 正式评测：贴近官方口径
```

实测（RTX 5090，10 任务 × 1 次）：**成功率 9/10 = 90%**，与官方水平同一量级。日志、兼容修复和回放方法见
[docs/reports/libero_eval/](docs/reports/libero_eval/README.md)。

现有权重只能测 `libero_spatial`，因为脚本令 `unnorm_key = task_suite_name`。其他套件需要先下载对应的微调权重：

```text
libero_spatial -> models/openvla-7b-finetuned-libero-spatial（已下载）
libero_object  -> models/openvla-7b-finetuned-libero-object
libero_goal    -> models/openvla-7b-finetuned-libero-goal
libero_10      -> models/openvla-7b-finetuned-libero-10
```

详细说明见 [05_LIBERO_EVAL_NOTES.md](docs/05_LIBERO_EVAL_NOTES.md)。

## 6. 阶段③ · 自定义 MuJoCo 闭环（sim 包）

**场景**：`sim/scenes/tabletop_scene.xml`，一张桌面加一个红方块 `red_cube`，机器人为 Franka Panda。
`agentview` 画面送入模型，`overview` 用来出视频，`wristview` 可选，用于调试。模型为 `openvla-7b` base + `bridge_orig`。

```bash
# 1) dry-run：不加载 7B 模型，验证 MuJoCo、渲染、控制、视频和 JSON 输出
python -m sim --dry-run --policy-steps 2 --sim-steps-per-policy 2

# 2) 真实模型驱动的闭环
python -m sim --motion-mode ik --policy-steps 20 --sim-steps-per-policy 30 \
  --unnorm-key bridge_orig --attn-implementation sdpa
```

`scripts/03_sim_rollout.py` 跑的是同一个闭环，只把策略加载换成阶段①的 repos/openvla 源码离线加载方式，
所以微调 checkpoint 也能用。`--motion-mode` 的三种模式见 §3.5。

**边界**：base 没有在该场景上微调，IK 也只控制末端位置，所以这一阶验证的是工程链路，不保证能抓起方块。
模块细节见 [sim/README.md](sim/README.md)。

## 7. 阶段④ · 自建 LIBERO 场景复刻（进行中）

目标：不依赖 robosuite/LIBERO，在自建的 `sim/scenes/libero_spatial_scene.xml` 上复刻 LIBERO 的观测预处理
（`sim/libero_obs.py`）和 OSC 控制（`sim/libero_controller.py`），让 LIBERO 微调模型闭环运行。

| 里程碑 | 脚本 | 内容 | 现状 |
|---|---|---|---|
| M1 | `verify_libero_scene.py` | 编译场景，渲染 agentview，与官方视角对比 | ✅ 有产物 |
| M2 | `04_libero_scene_single_step.py` | 观测预处理复刻 + 微调模型单步推理 | ✅ 输出范围与夹爪二值化检查通过 |
| M3 | `05_libero_scene_control_test.py` | 用合成 action 开环测试 OSC 等价控制的保真度 | ✅ `all_pass=True` |
| M4 | `calibrate_libero_scene.py` → `06_libero_scene_rollout.py` | 用官方 env 标定初始布局，再跑完整闭环 | 🚧 首轮 150 步没抓起碗，随后做了布局标定（`sim/assets/calibration/`），标定后的复测结果未记录 |

```bash
python scripts/06_libero_scene_rollout.py --policy-steps 150
```

## 8. 环境与兼容性

```text
硬件/框架  RTX 5090 (sm_120) / CUDA 12.8 / PyTorch 2.11.0+cu128（官方基准是 PyTorch 2.2 / A100）
仿真      mujoco 3.10.0 / robosuite 1.4.1 / gym 0.26.2（同一个 .venv 共存）
```

openvla、LIBERO 以 editable 方式安装。重装时必须加 `--no-deps`，避免覆盖已验证的 PyTorch/CUDA：

```bash
python -m pip install -e repos/openvla --no-deps --no-build-isolation
python -m pip install -e repos/LIBERO  --no-deps --no-build-isolation
python -c "import libero, prismatic; print('import OK')"
```

**版本兼容坑**（阶段②由 `02_libero_eval.py` 自动处理，不需要改子模块）：

```text
1. libero_source_path.pth 残留旧路径     → import libero 失败（应指向 repos/LIBERO）
2. wandb / protobuf 版本不匹配            → 关闭 wandb 时注入空 stub，跳过顶层 import
3. PyTorch≥2.6 torch.load 默认 weights_only=True → LIBERO init_states 含 numpy 对象，需改回 False
4. robosuite 1.4.x × mujoco≥3 的 mj_fullM 签名变更 → 注入 numpy 兼容 shim
```

> `mujoco 3.10` 同时被 `mink`（sim 包 IK）和 `robosuite` 依赖，**不能降级**，所以选择在封装层做 shim。

两套仿真的对比：

| | LIBERO（阶段②） | 自建 sim（阶段③④） |
|---|---|---|
| 引擎 | robosuite 1.4.1 + MuJoCo | MuJoCo 原生 API |
| 模型 | LIBERO 微调 + `libero_spatial` | ③ base + `bridge_orig`；④ LIBERO 微调 |
| 控制 | robosuite OSC_POSE | ③ IK / demo / direct；④ OSC 等价控制器 |
| 用途 | 标准**成功率评测** | 闭环工程验证，**不代表成功率** |

## 9. 目录结构与输出

```text
100-openVLA/
├── README.md / START_HERE.md
├── docs/                          # 主题文档 01-09（06 已删，编号缺口正常）+ reports/
├── config/demo_predict_action.yaml
├── scripts/
│   ├── 01_openvla_verify.py       # ① 单步（源码离线加载）
│   ├── demo_predict_action.py     # ① 单步（trust_remote_code 版）
│   ├── 02_libero_eval.py          # ② LIBERO 官方评测封装
│   ├── 03_sim_rollout.py          # ③ 闭环（源码加载 + 复用 sim）
│   ├── verify_libero_scene.py     # ④ M1
│   ├── 04_libero_scene_single_step.py   # ④ M2
│   ├── 05_libero_scene_control_test.py  # ④ M3
│   ├── calibrate_libero_scene.py  # ④ M4 前置标定
│   └── 06_libero_scene_rollout.py # ④ M4
├── sim/                           # MuJoCo 闭环包（Franka Panda），入口 python -m sim
├── repos/openvla, repos/LIBERO    # git submodule
├── .libero/config.yaml            # 项目内 LIBERO 路径配置
└── outputs/                       # 运行产物（gitignore）
```

```text
③ sim     outputs/franka_openvla/franka_openvla_<ts>.{mp4,json}
          JSON 重点字段：action / arm_delta / ik_info / target_ctrl / ee_site_pos / cube_pos / gripper_state / timing
④ 复刻    outputs/libero_scene_{verify,single_step,control_test,rollout}/
② LIBERO  repos/openvla/experiments/logs/（日志）、repos/openvla/rollouts/（MP4）；精选副本在 docs/reports/libero_eval/
```

无显示器时视频用 EGL 离屏渲染；有显示器时可以用 `ffplay` / `mpv` 回放 MP4。

## 10. 文档导航

| 文档 | 主题 |
|---|---|
| [01_OPENVLA_MODEL_NOTES.md](docs/01_OPENVLA_MODEL_NOTES.md) | 模型结构、base 与微调 checkpoint、推理流程 |
| [02_ACTION_SPACE_AND_CONTROL.md](docs/02_ACTION_SPACE_AND_CONTROL.md) | 7D action、`unnorm_key`、action 到机器人控制 |
| [03_MUJOCO_SIM_NOTES.md](docs/03_MUJOCO_SIM_NOTES.md) | MuJoCo XML、camera/site/joint/actuator、headless 渲染、IK |
| [04_SIMULATION_PLAN.md](docs/04_SIMULATION_PLAN.md) | 单步推理、dry-run、base + MuJoCo、LIBERO 路线 |
| [05_LIBERO_EVAL_NOTES.md](docs/05_LIBERO_EVAL_NOTES.md) | LIBERO 任务套件、checkpoint 对应关系、常见问题 |
| [07_TRAINING_AND_FINETUNING.md](docs/07_TRAINING_AND_FINETUNING.md) | 训练数据格式、ActionTokenizer、LoRA、全量微调 |
| [08_TERMS_AND_PARAMS.md](docs/08_TERMS_AND_PARAMS.md) | 术语和 CLI 参数速查 |
| [09_ENVIRONMENT_RTX5090.md](docs/09_ENVIRONMENT_RTX5090.md) | RTX 5090 环境、base demo 与 norm_stats 记录 |
| [reports/libero_eval/README.md](docs/reports/libero_eval/README.md) | LIBERO 评测记录（90%、兼容修复、回放） |
| [sim/README.md](sim/README.md) | 自定义 MuJoCo 仿真模块说明 |

推荐学习顺序：§3 模型输入输出 → ① + docs/01、02 → ② + docs/05 → ③ + docs/03、sim/README → ④ → docs/07 训练微调。
