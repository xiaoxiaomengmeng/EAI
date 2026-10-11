# PI0 + UR5e MuJoCo 仿真：方块放入马克杯 部署指南

> 目标：在 MuJoCo 的 UR5e + Robotiq 2F85 场景中，用真实 PI0 策略（PaliGemma 2B + Action Expert 300M + flow-matching action head）完成「抓起彩色方块 → 放入马克杯」闭环，**不使用 lerobot 的评测/环境框架**，仿真、数据采集、推理桥接全部自主可控。
>
> 本文同时作为 IK（逆运动学）学习材料。

---

## 0. 结论速览（先看这里）

1. **不用 lerobot 跑评测：完全可以。** MuJoCo 环境、控制循环、相机渲染、数据格式都自己写，lerobot 只在「训练」和「模型定义」两个环节出现。
2. **但 `pi0_base` 不能零样本完成任务。** 它是预训练基座（32 维 ALOHA 动作空间、3 相机），没见过 UR5e 和这个桌面。**必须先在 MuJoCo 里采数据 → 微调 PI0 → 再部署推理**，这是本项目最大的工作量（没有捷径，任何基座都一样）。
3. **推理端最省事的脱钩方式不是从零重写，而是「vendor lerobot 的 pi0 策略模块」**：把 `lerobot/policies/pi0/` 下模型定义和预处理链拷进本项目当普通 Python 包，权重用 `model.safetensors` 直接加载。完全从零重写约 800–1200 行且容易和权重对不上，不推荐。
4. 动作空间建议选 **关节空间（6 关节角 + 1 夹爪）**，与 PI0/ACT/ALOHA 主流训练范式一致，IK 只用于遥操作采数阶段，不进策略推理闭环（降低实时性和稳定性风险）。
5. 已验证的基础事实：UR5e 模型 37 项资产核查全 PASS；`grip` actuator **0=闭合 / 255=张开，与真机协议相反**；场景无相机传感器，需要自己加。

---

## 1. 总体架构

```
┌────────────────────────── 训练时（离线） ──────────────────────────┐
│ 遥操作采数 (view_ur5e --drag / 键盘 / 3D 鼠标)                       │
│   MuJoCo 渲染: base相机 + wrist相机 ──┐                              │
│   state: 6关节角+夹爪(7维, pad到32)   ├─> episode (lerobot dataset) │
│   action: 7维(pad到32) + prompt文本 ──┘    parquet/jsonl + mp4     │
│                  │                                                   │
│                  ▼                                                   │
│   lerobot 训练脚本 (PI0Policy, 只在训练环节用)                       │
│   输入3×224²图像 + 32维state + prompt → 50步action chunk             │
│                  │                                                   │
│                  ▼ 产出微调后权重 (config.json + safetensors)        │
└──────────────────────────────────────────────────────────────────────┘
┌────────────────────────── 推理时（在线，无 lerobot 环境） ───────────┐
│ 自主 MuJoCo 环境 (sim_robot/, 纯 mujoco + numpy + torch)             │
│  ① mj_step 后: Renderer 渲染 3 路相机、读 6 关节+夹爪 state           │
│  ② 预处理: resize224 / MEAN_STD 归一化 / prompt tokenize / pad       │
│  ③ PI0 推理: 10 步 flow-matching 去噪 → 50×32 action chunk           │
│  ④ 取前 n_action_steps，前 7 维 → ctrl：                             │
│       6 关节角直接喂位置伺服；夹爪通道反向 (255-x) 后喂 grip          │
│  ⑤ Ruckig/线性插值按 50Hz 下发，循环                                  │
│  ⑥ 成功判定: 方块在杯口内且高度低于杯沿 → success                    │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 2. 可行性评估（三个关键问题）

### Q1：必须用 lerobot 吗？

| 方案 | 说明 | 工作量 | 推荐度 |
| --- | --- | --- | --- |
| A. 全用 lerobot | lerobot dataset + policy + 自定义 MuJoCo env 适配 lerobot 接口 | 环境要适配 lerobot 的 Gym 风格，受其抽象约束 | ★★☆ |
| **B. lerobot 仅作训练/模型库（推荐）** | 训练用 lerobot；推理时把 pi0 模型代码 vendor 成本地包，环境完全自研 | 中，推理桥接约 300–500 行 | ★★★★★ |
| C. 完全脱离 lerobot | 自己实现 PaliGemma 编码 + Expert + flow matching + 全部预处理 | 800–1200 行，权重对齐风险高 | ★★ |
| D. 导出 ONNX/TensorRT | 先在 lerobot 里导出，再用 onnxruntime 直推 | 中，调试难度高，PI0 动态 shape 多 | ★★★（后期提速再做） |

**推荐 B**：lerobot 本质只提供两样东西——PI0 的 PyTorch 模型代码 和 训练循环。前者可以拷出来当库用（Apache-2.0，合规），后者只在训练机上跑。你的 MuJoCo 仿真和评测完全不 import lerobot。

### Q2：pi0_base 现在能直接完成任务吗？

**不能。** 实测其配置：

- `paligemma_variant=gemma_2b`、`action_expert_variant=gemma_300m`、`chunk_size=50`、`num_inference_steps=10`
- 输入：3 路相机（`base_0_rgb` / `left_wrist_0_rgb` / `right_wrist_0_rgb`，3×224×224）+ `observation.state`（32 维）
- 输出：`action` 32 维（ALOHA 双臂移动平台维度，单臂 UR5e 用前 7 维、其余补 0）
- 预处理链（`policy_preprocessor.json`）：rename → batch → 换行处理 → **tokenizer** → device → **relative_actions（动作转增量）** → **MEAN_STD 归一化**

基座没在 UR5e 桌面数据上微调时，输出的动作分布与你的场景无关。必须走「采数 → 微调」。

### Q3：工作量与硬件

| 阶段 | 内容 | 预计工作量 |
| --- | --- | --- |
| 场景改造 | 加 2~3 路相机、状态/奖励接口、稳定抓物参数整定 | 1–2 天 |
| 数据采集 | 遥操作采 50–100 条成功轨迹（拖拽 IK + 夹爪） | 1–2 天 |
| 微调 PI0 | LoRA 或全参，5090 32G，LoRA 数小时级 | 0.5–1 天 |
| 推理桥接 | vendor 模型代码 + 预处理复现 + chunk 下发 | 2–3 天 |
| 闭环调参 | 时序、夹爪阈值、动作平滑 | 1–3 天 |

---

## 3. 现有资产盘点

| 资产 | 路径 | 用途 |
| --- | --- | --- |
| UR5e+2F85 模型 | `RynnMotion/models/3.robot_arm/21.ur5e/` | 仿真本体（22kg 真实惯量、25 传感器） |
| 抓取场景 | 同上 `scene/scene_pickplace.xml` | 6 个 2cm freejoint 方块 + 固定马克杯 |
| Pinocchio 运动学版 | 同上 `mjcf/ur5e_pinocchio.xml` | RynnMotion C++ IK 用 |
| 自主查看器 | `sim_robot/view_ur5e.py` | 已有：关节控制、`--drag` IK 跟随、场景切换 |
| 资产核查 | `sim_robot/verify_ur5e.py` | 37 项自动核查 |
| PI0 基座权重 | `101-pi-serial-models/models/pi0_base/` | lerobot 格式预训练权重 |
| lerobot 源码 | `101-pi-serial-models/repos/lerobot-main/` | 训练 + 模型代码来源 |
| PI0 训练脚本 | `101-pi-serial-models/scripts/22_train_pi0.sh` | 微调模板 |
| PI0 原理文档 | `101-pi-serial-models/docs/26-PI0模型原理与源码对照详解-v1.0.md` | 架构细节 |

---

## 4. 需要下载 / 安装的代码

### 4.1 推理环境（建议新建独立 venv，复用 cosmos 思路）

必需：

```bash
# 在任意 py3.10/3.12 环境中
pip install torch  # 5090 用 2.9.1+cu128（你已有 uv 缓存，秒装）
pip install mujoco==3.9 numpy pillow transformers safetensors sentencepiece protobuf
pip install imageio opencv-python   # 录视频/resize
```

- **`transformers` 是必须的**：PI0 的视觉/语言塔是 PaliGemma（SigLIP + Gemma），tokenizer 和视觉预处理直接复用。
- **不需要** qpOASES / Pinocchio（除非推理闭环要走末端控制，见 §6）。

### 4.2 PI0 模型代码（vendor，不 pip install lerobot）

从 `101-pi-serial-models/repos/lerobot-main/src/lerobot/policies/pi0/` 拷贝（实际文件结构）：

```
pi0/
├── configuration_pi0.py     # PI0Config（你的 config.json 即序列化自它）
├── modeling_pi0.py          # PI0Policy：PaliGemma 包装 + Expert 300M + flow matching
├── processor_pi0.py         # PI0 专用前/后处理链（tokenize / relative action / 归一化）
└── __init__.py
```

另需依赖的通用模块：`lerobot/common/policies/` 下的 normalization（MEAN_STD）等基础类。把它们拷到 `sim_robot/pi0_runtime/` 作为本地包，逐文件删掉对 dataset/env/训练器的无关 import。加载方式：

```python
# 伪代码：权重直接读 safetensors，配置读 config.json
from pi0_runtime import PI0Policy, PI0Config
cfg = PI0Config.from_pretrained("101-pi-serial-models/models/pi0_base")
policy = PI0Policy.from_pretrained("101-pi-serial-models/models/pi0_base", config=cfg)
policy.eval()
```

### 4.3 训练环节（保留 lerobot 原生）

训练直接用现成的 `101-pi-serial-models` 环境和 `22_train_pi0.sh`，数据集用采数阶段产出的 lerobot dataset（parquet + mp4）。这样训练侧零额外开发。

---

## 5. IK 知识详解（学习章节）

> 本项目中 IK 的定位：**遥操作采数**用 IK 把人手/目标位姿转成关节角；如果策略输出末端位姿，则推理闭环也要 IK。理解它对调试和后续做跨本体泛化很重要。

### 5.1 正运动学与雅可比

机械臂构型用广义坐标 `q ∈ ℝⁿ`（UR5e 是 6 个关节角）描述。正运动学：

```
x = FK(q)          # x: 末端位姿（位置 xyz + 姿态）
```

对时间求导得到微分关系：

```
ẋ = J(q) q̇        # J 即雅可比矩阵
```

UR5e 只控位置时 J 是 **3×6**；位姿全控时是 6×6（上 3 行线速度、下 3 行角速度）。MuJoCo 中 `mj_jacSite(model, data, jacp, jacr, site_id)` 一步算出，Pinocchio 中是 `computeJointJacobians` + `getFrameJacobian`——这就是 RynnMotion 要单独维护 `ur5e_pinocchio.xml` 的原因。

### 5.2 微分逆运动学（DLS，阻尼最小二乘）

已知期望末端速度 `ẋ_d`，反解关节速度。朴素伪逆解：

```
q̇ = J⁺ ẋ_d = Jᵀ(JJᵀ)⁻¹ ẋ_d
```

问题：接近奇异点（臂伸直、关节共线）时 `JJᵀ` 接近奇异，`q̇` 爆炸，表现为机械臂突然甩动。**DLS（Levenberg-Marquardt）** 加入阻尼 λ：

```
q̇ = Jᵀ (JJᵀ + λ²I)⁻¹ ẋ_d
```

λ>0 保证矩阵永远可逆，代价是奇异点附近跟踪变慢——用「速度」换「稳定」，工业界主流做法。`view_ur5e.py --drag` 用的就是它（λ=0.05）：

```python
err = target_pos - ee_pos                    # 末端位置误差
dq  = J.T @ np.linalg.solve(J @ J.T + damp**2 * np.eye(3), err)
q   = clip(q + clip(dq, -0.1, 0.1), q_lo, q_hi)   # 限速 + 关节限位
```

每物理步（1ms）迭代 3 次，实测 22cm 偏移 2000 步收敛误差 0.0000m。

### 5.3 QP 形式（RynnMotion 的 DiffIKQP）

DLS 的 QP 等价表述：

```
min_{q̇}  ‖J q̇ − ẋ_d‖²  +  λ²‖q̇‖²
s.t.      q̇_min ≤ q̇ ≤ q̇_max          （速度限位）
          关节位置/加速度限位
          零空间任务（如保持期望臂型）
```

RynnMotion C++ 侧构造 `H = JᵀJ`、`g = −2Jᵀẋ_d`，用 **qpOASES** 求解（`motion/utils/kinematics/diff_ik.cpp`），比纯 DLS 多了硬约束和零空间能力。OSC 模块 yaml 里 `solver: qp` 走这条，`solver: pinv` 走简单伪逆（`motion/module/osc/osc.cpp`）。

### 5.4 OSC（操作空间控制）

IK 只解运动学，OSC 是动力学层控制：在末端任务空间设计 PD（`F = Kp(x_d−x) + Kd(ẋ_d−ẋ)`），再用动力学模型把末端力映射成关节力矩：

```
τ = Jᵀ F + 重力补偿 + 零空间项
```

RynnMotion 的 OSC 模块流程：末端位姿误差 → 期望末端速度 → DiffIK 出关节速度/角度 → 动力学转换 → 关节指令。它是「末端策略」落地到关节伺服之间的完整桥梁。

### 5.5 本仓库三套 IK 对照

| 实现 | 文件 | 雅可比来源 | 求解 |
| --- | --- | --- | --- |
| 迭代 IK（位姿级） | `motion/utils/kinematics/kine_pin.cpp` `PinKine::ikPos` | Pinocchio | DLS 迭代 |
| 微分 IK（QP） | `motion/utils/kinematics/diff_ik.cpp` `DiffIKQP` | Pinocchio | qpOASES，带限位/零空间 |
| Python 版 | `python/src/RynnMotion/algorithms/diff_ik.py`、`osc.py` | Pinocchio | scipy |
| **我们的采数脚本** | `sim_robot/view_ur5e.py --drag` | MuJoCo `mj_jacSite` | numpy DLS |

机器人选择是数据驱动的：`config/robot.yaml` 中每台机器人配 `pino_mjcf` 字段，UR5e 指向 `ur5e_pinocchio.xml`，所以同一套 IK 代码对所有机器人通用。

### 5.6 策略该输出关节还是末端？

RynnMotion 的动作消息 `act_command`（`common/lcm/lcmMotion/`）**双通道并存**：`jointPos/jointVel`（关节）与 `eePos/eeQuat`（末端），外加 `gripperPos` 和 `workMode` 切换。

| | 关节策略 | 末端策略 |
| --- | --- | --- |
| 策略输出 | 6~7 维关节角 | xyz+quat（6/7 维） |
| 推理闭环 | 直接喂位置伺服，Ruckig 平滑 | 必须在线 OSC/DiffIK |
| 实时性 | 好（无迭代求解） | 依赖 IK 收敛 |
| 跨本体泛化 | 差（绑定关节定义） | 好（1 主手控多从臂靠它） |
| 与 PI0 主流数据 | **一致（ALOHA/lerobot 默认）** | 需自己定义动作空间 |

**本项目选关节策略**：PI0 基座本来就按关节动作训练，采数时 IK 只用于「人怎么示教」，记录下来的是关节轨迹；策略学会后直接输出关节角，推理时不需要 IK，闭环简单可靠。

---

## 6. 观测、动作与场景改造

### 6.1 加相机（场景现状没有相机传感器）

`scene_pickplace.xml` 里的 camera 仅供查看器渲染，不能产生图像数据。方案：**不改原文件**，仿照 `--drag` 的注入法，加载前往 xml 注入 2 路相机：

- `base_camera`：第三人称俯视（桌面全局）
- `wrist_camera`：装在 `wrist_3_link` 上的随动相机（手眼视角）

推理时用 `mujoco.Renderer` 按相机名渲染 224×224 RGB：

```python
renderer.update_scene(data, camera="base_camera")
img = renderer.render()          # HWC uint8
```

PI0 配置要 3 路相机，单臂场景可以把 `right_wrist` 与 `left_wrist` 喂同一张腕部图（微调后模型只依赖有效信息），或微调时改配置为 2 路。

### 6.2 state / action 定义（32 维填充）

```python
state[:7]  = [q1..q6, gripper_norm]   # gripper_norm 归一化到 [0,1]
action[:7] = [q1..q6, gripper_cmd]    # 其余 25 维补 0
```

- 关节角用弧度，夹爪通道建议统一映射到 0~1 后再进 MEAN_STD。
- **夹爪反向**：MuJoCo 模型 0=闭合、255=张开；下发时 `ctrl = 255 * (1 - action_grip)`。
- 训练数据和推理环境必须用完全相同的填充约定和角度符号。

### 6.3 相对动作（relative_actions）

预处理链含 `relative_actions_processor`：训练时动作被转成相对当前状态的增量（delta），推理输出后要加回当前关节角。微调时保持该 processor 默认即可，但**推理桥接必须复现这一步**，否则动作会整体偏移。这是脱离 lerobot 时最容易踩的坑之一。

### 6.4 归一化统计量

MEAN_STD 的 mean/std 来自**你的训练数据集**（不是基座的旧统计）。微调产出的 checkpoint 里会带新统计（或在 dataset stats 文件中），推理时用微调后的数值，不能用 base 目录里的。

### 6.5 任务成功判定

- 取每个方块的 freejoint body 世界坐标；
- 杯口中心与半径从马克杯几何推算（杯壁由一圈 capsule geom 组成，读 mug body 子 geom 的位置）；
- 成功条件：方块 xy 在杯口半径内 **且** z 低于杯沿 **且** 持续 N 帧（防弹跳出杯误判）。

---

## 7. 实施路线

### 阶段 0：环境与基线（0.5 天）

- [ ] `view_ur5e.py --scene pickplace --drag` 拖臂到方块上方，确认能视觉上接近方块
- [ ] 用 Control 面板/脚本调夹爪 ctrl，确认 2cm 方块能被夹起（摩擦/接近方向整定）
- 风险：方块 20mm vs 最小指垫间距 14.9mm，靠硅胶垫挤压摩擦抓取；若抓不稳可把方块改到 15mm 或提高指垫 friction。

### 阶段 1：场景与环境封装（1–2 天）

新建 `sim_robot/ur5e_cube_env.py`：`reset()` / `step(action_chunk)` / `get_obs()` / `is_success()`；注入相机、固定随机化（方块位置 xy 随机、杯位固定）；Renderer 出图。

### 阶段 2：遥操作采数（1–2 天）

扩展 `view_ur5e.py` 为录制器（参考 RynnMotion 的 `mj_recorder` 思路，但自写更简单）：

- 拖动红球做末端示教，后台 DLS-IK 实时转关节；快捷键控制夹爪开合；
- 每条轨迹记录：3 路图（或 2 路）、7 维 state、7 维 action、文本 prompt（如 `"put the red cube into the mug"`）；
- 导出 lerobot dataset 目录格式（`data/chunk-000/parquet` + `videos/`），供训练直接消费；
- 目标 50–100 条成功轨迹，方块颜色/位置随机。

### 阶段 3：微调 PI0（0.5–1 天）

用 `101-pi-serial-models/scripts/22_train_pi0.sh` 改数据路径训练；5090 32G 优先 LoRA（参考项目既有经验）；保留 32 维、50 chunk、10 去噪步配置不变，减少对齐成本。

### 阶段 4：无 lerobot 推理桥接（2–3 天）

- vendor `pi0/` 模型代码到 `sim_robot/pi0_runtime/`；
- 复现预处理：resize224、STATE/ACTION MEAN_STD、relative action 反变换、PaliGemma tokenizer；
- 10 步 flow-matching 去噪出 50×32 chunk；
- 时序策略：每 10 个控制步重推理一次，期间顺序执行 chunk（temporal ensembling 可后做）。

### 阶段 5：闭环评测（1–3 天）

20 个随机种子位置，统计入杯成功率；输出视频；逐步加动作平滑（Ruckig 或简单线性插值）、夹爪迟滞阈值。

---

## 8. 风险与坑清单

1. **零样本幻觉**：base 模型直接推理只会输出无意义动作，别在这一步浪费时间调试。
2. **relative actions 忘反变换**：动作整体偏移的最常见原因。
3. **归一化统计用错**：必须用微调数据集的 mean/std。
4. **夹爪方向反**：模型 0 闭 / 255 开，真机协议相反。
5. **控制频率不匹配**：训练数据 50Hz（20ms），MuJoCo 物理 1ms——action 每 20 个物理步切换一次，中间插值。
6. **chunk 时序**：PI0 一次出 50 步（1 秒），不要每步都重新推理（10 步扩散很贵），按 n_action_steps 滚动执行。
7. **32 维填充**：7 维以外必须严格补 0，训练/推理一致。
8. **图像通道与分辨率**：PI0 输入 224×224（注意 smoke test 脚本里出现过 320，以微调 config 的 `image_resolution=[224,224]` 为准），HWC uint8 → CHW float 的转换在 processor 内，别重复归一化。
9. **GPU 显存**：PI0 推理约 5–8GB，5090 无压力；首次加载 PaliGemma tokenizer 需联网或提前缓存（国内设 HF 镜像/离线）。

---

## 9. 验收标准

- 数据：≥50 条成功轨迹，回放检查关节/夹爪/视频对齐；
- 训练：loss 正常下降，checkpoint 可被 vendor 的运行时加载；
- 推理：固定 10 个测试位置成功率 ≥70%（入门级目标）；
- 桥接：整个推理进程不 import lerobot（`grep -r lerobot sim_robot/pi0_runtime` 仅剩 license 注释）。
