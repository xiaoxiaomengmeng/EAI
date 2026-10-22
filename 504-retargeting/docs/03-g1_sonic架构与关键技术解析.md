# g1_sonic 架构与关键技术解析

> **证据等级**：实测背书型。所有机制描述均经源码核实（2026-10-21 七模块拆分后状态），验证数据标注「实测」；live 模式、SONIC 上游训练细节未实测，标注「待核」。
> 操作细节（标定步骤、场景参数、视角控制）在 [g1_sonic/README.md](../pico2g1/g1_sonic/README.md)，本文只写原理与架构；部署实测（setup.sh、权重下载、冒烟与回放判据）见 [01-pico2g1 部署复现实测记录](01-pico2g1部署复现实测记录.md)；g1_ik 链路的原理对照见 [02-g1_ik 架构与关键技术解析](02-g1_ik架构与关键技术解析.md)。

## 1. 定位与总览

`pico2g1/g1_sonic` 是三条链路中**策略驱动**的一条：不直接解 IK，而是把人体动捕转成 SMPL 参考，喂给 NVIDIA GEAR 的 SONIC v1.1 策略（encoder/decoder 两个 ONNX，CPU 推理），由策略输出 29 维关节目标、MuJoCo PD 伺服物理步进执行。与 g1_ik 的本质差异：g1_ik 是**运动学回放**（solve 结果直接写 qpos，物理不响应接触），g1_sonic 是**动力学控制**（物理步进会真的推道具、靠策略维持平衡）。

| 属性 | 值 |
|---|---|
| 控制频率 | 50 Hz（推理与物理推进同拍；物理以 10 子步 × timestep=1/(50×10) 展开） |
| 策略 | SONIC v1.1（`nvidia/GEAR-SONIC`），onnxruntime CPU（实测：不占 GPU） |
| 求解对象 | G1 本体 29 DoF（策略输出目标）+ Wuji Hand 左右各 20 关节（重定向直写） |
| 参考窗口 | `FUTURE_FRAMES = 10`（喂给策略的参考轨迹长度）；状态历史 `HISTORY_FRAMES = 10` |
| qpos 布局 | 7 浮动基座 + 29 本体 + 40 手部 = 76（互动场景物体追加在后） |
| 界面 | PySide6：MuJoCo 主画面（异步渲染）+ 三体相机 + 2D 人体/双手线框 + 姿态调节面板 |

端到端数据流（两个入口汇合同一条消费链，与 g1_ik 同形，多出策略环节）：

```
replay: session 目录/parquet ─┐
                              ├─> sonic_recording 加载/过滤/重采样 50 Hz
live:   ws://设备:9990 ───────┘            │
                                           ▼
                              sonic_source 帧生成（节拍/订阅/手套融合）
                                           │  LatestFrame 单槽总线（后台线程生产）
                                           ▼
              Qt 50 Hz 定时器消费 → convert（sonic_smpl，PICO→SMPL，输出 10 帧参考）
                                       → SonicPolicy.target（encoder+decoder ONNX）← 29 维目标
                                       → Simulator.advance：PD 伺服 10 子步 + 手部直写
                                           │
                                           ▼
                    sonic_window：MuJoCo 异步渲染 / 三相机 / 录制 / 推流 / 姿态调节
```

## 2. 架构

### 2.1 七模块平铺 + 瘦入口

2026-10-21 由 1646 行单文件重构而来：67 行瘦入口 + 7 个 `sonic_` 前缀平铺模块，依赖**严格单向无循环**（拆分模式与 g1_ik 同构，见 02 §2.1）：

```
sonic_recording  ←  sonic_config
sonic_model      ←  sonic_config
sonic_smpl       ←  sonic_config
sonic_policy     ←  sonic_config + sonic_model
sonic_source     ←  sonic_config + sonic_recording + sonic_smpl
sonic_window     ←  sonic_policy + sonic_source（+ 上述全部底层）
launch_g1_sonic.py  ←  sonic_window（并 re-export 四符号，见 2.2）
```

| 模块 | 行数 | 职责 |
|---|---|---|
| [sonic_config.py](../pico2g1/g1_sonic/sonic_config.py) | 123 | 常量表（关节名/SMPL 索引/电机增益与动作缩放）+ 资产路径；首次导入时把仓库根插入 `sys.path` |
| [sonic_model.py](../pico2g1/g1_sonic/sonic_model.py) | 91 | `build_model`：G1 + Wuji Hand MJCF 拼装、场景注入 |
| [sonic_recording.py](../pico2g1/g1_sonic/sonic_recording.py) | 231 | parquet 加载与过滤、live 消息解析、slerp/重采样、免 torch 读骨架 pkl |
| [sonic_smpl.py](../pico2g1/g1_sonic/sonic_smpl.py) | 129 | `convert`（PICO 24 关节 → SMPL 参考与 29 维腕/肘目标）、`apply_rotation_sources`、`WujiMapper` |
| [sonic_policy.py](../pico2g1/g1_sonic/sonic_policy.py) | 159 | `SonicPolicy`（encoder/decoder ONNX、10 帧历史、朝向对齐）+ `Simulator`（PD 伺服、10 子步、手部直写） |
| [sonic_source.py](../pico2g1/g1_sonic/sonic_source.py) | 139 | `replay_frames`/`live_frames` 帧生成、`LatestFrame` 帧总线 |
| [sonic_window.py](../pico2g1/g1_sonic/sonic_window.py) | 870 | `SourceView`/`MuJoCoView`/`Window`（异步渲染、三相机线程、推流、姿态调节） |
| [launch_g1_sonic.py](../pico2g1/g1_sonic/launch_g1_sonic.py) | 67 | argparse（replay/live）+ `run`/`main` + re-export |

### 2.2 三条结构性不变量（改代码前必读）

与 02 §2.2 同构的三条，值不同：

1. **路径推导集中在 sonic_config.py**：`REPO_ROOT`/`G1_XML`/`WUJI_ROOT`/`SKELETON_PATH` 基于模块自身路径推导（可被 `G1_DESCRIPTION`/`WUJI_ROOT` 覆盖），且首次 import 时把仓库根插入 `sys.path`——`from sim_scenes import ...` 依赖这一步，导入顺序不可调。
2. **入口 re-export 四个名字不可删**：`build_model`/`Simulator`/`load_skeleton`/`SMPL_OUTPUT_JOINTS`。tests/test_scenes.py 按文件路径加载入口后直接取这四个符号，删掉会**静默**打断测试。
3. **模块名 `sonic_` 前缀防撞名**：tests 加载时 `sys.path` 同时含 g1_ik 与 g1_sonic 两目录，无前缀会撞名。

### 2.3 与 sim_scenes 共享层的关系

与 g1_ik 共用同一[薄共享层](../pico2g1/sim_scenes/)（场景解析、LeRobot 录制、推流、渲染视图基建），sonic 侧额外用到的部分：`PulseSession`（互动场计分与回合，`for_model` 建实例、每子步 `update`）、`LockChain`/`SceneView`/`fit_renderer`（异步渲染锁协议基建）、`SourceViewStreams`（源面板图捕获）。g1_sonic_bridge 无界面，完全不碰这层。

### 2.4 线程模型

g1_ik 是「生产者 + Qt 定时器」两角色（02 §2.4）；g1_sonic 在此之上加了**三个常驻渲染角色**，是三链路线程模型最复杂的一条：

| 线程/角色 | 职责 | 关键交互 |
|---|---|---|
| 数据生产（daemon） | `publish_frames` 跑帧生成器 | 写 `LatestFrame` 单槽（与 g1_ik 同语义：取走即清空，只保最新帧） |
| Qt 主线程 20 ms tick | `consume_frame`：推理 `plan`（锁外，只读 `data`）→ 持 `scene_lock` `advance` 物理推进 → 更新源面板 → 请求异步渲染 → 录制 `capture` | 唯一修改 `data` 的线程 |
| `sonic-render` | 等待渲染请求，短持锁快照后在**自家 GL 上下文**渲染主画面 | 结果双投：`async_result`（GUI 显示）+ `publish_frame`（推流源） |
| `sonic-camera-{head,left,right}` ×3 | 各自 30 fps 渲染一个体相机，无人观看时释放 GL 上下文 | 写 `camera_images`/`camera_rendered_at` |
| `sonic-publish` | 把主画面帧与三相机帧（0.1 s 宽限等齐）+ 源面板图配成一路发布 | `publish_generation` 防旧帧重发 |

锁协议：`scene_lock`（物理推进 vs 渲染快照互斥）、`render_lock`（RLock，渲染参数：相机/阴影/取景）、快照时按 `LockChain(render_lock, scene_lock)` 双持、`camera_lock`/`publish_lock` 各管各的交换槽。MuJoCo Renderer 只能在创建它的线程用——这是 worker 与三相机各自建 GL 上下文的根因。

**线程错误队列**：worker 线程不能碰 Qt 控件，异常统一进 `thread_errors` 队列、由 GUI tick 打印并报状态栏。没有它，渲染线程死掉的表现是「画面冻结而缩放还能动」（缩放在 GUI 线程同步重绘）——极难排查的假象。

## 3. 关键技术

### 3.1 模型装配 build_model（sonic_model.py）

与 g1_ik 的 `build_model` 同构（02 §3.1 的 XML 手术：meshdir 重定向、删原厂手部 mesh、腕下装 docking adapter（`pos 0.0415`）+ Wuji 手掌（`pos 0.066`）同用绕 y 轴 90° 四元数、挂三相机、场景注入跳过自带 light）。两处差异：① 加载后 `model.opt.timestep = 1/(50×10)`——物理子步化是策略链路的必需项（PD 伺服稳定性），g1_ik 不改 timestep；② `Simulator` 在模型上建 `PulseSession` 计分。**实测**：`build_model()` 输出 `nq=76`。

### 3.2 免 torch 读骨架 pkl（sonic_recording.py）

`assets/human_joints_info.pkl` 是 `torch.save` 写的 zip 归档（SMPL 55 关节的静态 `J (55,3)` 与 `parents (55,)`）。sonic 的 venv 刻意不装 torch（ONNX 推理不需要，省 ~590 MB），`load_skeleton` 手写 `_TorchArchiveUnpickler`：拦截 `_rebuild_tensor_v2` 按存储 dtype 直接 `np.frombuffer` 映射载荷、`persistent_load` 从 zip 内 `data/<key>` 读字节。对照：g1_sonic_bridge 因 dex-retargeting 强制依赖 torch，那边直接 `torch.load`（04 §3.5）。

### 3.3 PICO 24 关节 → SMPL 参考（sonic_smpl.py）

`convert` 是整条链路的数学核心，五步：

1. 全局旋转右乘 `Y_180`（PICO 前向与 SMPL 前向相差 180°）。
2. 局部旋转 = `parentᵀ @ child`（`PARENTS_24` 拓扑）。
3. 55 关节 FK（根过 `Y_UP_TO_Z_UP` 转 z-up）后按 `SMPL_OUTPUT_JOINTS = [0..21, 39, 54]` 取 **24 个关节位置**，再经 `body_rot = root_z_up @ SMPL_BASE_INV` 转到躯干坐标——`smpl_joints`。
4. 肘部 swing-twist 分解（`swing_euler`）：把肘部旋转投影掉扭转分量只留摆动——G1 肘是单自由度铰链，扭转对策略是无意义输入。
5. 29 维 `joint_pos` 只填肘/腕 6 列（IsaacLab 序 23-28，`WRIST_ISAACLAB`；右侧按镜像取负）。

输出 dict 三键：`smpl_joints [10,24,3]`、`body_quat [10,4]`（wxyz）、`joint_pos [10,29]`——第一维是 `FUTURE_FRAMES` 的参考轨迹。`apply_rotation_sources` 在 convert 之前做两件事：腕四元数替换（tracker 为准，位置仍用 24 关节）+ 四关节局部欧拉修正（`rotation_offsets.json`，界面可调，见 3.6）。`WujiMapper` 与 g1_ik 同款（21 关键点 → wuji-retargeting 解析解，x 镜像翻转、关节序校验，详见 02 §3.6）。

### 3.4 SonicPolicy 推理管线（sonic_policy.py）

**encoder 输入 1751 维**的布局（`encoder_input` 拼装）：

| 区间 | 维数 | 内容 |
|---|---|---|
| `[0]` | 1 | 常数 2.0 |
| `[911:1631]` | 720 | `smpl_joints`：10 帧 × 24 关节 × 3 |
| `[1631:1691]` | 60 | 参考朝向：10 帧 × 6（旋转矩阵前两列展平） |
| `[1691:1751]` | 60 | `joint_pos` 肘腕列：10 帧 × 6（`WRIST_ISAACLAB`） |

其余区间留零。**实测**（2026-10-21 onnx 加载冒烟，见 01 §3.3）：encoder `obs_dict [1,1751] → encoded_tokens [1,64]`。

**朝向对齐**：首帧记录 `heading_alignment = heading(robot) @ heading(ref₀)⁻¹`（heading 取绕竖直轴的 yaw），此后每帧 `relative = heading(robot)⁻¹ @ alignment @ ref`——参考轨迹被整体转到机器人当前朝向的坐标系里追踪。含义：机器人初始朝向与录制朝向无需一致，策略追踪的是**相对动作**。「重置机器人」按钮同时清空 10 帧历史、`heading_alignment` 与 `DEFAULT_ANGLES` 复位。

**decoder 输入 994 维** = encoder token 64 + 10 帧状态历史 930（每帧 93 = IMU 角速度 3 + qpos 偏差 29 + qvel 29 + last_action 29 + 重力投影 3）。历史 `deque(maxlen=10)`，开局用首个观测填充 10 份；qpos 偏差相对 `DEFAULT_ANGLES`（SONIC 训练 rest pose）计算，关节序经 `MUJOCO_TO_ISAACLAB` 重排。输出 29 维 action：`target = DEFAULT_ANGLES + action[ISAACLAB_TO_MUJOCO] × ACTION_SCALE`，非 finite 直接抛错。**实测**：decoder `obs_dict [1,994] → action [1,29]`。

### 3.5 PD 伺服与动作缩放（Simulator）

增益按 4 型电机参数表（`ARMATURE`/`EFFORT`）推导：`KP = ARMATURE × (20π)²`、`KD = 4 × ARMATURE × 20π`（踝/足 6 个关节 ×2 增强）；`ACTION_SCALE = 0.25 × EFFORT / (ARMATURE × ω²)`——动作限幅等价于各关节 ±25% 最大静态力矩的偏转角，从物理上封顶策略输出。

`advance`：10 子步（每子步 `control` PD + `mj_step` + `pulse.update`）→ 手部 qpos 直写 clip + `qvel` 清零 + `mj_forward` → 记录 `commanded_qpos`（物理结果但身体 29 维换成 target 的混合量，供录制当 action 侧证据）。

**plan/advance 分离**：`plan` 只读 `data`（推理不推进物理），调用方在 `scene_lock` **外**跑推理、锁内只做 `advance`——推理耗时不出现在物理临界区里，渲染线程不会被 ONNX 推理卡住。

### 3.6 姿态调节闭环（sonic_window.py）

四关节（左右肘 + 左右腕 tracker）局部 XYZ 欧拉修正，写 `rotation_offsets.json`（本地不入库，启动自动加载，非有限值静默忽略）。对照 g1_ik 的 19 滑杆 `tuning.json`（02 §3.7）：sonic 只修**输入侧朝向**，不碰策略与增益。调节语义：replay 置 `pause_event` 挂起 / live 继续收包但锁定画面；每拖一格用锁定的当前帧重算 convert（repeat 成 10 帧参考）并完整跑一遍 `plan`——策略反应立即可见；「保存并继续」落盘并恢复。调节只影响喂给策略的参考，右侧 2D 面板始终显示未经修正的原始数据。

### 3.7 异步渲染、推流与录制（sonic_window.py）

渲染架构与锁协议见 2.4。推流配对：publish 线程把主画面帧、三体相机帧（0.1 s 宽限等齐 `captured_at`）、GUI 线程捕获的源面板图配成一路 MJPEG 帧发布（八路推流，默认 8876，共享层实现）。

录制：`RecordingControls` 挂在窗口（`action_semantics` 声明「Clipped SONIC position targets for body joints; clipped hand retargeting targets for fingers.」），每 tick `recording.capture(data, commanded_qpos)`；三路视频在 finalize 阶段由录制状态机重渲染。**关键取舍**：worker 渲染/推流线程死亡走 `thread_errors` 队列，**不置** `failed`——录制只存 qpos/action、视频靠 finalize 重渲染，死预览不该毁掉一段好数据；数据流本身的错误（`latest.error`）才会置 `failed` 停录停解算。`closeEvent` 在录制/导出忙时延迟关窗（等 `settled`），关窗时逐线程 `join(2s)` 并关闭全部 GL 资源。

## 4. 验证记录

| 验证项 | 时间 | 结果 | 证据 |
|---|---|---|---|
| 依赖冒烟 + onnx 加载 | 2026-10-21 | ✅ 11 包全通；encoder 1751→64、decoder 994→29 | 01 §3.3 实测输出 |
| 七模块拆分五级回归 | 2026-10-21 | ✅ pyflakes/py_compile 15 文件、tests 7/7、GUI 真跑 25 s 存活、桥一致性对照 | `MuJoCo ready: 76 qpos` |
| replay 经 scripts/02 实跑 | 2026-10-21 | ✅ GUI 起窗、策略驱动、物理步进推道具、DISPLAY 探测 `:1` 生效 | 01 §3.4 实测 |
| GUI 录制（sonic 侧单独实跑） | — | ⚠️ 待核 | 录制状态机与 g1_ik 共用（02 §3.8），g1_ik 侧已实跑 |
| live 模式 | — | ⚠️ 待设备 | WebSocket 链路代码已核实 |

## 5. 边界与交叉引用

- **pico2g1 内**：[g1_ik](../pico2g1/g1_ik/)（运动学 IK 链路，原理见 02）；[g1_sonic_bridge](../pico2g1/g1_sonic_bridge/)（数据桥，原理见 04——桥是给独立 SONIC 进程喂数据的，与本地这条 GUI 链路不直接相连）。三方法互不 import、各自 .venv。
- **共享层**：[sim_scenes/](../pico2g1/sim_scenes/) 的职责分组与边界见 02 §2.3。
- **部署实测**：环境搭建、权重下载（本机必须 hf-mirror）、冒烟与回放判据全部在 [01-pico2g1 部署复现实测记录](01-pico2g1部署复现实测记录.md)。
- **上游（待核范围）**：SONIC v1.1 的训练细节（obs 1751 维其余留零区间的语义、`obs[0]=2.0` 的含义）属 NVIDIA GEAR 侧知识，本文只描述输入输出的已核实事实。
- **本系列文档**：`504-retargeting/docs/` 按序号顺延命名，本文为 03。
