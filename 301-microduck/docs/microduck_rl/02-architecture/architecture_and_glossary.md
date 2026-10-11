# microduck_rl 项目整体理解与关键概念详解

> 本文是对 [`microduck_rl/`](../../../microduck_rl/) 仓库的**学习与梳理笔记**，分两部分：
> - **第一部分**：项目整体理解——定位、技术栈分层、代码结构、三条技术链路、核心契约、工程文化。
> - **第二部分**：关键概念详解——逐个解读项目涉及的框架、算法、硬件、语言术语（mjlab 插件、MuJoCo Warp、rsl_rl、PPO、BAM、ONNX、Rust、61-D 契约、`uv run train` 调用链路等），每个概念都贴着本项目的代码讲，并附具体例子与踩坑。
>
> 配套文档：[`desktop_quickstart.md`](../01-quickstart/desktop_quickstart.md)（台式机部署与首次训练）。
> 仓库官方文档：[`README.md`](../../../microduck_rl/README.md) 与 [`AGENTS.md`](../../../microduck_rl/AGENTS.md)。

---

# 第一部分 · 项目整体理解

## 1. 一句话定位与生态位

**microduck_rl 是 microduck 双足鸭子机器人的"策略训练场"**——一个约 19k 行 Python 的 [mjlab](https://github.com/mujocolab/mjlab) 插件包，用 MuJoCo Warp 做 GPU 并行物理、rsl_rl 的 PPO 做训练、Rhoban 的 BAM 做高保真舵机模型，把 27 个运动技能训成 ONNX，再发布到 HF Hub 供隔壁 Rust 仓库 [`microduck`](../../../microduck/)（机器人运行时）在 RK3566 板子上加载执行。

它在整个生态里是一条**单向数据流的源头**：

```text
microduck_rl (本仓库, Python/GPU)          microduck (Rust runtime, RK3566 板子)
─────────────────────────────────         ──────────────────────────────────────
MuJoCo Warp 仿真 + PPO 训练                  robotd 50Hz 控制循环
  → checkpoint (model_*.pt)                   → 加载 policy.onnx
  → export ONNX (烘焙 obs normalizer)   ───►    → obs[1,61] → policy → action[1,14]
  → publish 到 HF Hub (schema-2 manifest)      → safety 层 clamp → 15 舵机
```

关键：**训练侧和部署侧靠一个 61-D 观察 / 14-D 动作的契约对齐**。这个契约在两个仓库里各自实现、互相不能 import，靠测试和启动断言锁住一致性。

- 机器人本体：约 **25 cm、800 g** 的双足鸭子，**14 个 Dynamixel XL330 舵机**驱动（外加真机上一个不被策略驱动的 `mouth` 关节，共 15 个出线关节）。
- 出品方 [Pollen Robotics](https://pollen-robotics.com/microduck)，许可证 Apache-2.0。
- 能力：走路、装轮子后滚动、用喙捡东西、被推倒自己站起、坐、踢球、前滚翻、原地自旋。

---

## 2. 技术栈与框架分层

这个项目**不是从零造 RL 框架**，而是站在四层现成组件上做"插件式扩展"：

| 层 | 组件 | 版本 pin | 职责 | microduck_rl 对它做了什么 |
|---|---|---|---|---|
| 物理引擎 | **MuJoCo Warp** (`warp-lang`) | 1.12.0 | GPU 上并行跑数千个环境 | 只提供 XML 模型 + 调 solver 参数 |
| RL 框架 | **mjlab** | 1.3.0 | Manager-based env、task registry、terrain、velocity 基类 | 注册 27 个 task、继承 `VelocityOnPolicyRunner`、monkey-patch 4 处 |
| 训练算法 | **rsl_rl** (PPO) | 随 mjlab | On-policy PPO、obs normalizer、ONNX 导出 | patch `compute_returns` 消毒 NaN advantage |
| 执行器模型 | **BAM** (`better-actuator-models`) | git `mjlab_frictionloss` 分支 | XL330 舵机的电压控制 + back-EMF + 负载相关摩擦 | 子类化加 `friction_scale` DR + 编码器透过齿隙反馈 |

> 每个术语的详细解读见[第二部分](#第二部分--关键概念详解)。

**框架哲学**：mjlab 是"manager-based"架构——一个 env 由若干 manager（observation / action / reward / termination / event / command / curriculum）组成，每个 manager 里是一堆具名 term，每个 term = 一个函数 + 权重 + 参数。microduck_rl 的绝大部分代码就是**往这些 manager 里塞 term**，而不是写新的 env 循环。（详见 [§A.7](#a7-mjlab-框架哲学manager-based-架构详解)）

---

## 3. 代码结构（按职责分层）

```text
src/mjlab_microduck/
├── tasks/                    ← 核心：任务定义层（19 个文件，占代码量大头）
│   ├── __init__.py           (278)  任务注册表 + MicroduckOnPolicyRunner
│   ├── mdp.py                (7188) ★ 所有 obs/reward/event/curriculum/command 函数 + 4 个 patch
│   ├── microduck_velocity_env_cfg.py (949)  行走主 recipe，其他 env 的模板
│   ├── microduck_standup_env_cfg.py  (1159) 起身（最大的单任务 cfg）
│   ├── microduck_sitstand/ground_pick/ball_kick/roulade/spin/... 各任务 cfg
│   ├── backlash.py           (92)   把任意 env cfg 变换成齿隙变体
│   ├── symmetry.py           (169)  61-D 布局的左右镜像表（对称 loss）
│   └── slope_terrain.py      (115)  斜坡地形
│
├── robot/                    ← 机器人模型层
│   ├── microduck_constants.py (275) ★ 8 套 XML 变体 + EntityCfg + BAM 执行器参数 + HOME 姿态
│   ├── microduck/*.xml        模型文件（walk / groundcontact / rollers / backlash / ball / apartment）
│   ├── microduck/assets/*.stl 网格
│   └── add_backlash.py       (139)  后处理器：给每个 servo 关节串一个 ±1° 齿隙铰
│
├── actuator/                 ← 执行器层
│   └── friction_dr_bam.py    (112)  BAM → FrictionDRBam → BacklashEncoderBam 三层继承
│
├── sim/                      ← 反向 sim2real 层
│   ├── body_server.py        (545) ★ 把 MuJoCo 鸭子经 TCP 喂给真机 robotd
│   └── tof.py                (92)   模拟 VL53L5CX 8×8 深度传感器
│
├── publish/                  ← 发布层
│   ├── cli.py                (227)  `uv run publish`：导出→gate→上传 HF Hub
│   └── manifest.py           (386)  schema-2 manifest + shape gate + smoke run（无 torch 依赖）
│
├── export.py                 (311) ★ checkpoint → ONNX 的唯一安全路径
├── hf_jobs.py                (483)  `--hf-jobs`：提交到 HF 托管 GPU
├── train_hook.py             (64)   从 mjlab.tasks 入口拦截 --hf-jobs
└── train_cli.py              (36)   与 mjlab 等价的 train shim
```

**依赖方向**（单向，无环）：

```text
tasks/*_env_cfg.py  →  tasks/mdp.py  →  mjlab + rsl_rl
        ↓                    ↓
robot/microduck_constants.py  actuator/friction_dr_bam.py  →  bam
        ↓
    robot/microduck/*.xml  (MuJoCo 模型)

publish/cli.py  →  export.py  →  mjlab.rl.exporter_utils
publish/manifest.py  →  (纯 onnx/json，无 torch，可在无 GPU 笔记本上跑测试)
sim/body_server.py  →  sim/tof.py  →  (纯 mujoco，独立于训练栈)
```

---

## 4. 三条技术链路

### 4.1 链路 A：训练链路（命令 → checkpoint）

```text
uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096
      │
      ├─(1) console script `train` → 实际是 mjlab 的 train（本仓库的 shim 故意输掉命名冲突）
      ├─(2) mjlab 加载 `mjlab.tasks` 入口点 → import mjlab_microduck.tasks
      │        └─ 模块顶层第一件事：maybe_submit_to_hf_jobs()  ← 若有 --hf-jobs 在此拦截并 SystemExit
      │        └─ 否则：register_mjlab_task() × 27，把 task_id → (env_cfg, rl_cfg, runner_cls) 建表
      ├─(3) 按 task_id 取出 env_cfg（make_microduck_velocity_env_cfg()）
      ├─(4) 构建 ManagerBasedRlEnv → Scene 编译 MuJoCo 模型 → warp kernel 上 GPU
      ├─(5) MicroduckOnPolicyRunner(env, rl_cfg)
      └─(6) runner.learn() → PPO 循环：每 iteration = 24 步 × num_envs 并行 rollout
               → reward → compute_returns(被 patch 消毒) → PPO 更新 → 每 250 iter 存 model_<iter>.pt
```

> `uv run train` 的完整调用链路（含命名冲突的来龙去脉）见 [§F.3](#f3-uv-run-train-的调用链路)。

**训练循环里一步的数据流**（manager-based 架构的精髓）：

```text
observation manager  → 拼出 actor obs[61] 和 critic obs[61+特权]
       ↓
policy (actor MLP)   → action[14]（关节位置增量）
       ↓
action manager       → 缩放 + 加到 default joint pos → 目标位置
       ↓
BAM 执行器           → 电压控制律：back-EMF + Coulomb/Stribeck/负载摩擦 + 延迟 → 力矩
       ↓
MuJoCo Warp          → mj_step × decimation，物理前进
       ↓
reward manager       → 各 term 加权（NaN 被 patch 消毒成 0）
       ↓
termination manager  → 判断哪些 env 该 reset
       ↓
event manager        → reset 时重采样 DR（CoM/质量/摩擦/IMU 朝向/编码器偏置/push）
       ↓
curriculum manager   → 按 iteration 推进 stage（权重、命令范围、地形难度、push 强度）
```

### 4.2 链路 B：部署链路（checkpoint → 真机）

设计上**层层设卡防止手滑**：

```text
model_3000.pt (checkpoint)
      │  uv run scripts/export.py <task> --wandb-run-path ent/proj/run --checkpoint 3000
      ▼
export.run_export()  ← ★唯一安全路径★
      │  runner.export_policy_to_onnx() 发射 actor(normalizer(obs))
      │  obs normalizer 被"烘焙"进图 → 真机跑的就是训练看到的，喂原始 obs 即可
      ▼
policy.onnx  [1,61] → [1,14]
      │  uv run publish --task ... --repo user/microduck-my-walk --kind perpetual --slot walk
      ▼
publish/cli.py run()
      ├─ _resolve_weights()：从 checkpoint 走 export，或直接用现成 onnx
      ├─ check_onnx()：★shape gate★ 拒绝非 61→14（51-D 老 policy 明确拒绝）
      ├─ smoke_run_onnx()：喂零 obs 跑 50 步，拒绝 non-finite / std=0（常数输出）
      ├─ build_manifest() + validate_manifest()：schema-2
      ├─ render_readme()：生成 model card（含 robotctl 安装命令）
      └─ HfApi.upload_folder()：传 policy.onnx + manifest.json + README.md
      ▼
HF Hub repo  →  真机: sudo robotctl policy load walk user/microduck-my-walk
```

**为什么这么设卡**：`manifest.py` 顶部写明——daemon 侧（`microduck` 仓库的 `updater/src/policy.rs` + `robotd-params`）读同一套 schema，"the daemon refuses a policy whose manifest disagrees with these"。这个模块**故意不 import mjlab/torch**，所以测试能在笔记本上毫秒级跑、CLI 能在无 GPU 时验证 ONNX。

`export.py` 的注释点破核心陷阱：in-sim `play` 自己会应用 normalizer，**会掩盖手转 checkpoint 忘了烘焙 normalizer 的 bug**——真机上策略看到未归一化 obs，行为完全崩。所以"never convert by hand"。

### 4.3 链路 C：反向 sim2real（body_server）

一条**方向相反**的链路：不是把策略部署到真机，而是**把仿真里的鸭子身体喂给真机的完整软件栈**。

```text
uv run duck-body --ducks 4 --port 7801        robotd --sim 127.0.0.1:7801
      │                                              │
   MuJoCo World（4 只鸭子共享一个地板，能互相撞）      真机 robotd 的全部代码：
      │  每 20ms 一个 control tick（batch 4×5ms）      50Hz 循环 / ONNX 策略 / safety /
      ▼                                              跌倒检测 / 里程计 / kinematics / IPC
   Body.sensors() ──TCP JSON(NDJSON)──────────────► RobotIo trait 的 sim 实现
   {positions[15], velocities[15], currents_ma[15],      ↑
    imu{gyro,gravity,quat}, trunk_z, sim_time}      只有这个进程知道"没有真机器人"
      ▲                                              │
      └──────────── set_targets(write) ◄─────────────┘
```

**两个"这边故意拥有"的映射**（原文重点）：
1. **15 关节出线，14 关节进模型**：daemon 按 `JOINT_NAMES` 索引，含 index 9 的 `mouth`（真机有、walk 模型没有），body_server 负责插入/丢弃，"daemon 不该知道这件事"。
2. **给重力，不只给朝向**：策略观察躯干系下的 projected gravity，MuJoCo 只给朝向四元数，所以这边做旋转——**和真机 IMU 的 SFLP 滤波器是同一套算术，在同一根线的两侧**。

配套细节：`TIMESTEP=0.005`（BAM 拟合、接触 solref、armature 都在此步长下调过，不是性能旋钮）；用 `scene.xml` 而非 `scene_walk.xml`（后者 actuator default class 带 `contype="0" conaffinity="0"`，机器人会穿过地板）；鸭子"held until daemon takes it"（双足静态姿态不稳定，纯位置控制一秒内就倒，而 robotd 启动时故意不上扭矩）。

---

## 5. 核心契约：61-D obs / 14-D action

整个项目**最重要的不变量**，训练、导出、部署三处必须字节级对齐。权威布局定义在 [`symmetry.py`](../../../microduck_rl/src/mjlab_microduck/tasks/symmetry.py) 的模块注释里：

```text
actor obs [61] = 本体感受 48 + 命令块 13
  [0:3]   base_ang_vel       陀螺仪 (roll, pitch, yaw — body-frame IMU)
  [3:6]   projected_gravity  躯干系重力方向 (gx, gy, gz)
  [6:20]  joint_pos_rel      14 关节位置（相对 default 姿态）
  [20:34] joint_vel_rel      14 关节速度
  [34:48] last_action        上一步的 14 维动作
  [48:51] twist command      (lin_vel_x, lin_vel_y, ang_vel_z)
  [51:55] head command       (neck_pitch, head_pitch, head_yaw, head_roll 增量)
  [55:61] body command       (x, y, z, roll, pitch, yaw 增量)

action [14] = 14 个 servo 关节的目标位置增量
```

14 维关节块内的顺序（来自 `robot_walk.xml` body tree）：

```text
0: left_hip_yaw    5: neck_pitch    9:  right_hip_yaw
1: left_hip_roll   6: head_pitch    10: right_hip_roll
2: left_hip_pitch  7: head_yaw      11: right_hip_pitch
3: left_knee       8: head_roll     12: right_knee
4: left_ankle                       13: right_ankle
```

**为什么所有 policy 共享这个布局**（README 原话）：这正是**运行时策略热切换**成为可能的原因——daemon 可以在 walk / stand / sitstand / ground_pick / kick / roulade 之间无缝切换，因为它们吃同样形状、同样语义的 obs。

命令块的 13 维是**零填充约定**：不用的槽填零但保留维度。比如行走任务的 `body_pose(6)` 权重是 0（"infra kept intact but DISABLED"），但 obs 槽和 command 都活着，好让 standup env 直接抬权重、扩范围就能用。相位编码（ground_pick 的 `[cos2πφ, sin2πφ, 0]`）、姿态标志（sitstand 的 `[sit_flag, 0, 0]` 住在 twist vx 槽）都塞在这 13 维里。

> 61-D / 14-D 的逐维解读见 [§E](#e-组契约61-d-obs--14-d-action)。

---

## 6. 机器人模型体系（8 套 XML + passive_ 约定）

[`microduck_constants.py`](../../../microduck_rl/src/mjlab_microduck/robot/microduck_constants.py) 定义了一族碰撞模型，按任务需要选：

| 模型 | 碰撞范围 | 用于 |
|---|---|---|
| `robot_walk.xml` | 脚 + 少量 | Velocity 行走（碰撞越少 solver 越稳） |
| `robot_groundcontact.xml` | 接触地面的部件（sole/腿/躯干壳/头壳/喙/电池/髋） | standup / ground-pick / sitstand / roulade |
| `robot_allcollisions.xml` | 每个部件都带碰撞 geom（70 geoms） | 2026-09 导出，暂无任务用（未来需要全接触的 env） |
| `robot_groundcontact_rollers.xml` | + 被动轮铰 `passive_*wheel` | 轮滑任务 |
| `*_backlash.xml` × 3 | 每个 servo 串一个 `passive_<joint>_backlash` ±1° 齿隙铰 | Backlash 变体任务 |
| `ball.xml` | 70mm/15g 自由球 | BallKick |

**`passive_` 命名约定是整套设计的黏合剂**：所有非驱动关节（轮子、齿隙铰、喙连杆）都带 `passive_` 前缀，于是每个选择器只要写 `^(?!passive_).*` 就自动排除它们——动作空间恒定 14 维、obs 恒定对齐、导出不会因 passive 关节 KeyError。这个约定让 [`backlash.py`](../../../microduck_rl/src/mjlab_microduck/tasks/backlash.py) 的变换器只需 90 行就能把任意 env 变成齿隙变体。

**执行器三层继承**（[`friction_dr_bam.py`](../../../microduck_rl/src/mjlab_microduck/actuator/friction_dr_bam.py)）：

```text
bam.mjlab.BamActuator（电压控制 + back-EMF + 摩擦）
   └─ FrictionDRBamActuator   +friction_scale：每 env 缩放 BAM 速度无关摩擦预算（stiction/齿轮箱）
        └─ BacklashEncoderBamActuator  固件 PD 读编码器"透过齿隙"：cmd.pos = qpos[main]+qpos[backlash]
                                        （真机磁编码器在齿隙输出侧，模拟这个事实）
```

---

## 7. mdp.py 的分层（7188 行的组织）

这个巨型文件是项目的"函数库"，按段落组织：

| 行范围 | 段落 | 内容 |
|---|---|---|
| 1–113 | **4 个 monkey-patch** | Patch 1 `RewardManager.compute` 消毒 NaN reward；Patch 2 `PPO.compute_returns` 消毒 NaN advantage；Patch 3（已在 mjlab 1.3.0 迁移中移除）；Patch 4 `get_base_metadata` 导出时过滤 passive_* 关节 |
| 119–156 | **辅助函数** | `_servo_joint_ids/pos/vel/default`：按名字解析 servo-only 视图，抗 passive_* interleave |
| 158–1833 | **通用构件** | 行走/起身/蹲/轮子的 reward、obs、termination、command、curriculum |
| 1834–3140 | **Ground Pick** | 分段相位门控（下降/停顿/上升/休息）、mouth 接近、载荷力 |
| 3141–5031 | **Domain Randomization Events** | IMU 朝向、CoM、质量惯量、BAM 摩擦、push、命令 curriculum |
| 5032–5691 | **Unified pose command** | head_pose tracking、body_pose 6D、UniformPoseCommand、sit/stand posture |
| 5692–5855 | **BallKick** | 球 reset、kick reward、critic-only 球观察 |
| 5856–6132 | **SPIN** | 相位包络驱动的自旋 |
| 6133–6593 | **Backlash + SitStand** | 编码器透过齿隙 obs、posture 复合奖励 |
| 6594–7188 | **Roulade** | 前滚翻的累积器、over-the-head latch、进度率门控 |

> MDP（马尔可夫决策过程）是什么、monkey-patch 是什么，见 [§B](#b-组训练算法概念)。

---

## 8. 任务矩阵（27 个注册任务）

[`tasks/__init__.py`](../../../microduck_rl/src/mjlab_microduck/tasks/__init__.py) 注册了 14 个任务族，多数有 Flat/Rough 两版 + Backlash 变体：

| 任务族 | task_id 前缀 | 技能 | 特殊机制 |
|---|---|---|---|
| Velocity | `Mjlab-Velocity-{Flat,Rough}` | 行走 + 头部/身体姿态跟踪 | 主 recipe，head_pose_bias 修复 DC 下垂 |
| VelStand | `Mjlab-VelStand-*` | 行走 + 跌倒恢复 + 身体控制合一 | run-5→6→7 演化弧，recovery gated on tilt |
| StandUp | `Mjlab-StandUp-*` | 从趴/仰/坐起身 | reverse curriculum，mid-recovery spawn |
| SitStand | `Mjlab-SitStand-*` | 命令式坐↔站 | posture flag 在 twist vx 槽，push 延迟引入 |
| GroundPick | `Mjlab-GroundPick-*` | 蹲下用喙尖碰地再起 | 相位编码 `[cos2πφ,sin2πφ,0]`，温柔 push ±0.15 |
| BallKick | `Mjlab-BallKick-Flat` | 右脚踢球 | actor 对球盲、critic 看球（非对称） |
| Rollers | `Mjlab-Velocity-Flat-*-Rollers` | 轮滑 | cmd_x: 0=滑行 >0=推 <0=刹 |
| Swizzle | `Mjlab-Velocity-Swizzle` | 轮滑 swizzle | 从 stride 派生，删 anti-swizzle 项 |
| RollerCrouch | `Mjlab-RollerCrouch-Flat` | 轮上蹲 glide | 真机 read_pose 读姿态 |
| RollerSlope | `Mjlab-RollerSlope-Flat` | 斜坡滚 | FlatRampTerrain，重力自然滚 |
| RollerStandUp | `Mjlab-RollerStandUp-Flat` | 轮上起身 | 滚动摩擦反向 curriculum |
| Spin | `Mjlab-Spin-Flat` | 原地自旋 | roller 物理 + ground_pick 相位，对称关 |
| Roulade | `Mjlab-Roulade-Flat` | 前滚翻 | 支撑门控累积器防 ballistic whip |
| （以上多数） | `*-Backlash-*` | 齿隙变体 | `make_backlash_variant()` 自动变换 |

---

## 9. 工程文化：几个体现品味的决策（保留案例与推理）

这个项目的注释密度极高，且**几乎每个非显而易见的决策都附一次真机失败或几周调试**。几个典型：

**1. 符号约定"bit four envs"**：`mdp.py` 里有两种惩罚风格——mjlab 基础成本返回 ≥0（配负权重），microduck 自写的 `*_penalty`/`*_l1` 返回 ≤0（已自 negated，配正权重）。给自 negated 的惩罚再加负权重 = 双重否定 = **奖励违规行为**。真实案例：`roller_standup` 的 `gentle_rise` 用 `trunk_vertical_accel_penalty`（返回 `-|a_z|`）乘继承来的 `-0.02`，等于 `+0.02·|a_z|`——躯干越暴力加速 policy 越被付钱，日志确认 `Episode_Reward/gentle_rise=+0.0118`（run `vweolw91`，唯一记录为正的惩罚项）。

**2. head 下垂不能用收紧容差修**：头走路时俯冲 ~15°（run `ww1g2198` 实测 14.6° 均误差）。run `5yay13u4` 试着把 `head_pose_tracking` 的 `fine_std` 收到 0.1，结果策略 iter 300 前**完全停止行走**（air_time 1.01→0.02，熵 10.9→1.9 崩溃）。推理：280g 的头占机器人体重 38%，迈步时**必然**振荡，瞬时紧容差是**不可逃逸**的税（0.77/step，占 air_time 奖励 76%），站着不动反而得分高。正解：只对**可逃逸的 DC 偏差**收税——1s EMA 上的 L1（`head_pose_bias`），走路的策略在最优处一分钱不花。

**3. CoM curriculum 上限 ±15mm**：曾设 ±30mm，超出脚支撑多边形，"An accumulating CoM randomizer once degraded every long run for months"。

**4. push range ±0.3**：曾 ±0.5，训出"永久紧张步态"（2026-07 audit 收窄）。原文：一个大于最大行走速度（0.4）的 additive kick 每 3-6s 来一次，会训出永久紧张的跌倒恢复步态。

**5. aarch64 torch 的两个静默断点**：PyPI 的 aarch64 torch wheel 是 CPU-only（`torch.version.cuda is None`），mjlab 的 `select_gpus()` 索引空列表在第一步训练前就 `IndexError`。修法（`pyproject.toml` 的 `[tool.uv.sources]` 路由到 cu129 索引）有两个静默失效点，都被 `test_aarch64_cuda_torch.py` 锁住——两种情况 `uv sync` 都成功，只有启动 run 时才发现。

**6. `train` console script 命名冲突**：本仓库的 `train` 故意和 mjlab 的撞名（last-writer-wins，mjlab 赢）。但**必须保持声明**——删掉后 `uv sync` 会卸载 `.venv/bin/train`（按本仓库 RECORD）而不重装 mjlab 的，`uv run train` 落到 PATH 上任何叫 `train` 的东西（2026-08-31 落到 liblinear 的）。

**7. Critic-only `_safe` obs**：2026-08-21 Velocity2-Rough-Backlash 崩溃——critic 的传感器 term（foot_contact_forces/height/air_time）读 raycast/contact 数据，MuJoCo 可能在状态还干净时返回 non-finite，一个 NaN 经 rsl_rl 的 check_nan 杀掉整个 run。修法：这三个 term 在 critic 侧换成 `*_safe` 版本，"sanitizing costs the policy nothing"（actor 看不到）。

**8. 非对称 actor-critic**：BallKick 的 actor 对球盲（真机无球感知），critic 看球位姿。这是"训练时用特权信息、部署时策略只吃能拿到的"的标准手法。

---

## 10. 测试策略

`tests/` 里 19 个测试文件**全是 CPU-only、无 GPU、毫秒级**——它们不跑仿真，而是锁住不变量：

- `test_aarch64_cuda_torch.py`：pyproject 的 torch 路由 + lockfile 的 wheel 来源 + 版本跨平台一致
- `test_hf_jobs_flag.py`：`train` script 声明不被删（否则 `uv sync` 删掉它）
- `test_publish_manifest.py`：manifest schema-2 的所有拒绝规则
- `test_nan_guard.py` / `test_obs_nan_guard.py`：patch 1/2 生效
- `test_*_cfg.py`：各 env cfg 能构建、关节索引锁死、符号约定
- `test_infer_policy_bam.py`：`infer_policy.py` 里**镜像**（不 import）的 BAM 参数和 `microduck_constants.py` 一致（镜像是为了避免拖进 mjlab/torch/warp 的 ~16s 导入）

---

## 11. 总结：这个项目"是什么"

一句话——**microduck_rl 是一个高度工程化的 mjlab 插件，它把"训一只会走会滚会翻身会踢球的鸭子"这件事，拆成了可组合的 manager term + 一族碰撞模型 + 一套贯穿训练/导出/部署的 61-D 契约，并用大量 monkey-patch、shape gate、smoke run 和 CPU-only 测试，把每一个曾经真机失败过的坑固化成不变量。**

它最值得学的地方不在 RL 算法（那是 mjlab/rsl_rl 的），而在**如何让一个 sim2real 管线不出事**：契约单一真相源、部署路径唯一化、层层 gate、注释说"为什么"而非"做什么"、每个非显而易见决策配一个测试。

---

# 第二部分 · 关键概念详解

> 按主题分组：**A** 框架与仿真 · **B** 训练算法 · **C** 执行器与部署 · **D** 语言 · **E** 契约 · **F** 命令链路。
> 每个概念都先讲"是什么"，再讲"在本项目里扮演什么角色"，尽量附具体例子。

## A 组：框架与仿真

### A.1 什么是 mjlab 插件包，它的作用是什么

**mjlab** 是一个基于 MuJoCo Warp 的机器人 RL 训练框架（`mujocolab/mjlab`），它提供：manager-based 环境骨架、任务注册表（task registry）、地形生成器、一组预置任务基类（如 velocity 行走）、以及和 rsl_rl 的对接。

**"插件包"的含义**：microduck_rl **不修改 mjlab 源码**，而是作为一个独立的 Python 包，通过 mjlab 预留的**入口点（entry point）机制**把自己"挂"进去。具体在 [`pyproject.toml`](../../../microduck_rl/pyproject.toml) 里：

```toml
[project.entry-points."mjlab.tasks"]
mjlab_microduck = "mjlab_microduck.tasks"
```

这行声明的作用是：当 mjlab 启动时（`import mjlab`），它的 `__init__.py` 会在模块作用域调用 `_import_registered_packages()`，自动 import 所有注册在 `mjlab.tasks` 这个 entry-point 组下的包——于是 `mjlab_microduck.tasks` 被执行，里面的 `register_mjlab_task()` × 27 就把 microduck 的任务塞进了 mjlab 的全局任务表。

**它解决什么问题**：
1. **解耦**——microduck 的机器人模型、奖励、课程都是"业务逻辑"，不该污染通用框架。插件机制让框架和业务分离。
2. **即插即用**——`uv sync` 装上 microduck_rl 后，`uv run train Mjlab-Velocity-Flat-MicroDuck` 就能用，不需要改 mjlab 一行代码。
3. **一条无法被安装顺序夺走的钩子**——`--hf-jobs` 的拦截正是利用"mjlab 自己会 import 这个插件"的事实（见 [§F.3](#f3-uv-run-train-的调用链路)），因为这是 mjlab 的导入路径，任何安装顺序都保证它会执行。

**类比**：就像浏览器和它的扩展——浏览器（mjlab）提供内核和扩展 API，扩展（microduck_rl）用清单文件（`pyproject.toml` 的 entry-points）声明"我要在启动时被加载"，然后往内核里注册自己的功能。

---

### A.2 什么是 MuJoCo Warp，它和 MuJoCo 什么区别

**MuJoCo**（Multi-Joint dynamics with Contact）是 DeepMind 开源的物理仿真引擎，专门仿真带关节、带接触的刚体动力学。它是机器人 RL 的事实标准仿真器之一。经典 MuJoCo 是 **C 写的、单线程 CPU** 仿真：一次仿真一个机器人（或一个批次但在 CPU 上串行）。

**MuJoCo Warp** 是 MuJoCo 的 **GPU 重写版**（NVIDIA 和 Google DeepMind 合作），把物理仿真搬到 GPU 上，用 NVIDIA 的 Warp 运行时（见 [§A.3](#a3-什么是-warp-lang)）**同时并行仿真成千上万个环境**。

| 维度 | 经典 MuJoCo | MuJoCo Warp |
|---|---|---|
| 硬件 | CPU | GPU（CUDA） |
| 并行度 | 单环境/串行批次 | 数千环境同时仿真（本项目用 4096） |
| 用途 | 精确单次仿真、可视化、部署演练 | RL 训练（吞吐量优先） |
| API | C API / `mujoco` Python 绑定 | 经 mjlab 封装 |

**为什么 RL 训练必须用它**：PPO 是 on-policy 算法，要海量交互数据。在 CPU 上一只鸭子一只鸭子地仿真，训一个行走策略要几周；在 GPU 上 4096 只鸭子并行，1-2 小时就能出可用步态。这就是 [§4.1](#41-链路-a训练链路命令--checkpoint) 里"每 iteration = 24 步 × 4096 envs ≈ 100k env steps"能达到的原因。

**本项目的分工**：
- **训练**用 MuJoCo Warp（GPU，4096 并行）——经 mjlab 封装，microduck_rl 只碰 XML 模型和 cfg。
- **部署演练**用经典 MuJoCo（CPU，单环境）——[`scripts/infer_policy.py`](../../../microduck_rl/scripts/infer_policy.py) 和 [`sim/body_server.py`](../../../microduck_rl/src/mjlab_microduck/sim/body_server.py) 都是 `import mujoco`，因为真机运行时是单只鸭子，不需要 GPU 并行。

> 一个关键约束：训练用的 `TIMESTEP`、BAM 执行器拟合、接触 `solref`、关节 `armature` 都是在特定步长下调好的，所以 body_server 也用 `TIMESTEP=0.005`——"0.002 gives a duck whose legs reach the right angles and still cannot hold itself up"，物理参数是配套的一整套，不能只改步长。

---

### A.3 什么是 warp-lang

**Warp**（PyPI 包名 `warp-lang`，import 名 `warp`）是 **NVIDIA 开源的 Python GPU 计算运行时**。它让你用 Python 写 GPU kernel（类似 CUDA C，但语法是 Python），Warp 在运行时把它 JIT 编译成 GPU 代码。

**它和 MuJoCo Warp 的关系**：MuJoCo Warp 的物理引擎就是用 Warp 写的 GPU kernel。所以 `warp-lang` 是 MuJoCo Warp 的**底层依赖**——mjlab 跑起来时需要它。

**本项目怎么对待它**：只在 [`pyproject.toml`](../../../microduck_rl/pyproject.toml) 里 pin 版本 `warp-lang==1.12.0`（和 mjlab 1.3.0 配套），代码里**几乎不直接 import warp**——它被 mjlab 和 MuJoCo Warp 封装掉了。pin 死版本是因为 warp 的 kernel ABI 和 mujoco-warp 强耦合，版本漂移会导致编译失败或数值不一致。

**类比**：Warp 之于 MuJoCo Warp，就像 CUDA 之于一个 CUDA 应用——是底层运行时，应用开发者（microduck_rl）通常不直接碰，但必须保证版本匹配。

---

### A.4 什么是 terrain（地形系统）

**terrain** 指 mjlab 提供的**地形生成与导入系统**（`mjlab.terrains`）。RL 训练的机器人不能只在平地上走，否则一到真机遇到地砖、草地、斜坡就摔。terrain 系统程序化生成各种地形，让机器人在训练时就见过。

mjlab 的两种地形模式：
- **`terrain_type="plane"`**：无限大平地（Flat 任务用）。
- **`terrain_type="generator"`**：程序化生成的崎岖地形（Rough 任务用），由 `TerrainGeneratorCfg` 描述。

本项目的地形配置在 [`microduck_velocity_env_cfg.py`](../../../microduck_rl/src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py) 的 `MICRODUCK_ROUGH_TERRAINS_CFG`：

```python
import mjlab.terrains as terrain_gen
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
```

**几个实践细节**（保留踩坑）：
- **台阶高度 ≤ 1.5cm**：microduck 腿短，太高跨不过去。
- **`random_grid` 用 0.45m 格子**：曾想用小格子（0.12m）做更细的崎岖，但 `BoxRandomGridTerrainCfg` 的格子越小、每个 patch 的 box 越多——0.12m 会变成 66×66=4356 boxes/patch，总 ~261K boxes，直接 **GPU OOM**。0.45m 是 17×17=289 boxes/patch，可控。
- **`_soften_terrain_contacts`**：崎岖地形里相邻 box 高度不同形成硬边，会让接触 solver 产生 NaN 力。修法是把 box 的 `solref` 从 0.02 调到 0.04（更软），减少 NaN。
- **斜坡地形**单独在 [`slope_terrain.py`](../../../microduck_rl/src/mjlab_microduck/tasks/slope_terrain.py) 实现 `FlatRampTerrainCfg`（平地 + 斜坡 + runout），供 RollerSlope 任务用。

**terrain curriculum**：地形难度不是一上来就最难，而是随训练推进（`terrain_levels_slope` 等 curriculum 函数按 iteration 升级难度），这叫**课程学习**（curriculum learning）。

---

### A.5 什么是 velocity 基类

mjlab 内置了一个**速度命令行走任务**的完整实现（`mjlab.tasks.velocity`），它是"让机器人跟踪一个速度指令（前进/侧移/转向）走路"这个最经典 locomotion 任务的**模板基类**。microduck 的行走 env 不是从零写，而是**继承并改造**它。

涉及的几个基类：

| 基类 | 来自 | 作用 |
|---|---|---|
| `make_velocity_env_cfg()` | `mjlab.tasks.velocity.velocity_env_cfg` | 生成一个通用 velocity env cfg（含 obs/reward/command 的基础 term） |
| `UniformVelocityCommand` | `mjlab.tasks.velocity.mdp.velocity_command` | 速度命令 term：每 env 采样一个 (lin_x, lin_y, ang_z) 目标速度 |
| `VelocityOnPolicyRunner` | `mjlab.tasks.velocity.rl` | 训练 runner（包 rsl_rl 的 OnPolicyRunner，加 velocity 任务的导出逻辑） |

**microduck 怎么用**（[§4.1](#41-链路-a训练链路命令--checkpoint) 的 (3)(5) 步）：

```python
# 1. 拿基类 cfg 当起点，再逐项改造
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
cfg = make_velocity_env_cfg()
cfg.scene.entities["robot"] = MICRODUCK_WALK_ROBOT_CFG      # 换成 microduck 机器人
cfg.observations["actor"].terms["joint_pos"]...             # 改 obs（排除 passive_*、加延迟/噪声）
cfg.rewards["head_pose_tracking"] = RewardTermCfg(...)      # 加 microduck 特有的奖励
cfg.commands["twist"] = VelocityCommandCommandOnlyCfg(...)  # 换命令实现

# 2. 继承基类 runner，只加一个 pickle 修复
class MicroduckOnPolicyRunner(VelocityOnPolicyRunner):
    def __init__(self, env, train_cfg, ...):
        super().__init__(...)
        # 把 symmetry_cfg 里不可 pickle 的 MjSpec(_env) 从待序列化 dict 剔除
```

**为什么继承而非重写**：velocity 基类已经把"速度跟踪奖励、命令重采样、观测布局、PPO 对接"这些通用逻辑写好了，microduck 只需要**替换机器人、加头部/身体姿态命令、调 DR 和 curriculum**。这是框架复用的典型——基类提供 80% 通用骨架，插件补 20% 业务特化。

**派生关系**：项目里几乎所有 env 都以 velocity env 为模板派生。比如 Swizzle env "从 stride env 派生：删 `_ANTI_SWIZZLE` 项、加 `leg_symmetry`+`grounded`"；Spin env 是"roller 物理 + ground_pick 的 phase 机制"的 hybrid。

---

### A.6 "调 solver 参数"怎么理解

**solver（求解器）** 是物理引擎里解接触约束的数值求解器。MuJoCo 用一个叫 **PGS/Newton + line search** 的迭代求解器算"两个物体接触时不互相穿透、摩擦力多大"。每步仿真，solver 要迭代若干次逼近真解。

**"调 solver 参数"就是调这些迭代次数和收敛容差**，在**精度、稳定性、速度**之间权衡。本项目碰的几个参数：

| 参数 | 含义 | 本项目的值/调整 |
|---|---|---|
| `nconmax` | 最大同时接触点数 | velocity rough 用 200（默认 35 太小，机器人摔倒时多 link 撞多 box 会爆） |
| `iterations` | Newton 求解器迭代次数 | rough terrain 调到 30 |
| `ls_iterations` | line search 迭代次数 | 调到 50 |
| `solref` | 接触的"弹簧-阻尼"参考（软硬度） | `_soften_terrain_contacts` 把 box 从 0.02 调到 0.04（更软，减 NaN） |
| `timestep` | 仿真步长 | 0.005（训练与 body_server 一致；scene XML 自带 0.002，但被 sim cfg 覆盖） |

**为什么要调**（保留推理）：
- **接触爆炸 → NaN**：崎岖地形里机器人摔倒，多个 body link 同时撞多个 box，接触点数瞬间暴涨。如果 `nconmax` 不够，接触被截断，物理失真；如果 solver 迭代不够，接触约束没收敛，力变成 NaN。NaN 一旦进 reward，经 [§B.6](#b6-什么是-monkey-patch猴子补丁) 的 patch 链会杀掉整个 run。
- **硬边 → 不稳定**：相邻 box 高度差形成刀刃状硬边，solver 在硬边上算出的接触力数值病态。把 `solref` 调软（0.02→0.04）让接触像"软弹簧"，力变化平滑，NaN 减少。

**类比**：solver 参数就像拍照时的"对焦次数"和"快门"——对得越久（迭代越多）越清晰（精确）但越慢，场景越复杂（接触越多）越需要多对几次，否则糊（NaN）。

---

### A.7 mjlab 框架哲学：Manager-Based 架构详解

这是理解整个项目的**钥匙**。mjlab 的核心设计哲学是：**一个 RL 环境不是一个铁板一块的 `step()` 函数，而是一组正交的"管理器（manager）"，每个管理器装一堆可插拔的"项（term）"。**

#### 传统写法 vs manager-based 写法

**传统 RL env**（如 Gym 风格）：你写一个类，`step()` 里手动算 obs、手动算 reward、手动判断 done：

```python
class MyEnv:
    def step(self, action):
        obs = self._compute_obs()          # 手写拼接
        reward = -self._fall_penalty() + self._speed_reward()  # 手写加权
        done = self._check_fallen()        # 手写判断
        return obs, reward, done, info
```

问题：obs/reward/termination 的逻辑全耦在 `step()` 里，想加一个奖励项要改 `step()`，想给某个 env 变体去掉一项要写 if-else，复用困难。

**manager-based 写法**（mjlab）：你把每个关注点拆成独立的 **term**，用**配置（cfg）**声明式地组装：

```python
cfg.rewards["head_pose_tracking"] = RewardTermCfg(
    func=microduck_mdp.head_pose_tracking,   # 一个纯函数
    weight=2.0,                               # 权重
    params={"command_name": "head_pose", "std": 0.5},  # 参数
)
cfg.observations["actor"].terms["joint_pos"] = ObservationTermCfg(
    func=mdp.joint_pos_rel, noise=Unoise(-0.001, 0.001), ...
)
cfg.terminations["fallen"] = TerminationTermCfg(func=..., time_out=False)
```

然后 mjlab 的 **RewardManager / ObservationManager / TerminationManager** 在运行时自动遍历所有 term、调用函数、按权重求和、拼接 obs。**你从不写 `step()` 里的奖励循环**——框架替你跑。

#### 七个 manager

| Manager | 管什么 | term 配置类 | 例子 |
|---|---|---|---|
| **Observation** | 拼 obs（分 actor/critic 两组） | `ObservationTermCfg` | `joint_pos`、`projected_gravity`、`head_command` |
| **Action** | 把 policy 输出变成物理指令 | `JointPositionActionCfg` | 14 维关节位置增量 |
| **Reward** | 加权求和所有奖励项 | `RewardTermCfg` | `head_pose_tracking`(w=2.0)、`upright`(w=2.0) |
| **Termination** | 判断 episode 何时结束 | `TerminationTermCfg` | `fallen_state`、`robot_state_is_nan` |
| **Event** | reset/startup/interval 时改状态（DR 都在这） | `EventTermCfg` | `randomize_com`、`push_by_setting_velocity` |
| **Command** | 生成给机器人的指令 | `UniformVelocityCommandCfg` | twist、head_pose、body_pose |
| **Curriculum** | 随训练推进调参数 | `CurriculumTermCfg` | `com_range_curriculum`、`push_curriculum` |

#### 举例：这个哲学如何贯穿本项目

**例 1——加一个奖励项 = 加一行 cfg**。velocity env 要给"头部姿态跟踪"加奖励，不需要碰任何循环逻辑：

```python
cfg.rewards["head_pose_tracking"] = RewardTermCfg(
    func=microduck_mdp.head_pose_tracking, weight=2.0,
    params={"command_name": "head_pose", "std": 0.5},
)
```

`head_pose_tracking` 是 [`mdp.py`](../../../microduck_rl/src/mjlab_microduck/tasks/mdp.py) 里一个 60 行的纯函数（输入 env，输出每 env 的奖励张量）。RewardManager 每步自动调它、乘 weight、加进总奖励。

**例 2——一个 env 变体 = 增删几个 term**。Swizzle env 从 stride env 派生，整个"改造"就是：

```python
# 删掉不想要的反 swizzle 项
for name in ("single_support", "glide", "skating_air_time", "gait_symmetry", "hip_roll_neutral"):
    cfg.rewards.pop(name)
# 加 swizzle 需要的项
cfg.rewards["leg_symmetry"] = RewardTermCfg(func=microduck_mdp.leg_symmetry_reward, ...)
cfg.rewards["grounded"] = RewardTermCfg(func=microduck_mdp.grounded_reward, ...)
```

不用重写 env，只是"换零件"。

**例 3——Backlash 变体 = 一个 90 行的变换器**。[`backlash.py`](../../../microduck_rl/src/mjlab_microduck/tasks/backlash.py) 的 `make_backlash_variant(cfg)` 接收任意 env cfg，做四件事：换 robot cfg、把 `joint_pos`/`joint_vel` obs 的 `func` 换成 `joint_pos_rel_backlash`、把 `dof_pos_limits` 奖励的 `asset_cfg` 缩到 servo 关节、给 pose 奖励加齿隙排除。**15 个 backlash 任务全靠这一个函数生成**，没有一个重复的 env 类。

**例 4——actor/critic 非对称 = 两组 obs term**。ObservationManager 管 `actor` 和 `critic` 两个独立组。BallKick 让 actor 组不含球 obs（真机看不到球），critic 组含球位姿（训练时特权信息）。这只是"往不同组塞不同 term"，框架天然支持。

#### 这个哲学的代价

manager-based 不是没有缺点，本项目也踩了：
- **term 之间共享可变对象**——base cfg 的 term 对象/params dict 在多次 `make()` 调用间共享，原地改一个会泄漏到另一个。所以代码里到处是 `deepcopy(...)` 再改（velocity env 的 obs/command 改造每处都先 deepcopy）。
- **函数签名靠约定**——每个 reward/obs 函数必须遵守 `(env, asset_cfg, **params) -> Tensor` 的隐式契约，写错签名要到运行时才炸。
- **符号约定容易错**——reward 函数返回正还是负、weight 配正还是负，框架不管，全靠人遵守（于是有了 [§9.1](#9-工程文化几个体现品味的决策保留案例与推理) 的"双重否定"bug）。

---

## B 组：训练算法

### B.1 什么是 rsl_rl，怎么理解 "RSL"

**RSL = Robotic Systems Lab**（机器人系统实验室），是**苏黎世联邦理工学院（ETH Zürich）**下属的机器人实验室，以腿足机器人研究闻名（ANYmal 四足机器人就出自这里）。

**rsl_rl** 是这个实验室开源的**强化学习算法库**，专门给腿足/人形机器人 locomotion 训练用。它不仿真物理（那是 MuJoCo/Isaac 的事），只负责**RL 算法本身**：PPO 实现、策略网络（actor-critic MLP）、经验回放、obs 归一化、ONNX 导出。

**怎么理解这个命名**：
- `rsl` = 出品方（ETH Robotic Systems Lab）
- `rl` = reinforcement learning（强化学习）
- 合起来就是"RSL 实验室的 RL 库"。

**它为什么流行**：rsl_rl 是**最早把 PPO 大规模用于 GPU 并行腿足训练**的库之一（配合 Isaac Gym/Lab），代码精简、和并行仿真器对接成熟。很多机器人 RL 项目（包括 mjlab）都直接用它或它的衍生版，而不是自己写 PPO。

**本项目的关系**：mjlab 在底层用 rsl_rl 跑 PPO。microduck_rl **不直接调 rsl_rl 的 API**，而是经 mjlab 的 `RslRlOnPolicyRunnerCfg` / `RslRlPpoAlgorithmCfg` 配置类间接使用。唯一直接碰 rsl_rl 内部的地方是 [§B.6](#b6-什么是-monkey-patch猴子补丁) 的 Patch 2——patch 它的 `PPO.compute_returns` 消毒 NaN advantage。

> 一个版本细节：mjlab 1.3.0 迁移到 rsl_rl 5.0.1 后，rsl_rl 内部不再有 `ActorCritic` 类（换成了 `MLPModel` + `EmpiricalNormalization` 的组合），所以本项目原来的 Patch 3（patch ActorCritic）被移除了。这体现了"依赖上游库就要跟着它的重构走"的现实。

---

### B.2 PPO 是什么

**PPO = Proximal Policy Optimization**（近端策略优化），是 OpenAI 2017 年提出的强化学习算法，目前是机器人 locomotion 训练的**默认选择**。

**它要解决的问题**：策略梯度方法（policy gradient）直接用"哪个动作带来高回报就增加它的概率"来更新策略，但**步子迈太大会崩**——一次更新太猛，新策略和旧策略差太远，性能断崖式下跌且无法恢复。

**PPO 的核心思想**："近端（proximal）"= 限制每次更新的幅度，让新策略**不要离旧策略太远**。具体用一个 **clipped surrogate objective**（截断代理目标）：

```text
ratio = π_new(a|s) / π_old(a|s)          # 新旧策略对同一动作的概率比
objective = min( ratio·A,  clip(ratio, 1-ε, 1+ε)·A )   # A 是 advantage，ε 通常 0.2
```

含义：如果某个动作的 advantage 是正的（好动作），想增加它的概率，但 ratio 一旦超过 `1+ε`（概率涨太多），目标就被"截断"不再给奖励——**防止一步更新过猛**。

**为什么机器人训练爱用它**：
1. **稳定**——不像早期策略梯度那样容易崩，适合训几百万步。
2. **样本效率够用**——配合 GPU 并行仿真（4096 envs），on-policy 的"数据用完即弃"缺点被吞吐量弥补。
3. **超参鲁棒**——不太需要精调，一套默认参数能训大多数 locomotion 任务。

**本项目的 PPO 配置**在 `rl_cfg.py` 里，通过 `RslRlPpoAlgorithmCfg` 设置 clip 范围、学习率、entropy 系数等。[§9.2](#9-工程文化几个体现品味的决策保留案例与推理) 提到 run `5yay13u4` 的"熵 10.9→1.9 崩溃"就是 PPO 的 entropy（探索项）塌缩——策略过早收敛到"站着不动"。

---

### B.3 什么是 On-policy PPO

**on-policy（同策略）** 是 RL 算法的一个分类，和 **off-policy（异策略）** 相对，区别在于**"用来更新策略的数据，是不是当前策略自己产生的"**。

| | on-policy | off-policy |
|---|---|---|
| 数据来源 | **必须是当前策略**刚产生的经验 | 可以是**任意策略**（包括旧策略、别人）产生的，存在 replay buffer 里反复用 |
| 代表算法 | PPO、A2C、TRPO | DQN、SAC、DDPG |
| 数据用完 | **即弃**（更新一次后数据就"过期"了） | 可重复采样多次 |
| 样本效率 | 低（要海量新数据） | 高 |
| 稳定性 | 高 | 较难调 |

**为什么 PPO 是 on-policy**：PPO 的 clipped objective 里那个 `ratio = π_new/π_old`，要求 `π_old` 是**刚刚**采集数据的那个策略。一旦你更新了几步，`π_new` 和采集数据的 `π_old` 差太远，ratio 失真，这批数据就不能再用了——所以 PPO 采一批数据、更新几轮（epoch）、就扔掉重采。

**on-policy 的代价与弥补**：on-policy 样本效率低（数据用一次就扔），在 CPU 仿真时代是致命缺点。但**GPU 并行仿真改变了格局**——本项目一次采 4096 envs × 24 步 ≈ 10 万条经验，"海量新数据"变得廉价，于是 on-policy 的稳定性优势压倒样本效率劣势。这就是为什么现代 legged robot RL 几乎清一色 on-policy PPO + GPU 并行。

**本项目的体现**：[§4.1](#41-链路-a训练链路命令--checkpoint) 训练循环里，每 iteration 采 24 步 rollout → 算 advantage → PPO 更新几个 epoch → 丢弃这批数据 → 下一 iteration 重采。`MicroduckOnPolicyRunner` 继承自 mjlab 的 `VelocityOnPolicyRunner`，名字里的 "OnPolicy" 就是指这个。

---

### B.4 什么是 obs normalizer（观测归一化器）

**obs normalizer** 是 rsl_rl 里的一个组件（mjlab 1.3.0 里叫 `EmpiricalNormalization`），作用是**把观测值在线归一化到零均值、单位方差**，再喂给策略网络。

**为什么需要它**：神经网络的输入如果各维度量纲差异大（比如 `base_ang_vel` 是 ±3 rad/s，`joint_pos` 是 ±0.5 rad，`projected_gravity` 是 ±1），网络很难学——大数值维度会主导梯度。归一化把每一维都拉到差不多的尺度，训练更稳更快。

**它怎么工作**（经验归一化）：训练过程中**在线统计**每一维 obs 的 running mean 和 running variance（用 Welford 算法增量更新），然后：

```text
obs_normalized = (obs - running_mean) / sqrt(running_variance + ε)
```

**本项目的关键陷阱——"烘焙"（bake）**：这是 [§4.2](#42-链路-b部署链路checkpoint--真机) 部署链路的核心。normalizer 的 mean/var 是**训练时学到的统计量**，真机部署时也必须用**同一套** mean/var 归一化，否则策略看到的输入分布和训练时不一样，行为全崩。

mjlab 1.3.0 的做法（也是 `export.py` 注释强调的）：导出 ONNX 时，把 normalizer **烘焙进图**——即 ONNX 网络的第一个算子就是 `(obs - mean)/std`，mean/std 作为常量嵌在图里。这样真机只要喂**原始 obs**，ONNX 内部自己归一化，跑的就是训练看到的。

```text
ONNX 图 = normalizer(obs) → actor_MLP(normalized_obs) → action
          └── mean/std 烘焙成常量 ──┘
```

**为什么"never convert by hand"**：`export.py` 注释点破——如果你手动把 checkpoint 转成 ONNX 但忘了烘焙 normalizer，在仿真里 `play` 时**看不出问题**（因为 play 代码自己会应用 normalizer），但一到真机（喂原始 obs 给没烘焙的图）策略就崩。这个 bug 被 in-sim play 掩盖，所以必须走 `run_export()` 这条唯一安全路径。

---

### B.5 什么是 MDP（马尔可夫决策过程）

> 注：用户问题里写作 "mpd"，正确缩写是 **MDP**（Markov Decision Process）。本项目代码里 `mdp.py` 这个文件名正是取自这个概念。

**MDP = Markov Decision Process（马尔可夫决策过程）**，是强化学习的**数学框架**，把"一个智能体在环境里做决策"形式化成一个五元组：

```text
(S, A, P, R, γ)
  S = 状态空间（state）        机器人当前的关节角、速度、朝向……
  A = 动作空间（action）        14 维关节位置增量
  P = 转移概率（transition）    在状态 s 做动作 a 后到 s' 的概率（由物理引擎决定）
  R = 奖励函数（reward）        在 s 做 a 得到多少分
  γ = 折扣因子（discount）      未来奖励打多少折（越远越不值钱）
```

**"马尔可夫"的含义**：下一个状态只取决于**当前状态 + 当前动作**，与更早的历史无关（"无记忆"）。这正是为什么 obs 里要放 `last_action`——因为当前状态本身不含"我上一步做了什么"，要显式喂给策略，让它近似满足马尔可夫性。

**MDP 和本项目代码的对应**：[`mdp.py`](../../../microduck_rl/src/mjlab_microduck/tasks/mdp.py) 这个 7188 行的文件，名字就是 MDP，里面装的全是**构成 MDP 各要素的函数**：

| MDP 要素 | mdp.py 里的函数 | 例子 |
|---|---|---|
| 状态 → 观测（obs） | observation 函数 | `joint_pos_rel`、`projected_gravity`、`base_ang_vel` |
| 奖励 R | reward 函数 | `head_pose_tracking`、`upright`、`foot_slip` |
| 终止（episode 边界） | termination 函数 | `fallen_state`、`robot_state_is_nan` |
| 状态转移的随机性 | event 函数（DR） | `randomize_com`、`push_by_setting_velocity` |
| 命令（给 MDP 加目标） | command 函数 | `UniformVelocityCommand`、`GroundPickPhaseCommand` |

所以 `mdp.py` 可以理解为"**这个机器人 MDP 的所有零件库**"——每个 env cfg 从库里挑零件、配权重，组装出一个具体的 MDP。

**目标**：RL 要学的是一个策略 π(a|s)，最大化期望累积折扣回报 `E[Σ γ^t · R_t]`。PPO（[§B.2](#b2-ppo-是什么)）就是求解这个最优策略的算法。

---

### B.6 什么是 monkey-patch（猴子补丁）

**monkey-patch** 是一种编程技巧：**在运行时动态替换掉一个已存在的函数/方法/类**，而不修改它的源码。名字来自"guerrilla patch"（游击补丁）的谐音讹变，带点"偷偷改"的意味。

**Python 里怎么做**：因为 Python 的函数/方法都是对象，可以在运行时重新赋值：

```python
import some_module
original = some_module.some_function          # 存下原函数
def patched(*args, **kwargs):                 # 写自己的版本
    result = original(*args, **kwargs)        # 通常先调原函数
    return sanitize(result)                   # 再加自己的逻辑
some_module.some_function = patched           # 偷梁换柱
```

之后所有调用 `some_module.some_function` 的地方，跑的都是 `patched`——**包括第三方库内部的调用**。

**为什么用（而非改源码/fork）**：当你依赖一个第三方库（mjlab、rsl_rl），发现它有个 bug 或需要加个保护，但你**不想 fork 维护一份**（升级困难），就可以 monkey-patch——在自己的包导入时替换掉它的方法。代价是"隐式、全局生效、难追踪"，所以要打印日志声明"我 patch 了"。

**本项目的 4 个 patch**（[`mdp.py`](../../../microduck_rl/src/mjlab_microduck/tasks/mdp.py) 顶部，import 时立即生效，并打印 `[mdp] Patches 1-2 active` / `[mdp] Patch 4 active`）：

| Patch | 替换的目标 | 解决什么问题 |
|---|---|---|
| **Patch 1** | `RewardManager.compute` | 用 `torch.nan_to_num` 消毒每步的 reward 结果和 `_episode_sums` 累积器。**在 NaN 进 advantage 计算之前**就拦下，否则 rsl_rl 的 check_nan 会杀掉整个 run |
| **Patch 2** | `rsl_rl.PPO.compute_returns` | 消毒 advantages 和 returns（GAE 的 NaN 会绕过 Patch 1，所以这里补一道） |
| **Patch 3** | ~~`ActorCritic`~~ | **已移除**——mjlab 1.3.0 迁移到 rsl_rl 5.0.1 后没有 `ActorCritic` 类了 |
| **Patch 4** | `get_base_metadata`（ONNX 导出路径） | 过滤掉 `passive_*` 关节，否则导出时 passive 关节会让 joint_names 元数据对不上、KeyError |

**Patch 1 的注释原文**点破了防御链的设计思想：

> "A single NaN from contact-solver instability on rough terrain would otherwise propagate through the reward → advantage → gradient chain and kill the entire training run via rsl_rl's check_nan."

即：粗糙地形上接触 solver 偶尔算出 NaN 力 → NaN 进 reward → 进 advantage → 进梯度 → rsl_rl 的 `check_nan` 检测到就**终止整个训练**。与其让一次偶发 NaN 毁掉几小时的 run，不如**消毒成 0 继续训**。这是"防御性编程"在 RL 训练里的典型应用——配合 [§9.7](#9-工程文化几个体现品味的决策保留案例与推理) 的 critic-only `_safe` obs，构成两道 NaN 防线。

**为什么 patch 要放在 `mdp.py` 顶部、且 `mdp.py` 被所有 env cfg import**：这样无论训哪个任务，只要 import 了 env cfg → import 了 mdp → patch 就生效。这是"全局防御"的布置技巧。

---

## C 组：执行器与部署

### C.1 什么是 Rhoban

**Rhoban** 是一个**法国的机器人研究团队/组织**（与 Bordeaux 大学、Inria 关系密切），专注于人形与腿足机器人的运动控制。它在社区里以两件事闻名：
1. **Reachy / Poppy** 系列机器人项目的参与。
2. **BAM（Better Actuator Models）**——一套高保真舵机模型库，正是本项目执行器建模的核心（见 [§C.2](#c2-什么是-bam什么是高保真舵机模型)）。

**和本项目的关系**：microduck 机器人本身是 [Pollen Robotics](https://pollen-robotics.com/microduck) 出品（Pollen 也是法国波尔多系的机器人公司，和 Rhoban 同处一个生态圈），而它的**舵机物理建模**用了 Rhoban 的 BAM。这是法国机器人圈"硬件团队 + 建模团队"协作的典型。

**为什么要提 Rhoban**：BAM 不是 microduck 自研，是引入的第三方研究成果。理解这一点有助于明白——microduck_rl 的"高保真"不是营销词，而是**站在一个专门研究舵机动力学的团队成果之上**。

---

### C.2 什么是 BAM，什么是高保真舵机模型

**BAM = Better Actuator Models**，是 Rhoban 开源的**舵机（电机+齿轮箱）高保真物理模型库**（PyPI/git 包名 `better-actuator-models`，本项目用它的 `mjlab_frictionloss` 分支）。

#### 先理解"为什么需要高保真舵机模型"

**普通仿真里的舵机**通常被简化成一个理想的"位置伺服"或"力矩源"：你给一个目标角度，仿真里关节就瞬间/平滑地到位，力矩线性正比于误差（`τ = kp·(target - pos) - kd·vel`）。这对很多任务够用。

**但真机舵机远没这么简单**，尤其 microduck 用的 **Dynamixel XL330**（小型串行总线舵机）：
- **back-EMF（反电动势）**：电机转动时会产生反向电压，转速越高、可用力矩越小——这是直流电机的基本物理，理想模型完全忽略。
- **非线性摩擦**：齿轮箱有 **Coulomb 摩擦**（恒定阻力）、**Stribeck 摩擦**（低速时阻力反而大，导致"粘滑"stiction）、**负载相关摩擦**（压得越紧摩擦越大）。这些让真机舵机在低速、轻载时行为和一个理想伺服差很远。
- **电压依赖**：电池电压从 8.2V 掉到 6.5V，舵机能输出的力矩显著下降。
- **控制延迟**：真机固件的 PD 控制有采样/通信延迟，指令和实际力矩之间隔了几拍。

**如果仿真里用理想舵机**：策略在仿真里学会的步态，到真机上因为摩擦/延迟/back-EMF 完全对不上，直接摔——这就是 **sim2real gap（仿真到现实的差距）** 的主要来源之一。

**高保真舵机模型**就是把这些真实物理效应都建模进去，让仿真里的舵机"像真的一样难驱动"，策略在仿真里就被迫学会应对这些非理想特性。

#### BAM 在本项目的配置

[`microduck_constants.py`](../../../microduck_rl/src/mjlab_microduck/robot/microduck_constants.py) 里的 `_BAM_ACTUATOR_KWARGS`：

```python
_BAM_ACTUATOR_KWARGS = dict(
    motor_name="xl330",          # 用 XL330 舵机的实测参数
    model="m6",                  # BAM 的第 6 代模型（最完整的那档）
    target_names_expr=r"^(?!passive_).*",  # 只驱动非 passive 关节（14 个 servo）
    kp_fw=200.0,                 # 固件 PD 的比例增益
    vin_range=(6.5, 8.2),        # 输入电压范围（DR 在此区间随机）
    vin_drop_gain_range=(0.0, 0.2),  # 电压跌落增益
    vin_min=6.0,                 # 最低电压（低于此舵机行为异常）
    delay_min_lag=3, delay_max_lag=6,  # 控制延迟 3-6 拍（DR 随机）
)
```

注意 `target_names_expr=r"^(?!passive_).*"`——这又是 [§6](#6-机器人模型体系8-套-xml--passive_-约定) 的 `passive_` 约定在起作用：BAM 只驱动 14 个真 servo，轮子/齿隙铰这些 passive 关节不施加电压控制。

#### 本项目的三层执行器继承

[`friction_dr_bam.py`](../../../microduck_rl/src/mjlab_microduck/actuator/friction_dr_bam.py) 在 BAM 之上又加了两层：

```text
bam.mjlab.BamActuator（Rhoban 的电压控制 + back-EMF + 摩擦）
   └─ FrictionDRBamActuator
        加 friction_scale(num_envs, 1)：每 env 缩放 BAM 的"速度无关摩擦预算"
        （stiction + 齿轮箱摩擦）。配合 randomize_bam_friction event，
        每 episode 重采样（non-accumulating），模拟每台舵机摩擦个体差异
        └─ BacklashEncoderBamActuator
             固件 PD 读编码器"透过齿隙"：
             cmd.pos = qpos[main] + qpos[backlash]·mask
             （真机磁编码器装在齿隙输出侧，看到的是含齿隙的位置）
             cmd.vel 故意留电机侧（back-EMF/摩擦是转子物理）
```

**这三层体现了 sim2real 的精细程度**：不只是"用 BAM"，还在 BAM 上加了**摩擦域随机化**（让策略对每台舵机的摩擦差异鲁棒）和**齿隙编码器建模**（连"编码器装在齿轮哪一侧"这种机械细节都模拟）。

> 相关的真机硬件背景：microduck 的舵机总线用 **Dynamixel XL330（Protocol 2.0）**，15 个出线关节（左腿 5 + 颈/头/嘴 5 + 右腿 5），其中策略只驱动 14 个（`mouth` 不参与 locomotion）。IMU 是一颗伪装成 Dynamixel 设备（ID 200）的 LSM6DSV16X，和舵机共享同一条 1Mbps 总线——这解释了为什么 obs 里的 `projected_gravity` 和 `base_ang_vel` 来自真机 IMU，而 body_server 要用同一套算术在仿真侧重建它（[§4.3](#43-链路-c反向-sim2realbody_server)）。

---

### C.3 ONNX 是什么

**ONNX = Open Neural Network Exchange**（开放神经网络交换格式），是一个**开放的模型表示标准**。它把训练好的神经网络存成一个 `.onnx` 文件，里面是**框架无关的计算图**（一串算子：MatMul、Add、Relu……）+ 权重。

**它解决什么问题**：训练框架（PyTorch/TensorFlow）和部署环境（机器人板子、手机、浏览器）往往不一致。你不能指望一块 RK3566 嵌入式板子装个完整的 PyTorch 来跑推理——太重。ONNX 提供**中间格式**：

```text
PyTorch 训练 → 导出 .onnx → ONNX Runtime（轻量推理引擎）在目标平台加载执行
```

**核心优势**：
1. **跨框架**——PyTorch 训的，可以用 ONNX Runtime / TensorRT / CoreML 等任意推理引擎跑。
2. **轻量部署**——目标平台只需一个 ONNX Runtime（几 MB），不需要整个训练框架（几百 MB + Python 生态）。
3. **优化**——推理引擎可以对图做算子融合、量化、平台特化加速。
4. **自包含**——权重和计算图都在一个文件里，甚至能把预处理（如 obs normalizer）烘焙进去（见 [§B.4](#b4-什么是-obs-normalizer观测归一化器)）。

**本项目的 ONNX 契约**（[`manifest.py`](../../../microduck_rl/src/mjlab_microduck/publish/manifest.py)）：

```python
OBS_LEN = 61        # 输入必须是 [1, 61]
ACTION_LEN = 14     # 输出必须是 [1, 14]
POLICY_FILE = "policy.onnx"
```

`check_onnx()` 在发布前**强制校验**图的输入输出形状——拒绝 51-D 的老 policy（布局变过）。`smoke_run_onnx()` 进一步用 ONNX Runtime 真跑 50 步，确认输出不是 NaN、不是常数（std≠0）。

**为什么用 ONNX 而非直接传 PyTorch checkpoint**：真机运行时是 **Rust**（[§C.4](#c4-rust-是什么它和-cpython-什么区别和优势)），Rust 生态里加载 PyTorch `.pt` 很别扭（要 libtorch C++ 绑定），而 ONNX Runtime 有成熟的 Rust/C API。所以 `.pt`（训练）→ `.onnx`（部署）的转换是**跨语言边界的必经之路**，这也是为什么 `export.py` 要把 normalizer 烘焙进图——让 Rust 侧只需喂原始 obs，不必自己实现归一化逻辑。

---

### C.4 Rust 是什么，它和 C++、Python 什么区别和优势

**Rust** 是 Mozilla 2010 年起开发、2015 年 1.0 的**系统级编程语言**，主打**内存安全 + 无垃圾回收（GC）+ 高并发**。隔壁的机器人运行时仓库 [`microduck`](../../../microduck/)（robotd、updater、configd 等）就是纯 Rust 写的。

#### 三种语言的定位对比

| 维度 | **Python** | **C++** | **Rust** |
|---|---|---|---|
| 抽象层级 | 高（解释型） | 低（编译型） | 低（编译型） |
| 内存管理 | 自动 GC | 手动（new/delete，易泄漏/悬垂） | **所有权系统**（编译期保证安全，无 GC） |
| 运行速度 | 慢（解释执行） | 快 | 快（与 C++ 同量级） |
| 开发速度 | 快 | 慢 | 中（编译器严格，但拦住大量 bug） |
| 并发安全 | GIL 限制 | 手动加锁，易 data race | **编译期防 data race**（"无畏并发"） |
| 典型用途 | 脚本、ML 训练、原型 | 游戏引擎、高频交易、嵌入式 | 系统软件、WebAssembly、嵌入式、CLI 工具 |

#### Rust 的核心创新：所有权（ownership）

Rust 最大的特点是用**编译期的所有权 + 借用检查（borrow checker）**保证内存安全，而不需要 GC：
- 每个值有唯一的"所有者"，所有者离开作用域时值被自动释放（无泄漏）。
- 你可以"借用"（引用）值，但编译器强制：要么多个不可变借用，要么一个可变借用——**从根上杜绝了悬垂指针、double-free、data race**。

代价是**学习曲线陡**（要和借用检查器"搏斗"），收益是**一旦编译通过，整类内存 bug 就不可能存在**。

#### 为什么 microduck 运行时选 Rust，而训练选 Python

这是一个**按场景选语言**的教科书案例：

**训练侧用 Python（microduck_rl）**：
- ML 生态全在 Python（PyTorch、mjlab、rsl_rl、numpy）。
- 训练是离线、一次性的，慢一点无所谓，开发迭代速度优先。
- 大量调参、实验，Python 的动态灵活正合适。

**运行时侧用 Rust（microduck）**：
- 真机控制是 **50Hz 硬实时循环**——每 20ms 必须算完 obs→policy→action→safety→下发，**错过一拍机器人就可能摔**。Python 的 GC 停顿和解释开销无法保证这种确定性延迟，Rust 可以。
- 嵌入式板子（RK3566）**资源有限**——Rust 编译出的原生二进制小、无运行时依赖、内存占用可控。
- 机器人是**会动到会伤人的物理设备**——Rust 的内存安全 + 并发安全意味着"段错误/data race 导致失控"这类灾难性 bug 在编译期就被拦住。
- 长期运行的守护进程（robotd 开机即跑）不能有内存泄漏——Rust 的所有权系统天然免疫。

#### 两者的边界：ONNX

Python 训出的策略怎么交给 Rust 跑？就是 [§C.3](#c3-onnx-是什么) 的 **ONNX**——Rust 侧用 ONNX Runtime 的 C/Rust 绑定加载 `policy.onnx`，喂 61-D obs（从 Rust 写的舵机总线/IMU 驱动采集），拿 14-D action，经 Rust 写的 safety 层 clamp 后下发给舵机。**Python 和 Rust 通过 ONNX 这个语言无关的格式握手**，各干各最擅长的事。

> 一句话总结这个分工：**Python 负责"学会走路"（离线、灵活、生态丰富），Rust 负责"安全地走起来"（实时、可靠、资源受限）。** 这正是现代机器人系统"训练用 Python、部署用 Rust/C++"的典型架构。

---

## D 组：契约与命令链路

### D.1 61-D 都包含什么（逐维详解）

61 维 actor 观察 = **48 维本体感受（proprioception）+ 13 维命令块（command）**。本体感受是"机器人自己身体现在什么状态"，命令块是"我希望它做什么"。逐维拆解（权威来源 [`symmetry.py`](../../../microduck_rl/src/mjlab_microduck/tasks/symmetry.py) 模块注释）：

#### 第一段：本体感受 48 维

| 索引 | 名称 | 维数 | 物理含义 | 数据来源（真机） |
|---|---|---|---|---|
| `[0:3]` | `base_ang_vel` | 3 | 躯干角速度 (roll, pitch, yaw 三轴)，body-frame | IMU 陀螺仪（LSM6DSV16X） |
| `[3:6]` | `projected_gravity` | 3 | 重力方向在躯干坐标系下的投影 (gx, gy, gz) | IMU 加速度计 → 姿态解算 |
| `[6:20]` | `joint_pos_rel` | 14 | 14 关节当前位置 **相对 default 姿态**的偏差 | 舵机磁编码器 |
| `[20:34]` | `joint_vel_rel` | 14 | 14 关节当前角速度 | 舵机编码器差分 |
| `[34:48]` | `last_action` | 14 | 上一步策略输出的 14 维动作 | 软件状态（上一拍缓存） |

**几个设计要点**：
- **`projected_gravity` 而非原始加速度计**：`USE_PROJECTED_GRAVITY=True`。原始加速度计读数 = 重力 + 运动加速度，混在一起；投影重力是"纯姿态"信息（躯干相对水平面怎么倾斜），更干净。body_server 用 `quat → -rotation.T[:,2]` 在仿真侧重建它，**和真机 IMU 的 SFLP 滤波器同一套算术**。
- **`joint_pos_rel` 是"相对 default"而非绝对角度**：default 姿态就是 [§6](#6-机器人模型体系8-套-xml--passive_-约定) 提到的 `HOME_FRAME`（STAND2 站姿）。用相对值让策略学"偏离标准姿态多少"，比学绝对角度更鲁棒（换机器人零点不用重训）。
- **`last_action` 为什么在 obs 里**：呼应 [§B.5](#b5-什么是-mdp马尔可夫决策过程) 的马尔可夫性——当前物理状态不含"我上一步下了什么指令"，但动作的效果有延迟（BAM 的 3-6 拍 lag），策略需要知道上一步指令才能推断当前真实意图。
- **延迟建模**：`base_ang_vel`/`projected_gravity` 的 `delay_max_lag=1`（曾设 3=60ms，2026-07 audit 发现真机 dxl IMU 其实很快，±20ms，遂降到 1）。

#### 第二段：命令块 13 维

| 索引 | 名称 | 维数 | 含义 |
|---|---|---|---|
| `[48:51]` | `twist` command | 3 | 速度指令 (lin_vel_x 前进, lin_vel_y 侧移, ang_vel_z 转向) |
| `[51:55]` | `head` command | 4 | 头部姿态增量 (neck_pitch, head_pitch, head_yaw, head_roll)，相对 HOME |
| `[55:61]` | `body` command | 6 | 身体姿态增量 (x, y, z, roll, pitch, yaw)，相对 HOME |

**命令块的"零填充约定"**：13 维是**所有任务共用的最大命令空间**，不用的槽填零但保留维度。这样不同任务的 policy 输入形状一致（都是 61），才能热切换。例子：
- 行走任务：`twist` 用满（采样 lin_x∈[-0.4,0.4] 等），`head` 用作主要跟踪目标，`body` 权重=0（DISABLED 但槽活着）。
- 起身任务：`body_pose` 权重抬高、范围扩大，指挥机器人从趴到站。
- ground_pick：相位编码 `[cos2πφ, sin2πφ, 0]` 塞进 twist 槽。
- sitstand：姿态标志 `[sit_flag, 0, 0]` 住在 twist vx 槽。

#### critic 观察（特权信息）

上面是 **actor** 看的 61 维（部署时真机也只有这些）。训练时 **critic** 还额外看"特权信息"（privileged info）——真机拿不到但仿真里有的：球的位姿（BallKick）、脚接触力、脚离地高度、地形高度图等。这叫**非对称 actor-critic**（[§9.8](#9-工程文化几个体现品味的决策保留案例与推理)）：训练时 critic 用特权信息加速学习，部署时只导出 actor（盲于特权），真机照常跑。

---

### D.2 14-D 是什么

**14-D action = 14 个 servo 关节的目标位置增量**。策略网络输出 14 个浮点数，每个对应一个驱动关节，**不是直接的角度，而是"相对 default 姿态的偏移量"**。

**从 action 到物理力矩的链路**：

```text
policy 输出 action[14]（增量，范围约 [-1, 1]）
      │
      ▼  action manager（JointPositionActionCfg）
target_pos = default_joint_pos + action_scale · action[14]
      │     （action_scale 把 [-1,1] 映射到合理的弧度偏移）
      ▼  BAM 执行器（BacklashEncoderBamActuator）
固件 PD：cmd.pos = target_pos（透过齿隙读编码器）
        τ = 电压控制律(back-EMF, Coulomb/Stribeck/负载摩擦, 延迟 3-6 拍)
      │
      ▼  MuJoCo Warp
施加力矩 → mj_step × decimation → 关节运动
```

**为什么是 14 而非 15**：真机有 15 个出线关节（含 `mouth`），但 `mouth`（喙开合）不参与 locomotion，被 `passive_` 约定排除在动作空间外（[§6](#6-机器人模型体系8-套-xml--passive_-约定)）。所以策略只驱动 14 个：左腿 5 + 颈/头 4（neck_pitch/head_pitch/head_yaw/head_roll）+ 右腿 5。

**14 维的关节顺序**（和 obs 里的 joint 块一致，来自 `robot_walk.xml` body tree）：

```text
0: left_hip_yaw    5: neck_pitch    9:  right_hip_yaw
1: left_hip_roll   6: head_pitch    10: right_hip_roll
2: left_hip_pitch  7: head_yaw      11: right_hip_pitch
3: left_knee       8: head_roll     12: right_knee
4: left_ankle                       13: right_ankle
```

**61-D 和 14-D 的关系**：这就是 [§5](#5-核心契约61-d-obs--14-d-action) 说的核心契约——`policy.onnx` 是一个 `[1,61] → [1,14]` 的函数。发布前 `check_onnx()` 强制校验这个形状，`smoke_run_onnx()` 把 14-D action 反馈回 obs 的 `[48-13:48]`（last_action 槽）跑 50 步闭环，确认不发散。

---

### D.3 `uv run train` 的调用链路

> 用户问："uv run train 是内联函数接口，还是自定义？" —— **答案：它是 console script（控制台脚本入口），在 `pyproject.toml` 里自定义声明，但本仓库故意让它和 mjlab 同名并"输掉"命名冲突，实际执行的是 mjlab 的 train 实现。** 下面讲清楚这个绕人的设计。

#### 什么是 console script

`uv run train` 里的 `train` 不是 Python 内置函数，也不是 shell 命令，而是一个 **console script（控制台脚本）**——Python 包安装时根据 `pyproject.toml` 的声明，在 `.venv/bin/` 下生成的一个可执行入口。本仓库的声明：

```toml
[project.scripts]
train = "mjlab_microduck.train_cli:main"     # 故意与 mjlab 撞名
duck-body = "mjlab_microduck.sim.body_server:main"
publish = "mjlab_microduck.publish.cli:main"
```

`uv run train ...` 本质是：uv 激活 `.venv`，然后执行 `.venv/bin/train`，它调用声明里指向的 `main()` 函数。

#### 命名冲突的来龙去脉（保留踩坑）

问题：**mjlab 自己也声明了一个叫 `train` 的 console script**。两个 distribution 声明同名 script，安装时是 **last-writer-wins（后写者赢）**，而 mjlab 赢了——`uv sync` 后 `.venv/bin/train` 实际指向 `mjlab.scripts.train:main`，本仓库的 `train_cli.py` 包装器**根本不会被调用**。

那为什么还要保留这个声明？[`train_cli.py`](../../../microduck_rl/src/mjlab_microduck/train_cli.py) 的注释解释：删掉声明**不会**把名字还给 mjlab——反而会让下次 `uv sync` 按本仓库的 RECORD **卸载** `.venv/bin/train`，而没有东西重装 mjlab 的。于是 `train` 从 venv 里彻底消失，`uv run train` 落到 PATH 上任何叫 `train` 的程序——2026-08-31 就落到了一台机器上 liblinear 的 `train`，它回答 `can't open input file Mjlab-Velocity-Flat-MicroDuck`。

所以结论：**冲突保留，但做成无害**——无论哪个 shim 赢，`train` 行为一致，`--hf-jobs` 标志在唯一一处处理。

#### 完整调用链路

**关键洞察**：`--hf-jobs` 标志**不在 console script 里处理**（因为那个 script 可能根本不被调用），而是从 **mjlab.tasks 插件入口**拦截——这是 mjlab 自己的导入路径，任何安装顺序都夺不走（呼应 [§A.1](#a1-什么是-mjlab-插件包它的作用是什么)）。

```text
uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096 [--hf-jobs]
      │
      ▼
.venv/bin/train  →  mjlab.scripts.train:main（mjlab 赢了命名冲突）
      │
      │  执行 `from mjlab.scripts.train import main` 时，触发 `import mjlab`
      ▼
mjlab/__init__.py（模块作用域）
      │  调用 _import_registered_packages()
      │  → 遍历 "mjlab.tasks" entry-point 组
      ▼
import mjlab_microduck.tasks        ← 本仓库的插件被自动加载
      │
      │  tasks/__init__.py 模块顶层【第一件事】：
      ▼
maybe_submit_to_hf_jobs()           ← train_hook.py
      │
      ├─【分支 A】argv 里有 --hf-jobs 且 argv[0] 是 train 且不在 HF job 内：
      │     from mjlab_microduck.hf_jobs import submit
      │     sys.exit(submit([...]))   ← SystemExit 是 BaseException，
      │                                穿过 loader 的 except Exception，
      │                                直接结束进程，本地训练器永不启动
      │                                （作业提交到 HF 托管 GPU）
      │
      └─【分支 B】没有 --hf-jobs（普通本地训练）：
            函数 no-op 返回，继续执行 tasks/__init__.py 余下部分
            → register_mjlab_task() × 27，建好 task_id → (env_cfg, rl_cfg, runner) 表
      │
      ▼
回到 mjlab.scripts.train:main
      │  tyro 两阶段解析 argv（第一阶段解析出 task_id，第二阶段按 task 解析其余参数）
      │  按 task_id 取出 env_cfg = make_microduck_velocity_env_cfg()
      ▼
构建 ManagerBasedRlEnv → MicroduckOnPolicyRunner → runner.learn()
      │
      ▼
PPO 训练循环（[§4.1](#41-链路-a训练链路命令--checkpoint)）→ model_<iter>.pt
```

#### 为什么这个设计巧妙

1. **拦截点在 mjlab 自己的导入路径上**——`import mjlab` 必然触发插件加载，必然执行 `maybe_submit_to_hf_jobs()`，**任何安装顺序都保证它先于 tyro 解析 argv 运行**。如果放在 console script 里，script 不被调用就失效（这正是 2026-08-31 的 bug：tyro 先报 `Unrecognized options: --hf-jobs`）。
2. **`SystemExit` 穿透 `except Exception`**——loader 用 `except Exception` 包住插件导入（防一个插件坏掉拖垮全部），但 `SystemExit` 继承自 `BaseException` 而非 `Exception`，所以能穿透出去终止进程。这是 Python 异常层级的一个精妙利用。
3. **`MICRODUCK_IN_HF_JOB=1` 防递归**——在 HF job 容器内，`submit()` 会设这个环境变量，让容器里的 `uv run train` 永远走"本地训练"分支，不会再次提交（无限递归）。
4. **逃生舱**——万一拦截失效，`uv run scripts/hf/train_hf.py <task>` 直接调 `submit()`，绕过整个 console script 机制。

> 一句话：`uv run train` 表面是个普通命令，背后是"console script 命名冲突 + 插件入口拦截 + BaseException 穿透"三层机制的协作，全部为了让 `--hf-jobs` 这个标志**无论谁拥有 train 都能生效**。

---

## 结语：把概念串成一条线

回到最开始那张图，现在每个环节都能对应到具体概念了：

```text
【训练】mjlab 插件(A.1) 注册任务 → MuJoCo Warp(A.2/A.3) 在 GPU 并行仿真
        → manager-based 架构(A.7) 组装 MDP(B.5)：obs(61-D, D.1) / action(14-D, D.2) / reward
        → BAM 高保真舵机(C.2) 让仿真接近真机 → terrain(A.4) + DR 增加多样性
        → rsl_rl(B.1) 的 on-policy PPO(B.2/B.3) 训练 → obs normalizer(B.4) 在线归一化
        → checkpoint，NaN 被 monkey-patch(B.6) 消毒保护

【导出】export.py 把 normalizer 烘焙进 ONNX(C.3) → policy.onnx [1,61]→[1,14]

【部署】publish 经 shape gate + smoke run 上传 HF Hub
        → Rust(C.4) 运行时 robotd 加载 ONNX → 50Hz 实时控制真机

【反向】body_server 把 MuJoCo 鸭子经 TCP 喂给真机 Rust 栈，验证整条链路
```

这个项目最值得记住的，不是某个算法或某个框架，而是**它如何把"仿真里学会"到"真机上跑通"之间的每一道缝隙，都用契约、gate、patch 和测试堵住**——61-D 契约堵训练/部署的形状缝隙，ONNX 烘焙堵 normalizer 缝隙，monkey-patch 堵 NaN 缝隙，shape gate 堵版本缝隙，CPU-only 测试堵回归缝隙。

---

## 附录：术语速查表

| 术语 | 全称 / 含义 | 详见 |
|---|---|---|
| **mjlab** | 基于 MuJoCo Warp 的 manager-based 机器人 RL 框架 | [§A.1](#a1-什么是-mjlab-插件包它的作用是什么) |
| **mjlab 插件包** | 经 entry-point 机制挂进 mjlab 的独立包，不改其源码 | [§A.1](#a1-什么是-mjlab-插件包它的作用是什么) |
| **MuJoCo** | DeepMind 的 CPU 刚体动力学仿真引擎 | [§A.2](#a2-什么是-mujoco-warp它和-mujoco-什么区别) |
| **MuJoCo Warp** | MuJoCo 的 GPU 并行重写版，训练用 | [§A.2](#a2-什么是-mujoco-warp它和-mujoco-什么区别) |
| **warp-lang** | NVIDIA 的 Python GPU 计算运行时，MuJoCo Warp 的底层 | [§A.3](#a3-什么是-warp-lang) |
| **terrain** | mjlab 的程序化地形生成系统（plane / generator） | [§A.4](#a4-什么是-terrain地形系统) |
| **velocity 基类** | mjlab 内置的速度命令行走任务模板，microduck 继承改造 | [§A.5](#a5-什么是-velocity-基类) |
| **solver 参数** | 接触求解器的迭代/容差/软硬度，权衡精度-稳定-速度 | [§A.6](#a6-调-solver-参数怎么理解) |
| **manager-based** | mjlab 哲学：env = 一组 manager，每个装可插拔 term | [§A.7](#a7-mjlab-框架哲学manager-based-架构详解) |
| **rsl_rl** | ETH Robotic Systems Lab 的 RL 算法库（PPO 实现） | [§B.1](#b1-什么是-rsl_rl怎么理解-rsl) |
| **PPO** | Proximal Policy Optimization，截断式策略优化算法 | [§B.2](#b2-ppo-是什么) |
| **on-policy** | 用当前策略产生的数据更新（PPO 属此类，数据用完即弃） | [§B.3](#b3-什么是-on-policy-ppo) |
| **obs normalizer** | 观测在线归一化器（EmpiricalNormalization），导出时烘焙进 ONNX | [§B.4](#b4-什么是-obs-normalizer观测归一化器) |
| **MDP** | Markov Decision Process，RL 的数学框架（mdp.py 得名于此） | [§B.5](#b5-什么是-mdp马尔可夫决策过程) |
| **monkey-patch** | 运行时动态替换第三方库的函数，本项目 4 处（NaN 消毒等） | [§B.6](#b6-什么是-monkey-patch猴子补丁) |
| **Rhoban** | 法国机器人研究团队，BAM 的出品方 | [§C.1](#c1-什么是-rhoban) |
| **BAM** | Better Actuator Models，Rhoban 的高保真舵机模型库 | [§C.2](#c2-什么是-bam什么是高保真舵机模型) |
| **高保真舵机模型** | 建模 back-EMF/非线性摩擦/电压依赖/延迟，缩小 sim2real gap | [§C.2](#c2-什么是-bam什么是高保真舵机模型) |
| **ONNX** | Open Neural Network Exchange，框架无关的模型部署格式 | [§C.3](#c3-onnx-是什么) |
| **Rust** | 内存安全 + 无 GC 的系统语言，microduck 运行时用它 | [§C.4](#c4-rust-是什么它和-cpython-什么区别和优势) |
| **61-D obs** | 48 本体感受 + 13 命令块，actor 观察契约 | [§D.1](#d1-61-d-都包含什么逐维详解) |
| **14-D action** | 14 个 servo 关节的目标位置增量 | [§D.2](#d2-14-d-是什么) |
| **console script** | pyproject.toml 声明的命令行入口（`uv run train` 的 train） | [§D.3](#d3-uv-run-train-的调用链路) |
| **passive_ 约定** | 非驱动关节前缀，选择器 `^(?!passive_).*` 自动排除 | [§6](#6-机器人模型体系8-套-xml--passive_-约定) |
| **DR** | Domain Randomization，域随机化（CoM/质量/摩擦/IMU/push 等） | [§4.1](#41-链路-a训练链路命令--checkpoint) |
| **sim2real** | 仿真到现实的迁移，本项目用 BAM/DR/ONNX gate 缩小差距 | [§C.2](#c2-什么是-bam什么是高保真舵机模型) |

---

> 本文为学习梳理笔记，基于仓库代码与注释整理。配套：[`desktop_quickstart.md`](../01-quickstart/desktop_quickstart.md)（部署训练）、[仓库 README](../../../microduck_rl/README.md)、[AGENTS.md](../../../microduck_rl/AGENTS.md)。
