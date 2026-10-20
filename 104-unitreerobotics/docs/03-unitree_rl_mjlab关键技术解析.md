# unitree_rl_mjlab 关键技术解析（初稿 v0.1）

> 定位：[docs/02](02-unitree_rl_mjlab环境搭建与验证.md) 讲「怎么跑」（环境 / 命令 / 判据 / 踩坑），本文讲「跑的是什么」——从源码与本机实测拆解项目用到的关键技术。
> 证据约定：标「实测」的数据来自本机两个训练 run（2026-10-19~20）；标「源码」的可在 `repos/unitree_rl_mjlab` 中直接定位。

## 1. 技术栈总览

| 层 | 技术 | 版本（已验证） | 在本项目中的作用 |
|---|---|---|---|
| 物理仿真 | MuJoCo + MuJoCo Warp | 3.5.0 + 3.5.0 | 同一套 MJCF 模型语义，物理 step 整体在 GPU 上 batch 执行 |
| GPU kernel | warp-lang | 1.12.0 | NVIDIA 的 Python→CUDA kernel 框架，MuJoCo Warp 的执行底座 |
| RL 环境框架 | mjlab | 1.2.0 | IsaacLab 风格的「管理器式」环境组装 + 任务注册 |
| RL 算法 | rsl-rl-lib | 5.0.1 | on-policy PPO（actor-critic MLP + adaptive lr） |
| 任务插件 | unitree velocity / tracking | — | 8 机型速度跟踪 + G1 动作模仿，29 个注册任务 |
| 部署 | ONNX → C++ | — | policy 导出 → unitree_mujoco 仿真回放 → unitree_sdk2 实机（未实测） |

数据流一行图：

```
MJCF 机器人模型 ──► MuJoCo Warp（2048 envs GPU 并行仿真）──► PPO（rsl-rl）──► ONNX ──► C++ 仿真回放 / 实机部署
      ▲                        ▲
   动捕 CSV→NPZ          域随机化 events
```

## 2. MuJoCo Warp：物理仿真整体搬上 GPU

- **是什么**：MuJoCo 官方 GPU 后端。把约束求解、碰撞、积分等物理 step 重写为 warp-lang kernel，模型描述（MJB）与 CPU 版语义对齐——同一份 MJCF，CPU/GPU 两跑。
- **本项目的收益（实测）**：G1 velocity 2048 envs 并行，稳定 **约 87k steps/s**、单 iter 0.52~0.56s；CPU 版 MuJoCo 同规模通常慢一个数量级以上。
- **JIT 编译**：首次运行按模型编译全套 kernel（实测约 30s，单 kernel 最长 10.3s），落盘 `~/.cache/warp/1.12.0`，此后启动直接加载。
- **版本强配对（踩坑实证）**：mujoco-warp 3.5.0 引用的枚举 `mjENBL_MULTICCD` 在 mujoco 3.16 中已被删除——装成 mujoco 3.16 直接 `AttributeError`。warp 生态的 C++ 扩展接口对版本敏感，必须用验证过的配对（详见 docs/02 §4）。

## 3. mjlab：管理器式 RL 环境框架

- **是什么**：IsaacLab 设计思想在 MuJoCo 上的实现。环境不是手写 step/reset，而是组装一份 `ManagerBasedRlEnvCfg`，由各管理器在运行时驱动：

| 管理器 | 职责 | 本项目实例（源码：`src/tasks/velocity/config/g1/env_cfgs.py`） |
|---|---|---|
| scene | 机器人 + 地形 + 传感器 | `get_g1_robot_cfg()` + plane/rough 地形 + ContactSensor/RayCastSensor |
| actions | 动作空间 | `JointPositionAction`（关节位置目标 + G1_ACTION_SCALE 缩放） |
| observations | actor/critic 观测组 | 非对称：critic 额外拿 foot_height 等「特权」观测量 |
| rewards | 奖励项列表 | velocity 16 项组合（§4）；tracking 分层跟踪项（§5） |
| commands | 指令采样 | velocity: 均匀速度指令；tracking: motion 参考帧（三模式，§5） |
| events | 域随机化 / 扰动 | `foot_friction`、`base_com`、`push_robot`（§4） |
| terminations | 终止条件 | velocity: 摔倒 / 超时；tracking: 跟踪误差超阈值（§5） |
| curriculum | 课程学习 | velocity G1-Flat 明确置空 `cfg.curriculum = {}`（全程固定难度） |

- **任务注册机制（源码）**：`src/tasks/__init__.py` 用 `import_packages` 动态导入子包，各机型 `config/<robot>/__init__.py` 调用 `register_mjlab_task` 注册四件套（env_cfg / play_env_cfg / rl_cfg / runner_cls）。无 entry-point，train/play 须在仓库根目录执行。
- **插件复用模式**：Unitree 任务基于 mjlab 内置 velocity 模板（`make_velocity_env_cfg()`）做机型覆盖——换机器人 = 换 entity 配置 + 微调 reward/events 参数，任务骨架不重写。
- **play 模式机制**：`play=True` 时同一份 cfg 打补丁——关观测噪声（`enable_corruption=False`）、去推扰（pop `push_robot`）、episode 无限长、指令范围收窄。训练/回放行为差异全部显式可见。

## 4. velocity 任务：指令速度跟踪（G1 实测通过）

**任务定义**：每个 env 独立采样速度指令 (vx, vy, ωz)，奖励机器人基座实际速度贴合指令、同时保持人形步态稳定。

**reward 组合术（实测 16 项，五类分工）**：

| 类别 | 项 | 作用 |
|---|---|---|
| 任务 | track_linear_velocity、track_angular_velocity | 主目标：速度贴合（exp 衰减误差） |
| 姿态 | body_orientation_l2、pose | 躯干竖直 + 分关节参考姿态（见下） |
| 能量/平滑 | action_rate_l2、joint_acc_l2、joint_pos_limits、angular_momentum | 压动作抖动、限关节加速度、防角动量发散 |
| 步态塑形 | foot_gait、foot_clearance、foot_slip、soft_landing、stand_still | 摆动/支撑时序、抬脚高度、防打滑、轻着陆 |
| 安全 | self_collisions、is_terminated | 自碰惩罚（力阈值 10N）、摔倒重罚 |

**分关节步态塑形（源码注释即设计文档，`env_cfgs.py` L86-132）**：`pose` 项对每个关节给独立的参考姿态 std——髋 pitch/膝 0.5（松，允许自然迈腿）、踝 roll 0.1（紧，保平衡）、腰 0.1~0.15（紧，躯干稳）、肩/肘 0.1~0.25（中等，自然摆臂）、腕最松。**松紧即权限**：想让策略自由发挥的关节给大 std，必须严格的给小 std。还有 std_standing/std_walking/std_running 三档随速度指令切换。

**域随机化（events，Sim-to-Real 核心手段）**：`foot_friction`（脚底 14 个 collision 几何摩擦随机）、`base_com`（torso 质心扰动）、`push_robot`（推扰，play 模式移除）；配合观测噪声（`enable_corruption`）。让策略在训练时就见过「物理参数漂移」的世界。

**Flat 与 Rough 的差异**：Rough 启用地形课程（`terrain_generator.curriculum = True`）+ `terrain_scan` 射线传感器（pelvis 高度扫描进观测）；Flat 是纯平面并删掉 height_scan 观测与地形课程——先用 Flat 跑通链路，再上 Rough 练感知。

## 5. tracking 任务：动作模仿（G1 舞蹈实测通过）

**动捕管线（实测）**：CMU 格式动捕 CSV（30fps，3945 帧）→ `csv_to_npz.py` 插值重采样为 50fps NPZ（与仿真控制频率对齐）→ 12MB 参考动作。

**分层跟踪设计（源码：`src/tasks/tracking/mdp/`）**：奖励与终止都分两层——
- **anchor 层（全局根）**：`motion_global_root_pos/ori`——根位置/朝向贴参考轨迹（决定「跳到哪」）；
- **body 层（全身相对）**：`motion_body_pos/ori/lin_vel/ang_vel`——全身关键 body 相对根的位姿速度（决定「动作像不像」）。
- 对称的 error_* 指标族（error_anchor_*/error_body_*/error_joint_*）直接进 tensorboard，训练健康度可视化。

**motion 采样三模式（`mdp/commands.py` `_resample_command`，注意！G1 dance 用的是 start）**：

| 模式 | 行为 | 适用 |
|---|---|---|
| `start`（G1 dance 实际配置） | 每次 episode 从动作第 0 帧开始 | 完整表演一段（如舞蹈） |
| `uniform` | 全时间轴均匀随机取起始帧 | 泛化到任意片段 |
| `adaptive` | **失败感知采样**（内置未启用）：动作时间轴分 bin，episode 在哪个 bin 摔倒该 bin 失败计数 +1，采样概率 ∝ 失败计数 + 均匀先验，再经 1D 卷积平滑相邻 bin——「哪里摔倒就在哪多练」；熵指标 sampling_entropy 监控聚焦程度 | 长动作、高难度任务 |

实测教训：之前误以为 dance 训练启用了自适应课程，源码核实是 `start`——**读配置别猜**。

**终止阈值**：`anchor_pos` / `anchor_ori` / `ee_body_pos` 超阈值即终止（实测：ee_body_pos 终止率从开局 400/iter 降到 0.29/iter——阈值构成「够不到参考动作就重来」的硬约束）。

## 6. PPO 训练（rsl-rl）与曲线诊断方法论

**配置（实测 run 的 params/agent.yaml）**：actor-critic MLP 512-256-128 + elu；lr 1e-3 起、adaptive 调度（desired_kl=0.01——每轮 update 后按实际 KL 与目标的比值缩放 lr，KL 超标降 lr、不足升 lr）；观测在线归一化；高斯策略 std 可学习（实测 1.0 → 0.3，探索收敛的直观信号）。

**两个实测案例（同一套算法，两种命运）**：

| | velocity G1-Flat | tracking dance |
|---|---|---|
| 曲线形状 | iter 5000 见顶（39.4）后**断崖塌陷**，仅恢复七成 | **单调爬升** 30000 iters，峰值在 iter 29079 |
| 课程 | 无（curriculum 置空） | 无（start 模式；adaptive 未启用） |
| 诊断结论 | PPO adaptive lr 长跑失稳，后半程白烧 | 任务难、梯度足，长训有效 |

**诊断方法论（两次实战验证）**：reward 断崖先查两件事再归因——①源码核实是否有 curriculum/命令范围变化（有 → 任务变难是正常台阶）；②无课程则断崖+部分恢复指向策略失稳 → **回退到峰值 checkpoint 而非用末位**。两个 run 的最佳 checkpoint（5000 / 29000）都不在终点，「末 checkpoint ≠ 最佳 checkpoint」在本仓库已两次实证。

## 7. Sim-to-Real 部署链（源码调研，未实测）

```
train.py（GPU 训练）
  └─ save 时自动导出 ONNX ──► simulate/（unitree_mujoco：C++ 仿真回放验证）
                                  └─ deploy/（unitree_sdk2：真机 G1 部署）
```

- velocity 导出 `policy.onnx`（858KB 纯网络）；tracking 导出 `<时间戳>.onnx`（约 7MB，**motion 参考数据被 register_buffer 打包进 ONNX**——部署侧无需单独分发动作文件，这个设计值得学习）。
- **坑（实测发现）**：ONNX 是训练收尾时用**末位权重**导出的；若最佳 checkpoint 不是末位（两个 run 都不是），部署前需从峰值 .pt 重新导出。
- **No-State-Estimation 变体的意义**：真机上拿不到仿真里的真值速度/位置等状态，该变体的观测只含本体传感器可得的量（IMU、关节编码器等），是「策略能否上真机」的关键分水岭；State-Estimation 变体则供仿真内研究。

## 8. 本机实测数据速查

| 指标 | velocity G1-Flat | tracking dance |
|---|---|---|
| 规模 | 2048 envs × 10001 iters | 2048 envs × 30000 iters |
| 耗时 | 94 分钟（0.56s/iter） | 4.35 小时（0.52s/iter） |
| 吞吐 | ~87k steps/s | ~78k steps/s（推算） |
| mean_reward | -0.93 → 27.45（峰 39.4 @4965） | -2.0 → 26.2（峰 @29079） |
| 最佳 checkpoint | model_5000 | model_29000 |
| 关键健康信号 | episode 拉满 1000 步、fell_over 归零 | ee_body_pos 终止 400→0.29、episode 462/500 |
| 遗留问题 | 5000→6000 塌陷未根治（可收紧 desired_kl 或缩 max_iterations） | ONNX 为末位权重，部署需重导 |

---

> 初稿说明：§2/§3/§4/§5/§6 均经源码定位与本机实测双重验证；§7 仅源码调研。结构按「总览 → 仿真 → 框架 → 两类任务 → 算法 → 部署 → 数据」组织，待复核后精简。
