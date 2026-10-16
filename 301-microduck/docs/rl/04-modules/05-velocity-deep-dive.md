# velocity 精讲 —— 逐段拆解 `microduck_velocity_env_cfg.py`

> 这是 `04-modules/` 模块击破系列的 **#5**，也是整条学习脊柱的本体。
> 目标：把 [`microduck_velocity_env_cfg.py`](../../../microduck_rl/src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py)（950 行）**逐段拆开**，让你对照源码精读，读完就懂"一只鸭子走路这件事，是怎么被翻译成一堆 reward / obs / event / curriculum 的"。

**怎么用这份文档**

- 左边开这份文档，右边开源码，按节对照读。每节都标了**源码行号**。
- 每段讲三件事：**配了什么** → **为什么这么配** → **真实案例**（注释里的 run 编号、调参演化、踩坑复盘全部保留——这是这个仓库最值钱的部分）。
- 遇到不懂的 RL 术语，查 [`architecture_and_glossary.md`](../02-architecture/architecture_and_glossary.md) 第二部分（A–F 六组概念词典）。
- 想看"一条命令跑起来的全链路"，读 [`training_walkthrough.md`](../03-walkthrough/training_walkthrough.md)。

**为什么 velocity 是脊柱**

- README 称它是 "The main locomotion task"，是其余 20+ 个任务的**基类 / 配方源头**——BallKick 的注释就写着它的 DR 配方 `copied from the standup env (which is itself matched to velocity — the recipe with proven transfer)`。
- 吃透 velocity = 一次掌握所有 microduck 任务的骨架 + 那套"验证过能迁移到真机"的配方。

---

## 第 0 节 · 30 秒地图：这个文件长什么样

950 行，五大块：

```
L1-17     docstring   任务"需求说明书"（作者留给你的"为什么"）
L19-88    模块级常量   所有可调旋钮 + DR 开关总表
L89-122   imports     关键：从上游 mjlab 基类 make_velocity_env_cfg 起步
L125-190  地形定义     MICRODUCK_ROUGH_TERRAINS_CFG + 接触软化 spec_fn
L193-909  make_microduck_velocity_env_cfg()   主体，组装一个完整 env
            ├ L199-273  std字典 / 传感器 / 基类起步 / robot / action
            ├ L275-359  REWARDS（+ L703-745 头姿 reward）
            ├ L361-529  EVENTS + TERMINATIONS + DR 事件
            ├ L531-635  OBSERVATIONS
            ├ L637-701  COMMANDS
            ├ L747-774  TERRAIN（flat / rough 分支）
            └ L776-907  CURRICULUM
L912-949  MicroduckRlCfg   网络结构 + PPO 超参（"怎么训"）
```

**一句话本质：**

> velocity env = **上游 mjlab 的 `make_velocity_env_cfg()` 基类**（L263 起步）+ **microduck 的"改 / 删 / 加"定制**（L263 之后全部）。

**学会"改/删/加"读法**（读任何继承式配置的钥匙）：

| 你看到的代码 | 含义 |
|---|---|
| `cfg.rewards["pose"].weight = 1.0` | **改** —— `pose` 是基类已有 term，microduck 只调参数 |
| `del cfg.observations["actor"].terms["base_lin_vel"]` | **删** —— 基类给了，但 microduck 不要（真机拿不到） |
| `cfg.rewards["head_pose_tracking"] = RewardTermCfg(...)` | **加** —— 基类没有，microduck 新增 |

用这三招，你**不用读 mjlab 基类源码**，就能反推出"基类给了什么、microduck 动了什么"。本文末尾【附录】有一张完整的**改/删/加速查表**。

---

## 第 1 节 · docstring：任务的"需求说明书"（L1-17）

velocity = **速度命令跟踪 + 头姿命令**。reward 是 locomotion-focused（lean tracking + gait/feet，action-rate 由 curriculum 逐步收紧）。docstring 里作者列了 5 个"你一定会问为什么"的设计决策：

1. **`foot_slip` 保持 -0.1（故意弱）** —— 更强的值对这只机器人"以枢轴转为主"（pivot-heavy）的转向太限制。
2. **固定、适度的命令范围**（`ang ±1.0` 让转向可学），**而非**一个不断加宽、最终超出机器人能力的 curriculum。
3. **turn-in-place（原地转）**：15% 的 env 专门拿 `lin=0 + |ang| ∈ [0.4, 1.0]`。原因来自 2026-07 审计：如果只做独立均匀采样，"原地旋转"只占 ~2% 的数据 → 几乎没被训练到。
4. **`head_pose_tracking` 作为主目标**，外加一个基于 EMA 的 `head_pose_bias` 惩罚，**只对"可逃逸的 DC 头下垂"定价**（第 5 节详解，这是全文最精彩的一处）。
5. **`body_pose` 跟踪的基础设施保留但 DISABLED（weight 0）**，好让 obs 槽位对那些真正用它的 env（如 standup）保持活跃。

> **读法教学**：读任何 `*_env_cfg.py`，**先读 docstring**。它是作者的"设计意图声明"，后面每一段代码都是它的展开。带着这 5 条去读，你会不断看到"哦，这就是第 3 条说的 turn-in-place"。

---

## 第 2 节 · 模块级常量与 DR 开关（L19-88）

这一段是**全文件的"旋钮总表"**。分三组看。

### 2.1 训练节奏常量（L22-28）

| 常量 | 值 | 作用 |
|---|---|---|
| `NUM_STEPS_PER_ENV` | 24 | 每个 rollout 采 24 步再更新一次网络（串到 rl_cfg 的 `num_steps_per_env=24`）。**curriculum 的 step 单位 = iter × 24**（第 9 节会反复用到） |
| `TURN_IN_PLACE_FRACTION` | 0.15 | 15% env 专练原地转（对应 docstring 第 3 条） |
| `ENABLE_SYMMETRY` | False | 走路任务不开对称数据增强（对称=左右镜像，见 architecture B 组；BallKick 单脚任务也关） |

### 2.2 域随机化（DR）开关（L30-42）

DR = Domain Randomization，**sim2real 的核心手段**：训练时故意把物理参数随机化，逼策略学得鲁棒，这样真机的"参数偏差"就被覆盖在训练分布里了。

| 开关 | 状态 | 备注 |
|---|---|---|
| `ENABLE_COM_RANDOMIZATION` | ✅ | 躯干质心随机 |
| `ENABLE_HEAD_COM_RANDOMIZATION` | ✅ | 头组件质心随机 |
| `ENABLE_KP_RANDOMIZATION` / `KD` | ❌ | 注释标 `Was True`——曾经开过，后来关 |
| `ENABLE_MASS_INERTIA_RANDOMIZATION` | ✅ | 质量+惯量 |
| `ENABLE_JOINT_FRICTION_RANDOMIZATION` | ✅ | 走 BAM 专属路径（第 6 节） |
| `ENABLE_ARMATURE_RANDOMIZATION` | ✅ | 反射转子惯量 |
| `ENABLE_VELOCITY_PUSHES` | ✅ | 随机推力扰动 |
| `ENABLE_IMU_ORIENTATION_RANDOMIZATION` | ✅ | 模拟 IMU 安装误差 |
| `ENABLE_ENCODER_BIAS` | ✅ | 每 env 固定关节编码器偏置 |
| `ENABLE_BASE_ORIENTATION_RANDOMIZATION` | ❌ | 初始倾斜 |

### 2.3 DR 范围 + 真实调参案例（L55-88）

- **`VELOCITY_PUSH_RANGE` = ±0.3（原 ±0.5）** ← **重点案例**（L80-83）：
  > 2026-07 审计发现：一个**大于最大步行速度(0.4)的加性推力**每 3-6 秒踢一次，会训练出一种**永久紧张的"防摔恢复"步态**。改成 ±0.3 后，既保留了抗推鲁棒性，又让"更平静的步态"成为最优解。

  这是一个典型的"过度随机化反而毁掉步态"的教训——DR 不是越强越好。

- **`IMU_ORIENTATION_RANDOMIZATION_ANGLE` = 6°**（L84）：注意是**零中心、随机轴**——训练的是对"失准**幅度**"的容忍，**不是** pitch 偏置。真板上系统性的 ~5° pitch 偏移是在 runtime 源头修正的（`imu-pitch-offset`），不在这里。
- **`ENCODER_BIAS_RANGE` = ±0.015**（≈ ±0.86°/关节，每 env 恒定）。
- **`HEAD_BODY_NAMES`**（L66-72）里藏着一个**诚实的踩坑注释**：`bearing_roll` 其实**不是**头部 body（它是右 hip-yaw 连杆），"一直被错误地列在这里，仅为保留既有 DR 行为而留着"。

> **读法教学**：常量区的注释里，凡是 `was X → now Y` + 日期审计（2026-07 / 2026-08）的，都是**调参史的化石层**。别跳过——它们告诉你"这个值不是拍脑袋定的，是撞过墙改出来的"。

---

## 第 3 节 · 地形（L125-190）

⚠️ **只有 `rough=True` 才会用到这一节**；flat 训练（默认）的地形就是一个 `plane`（见 L748-750）。

### 3.1 `MICRODUCK_ROUGH_TERRAINS_CFG`（L125-169）

比 mjlab 默认的 rough terrain **温和得多**——因为这只机器人**只能把脚抬起 ~1-2 cm**，所以台阶高度封顶 1.5 cm。四种子地形：

| 子地形 | proportion | 参数 |
|---|---|---|
| `flat` | 0.25 | 平地 |
| `pyramid_stairs` | 0.25 | 台阶 ≤ **1.5 cm**（默认是 10 cm） |
| `random_grid` | 0.30 | 随机凹凸 ≤ 1 cm（鹅卵石感） |
| `pyramid_slope` | 0.20 | 缓坡 1.7°→5.7° |

这里有两个**用 GPU 和崩溃换来的踩坑**（原文注释保留）：

- **踩坑 1（OOM）**：`random_grid` 的 `grid_width` 若设 0.12，在 8m patch 上 = 66×66 = 4356 boxes/patch → 总计 ~261K → **显存 OOM**。改成 **0.45** → 17×17 = 289 boxes/patch → ~17K。且 0.45 必须**不能整除** terrain size(8.0)：0.45 × 17 = 7.65 ✓（留出 border）。
- **踩坑 2（穿地）**：`BoxInvertedPyramidStairsTerrainCfg` 被**移除**——它会把 `env_origin_z` 设到坑底（负值），导致 reset 时 `root_z = 0.12 + env_origin_z ≈ −0.10 m`，机器人被放在坑地板**下方** → 直接穿过地面坠落。

### 3.2 `_soften_terrain_contacts`（L172-190，一个 spec_fn）

box 地形把相邻 geom 放在不同高度，高度突变处的**硬边**会让脚落上去时接触法线不稳定 → MuJoCo 求解器产生**脉冲式 NaN 力**。解法：把地形 body 里每个 geom 的 `solref` 时间常数 **0.02 → 0.04**（接触弹簧软 2×），足以 damp 掉不稳定，又不明显改变宏观走路物理。

> **读法教学**：地形是 NaN 的高发区。这一节几乎每个数字背后都有一次"训练跑到一半 NaN 崩了"的经历。

---

## 第 4 节 · make 函数骨架：从基类起步（L193-273）

函数签名 `make_microduck_velocity_env_cfg(play=False, rough=False)`：`play`=可视化推理模式，`rough`=崎岖地形。L263 是**分水岭**——之前是"备料"，之后是"在基类上覆盖"。

### 4.1 备料：两套 pose 标准差字典（L199-215）

`pose` reward 需要"每个关节该有多贴近目标"的标准差。velocity 给了**两套**：命令≈0 时用 `std_standing`（更紧，把机器人锁在 HOME），走路时用 `std_walking`（更松，允许摆动）。

- **案例**：`hip_roll` 的 std 走了 `0.1 → 0.06 → 0.05` 的下调史——目的是**保持 5° 内八的站姿**（让脚掌平贴地面 sole sits flat），阻止腿"外撇到竖直"（stop leg splay）。

### 4.2 备料：三个传感器（L217-260）

| 传感器 | 作用 |
|---|---|
| `feet_ground_cfg` | 脚-地接触（**严格 LEFT、RIGHT 顺序**，geom pattern 匹配 `left/right_foot_collision`，reduce=netforce，`track_air_time=True`） |
| `self_collision_cfg` | trunk_base 子树自撞检测 |
| `foot_height_scan_cfg` | **mjlab 1.3.0 新机制**：`foot_height` obs 和 `foot_clearance/foot_swing_height` reward 现在由每脚的**地形高度射线传感器**驱动（旧版基于 site_pos），对标 microban 的 `foot_height_scan` |

### 4.3 分水岭：`cfg = make_velocity_env_cfg()`（L263）

这一行从**上游 mjlab 基类**拿到一个"标准 velocity env"的完整配置（含 `pose / upright / air_time / track_*_velocity / foot_* / action_rate_l2` 等一整套 locomotion term 的骨架）。**后面 640 行全是在这个 `cfg` 上做"改/删/加"。**

### 4.4 robot 与 action（L266-273）

- `cfg.scene.entities = {"robot": MICRODUCK_WALK_ROBOT_CFG}` —— 塞进 microduck 走路机器人（见 [`../04-modules/README.md`](../README.md) 的 robot 模块）。
- `cfg.viewer.body_name = "trunk_base"` —— 可视化时相机跟随躯干。
- action：`joint_pos.scale = 1.0` —— 动作就是**目标关节角**（scale=1.0 即绝对角度，不是缩放增量）。actuator 细节见 architecture C 组。

---

## 第 5 节 · REWARDS：鸭子为什么要这么走（L275-359 + L703-745）

**这是全文的重头戏**。reward 是 RL 的"灵魂"——它定义了"什么是好的走路"。velocity 的每一个 weight 都是调出来的。

> **符号约定**（贯穿全仓库）：**正 = 奖励，负 = 惩罚**。一条铁律是"每个 penalty 项算出来必须 ≤ 0"（见 [`desktop_quickstart.md`](../01-quickstart/desktop_quickstart.md) §8.2），否则惩罚会变成奖励。

我把 reward 按**功能**分成 5 组（源码里是交错出现的，这里重新归类方便理解）：

### (a) 姿态组：站直、别乱晃

- **`pose`（L277-289，weight 1.0）**：**只作用于腿关节**——`asset_cfg` 用正则 `^(?!passive_|.*neck.*|.*head.*).*` 明确**排除了 neck/head**。
  > **为什么排除头？**（L280-284）头/颈是由 `head_pose_tracking` **命令驱动**的。如果头也进 `pose` reward，就会出现"`pose` 把头拉向 HOME、而 `head_pose_tracking` 把头拉向命令"的打架；当命令很大时 `head_pose_tracking` 的梯度衰减，`pose` reward 占上风 → 策略学会**"干脆忽略头姿命令"**。所以头必须从 `pose` 里摘出去。

- **`upright`（L292-301，weight 2.0，std²=0.05）**：作用于 `trunk_base`，奖励躯干竖直。
  > **案例**：weight `1.0 → 2.0`、std² `0.1 → 0.05`。2026-07 的 pitch-vs-speed 评测发现：策略走路时带着 **+2–4° 的稳态前倾**（p90 ~6-8°），且 ~2/3 的"高速下被推倒"是**向前摔**。在 weight 1.0 / std² 0.1 时，4° 前倾每步只花 ~0.05 → **几乎免费**；提到 2.0 / 0.05 后每步花 ~0.19 → 有足够的梯度把躯干"扶正"，同时瞬态前倾（推倒恢复、加速）仍负担得起。

### (b) 步态组：抬脚、别拖、别滑

- **`air_time`（L330-336，weight 3.0）**：奖励"脚在空中停留时长"落在窗口 `[0.125, 0.300]s` 内 → 逼出正常迈步。
  > **为什么"零命令时该站着不动"不用一个显式的 stillness 惩罚？** 因为它是靠 `standing_envs` **curriculum** 教的（到 iter 2000 有 25% 的 env 是"站立 env"，见第 9 节），而不是靠一个"别迈步"的 term。

- **`foot_clearance`（L351-352）/ `foot_swing_height`（L354-355）**：`target_height` 都从 **0.01 → 0.02**——前者惩罚"拖脚"（dragging），后者强制"把脚抬起来"。mjlab 1.3.0 里 `foot_swing_height` 完全由传感器驱动（无 asset_cfg）。
- **`foot_slip`（L306-315，weight -0.1）**：**故意弱**（不是 -1.0）——对应 docstring 第 1 条：-1.0 对"枢轴转向"太限制。

### (c) 速度跟踪组：velocity 的核心目标 ⭐

- **`track_linear_velocity`（L342-343，weight 2.0，std²=0.1）**
- **`track_angular_velocity`（L344-345，weight 2.0，std²=0.5）** —— 转向的 std 更松（0.5 vs 0.1），因为精确跟踪角速度比线速度难。

这两项是"velocity"这个任务名字的由来：**跟上给定的 (vx, vy, ωz) 速度命令**。

### (d) 平滑 / 正则组：别抖、别自撞

| term | weight | 作用 |
|---|---|---|
| `action_rate_l2`（L347-349） | -0.1（stage-0） | 动作平滑；由 `action_rate_weight` curriculum **ramp 到 -1.0**（第 9 节）。它对**所有** action 维（含 neck）求和 |
| `body_ang_vel`（L310, 338） | -0.05 | 抑制躯干乱转 |
| `angular_momentum`（L339） | -0.02 | 抑制角动量 |
| `self_collisions`（L323-327，**microduck 新增**） | -1.0 |  discourage 腿撞进躯干电池仓；注释坦承"有关节限位其实够不到身体，但一个正信号能让它保持远离" |
| `soft_landing`（L317） | —— | **被 `pop` 删除**（基类有，microduck 不要） |

### (e) 头姿组（microduck 新增，L703-745）—— 全文最精彩

- **`head_pose_tracking`（L710-714，weight 2.0，std=0.5）**：跟踪 4 个头姿命令。
  > **为什么 std=0.5？**（L704-709）在 ±1.0 rad 的满命令下，一个"完全不跟踪"的策略仍能拿到 per-joint reward `exp(-(1/0.5)²)=exp(-4)≈0.018`——**小但非零的梯度**，所以 curriculum 把命令范围加宽时**不会杀死学习信号**。最终 reward 是 4 个关节的**均值** → 部分跟踪 = 部分奖励（没有 all-or-nothing 的悬崖）。

- **`body_pose_tracking_6d`（L717-727，weight 0.0）**：**DISABLED**——基础设施完整保留，但权重为 0，只为让 obs 槽位和 command 对那些"会 raise weight"的 env（standup）保持活跃。

- **`head_pose_bias`（L729-745，weight 0.0，由 curriculum ramp）**：⭐ **这是整个 velocity 最深刻的一课**，原文复盘完整保留：

  > **现象**：头在走路时会**下垂 ~15°**（实测 run `ww1g2198`：`head_pose_tracking` = 1.544/2.0 → 14.6° 平均关节误差）。
  >
  > **失败的修法**：run `5yay13u4` 试着**收紧** `head_pose_tracking` 的 `fine_std=0.1` → 策略在 **iter 300 就完全停止走路**（`air_time` 1.01 → 0.02，峰值脚高 15mm → 2mm，熵 entropy 崩塌 10.9 → 1.9）。
  >
  > **为什么失败**：一个**瞬时的紧容差**每步向走路收税 **0.77/step = 整个 air_time 奖励的 76%**，而且是 **UNESCAPABLE（不可逃逸）**——一个 280g 的头（占机器人体重 38%）在迈步时**物理上必须 oscillate**。于是"站着不动"得分更高 → 策略就选择站着不动。
  >
  > **正确的修法**：DC 偏置（不同于振荡）是 **IS escapable（可逃逸）** 的——把颈部命令**向上 bias** 就能抵消重力下垂。所以**只对那部分定价**：对误差的 **1 秒 EMA 做 L1 惩罚**。在最优解处，这对一个走路的策略**花费为 0**。

> **读法教学（务必记住这一课）**：**惩罚只能对"策略能改变的东西"定价。** 一个"看起来合理"的紧容差，如果它惩罚的是策略**无法逃逸**的物理必然（走路时头必须晃），策略就会用"躺平"来逃避——你会看到它学会站着不动，而不是走得更好。这个 `head_pose_bias` 的 EMA 设计，就是"只罚可逃逸的 DC 分量"的教科书级例子。

---


## 第 6 节 · EVENTS + TERMINATIONS：每步 / 每 reset 发生什么（L361-529）

**event** = "在特定时机注入变化"的钩子，时机由 `mode` 决定：`startup`（开局一次）、`reset`（每个 episode 重置时）、`interval`（每隔几秒）。DR 和扰动都靠 event 实现。

### 6.1 基础 event（L364-406）

| event | mode | 作用 |
|---|---|---|
| `expand_bam_friction_fields`（**加**） | startup | BAM（`mjlab_frictionloss` 分支）每步写 per-env 的 `dof_frictionloss/dof_damping`；这个 **no-op event 只为把那些字段注册进来**，以便 per-world 展开 |
| `reset_action_history`（**加**） | reset | 重置动作历史缓冲 |
| `foot_friction`（**改**） | reset | 脚底摩擦范围 `(0.7, 1.3)`——注释："Grippier footpad"，从 `(0.3, 1.2)` **收窄**而来 |
| `reset_base`（**改**） | reset | 重置高度 `z ∈ (0.12, 0.13)` |
| `push_robot`（**加**，条件） | interval | 每 3-6s（play 模式 0.5-1s 更频繁便于观察）给一个 ±0.3 m/s 的速度推力 |

### 6.2 `nan_state` 终止（L382-386，microduck 新增）⭐

```python
cfg.terminations["nan_state"] = TerminationTermCfg(
    func=microduck_mdp.robot_state_is_nan, time_out=False,
    params={"sensor_names": (feet_ground_cfg.name,)})
```

> **为什么**：MuJoCo 在极端接触脉冲下会产生 **NaN 关节位置**。立即终止 → reset 回有效状态，**赶在 NaN 传播进 obs buffer、损坏网络权重之前**。这是一道"数值防火墙"。

### 6.3 DR 事件（L408-529）—— 以及一个致命陷阱：BAM 让 stock DR 变 no-op

这一节最重要的一课：**升级框架 / 换了 actuator 模型后，很多"标准 DR 操作"会静默失效（silent no-op）**。原文注释保留了两个血泪案例：

- **`randomize_mass_inertia`（L452-469）**：用 `dr.pseudo_inertia`（startup 模式，`alpha` 从 ±5% 反推 `e^(2α)∈[0.95,1.05]`，同时缩放 mass 和 inertia 且 CoM 不变，故不与 `randomize_com` 冲突）。
  > **它替换了旧的 custom `randomize_mass_and_inertia`——那个在 mjlab 1.3.0 下是 silent no-op**（直接写 per-env `body_mass/body_inertia` 不会被展开，会 collapse 成单个共享值）。**你以为在随机化质量，其实根本没随机**——这种 bug 极难发现。

- **`randomize_joint_friction`（L471-483）**：走 **BAM-native 路径** `microduck_mdp.randomize_bam_friction`。
  > **为什么不用 stock `dr.dof_frictionloss`？** 因为在 BAM 下 MuJoCo 的 `dof_frictionloss` 被 **zeroed**，stock 操作是 no-op。必须走 `FrictionDRBamActuator.friction_scale` 钩子。

- 同理 `randomize_joint_damping`（L485-498，本就关闭）注释也写明"BAM 下 no-op"；而 `randomize_armature`（L500-512）注释强调 **`armature` DOES affect BAM**（BAM 会 set `dof_armature≈0.0018`，不被 zero），所以 stock `dr.joint_armature` 有效。
- `randomize_com` / `randomize_head_com`（L413-434）：`dr.body_ipos`，reset 模式，`add` 操作。注释说明 mjlab 1.3.0 的 stock `dr.*`（add/scale）**天生非累积**（每次 reset 从编译期默认值读），替代了 microduck 旧的"先还原再加"的 workaround。
- IMU orientation（L514-517）：**不在 event 层做**，改到 obs 层（第 7 节）——旧的 event 版写 `site_quat`，mjlab 1.3.0 下既不 per-env 展开也不被 obs 读 → no-op。

> **读法教学**：这一节是"框架升级踩坑"的集中营。凡是注释里出现 **no-op / silent no-op / was replaced** 的，都在提醒你：**配置写了 ≠ 生效了**。验证 DR 是否真的生效，是 sim2real 调试里最容易忽略的一环。

---

## 第 7 节 · OBSERVATIONS：策略能看到什么（L531-635）

obs = 策略每步的"感官输入"。**设计第一原则：真机能不能拿到这个量。** 核心机制是 **actor / critic 非对称**：

- **actor**（部署时真正用的策略）：只看**真机传感器能拿到**的量。
- **critic**（只在训练时估值用）：可以看**上帝视角（privileged）**——真机拿不到的真值。

### 7.1 删 / 加：actor 与 critic 的信息差（L532-543）

| 操作 | 项 | 原因 |
|---|---|---|
| **删** | `actor` 的 `base_lin_vel` | 真机没有直接的线速度传感器（只能靠 IMU + 运动学估） |
| **删** | `actor` + `critic` 的 `height_scan` | microduck 没有给策略用的 body-mounted 地形传感器（对标 microban） |
| **加** | `critic` 的 `base_lin_vel` | privileged 真值，只喂 critic 帮助估值 |

### 7.2 改：把真机传感器的"不完美"注入 sim（L560-635）

- **延迟 lag（L560-570）**：`base_ang_vel` / `projected_gravity` 的 `delay_max_lag = 1`（**原为 3**，即 60ms 最坏情况）——2026-07 审计发现真实 dxl IMU 通路很快（±20ms envelope），3 步延迟过头了。
- **`joint_vel` 的 1-step lag（L604-613）**：Dynamixel 固件用"上一个位置采样窗口的滑动平均"算 `present_velocity`，所以策略读到的速度**天然慢 ~1 个控制周期**。这里显式建模这个滞后，**阻止策略依赖瞬时 qdot 反馈**。
- **obs 噪声（L586-590）**：全部**大幅调小**——`base_ang_vel ±0.03`（was 0.2）、`gravity ±0.01`（was 0.15）、`joint_pos ±0.001`（was 0.05）、`joint_vel ±0.25`（was 2.0）。
- **IMU 安装误差 DR（L592-602，actor only）**：把 `base_ang_vel` / `projected_gravity` 的 func 换成 `*_imu_misaligned` 版（每 env 恒定旋转 ≤6°），**critic 保持真值**。
- **encoder-bias DR（L626-635）**：基类模板采样了每 env 恒定的编码器偏置，但 `joint_pos` 要 `biased=True` 才会用它。**喂给 actor 的是带偏置的（真编码器报告值），critic 用真值（privileged）**。
- **passive 关节排除（L615-624）**：把 jaw linkage 的 `passive_*` 关节从 `joint_pos/vel` obs 排除，使 **obs 维度匹配 action 维度（14）而非原始关节数（16）**。
  > 注意这里**先 deepcopy 每个 term**——因为 actor 和 critic 从基类模板**共享同一批 term 对象 / params 字典**，改一个会泄漏到另一个（例如上面的 `biased` flag）。这是"共享可变默认值"的经典 Python 陷阱。

### 7.3 critic 专属的 NaN 消毒（L572-584）⭐ 真实崩溃复盘

```python
for _term, _safe in (("foot_contact_forces", ...foot_contact_forces_safe),
                     ("foot_height", ...foot_height_safe),
                     ("foot_air_time", ...foot_air_time_safe)):
    if _term in cfg.observations["critic"].terms:
        cfg.observations["critic"].terms[_term].func = _safe
```

> **为什么**：`nan_state` 只查 joint + root state，但 critic 这几个 term 读的是 **raycast / contact 传感器数据**——MuJoCo 可能在 state 仍然干净时就返回 non-finite 值。**这是 `nan_state` 唯一保护不了的 obs 路径**。这里单个 NaN 会通过 rsl_rl 的 `check_nan` **杀死整个训练 run**——这就是 **2026-08-21 的 `Velocity2-Rough-Backlash` 崩溃**。因为这些 term 只在 critic，消毒对策略**零成本**。

> obs 的 61 维统一布局（`[twist(3), head_pose(4), body_pose(6)]` + 关节/IMU 等）详见 [`architecture_and_glossary.md`](../02-architecture/architecture_and_glossary.md)，此处不重复。

---

## 第 8 节 · COMMANDS：给策略下的指令（L637-701）

command = 每步喂给策略的"任务目标"。velocity 有三个命令通道：

### 8.1 `twist`（速度命令，L637-654）

- **先 deepcopy（L640）**：`make_velocity_env_cfg()` 返回的对象带**共享可变引用**；standup / ground_pick 等 env 会**就地 mutate** `commands["twist"]` 把 ranges 清零。不 deepcopy 就会**互相污染**——这是多 env 共享配置对象的经典坑。
- **固定范围（L648-650）**：`lin_vel_x (-0.4,0.4)`、`lin_vel_y (-0.3,0.3)`、`ang_vel_z (-1.0,1.0)`，**不加宽 curriculum**。
  > **案例**：曾经 ramp 到 `lin ±0.4 / ang ±2.0`，**outpaced 机器人能力**，对应一次 post-iter-1000 的 reward / episode-length 下滑。`ang ±1.0` 是关键改动——它让"转向"变得可学。
- `rel_standing_envs=0.02`（小但非零，curriculum 会 ramp）、`rel_turn_in_place_envs=0.15`（对应 docstring 第 3 条）。

### 8.2 `head_pose`（4D，L656-674）+ `body_pose`（6D，L675-689）

- **`head_pose`**：`UniformPoseCommandCfg`，4 个"相对 HOME 的增量"，关节顺序 `neck_pitch, head_pitch, head_yaw, head_roll`。初始范围很小（±0.05 等）让输入神经元从 step 0 就活跃，curriculum 再加宽。每关节的 final cap 反映**机械可达增量**（XML 限位减 HOME 偏移，留 ~10% 余量）：pitch ±1.10 rad、yaw ±1.40、roll ±0.31。
- **`body_pose`**：6D 增量 `[x,y,z,roll,pitch,yaw]`。vel env **带着这个槽只为 runtime obs 形状一致**，权重极小不主导策略；standup env 才 raise weight + 加宽。

### 8.3 obs 顺序（L691-701）

`head_command` / `body_command` 追加到 actor 和 critic。**顺序至关重要**：runtime obs 布局依赖 `[twist(3), head_pose(4), body_pose(6)]` 这个固定次序。

---

## 第 9 节 · CURRICULUM：训练难度怎么爬（L776-907）

curriculum = "先易后难"的自动化。**step 单位 = iter × 24**（`NUM_STEPS_PER_ENV`）。velocity 有 7 个爬坡项 + 2 个删除：

| curriculum | 函数 | 爬坡内容 |
|---|---|---|
| `action_rate_weight`（L778-791） | `reward_weight` | `action_rate_l2` 权重 **-0.1 → -1.0**（by iter 1500）。先让步态 bootstrap，再收紧平滑 |
| `standing_envs`（L794-807） | `standing_envs_curriculum` | 站立 env 占比 **0.02 → 0.25**（by iter 2000）。走路稳了再教"该停就停" |
| `head_pose_range`（L816-829） | `pose_command_range_curriculum` | 头姿命令范围 5 阶段 **5% → 100%** 每关节 cap |
| `body_pose_range`（L833-848） | 同上 | 保持小（单阶段），standup 覆盖 |
| `com_range`（L851-869） | `com_range_curriculum` | CoM 随机范围 **0.003 → 0.015**（封顶，见下） |
| `head_com_range`（L872-885） | 同上 | 头 CoM 范围 **0.003 → 0.01**（封顶 ±10mm） |
| `head_pose_bias_weight`（L896-907） | `reward_weight` | `head_pose_bias` 权重 **0（iter<600）→ 3.0**（by 1500） |
| ~~`terrain_levels`~~（L888-889） | —— | **删**（非 rough 时） |
| ~~`command_vel`~~（L890） | —— | **删**（基类的速度范围爬坡不要，改固定范围） |

**两个精妙的"先关后开 / 封顶"决策**：

- **`com_range` 封顶 ±15mm**（L857-862，2026-07 审计）：
  > 之前 ramp 到 **±30mm 超出了脚的支撑多边形**（脚跟只在脚踝后 20mm）→ 随机化后的 CoM 可能**完全坐在支撑外** → 逼出一种"又宽又快的过度反应步态"，让**向后平衡变得不可训练**。回归时间线和 ramp 增加完全吻合：`0.015 → 0.02 → 0.03`，策略一个比一个差。
- **`head_pose_bias_weight` iter 600 前恒为 0**（L892-895）：
  > 因为"姿态精度项"在**步态还没形成前是个干扰**。等到 weight 3.0 时，15° 残余偏置每步花 0.79、2° 偏置花 0.10——正好接上第 5 节那个"只罚可逃逸 DC 分量"的设计。

> **读法教学**：curriculum 的三处智慧——① `action_rate` **先松后紧**（先学会走，再走得平滑）；② `head_pose_bias` **先关后开**（先有步态，再抠姿态）；③ `com_range` **封顶**（随机化一旦超过物理支撑多边形，任务就不可学了）。

---

## 第 10 节 · MicroduckRlCfg：网络与 PPO（L912-949）

这一坨是**"怎么训"**（网络结构 + PPO 超参），和前面 700 行的**"训什么"**（env）分开。

- **actor 网络**：`hidden_dims=(512,256,128)`，`elu`，`obs_normalization=True`，动作分布 `GaussianDistribution(init_std=1.0, std_type="scalar")`。
- **critic 网络**：同结构 `(512,256,128)` + elu + obs_norm（无分布——critic 只输出一个 value 标量）。
- **`PpoWithSymmetryCfg`**：`clip_param=0.2`、`entropy_coef=0.01`、`num_learning_epochs=5`、`num_mini_batches=4`、`learning_rate=1e-3`、`schedule="adaptive"`、`gamma=0.99`、`lam=0.95`、`desired_kl=0.01`、`max_grad_norm=1.0`、`symmetry_cfg=None`（因 `ENABLE_SYMMETRY=False`）。
- **wandb / 产出**：`wandb_project="mjlab_microduck"`、`experiment_name="velocity"`（= checkpoint 目录名）、`run_name="velocity"`（拼成 `<datetime>_velocity`）、`save_interval=250`、`num_steps_per_env=24`、`max_iterations=50000`。

> **两处一致性要注意**：① `num_steps_per_env=24` 必须和 L22 的 `NUM_STEPS_PER_ENV=24` 一致（curriculum 的 step 换算依赖它）；② `experiment_name` 决定 checkpoint 目录，**一路串到后续的 export / infer / publish**。PPO 各超参的含义见 [`architecture_and_glossary.md`](../02-architecture/architecture_and_glossary.md) B 组，不重复。

---

## 第 11 节 · 串起来：velocity 在训练全流程里的位置

```
uv run train Mjlab-Velocity-Flat-MicroDuck        ← 命令（见 walkthrough 阶段 4）
        │  task_id 在 tasks/__init__.py 注册，指向 ↓
        ▼
make_microduck_velocity_env_cfg(rough=False)      ← 本文逐段拆解的这个函数
        │  = make_velocity_env_cfg() 基类 + 改/删/加
        ▼
ManagerBasedRlEnv（obs/action/reward/event/termination/curriculum 各 manager 生效）
        │  + MicroduckRlCfg（PPO + 网络）
        ▼
训练循环 → wandb 曲线 → checkpoint（logs/…/velocity/…）
        ▼
export ONNX → infer 验证 → publish 到机器人     ← 见 walkthrough 阶段 6-7
```

- **velocity 是脊柱**：`VelStand / StandUp / SitStand / GroundPick / BallKick / Roller*` 等任务，都是"从 velocity 这套配方分叉 + 局部改写"。你在这个文件里学到的"改/删/加 + DR + curriculum"套路，**换任何一个任务都成立**。
- 想知道每个入口命令具体在哪、怎么跑，查 [`entry_points_map.md`](../07-reference/entry_points_map.md)。

---

## 第 12 节 · 精读检查清单 + 下一步

**读完这份精讲，你应该能不看文档回答：**

1. velocity 的 **actor 能看到 `base_lin_vel` 吗**？为什么？（不能——真机没有线速度传感器；只有 critic 作为 privileged 信息看）
2. 为什么 **`foot_slip` 只有 -0.1** 这么弱？（-1.0 对"枢轴转向"太限制）
3. **`head_pose_bias` 为什么用 1s EMA、而不是瞬时紧容差**？（瞬时容差不可逃逸——走路时头必须晃——策略会躺平站着不动；只对可逃逸的 DC 分量定价）
4. **turn-in-place 为什么要单独分 15% 的桶**？（独立均匀采样下原地转只占 ~2% 数据，几乎没被训练）
5. rough 地形为什么要**软化接触 + 提高 `nconmax`/`iterations`**？（硬边接触产生 NaN 力；接触溢出会静默丢弃 → 突然解压 → NaN）
6. 为什么 `randomize_mass_inertia` 用 **`pseudo_inertia`** 而不是直接写 `body_mass`？（mjlab 1.3.0 下直接写是 silent no-op，collapse 成单值）
7. curriculum 里的 `step` 单位是什么？（iter × 24）

**动手练习**（呼应 [`desktop_quickstart.md`](../01-quickstart/desktop_quickstart.md)）：把 `track_linear_velocity.weight` 从 `2.0` 改成 `1.0`，跑一个 smoke test，观察 wandb 上 `Episode_Reward/track_linear_velocity` 和实际前进速度的变化——**亲手改一个旋钮，比读十遍都记得牢**。

**下一步读什么：**

- **`04-modules/` #4「mdp.py 词库」**：本文反复出现的 `head_pose_bias_penalty`、`robot_state_is_nan`、`reward_weight`、`standing_envs_curriculum` 等函数的**具体实现**（EMA 到底怎么算、NaN 到底查哪些字段），都在 [`mdp.py`](../../../microduck_rl/src/mjlab_microduck/tasks/mdp.py)（7188 行）里。
- **实战**：基于 velocity 的走路 + BallKick 的球 obs/reset，动手做一个"鸭子走到球旁、用喙碰球"的最小可行新 task，走通 `注册 → env_cfg → reward → 训练 → 导出` 全流程。

---

## 附录 · 「改 / 删 / 加」完整速查表

> 这张表是 velocity env 相对上游 mjlab 基类的**完整 diff**——不用读基类源码，就能看清 microduck 到底动了什么。

**Rewards**（L275-359, L703-745）

| term | 操作 | 关键改动 |
|---|---|---|
| `pose` | 改 | weight 1.0；**只作用腿关节**（排除 neck/head/passive） |
| `upright` | 改 | weight 1.0→**2.0**，std²→0.05，body=trunk_base |
| `track_linear_velocity` | 改 | weight 2.0，std²=0.1 |
| `track_angular_velocity` | 改 | weight 2.0，std²=0.5 |
| `air_time` | 改 | weight 3.0，窗口 [0.125,0.300] |
| `foot_clearance` / `foot_swing_height` | 改 | target_height 0.01→**0.02** |
| `foot_slip` | 改 | weight **-0.1**（故意弱） |
| `body_ang_vel` / `angular_momentum` | 改 | -0.05 / -0.02 |
| `action_rate_l2` | 改 | -0.1（stage-0，curriculum ramp） |
| `soft_landing` | **删** | pop 掉 |
| `self_collisions` | **加** | -1.0 |
| `head_pose_tracking` | **加** | 2.0，std 0.5 |
| `body_pose_tracking` | **加** | 0.0（禁用保活） |
| `head_pose_bias` | **加** | 0.0（curriculum ramp，EMA） |

**Observations**（L531-635）

| term | 操作 | 关键改动 |
|---|---|---|
| `actor.base_lin_vel` / `*.height_scan` | **删** | 真机拿不到 |
| `critic.base_lin_vel` | **加** | privileged |
| `base_ang_vel` / `projected_gravity` | 改 | lag 3→1、noise 调小、IMU 失准 func（actor only） |
| `joint_pos` | 改 | noise 调小、排除 passive、encoder-bias（actor only） |
| `joint_vel` | 改 | noise 调小、排除 passive、1-step lag |
| `critic.foot_contact_forces/height/air_time` | 改 | 换 `*_safe` 版（NaN 消毒） |
| `head_command` / `body_command` | **加** | 顺序 [twist,head,body] |

**Events / Terminations**（L361-529）

| term | 操作 | 关键改动 |
|---|---|---|
| `foot_friction` | 改 | ranges (0.3,1.2)→**(0.7,1.3)** |
| `reset_base` | 改 | z (0.12,0.13) |
| `encoder_bias` | 改 | bias_range；actor obs biased=True |
| `expand_bam_friction_fields` / `reset_action_history` | **加** | BAM 注册 / 重置动作史 |
| `push_robot` | **加** | ±0.3（条件） |
| `randomize_com/head_com/armature` | **加** | dr.*（条件） |
| `randomize_mass_inertia` | **加** | `pseudo_inertia`（替换 silent no-op） |
| `randomize_joint_friction` | **加** | BAM-native（stock 是 no-op） |
| `nan_state` | **加** | NaN 数值防火墙 |

**Commands / Curriculum / Scene / Sim**

| term | 操作 | 关键改动 |
|---|---|---|
| `commands.twist` | 改 | deepcopy + 固定范围 + turn_in_place 0.15 |
| `commands.head_pose` / `body_pose` | **加** | 4D / 6D 命令 |
| `curriculum.terrain_levels` / `command_vel` | **删** | 基类默认爬坡不要 |
| `curriculum.action_rate_weight` 等 7 项 | **加** | 见第 9 节 |
| `scene.entities.robot` | 改 | MICRODUCK_WALK_ROBOT_CFG |
| `scene.sensors` | **加** | feet_ground / self_collision / foot_height_scan |
| `scene.terrain` | 改 | flat=plane / rough=generator+软化 spec_fn |
| `sim.nconmax` / `iterations` / `ls_iterations` | 改 | rough 时 200 / 30 / 50 |
| `actions.joint_pos.scale` | 改 | 1.0 |

---

*本文是 `04-modules/` 模块击破系列 #5。返回 [`04-modules/README.md`](../README.md) 看模块清单，或回 [`../../README.md`](../../README.md) 主导航。*
