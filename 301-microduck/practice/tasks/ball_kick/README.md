# ball_kick 高踢任务 · RL 全链路实战（v4 脖子夹球 12cm → v5 机制感知干净踢射 ~70cm）

> **本地复现侧**（`practice/`）的第一个 microduck 任务复现。以「microduck 踢球，看能踢多高」为脊柱，
> 把 RL 全链路走一遍：环境搭建 → 任务注册 → reward 设计 → PPO 训练 → 回放/预览可视化。
>
> 这不是一份「一次成功」的教程，而是一份**五版迭代排查实录**：v1→v4 踩了三个坑（几何不可达 ×2、奖励被站姿支配 ×1），
> v4 让球被挑到 12cm 但 it3000 即 plateau，且回放暴露这 12cm 实为**脖子夹球**（reward hacking）；v5 用「机制感知 reward」（门控+颈接触惩罚）废掉 exploit，
> **仅 1000 iter 就把干净踢射顶到 mean ~70cm、14/16 ≥50cm、机制 16/16 脚腿**。踩坑与判据全部写进[第六章](#六reward-设计与五版排查实录v1v5)，这才是最有价值的部分。
>
> 上级导航：[`practice/README.md`](../../README.md) ｜ 总结侧入口：[`docs/rl/README.md`](../../../docs/rl/README.md)
>
> **红线**：全程 **不改上游子模块 `microduck_rl`**，所有新代码落在本目录；env/reward 改动走「复用官方 cfg + 覆盖」的封装手法。验收：`git -C ../../../microduck_rl status` 干净。

---

## 一、为什么是踢球 & 学什么

`microduck_rl/src/mjlab_microduck/tasks/` 下有 17 个已注册任务（velocity 走路、standup 起身、roulade 前滚翻、spin 旋转、ball_kick 踢球……）。选 **ball_kick** 当学习脊柱，因为它：

- **目标可量化、有趣**：球能踢多高 / 多远，肉眼可见，reward 曲线和行为直接对应；
- **麻雀虽小五脏俱全**：一份 662 行的 env cfg 覆盖了 RL 工程的全部要素——场景组装、接触传感器、域随机化(DR)、curriculum、非对称 actor-critic、reward 塑形、PPO 超参；
- **改 reward 就能改行为**：不用碰网络、不用碰算法，是体会「奖励函数如何决定策略」的最佳沙盒。

**学习目标**：跟着这个任务搞懂——软件框架长什么样、每个模块干什么、新任务入口怎么从零长出来、reward 怎么写、遇到「目标学不动」时怎么系统排查。

---

## 二、软件框架图

本任务是「**在别人的完整 RL 框架上，用一层薄薄的 shim 注入一个自定义变体**」。三层关系与数据流：

```
                          ┌───────────────────────────────────────────┐
   三条入口（practice）     │ train_high_kick.py   play_high_kick.py     │
                          │ preview_scene.py                            │
                          └──────────────────┬────────────────────────┘
                                             │ register_high_kick_task()
                                             │ （把变体塞进全局 _REGISTRY）
                                             ▼
                          ┌───────────────────────────────────────────┐
   shim 组装层（不改子模块） │ high_kick_cfg.py :: make_high_kick_env_cfg │
                          │   ① entities += pedestal  (pedestal.xml)   │
                          │   ② events: reset_ball → 摆球到桩顶         │
                          │   ③ rewards: 删 forward/overshoot/upright   │
                          │      + ball_lift / ball_vertical_velocity   │
                          │      站姿项 ×0.25、动作约束改轻              │
                          │   ④ terminations −= fell_over               │
                          │   ⑤ curriculum −= action_rate ramp          │
                          │   HIGH_KICK_RL_CFG = deepcopy(官方) 改名     │
                          │  依赖: high_kick_rewards.py · pedestal.xml  │
                          └─────────┬───────────────────────┬───────────┘
              复用（import，只读）    │                       │ 注册进 _REGISTRY
       ┌────────────────────────────┘                       └────────────────────────┐
       ▼                                                                              ▼
┌────────────────────────────────┐                     ┌──────────────────────────────────────┐
│ 子模块 microduck_rl（上游，只读）│                     │ mjlab 框架（GPU 并行 RL 基础设施）       │
│  make_microduck_ball_kick_env_  │                     │  tasks/registry._REGISTRY             │
│    cfg()  ← 662 行完整配方       │                     │  ManagerBasedRlEnv(scene/managers)    │
│  MicroduckBallKickRlCfg (PPO)   │                     │  launch_training / run_play           │
│  MicroduckOnPolicyRunner        │                     │  OffscreenRenderer(EGL) / Viewer      │
│  tasks/mdp.py（reward/obs/event）│                     │  MuJoCo Warp（GPU 批量物理步进）       │
│  robot/microduck/ball.xml(15g球)│                     │  rsl_rl PPO（actor-critic）           │
└────────────────────────────────┘                     └───────────────────┬──────────────────┘
                                                                           ▼
                                          logs/rsl_rl/ball_kick_aggr_right/<时间戳>/
                                            model_N.pt · *.onnx · events.tfevents · params/ · wandb(offline)
```

**三条链路各走一遍**（都从 `register_high_kick_task()` 起步，保证 registry 里有我们的 task）：

| 链路 | 入口 | 调用栈 | 产物 |
|---|---|---|---|
| **训练** | `train_high_kick.py` | 注册 → `tyro.cli(TrainConfig)` → `launch_training(TASK_ID)` → mjlab 建 env + `MicroduckOnPolicyRunner` 跑 PPO | `logs/rsl_rl/ball_kick_aggr_right/<ts>/model_N.pt` + tfevents + wandb |
| **回放** | `play_high_kick.py` | 注册 → `tyro.cli(PlayConfig)` → `run_play(TASK_ID)` → `load_env_cfg(play=True)` + `runner.load(ckpt)` + viewer | 交互画面 / mp4 |
| **预览** | `preview_scene.py` | `make_high_kick_env_cfg(play=True)` → `ManagerBasedRlEnv(render_mode="rgb_array")` → EGL 离屏渲染 | `preview_scene.png` |

**关键工程点**：`_REGISTRY` 是 mjlab 的模块级全局 dict，`register_mjlab_task(...)` 运行时可追加（task_id 唯一即可）。所以我们能在**子模块外部**注入一个新 task，让官方的 `launch_training`/`run_play` 原样命中它——一行子模块代码都不用改。

---

## 三、模块作用与技术栈

本目录 7 个文件，职责单一、边界清晰：

| 文件 | 作用 | 关键技术 |
|---|---|---|
| [`high_kick_cfg.py`](high_kick_cfg.py) | **组装层核心**：造 env cfg（覆盖 entities/events/rewards/terminations/curriculum）+ deepcopy 官方 rl cfg 改名 | `mujoco.MjSpec.from_file`、`EntityCfg`、`RewardTermCfg`、`EventTermCfg`、`deepcopy`、yaw-frame 坐标变换 |
| [`high_kick_rewards.py`](high_kick_rewards.py) | **reward 函数库**：v4 机制盲 4 函数（离地高度 / 向上速度 / 超起始高增量 / 击落惩罚）+ v5 机制感知 4 函数（颈接触 mask / 门控 lift / 门控向上速度 / 颈接触惩罚），返回 `[num_envs]` | `ball.data.root_link_pos_w[:,2]`、`root_link_lin_vel_w[:,2]`、`clamp`、`nan_to_num`、`sim.data.xpos` 世界坐标距离 |
| [`pedestal.xml`](pedestal.xml) | **场景资产**：垫高用的细桩 tee（圆柱 r=10mm、顶面 z=0.045），把球架到脚够得着又能从下方挑的高度 | MJCF、`freejoint` + 5kg 重块（dynamic entity，规避 static welded 的不确定性） |
| [`train_high_kick.py`](train_high_kick.py) | **训练入口**：注册变体 task → tyro 解析 → 复用官方训练管线 | `register_mjlab_task`、`tyro.cli(TrainConfig)`、`launch_training`、`MUJOCO_GL=egl` |
| [`play_high_kick.py`](play_high_kick.py) | **回放入口**：复用 train 的注册函数 → 从本地 checkpoint 回放 | `run_play(TASK_ID, PlayConfig(checkpoint_file=...))`、native/viser viewer |
| [`preview_scene.py`](preview_scene.py) | **headless 预览**：无 DISPLAY 时把场景离屏渲成 4 方位角拼图 PNG | `render_mode="rgb_array"`、`OffscreenRenderer`(EGL)、`ViewerConfig`、`env._offline_renderer._cam`、PIL 拼图 |
| [`.gitignore`](.gitignore) | 忽略 `logs/`、`*.mp4`、`preview_scene.png` 等可再生产物 | — |

### 3.1 组装层怎么「覆盖」官方配方（最关键的机制）

`make_high_kick_env_cfg()` 不重写配方，而是**先拿官方的完整 cfg，再逐项增删改**（cfg 都是可变 dataclass）：

```python
cfg = make_microduck_ball_kick_env_cfg(play=play, kick_foot=kick_foot)  # 官方 662 行配方
cfg.scene.entities["pedestal"] = PEDESTAL_CFG          # ① 加垫高细桩
cfg.events["reset_ball"] = EventTermCfg(func=reset_ball_on_pedestal, ...)  # ② 换 reset：球摆到桩顶
cfg.rewards.pop("ball_forward_velocity", None)         # ③ 删「滚地球」主奖励
cfg.rewards["ball_lift"] = RewardTermCfg(func=hk.ball_lift_above_start, weight=30.0, ...)  # 加挑高主奖励
cfg.terminations.pop("fell_over", None)                # ④ 摔倒不再中断回合
cfg.curriculum.pop("action_rate_weight", None)         # ⑤ 删动作惩罚的 ramp
```

> **为什么 `RewardTermCfg.func` 能挂我们自己写的函数**：它只是一个「可调用对象」，运行时解析，签名 `(env, **params) -> Tensor[num_envs]` 即可，不要求来自子模块 `mdp.py`。这是「不改子模块也能加 reward」的根本依据。

### 3.2 非对称 actor-critic（继承自官方，值得理解）

冒烟日志会打印 `Actor shape (61,)` / `Critic shape (80,)`，差 19 维 = **球的特权观测**（`ball_pos_in_base` + `ball_vel_in_base`）。含义：**actor（真机要部署的策略）对球是盲的**，只靠本体感觉开环挥腿；critic 训练时能看到球，用来更准地估计「这一脚的收益」。这是 sim2real 的常见设计——真机没有球传感器，所以策略不能依赖球状态。（v4 之后若仍学不动，第一个诊断就是「暂时给 actor 也喂球状态」验证可学性，详见 [6.5](#65-可复用的排查方法论三级排查--rollout-探针)。）

---

## 四、如何新增一个 task（结合高踢任务，7 步）

这套流程适用于「想在 microduck_rl 之上加一个自己的任务变体，但不改子模块」。以高踢任务为实例：

**第 1 步 · 找官方配方工厂。** 在 `microduck_rl/src/mjlab_microduck/tasks/` 找到最接近的已有任务（这里是 `microduck_ball_kick_env_cfg.py`），读它的 `make_*_env_cfg()` 和 `*RlCfg`——你要复用的就是这两样。

**第 2 步 · 写组装函数。** 新建 `high_kick_cfg.py`，`make_high_kick_env_cfg()` 里调官方工厂拿 cfg，再按 [3.1](#31-组装层怎么覆盖官方配方最关键的机制) 的方式覆盖 entities/events/rewards/terminations/curriculum。**只覆盖你要改的，其余全部继承。**

**第 3 步 · 写自定义 reward（如需）。** 新建 `high_kick_rewards.py`，函数签名 `(env, **params) -> Tensor[num_envs]`，从 `env.scene["ball"].data` 取物理量。挂进 `cfg.rewards["名字"] = RewardTermCfg(func=..., weight=..., params={...})`。

**第 4 步 · 加场景资产（如需）。** 新建 MJCF（这里 `pedestal.xml`），用 `EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_file(xml), init_state=...)` 挂进 `cfg.scene.entities`。**注意：robot 必须保持字典第一个 entity**（reset 事件按 `qpos[:,0:7]` 写机器人根）。

**第 5 步 · deepcopy rl cfg 改名。** `HIGH_KICK_RL_CFG = deepcopy(官方RlCfg)`，改 `experiment_name`/`run_name`（决定日志目录，务必与原版区分）、`max_iterations`。rl cfg 非 frozen，可直接赋值。

**第 6 步 · 写注册 + 训练/回放入口。** `train_high_kick.py`：`import mjlab_microduck.tasks`（触发官方 17 个 task 注册）→ `register_mjlab_task(TASK_ID, env_cfg, play_env_cfg, rl_cfg, runner_cls)` → `tyro.cli(TrainConfig)` → `launch_training(TASK_ID, args)`。`play_high_kick.py` 复用同一个注册函数 + `run_play`。

**第 7 步 · 走验证阶梯**（前一阶过了才进下一阶）：

```bash
# ① 静态验证：cfg 键/权重/维度对不对（秒级，不训练）
python -c "import high_kick_cfg as C; cfg=C.make_high_kick_env_cfg(True); print(list(cfg.rewards)); print(cfg.terminations)"
# ② 冒烟：64 envs × 1 iter，确认管线 EXIT=0、日志出现新 reward 键
WANDB_MODE=offline ../../../.venv/bin/python train_high_kick.py --env.scene.num-envs 64 --agent.max-iterations 1
# ③ 预览场景：确认几何摆位对（headless 也能看）
MUJOCO_GL=egl ../../../.venv/bin/python preview_scene.py
# ④ 真训练：512~2048 envs × 数百~数千 iter，盯目标 reward 曲线
# ⑤ rollout 探针：加载 checkpoint 把 reward 换算成物理量（球实际挑多高 cm）
```

> **验收红线**：每步之后 `git -C ../../../microduck_rl status --short` 必须为空（子模块零改动）。

---

## 五、核心洞察：官方「轻触版」vs 我们的「高踢版」

仓库里已有一份完整的官方 ball_kick（`Mjlab-BallKick-Flat-MicroDuck`），优化目标和我们**恰好相反**——这是最生动的教材。读源码 [`microduck_ball_kick_env_cfg.py`](../../../microduck_rl/src/mjlab_microduck/tasks/microduck_ball_kick_env_cfg.py) 发现官方注释白纸黑字写着「第一版用 linear reward capped at 5 m/s，kicked much harder than needed，所以特意调成 gentle tap（轻触）」：

- 只奖励**水平前向速度** `ball_forward_velocity`（取 `vel_xy[:, :2]`，**完全不看 Z 轴**），cap 在 1.0 m/s；
- 外加 `ball_speed_overshoot`（−4.0）**主动惩罚踢太快**。

即官方策略被精心设计成「温柔轻推球滚地」。我们要「尽量把球挑高」。v4 最终差异：

| reward / 终止项 | 官方轻触版 | 高踢 v4 | 改动理由 |
|---|---|---|---|
| `ball_lift`（超起始高增量） | ❌ 无 | ➕ **w=30（主奖励）** | 「踢多高」的直接目标，dense 给分，压倒站姿 |
| `ball_vertical_velocity`（向上速度） | ❌ 无 | ➕ **w=10（辅奖励）** | 起球瞬间给梯度，最早的稠密 bootstrap |
| `ball_forward_velocity` | ✅ 12.0（主），cap 1.0 | ❌ **删除** | 它是「滚地球」歧途的元凶 |
| `ball_speed_overshoot` | ✅ −4.0 | ❌ **删除** | 压制猛踢，与「踢高」直接冲突 |
| `upright` | ✅ 2.0 | ❌ **删除** | 高斯强力对抗前倾/摆腿（roulade 同款先例） |
| `fell_over`（终止） | ✅ 有 | ❌ **删除** | 摔倒不再中断回合；躺平拿不到 lift＝隐式惩罚 |
| `support_foot_grounded`/`pose_stand_*`/`height_stand` | ✅ 2.0/2.0/1.0/1.0 | ♻️ **×~0.25** | 站姿降为软偏好，压不过踢高 |
| `action_rate_l2`（+ curriculum ramp 到 −1.0） | ✅ −0.1→−1.0 | ♻️ **−0.05 且删 ramp** | 放开爆发性挥腿，长跑不再被压死 |

---

## 六、reward 设计与五版排查实录（v1→v5）

这是本文档的核心。**目标学不动时怎么系统排查**，比「最后配对了什么权重」更值得记。五版演化：

### 6.1 v1 贴地版 —— 失败于「几何不可达」

- **做法**：球贴地 spawn（球心 z = 球半径 0.035，球底与地面零缝隙）；主奖励 `ball_height_above_ground`（离地高度，w=8）；保留 `ball_forward_velocity`（降为 w=2）。
- **结果**：300 iter，`mean_reward` 0.65→17.9 但涨幅几乎全来自站姿，`ball_height` 全程 ≈ 0（峰值 <0.1mm）。机器人只学会「站稳蹭球滚动」。
- **根因（世界坐标坐实）**：microduck 脚是**扁平 sole**，球贴地时脚插不进球下方 → 无法「从下往上铲」，接触力竖直分量恒为向下或水平 → **垂直挑高在贴地设定下物理不可达**。
- **教训**：动作空间里不存在的解，任何 reward 塑形都优化不出来。**先修几何，别急着调参。**

### 6.2 v2 实体 box 垫高版 —— 同一个几何陷阱搬了位置

- **做法**：造 `pedestal.xml`（初版 3cm 见方**实体 box** + 5kg + freejoint），把球垫到 box 顶（球心 z=0.08）；主奖励换成 `ball_lift_above_start`（基准 = 起始高 0.08，**静止不白给分**）。
- **结果**：300 iter 仍失败。headless rollout 探针给出决定性证据——整回合**球 z 最大值恰等于起始高 0.0800**、lift 计数 **0/4800**、drop 计数 **181/4800**。
- **根因**：实体 box 顶面正好**堵死球的正下方**，脚依旧只能碰到球的侧面/上半部，接触力竖直分量必然向下 → 和贴地版被地板挡住是**同一个几何陷阱，只是把「地板」搬到了球底下**。
- **教训**：垫高≠可达。支撑物本身也会挡住「从下方挑」的通道。

### 6.3 v3 细桩 tee 版 —— 几何可达了，却失败于「奖励被站姿支配」

- **做法**：`pedestal.xml` 改成**半径 10mm 的细圆柱桩**，球赤道下方（z≈0.055~0.07）在桩外侧空出 ~18mm 的槽，让扁平 toe 能从侧面扫进球下方做真正的「斜向上挑」；另加事件型 `ball_drop_penalty`（只罚向下击落那一刻，w=−3，阈值 0.06 留 2cm 容差带避免静止假阳性）。
- **结果**：300 iter，`ball_lift` 仍贴 0（峰值 ≤ 0.5mm），但 `mean_reward` 早饱和到 20.03（全站姿）。
- **根因（这次不是几何）**：几何**已经可达**（细桩让 toe 够得到球下方），但站姿奖励（~20）压倒一切，且 `fell_over` 终止 + `upright` 把「冒险挥腿」的期望收益压成负 → 策略**理性地选「躲球站着刷分」**。`ball_drop` 惩罚还额外把「碰球」变成负收益，加剧回避。
- **教训**：**几何可达只是必要条件。** 若目标项贴 0 而 mean_reward 饱和，病在「奖励支配性」，不是 iter 不足——此时加轮数只会把「躲球站着」磨得更牢。

### 6.4 v4 运动员式猛踢重平衡 —— 突破

- **用户定调**：「像运动员踢球，把球踢得更高是目的，摔倒是其次；能站稳最好，但为了踢高站不稳也行。」
- **做法**（照官方 roulade 前滚翻任务 `del upright` + `del fell_over` 的同款先例）：
  1. `ball_lift` 权重 **8→30**，压倒站姿；`ball_vertical_velocity` **5→10**；
  2. **删 `upright`**（对抗前倾摆腿）、**删 `fell_over` 终止**（摔倒不再中断回合）；
  3. 站姿项 `support/pose_legs/pose_neck/height` **×~0.25**（降为软偏好）；
  4. `action_rate_l2` 改 **−0.05 且删 curriculum ramp**、`body_ang_vel`/`angular_momentum` 减小（放开爆发性挥腿）；
  5. **删 `ball_drop`**（接触自由化，方向完全交给 lift/vertvel 塑造）。
- **结果（300 iter，512 envs）**：

  | 指标 | it0 | it150 | it225 | it299 | 判读 |
  |---|---|---|---|---|---|
  | `ball_lift` | 0.0000 | 0.0014 | 0.0731 | **0.1593** | **单调爬升、未饱和** |
  | `ball_vertical_velocity` | 0.0005 | 0.030 | 0.075 | 0.095 | 同步爬升 |
  | `mean_reward` | 0.003 | 4.03 | 6.30 | 7.26 | 比 v3 低（站姿被砍），仍在涨 |

- **rollout 探针换算（model_299.pt，64 envs × 400 步）**：球**峰值被挑高 2.1cm**（球心峰 z=0.101m）、每 env 峰值 mean 1.6cm；**球不再被拍落**（掉地 env-step 仅 4/25600 = 0.015%，对比 v2 box 版 3.7%）。
- **结论**：**第一次让球真正被挑起来且方向正确**（向上轻挑而非拍飞）。幅度还小（~2cm），但趋势陡升未饱和 → **这才是「该加迭代轮数」的情形**（v3 是平的、不该加）。
- **v4 大训练终局（4096 envs × 23875 iter，用户手动终止于平台期）**：rollout 探针（model_23750.pt）**每 env 峰值挑高 mean 12.1cm / max 13.0cm**（球心峰 z≈0.21m、齐机身头顶），32 envs 全 ≥11cm（稳态技能非运气）、掉地率 2.4%（敢发力）。但 `ball_lift` **it3000 即跳到 ~1.0 后平 20000 iter**（峰 1.14@it7136）——加训练量的收益 it3000 就吃满；伴随数值不稳定累积（mean_reward 15 个灾难负尖峰、最狠 −1.99e8@it17834，频率随训练递增，PPO clip 每次拉回、不致命）。**12cm 是当前配方+几何+探索的局部最优，再往上不是加 iter 能解决的。**

### 6.5 可复用的排查方法论（三级排查 + rollout 探针）

目标项长期零增长时，**按序**排查三类瓶颈，别一上来就加 iter 或瞎调权重：

1. **① 几何可达性**：接触件能否从**正确方向**作用到目标？→ 用世界坐标 / rollout 探针验证（如「球 z 上界是否恰等于起始高」）。不可达就**改摆放几何**（垫高 / 细桩 tee 让末端扫入目标下方）。
2. **② 奖励支配性**：几何已可达但目标仍贴 0、而 `mean_reward` 早饱和？→ 病在站姿/平衡/防摔项**压倒**目标项。杠杆是**重平衡 reward**（抬目标权重、删对抗项、删防摔终止、站姿降为软偏好）。
3. **③ 训练量**：**仅当**目标项已出现**非零且持续上升**的趋势时，加 iter 才有正收益。

> **判据一句话**：目标项**平贴 0（峰值≈噪声）+ mean_reward 饱和** → 病在 ①/②，别加 iter；目标项**有非零峰值且趋势上升未饱和** → 才回到 ③ 加训练量。
>
> **rollout 探针**是把「reward 数字」翻译成「物理量」的关键工具：加载 checkpoint、关掉终止、跑几十~几百步，记录目标物的世界坐标轨迹（球 z 峰值、掉落计数），比只看 reward 曲线更能定性。它是一次性临时脚本（用完即删），核心 API：`RslRlVecEnvWrapper(env)` → `runner.load(ckpt, load_cfg={"actor":True})` → `runner.get_inference_policy()` → 手动 step。

**当前状态（v4 终局）**：①②已过、已执行（大训 23875 iter）——峰值挑高冲到 **12cm**，但 **it3000 即 plateau**，③ 的收益已吃满。再往上（12cm→30~50cm）是**新天花板**：探索熵坍缩 / 细桩单击物理上限 / 数值不稳定，需换杠杆（喂球位姿诊断、峰值高度 reward、重启探索），**不是加 iter**。

### 6.6 v5 机制感知奖励 —— 废掉脖子夹球，干净踢射 ~70cm（第四类瓶颈：奖励机制盲）

- **触发**：v4 大训回放截图暴露「12cm 挑高」实为**脖子夹着球升高**，不是脚踢。rollout 探针（model_23750.pt，8 envs×300 步）坐实：峰值时刻最近 link **颈组 8/8**（neck_pitch×7、trunk_base×1）、脚组 0/8；>5cm 的 1097 个 env-step **100% 贴颈组、0% 贴脚组**。
- **根因**：v4 的 `ball_lift`/`ball_vertical_velocity` 只测**结果**（球多高/向上多快）不测**机制**（是谁弄高的）。脖子（neck_pitch 直驱、位置贴桩顶、jaw 兜住 15g 轻球）scoop 站起比单脚甩腿挑容易一个数量级 → 策略理性选脖子。**目标项达标 ≠ 机制正确。**
- **排除钝杠杆**：改球尺寸/质量同时增大两条路径难度、不选择性废 exploit；限 neck_pitch 关节则 trunk 仍可驮球。故选**机制感知 reward**。
- **做法**（v4 四函数一字不改，新增一套可切换；`HIGH_KICK_REWARD_MODE=v4|v5`，默认 v5）：
  1. `_neck_contact_mask()`：球心距颈组任一 link（neck/neck_pitch/jaw_soft/trunk_base）< 0.075m 判「被夹」，纯几何代理、不依赖接触传感器；
  2. `ball_lift_clean` = lift × (1−mask)、`ball_vertical_velocity_clean` 同款门控 → 脖子路径恒 0 分；
  3. `ball_neck_contact_penalty`（w=−5）= mask → 显式负分，与门控互补；
  4. 实验名按模式分开：v4→`ball_kick_aggr_right`、v5→`ball_kick_clean_right`（日志/ckpt 互不覆盖，可对照重训）。
- **决定性验证（旧 ckpt 喂新 reward）**：把 v4 夹球策略 model_23750.pt 喂进 v5 奖励，高挑高时段 8.9cm→**0.06cm（清零 99%）**、颈接触 mask 0.990 → exploit 被彻底废掉后才开训。
- **v5 训练终局（4096 envs × 1000 iter，用户手动停于 it1000）**：
  - 曲线：`ball_lift` 0→0.13(it100)→0.78(it200)→1.1(it400)→**1.35(it1000)** 高位缓爬未硬 plateau；`ball_neck_contact` 全程≈0（mean last100 = −0.0006）；`ball_vertical_velocity`→1.13；`Mean action std` **卡 1.57 走平（未结晶）**；`action_rate_l2`≈−2.56（稳定极抖）。
  - rollout 探针（model_1000.pt，16 envs×300 步）：峰值挑高 **mean 69.9 / median 71.6 / max 96.4 / min 26.1 cm**；**≥30cm 15/16、≥50cm 14/16**；机制归属 **16/16 全 LEG/FOOT、贴颈=0**。对比 model_500（mean 62.1、1 个 18cm 离群、1 个 NECK）更稳更干净。
- **结论**：v5 用 **1/24 的训练量**（1000 vs 23875 iter）把挑高从 12cm 顶到 **~70cm**，且机制正确、一致性好。**剩余短板是动作形态**：暴力翻扑 launch、std 未结晶、action 极抖——影响「受控/可部署」而非「高」。v6 改进方向见[第十章](#十验证阶梯与后续路线)。

---

## 七、怎么跑

所有命令在**本目录**（`practice/tasks/ball_kick/`）下执行，训练用**根 `.venv`**（相对本目录为 `../../../.venv`）。

### 7.1 预览场景（看细桩 + 球 + 脚的几何，训练前/后都可用）

想先「看一眼」场景摆位（细桩 tee 是否立着、球是否稳坐桩顶、脚够不够得着），分带屏幕 / 不带屏幕两种。

**不带屏幕（headless，推荐、最稳）**：EGL 离屏渲染成一张 4 方位角拼图 PNG，直接读图：

```bash
MUJOCO_GL=egl ../../../.venv/bin/python preview_scene.py            # 默认输出 ./preview_scene.png
MUJOCO_GL=egl ../../../.venv/bin/python preview_scene.py my.png     # 指定输出路径
```

[`preview_scene.py`](preview_scene.py) 用 `render_mode="rgb_array"` 建单环境（`num_envs=1`、`max_extra_envs=0`，画面干净），reset 后把相机 `lookat` 对准球，从 4 个方位角各渲一帧拼成 2×2（机器人朝向随机，多角度确保看清）。产物 `preview_scene.png` 已 gitignore（可再生）。

**带屏幕（有 DISPLAY）**：走 mjlab 原生交互 viewer，可拖拽/缩放。用 `--agent zero` 可不加载 checkpoint、只看场景几何：

```bash
../../../.venv/bin/python play_high_kick.py --agent zero --viewer native --num-envs 1
```

> ⚠️ 两个已知坑：① `native` 需要 `DISPLAY`，headless 机器起不来；② `viser` 网页 viewer 在官方 play 管线的 velocity-command slider 处有 assert bug（`add_slider` 的 `max>=value>=min`）会崩，本任务暂不可用——所以 headless 一律走 `preview_scene.py`。

### 7.2 训练（先冒烟，再逐步加大；v4/v5 用环境变量切换）

reward 版本由环境变量 `HIGH_KICK_REWARD_MODE` 切换（[high_kick_cfg.py](high_kick_cfg.py) 读取，**默认 v5**）：`v4`=机制盲四函数（实验名 `ball_kick_aggr_right`）、`v5`=门控+颈接触惩罚（实验名 `ball_kick_clean_right`）。两者日志/ckpt 目录独立，可对照重训互不覆盖。

```bash
# 链路冒烟：64 envs × 1 iter，约 1 分钟，确认能跑起来（看 EXIT=0 + 日志出现 ball_lift）
WANDB_MODE=offline MUJOCO_GL=egl ../../../.venv/bin/python train_high_kick.py --env.scene.num-envs 64 --agent.max-iterations 1

# v4（机制盲，复现「脖子夹球 12cm」那条线）：4096 envs × 23875 iter
HIGH_KICK_REWARD_MODE=v4 WANDB_MODE=offline MUJOCO_GL=egl ../../../.venv/bin/python train_high_kick.py --env.scene.num-envs 4096 --agent.max-iterations 23875

# v5（机制感知；复现「干净踢射 ~70cm」那条线）：4096 envs × 1000 iter 已足够达标
HIGH_KICK_REWARD_MODE=v5 WANDB_MODE=offline MUJOCO_GL=egl ../../../.venv/bin/python train_high_kick.py --env.scene.num-envs 4096 --agent.max-iterations 1000
# 等价写法（默认即 v5，环境变量可省略）：
WANDB_MODE=offline MUJOCO_GL=egl ../../../.venv/bin/python train_high_kick.py --env.scene.num-envs 4096 --agent.max-iterations 1000
```

产物落在 `./logs/rsl_rl/<实验名>/<时间戳>/`（v4=`ball_kick_aggr_right`、v5=`ball_kick_clean_right`；内含 `model_N.pt` / `*.onnx` / `events.out.tfevents.*` / `params/*.yaml`），已被本目录 `.gitignore` 忽略。`./wandb/offline-run-<时间戳>-<id>/` 与日志目录同起始时间戳，可互相对照（如 `offline-run-20261017_212745-*` ↔ `2026-10-17_21-27-41_*`）。

### 7.3 看训练曲线（TensorBoard）

```bash
../../../.venv/bin/tensorboard --logdir logs/rsl_rl --port 6006    # 同时看 v4(aggr) 与 v5(clean) 两条线
```

**v4/v5 唯一该盯的是 `Episode_Reward/ball_lift` 有没有持续往上爬**（不是 mean_reward——站姿被我们主动砍了，总分会比 v3 低，这是预期不是退步）；v5 额外盯 `Episode_Reward/ball_neck_contact` **绝对值往 0 走**（很负=又在夹球）。辅助看 `Episode_Reward/ball_vertical_velocity`、`Policy/mean_std`（v5 若卡高位不降=策略没结晶）、`Episode_Metrics/mean_action_acc`。注意 `fell_over` 终止已删，日志里不再有摔倒率统计。每次 iteration 控制台那一整块里**每个字段的含义与代码出处**见 [7.5](#75-读懂训练日志每个字段什么意思从哪行代码来)。

### 7.4 如何 Play（回放训练好的策略，看那一挑）

两个版本的最终 checkpoint（回放时按版本选）：

| 版本 | checkpoint | 实测峰值挑高 | 机制 |
|---|---|---|---|
| v4 机制盲 | `logs/rsl_rl/ball_kick_aggr_right/2026-10-17_01-15-28_ball_kick_aggr_right/model_23750.pt` | ~12cm | ❌ 脖子夹球 |
| v5 机制感知 | `logs/rsl_rl/ball_kick_clean_right/2026-10-17_21-27-41_ball_kick_clean_right/model_1000.pt` | mean ~70cm / max 96cm | ✅ 16/16 脚腿 |

分带屏 / 不带屏两种。

**不带屏幕（headless，本机推荐）**：官方 `run_play` 最后必进 `viewer.run()`（native 要 DISPLAY、viser 有 slider bug），headless 跑不通；改用 [`play_to_video.py`](play_to_video.py) 把训练好的策略 rollout **逐帧离屏渲染成 H.264 mp4**（相机水平跟球、叠加 step/球高/峰值标注）：

```bash
MUJOCO_GL=egl ../../../.venv/bin/python play_to_video.py                   # 默认 v4 model_23750.pt → ./play_high_kick.mp4
MUJOCO_GL=egl ../../../.venv/bin/python play_to_video.py <ckpt.pt> my.mp4  # 指定 checkpoint / 输出
# v5 干净踢射回放（本文所有 ~70cm 截图/视频的来源）：
MUJOCO_GL=egl ../../../.venv/bin/python play_to_video.py \
  logs/rsl_rl/ball_kick_clean_right/2026-10-17_21-27-41_ball_kick_clean_right/model_1000.pt play_v5_it1000.mp4
```

产物 `*.mp4`（640×480@50fps、libx264/yuv420p 通用可播、~6s 含一次完整挑球）已 gitignore。

**带屏幕（有 DISPLAY）**：官方交互 viewer，可拖拽 / 热换 checkpoint；`--no-terminations` 让回合不被摔倒/超时打断、看完整动作；`--video True` 另录 mp4 到 `<run>/videos/play/`：

```bash
CKPT=logs/rsl_rl/ball_kick_clean_right/2026-10-17_21-27-41_ball_kick_clean_right/model_1000.pt   # 或 v4 的 model_23750.pt
../../../.venv/bin/python play_high_kick.py --checkpoint-file $CKPT --viewer native --num-envs 1
../../../.venv/bin/python play_high_kick.py --checkpoint-file $CKPT --viewer native --no-terminations --video True
```

> ⚠️ headless 别用 `play_high_kick.py`（必崩）；它只适合有 DISPLAY 的交互回放。纯看几何（不看动作）用 7.1 的 `preview_scene.py`。

### 7.5 读懂训练日志（每个字段什么意思、从哪行代码来）

每次 iteration 控制台会打印一大块（以 v5 `ball_kick_clean_right` it99 那屏为例）。这块由 **rsl_rl 的 `Logger.log()`**（`.venv/.../rsl_rl/utils/logger.py`）拼出来，数据来自三处：PPO 算法（损失/std）、Logger 自己的 buffer（回报/回合长度/耗时）、以及 **mjlab 各 manager 塞进 `info` 的 extras**（`Episode_Reward/*`、`Episode_Termination/*`、`Curriculum/*`、`Metrics/*`、`Episode_Metrics/*`）。按五组读：

**A. 头部与性能**（`logger.py::Logger.log`）

| 字段 | 含义 | 代码出处 |
|---|---|---|
| `Learning iteration 99/3000` | 当前 iter / 总 iter | `it/total_it`，total=`--agent.max-iterations` |
| `Run name` | 实验名（=日志目录名） | `cfg["run_name"]`，即 `HIGH_KICK_RL_CFG.run_name` |
| `Total steps` | 累计采集的环境步数 | `tot_timesteps += num_steps_per_env×num_envs` |
| `Steps per second` | 采样+学习总吞吐（FPS） | `fps = collection_size/(collect+learn)` |
| `Collection time` | 环境 rollout 采样耗时 | `collect_time` |
| `Learning time` | PPO 网络更新耗时 | `learn_time` |
| `Iteration time / Time elapsed / ETA` | 单 iter 耗时 / 已跑 / 预计剩余 | 底部 footer |

**B. PPO 损失与探索度**（`loss_dict` 由 `rsl_rl/algorithms/ppo.py::update()` 返回，logger 打 `Mean {key} loss`）

| 字段 | 含义 | 怎么读 |
|---|---|---|
| `Mean value loss` | critic 价值损失（V(s) 与实测 return 的 MSE） | 反映价值拟合；持续很大=价值没学好 |
| `Mean surrogate loss` | PPO clip 策略梯度目标 | 通常为负，越负=本批更新收益越大 |
| `Mean entropy loss` | 策略高斯分布的熵（探索度） | 与 action std 同向；**掉太快=探索坍缩**（v5 要盯） |
| `Mean action std` | 策略输出高斯 std 均值（可学习参数） | 从 ~0.5 缓慢下降=变确定；**v5 早期别让它塌到 ~0**（还需探索脚踢） |

**C. 总体回报**（Logger 的 `rewbuffer`/`lenbuffer`，各存最近 100 个**完成回合**）

| 字段 | 含义 | 怎么读 |
|---|---|---|
| `Mean reward` | 最近 100 回合**总回报**均值 | 训练主曲线；v4/v5 主动砍了站姿，绝对值比 v3 低是**预期**不是退步 |
| `Mean episode length` | 最近 100 回合步数均值 | 本任务删了 `fell_over`，几乎全靠 `time_out` 结束 → 应≈上限（截图 250.00） |

**D. 逐 reward 项 `Episode_Reward/<term>`**（mjlab `managers/reward_manager.py`）

代码语义：`compute()` 每步累加 `value = func(env)×weight×dt` 到 `_episode_sums`；回合 reset 时 `Episode_Reward/term = mean(_episode_sums)/max_episode_length_s`（`reward_manager.py:108`）。即**该回合内「weight×原始reward」对时间的平均（每秒平均贡献）**，不是单步瞬时值。本任务各项：

| 字段 | 组 | 怎么读 |
|---|---|---|
| `ball_lift` | **主目标**（v5=`ball_lift_clean` 门控版） | **唯一硬指标，要持续往上爬**；v5 里只有「脱离脖子的干净腾空」才计分 |
| `ball_vertical_velocity` | 辅目标（v5=门控版） | 起球爆发力，与 lift 同向爬 |
| `ball_neck_contact` | **v5 新增惩罚**（w=−5） | **绝对值要往 0 走**；持续很负=策略还在脖子夹球（exploit 没死） |
| `support_foot_grounded`/`pose_stand_legs`/`pose_stand_neck`/`height_stand` | 站姿软偏好（×0.25） | 正分，参考即可，不应压过 ball_lift |
| `action_rate_l2`/`body_ang_vel`/`angular_momentum` | 动作/角速度约束 | 负=违反程度；猛踢时为负属正常 |
| `dof_pos_limits`/`self_collisions` | 关节限位/自碰 | 负分；`self_collisions` 大幅变负要留意 |

**E. 课程 / 指令 / 终止 / 步级 metric**（mjlab 各 manager 的 extras）

| 前缀 | 含义 | 代码出处 |
|---|---|---|
| `Curriculum/<term>` | 域随机化课程当前 level（`com_range`/`head_com_range`/`push_magnitude` 的幅度） | `curriculum_manager.py:103/107` |
| `Metrics/<name>/<metric>` | 速度指令跟踪误差（`twist/error_vel_xy`、`error_vel_yaw`）；本任务不追指令，参考 | `command_manager.py:246` |
| `Episode_Termination/<term>` | 本 iter 平均每次 reset 因该项结束的 **env 个数**；`time_out`≈16.8 正常，**`nan_state`>0 要警惕数值爆炸** | `termination_manager.py:95` |
| `Episode_Metrics/<key>` | 步级 metric 的回合平均（如 `mean_action_acc` 动作跟踪精度；从乱抖变有目的会先降后稳） | `metrics_manager.py:109` |

> **v5 这一屏怎么读（it99 示例）**：`ball_lift=0.1214` 已非零（门控后仍有干净挑高，好苗头）、`ball_neck_contact=−0.0011` 几乎不触发（早期随机策略还没形成夹球）、`Mean action std=0.49` 探索充足、`Mean episode length=250` 确认无摔倒终止。**接下来盯三条**：`ball_lift`↑、`ball_vertical_velocity`↑、`ball_neck_contact`→0；若 `ball_lift` 死贴 0 而 std 先塌，则是探索不足，上 entropy / action-noise 杠杆而非加 iter。

---

## 八、验证实测记录

在本机（RTX 5090 / py3.12 / 根 .venv）实测：

| 检查项 | 命令 / 判据 | 结果 |
|---|---|---|
| 新任务注册可见 | `list_tasks()` 含 `Mjlab-BallKickHigh-Flat-MicroDuck`，与官方轻触版并存 | ✅ |
| v4 reward 覆盖正确 | `ball_lift` w=30、`ball_vertical_velocity` w=10；`upright`/`ball_drop`/`ball_forward_velocity`/`ball_speed_overshoot` 均 ABSENT；站姿项 0.5/0.5/0.3/0.5 | ✅ |
| v4 终止/curriculum | `fell_over` 已删（terminations 只剩 time_out/out_of_terrain_bounds/nan_state）；`action_rate_weight` ramp 已删 | ✅ |
| 非对称维度 | Actor 61（盲踢）/ Critic 80（多 19 维球特权） | ✅ 继承正确 |
| 训练管线跑通 | 64 envs × 1 iter | ✅ **EXIT=0** |
| 场景几何核实 | 球质量 15g（机器人 737g 的 1/49，一碰必飞）；球下半部 4.5~8cm 落在脚从地面(2.3cm)摆到髋(9.7cm)的包络内 | ✅ 够得着、不太重 |
| **v4 目标学起来了** | 512×300：`ball_lift` 0→0.159 陡升未饱和；rollout 探针球峰值挑高 2.1cm、drop 率 0.015% | ✅ **突破** |
| **v4 大训练终局** | 4096×23875：峰值挑高 12.1cm(mean)/13.0cm(max)、球心峰 0.21m；it3000 后 plateau；15 个负尖峰不致命 | ✅ **12cm 挑高 + plateau 定位** |
| **v4 的 12cm 是脖子夹球** | rollout 探针（model_23750）：峰值最近颈组 8/8、脚组 0/8；>5cm 的 1097 step 100% 贴颈 | ✅ **reward hacking 坐实** |
| **v5 门控清零 exploit** | v4 ckpt 喂 v5 奖励：高挑高 8.9→0.06cm（清零 99%）、颈接触 mask 0.990 | ✅ **exploit 废掉** |
| **v5 训练终局** | 4096×1000：`ball_lift`→1.35 未饱和、`ball_neck_contact`≈0；探针 model_1000 峰值挑高 mean 69.9/max 96.4cm、≥50cm 14/16、机制 16/16 脚腿 | ✅ **干净踢射达标** |
| **子模块零改动** | `git -C ../../../microduck_rl status --short` | ✅ **空（干净）** |

---

## 九、文件清单

```
ball_kick/
├── README.md              ← 本文档：框架图 + 模块讲解 + 新增 task 教程 + 五版排查实录 + v6 路线
├── high_kick_cfg.py       ← 【组装层】造 env cfg / rl cfg，覆盖 entities/events/rewards/terminations/curriculum
├── high_kick_rewards.py   ← 【reward 库】v4 机制盲 4 函数 + v5 机制感知 4 函数（颈接触门控/惩罚）
├── pedestal.xml           ← 【场景资产】垫高细桩 tee MJCF（圆柱 r=10mm、5kg、freejoint，顶面 z=0.045）
├── train_high_kick.py     ← 【训练入口】注册新 task + 复用 launch_training
├── play_high_kick.py      ← 【回放入口】本地 checkpoint 交互回放（复用 run_play，需 DISPLAY）
├── play_to_video.py       ← 【headless 回放】训练策略 rollout→逐帧渲染→H.264 mp4（相机跟球+球高标注）
├── preview_scene.py       ← 【预览工具】headless 离屏渲染场景（4 方位角拼图 PNG）
├── .gitignore             ← 忽略 logs/ / *.mp4 / preview_scene.png
└── notes/                 ← reward 迭代记录、曲线截图、踩坑（边做边写）
```

---

## 十、验证阶梯与后续路线

遵循「前一阶通过才进下一阶」：

1. ✅ **链路通**：注册可见 + reward 覆盖正确 + 1 iter EXIT=0 + 子模块零污染
2. ✅ **几何可达**：细桩 tee 让 toe 能扫进球下方（v3 修好，v4 rollout 探针确认球被向上挑而非拍落）
3. ✅ **奖励支配破除**：v4 重平衡后 `ball_lift` 0→0.159 陡升未饱和（球峰值挑高 2cm）
4. ✅ **加训练量（已执行并终止）**：4096 envs × 23875 iter → 峰值挑高 **12cm**，但 it3000 即 plateau；回放进一步暴露这 12cm 是**脖子夹球**（reward hacking，见 6.6）
5. ✅ **机制感知 reward v5（已执行并达标）**：门控+颈接触惩罚废掉 exploit；4096 envs × **1000 iter** → 干净踢射 **mean ~70cm / max 96cm / ≥50cm 14/16 / 机制 16/16 脚腿**
6. ⏭️ **v6：从「踢得高」到「踢得受控」（当前）**：见下方 10.1 改进方向
7. **（可选）导出**：复用子模块 `export.py` 导出 ONNX，理解 sim2real 的 normalizer 烘焙

### 10.1 v6 改进方向（按优先级）

v5 已解决「高」和「机制正确」，剩「受控 / 结晶 / 可迁移」。按杠杆性价比排序：

0. **确定性探针（先诊断再动训练）**：把采样换成**均值动作** rollout model_1000——若挑高大幅塌缩，说明技能依赖噪声（std 卡 1.57），v6 第一任务是「结晶」；若保持，则直接进形态塑形。这是一次性探针，不改训练。
1. **结晶策略（去噪声依赖）**：`Mean action std` 卡 1.57 走平=探索没收敛。杠杆：降 `init_std` / 加 std 衰减 schedule / 后程降 entropy coef。判据：确定性探针挑高 ≈ 随机探针。
2. **形态塑形（受控踢射替代翻扑）**：抬 `body_ang_vel`/`angular_momentum` 权重压全身翻腾；新增「接触瞬间躯干直立」reward（球向上速度>阈值的窗口内 trunk pitch 近 0 才给分）；新增「支撑脚着地」强制单腿踢站姿。目标：挑高可略降，先换动作可迁移。
3. **高度梯度重构（若还要更高）**：`max_lift=0.5` cap 使 >50cm 无梯度（94~114cm 属无引导过冲）；改 log/soft cap，或加**滞空时间（airtime）** reward 奖励「持续高」而非「瞬时高」。
4. **成功率 metric + 数值稳定**：`Episode_Metrics` 加二值成功 metric（峰值≥X cm）盯成功率而非只看均值；盯 `nan_state` 与 mean_reward 负尖峰（v4 曾 −1.99e8），必要时接触 damping/限力。
5. **sim2real 准备（远期）**：15g 球过轻不真实，加球质量/摩擦 DR；翻扑形态无法上真机，#2 是前提；必要时接触瞬间给 actor 喂球位姿验证上限。

---

## 十一、预期管理（物理边界）

- microduck 是 25cm / 737g 双足，球 70mm / **15g**（极轻，一碰必飞）。踢「远」（水平）比踢「高」（垂直挑射）容易得多——高踢要求摆动脚有向上的速度分量，单脚支撑时向上摆腿会破坏平衡。
- **v4 证明「垂直挑高」物理可达且能学**（12cm，但实为脖子夹球）；**v5 证明废掉 exploit 后脚腿踢射能更高**：1000 iter 即 mean ~70cm / max 96cm，已越过当初 0.3~0.5m stretch goal 的下半段。
- 当前边界不再是「高度」而是**形态与可部署性**：翻扑 launch、std 未结晶、action 极抖——v6 的主战场是「受控、结晶、可迁移」，而非继续加高度。
- 球 15g 是仿真便利（1/49 体重、一碰必飞），与真球手感差异大；sim→real 迁移结论 需经球质量 DR 重验。

# 口播文案 1

哟，今天给大家看个我最近捯饬出来的玩意儿，一台小型双足机器人，把小球往天上垂直踢。

机器人叫 microduck，身高二十五厘米，体重七百多克。球是个直径七厘米、只有十五克的小轻球，还不到机器人体重的五十分之一，一碰就飞。听着挺简单？踢远确实容易，踢高可太难了，单腿向上摆的那一下，机器人自己就得摔。

这事我迭代了整整五个版本。前两版球放在地上和台子上，练完发现球根本起不来，翻数据才明白，脚插不到球底下，几何上就不可能。第三版用细桩把球架高，几何通了，机器人却学会站着不动，毕竟站姿也得分。第四版把踢球的奖励狠狠调高，它真练出来了，球能起十几厘米。可回放越看越不对劲，球不是脚踢起来的，是被脖子夹着顶上去的。奖励只问球多高，不问谁碰的，它就专挑漏洞走。第五版让奖励认机制，球只要贴着脖子，分数直接作废。重训之后才叫干净，十六次测试全是脚腿起球，平均高度约七十厘米，最高冲到九十六厘米，快一米了。

五个版本最大的体会是，机器人不懂人话，只认奖励。规则写错，它一定找最省力的漏洞，治它的办法不是换更聪明的算法，而是把规则写对。

过程画面和回放视频我都存在项目里了，想看的评论区聊。


# 口播文案 2

一只25厘米高的小鸭子机器人，能把球踢到一米高吗？关注我视频号的朋友应该记得 microduck，之前给大家介绍过。今天我在本地给它搭了个训练场，用强化学习逼它学会一件事，把小球垂直踢上天。
球是直径7厘米、只有15克的轻球。训练全部在物理仿真里完成，一张显卡上并行跑4096个环境，相当于四千多个平行世界里的鸭子在同时练球。
强化学习的核心是奖励函数设计，你得用数字告诉策略什么叫做得好。我的目标是把球挑高，奖励就盯球的离地高度和向上的速度。这件事我迭代了五个版本。前两版输在场景几何，脚插不到球底下，练多久都白搭。第三版几何通了，策略却只肯站着刷站姿分。
第四版把踢球奖励拉高，球真起来了十几厘米，可回放一看，是脖子夹着球顶上去的。这就是圈内说的 reward hacking，奖励作弊，算法不理解你的意图，它只最大化数字。第五版我把奖励改成机制感知的，用几何距离判断球是否贴着脖子，贴上了高度分直接作废，再叠一项负惩罚。重训之后，十六次测试全部是脚腿把球踢起来的，平均高度约七十厘米，最高96厘米。
现在为了把小球踢得更高，microduck 是尽可能地在变换姿态，翻跟头、前扑、倒立，什么招都用上了。
我的体会就一句：强化学习里奖励函数就是产品规格，规格有漏洞，策略一定钻。与其换更强的算法，不如先把规则写对。

