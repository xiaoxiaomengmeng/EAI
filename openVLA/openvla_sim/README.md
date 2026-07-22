# openvla_sim：OpenVLA Base 多机器人仿真 Demo

本目录是 OpenVLA 的自定义 MuJoCo 闭环仿真测试代码，支持 Franka Emika Panda 和 WX250S 双机器人平台，目标是验证：

```text
MuJoCo 场景 (Franka / WX250S)
  -> 固定相机 RGB 图像
  -> OpenVLA base 模型预测 7DoF action
  -> action_adapter 转换为 actuator ctrl
  -> MuJoCo step
  -> 保存视频和 action 日志
```

该 demo 不依赖 LIBERO 微调 checkpoint，也不代表标准 LIBERO 成功率评测。

## 0. OpenVLA 如何适配具体机器人

这套仿真代码的关键理解是：OpenVLA 通常不是靠 prompt 里出现 `Franka`、`UR5` 或 `WidowX` 这样的机器人名字来完成机械臂适配，而是靠动作统计、动作空间约定和机器人控制适配层完成对齐。

完整链路可以理解为：

```text
RGB 图像 + instruction
  -> OpenVLA 生成 7 个 action token
  -> 使用 unnorm_key 对应的 norm_stats 反归一化
  -> 得到连续 7DoF action: [dx, dy, dz, droll, dpitch, dyaw, gripper]
  -> action_adapter 转换为当前机器人 actuator ctrl
  -> MuJoCo 执行控制
```

因此，本项目里真正“告诉模型/代码该按哪种动作尺度解释输出”的位置主要是：

```text
PolicyConfig.unnorm_key = "bridge_orig"
```

`bridge_orig` 表示使用 OpenVLA base 模型中 BridgeData / WidowX 风格的动作统计做反归一化。它在 OpenVLA 官方 OXE 数据转换代码中对应 `Original Version of Bridge V2 from Project Website`，可理解为 BridgeData V2 原始版动作统计 key。它决定 action token 被解码成多大的 `dx/dy/dz/rotation/gripper` 连续值；同样的 token，换成不同数据集或机器人统计，物理动作尺度可能不同。当前本地 `openvla-7b` 支持的完整 `norm_stats` key 列表见根目录 [DAMO_README.md](file:///home/creator/ws/openVLA/DAMO_README.md)。

机器人名字本身主要影响的是控制适配层，而不是 OpenVLA 自然语言输入：

```text
--robot franka
  -> robots/franka.py 定义 actuator 顺序、home ctrl、gripper 范围、ee_site 插入位置
  -> action_adapter / motion_modes/ik.py 将 7DoF action 转成 Franka 关节 position ctrl

--robot wx250s
  -> robots/wx250s.py 定义 WX250S 对应 actuator 和 gripper 配置
  -> action_adapter 使用同一 7DoF action 语义映射到 WX250S 控制量
```

对 Franka 来说，当前默认 `ik` 模式只使用 OpenVLA action 的前三维 `[dx, dy, dz]` 作为末端位置增量，通过 `ee_site + mj_jacSite + 阻尼最小二乘 IK` 求出前 6 个关节控制量；`action[6]` 用于夹爪控制。`droll/dpitch/dyaw` 已记录在日志中，但当前还没有完整姿态控制。

这也解释了当前 demo 的边界：base `openvla-7b` + `bridge_orig` 可以验证图像、语言、动作预测和控制适配链路，但如果要在 Franka 自定义场景或 LIBERO 中追求高成功率，通常还需要对应数据集/机器人的动作统计、微调 checkpoint，或者更完整的控制器与抓取状态机。

## 1. 目录结构

### action_adapter.py

负责把 OpenVLA 7DoF action 转换成机器人的 position actuator 控制目标，支持多机器人平台。

当前支持：

```text
--motion-mode ik      # 默认，使用 ee_site + MuJoCo Jacobian 做位置 IK
--motion-mode demo    # 演示友好映射，机械臂主体运动更明显
--motion-mode direct  # 原始直连映射，action[:6] 直接加到 6 个关节
```

### cli.py

命令行入口，负责解析参数并组装配置。

可以通过下面两种方式运行：

```bash
python run_openvla_sim.py
python -m openvla_sim
```

支持机器人选择（默认 franka）：

```bash
python -m openvla_sim --robot franka
python -m openvla_sim --robot wx250s
```

### config.py

集中管理默认路径和配置 dataclass，包括：

```text
DEFAULT_MODEL_PATH
SceneConfig
RenderConfig
PolicyConfig
ControlConfig
RolloutConfig
```

### motion_mode.py

三种运动模式的统一抽象文件，定义 `MotionMode` 基类和 `build_motion_mode()` 工厂函数，支持多机器人。

### motion_modes/ik.py

实现 `IKMotionMode`，负责末端 `ee_site` 位置 IK。

### motion_modes/demo.py

实现 `DemoMotionMode`，负责演示友好的手写运动映射。

### motion_modes/direct.py

实现 `DirectMotionMode`，负责 OpenVLA action 到关节增量的直接映射。

### mujoco_env.py

负责通用 MuJoCo 环境，支持多机器人平台：

```text
动态生成机器人 + table + red cube 场景
给机器人夹爪临时插入 ee_site
初始化 mujoco.MjModel / MjData
固定相机渲染 RGB 图像
使用 mj_jacSite 计算末端 site 位置雅可比
执行 data.ctrl 控制
记录 ee_site / gripper / red cube 的距离
```

### robots/

机器人配置目录，每个机器人一个配置文件：

```text
robots/base.py      # RobotConfig 基类定义
robots/franka.py    # Franka Emika Panda 配置
robots/wx250s.py    # WX250S 配置
```

### openvla_policy.py

负责 OpenVLA 模型：

```text
加载 AutoProcessor
加载 AutoModelForVision2Seq
使用 bridge_orig 执行动作反归一化
输出 7DoF action
```

### runner.py

负责闭环 rollout：

```text
render image
predict action
adapt action to ctrl
mujoco step
save mp4 / json
```

## 2. 为什么机械臂仍不一定能抓到 red cube

现在已经实现了末端 site + 位置 IK，但仍不能把它等同于完整抓取策略。核心原因有三个：

### 2.1 当前 IK 只控制末端位置，不控制完整抓取行为

OpenVLA 输出的动作语义更接近末端执行器 delta pose：

```text
[dx, dy, dz, droll, dpitch, dyaw, gripper]
```

但 WX250S MuJoCo 模型当前接收的是关节 position actuator：

```text
[waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate, gripper]
```

当前已经在 `wx250s/gripper_link` 前方临时插入：

```text
ee_site
```

并通过 MuJoCo 的 `mj_jacSite` 计算位置雅可比，用阻尼最小二乘求解关节增量：

```text
OpenVLA [dx, dy, dz]
  -> ee_site 目标位置
  -> mj_jacSite 位置雅可比
  -> 阻尼最小二乘 IK
  -> 关节 position actuator ctrl
```

但这个 IK 目前只控制末端位置，没有控制：

```text
末端姿态
夹爪对齐方向
接触稳定性
闭合时机
抓取成功判定
```

所以它比 demo/direct 更接近真实机器人控制，但仍不是完整 pick-and-place 策略。

### 2.2 red cube 位置会影响难度

之前 red cube 在：

```text
x=0.34, y=0.04, z=0.055
```

对当前非 IK 映射来说偏难。现在默认改为更居中的：

```text
x=0.28, y=0.0, z=0.055
```

可以通过参数调整：

```bash
--cube-x 0.28 --cube-y 0.0 --cube-z 0.055
```

### 2.3 openvla-7b base 没有针对当前仿真场景微调

当前使用的是 base 模型：

```text
/home/creator/.cache/modelscope/models/zixiaoBios--openvla-7b/snapshots/master
```

它与 BridgeData / WidowX 更接近，但没有针对当前自定义 MuJoCo 场景、相机视角、cube 位置和控制器微调。

因此第一版目标是工程链路跑通，而不是成功抓取。

## 3. 当前改进点

相比最早的单文件版本，现在有这些改进：

```text
代码拆分到 openvla_sim/
支持多机器人平台：Franka Emika Panda（默认）、WX250S
新增 --robot 参数选择机器人
新增 robots/ 目录管理机器人配置
cube 默认放近、放中间
新增 --cube-x / --cube-y / --cube-z / --cube-size
新增 ee_site，位于夹爪前方
新增 --motion-mode ik，使用 MuJoCo site Jacobian 做位置 IK
新增 --reach-bias，让 IK/demo 模式更容易向 red cube 靠近
日志新增 ee_site_pos、ik_info、gripper_cube_distance、gripper_state
保留旧入口 run_openvla_sim.py
支持 python -m openvla_sim
```

`gripper_cube_distance` 是判断机械臂是否靠近目标的关键指标。如果视频看不清，可以直接看 JSON 日志中这个值是否下降。

## 4. 借用了哪些关键信息

本实现不是直接复制 OpenVLA / LIBERO 官方评测脚本，而是借用了以下关键信息：

### 4.1 OpenVLA 论文与模型信息

```text
BridgeData V2 / WidowX 是 OpenVLA 论文中的重要评估平台
openvla-7b base 模型包含 bridge_orig norm_stats
OpenVLA action 语义接近 [dx, dy, dz, droll, dpitch, dyaw, gripper]
```

因此本 demo 使用：

```text
unnorm_key = bridge_orig
```

### 4.2 mujoco_menagerie 的机器人模型信息

**Franka Emika Panda**（默认）：

从 `franka_emika_panda/panda.xml` 借用了：

```text
关节名称: actuator1 ~ actuator7
夹爪关节: finger_joint1 / finger_joint2
actuator 顺序: 7 个 arm position actuator + 1 个 gripper actuator
home keyframe: qpos / ctrl 初始姿态
panda_hand 结构: 用于插入 ee_site
```

**WX250S**：

从 `trossen_wx250s/wx250s.xml` 借用了：

```text
关节名称: waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate
夹爪关节: left_finger / right_finger
actuator 顺序: 6 个 arm position actuator + 1 个 gripper actuator
home keyframe: qpos / ctrl 初始姿态
gripper_link 结构: 用于插入 ee_site
```

### 4.3 MuJoCo 官方能力

使用 MuJoCo Python API：

```text
mujoco.Renderer          # headless RGB 渲染
mujoco.mj_jacSite        # 计算 ee_site 位置雅可比
mujoco.mj_step           # 推进仿真
model.jnt_qposadr/dofadr # 找到关节 qpos/dof 索引
```

### 4.4 LIBERO 借鉴点

没有依赖 LIBERO benchmark，但借鉴了它的 rollout 思路：

```text
reset env
render observation image
query policy action
step env
record video / logs
```

## 5. 运行前提

```bash
cd /home/creator/ws/openVLA
source .venv/bin/activate
```

确认不要使用 conda base：

```bash
which python
python --version
```

## 6. Dry-run 验证

不加载 7B 模型，只验证 MuJoCo、渲染、控制和视频保存：

```bash
python -m openvla_sim \
  --dry-run \
  --policy-steps 2 \
  --sim-steps-per-policy 2
```

兼容旧入口：

```bash
python run_openvla_sim.py \
  --dry-run \
  --policy-steps 2 \
  --sim-steps-per-policy 2
```

## 7. 运行真实 OpenVLA base 闭环

最小真实模型验证：

```bash
python -m openvla_sim \
  --policy-steps 1 \
  --sim-steps-per-policy 1 \
  --attn-implementation sdpa
```

完整一点的 IK rollout：

```bash
python -m openvla_sim \
  --policy-steps 20 \
  --sim-steps-per-policy 30 \
  --motion-mode ik \
  --max-joint-delta 0.08 \
  --reach-bias 0.02 \
  --ik-position-gain 1.0 \
  --ik-max-position-delta 0.05 \
  --ik-damping 0.05 \
  --cube-x 0.28 \
  --cube-y 0.0 \
  --attn-implementation sdpa
```

如果想对比旧的演示映射：

```bash
python -m openvla_sim \
  --motion-mode demo \
  --joint-gain 3.0 \
  --max-joint-delta 0.08
```

## 8. 常用调参

### 8.1 调整 IK 末端运动幅度

```bash
--ik-position-gain 1.5 --ik-max-position-delta 0.06
```

如果末端抖动或动作过大：

```bash
--ik-position-gain 0.5 --ik-max-position-delta 0.03 --ik-damping 0.08
```

### 8.2 调整 demo/direct 动作幅度

```bash
--joint-gain 4.0 --max-joint-delta 0.10
```

如果动作过大：

```bash
--joint-gain 2.0 --max-joint-delta 0.04
```

### 8.3 调整目标位置

```bash
--cube-x 0.24 --cube-y 0.0
```

如果 cube 太远，先降低 `cube-x`；如果偏左/右，调整 `cube-y`。

### 8.4 关闭 reach bias

```bash
--reach-bias 0.0
```

这可以更纯粹地观察 OpenVLA action 本身对 IK 或 demo 映射的影响。

### 8.5 保存单帧图像

```bash
--save-frames
```

## 9. 输出位置

默认输出：

```text
/home/creator/ws/openVLA/outputs/wx250s_openvla/
```

包括：

```text
wx250s_openvla_<timestamp>.mp4
wx250s_openvla_<timestamp>.json
```

JSON 中重点看：

```text
timing                  # 顶层汇总耗时
action
arm_delta
ik_info
target_ctrl
ee_site_pos
gripper_pos
cube_pos
gripper_cube_distance
records[*].timing       # 每个 policy step 的细分耗时
```

## 10. 性能耗时统计

每次 rollout 的 JSON 顶层会写入 `timing` 汇总字段，常用字段包括：

```text
total_runtime_sec       # 从 run() 开始到最终 JSON 写完的总耗时
pre_save_runtime_sec    # 保存视频/JSON 前的运行耗时
env_init_sec            # MuJoCo 环境初始化耗时
policy_load_sec         # policy 加载总耗时；真实模型时约等于加载 OpenVLA
processor_load_sec      # AutoProcessor 加载耗时
model_load_sec          # AutoModelForVision2Seq 加载耗时
model_prepare_sec       # model.to/eval 等准备耗时
save_video_sec          # MP4 保存耗时
save_json_sec           # JSON 写入耗时
save_outputs_sec        # 保存输出总耗时
```

每个 `records[*].timing` 会记录单步耗时：

```text
render_sec              # 渲染给 OpenVLA 的 RGB 图像耗时
frame_to_image_sec      # numpy frame 转 PIL image 耗时
policy_predict_sec      # 一次 policy.predict 总耗时
preprocess_sec          # prompt/image -> tensor 耗时
model_forward_sec       # model.predict_action 耗时
postprocess_sec         # action 转 numpy 等后处理耗时
motion_mode_sec         # action -> ctrl 的运动模式转换耗时
mujoco_step_sec         # 当前 action 下 MuJoCo 连续 step 总耗时
mujoco_step_avg_sec     # 单个 MuJoCo step 平均耗时
diagnostics_sec         # 末端/cube 距离诊断耗时
step_total_sec          # 当前 policy step 总耗时
```

当 `--motion-mode ik` 时，还会记录：

```text
ik_solve_sec            # IK 求解总耗时
ik_forward_sec          # mj_forward 耗时
ik_jacobian_sec         # mj_jacSite 耗时
ik_linear_solve_sec     # 阻尼最小二乘线性求解耗时
```

顶层还会自动生成 `avg_*` 和 `total_*` 汇总，例如：

```text
avg_policy_predict_sec
total_policy_predict_sec
avg_mujoco_step_sec
total_mujoco_step_sec
avg_mujoco_step_avg_sec
avg_motion_mode_sec
```

## 11. 优化记录

### 11.1 2026-07-21 优化

#### 问题 1：红色方块尺寸过大，夹爪无法抓住

**现象**：默认 `cube_size=0.03`，实际边长为 6cm，但 WX250S 夹爪最大开合距离约 4.4cm（单边滑动范围 0.015~0.037），导致方块无法被夹爪抓住。

**修复**：将默认 `cube_size` 从 `0.03` 修改为 `0.015`（边长 3cm），确保夹爪能够抓住。

**文件**：`config.py`

```python
cube_size: float = 0.015  # 原为 0.03
```

#### 问题 2：夹爪过早闭合

**现象**：从运行日志可以看到，step=001 时 action[6]=0.996（接近闭合），但距离方块还有 0.19m，导致机械臂在距离目标较远时就闭合夹爪，无法抓住目标。

**修复**：增加基于距离的夹爪延迟闭合逻辑：当末端到方块的距离大于阈值时，强制保持夹爪张开。

**新增参数**：`--gripper-close-threshold`，默认值 0.05（5cm）

**文件**：
- `motion_mode.py`: 修改 `gripper_ctrl()` 方法，增加距离判断
- `config.py`: 在 `ControlConfig` 中增加 `gripper_close_threshold` 参数
- `cli.py`: 增加命令行参数 `--gripper-close-threshold`
- `motion_modes/ik.py`, `demo.py`, `direct.py`: 更新 `to_ctrl()` 方法签名，传递距离参数
- `action_adapter.py`: 更新 `to_ctrl()` 方法签名，传递距离参数
- `runner.py`: 在控制前获取距离并传递给 action adapter，同时记录夹爪状态到 JSON 日志

**逻辑**：
```python
def gripper_ctrl(self, gripper_action: float, distance: float | None = None) -> float:
    gripper_open = 0.037
    gripper_closed = 0.015
    if distance is not None and distance > self.config.gripper_close_threshold:
        return gripper_open
    return gripper_open if gripper_action > 0.5 else gripper_closed
```

#### 问题 3：缺少夹爪状态诊断信息

**现象**：JSON 日志中没有记录夹爪的开合状态，难以分析抓取失败原因。

**修复**：在每个 policy step 的记录中增加 `gripper_ctrl`（夹爪控制值）和 `gripper_state`（"open" 或 "closed"）字段。

**文件**：`runner.py`

**新增字段**：
```json
{
  "gripper_ctrl": 0.037,
  "gripper_state": "open"
}
```

#### 问题 4：ee_site 位置与实际夹爪抓取点不一致

**现象**：ee_site 设置在 0.11m 处，但实际夹爪接触点应该在 0.108m 处（gripper_link 位置 0.065m + finger_link 位置 0.066m + 接触点位置 0.042m - 几何偏移 ≈ 0.108m）。位置偏差导致 IK 目标点与实际抓取点不重合，机械臂在距离方块约 1.9cm 时就停止。

**修复**：将 ee_site 的位置从 0.11m 修改为 0.108m。

**文件**：`mujoco_env.py`

```python
site_xml = '                  <site name="ee_site" pos="0.108 0 0" size="0.008" rgba="0 1 0 1"/>\n'
```

#### 问题 5：reach_bias 太小，机械臂难以到达目标

**现象**：reach_bias=0.02，每次最多移动 2cm，机械臂在接近方块时速度太慢，容易陷入局部最优。

**修复**：将 reach_bias 从 0.02 增加到 0.03。

**文件**：`config.py`

```python
reach_bias: float = 0.03  # 原为 0.02
```

#### 问题 6：夹爪闭合后没有下压动作

**现象**：夹爪闭合后机械臂没有向下施加压力，导致方块可能没有被牢固抓住。

**修复**：增加 `grip_down_offset` 参数，在夹爪闭合后自动向下移动一定距离（默认 1.5cm）。

**新增参数**：`--grip-down-offset`，默认值 0.015（1.5cm）

**文件**：
- `config.py`: 在 `ControlConfig` 中增加 `grip_down_offset` 参数
- `cli.py`: 增加命令行参数 `--grip-down-offset`
- `mujoco_env.py`: 修改 `solve_ik_ctrl()` 方法，增加下压偏移逻辑
- `motion_modes/ik.py`: 判断夹爪状态并传递下压参数

**逻辑**：
```python
if gripper_closed and grip_down_offset > 0:
    target_pos[2] -= grip_down_offset
```

### 11.2 优化后的运行效果

优化后，机械臂会：
1. 根据 instruction 语义执行抓取任务（关键优化：修改 instruction 为 "pick up the red cube and lift it"）
2. 在距离方块较远时保持夹爪张开（安全机制）
3. 只有当距离小于 `gripper_close_threshold`（默认 5cm）时才允许闭合夹爪
4. 夹爪闭合后自动向下移动 `grip_down_offset`（默认 1.5cm）确保牢固抓住
5. JSON 日志中记录每个 step 的夹爪状态，便于分析
6. ee_site 位置更精确，IK 目标点与实际抓取点一致

### 11.4 关键优化：修改 instruction 而非代码

**核心原则**：OpenVLA 是 VLA (Vision-Language-Action) 模型，其 action 输出完全受 instruction 语义引导。与其在代码层硬编码抓取状态机，不如直接修改 instruction 让模型知道要做什么。

**优化前后对比**：

| 阶段 | 之前 | 现在 |
|------|------|------|
| instruction | `"move the robot arm toward the red cube"` | `"pick up the red cube and lift it"` |
| 模型行为 | 只会靠近方块 | 会靠近、闭合夹爪、抬起 |
| 代码逻辑 | 添加了复杂的 grasp_state 状态机 | 简单的距离延迟闭合 + 下压 |

**设计哲学**：
- **instruction 驱动**：VLA 模型的核心能力就是理解自然语言指令，应该充分利用这一点
- **代码层只做安全兜底**：`gripper_close_threshold` 作为安全机制，防止模型在远距离时误闭合夹爪
- **避免过度工程化**：不需要在代码层模拟抓取流程，让模型自己决策

### 11.3 新增参数说明

```bash
--gripper-close-threshold 0.05  # 夹爪闭合距离阈值（米）
--grip-down-offset 0.015        # 夹爪闭合后下压距离（米）
```

设置为 0 可禁用相关功能，恢复原始行为：
```bash
--gripper-close-threshold 0.0  # 禁用距离限制
--grip-down-offset 0.0         # 禁用下压动作
```

## 12. 下一步建议

现在已经完成末端 site + 位置 IK，以及夹爪延迟闭合逻辑。若要进一步提高抓取效果，下一步建议：

1. 加入末端姿态控制，而不只是位置 IK；
2. 根据 cube 位置设计接近、下压、闭合、抬起的阶段式策略；
3. 更精确地设置 ee_site 到夹爪中心的位置；
4. 调整相机视角，使图像更接近 BridgeData / WidowX 分布；
5. 增加接触检测和 grasp success 判定；
6. 如果要追求成功率，再考虑微调或使用任务专用 checkpoint。
