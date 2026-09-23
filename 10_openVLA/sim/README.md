# sim：OpenVLA Base × Franka Panda MuJoCo 闭环仿真

本目录是 OpenVLA 的自定义 MuJoCo 闭环仿真测试代码，**只使用 Franka Emika Panda**（对齐 LIBERO 的机器人平台，便于参照）。目标是验证：

```text
MuJoCo 场景 (Franka Panda)
  -> 固定相机 RGB 图像
  -> OpenVLA base 模型预测 7DoF action
  -> action_adapter / IK 转换为 actuator ctrl
  -> MuJoCo step
  -> 保存视频和 action 日志
```

该 demo 不依赖 LIBERO 微调 checkpoint，也**不代表标准 LIBERO 成功率评测**（那条链路见 [../scripts/02_libero_eval.py](../scripts/02_libero_eval.py) 与 [../docs/reports/libero_eval/](../docs/reports/libero_eval/README.md)）。

## 0. OpenVLA 如何适配 Franka

关键理解：OpenVLA 通常不是靠 prompt 里出现 `Franka` 这样的机器人名字来完成机械臂适配，而是靠动作统计（`norm_stats`）、动作空间约定和机器人控制适配层完成对齐。

完整链路：

```text
RGB 图像 + instruction
  -> OpenVLA 生成 7 个 action token
  -> 使用 unnorm_key 对应的 norm_stats 反归一化
  -> 得到连续 7DoF action: [dx, dy, dz, droll, dpitch, dyaw, gripper]
  -> action_adapter / IK 转换为 Franka actuator ctrl
  -> MuJoCo 执行控制
```

真正“告诉代码该按哪种动作尺度解释输出”的位置是：

```text
PolicyConfig.unnorm_key = "bridge_orig"
```

`bridge_orig` 表示使用 OpenVLA base 模型中 BridgeData / WidowX 风格的动作统计做反归一化，在 OpenVLA 官方 OXE 数据转换代码中对应 `Original Version of Bridge V2 from Project Website`。它决定 action token 被解码成多大的 `dx/dy/dz/rotation/gripper` 连续值。当前本地 `openvla-7b` 支持的完整 `norm_stats` key 列表见 [docs/09_ENVIRONMENT_RTX5090.md](../docs/09_ENVIRONMENT_RTX5090.md)。

机器人配置影响的是控制适配层：

```text
--robot franka（唯一选项，默认）
  -> robots/franka.py 定义 actuator 顺序、home ctrl、gripper 范围、ee_site 插入位置
  -> action_adapter / motion_modes/ik.py 将 7DoF action 转成 Franka 关节 position ctrl
```

当前默认 `ik` 模式只使用 OpenVLA action 的前三维 `[dx, dy, dz]` 作为末端位置增量，通过 `ee_site + mj_jacSite + 阻尼最小二乘 IK` 求出前 7 个关节控制量；`action[6]` 用于夹爪控制。`droll/dpitch/dyaw` 已记录在日志中，但当前还没有完整姿态控制。

这也界定了 demo 的边界：base `openvla-7b` + `bridge_orig` 可以验证图像、语言、动作预测和控制适配链路，但若要在自定义场景中追求高成功率，通常还需要对应数据集的动作统计、微调 checkpoint 或更完整的抓取状态机。

## 1. 目录结构

```text
sim/
├── cli.py             # 命令行入口（python -m sim）
├── config.py          # 默认路径与配置 dataclass
├── mujoco_env.py      # MuJoCo 环境、场景注入、相机渲染、IK 求解
├── openvla_policy.py  # OpenVLA 模型加载与推理
├── action_adapter.py  # 7DoF action -> actuator ctrl 分发
├── motion_mode.py     # MotionMode 基类 + build_motion_mode() 工厂
├── motion_modes/
│   ├── ik.py          # IKMotionMode：ee_site 位置 IK（默认）
│   ├── demo.py        # DemoMotionMode：演示友好映射
│   └── direct.py      # DirectMotionMode：action[:6] 直接映射关节
├── robots/
│   ├── base.py        # RobotConfig 基类
│   └── franka.py      # Franka Emika Panda 配置（唯一机器人）
├── runner.py          # 闭环 rollout 主循环、视频/JSON 输出
└── scenes/
    └── tabletop_scene.xml  # 桌面 + 红方块场景模板
```

### 运动模式（`--motion-mode`）

```text
ik      # 默认，使用 ee_site + MuJoCo Jacobian 做位置 IK
demo    # 演示友好映射，机械臂主体运动更明显
direct  # 原始直连映射，action[:6] 直接加到关节
```

### robots/franka.py 关键字段

```text
robot_dir       assets/mujoco/mujoco_menagerie/franka_emika_panda（panda.xml）
actuator_names  actuator1 ~ actuator7（臂）+ actuator8（夹爪）
arm_joint_names joint1 ~ joint7
gripper_body    hand
ee_site_pos     (0, 0, 0.108)  插入到 left_finger 前
gripper 范围    255=全开 / 0=闭合（home_ctrl 夹爪位=255，即初始张开）
```

> `target_ctrl` 长度为 8：索引 0-6 是 7 个臂关节，索引 7（即 `[-1]`）才是夹爪控制量；读写夹爪一律用 `target_ctrl[-1]`，误用 `[6]` 会把臂关节当夹爪、产生“夹爪全程闭合”的假象。

## 2. 为什么机械臂仍不一定能抓到 red cube

已实现末端 site + 位置 IK，但仍不是完整抓取策略，核心原因：

### 2.1 当前 IK 只控制末端位置

OpenVLA 输出的动作语义接近末端执行器 delta pose `[dx, dy, dz, droll, dpitch, dyaw, gripper]`，当前实现：

```text
OpenVLA [dx, dy, dz]
  -> ee_site 目标位置
  -> mj_jacSite 位置雅可比
  -> 阻尼最小二乘 IK
  -> 前 7 关节 position actuator ctrl
```

IK 只控制末端位置，未控制：末端姿态、夹爪对齐方向、接触稳定性、闭合时机、抓取成功判定。因此它比 demo/direct 更接近真实控制，但仍不是完整 pick-and-place 策略。

### 2.2 red cube 位置与尺寸影响难度

当前默认（`config.py` 的 `SceneConfig`）：

```text
cube_x=0.42  cube_y=-0.16  cube_z=0.045  cube_size=0.025（边长约 5cm，适配 Panda 夹爪）
```

可通过参数调整：

```bash
--cube-x 0.30 --cube-y 0.0 --cube-z 0.045 --cube-size 0.025
```

> 红方块尺寸建议 0.025（5cm）或 0.02（4cm）以适配 Panda 夹爪，避免过大导致夹爪无法闭合。

### 2.3 openvla-7b base 未针对当前场景微调

base 模型与 BridgeData / WidowX 更接近，但没有针对当前自定义 MuJoCo 场景、相机视角、cube 位置和控制器微调。因此第一版目标是工程链路跑通，而不是稳定抓取。

## 3. 运行前提

```bash
cd /home/creator/ws/10_openVLA
source .venv/bin/activate
which python && python --version   # 确认不是 conda base
export MUJOCO_GL=egl                # 无显示器时（须在 import mujoco 前设置）
```

## 4. Dry-run 验证

不加载 7B 模型，只验证 MuJoCo、渲染、控制和视频保存：

```bash
python -m sim --dry-run --policy-steps 2 --sim-steps-per-policy 2
```

## 5. 运行真实 OpenVLA base 闭环

最小真实模型验证：

```bash
python -m sim --policy-steps 1 --sim-steps-per-policy 1 --attn-implementation sdpa
```

完整一点的 IK rollout：

```bash
python -m sim \
  --policy-steps 20 \
  --sim-steps-per-policy 30 \
  --motion-mode ik \
  --max-joint-delta 0.08 \
  --reach-bias 0.03 \
  --ik-position-gain 1.0 \
  --ik-max-position-delta 0.05 \
  --ik-damping 0.05 \
  --cube-x 0.30 \
  --cube-y 0.0 \
  --attn-implementation sdpa
```

对比旧的演示映射：

```bash
python -m sim --motion-mode demo --joint-gain 3.0 --max-joint-delta 0.08
```

## 6. 常用调参

### 6.1 IK 末端运动幅度

```bash
--ik-position-gain 1.5 --ik-max-position-delta 0.06   # 加大
--ik-position-gain 0.5 --ik-max-position-delta 0.03 --ik-damping 0.08   # 末端抖动时收敛
```

### 6.2 demo/direct 动作幅度

```bash
--joint-gain 4.0 --max-joint-delta 0.10   # 加大
--joint-gain 2.0 --max-joint-delta 0.04   # 动作过大时收敛
```

### 6.3 目标位置

```bash
--cube-x 0.30 --cube-y 0.0   # cube 太远先降 cube-x；偏左右调 cube-y
```

### 6.4 关闭 reach bias / 保存单帧

```bash
--reach-bias 0.0    # 更纯粹观察 OpenVLA action 本身对映射的影响
--save-frames       # 保存单帧图像
```

### 6.5 夹爪安全兜底

```bash
--gripper-close-threshold 0.05   # 末端离 cube 大于该距离时强制夹爪张开（防远距离误闭合）
--grip-down-offset 0.025         # 夹爪闭合后自动下压距离，确保牢固抓住
```

设为 0 可禁用相关功能，恢复原始行为：

```bash
--gripper-close-threshold 0.0   # 禁用距离限制
--grip-down-offset 0.0          # 禁用下压动作
```

## 7. 输出位置

默认输出：

```text
/home/creator/ws/10_openVLA/outputs/franka_openvla/
├── franka_openvla_<timestamp>.mp4
├── franka_openvla_<timestamp>.json
└── franka_openvla_<timestamp>.log
```

JSON 重点字段：

```text
timing                  # 顶层汇总耗时
action / arm_delta / ik_info / target_ctrl
ee_site_pos / gripper_pos / cube_pos
gripper_cube_distance   # 末端到方块距离，判断是否靠近目标的关键指标
gripper_ctrl / gripper_state   # 夹爪控制值与 open/closed 状态
records[*].timing       # 每个 policy step 的细分耗时
```

`gripper_cube_distance` 是判断机械臂是否靠近目标的关键指标；视频看不清时，直接看 JSON 里这个值是否下降。

## 8. 性能耗时统计

每次 rollout 的 JSON 顶层写入 `timing` 汇总字段：

```text
total_runtime_sec / pre_save_runtime_sec / env_init_sec
policy_load_sec / processor_load_sec / model_load_sec / model_prepare_sec
save_video_sec / save_json_sec / save_outputs_sec
```

每个 `records[*].timing` 记录单步耗时：

```text
render_sec / frame_to_image_sec / policy_predict_sec
preprocess_sec / model_forward_sec / postprocess_sec
motion_mode_sec / mujoco_step_sec / mujoco_step_avg_sec
diagnostics_sec / step_total_sec
```

`--motion-mode ik` 时额外记录：

```text
ik_solve_sec / ik_forward_sec / ik_jacobian_sec / ik_linear_solve_sec
```

顶层还会自动生成 `avg_*` / `total_*` 汇总（如 `avg_policy_predict_sec`、`total_mujoco_step_sec`）。

## 9. 关键调参逻辑（经验）

围绕“靠近—闭合—下压”的抓取过程，几处关键机制：

```text
1. 方块尺寸：cube_size 用 0.025（5cm）适配 Panda 夹爪，过大无法闭合。
2. 夹爪延迟闭合：末端到方块距离 > gripper_close_threshold 时强制夹爪张开，
   避免模型在远距离就闭合夹爪（gripper_ctrl 返回 gripper_open=255）。
3. 闭合后下压：夹爪闭合后按 grip_down_offset 向下移动，确保牢固抓住。
4. ee_site 精度：ee_site 位于 hand 坐标系 (0,0,0.108)，与实际夹爪抓取点对齐，
   避免 IK 目标点与真实抓取点错位导致提前停在方块上方。
5. instruction 驱动：VLA 的动作受 instruction 语义引导，优先用清晰指令
   （如 "pick up the red cube and lift it"）而非在代码层硬编码抓取状态机；
   代码层只做安全兜底（距离门限、下压）。
```

## 10. 下一步建议

1. 加入末端姿态控制，而不只是位置 IK；
2. 设计接近、下压、闭合、抬起的阶段式策略；
3. 更精确地设置 ee_site 到夹爪中心的位置；
4. 调整相机视角，使图像更接近 BridgeData / WidowX 分布；
5. 增加接触检测和 grasp success 判定；
6. 若追求成功率，再考虑微调或使用任务专用 checkpoint。
