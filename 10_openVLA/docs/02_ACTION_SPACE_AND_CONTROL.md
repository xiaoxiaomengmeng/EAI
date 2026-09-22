# OpenVLA Action Space 与机器人控制

本文集中说明 OpenVLA 输出动作的语义、`norm_stats` / `unnorm_key` 的作用，以及动作如何映射到 MuJoCo 机器人控制量。

## 1. 7D action 语义

OpenVLA 常见输出是 7D action：

```text
[dx, dy, dz, droll, dpitch, dyaw, gripper]
```

含义：

```text
dx / dy / dz             末端执行器位置增量
droll / dpitch / dyaw    末端执行器姿态增量
gripper                  夹爪开合信号
```

它不是天然等同于某个机器人所有关节角，也不能直接喂给所有 MuJoCo actuator。

## 2. norm_stats 与 unnorm_key

`norm_stats` 是动作归一化统计。训练时，真实机器人动作会被归一化；推理时，模型输出需要根据同一套统计反归一化。

`unnorm_key` 是选择哪套动作统计的 key。

例如：

```text
bridge_orig
```

表示 BridgeData V2 原始版动作统计，即 `Original Version of Bridge V2 from Project Website`。

关键理解：

```text
prompt 里的机器人名字不会自动改变动作尺度
unnorm_key 才决定 action token 被解释成多大的连续动作
robot adapter / IK / controller 才决定具体机器人怎么执行
```

## 3. base 模型与 LIBERO key

当前 `openvla-7b` base 模型适合用：

```text
bridge_orig
```

LIBERO 评测脚本会使用：

```text
libero_spatial
libero_object
libero_goal
libero_10
libero_90
```

如果 base 模型中没有这些 key，官方脚本会报错，或即使绕过检查也会出现动作尺度不匹配。

## 4. 机器人适配链路

完整链路可以理解为：

```text
RGB 图像 + instruction
  -> OpenVLA 生成 action token
  -> unnorm_key 对应 norm_stats 反归一化
  -> 连续 7D action
  -> action adapter
  -> motion_mode / IK / gripper controller
  -> MuJoCo actuator ctrl
```

当前项目中对应实现：

```text
sim/openvla_policy.py      # 模型加载和 predict_action
sim/action_adapter.py      # action -> ctrl 统一入口
sim/motion_mode.py         # motion_mode 抽象和工厂
sim/motion_modes/ik.py     # ee_site + Jacobian IK
sim/motion_modes/demo.py   # 演示友好映射
sim/motion_modes/direct.py # 直接关节增量映射
```

## 5. motion_mode 三种模式

当前 CLI 统一使用 `--motion-mode`。

```text
ik      默认推荐，使用 ee_site + MuJoCo mj_jacSite 做位置 IK
demo    演示友好映射，让机械臂运动更明显
direct  action[:6] 直接映射为关节增量，用于对照
```

运行示例：

```bash
python -m sim --robot wx250s --motion-mode ik
python -m sim --robot wx250s --motion-mode demo
python -m sim --robot wx250s --motion-mode direct
```

## 6. IK 模式

IK 模式主要使用 OpenVLA action 的前三维：

```text
action[:3] = [dx, dy, dz]
```

控制流程：

```text
OpenVLA [dx, dy, dz]
  -> ee_site 目标位置
  -> mj_jacSite 位置雅可比
  -> 阻尼最小二乘 IK
  -> arm joint delta
  -> actuator position ctrl
```

当前 IK 只控制末端位置，尚未完整控制：

```text
末端姿态
夹爪对齐方向
接触稳定性
完整抓取成功判定
```

## 7. Franka 控制链路

Franka 配置由 `sim/robots/franka.py` 管理。

核心信息：

```text
7 个 arm position actuator
1 个 gripper actuator
使用 panda hand 附近插入 ee_site
默认可使用 ik / demo / direct 三种 motion_mode
```

在默认 IK 模式下，action 前三维被转换为末端位置目标，再由 MuJoCo Jacobian 求解关节增量。

## 8. WX250S 控制链路

WX250S 配置由 `sim/robots/wx250s.py` 管理。

核心信息：

```text
6 个 arm position actuator
1 个 gripper actuator
使用 gripper_link 附近插入 ee_site
BridgeData / WidowX 风格 action 与该平台语义更接近
```

这也是 `bridge_orig` 更自然适配 WX250S / WidowX 风格验证的原因。

## 9. 抓取逻辑边界

当前 `sim` 的目标是工程闭环验证，不等于完整抓取策略。

已支持：

```text
图像渲染
OpenVLA action 预测
action -> ctrl 映射
IK 位置控制
夹爪距离阈值限制
下压偏移
JSON 诊断记录
```

仍需进一步完善：

```text
姿态 IK
接触检测
grasp success 判定
阶段式 approach / grasp / lift 策略
任务专用微调或数据采集
```
