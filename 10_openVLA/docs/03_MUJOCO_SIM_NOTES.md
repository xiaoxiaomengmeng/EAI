# MuJoCo 仿真笔记

本文记录 OpenVLA 工作区中 MuJoCo 仿真的关键概念、当前 `sim` 的实现方式，以及常见兼容性问题。

## 1. MuJoCo 在本项目中的角色

当前项目中有两类 MuJoCo 相关路径：

```text
LIBERO                         标准 benchmark，底层基于 MuJoCo / robosuite
sim                    自定义 MuJoCo demo，支持 Franka 和 WX250S
```

二者定位不同：

```text
LIBERO      用于标准仿真评测和成功率统计
sim 用于本地工程验证、控制适配、视频和日志调试
```

## 2. sim 场景流程

`sim` 的闭环流程：

```text
动态构建 MuJoCo 场景
  -> 加载机器人 XML
  -> 添加 table / red cube / camera / ee_site
  -> mujoco.MjModel / MjData 初始化
  -> Renderer 渲染 RGB
  -> OpenVLA 预测 action
  -> motion_mode 转 ctrl
  -> mj_step 推进仿真
  -> 保存 mp4 / json
```

核心文件：

```text
sim/mujoco_env.py
sim/runner.py
sim/robots/franka.py
sim/robots/wx250s.py
```

## 3. MuJoCo XML 基本概念

常见元素：

```text
body       刚体节点，表示机器人连杆或场景物体
joint      自由度，例如 revolute / slide / free joint
actuator   控制器，例如 position actuator
site       标记点，可用于末端位置、传感器或可视化
camera     相机，Renderer 根据它生成图像
asset      mesh / texture / material 等资源
worldbody  世界坐标系下的 body / camera / light
```

## 4. mujoco_menagerie 机器人模型

当前自定义仿真借用了 `mujoco_menagerie` 中的机器人模型。

Franka：

```text
franka_emika_panda/panda.xml
7 个 arm position actuator
finger_joint1 / finger_joint2 作为夹爪相关关节
panda hand 附近插入 ee_site
```

WX250S：

```text
trossen_wx250s/wx250s.xml
waist / shoulder / elbow / forearm_roll / wrist_angle / wrist_rotate
left_finger / right_finger 作为夹爪相关关节
gripper_link 附近插入 ee_site
```

## 5. ee_site 与 IK

`ee_site` 是当前 IK 控制的关键。它临时插入到夹爪前方，用作末端位置代表点。

IK 求解流程：

```text
mj_forward
  -> 获取 ee_site 当前世界坐标
  -> mj_jacSite 计算位置雅可比
  -> 阻尼最小二乘求关节增量
  -> clip max_joint_delta
  -> 写入 actuator ctrl
```

优势：

```text
不需要直接训练关节控制模型
可以把 OpenVLA 的 [dx, dy, dz] 解释为末端位置增量
多机器人可共用同一抽象
```

限制：

```text
当前只做位置 IK
姿态控制尚未完整使用
接触和抓取稳定性仍需额外逻辑
```

## 6. camera 与 policy input

当前仿真中相机有分工：

```text
policy camera  给 OpenVLA 的输入图像
video camera   保存 rollout 视频和调试视角
```

常用参数：

```bash
--policy-camera agentview
--video-camera overview
--camera <name>              # 同时设置 policy 和 video camera
```

注意：MuJoCo XML 中 `<camera>` 必须放在 `<worldbody>` 内，不能作为顶层元素。

## 7. headless 渲染

在服务器或无桌面环境运行 MuJoCo Renderer 时，建议设置：

```bash
export MUJOCO_GL=egl
```

该变量必须在导入 `mujoco` 前生效。否则可能出现 OpenGL platform library 未加载相关错误。

## 8. 常见 XML 兼容问题

当前工作区已遇到过这些问题：

```text
<global distance="..."> 属性在当前 MuJoCo schema 中不兼容
<camera> 不能作为顶层元素，必须放到 worldbody 内
完整 mujoco XML 不能直接 include 到另一个 worldbody 内
机器人 XML 应通过注入 asset/worldbody 方式构建场景
```

## 9. 输出与诊断

`sim` 默认输出：

```text
outputs/<robot>_openvla/<timestamp>.mp4
outputs/<robot>_openvla/<timestamp>.json
```

JSON 中重点字段：

```text
action
arm_delta
target_ctrl
ee_site_pos
gripper_pos
cube_pos
gripper_cube_distance
gripper_state
ik_info
timing
records[*].timing
```

其中 `gripper_cube_distance` 是判断末端是否靠近目标的关键指标。

## 10. 性能耗时统计

顶层 `timing` 常见字段：

```text
total_runtime_sec
policy_load_sec
processor_load_sec
model_load_sec
model_prepare_sec
env_init_sec
save_video_sec
save_json_sec
```

单步 `records[*].timing` 常见字段：

```text
render_sec
policy_predict_sec
preprocess_sec
model_forward_sec
postprocess_sec
motion_mode_sec
mujoco_step_sec
mujoco_step_avg_sec
ik_solve_sec
```

这些字段用于判断瓶颈在模型推理、IK、MuJoCo step 还是视频/JSON 保存。
