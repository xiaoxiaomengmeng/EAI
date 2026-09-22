# BridgeData / WidowX 笔记

本文记录 BridgeData V2、WidowX / WX250S、`bridge_orig` 与 OpenVLA 官方实机评测脚本的关系。

## 1. BridgeData V2 是什么

BridgeData V2 是 Berkeley / Bridge 系列机器人操作数据集，主要包含 WidowX 系列机器人在真实场景中的操作轨迹。

OpenVLA base 模型与 BridgeData / WidowX 有较强关系，因此当前本地 base 推理和 `sim` 常用：

```text
unnorm_key = bridge_orig
```

## 2. bridge_orig

`bridge_orig` 全称可理解为：

```text
Original Version of Bridge V2 from Project Website
```

它是动作反归一化统计 key，不是机器人名字。

含义：

```text
使用 BridgeData V2 原始版动作统计
将模型输出 action token 解释为 Bridge/WidowX 风格 7D 连续动作
```

关键理解：

```text
bridge_orig 决定动作尺度
机器人控制器决定如何执行动作
prompt 里的 WidowX / Franka 文字不会替代控制适配层
```

## 3. WidowX 与 WX250S

OpenVLA 官方 Bridge 评测面向 WidowX / Bridge V2 真实机器人环境。

当前 `sim` 支持 WX250S，是因为：

```text
WX250S 和 WidowX 风格机械臂更接近
BridgeData / WidowX 动作语义更适合用 bridge_orig 做工程验证
mujoco_menagerie 中有可用的 trossen_wx250s 模型
```

但这不等于官方 Bridge 评测已经被离线复现。

## 4. 官方 Bridge/WidowX 入口

入口文件：

```text
repos/openvla/experiments/robot/bridge/run_bridgev2_eval.py
```

脚本说明中明确写的是：

```text
Runs a model in a real-world Bridge V2 environment.
```

也就是说它是实机评测，不是离线 MuJoCo 仿真。

## 5. 运行依赖

官方 Bridge/WidowX 评测需要：

```text
widowx_envs
WidowXClient
WidowX / Bridge V2 机器人控制服务
相机 topic 数据源
host_ip / port 连接配置
真实或兼容的机器人控制端
```

核心依赖位置：

```text
repos/openvla/experiments/robot/bridge/bridgev2_utils.py
repos/openvla/experiments/robot/bridge/widowx_env.py
```

其中 `widowx_env.py` 依赖：

```python
from widowx_envs.widowx_env_service import WidowXClient
```

## 6. 默认配置

`run_bridgev2_eval.py` 中常见默认配置：

```text
host_ip = localhost
port = 5556
camera_topics = [{"name": "/blue/image_raw"}]
max_episodes = 50
max_steps = 60
control_frequency = 5
unnorm_key = bridge_orig
```

如果本机没有对应服务，脚本无法仅靠代码目录完成评测。

## 7. 与 sim 的关系

`sim` 是当前项目自定义的 MuJoCo demo：

```text
MuJoCo 场景
Franka / WX250S
red cube
OpenVLA base
bridge_orig
ee_site + IK
视频和 JSON 日志
```

它适合做：

```text
工程链路验证
动作适配调试
模型加载和推理耗时统计
不同 motion_mode 对比
```

它不等同于：

```text
官方 Bridge V2 实机评测
真实 WidowX 成功率
离线 BridgeData 数据集评测
```

## 8. 如果后续要做实机评测

建议先做环境预检：

```text
是否安装 widowx_envs
是否能连接 host_ip:port
相机 topic 是否存在
WidowXClient 是否能 reset / get_observation / step_action
openvla-7b checkpoint 是否可加载
bridge_orig 是否存在于 norm_stats
```

在没有真实或兼容服务前，不建议直接运行 `run_bridgev2_eval.py`。
