# 人形机器人互动场景

新增三套 14 × 12 m 的概念场地，均有哑光拼接地板、开放入口和机器人侧向通道。关卡通过真实接触驱动物件；状态栏显示分数与下一项任务。

| 参数 | 场景 | 玩法 | 满分 |
|---|---|---|---:|
| `--scene strike` | STRIKE 保龄球 | 推或踢橙球，击倒六个球瓶 | 60 |
| `--scene orbit` | ORBIT 桌面滑盘 | 沿桌侧走动，将三色滑盘推入各自 DOCK 并松手 | 90 |
| `--scene switchyard` | SWITCHYARD 操作闯关 | 拨手柄 → 开门 → 推箱入坞 → 走到终点 | 100 |

原有 `empty`、`xixi`、`pulse` 继续支持；默认仍为空场景。两套模拟器的 `replay` 和 `live` 均可使用新参数，无需安装额外运行依赖。

## 启动

在仓库根目录运行，替换设备 IP 或回放路径：

```sh
# SONIC 策略仿真
g1_sonic/.venv/bin/python g1_sonic/launch_g1_sonic.py live DEVICE_IP --scene strike
g1_sonic/.venv/bin/python g1_sonic/launch_g1_sonic.py replay /path/to/session --scene orbit

# 轻量 IK 姿态回放
g1_sonic/.venv/bin/python g1_ik/launch_g1_ik.py replay /path/to/session --scene switchyard
```

机器人从 `(0, -4.7)` 出生，保持原根部高度。即使尚未收到输入，窗口也先显示场景与机器人。默认镜头按机器人尺寸取景；切换“场景总览”可查看整个关卡。左键拖动围绕机器人旋转，滚轮缩放。阴影默认关闭，需要时追加 `--render-shadows`。

SONIC 的重置按钮会还原机器人、所有道具和分数。轻量 IK 可重新启动回放开启新一轮。普通回放结束或等待数据时物理暂停；标定模式沿用原有锁帧步进机制。

## 计分条件

- **STRIKE：** 球瓶倾斜超过 55°，且瓶体原点低于 0.5 m，持续 0.25 秒后计 10 分。仅将瓶子平移不会计分。球半径 0.23 m，质量 1.1 kg；每个瓶子由独立圆柱与球体组成。
- **ORBIT：** 桌面高度 0.82 m；滑盘半径 0.12 m、质量 0.22 kg。整枚滑盘必须进入对应目标体积，速度向量范数低于 0.08，脱离其他动态物体及机器人接触并停留 0.5 秒，每盘计 30 分。可沿桌侧分段推动，无须一次滑完整张桌。
- **SWITCHYARD：** 手柄向前转过约 37°并保持 0.15 秒（20 分）；门转过约 57°并保持 0.15 秒（20 分）；整个箱子进入绿区、停稳脱手 0.5 秒（40 分）；机器人骨盆投影在终点中心半径 0.55 m 内停留 0.3 秒（20 分）。按此顺序计分；道具一直可以碰撞，不会通过脚本瞬移或自动开门。

每个目标每轮只奖励一次；时间回退会清空游戏状态。`session.game.update(...)` 返回本次新增奖励事件，`snapshot()` 提供分数、满分、已完成目标及下一项任务，便于后续接入声音或训练奖励。

## 资源与代码

运行资源分别位于 `assets/strike_arena`、`assets/orbit_arena`、`assets/switchyard_arena`，合计约 1.23 MB。每个目录只有三个 XML、`design.json`、说明和实际引用的二进制 MSH；不加载 `.blend`、OBJ、生成脚本、预览图、虚拟环境或贴图。MSH 转换保留导出后的顶点和三角面，未进行几何简化。磁盘体积不等同于运行内存。

| 代码 | 职责 |
|---|---|
| `sim_scenes/registry.py` | 场景选择与资源位置 |
| `sim_scenes/__init__.py` | 合并机器人与环境、保留关节索引、道具步进、状态栏入口 |
| `sim_scenes/task_game.py` | 倾倒、目标区域、关节阈值、有序路点及计分 |
| `sim_scenes/pulse_game.py` | 保留原 PULSE 规则 |

当前仍沿用 `PulseSession` 类名和窗口的 `pulse` 属性，以兼容现有调用；它们也管理三套新场景。新模型通过 `play_scene_<key>` 标记识别场景，通过 `design.json` 的 `prop_joints` 识别道具关节。

视觉 geoms 使用 group 1、碰撞掩码为零、质量为零；碰撞 geoms 使用 group 3 和基本几何。框架在编译前将环境 affinity 设置为兼容 G1 身体及 Wuji 指尖的掩码，隐藏 group 3 仅影响显示。固定地面为 z=0 的无限平面，14 × 12 m 是可视场地范围。

机器人原 76 个 qpos、75 个速度自由度和 29 个 actuator 顺序不变；STRIKE 追加 49/42，ORBIT 追加 21/18，SWITCHYARD 追加 9/8。道具均为被动关节，没有新增 actuator，不会占用策略控制向量。

独立加载：

```python
import mujoco
model = mujoco.MjModel.from_xml_path("assets/orbit_arena/scene.xml")
data = mujoco.MjData(model)
option = mujoco.MjvOption()
option.geomgroup[3] = 0  # 隐藏碰撞替身，碰撞仍然有效
option.sitegroup[:] = 0
mujoco.mj_step(model, data)
```

独立 MJCF 包含物理行为；计分需要实例化 `TaskGame(model, design)` 并在步进后调用 `update`，机器人走终点的任务还需传入 `actor_xy`。框架已自动完成这些连接。

## 验证范围与交互限制

验证覆盖独立 MuJoCo 加载、静态零自由度、地面落球、墙和入口射线、真实接触推动球/盘/箱及转动手柄/门、分层渲染、资源换目录加载。集成测试覆盖两套机器人加载器、实际 Wuji 指尖接触、首帧前显示、计分边界及旧场景回归。

SONIC 保留原有全身策略，手部目前直接写关节姿态；轻量 IK 固定机器人姿态推进道具。上述验证证明场景和接触逻辑可用，尚未证明真实遥操作员能稳定完成全关或实现可靠力控抓取。没有额外添加键盘行走、鼠标抓物、导航策略或自动通关逻辑。

```sh
QT_QPA_PLATFORM=offscreen g1_sonic/.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

制作文件、脚本、预览与验证报告单独交付在 `play_worlds` 制作包中，不属于运行时依赖。静态布局可编辑 Blender 后重新导出；动态预览集合不参与静态导出，修改动态物体必须同步其描述和任务配置，详见制作包说明。
