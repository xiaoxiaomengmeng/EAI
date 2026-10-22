# G1 轻量 IK 仿真

这个工程不跑策略网络，直接用 MuJoCo 的雅可比 + `lsq_linear` 解 G1 的 29 个本体自由度，适合验证人体到机器人的映射关系本身是否合理，或者调整两者的尺度差异。

- 身体：PICO 24 关节位置/旋转 + 左右手 tracker 旋转 → 每帧线性化 IK → G1 29 个本体电机
- 手部：手套 21 个关键点（`wrist` + 每指 4 点 × 5 指）→ 已有 `wuji-retargeting`（adaptive_analytical 解析解配置）→ Wuji Hand 40 关节（左右各 20）
- 仿真和界面：MuJoCo + PySide6，3D 视图旁边是人体和双手的 2D 线框；进入参数标定后锁定当前帧调参
- 固定 50 Hz；`qpos` 共 76 = 7 基座 + 29 本体 + 40 手部
- 纯 CPU，不需要 GPU，也不用 torch

IK 每帧解一次带阻尼和关节限位的最小二乘：位置任务追踪左右腕、踝、膝、肘，旋转任务约束躯干朝向和左右腕、踝的朝向，首帧迭代 30 次、之后每帧 8 次，关节速度上限 6.0 rad/s。

## 代码结构

入口 `launch_g1_ik.py` 只保留 CLI 与装配（~70 行），职责拆在同目录 `g1_*.py` 平铺模块（依赖单向，无循环）：

| 模块 | 职责 |
|---|---|
| `g1_config.py` | 常量表（关节/边/调参默认值）+ 资产与 wuji 路径；首次导入时把仓库根插入 `sys.path` |
| `g1_recording.py` | parquet 加载、设备消息解析、时间戳重采样（slerp/interpolate） |
| `g1_model.py` | `build_model`：G1 + Wuji Hand MJCF 拼装与场景注入 |
| `g1_ik.py` | `FullBodyIK` 全身 IK + `WujiMapper`/`set_hands` 手部重定向 |
| `g1_source.py` | `replay_frames`/`live_frames` 帧生成、`load_tuning`、跨线程帧总线 |
| `g1_window.py` | `SourceView`/`MuJoCoView`/`TuningWindow` 三个界面类 |

改名或搬移注意：`tests/test_scenes.py` 按文件路径加载本入口并取 `build_model`/`FullBodyIK`/`TUNING_DEFAULTS`（入口 re-export，三个名字不可删）；`tuning.json` 与 `REPO_ROOT` 均按 `__file__` 同目录/上两级推导，模块需与本入口同级平铺。

## 安装

```bash
cd g1_ik
bash create_env.sh
```

`create_env.sh` 会：建独立 `.venv`；把 wuji-retargeting（含 `wuji-description` 子模块）克隆到仓库根的 `3rd/`。G1 模型与网格已随仓库放在 `../assets/g1_description/`，无需另外准备。整个过程不读取 `$HOME` 下任何路径，也不依赖另外两个工程。

只想拉代码不建环境（如提前预下载），可单独运行：`bash fetch_wuji.sh`（幂等，已存在则跳过 clone 并补齐子模块）。

wuji-retargeting 声明了 `wuji_sdk` 依赖（只随真实手套/手部硬件提供的私有 wheel，不在 PyPI 上），所以 `create_env.sh` 先单独装 `nlopt` 和 `pin>=3.8.0` 两个求解器依赖，再用 `--no-deps` 装 wuji-retargeting。这个仿真只用到 `wuji_retargeting.Retargeter`，它从不 import `wuji_sdk`。

## 落盘回放

入参是具体的 `observation.data.vr_full_body_*.parquet` **文件**，不接受 session 目录：

```bash
.venv/bin/python launch_g1_ik.py \
  replay ../session_20260828_161120/data/chunk-000/observation.data.vr_full_body_000000.parquet
```

手部 tracker 数据按文件名把 `vr_full_body` 替换成 `vr` 自动找到；手套数据替换成 `glove_left` / `glove_right` 自动找到。这三个文件都是**必需**的。腕部位置仍取 24 关节数据，腕部旋转直接取 `left_controller` / `right_controller`，不做 tracker 到人体腕部的首帧旋转对齐。

注意这里和 `g1_sonic` 不同，后者可以直接传 session 目录。

## 实时控制

```bash
.venv/bin/python launch_g1_ik.py live 192.168.2.111
```

连接 `ws://设备IP:9990/subscribe?token=rynn_dev`，订阅 `vr` 和 `glove` 两个 channel，设备端各以 90 Hz 推送，脚本内部重采样到 50 Hz。token 可以用 `--token` 覆盖。

在回放或实时命令末尾加 `--xixi-scene` 会加载西溪办公场景；不加时仍使用原来的空场景：

```bash
.venv/bin/python launch_g1_ik.py replay <full_body.parquet> --xixi-scene
```

## 标定与调参

收到第一帧后点“进入参数标定”，界面会锁定当前动作：落盘回放同时暂停；实时模式继续在后台接收，但画面停在进入标定时的当前帧。拖动滑杆会用同一帧持续重新求解，便于直接观察 G1 的变化。右侧原始人体中，黄色圆圈标出 8 个位置任务，红、绿、蓝轴标出 5 个旋转任务。

1. 启动回放或实时模式，等右侧出现人体和手套数据。
2. 选一帧站姿稳定、双臂和双腿都容易观察的动作，点“进入参数标定”。
3. 先调身体比例：臂长、腿长、肩宽和髋宽；再调任务权重、阻尼与左右手腕 XYZ 旋转。每次只改变一个滑杆，直接观察同一帧上的 G1 变化。
4. 如果人体方向或尺寸基准不正确，点“重新标定身体”，程序会以当前锁定帧重新计算。
5. 点“保存并继续”，结果写入 `tuning.json`，回放或实时控制随后继续。下次启动会自动加载保存值。

界面滑杆分三组（括号里是写进 `tuning.json` 的键名）：

| 组 | 滑杆 | 范围 | 步长 |
| --- | --- | --- | --- |
| 身体比例 | 臂长（`arm_scale`）、腿长（`leg_scale`）、肩宽（`shoulder_width`）、髋宽（`hip_width`） | 0.7 – 1.3 | 0.01 |
| 平滑与求解 | 手部响应（`hand_response`） | 0.05 – 1.0 | 0.01 |
| 平滑与求解 | 阻尼（`damping`） | 0.005 – 0.2 | 0.005 |
| IK 任务权重 | 手腕位置（`wrist_pos`）、脚踝位置（`ankle_pos`）、膝部位置（`knee_pos`）、肘部位置（`elbow_pos`） | 0.0 – 2.0 | 0.05 |
| IK 任务权重 | 躯干旋转（`torso_rot`）、手腕旋转（`wrist_rot`）、脚踝旋转（`ankle_rot`） | 0.0 – 1.0 | 0.05 |
| 手腕目标局部旋转 | 左右腕 X/Y/Z（`left_wrist_*`、`right_wrist_*`） | -180° – 180° | 1° |

“手腕旋转”是 IK 任务权重，只决定机器人多大程度跟随方向；“手腕目标局部旋转”才是实际的角度修正。X、Y、Z 分别对应红、绿、蓝轴，修正作用在首帧对齐后的 G1 手腕目标上。

三个按钮：「恢复默认」还原上表默认值；「重新标定身体」按当前锁定帧重新标定人体方向和尺寸；「保存并继续」把当前参数写进 `g1_ik/tuning.json`，退出标定并恢复动作。

`tuning.json` 是本地调参结果，不入库；下次启动会自动读取，文件不存在就用默认值。该工程只用于 MuJoCo 验证，不直接控制真机。


## 可选互动场景 PULSE

在 `replay` 或 `live` 命令末尾添加 `--scene pulse`，即可加载人形机器人互动场：弹簧按钮、推球入门、物块配对和有序路点。`--scene empty` 为原空场景，`--scene xixi` 为西溪工位场景；原 `--xixi-scene` 参数继续有效，与 `--scene` 互斥。

默认不渲染阴影，也不分配阴影贴图。需要阴影时，在 `replay` 或 `live` 命令末尾添加 `--render-shadows`，可与任意 `--scene` 组合使用；不加该参数即保持关闭。开启后按实际几何范围配置阴影，避免裁切和过大的采样方块。

精简运行资源位于 `assets/pulse_arena/`（相对仓库根目录）。两个加载器均将机器人放到 `(0, -3.8)` 出生点，保留根部高度；界面状态栏显示 PULSE 分数。机器人关节 / 控制器索引保持原顺序，可动物件在后面追加。

SONIC 使用原有策略和物理步进推动道具；重置按钮会同时恢复道具及分数。轻量 IK 在每个源数据帧中固定机器人姿态、只让道具响应接触，属于运动学近似，不代表动态平衡或可靠抓取。普通回放随源数据推进，等待数据或回放结束时暂停；进入标定后锁定输入帧，继续更新仿真供调参观察。此改动不添加手动移动或行走控制功能。

资源内容及加载说明见仓库的 `assets/pulse_arena/README.md`。接入指南及验证命令见 `docs/pulse_scene.md`。

```sh
.venv/bin/python launch_g1_ik.py live DEVICE_IP --scene pulse
.venv/bin/python launch_g1_ik.py replay /path/to/full_body.parquet --scene pulse
```

### 场景优先显示

初始化完成后先渲染场景与机器人，再启动数据读取线程；无数据时仍可查看场景。空场景、西溪、PULSE 均支持。默认按机器人尺寸取景，右侧可切换“场景总览 / 跟随机器人”。切换视角不依赖数据，不推进物理仿真；数据连接出错时保留已加载的场景。更改代码后需关闭旧窗口并重新启动模拟器。

### 鼠标视角控制

在左侧三维画面内：滚轮向上拉近、向下拉远；按住鼠标左键左右拖动旋转水平视角，上下拖动调整俯仰。触控板垂直滚动也支持缩放。无数据时同样有效，操作相机不会推进仿真。

鼠标操作始终以机器人躯干中心为 pivot；从场景总览开始操作时也切换为机器人支点。机器人移动时支点随之移动，并保留手动角度和距离。切换视角下拉框可恢复对应预设。俯仰与缩放均有限位，避免翻转和穿过观察中心。

默认镜头仅由 pelvis 子树内机器人的可视几何尺寸决定，忽略场景和道具尺寸。使用 45° 垂直视场、135° 方位角、-12° 俯仰；按方形渲染画面约 68% 的包围框占比计算距离，并瞄准机器人全身包围框中心。该距离在初始化时确定，运行中不会因抬手或场景变化而自动缩放。重新选择“跟随机器人”可恢复默认取景。
