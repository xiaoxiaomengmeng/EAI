# G1 SONIC 策略仿真

这个工程直接用 ONNX Runtime CPU 运行 SONIC v1.1，不依赖 GMR、TWIST2、CUDA 或 TensorRT；ONNX CPU 推理本身跨平台，Mac 与 Linux 均可部署。本机 Linux 部署实测记录见 [docs/01](../../docs/01-pico2g1部署复现实测记录.md)；架构与关键技术解析见 [docs/03](../../docs/03-g1_sonic架构与关键技术解析.md)。

- 身体：PICO 24 关节 + 双手 tracker → 标准 SMPL → SONIC v1.1 → G1 29 个本体电机
- 手部：手套 21 关键点 → 已有 `wuji-retargeting` → Wuji Hand 40 关节
- 仿真和界面：MuJoCo + PySide6，左侧显示 G1，右侧显示原始人体和双手
- 固定 50 Hz；落盘数据使用当前帧及之后 9 帧；实时数据保持最新帧，不额外等待 9 帧

## 代码结构

入口 `launch_g1_sonic.py` 只保留 CLI 与装配（~65 行），职责拆在同目录 `sonic_*.py` 平铺模块（依赖单向，无循环）：

| 模块 | 职责 |
|---|---|
| `sonic_config.py` | 常量表（关节名/SMPL 索引/电机增益与动作缩放）+ 资产路径；首次导入时把仓库根插入 `sys.path` |
| `sonic_model.py` | `build_model`：G1 + Wuji Hand MJCF 拼装与场景注入 |
| `sonic_recording.py` | parquet 加载、设备消息解析、时间戳重采样（slerp/interpolate）、免 torch 读骨架 pkl |
| `sonic_smpl.py` | `convert`（PICO 24 关节 → SMPL 参考与 29 维腕/肘目标）、`WujiMapper` 手部重定向 |
| `sonic_policy.py` | `SonicPolicy`（encoder/decoder ONNX 推理、10 帧状态历史、朝向对齐）+ `Simulator`（PD 伺服、10 子步、手部直写） |
| `sonic_source.py` | `replay_frames`/`live_frames` 帧生成、跨线程帧总线（`LatestFrame`） |
| `sonic_window.py` | `SourceView`/`MuJoCoView`/`Window` 三个界面类（含异步渲染、八路相机、推流、姿态调节） |

改名或搬移注意：`tests/test_scenes.py` 按文件路径加载本入口并取 `build_model`/`Simulator`/`load_skeleton`/`SMPL_OUTPUT_JOINTS`（入口 re-export，四个名字不可删）；ONNX 模型按 `ROOT/models` 同目录推导，模块需与本入口同级平铺。

## 安装

```bash
cd g1_sonic
bash setup.sh
```

`setup.sh` 会：建独立 `.venv`；把 wuji-retargeting（含 `wuji-description` 子模块）克隆到仓库根的 `3rd/`；从公开的 Hugging Face 仓库 `nvidia/GEAR-SONIC`（`sonic_v1_1/`）下载 SONIC v1.1 的 `model_encoder.onnx`、`model_decoder.onnx` 和 `observation_config.yaml` 到 `models/`（共约 191 MB，不需要 token）。G1 模型与网格、人体骨架已随仓库放在 `../assets/`（`g1_description/`、`human_joints_info.pkl`），无需另外准备。整个过程不读取 `$HOME` 下任何路径，也不依赖另外两个工程。

### 关于模型下载

huggingface.co 对单个连接限速到 ~1.5 KB/s，但这个限制是**按连接**算的（8 连接实测约 74 KB/s，24 连接约 513 KB/s）。所以脚本把每个文件切成 2 MB 分片、用 24 路并发 range 请求取回再拼接，191 MB 从两个多小时压到 8 分钟以内。屏幕上会看到：

```
Downloading model_decoder.onnx (149778638 bytes, 75 chunks, 24 parallel)
  round 1: 75/75 chunks outstanding
  model_decoder.onnx OK (149778638 bytes)
```

`round N: X/Y chunks outstanding` 是正常进度，不是报错：每轮只补上一轮没拿全的分片，最多 8 轮。分片暂存在 `models/.seg/`（已忽略），中断后重跑会接着下，不会从头开始。只有拼出来的字节数和服务器的 `Content-Length` 完全相等，文件才会从 `.part` 改名就位；对不上就报错退出，不留截断的 `.onnx`——否则 onnxruntime 会在加载时报一个看起来跟网络毫无关系的错。

## 落盘回放

可以直接传 session 目录：

```bash
.venv/bin/python launch_g1_sonic.py replay \
  ../session_20260828_161538
```

也可以传 `observation.data.vr_full_body_*.parquet` 文件。

## 实时控制

```bash
.venv/bin/python launch_g1_sonic.py live 192.168.2.111
```

装置端继续使用现有订阅服务：`ws://设备IP:9990/subscribe`，发送 `vr` 和 `glove` 两个 channel。`vr` 内除 PICO 原始 24 关节外，还需要 `joints` 中的 `left_controller`、`right_controller`；左右腕的位置仍取 24 关节数据，旋转直接取这两个手部 tracker，不做首帧旋转对齐。`glove` 内是左右手套各 21 个关键点；字段格式与现有装置文档保持一致。

在回放或实时命令末尾加 `--xixi-scene` 会加载西溪办公场景；不加时仍使用原来的空场景：

```bash
.venv/bin/python launch_g1_sonic.py replay ../session_20260828_161538 --xixi-scene
```

界面的“重置机器人”会同时清空 SONIC 的 10 帧状态历史和朝向标定。该工程只用于 MuJoCo 仿真验证，不直接控制真机。

## 姿态标定

1. 启动回放或实时模式，等右侧出现人体和手套数据。
2. 让画面停在能清楚观察目标关节的动作，点“进入姿态调节”。回放会暂停在当前帧；实时模式继续接收数据，但标定画面锁定在当前帧。
3. 选择左肘、右肘、左腕或右腕。X、Y、Z 分别对应红、绿、蓝轴；右侧虚线是原始方向，实线是修正后的方向。
4. 每次只调整一个轴并观察左侧 G1。方向不对时反向调整；“归零”只清除当前轴，“全部归零”清除四个关节的全部修正。
5. 点“保存并继续”，结果写入 `rotation_offsets.json`，回放或实时控制随后继续。下次启动会自动加载保存值。

回放和实时模式都提供“进入姿态调节”按钮。进入后锁定当前人体、tracker 和手套帧，仿真会持续使用这一帧，因此拖动滑杆时可以直接观察 G1 的变化。回放计时同时暂停；实时数据继续在后台接收，点“保存并继续”后恢复到最新动作。

调节面板可选择左肘、右肘、左腕、右腕，并分别调整局部 X/Y/Z 角度。X、Y、Z 在界面和原始视图中依次使用红、绿、蓝三种颜色；虚线表示原始坐标轴，实线表示修正后的坐标轴。每个轴可以单独归零，也可以全部归零。保存结果写入本地 `rotation_offsets.json`，下次启动自动加载。

右侧人体骨架和手套仍显示未经修正的原始数据；姿态调节时出现的彩色实线只用于显示当前修正结果，不会改写原始数据。


## 可选互动场景 PULSE

在 `replay` 或 `live` 命令末尾添加 `--scene pulse`，即可加载人形机器人互动场：弹簧按钮、推球入门、物块配对和有序路点。`--scene empty` 为原空场景，`--scene xixi` 为西溪工位场景；原 `--xixi-scene` 参数继续有效，与 `--scene` 互斥。

默认不渲染阴影，也不分配阴影贴图。需要阴影时，在 `replay` 或 `live` 命令末尾添加 `--render-shadows`，可与任意 `--scene` 组合使用；不加该参数即保持关闭。开启后按实际几何范围配置阴影，避免裁切和过大的采样方块。

精简运行资源位于 `assets/pulse_arena/`（相对仓库根目录）。两个加载器均将机器人放到 `(0, -3.8)` 出生点，保留根部高度；界面状态栏显示 PULSE 分数。机器人关节 / 控制器索引保持原顺序，可动物件在后面追加。

SONIC 使用原有策略和物理步进推动道具；重置按钮会同时恢复道具及分数。轻量 IK 在每个源数据帧中固定机器人姿态、只让道具响应接触，属于运动学近似，不代表动态平衡或可靠抓取。普通回放随源数据推进，等待数据或回放结束时暂停；进入标定后锁定输入帧，继续更新仿真供调参观察。此改动不添加手动移动或行走控制功能。

资源内容及加载说明见仓库的 `assets/pulse_arena/README.md`。接入指南及验证命令见 `docs/pulse_scene.md`。

```sh
.venv/bin/python launch_g1_sonic.py live DEVICE_IP --scene pulse
.venv/bin/python launch_g1_sonic.py replay /path/to/session --scene pulse
```

### 场景优先显示

初始化完成后先渲染场景与机器人，再启动数据读取线程；无数据时仍可查看场景。空场景、西溪、PULSE 均支持。默认按机器人尺寸取景，右侧可切换“场景总览 / 跟随机器人”。切换视角不依赖数据，不推进物理仿真；数据连接出错时保留已加载的场景。更改代码后需关闭旧窗口并重新启动模拟器。

### 鼠标视角控制

在左侧三维画面内：滚轮向上拉近、向下拉远；按住鼠标左键左右拖动旋转水平视角，上下拖动调整俯仰。触控板垂直滚动也支持缩放。无数据时同样有效，操作相机不会推进仿真。

鼠标操作始终以机器人躯干中心为 pivot；从场景总览开始操作时也切换为机器人支点。机器人移动时支点随之移动，并保留手动角度和距离。切换视角下拉框可恢复对应预设。俯仰与缩放均有限位，避免翻转和穿过观察中心。

默认镜头仅由 pelvis 子树内机器人的可视几何尺寸决定，忽略场景和道具尺寸。使用 45° 垂直视场、135° 方位角、-12° 俯仰；按方形渲染画面约 68% 的包围框占比计算距离，并瞄准机器人全身包围框中心。该距离在初始化时确定，运行中不会因抬手或场景变化而自动缩放。重新选择“跟随机器人”可恢复默认取景。
