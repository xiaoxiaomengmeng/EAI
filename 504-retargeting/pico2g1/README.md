# ego_retargeting

把 PICO 头显的人体动捕和手套数据，重定向到 Unitree G1 / Wuji Hand / Dex3 上，并在本机用 MuJoCo 做离线回放与实时验证。

仓库里有三个互相独立的方法，共用同一套输入（设备端 WebSocket 订阅服务，或 `session_*` 录制目录），但**代码互不 import、依赖互不重叠、各自建独立的 `.venv`**：

| 方法 | 目录 | 作用 | 额外依赖 |
| --- | --- | --- | --- |
| IK 解算 | `g1_ik/` | 不跑策略网络，用 MuJoCo 雅可比 + `lsq_linear` 直接 IK 解 G1 29 DoF，参数可在界面里实时调 | MuJoCo · PySide6 · wuji |
| SONIC 策略 | `g1_sonic/` | SONIC v1.1 强化学习策略（ONNX CPU）驱动 G1，带 PySide6 界面 | MuJoCo · PySide6 · onnxruntime · wuji |
| SONIC 数据桥 | `g1_sonic_bridge/` | 无界面，只做数据转换：把姿态转成 SONIC 需要的窗口格式，通过 ZMQ 发布给下游 SONIC 进程 | pyzmq · dex-retargeting · torch(CPU) |
| 数据工具 | `tools/` | 在线流抓包、离线 parquet 结构分析 | 见各脚本 |

其它目录：`assets/` 是入库的第三方静态资产（G1 的 `g1_description/`、人体骨架 `human_joints_info.pkl`、Dex3 手 `unitree_dex3_*.urdf`）；`3rd/` 是 setup 克隆的外部仓库（已忽略，不入库）；`session_*/` 是录制数据（已忽略）。

## 外部资产从哪来

- **G1 模型与网格**：已经拉好放进仓库 `assets/g1_description/`（`g1_gear_wbc.xml` + `meshes/`），克隆即用，不依赖本机任何目录。人体骨架 `assets/human_joints_info.pkl` 和 Dex3 手 `assets/unitree_dex3_*.urdf` 同样随仓库入库。
- **wuji-retargeting**（手部重定向，IK 与 Mac SONIC 需要）：由对应工程的 `setup.sh` 用 `git clone --recursive` 拉到仓库根的 `3rd/wuji-retargeting`（含 `wuji-description` 子模块）。
- **SONIC v1.1 ONNX 策略**（g1_sonic 需要，约 191 MB）：由 `g1_sonic/setup.sh` 从公开的 Hugging Face 仓库 `nvidia/GEAR-SONIC`（`sonic_v1_1/` 目录）下载到 `g1_sonic/models/`，无需 token。

路径均可用环境变量覆盖：`G1_DESCRIPTION` 指定 G1 资产目录，`WUJI_ROOT` 指定 wuji-retargeting 检出目录。默认全部落在仓库内，代码不读取 `$HOME` 下任何路径。

## IK 解算（`g1_ik`）

把人体关键点当作 IK 任务目标（腕/踝/膝/肘位置、躯干与腕部朝向、肘部 pole 约束），每帧线性化后用带阻尼和关节限位的最小二乘求解 Δq。适合验证映射关系本身是否合理，或调整人体到机器人的尺度差异。

```bash
cd g1_ik
bash create_env.sh     # 建 .venv、拉 wuji 到 ../3rd
.venv/bin/python launch_g1_ik.py \
  replay ../session_20260828_161538/data/chunk-000/observation.data.vr_full_body_000000.parquet
.venv/bin/python launch_g1_ik.py live 192.168.2.111
```

`replay` 的入参是 `observation.data.vr_full_body_*.parquet` 文件（不接受 session 目录），手套数据按文件名把 `vr_full_body` 替换成 `glove_left` / `glove_right` 自动找到。界面里的滑杆（`arm_scale`、`shoulder_width`、各任务权重、`damping` 等）会写入 `g1_ik/tuning.json`，是本地调参结果，不入库。细节见 [g1_ik/README.md](g1_ik/README.md)，架构解析见 [docs/02](../docs/02-g1_ik架构与关键技术解析.md)。

## SONIC 策略仿真（`g1_sonic`）

链路：PICO 24 关节 → 标准 SMPL → SONIC v1.1 ONNX → G1 29 个本体电机；手套 21 关键点 → wuji-retargeting → Wuji Hand 40 关节。固定 50 Hz，左侧画 G1，右侧画原始人体和双手。

```bash
cd g1_sonic
bash setup.sh          # 建 .venv、拉 wuji、下载 SONIC v1.1 onnx 到 models/
.venv/bin/python launch_g1_sonic.py replay ../session_20260828_161538
.venv/bin/python launch_g1_sonic.py live 192.168.2.111
```

`replay` 的入参可以是 session 目录，也可以是 `observation.data.vr_full_body_*.parquet` 文件。界面上的“重置机器人”会同时清空 SONIC 的 10 帧状态历史和朝向标定。细节见 [g1_sonic/README.md](g1_sonic/README.md)，架构解析见 [docs/03](../docs/03-g1_sonic架构与关键技术解析.md)。

## SONIC 数据桥（`g1_sonic_bridge`）

无界面，纯转换与转发：把设备/录制数据重采样到 50 Hz，按 5 帧滑窗打包，通过 ZMQ PUB 发布到 `tcp://*:5556`，topic 为 `pose`，由下游（Linux 上的）SONIC 进程消费。手部走的是 `assets/unitree_dex3_{left,right}.urdf` 的 Dex3 重定向，不需要 G1 模型也不需要 wuji。

```bash
cd g1_sonic_bridge
bash setup.sh          # 建 .venv，装 torch(CPU) / pyzmq / dex-retargeting
.venv/bin/python launch_g1_sonic_bridge.py \
  replay ../session_20260828_161538/data/chunk-000/observation.data.vr_full_body_000000.parquet
.venv/bin/python launch_g1_sonic_bridge.py live 192.168.2.111
```

`replay` 入参同样是 `observation.data.vr_full_body_*.parquet` 文件，文件名里必须含 `vr_full_body`，否则会直接报错。消息的字段与二进制布局见 [g1_sonic_bridge/README.md](g1_sonic_bridge/README.md)，架构解析见 [docs/04](../docs/04-g1_sonic_bridge架构与关键技术解析.md)。

## 数据排查工具

`tools/` 里的脚本可以借用任意一个工程的 `.venv` 运行（它们只依赖 `websockets` / `pyarrow` / `numpy`）：

```bash
# 抓设备端一条完整的 glove 消息和一条完整的 vr 消息，检查字段是否齐全
python tools/inspect_device_stream.py 192.168.2.111

# 看录制文件的 schema、每列宽度/空值、时间戳跨度与采样率、解码后的样本行
python tools/inspect_parquet.py session_20260828_161538
python tools/inspect_parquet.py <parquet 文件> --row 600 --rows 3
python tools/inspect_parquet.py <parquet 文件> --column wrist
```

## 设备接口与数据约定

- 实时订阅地址：`ws://设备IP:9990/subscribe?token=rynn_dev`，`--token` 可覆盖默认 token。
- 需要订阅两个 channel：`vr`（PICO 原始 24 关节 body tracking）和 `glove`（左右手套各 21 个关键点）。
- 各数据流采样率并不一致：手套约 90.9 Hz、VR 手部约 76.6 Hz、视频 30 Hz，跨流对齐必须按时间戳做，不能按帧号。
- 手套 parquet 里的 `wrist` 列是局部坐标占位值，真实腕部朝向在 `imu` 列。
- 手腕朝向使用 VR 的左右手 tracker，手指关节使用手套通道。回放需要配套的 `vr_full_body`、`vr`、`glove_left` 和 `glove_right` Parquet 文件。

三个方法都只用于本机 MuJoCo / 转换验证，不直接控制真机。


## 机器人同步视角

轻量 IK 与 Mac SONIC 界面的主画面下方默认显示三个相机画面：**头部、左手腕、右手腕**。主画面随窗口尺寸和数据采集区域的展开/收起调整渲染宽高，按可用区域铺满且不拉伸。三个视角使用主画面当前的同一份仿真状态同步刷新，支持所有场景及 replay / live；等待输入时显示初始场景，暂停或回放结束后保留最后画面。

头部相机位于 G1 头部前方，随躯干运动（模型没有独立头部关节）。手腕相机位于各自手腕局部下方 12 cm、前方 1.5 cm，向上倾斜 40°，以 55° 垂直视场观察五根手指的抓取细节，随手腕位置与旋转一起变化。主画面的拖动、缩放和总览切换不改变这三个相机的安装姿态。每路以 320 × 180 渲染，画面按比例适应窗口大小；实际刷新速度取决于场景和机器性能。

## 录制 LeRobot 数据集

两版窗口可通过主画面右上角的小箭头图标展开/收起数据采集区域；图标不占布局高度，收起时录制继续，悬停可查看状态。区域内可填写任务描述、选择保存位置，并“开始录制 / 停止并保存”。输出为 LeRobot v2.1：三路 **1280 × 720 / 30 fps / H.264** 视频，与 G1 + Wuji 的 **69 维实际关节、69 维目标动作和浮动基座位姿** 同步保存。默认目录为 `recordings/`，每次录制生成独立数据集，不覆盖已有数据。

录制时仅保存仿真轨迹，停止后后台渲染高清视频；使用已处理仿真步的时间轴，导出时统一重采样到 30 Hz。需安装支持 libx264 的 `ffmpeg` 及 `ffprobe`。操作、数据含义和失败恢复见 [录制说明](docs/lerobot_recording.md)。

录制结束后可用校验脚本验证产物是否为有效的 LeRobot v2.1（结构/schema/parquet 内容/视频逐帧计数，能诊断导出中断并给出恢复命令；`--deep` 可附加指定解释器的 lerobot 库加载验证）：

```bash
../scripts/04_validate_recording.py recordings/<录制名>       # 或直接传目录路径（用 g1_ik/.venv 运行）
# 中断态的离线恢复导出（在 pico2g1/ 下执行；无 DISPLAY 时 mujoco.Renderer 需 EGL 上下文）：
MUJOCO_GL=egl g1_ik/.venv/bin/python -m sim_scenes.lerobot_recording recordings/<录制名>
```

## 八路实时推流

右侧开启「实时推流」，或在 `replay` / `live` 命令后添加 `--stream`，即可将主画面、头部及左右手腕、身体正侧视和左右手原始关键点同步到其他机器。浏览器访问界面显示的局域网地址（默认端口 `8876`）；网页左侧显示仿真画面、右侧显示四路原始关键点；支持八路同步、单路放大、自动重连，以及独立 MJPEG 接口。后台编码并跳过积压帧，收起采集面板后仍可观看。参数和接口见 [推流说明](docs/live_streaming.md)。

## 可选互动场景 PULSE

在 `replay` 或 `live` 命令末尾添加 `--scene pulse`，即可加载人形机器人互动场：弹簧按钮、推球入门、物块配对和有序路点。`--scene empty` 为原空场景，`--scene xixi` 为西溪工位场景；原 `--xixi-scene` 参数继续有效，与 `--scene` 互斥。

默认不渲染阴影，也不分配阴影贴图。需要阴影时，在 `replay` 或 `live` 命令末尾添加 `--render-shadows`，可与任意 `--scene` 组合使用；不加该参数即保持关闭。开启后按实际几何范围配置阴影，避免裁切和过大的采样方块。

精简运行资源位于 `assets/pulse_arena/`（相对仓库根目录）。两个加载器均将机器人放到 `(0, -3.8)` 出生点，保留根部高度；界面状态栏显示 PULSE 分数。机器人关节 / 控制器索引保持原顺序，可动物件在后面追加。

SONIC 使用原有策略和物理步进推动道具；重置按钮会同时恢复道具及分数。轻量 IK 在每个源数据帧中固定机器人姿态、只让道具响应接触，属于运动学近似，不代表动态平衡或可靠抓取。普通回放随源数据推进，等待数据或回放结束时暂停；进入标定后锁定输入帧，继续更新仿真供调参观察。此改动不添加手动移动或行走控制功能。

资源内容及加载说明见仓库的 `assets/pulse_arena/README.md`。接入指南及验证命令见 `docs/pulse_scene.md`。

另有三套人形机器人游戏场：`--scene strike` 推球保龄球、`--scene orbit` 桌面滑盘、`--scene switchyard` 手柄 / 开门 / 推箱闯关。SONIC 与轻量 IK 的 replay / live 均已接入物理交互和计分，三套最小运行资源合计约 1.23 MB。玩法、启动命令、分层资源和验证范围见 [互动场景说明](docs/play_worlds.md)。
