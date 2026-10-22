# G1 SONIC 数据桥

这个工程没有界面，也不跑 SONIC 本身，只做数据转换：把录制或实时的人体/手套姿态转成 SONIC 期望的窗口格式，通过 ZeroMQ（ZMQ）的 PUB/SUB 模式发布给下游的 SONIC 进程。适合放在跑 SONIC 的那台机器上（通常是 Linux + GPU），本脚本自己只用 CPU。架构与关键技术解析见 [docs/04](../../docs/04-g1_sonic_bridge架构与关键技术解析.md)。

- 身体：PICO 24 关节位置/旋转 + 左右手 tracker 旋转 → 标准 SMPL → 29 维关节位置/速度 + 躯干朝向
- 手部：手套 4 个关键点（`wrist`、`thumbend`、`indexend`、`middleend`）→ `dex-retargeting` → Dex3 手 7 个关节（拇指 3、食指 2、中指 2）
- 输出：ZMQ PUB，绑定 `tcp://*:5556`（默认值，可环境变量 `SONIC_ENDPOINT` 覆盖——本机 5556 常被钉钉等进程占用），topic `pose`
- 固定 50 Hz，每条消息带 5 帧（`WINDOW = 5`）的滑动窗口

## 代码结构

入口 `launch_g1_sonic_bridge.py` 只保留 CLI 分发（~26 行），职责拆在同目录 `bridge_*.py` 平铺模块（依赖单向，无循环）：

| 模块 | 职责 |
|---|---|
| `bridge_config.py` | 常量表（关节名/SMPL 索引/Dex3 关键点映射/输出字段）+ 资产路径 |
| `bridge_source.py` | parquet 加载、设备消息解析、时间戳重采样（slerp/interpolate）、`load_rotation_offsets` |
| `bridge_smpl.py` | `convert`（PICO 24 关节 → SMPL 字段与 29 维腕/肘目标） |
| `bridge_hands.py` | Dex3 手部重定向（4 关键点 → 7 关节角，`dex-retargeting` 向量法） |
| `bridge_protocol.py` | ZMQ 消息打包/解包、窗口校验、发布端 socket（`SNDHWM=2`） |
| `bridge_modes.py` | `replay`（落盘转 50 Hz 滑窗）与 `live`（设备订阅 + 缓冲拼窗）两个驱动 |

改名或搬移注意：`tests/test_rotation_config.py` 用 AST 从 `bridge_config.py` 抽 `ROTATION_OFFSET_JOINTS`、从 `bridge_source.py` 抽 `load_rotation_offsets` 做隔离测试（不 import 入口，规避运行时依赖），两个函数/常量搬家时须同步该测试。

## 安装

```bash
cd g1_sonic_bridge
bash setup.sh
```

`setup.sh` 会建独立 `.venv`，装 `numpy`、`pyarrow`、`scipy`、`websockets`、`torch`、`pyzmq`、`dex-retargeting`。人体骨架和 Dex3 手的 URDF 已随仓库放在 `../assets/`（`human_joints_info.pkl`、`unitree_dex3_left.urdf`、`unitree_dex3_right.urdf`），无需另外准备；`ASSETS_DIR` 环境变量可以覆盖这个目录。整个过程不读取 `$HOME` 下任何路径，也不依赖另外两个工程。

这里的 `torch` **不能从依赖里去掉**，尽管本工程不跑任何推理。`dex-retargeting` 把 torch 声明为可选依赖，pip 不会自动装，但它的 `__init__.py` 开头就是一道 `if find_spec("torch") is None: raise ImportError`——不是用到才报错，而是只要 import 这个包就检查。而 `replay` 和 `live` 都会无条件调 `build_hand_retargeter`，所以缺了 torch 手部重定向直接走不下去。（旁边两个工程用的是 `wuji-retargeting`，不需要 torch，不能类比。）

脚本从 `download.pytorch.org/whl/cpu` 装 torch（这也是 `dex-retargeting` 报错信息里自己推荐的），避免在 Linux 上为一个不跑策略的数据转换器拉好几 GB 的 CUDA 版；该源同时也提供 macOS arm64 的 wheel。

## 落盘回放

入参是具体的 `observation.data.vr_full_body_*.parquet` **文件**，不接受 session 目录：

```bash
.venv/bin/python launch_g1_sonic_bridge.py \
  replay ../session_20260828_161120/data/chunk-000/observation.data.vr_full_body_000000.parquet
```

手部 tracker 数据按文件名把 `vr_full_body` 替换成 `vr` 自动找到；手套数据替换成 `glove_left` / `glove_right` 自动找到。这三个文件都是**必需**的。腕部位置仍取 24 关节数据，腕部旋转直接取 `left_controller` / `right_controller`，不做首帧对齐。

如需修正 tracker 安装方向，可在本工程目录创建 `rotation_offsets.json`，按 XYZ 度数填写 `left_elbow`、`right_elbow`、`left_wrist`、`right_wrist`；缺少的项默认为 0。

发布前会拿第一个窗口做一次 `pack_message` → `unpack_message` 往返自检，序列化或形状不对会在连接下游之前就暴露出来。

## 实时转发

```bash
.venv/bin/python launch_g1_sonic_bridge.py live 192.168.2.111
```

连接 `ws://设备IP:9990/subscribe?token=rynn_dev`，订阅 `vr` 和 `glove` 两个 channel，设备端各以 90 Hz 推送，脚本内部重采样到 50 Hz；攒满 5 帧窗口就发一条。token 可以用 `--token` 覆盖。

## 消息格式

下游订阅 `tcp://<桥接机器IP>:5556` 的 `pose` topic，每条消息的结构是：

```
b"pose" + <1280 字节 JSON 头，尾部用 \0 补齐> + <紧凑二进制载荷>
```

JSON 头形如 `{"v":3,"endian":"le","count":1,"fields":[{"name":...,"dtype":...,"shape":[...]}]}`，载荷按 `fields` 顺序、小端、C 连续排列。字段如下：

| 字段 | 形状 | 说明 |
| --- | --- | --- |
| `joint_pos` | (5, 29) | 29 维关节位置，5 帧 |
| `joint_vel` | (5, 29) | 29 维关节速度，5 帧 |
| `body_quat` | (5, 4) | 躯干朝向四元数，5 帧 |
| `smpl_joints` | (5, 24, 3) | SMPL 关节位置，取索引 `0..21` 加 `39`、`54` 共 24 个 |
| `smpl_pose` | (5, 21, 3) | SMPL 姿态旋转向量 |
| `frame_index` | (5,) | int64 帧号 |
| `left_hand_joints` | (7,) | Dex3 左手关节角 |
| `right_hand_joints` | (7,) | Dex3 右手关节角 |

两个手部字段**不带窗口**，只取窗口最后一帧的值。它们在线格式上是可选的（`validate_window` 对缺席宽容），但这个桥接两种模式下都会始终发出来：replay 必须能读到手套文件，live 每帧都从 `glove` channel 填入。下游若要兼容其它发布端，才需要处理缺席的情况。

发布前会校验上表形状、检查有无 `nan`/`inf`，并确认 `body_quat` 是单位四元数（容差 1e-4），不合规就抛异常而不是发出去。

脚本里的 `unpack_message()` 是这套格式的解包参考实现，下游可以直接照着写。该工程不写任何本地文件，也没有调参界面。
