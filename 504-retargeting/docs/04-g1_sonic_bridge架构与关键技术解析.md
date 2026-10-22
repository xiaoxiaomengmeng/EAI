# g1_sonic_bridge 架构与关键技术解析

> **证据等级**：实测背书型。所有机制描述均经源码核实（2026-10-21 六模块拆分后状态），验证数据标注「实测」；live 模式与下游 SONIC 进程联调未实测，标注「待核」。
> 操作细节（安装、订阅示例、消息字段操作表）在 [g1_sonic_bridge/README.md](../pico2g1/g1_sonic_bridge/README.md)，本文只写原理与架构；部署实测见 [01-pico2g1 部署复现实测记录](01-pico2g1部署复现实测记录.md)；另两条链路原理见 [02-g1_ik](02-g1_ik架构与关键技术解析.md) 与 [03-g1_sonic](03-g1_sonic架构与关键技术解析.md)。

## 1. 定位与总览

`pico2g1/g1_sonic_bridge` 是三条链路中**唯一的无界面纯数据链路**：不加载 MuJoCo 模型、不跑策略、不做物理仿真，只把人体/手套姿态转成 SONIC 参考格式，按 5 帧滑窗打包后经 ZMQ PUB 发布。设计意图：把「数据转换」从跑 SONIC 的机器上剥离——桥自己只吃 CPU，可部署在任意机器，下游 SONIC 进程（仓库外，另行准备）订阅 `pose` topic 消费。

| 属性 | 值 |
|---|---|
| 输出频率 | 50 Hz（replay 节拍 sleep / live 插值网格） |
| 窗口 | `WINDOW = 5`：每条消息 5 帧；手部字段除外（只取窗口末帧） |
| 传输 | ZMQ PUB，默认 `tcp://*:5556`（`SONIC_ENDPOINT` 环境变量覆盖），topic `pose`，`SNDHWM = 2` |
| 单条消息 | 5320 字节 = topic 4 + JSON 头 1280（\0 填充）+ 小端载荷 4036（实测：SUB 首帧 5320 字节） |
| 身体输出 | `joint_pos/joint_vel/body_quat/smpl_joints/smpl_pose` 各 5 帧 + `frame_index` |
| 手部输出 | Dex3 左右手各 7 关节角（`left/right_hand_joints`，形状 (7,)） |
| 依赖面 | torch CPU、pyzmq、pyarrow、scipy、websockets；**无** mujoco / PySide6 / onnxruntime / sim_scenes |

端到端数据流：

```
replay: vr_full_body parquet ─┐
                              ├─ load_* + 重采样 50 Hz（时间戳对齐、slerp）
live:   ws://设备:9990 ───────┘            │
                                           ▼
                          convert（bridge_smpl：PICO 24 关节 → SMPL 五字段）
                          Dex3 重定向（bridge_hands：4 关键点 → 7 关节角）
                                           │
                                           ▼
              make_window 5 帧滑窗 → validate_window → pack_message
                                           │
                                           ▼
                          ZMQ PUB（topic pose，SNDHWM=2）→ 下游 SONIC 进程
```

与另两条链路的关系：桥与 g1_sonic GUI 链路**不相连**（消息给独立 SONIC 进程）；SMPL 转换在 bridge_smpl 与 sonic_smpl 各有一份**语义同构**的实现（刻意复制，见 3.1）；手部走 Dex3 4 关键点，与另两条的 wuji 21 关键点不同源。

## 2. 架构

### 2.1 六模块平铺 + 瘦入口

2026-10-21 由 754 行单文件重构而来：27 行入口 + 6 个 `bridge_` 前缀平铺模块，依赖**严格单向无循环**：

```
bridge_source    ←  bridge_config
bridge_smpl      ←  bridge_config
bridge_hands     ←  bridge_config + bridge_source
bridge_protocol  ←  bridge_config
bridge_modes     ←  上述全部
launch_g1_sonic_bridge.py  ←  bridge_modes（仅 argparse 分发 replay/live）
```

| 模块 | 行数 | 职责 |
|---|---|---|
| [bridge_config.py](../pico2g1/g1_sonic_bridge/bridge_config.py) | 95 | 常量表（关节名/SMPL 索引/Dex3 向量索引与镜像旋转/输出字段/dtype 名表）+ 资产路径 |
| [bridge_source.py](../pico2g1/g1_sonic_bridge/bridge_source.py) | 247 | parquet 加载与过滤、设备消息解析、slerp/重采样、`load_rotation_offsets`（`torch.load` 读骨架） |
| [bridge_smpl.py](../pico2g1/g1_sonic_bridge/bridge_smpl.py) | 122 | `convert`：PICO 24 关节 → SMPL 五字段与 29 维腕/肘目标 |
| [bridge_hands.py](../pico2g1/g1_sonic_bridge/bridge_hands.py) | 69 | Dex3 重定向（4 关键点 → 7 关节角，`dex-retargeting` 向量法） |
| [bridge_protocol.py](../pico2g1/g1_sonic_bridge/bridge_protocol.py) | 109 | ZMQ pack/unpack、窗口校验、发布端 socket（`SNDHWM=2`） |
| [bridge_modes.py](../pico2g1/g1_sonic_bridge/bridge_modes.py) | 196 | `replay`（落盘转 50 Hz 滑窗）与 `live`（订阅 + 插值拼窗）两个驱动 |
| [launch_g1_sonic_bridge.py](../pico2g1/g1_sonic_bridge/launch_g1_sonic_bridge.py) | 27 | argparse 分发，无逻辑 |

### 2.2 结构性不变量（改代码前必读）

1. **不碰 sim_scenes、不注入 sys.path**：桥的 config 不做任何 `sys.path` 操作（对照两条 GUI 链路的 config 都注入仓库根）——无界面链路的边界就是连共享层都不 import；也因此桥的模块在任意 CWD 下可直接运行。
2. **tests 的 AST 耦合**：`tests/test_rotation_config.py` 从 bridge_config.py 抽 `ROTATION_OFFSET_JOINTS` 常量、从 bridge_source.py 抽 `load_rotation_offsets` 函数做隔离执行（不 import 入口，规避 torch/zmq 运行时依赖）。这两个符号搬家时须同步测试。
3. **`rotation_offsets.json` 与 g1_sonic 侧同名同格式但两份独立**：各写在自家目录下（`ROOT/rotation_offsets.json`），两边各调各的、互不读取；「修正 tracker 安装方向」的语义与 sonic 侧完全一致。

### 2.3 两种驱动的形态差异

| 维度 | replay | live |
|---|---|---|
| 数据源 | parquet 全量加载 | WebSocket 流式（vr + glove 双 channel，90 Hz 推送） |
| 重采样 | 先 `resample` 到 50 Hz 网格，再切窗 | 事件驱动：相邻两个 VR 包之间按 20 ms 网格插值出帧（见 3.4） |
| 手部 | `load_hand_joints` 一次性重定向全部帧 | 每采样帧增量 `retargeter.retarget`（低通状态在实例内） |
| 断流处理 | 无（数据有限） | gap ≤ 0 或 > 200 ms → 清 buffer + `reset()` 两手 retargeter 与 filter |
| 发布前自检 | 首窗 `pack → unpack` 往返 + `validate_window` | 无（逐窗 `validate` 由构造保证） |
| 节拍 | `started + index/FPS` 的 sleep 节拍 | 攒满 5 帧即发（while `next_timestamp` 循环） |

## 3. 关键技术

### 3.1 SMPL 转换与 sonic 侧的同源关系（bridge_smpl.py）

`convert` 与 [03 §3.3](03-g1_sonic架构与关键技术解析.md) 的 sonic_smpl.convert **语义同构**（Y_180 全局旋转 → `parentᵀ @ child` 局部旋转 → 55 关节 FK → `SMPL_OUTPUT_JOINTS` 24 关节 → `body_rot` 躯干系；肘 swing-twist 分解 + 腕欧拉合成 29 维 `joint_pos` 的 6 个肘腕列）。三处桥特有差异：

1. 输出多两键：`smpl_pose [N,21,3]`（21 关节旋转向量直出）与 `joint_vel`（**全零占位**——SONIC 参考的速度字段本桥不计算，格式占位）。
2. 输入显式校验：finite + 四元数模长 > 1e-8 不合规直接抛错（sonic 版依赖上游过滤，此处双保险）。
3. `swing_euler` 写法不同（手写索引 w/z 分量 vs 切片列），语义相同——两份实现**刻意独立维护**（三方法不抽公共库的既定设计），改公式须双侧同步核对，这是本链路的复现红线。

### 3.2 Dex3 手部重定向：4 关键点向量法（bridge_hands.py）

与另两条链路（wuji 21 关键点解析解）**不同源**，勿互相「纠正」。链路：

1. 手套侧只取 4 点：`GLOVE_POINTS = wrist / thumbend / indexend / middleend`（replay 从 21 列 parquet 取这 4 列；live 从 `pose_joints` 取索引 `[0, 4, 8, 12]`，即 Wrist 与三指尖）。
2. `GLOVE_TO_DEX3` 按左右手各一个固定旋转（互为镜像）把手套坐标转到 Dex3 URDF 坐标。
3. `hand_references` 按 `HAND_ORIGIN/HAND_TASK` 索引对构造 **6 个向量目标**：指尖间 3 个（食→拇、中→拇、中→食）+ 腕到指尖 3 个——`dex-retargeting` `dexpilot` 向量法的标准输入。
4. `RetargetingConfig(type="dexpilot", urdf=assets/unitree_dex3_{side}.urdf, 7 目标关节（拇指 3 + 食指 2 + 中指 2）, 3 指尖链接, wrist=base_link, low_pass_alpha=0.2)` 求解，输出按声明关节序 `order` 重排。
5. 求解初值语义（源码注释）：首解从关节限位中点出发，后续解以前帧结果热启动——所以断流必须 `reset()`，否则续流第一解会跳。

### 3.3 ZMQ 消息协议（bridge_protocol.py）

`pack_message`：`topic(4B) + JSON 头(1280B，\0 尾填充) + 小端紧凑载荷`。JSON 头 `{"v":3,"endian":"le","count":1,"fields":[{name,dtype,shape}…]}`，dtype 白名单 `f32/f64/i32/i64/bool/u8`（`DTYPE_TO_NAME` 双向表），载荷按 fields 顺序 C 连续排列。字段与字节构成（实测单帧 5320B 逐项吻合）：

| 字段 | 形状 | 字节 |
|---|---|---|
| `joint_pos` / `joint_vel` | (5, 29) f32 | 580 × 2 |
| `body_quat` | (5, 4) f32 | 80 |
| `smpl_joints` | (5, 24, 3) f32 | 1440 |
| `smpl_pose` | (5, 21, 3) f32 | 1260 |
| `left/right_hand_joints` | (7,) f32 | 28 × 2 |
| `frame_index` | (5,) i64 | 40 |

可靠性设计：`validate_window` 发布前校验全部形状 + finite + `body_quat` 单位四元数（atol 1e-4），不合规抛异常而不是发出去；replay 首窗再做一次 `pack → unpack` 往返自检，序列化问题在连接下游之前暴露。`SNDHWM = 2`：PUB 无订阅者或下游消费慢时旧消息直接丢弃——遥操作语义下宁可丢帧不要积压延迟。`unpack_message` 是这套格式的解包参考实现，下游照抄即可。

### 3.4 live 的插值重采样（bridge_modes.py）

live 不预知数据边界，采用**事件驱动插值**：每个 VR 包到达时，在「上一包 ↔ 当前包」之间按 20 ms 网格（`step_ns`）用 `interpolate`（位置线性 + 旋转 slerp）补出所有落点帧；手套在两个 glove 状态间按同一 alpha 线性插值。gap > 200 ms 判为断流：清空 `deque(maxlen=5)`、重置两手 retargeter 与低通 filter（衔接 3.2 的热启动语义）。帧由 **VR 节拍**驱动（glove 包只更新 `current_gloves`，vr 包到来才出帧），攒满 5 帧打包发送。

### 3.5 torch 依赖的由来（对照）

桥 venv 里 torch **不能去掉**：`dex-retargeting` 的 `__init__.py` 在 import 时就检查 torch（`find_spec` 为空直接 raise），而 replay/live 都无条件建手部 retargeter。既然 torch 已在依赖内，`load_skeleton` 直接 `torch.load`（`weights_only=False`）读骨架 pkl——对照 g1_sonic 为省 590 MB 刻意不装 torch、手写 archive unpickler（03 §3.2）：同一份 pkl 两种读法，各自服从所在链路的依赖取舍。torch 从 `download.pytorch.org/whl/cpu` 装 CPU 版，避免为不跑策略的转换器拉 CUDA 全家桶。

## 4. 验证记录

| 验证项 | 时间 | 结果 | 证据 |
|---|---|---|---|
| replay 端到端（转换→发布→外部订阅） | 2026-10-21 | ✅ SUB 实收 300/300 帧、首帧 5320 字节、topic `pose` | 01 §4 实测 |
| 六模块拆分回归（拆分前后逐字节一致） | 2026-10-21 | ✅ 同 session 重跑，SUB 收帧与拆分前完全一致 | 拆分回归记录 |
| tests/test_rotation_config（AST 隔离） | 2026-10-21 | ✅ 1/1 | tests 输出 |
| live 模式 | — | ⚠️ 待设备 | WebSocket 链路代码已核实 |
| 端到端联调（桥 → 下游 SONIC 进程） | — | ⚠️ 仓库内无消费端 | 下游另行准备（消息格式见 3.3 与 README） |

## 5. 边界与交叉引用

- **pico2g1 内**：[g1_sonic](../pico2g1/g1_sonic/)（本地 SONIC GUI 链路，原理见 03——桥与它不直接相连，给的是仓库外 SONIC 进程）；[g1_ik](../pico2g1/g1_ik/)（原理见 02）。三方法互不 import、各自 .venv；桥是唯一不碰 sim_scenes 共享层的一条。
- **部署实测**：setup.sh、端口占用（5556 → `SONIC_ENDPOINT`）与 replay 判据在 [01-pico2g1 部署复现实测记录](01-pico2g1部署复现实测记录.md)。
- **操作面**：安装细节（torch 来源）、订阅示例、消息字段操作表在 [g1_sonic_bridge/README.md](../pico2g1/g1_sonic_bridge/README.md)。
- **本系列文档**：`504-retargeting/docs/` 按序号顺延命名，本文为 04。
