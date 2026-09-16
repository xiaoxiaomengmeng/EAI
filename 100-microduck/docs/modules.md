# 模块详解（骨架 · 待逐个展开）

> 配合 [`README.md`](README.md) 总览使用。每个模块一小节，先给**一句话职责 + 关键源文件 + 官方文档指引 + 待展开要点**。
> 展开时可直接填充"待展开"小节；若某模块内容变多，可把它拆成 `modules/<name>.md` 独立文件，并在此处改为链接。
>
> 源文件路径均相对于 [`microduck/`](../microduck/)。官方设计文档在 [`microduck/docs/design/`](../microduck/docs/design/)。

## 索引

**守护进程（daemons）**
[`robotd`](#robotd) · [`updater`](#updater) · [`configd`](#configd) · [`btd`](#btd) · [`padd`](#padd) · [`mediad`](#mediad) · [`tof`](#tof)

**库（libraries）**
[`duck-ipc-proto`](#duck-ipc-proto) · [`duck-control`](#duck-control) · [`kinematics`](#kinematics) · [`odometry`](#odometry) · [`sounds`](#sounds) · [`pet-detect`](#pet-detect) · [`duck-detect`](#duck-detect) · [`robotd-params`](#robotd-params)

**工具（tools）**
[`robotctl`](#robotctl) · [`duckctl`](#duckctl) · [`xtask`](#xtask) · [`test-support`](#test-support)

---

# 守护进程

## robotd

> ✅ **已展开** → [`modules/robotd.md`](modules/robotd.md)（忠实梳理自 `robotd-design.md`，§1–§9 全覆盖）。以下为原始骨架，保留作索引。

**职责**：控制守护进程——50 Hz 控制循环、电机总线、策略推理、安全层、健康上报；还挂了 voice / theremin / chorale 等子系统。

**关键源文件**：
- `robotd/src/main.rs`（~4654 行，进程主体：socket、JSON-RPC、systemd、健康）
- `robotd/src/control.rs`（策略优先级链 + 数值默认）
- `robotd/src/intents.rs`（意图槽）、`robotd/src/chorale.rs`、`theremin.rs`、`sound.rs`、`soc.rs`（板载热区）
- 控制核心逻辑在库 `duck-control/`（见下）

**官方文档**：[`design/robotd-design.md`](../microduck/docs/design/robotd-design.md)

**待展开**：
- [ ] tick 的完整数据流（read → observe → policy → safety.apply → publish）
- [ ] bring-up 状态机（Limp → Homing → Ready）、`enable`/`init`/`relax` 语义
- [ ] 健康判定：哪些能进 verdict，哪些只是 description（battery/temperature 为何不 gate）
- [ ] limp-fall 跌倒预测（`duck_control::fall`）与电池耗尽关机
- [ ] 挂载子系统：sound / theremin / chorale / pet-detect 如何"挂在 tick 上但不阻塞它"

## updater

> ✅ **已展开** → [`modules/updater.md`](modules/updater.md)（忠实梳理自 `updater-design.md` + `restart-order.md` + `boot-recovery-net.md`，§1–§19 全覆盖：apply 时序、原子交换、三层回滚网、boot counter、健康门、hooks、签名信任、CI/晋升、决策表）。以下为原始骨架，保留作索引。

**职责**：更新引擎（engine）+ `updaterd` 守护进程——验证签名、解包、原子切换 `current`、健康门、自动回滚、boot counter、更新历史日志。

**关键源文件**：
- `updater/src/engine.rs`（~3828 行，核心引擎）
- `updater/src/verify.rs`（minisign 签名验证）、`manifest.rs`、`store.rs`、`journal.rs`（update-log.jsonl）
- `updater/src/hooks.rs`（preinstall/postinstall）、`preflight.rs`、`reconcile.rs`、`orphan.rs`、`robot.rs`（对 robotd 的接口）、`ipc.rs`、`config.rs`、`main.rs`
- 测试：`updater/tests/apply.rs`（故障注入驱动真实引擎）

**官方文档**：[`design/updater-design.md`](../microduck/docs/design/updater-design.md)、[`design/restart-order.md`](../microduck/docs/design/restart-order.md)、[`design/boot-recovery-net.md`](../microduck/docs/design/boot-recovery-net.md)

**待展开**：
- [ ] 发布产物格式（release format）与验证流程
- [ ] 原子切换 + 健康门 + auto-rollback 的确切时序
- [ ] boot counter / golden 恢复网
- [ ] component 版本线（daemon / model 独立 install/rollback/pin）
- [ ] 单飞锁（single-flight，`File::try_lock`）、掉电恢复

## configd

**职责**：wifi（经 NetworkManager）、机器人身份与名字（从 SoC serial 派生）、配对 PIN、手柄绑定（bonding）、重启、单元状态上报。**必须在 robotd 死掉时可达**。

**关键源文件**：
- `configd/src/main.rs`（~585 行）、`store.rs`（config.json + flock + rename）
- `nm.rs`（NetworkManager D-Bus 客户端，Linux-only）、`net.rs`、`bluez.rs`（~910 行，BlueZ 客户端，Linux-only）
- `identity.rs`、`pad.rs`（手柄配对）、`power.rs`、`units.rs`

**官方文档**：[`design/app-path-design.md`](../microduck/docs/design/app-path-design.md)

**待展开**：
- [ ] 为何 wifi 交给 NetworkManager 拥有（netplan/networkd 的不足）
- [ ] config store 的原子写与并发（flock + rename + inotify 尚未启用的原因）
- [ ] 身份派生（`duck-c51b` 从 `/proc/device-tree/serial-number`）与配对 PIN 为何不能来自身份
- [ ] 手柄绑定为何在 configd 而非 padd（需 root + BlueZ）

## btd

**职责**：BLE 前置门（front door）——GATT server、framing、路由 API 子集、BlueZ 后端、配对 agent。**无状态传输适配器**。

**关键源文件**：
- `btd/src/main.rs`、`session.rs`（~938 行）、`route.rs`（~658 行，路由子集）
- `bluez.rs`（~778 行）、`framing.rs`（BLE 分片重组）、`gatt.rs`、`adv.rs`（广播）、`pairing.rs`、`link.rs`、`upstream.rs`、`chorale.rs`（合唱信标）

**官方文档**：[`design/app-path-design.md`](../microduck/docs/design/app-path-design.md)

**待展开**：
- [ ] BLE 暴露的 API 子集（provisioning / status / update 触发+进度），为何 payloads 不走 BLE
- [ ] framing：窄带宽无线电上的分片/重组
- [ ] 配对与加密现状（默认未加密，§5.5）
- [ ] chorale 信标：btd 只搬运不做决策

## padd

**职责**：手柄 → intents，作为普通非特权 socket 客户端。还提供 raw input tap（`pad.input`）。

**关键源文件**：
- `padd/src/main.rs`（~663 行，gilrs 读取）、`tap.rs`（~766 行，raw input 流）

**官方文档**：[`design/robotd-design.md`](../microduck/docs/design/robotd-design.md) §4.3；[`robot/pair-a-gamepad.md`](../microduck/docs/robot/pair-a-gamepad.md)

**待展开**：
- [ ] 手柄按键映射（button mapping）→ intent
- [ ] 为何独立 crate（把 gilrs/libudev 挡在恢复 CLI 之外）
- [ ] 非特权设计：只有 input/robot 组成员身份
- [ ] 无手柄时安全（不发东西，deadman 兜住）

## mediad

**职责**：摄像头/麦克风、编码、WebRTC 远程网关、感知（perception/detect）、以及机器人自己托管的 Web 控制台。**最重的服务**。

**关键源文件**：
- `mediad/src/main.rs`、`pipeline.rs`（~1430 行，GStreamer 管线）
- `session.rs`（WebRTC 会话层）、`route.rs`、`upstream.rs`、`web.rs`（console）、`detect.rs`、`exposure.rs`（3A/rkaiq）、`producer.rs`、`config.rs`
- `mediad/webclient/`（控制台前端）

**官方文档**：[`design/remote-webrtc.md`](../microduck/docs/design/remote-webrtc.md)、[`design/webrtc-console.md`](../microduck/docs/design/webrtc-console.md)、[`project/media-bringup.md`](../microduck/docs/project/media-bringup.md)

**待展开**：
- [ ] PeerConnection 结构：video/audio track + control(可靠) / teleop(不可靠) 两个 datachannel
- [ ] 硬件 H.264 经 `mpph264enc` → `webrtcsink`（两个必须自建的 GStreamer 插件）
- [ ] mediad 作为远程网关如何代理 control 消息到各 owning service
- [ ] 感知就近：detect 发布 features 而非 frames
- [ ] 隐私：consent（每会话同意）与可见指示灯（硬件缺口）

## tof

**职责**：`tofd`——头部 VL53L5/8CX ToF 传感器，8×8 深度矩阵，在自己 socket 上以 15 Hz 发布；只发布不读取。无传感器的板子也照跑并如实说明。

**关键源文件**：
- `tof/src/main.rs`（~563 行）、`sensor.rs`（~424 行，含无法打开的 `Sensor`）、`status.rs`、`lib.rs`
- `tof/build.rs` + `tof/vendor/platform.c`（经 `linux/i2c.h`，仅 Linux 目标编译）
- `tof/src/` 的 `--fake` 路径供离板运行

**官方文档**：[`design/architecture.md`](../microduck/docs/design/architecture.md) §1；theremin 见 [`design/robotd-design.md`](../microduck/docs/design/robotd-design.md) §4.5

**待展开**：
- [ ] 为何独立于 robotd（上电上传 ~90KB 固件、I²C 与音频 codec 共享、多数鸭子没装）
- [ ] 为何不属于 mediad（深度是"总线上的传感器"不是"媒体管线"）
- [ ] `tof.stream` 订阅；消费者如何结合 `robot.state` 经 kinematics head FK 重投影
- [ ] macOS 上驱动测试为何不跑

---

# 库

## duck-ipc-proto

**职责**：所有 `*.` 命名空间的**线上契约（wire contract）**。只依赖 serde / serde_json / semver，因此恢复路径不会继承更新引擎的 http/tar/crypto 树。当前 API v14。

**关键源文件**：`duck-ipc-proto/src/lib.rs`（~4864 行，单文件契约）

**待展开**：
- [ ] 命名空间划分：`robot.*` / `update.*` / `net.*` / `pad.*` / `system.*`
- [ ] intents(notifications) vs requests(answered) 的消息族映射
- [ ] `JOINT_NAMES` 从此 crate 来（wire 按位置索引 joints/targets，const 断言防漂移）
- [ ] API 版本握手、`Call::is_mutating()` 授权分类

## duck-control

**职责**：控制核心——robot model · bus · IMU · RobotIo · observations · policy · safety。"读总线与写总线之间的一切"。no tokio / no socket / no systemd。

**关键源文件**：
- `duck-control/src/bus.rs`（~495 行，Dynamixel 总线 + RobotIo）
- `duck-control/src/safety.rs`（~580 行，独占写句柄、clamp、deadman、fall verdict）
- `duck-control/src/policy.rs`（~509 行，ONNX 加载/推理、runtime 探测）
- `duck-control/src/obs.rs`（~390 行，61-D 观测构建）、`model.rs`（15 关节常量、battery_percent）、`imu.rs`、`fall.rs`（limp-fall 预测）、`io.rs`

**官方文档**：[`design/robotd-design.md`](../microduck/docs/design/robotd-design.md) §1.3、§2

**待展开**：
- [ ] `RobotIo` trait 与两个实现（`DynamixelIo` / `FakeIo`）
- [ ] 总线常量（RAD_PER_SEC_PER_COUNT、EEPROM 寄存器校正、kP/kI/kD）
- [ ] 观测构建的 3 个"看似合理实则错误"的点（body 全零是 nominal、head 在 command 里、body 顺序 z/roll/pitch）
- [ ] safety 的两条无条件规则 + fall verdict 为何"只报告不 gate"
- [ ] `ort` 缺失会 panic → `ensure_runtime` 预探测

## kinematics

**职责**：MJCF 模型与正向运动学（FK）；head 与 hand 链。两个消费者（odometry 与 robotd 的 head FK），避免各存一份模型。

**关键源文件**：`kinematics/src/lib.rs`、`head.rs`、`hand.rs`、`math.rs`、`mjcf.rs`、`tof.rs`；`kinematics/assets/alpha/`、`kinematics/tests/`

**待展开**：
- [ ] MJCF 解析与链路
- [ ] head FK（用于 tof 重投影、gaze）
- [ ] hand 链
- [ ] 与 odometry 的接口

## odometry

**职责**：里程计——机器人去过哪里，基于足端接触（foot contact）+ IMU。作为控制循环里的 struct，每 tick 步进一次（用循环已读的采样）。

**关键源文件**：`odometry/src/lib.rs`（~348 行，`Odometry::alpha()`）

**官方文档**：[`design/robotd-design.md`](../microduck/docs/design/robotd-design.md) §4.4

**待展开**：
- [ ] 接触式里程计： sole corner 作为 ground anchor，anchor 切换避免跳变
- [ ] heading = IMU 积分 yaw（无磁力计、不修正漂移，纯相对运动）
- [ ] 输出 `odom: { position, yaw }` → robotctl monitor 的 path map
- [ ] 每 tick 两次链求值的成本

## sounds

**职责**：语音——合成（synthesis）、每机器人人格（personality）、合唱乐谱（chorale score）。

**关键源文件**：`sounds/src/lib.rs`、`synth.rs`、`voices.rs`、`personality.rs`、`rng.rs`、`stream.rs`（~838 行）、`main.rs`；`sounds/scores/`（乐谱）

**待展开**：
- [ ] 合成管线与 voice personality 参数
- [ ] chorale score 格式
- [ ] 与 robotd `sound.rs`（aplay 子进程，PCM 独占）的关系

## pet-detect

**职责**：一个 ~20KB CNN，从 onboard mic 的 40-band log-mel 窗口"听"摸头（head scratch）；在自己的 worker 里跑，robotd 离循环轮询。

**关键源文件**：`pet-detect/src/lib.rs`（~356 行）、`worker.rs`（~381 行）；`pet-detect/models/`、`pet-detect/training/`

**待展开**：
- [ ] 音频特征（log-mel）与模型输入输出
- [ ] worker 与 robotd 的轮询接口
- [ ] 训练流程

## duck-detect

**职责**：鸭子检测器，两条后端——ONNX（`ort`）与 RKNN（Rockchip NPU，`v2.3.2` runtime）。

**关键源文件**：`duck-detect/src/lib.rs`（~477 行）、`onnx.rs`、`rknn.rs`（~462 行）；`duck-detect/models/`

**官方文档**：[`project/npu-bringup.md`](../microduck/docs/project/npu-bringup.md)

**待展开**：
- [ ] NPU（RKNN）路径与 dlopen
- [ ] ONNX 路径
- [ ] 与 mediad detect 的关系；帧路径缺口

## robotd-params

**职责**：robotd 的启动参数——schema、defaults、validation。约十个值（不是 runtime 的 142 个）。

**关键源文件**：`robotd-params/src/lib.rs`（~1482 行）、`registry.rs`（~587 行）；`deploy/robotd.toml`

**官方文档**：[`design/robotd-design.md`](../microduck/docs/design/robotd-design.md) §4.2

**待展开**：
- [ ] 参数集：control rate、gains、action scale、low-pass alphas、max velocities、deadman timeout、policy paths、gate 阈值
- [ ] `policy.mode`（walk / roller）preset 如何决定每个未设字段
- [ ] 文件在 `/etc/robot/` 而非 release 内（挺过更新+回滚）；未注释值会被"冻结"的坑
- [ ] `[policy] enabled`：区分"不想要策略"（healthy）与"策略坏了"（unhealthy）

---

# 工具

## robotctl

**职责**：本地操作 CLI——`update` / `health` / `version` / `monitor` / `net` / `system` / `robot` / `pad` / `configure` / `quack` / `chorale` / `theremin` / `completions`。依赖 duck-ipc-proto（非 updater），以留在恢复路径上。**必须在坏掉的机器人上也能用**。

**关键源文件**：`robotctl/src/main.rs`（~4414 行）、`monitor.rs`（~4219 行，含 3D 视图/path map）、`configure.rs`（~1332 行）、`duck.rs`、`show.rs`、`path_map.rs`；`robotctl/assets/`

**官方文档**：[`robot/cheatsheet.md`](../microduck/docs/robot/cheatsheet.md)

**待展开**：
- [ ] 各子命令职责与对应 socket
- [ ] `health` 如何合并 robotd(硬件) + updaterd(软件) 且非零退出
- [ ] `version` 报"运行版 vs 安装版"两个数
- [ ] `monitor` 的 3D 机器人、loop、pad stream、ToF 矩阵、path map

## duckctl

**职责**：从笔记本控制机器人（今天走 BLE）。**从不发布、从不交叉编译**——workspace `default-members` 里被排除，`--workspace` 仍会 lint/test 它。

**关键源文件**：`duckctl/src/main.rs`（~2466 行）；`duckctl/examples/`

**官方文档**：[`robot/duckctl.md`](../microduck/docs/robot/duckctl.md)

**待展开**：
- [ ] 与 robotctl 的区别（笔记本端 vs 板端）
- [ ] BLE 连接路径
- [ ] 为何被 default-members 排除（避免在发布路径上为板子交叉编译一个蓝牙栈）

## xtask

**职责**：构建工具——package · sign · promote（byte-identical 晋升已验证）。从不发布。

**关键源文件**：`xtask/src/main.rs`（~1637 行）；`xtask/tests/`

**官方文档**：[`project/ci-setup.md`](../microduck/docs/project/ci-setup.md)、[`microduck/CONTRIBUTING.md`](../microduck/CONTRIBUTING.md) "Releasing"

**待展开**：
- [ ] package：如何把 ONNX Runtime 版本烘进 preinstall hook（单一真相源 + 测试断言）
- [ ] sign：minisign 签名（release-1/2/3 + team.dev）
- [ ] promote：staging → release 的 byte-identical 重签
- [ ] `--version` 与 Cargo.toml 一致性校验（拒绝不一致的 tag）

## test-support

**职责**：测试用的签名发布 fixtures。从不发布。

**关键源文件**：`test-support/src/lib.rs`（~285 行）；`test-support/examples/`

**待展开**：
- [ ] 提供哪些 fixture（签名 release 等）
- [ ] 被哪些测试消费（updater/tests、robotd/tests、xtask/tests）

---

*逐个展开时，把对应"待展开"清单替换为正文即可。*
