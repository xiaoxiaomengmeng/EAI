# Microduck 项目理解（学习笔记）

> 本目录是对 [`microduck/`](../../microduck/)（Rust runtime）与 [`microduck_rl/`](../../microduck_rl/)（RL 训练）两个仓库的**学习与梳理笔记**，非仓库自带的权威文档。
> 仓库官方文档：
> - `microduck`：[`microduck/docs/`](../../microduck/docs/)（`design/` 讲机制、`project/` 讲进度、`robot/` 讲使用）
> - `microduck_rl`：[`microduck_rl/README.md`](../../microduck_rl/README.md) 与 [`microduck_rl/AGENTS.md`](../../microduck_rl/AGENTS.md)
>
> 本笔记用中文书写，关键术语保留英文原文，便于对照代码与官方文档。
>
> - 总览（本文）：`microduck` 的整体架构、IPC 契约、代码布局、设计不变量、路线图、构建测试
> - 模块详解骨架：见 [`modules.md`](modules.md) —— `microduck` 每个模块一小节，留待逐个展开
> - `microduck_rl` 学习笔记：见 [`rl/`](../rl/) 子目录（RL 训练侧，与 `modules/` 平行）——**从 [`rl/README.md`](../rl/README.md) 进入**（导航 + 认知阶梯 + 学习路线图）。该目录按学习顺序编号：`00-primer/`(预备知识) · `01-quickstart/`(部署上手) · `02-architecture/`(架构与概念词典) · `03-walkthrough/`(全链路实战教程，**初学者主线**) · `04-modules/`(模块击破) · `05-source-walkthrough/`(源码走读) · `06-topics/`(专题深挖) · `07-reference/`(入口地图)。

---

## 1. 这是什么

一只约 **25 cm、800 g 的双足（鸭子形态）机器人**，用**强化学习策略（RL policy）**驱动运动。

- **本仓库 = 鸭子的"大脑"**：运行在 **Rockchip RK3566**（Radxa Zero 3W）上的一整套 **Rust** 守护进程（daemon）软件栈。
- 核心是一个 **50 Hz 控制循环（control loop）**，驱动 **15 个舵机（Dynamixel servo）**，加上无线电、摄像头，以及"不把机器人变砖（brick）"的更新机制。
- **策略训练**在隔壁仓库 [`microduck_rl`](../../microduck_rl/)：MuJoCo + PPO + domain randomisation，导出为 **ONNX**，本仓库加载运行。
- 出品方 [Pollen Robotics](https://pollen-robotics.com/microduck)，许可证 **Apache-2.0**。

**能力**：走路（walk）、装轮子后滚动（roll / `roller`）、用喙捡东西（ground pick）、被推倒后自己站起（get back up）、坐（sit）、踢球（kick）、以及用自己独特的嗓音"嘎嘎叫"（quack）、合唱（chorale）、特雷门琴（theremin）。

---

## 2. 核心架构：多个守护进程 + 一条 JSON-RPC 总线

一块板子上跑多个 **systemd** 守护进程，通过 **Unix socket** 上的 **JSON-RPC 2.0（NDJSON，一行一个 JSON 对象，newline 分帧）** 通信。所有客户端（App、Web 控制台、手柄、你的脚本）发送**完全相同的调用（same calls）**。

```text
手柄          手机          笔记本(ssh)      远端 peer        GitHub release
 │BLE/USB      │BLE            │robotctl       │WebRTC           │https
 ▼             ▼               ▼               ▼                 │
padd          btd           robotctl        mediad               │
 └── robot.* ──┴──── 同一套 API 的子集 ───────┘                  │
        │                                                        │
        ▼  每服务一个 unix socket · JSON-RPC 2.0 · NDJSON         │
   robotd (robot.*)   configd (net/pad/system)   updaterd ◄──────┘
   50Hz 控制循环       wifi/名字/配对             验证/切换/健康门/回滚
   │Dynamixel          │D-Bus                    │systemctl restart
   ▼                   ▼                         ▼
15 舵机 + IMU       BlueZ·NetworkManager    /opt/robot/daemon/current
```

### 服务清单（daemon）

| 服务 | 拥有（owns） | 监听 | 关键点 |
|---|---|---|---|
| **`robotd`** | 电机控制、感知、策略、安全、`robot.health` | `/run/robotd.sock` | **唯一能碰电机的东西**；50 Hz 循环独占 Dynamixel 串口总线 |
| **`configd`** | wifi、机器人身份/名字、配对 PIN、手柄绑定、重启 | `/run/configd.sock` | 配置必须在 `robotd` 挂掉时仍可达；经 D-Bus 驱动 BlueZ / NetworkManager |
| **`updaterd`** | 发布版（release）：验证、安装、原子切换、健康门、回滚 | `/run/updaterd.sock` | 见官方 `updater-design.md` |
| **`btd`** | 无状态——BLE 传输适配器（transport adapter） | BLE GATT service | 手机配置机器人的通道，只转发 API 子集 |
| **`padd`** | 无状态——手柄传输；提供 raw input tap | `/run/padd/pad.sock` | 非特权（unprivileged）socket 客户端 |
| **`mediad`** | 摄像头/音频管线；WebRTC 远程网关（remote gateway） | TCP `:8080`(console)/`:8443`(signalling) | 最重的服务，无自己的 unix socket |
| **`tofd`** | 头部 8×8 ToF 深度矩阵，只发布不读取 | `/run/tofd/tof.sock` | 独立 I²C 传感器；无传感器的板子也照跑并如实说明 |
| **`robotctl`** | 无——本地 CLI | — | **必须在坏掉的机器人上也能用** |

### 控制面 vs 数据面（control plane / data plane）

| | 控制面 | 数据面 |
|---|---|---|
| 内容 | 命令、配置、状态、感知事件（features） | 视频/音频帧 |
| 大小/频率 | 几十字节，≤100 Hz | ~27 MB/s（640×480 RGB @30fps） |
| 机制 | unix socket RPC | **从不跨 socket**（感知就近放在传感器旁，只传"特征"不传"帧"） |

---

## 3. 贯穿全局的设计不变量（invariants）

1. **`configd`、`updaterd`、`btd` 能在 `robotd` 死掉时存活**——它们是**恢复路径（recovery path）**。控制循环起不来的机器人，恰恰最需要重配/更新/回滚。（无 systemd 依赖、无 ML runtime、无 media stack）
2. **`robotd` 对安全有最终权威（authoritative on safety）**——客户端发的是**意图（intents）**（"以这个速度走""看那里""站起来"），安全层决定什么真正可执行；没有远程/本地客户端能绕过跌倒检测、关节/温度限制。
3. **控制循环永不阻塞在别的服务上**——跨服务读取都是 **last-value-wins 缓存**，绝不同步 RPC。
4. **每份状态单一写者（single writer）**——每个值恰好一个 owning service，其余只读或订阅。
5. **发布版是"切换"而非"打补丁"（swapped, not patched）**——整个目录落到 `releases/<version>/`，验证签名后移动 `current` 软链、重启单元，再问 `robotd` 是否健康；不健康就自动把旧版放回（auto-rollback），崩溃循环由 boot counter 兜底。

---

## 4. 代码布局（一个 Cargo workspace，无框架）

> 官方权威布局见 [`microduck/CONTRIBUTING.md`](../../microduck/CONTRIBUTING.md) "The layout"。

**守护进程**（各一个 crate、一个 systemd 单元，同一发布产物）：
`robotd/` `updater/`(engine + updaterd) `configd/` `btd/` `padd/` `mediad/` `tof/`(tofd)

**它们驱动的库**（no socket、no systemd、不自启动，编译器强制边界）：
- `duck-ipc-proto/` — **线上契约（wire contract）**；只依赖 serde/semver，所以恢复路径不会继承更新引擎的 http/tar/crypto 树
- `duck-control/` — **控制核心**：model · bus · IMU · RobotIo · observations · policy · safety。`safety` 独占 `RobotIo` 写句柄（borrow checker 强制），任何策略/客户端都无法越过它写电机
- `kinematics/` — MJCF 模型与正向运动学（FK）；head / hand 链
- `odometry/` — 里程计，基于足端接触（foot contact）+ IMU
- `sounds/` — 语音合成、每机器人人格（personality）、合唱乐谱（score）
- `pet-detect/` — 听"摸头"的小 CNN（onboard mic → 40-band log-mel）
- `duck-detect/` — 鸭子检测器（ONNX / RKNN NPU 两条后端）
- `robotd-params/` — robotd 启动参数：schema、defaults、validation

**工具（tools）**：
- `robotctl/` — 本地 CLI（含 `monitor`）
- `duckctl/` — 笔记本端客户端，**从不交叉编译、从不发布**（`default-members` 里被排除）
- `xtask/` — package · sign · promote（构建工具，从不发布）
- `test-support/` — 签名发布 fixtures（测试用，从不发布）

**其它目录**：`deploy/`（板子配置：updater.toml / robotd.toml / trust anchor / journald）、`policies/`（发布携带的 ONNX 网络）、`hooks/`（preinstall/postinstall，更新时在板子上跑）、`scripts/`（provision / dev-push / setup / 恢复脚本）、`docs/`（官方文档）。

### 控制循环（`robotd`）一个 tick 的数据流

```text
read()   一次 sync_read：IMU 板 + 15 舵机（寄存器 124–136）
decide   observation [f32;61] → policy → targets [f32;14] → clamp
write()  一次 sync_write：goal positions
publish  atomics 总是发布；state frame 仅当有订阅者
每 1s    slow_sensors()：voltage + per-joint temperature（寄存器 144–146）
```

- 观测布局：`[ gyro(3) | projected_gravity(3) | joint_pos(14) | joint_vel(14) | last_action(14) | command(13) ]`
- 策略优先级链：`roulade > kick > ground pick > sit/rise > stand(按 |twist| 或强制) > walk`
- 所有 alpha 策略都是 `obs[1,61] → actions[1,14]`，在**加载时**校验（不是推理时）
- `MissedTickBehavior::Skip`：保持原调度、丢弃错过的 tick（控制循环想要的行为）

---

## 5. 状态与更新（state & update）

**状态存放位置**（除 `releases/<ver>/` 外都能同时挺过更新与回滚）：

| 状态 | 拥有者 | 机制 |
|---|---|---|
| `/etc/robot/robotd.toml`、`updater.toml` | 每板配置 | 安装器写一次，从不覆盖 |
| `/var/lib/robot/config/config.json` | `configd` | 文件 + `flock` + `rename(2)` 原子写 |
| wifi 凭据 | **NetworkManager** | 我们从不存储，NM 自己持久化 |
| `/opt/robot/daemon/releases/<ver>/` | 更新系统 | 二进制、策略、默认值——原子替换 |
| `/opt/robot/daemon/current` | 更新系统 | 指向当前生效 release 的软链 |
| `/var/lib/robot/updater/update-log.jsonl` | updaterd | **更新历史**，每条 `fsync`，200 条上限，掉电不丢 |

**一次变更如何到达机器人**：`push 分支 → CI 构建并签名 release → robotctl update apply → updaterd 验证签名/解包/移动 current/重启单元 → 健康门（问 robot.health）→ 健康则保留，不健康则回滚`。

**可观测性（observability）**：每个服务 `tracing → stderr → journald`，第一行以 `warn` 打印自身身份（build/rev/exe/pid）。`robotctl health` 一条命令同时报硬件（来自 robotd）与软件（来自 updaterd），不健康/不可达时非零退出。

---

## 6. 当前状态（roadmap，截至 2026-08）

> 官方权威进度见 [`microduck/docs/project/roadmap.md`](../../microduck/docs/project/roadmap.md)。

- **已完成**：M1 闭环、M2 开发通道（dev channel）、M3 `robotd` 真的会走、M4 硬件 bring-up 收尾。`0.9.x` 是一只**会走、会滚、能看、能说话、能唱歌**的鸭子。`updaterd` 引擎完整；自 `0.2.0` 起每个发布都经 CI 流水线到达过真实板子。
- **进行中**：M5（`mediad`/WebRTC 已能在局域网内推流到浏览器，硬件 H.264 经 `mpph264enc`；剩：跨 LAN、SDK、隐私同意 consent）
- **待做**：M6 出厂就绪（配对 PIN、标定 calibration、恢复网 recovery net 实测、手机 App）、M7 故障诊断（support bundle）、M8 模型通道（从 HuggingFace Hub 单独发布策略）、M9 自主大脑（autonomous brain，最大未移植缺口）
- **测试**：Mac 上 **942 个测试全绿**，无需硬件/网络/Docker（靠 `FakeIo` 脚本化采样）
- **明确不做**：A/B image 更新、OS/kernel OTA、fleet dashboard/telemetry、delta 更新、分阶段灰度

---

## 7. 构建与测试

需要 **Rust 1.89+**（edition 2024）。机器人是 aarch64 Linux，开发在 Linux/macOS。

```bash
cargo test --workspace        # 无硬件、无网络、无 Docker
cargo fmt --all
cargo test -p <crate>         # 单个 crate
```

- **Linux** 需先装 C 库：`libudev-dev`、GStreamer 系列（`padd` 经 gilrs 绑 libudev，`mediad` 管线是 `cfg(target_os="linux")`）
- **macOS** 上 `cargo test --workspace` 就是全部（942 tests）；只有 ToF 驱动的两个测试因无 Linux I²C 而不跑
- 交叉编译到板子验证：`RUSTFLAGS="-D warnings" cargo clippy -p configd --all-targets --target aarch64-unknown-linux-gnu`
- 板端断言：`scripts/board-test.sh`（CI 里跑 60 项：回滚、篡改产物拒绝、boot-counter 恢复、socket 权限、peer-credential 授权等）
- 真机部署不发布：`scripts/dev-push.sh <user@board>`（默认 `cargo zigbuild`，或 `--docker` 在板子 userland 内构建）
- **发布只在 CI 里签名**，从不本地：打标签 `daemon-staging-vX.Y.Z`（预发布 → staging）或 `daemon-vX.Y.Z`（promote staging，同样字节重签）

---

## 8. 工程文化（值得学习的地方）

- **注释说"为什么"而非"做什么"**——一个东西为何是这样的，比代码本身活得更久。
- **每个非显而易见的决策都有一个测试**，测试注释写明它防止哪种故障——尤其是回滚路径（只在别处已出错时才运行，最容易被悄悄弄坏）。
- **"让坏状态无法表示（make the broken state unrepresentable）"**：例如 `safety` 用 borrow checker 独占电机写句柄；unix socket 让"误绑 0.0.0.0 暴露固件更新"这类 bug 根本不存在。
- **官方文档"一个机制由一个页面拥有"**，其余页面只写一句并链接——因为"一个事实写在六处会朝六个方向漂移"。
- 大量失败教训被固化成不变量（如"一份 ONNX Runtime 版本漂移曾让板子装上 `ort` 会 panic 的 1.20.1"→ 用单一真相源 + 测试断言两处一致）。

---

## 9. 关键术语表（glossary）

| 术语 | 含义 |
|---|---|
| **daemon** | 守护进程，systemd 管理生命周期/重启/顺序/watchdog |
| **intent（意图）** | 客户端发的高层命令（速度、注视目标、"坐下"），非原始关节写入 |
| **control loop / tick** | robotd 的 50 Hz 循环；一个 tick = read → decide → write → publish |
| **policy（策略）** | ONNX 神经网络，`obs[1,61] → actions[1,14]`，由 microduck_rl 训练 |
| **safety layer** | 独占 `RobotIo` 写句柄；拒绝非有限值、clamp 到执行器行程、deadman |
| **deadman / heartbeat** | 意图停止到达时速度归零（"stop 不是 limp"，站定是双足的安全态） |
| **RobotIo** | 总线读写抽象 trait；两个实现 `DynamixelIo` / `FakeIo`（测试用） |
| **release / current** | `releases/<ver>/` 目录 + `current` 软链，原子切换 |
| **health gate** | 更新后问 `robot.health`，不健康则 auto-rollback |
| **boot counter** | 崩溃循环兜底：连续启动失败则回退到 golden |
| **JSON-RPC 2.0 / NDJSON** | 线上契约；一行一个 JSON 对象，notification 无 id |
| **transport adapter** | btd/padd/mediad——同一 API 的薄适配层，不拥有机器人状态 |
| **last-value-wins** | 跨服务读取的缓存策略，控制循环永不同步等待 |
| **component** | 更新系统的版本线单元（daemon / model 各自独立版本、回滚、pin） |

---

*下一步：在 [`modules.md`](modules.md) 中逐个模块展开。*
