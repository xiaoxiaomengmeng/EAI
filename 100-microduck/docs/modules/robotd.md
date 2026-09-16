# robotd —— 控制循环（梳理）

> 忠实梳理自官方 [`microduck/docs/design/robotd-design.md`](../../microduck/docs/design/robotd-design.md)（956 行）。
> 保留原文的技术例子、问题背景、推理过程、实践细节与观点分歧；去掉作者身份与推广信息。
> 被吸收的原型是 `apirrone/microduck_runtime`（下文简称 **the runtime / 原型**）。
> 代码锚点：进程主体在 `robotd/`，控制核心在库 `duck-control/`。

**范围收窄**：只支持 **alpha 变体、Radxa 板、v2 `imu_to_dxl` 板**。v1/v1.5/v1.6、另外四种 IMU、三种摄像头和 Pi 全部砍掉，所有出厂策略都是 `alpha_*`。**轮式配置作为一个 params 开关存活**——`policy.mode = "roller"`（§4.2）——因为它选的是"一套策略 + 一份调参预设"，不是一个硬件变体。

---

## 1. 整体形态

一个进程、一条串行总线、一个 50 Hz 循环。循环在**一次事务**里读总线上全部 16 个设备，决定 15 个关节目标，再写回去。其它一切——客户端、健康、遥测——都挂在这个循环上，且**永远无法阻塞它**。

### 1.1 总线，以及谁拥有端口

15 个舵机和 `imu_to_dxl` 板共享**一条 UART**。没有第二条总线、没有第二个端口：

```text
                     robotd — 控制线程
                              │  duck_control::bus::DynamixelIo
                              │  serialport · TIOCEXCL
                              ▼
         /dev/ttyS2 · 1 Mbps · Dynamixel protocol v2
                              │
    ┌────────────┬────────────┴───────┬──────────────────┐
  id 200       20–24                30–34             10–14
 imu_to_dxl   左腿 5 舵机      颈·头·嘴 5 舵机       右腿 5 舵机
  v2 板
```

**IMU 是 `id 200`，和舵机在同一次 `sync_read` 里读**，因为硬件就是这样：v2 板挂在 Dynamixel 总线上，从舵机应答的同一块寄存器里吐出片上 **SFLP 四元数**。一块板、一条代码路径、**没有 IMU 抽象**。它在 id 向量里排第一，好在舵机突发之前先应答。

**同一时刻只有一个拥有者，且没有硬性强制**。`serialport` 会设 `TIOCEXCL`，让第二次*非特权*打开变成 `EBUSY`——但 `robotd.service` 以 root 运行（电机控制需要字符设备），而 root 不受这个标志约束。所以排他是**安排出来的、不是强制出来的**，每个潜在抢占者都被刻意挡在端口外：

- **控制循环**：守护进程运行期间一直拥有它。
- **`robotd init`**：自己打开端口，且以 root 身份**在守护进程运行时也会成功**——两个写者在一条总线上交错发包，读起来像硬件故障。所以它希望守护进程已停止，而这正是 `robot.init` 和 `robot.relax` 作为 IPC 方法存在的原因（§3.3）：守护进程从循环内部服务这两者，于是别的东西根本不必打开总线。`init` 是给"守护进程没在跑"的机器人用的逃生舱。
- **`serial-getty@ttyS2`**：Armbian 默认在 UART2 上跑登录控制台，而占着端口的 `agetty` 会让每个舵机对其它一切都不可见。`scripts/setup-board.sh` 会 mask 掉这个单元；`fuser -v /dev/ttyS2` 里出现 `agetty` 就是当初发现它的办法，也仍是回答"谁占着总线"的命令。
- **原型（runtime）**，在更粗的粒度上：它驱动同一条总线，所以一块板要么跑原型、要么跑 `robotd`，绝不同时跑，systemd 单元用 `Conflicts=` 声明这一点（§5.2）。

### 1.2 谁和 robotd 说话

```text
   ┌──────────┐   robot.move / robot.head      ┌─────────────────────┐
   │  padd    │───(notifications, 50 Hz)──────►│                     │
   │ 手柄     │   robot.stop / robot.enable    │   /run/robotd.sock  │
   └──────────┘───(requests, 有应答)─────────►│   JSON-RPC 2.0      │
   ┌──────────┐   robot.subscribe              │   NDJSON            │
   │ robotctl │──────────────────────────────► │                     │
   │ monitor  │◄──robot.state (抽稀)───────────│                     │
   └──────────┘                                │                     │
   ┌──────────┐   robot.health                 │                     │
   │ updaterd │──robot.safeToRestart──────────►│                     │
   │          │  robot.modelApi                └──────────┬──────────┘
   └──────────┘       on_apply: systemctl restart robotd  │
        └─────────────────────────────────────────────────┘
   ┌ 两个传输层 ───────────────────────────────┐
   │  mediad — WebRTC + JSON-RPC 中继           │  手机、浏览器、LLM
   │  btd    — BLE，同一 API 的子集             │  客户端从这里进来
   └────────────────────────────────────────────┘
```

每一个都说同样的两套词汇——**意图进（intents in）、状态出（state out）**——所以 `mediad` 是**中继（relay）**帧而非翻译它们（§3.1、§3.2）。`updaterd` 永远只是**问问题**：更新路径里没有任何东西能命令电机。

### 1.3 crate 边界

```text
duck-ipc-proto/  线上契约 —— 只有 serde；无 tokio、无 http、无 crypto
duck-control/    机器人模型·总线·IMU·RobotIo·obs·policy·safety
                 "读总线与写总线之间的一切"；无 tokio、无 socket、无 systemd
robotd/          进程本身：socket、JSON-RPC、systemd、健康上报
robotd-params/   启动参数：schema、defaults、validation（§4.2）
kinematics/      MJCF 模型、正向运动学、head 与 hand 链
odometry/        机器人去哪了，从足端接触 + IMU
sounds/          嗓音：合成、每机器人人格、合唱乐谱
pet-detect/      robotd 离循环轮询的相机侧检测器
robotctl/ padd/ updater/   CLI · 手柄→意图 · 引擎+updaterd
```

列表里 `robotd/` 以下的都是**它驱动的库**，不是服务：没有自己的 tokio runtime、没有 socket、systemd 不启动它们。它们独立成 crate 的理由和 `duck-control` 一样——**是编译器把守护进程的关注点挡在外面**；尤其 `kinematics` 有两个消费者（`odometry` 和 `robotd` 的 head FK），否则会各长一份模型副本。

`duck-control` 持有"读总线与写总线之间的一切"；`robotd` 是包在它外面的进程。编译器强制这条边界，这正是阻止守护进程关注点渗进控制代码的东西——也意味着这个 crate 日后能被拎进自己的仓库（若过渡期原型要消费它），而不必重写。

**`safety` 持有 `RobotIo`**，所以策略、控制器和每个客户端都只能*提议*目标，没有一个能*发送*目标。这是借用检查器，不是一条约定（§2.4）。

### 1.4 tick（一个周期）

50 Hz，一个 `tokio` 任务跑在自己的 runtime 上，好让 IPC 工作无法插到它前面。每 tick 两次总线事务，外加每秒一次第三笔：

```text
read()          一次 sync_read · IMU 板 + 15 舵机 · 寄存器 124–136   (§2.1)
decide          observation → policy → targets → clamp               (§2.2–§2.4)
write()         一次 sync_write · goal positions
publish         atomics 总是发；state frame 仅当有人订阅             (§4.1)
每 1 秒         slow_sensors() · 寄存器 144–146 · voltage + temperature (§2.1)
```

一个周期里数据去哪了：

```text
  Dynamixel 总线
       │  一次 sync_read：IMU 板 + 15 舵机，一次事务
       ▼
   Sensors ──────────┬──────────────────────► safety.observe ──► 跌倒了?(去抖)
   joints, IMU       ▼
              Observation::build  ◄──── Command ◄── gate(deadman) ◄── intent 快照
                     │              (twist, head, body = nominal)
                     │  [f32; 61]
                     ▼
              Policy::infer ──── roulade > kick > ground pick > sit/rise >
                     │           stand (按 |twist| 或强制) > walk
                     │  [f32; 14]   — 嘴(mouth)被排除
                     ▼
              home pose + scale × action ──► head 和 legs 上的一阶低通
                     │  [f64; 15] 提议的目标
                     ▼
        ╔═════════════════════════════════════════════╗
        ║  safety.apply   ← 拥有唯一的 RobotIo         ║
        ║  · 拒绝非有限值(non-finite)                  ║
        ║  · clamp 到执行器行程                        ║
        ║  · 无 fall gate —— 跌倒判定只报告            ║
        ╚═════════════════════════════════════════════╝
                     │  sync_write goal positions
                     ▼
              Dynamixel 总线
```

数据流没画出来的、围绕它的决策：

```text
  启动
    ├─ 打开总线，等待            没上电的板子用开关修，不是靠放弃循环
    ├─ Safety::new(io)          safety 拿走 RobotIo；别的都不能写
    ├─ read() → hold = 机器人当前所处的姿态    —— 启动时绝不移动
    └─ policy
         disabled ─► controller = None                    healthy
         loaded   ─► controller = Some
         failed   ─► controller = None + policy_error     unhealthy

  每 tick
    read ─┬─ ok  ─► 清零 consecutive-error 计数
          └─ err ─► count++, sensors = None   (tick 仍然跑)
    observe → 跌倒了?    发布；不 gate 任何东西
    driving = enabled ∧ policy loaded ∧ 本 tick 有 sensors ∧ ¬limp-fall
    edges ─┬─ 开始 driving ─► controller.reset()
           │                  否则陈旧的 last action、或锚在一分钟前位置上的滤波器，
           │                  会表现为一次猛冲(lurch)
           └─ 停止 driving ─► hold = 当前姿态，捕获一次
                              每 tick 重读会在重力下下垂(sag)
    driving ─┬─ limp-fall ─► 序列自己的目标与增益             (§2.4.1)
             ├─ yes ──────► step() → targets, gain, 活动网络名
             └─ no  ──────► targets = hold, 默认增益, "held"
    safety.apply(targets, hold, gain)
    publish ─┬─ atomics       总是     → robot.health, safeToRestart
             └─ state frame   仅当订阅 → robot.state
```

`driving` 的四个条件**每个都承重**。`本 tick 有 sensors` 是不显然的那个：一次失败的读取留不下可用来构建 observation 的东西，凭空捏一个等于给策略喂一个不存在的机器人。`¬limp-fall` 是那个"不是拒绝"的条件——当 limp-fall 序列拥有机器人时（§2.4.1），策略被刻意不驱动，目标改由序列给出。

循环仍是一个带 `interval` 的 `tokio` 任务，**没有做成实时（RT）**（§5.4）。相比原型有一处值得点名的改动：**`MissedTickBehavior::Skip`**。

- `Burst` 会把积压的 tick 背靠背补发，把电机命令一层层叠起来。
- `Delay` 错得更隐蔽——它在每个 tick 后把下一个 tick 排在 *now + period*，于是每次唤醒延迟都被**加进周期**而不是被吸收，循环跑得比自己配置的速率还慢。
- `Skip` 保持原调度、丢弃错过的 tick，这才是控制循环想要的。

把感知移出到 `mediad` 后，大部分和循环抢时间的东西就没了。

### 1.5 五条不变量

其余一切都围绕这五条安排。每条都在某处**被强制**，不只是被打算：

1. **`safety` 之上没有任何东西能写电机。** 它拥有唯一的 `RobotIo`；借用检查器就是强制手段（§2.4）。
2. **`robotd` 绝不因为"某进程启动了"就移动机器人。** 一次更新重启会让站着的机器人继续站着（§3.3）。
3. **控制循环绝不等待客户端。** 意图是原子加载，遥测是滞后即丢（drop-on-lag）（§4.1）。
4. **健康是被发布的，不是被问出来的。** 卡死的循环会把自己报成不健康，而不是挂住调用者（§3.4）。
5. **只有"能归咎于某个 release"的东西才可以进入健康判定。** 电池和温度是描述，绝不是回滚输入（§3.4）。

---

## 2. 控制路径

### 2.1 总线层与 `RobotIo`

对 `rustypot` 的薄封装：open、一次合并的 `sync_read`、`sync_write` 目标位置、torque enable、gains、慢速传感器读取、启动寄存器检查。**代码是新写的，但数字是从原型借来的**，每处都有注释说明：

- `RAD_PER_SEC_PER_COUNT = 0.229 × 2π/60`，以及位置 count↔弧度 的换算。
- 来自 `check_and_fix_config` 的 EEPROM 寄存器，在启动时**既断言又纠正**：`return_delay_time=0`、`baud_rate=3`、`pwm_slope=255`、`shutdown=52`。第一个是承重的——在 XL330 默认值 250 下，那是每设备 500 µs 的周转，16 个设备一 tick 就要 ~8 ms，占预算 40%。被恢复出厂或换上的舵机 arrives 时是 250，所以这个检查消除了整整一类"为什么它在这台机器人上这么慢"。`shutdown = 52` 是在过载、过热、输入电压故障时锁存的错误掩码。
- 位置 P 增益写入时 **I 和 D 都为零**（原型的 `--ki`/`--kd` 默认）。这些是 RAM 寄存器，断电会恢复舵机出厂值，而出厂 D 不是零：留着它会给舵机内部 PID 加阻尼，机器人在**同样 kP** 下明显发软。这不是谁做的调参选择，所以是钉死而非暴露。

**每 tick 两次总线事务，第三笔每秒一次。** tick 读 124–136 的连续块（pwm、current、velocity、position）。voltage 和 temperature 在 144–146，比它末尾晚 8 字节，中间隔着 12 字节没人要的 trajectory 寄存器——所以它们每秒在自己的事务里一起采一次（~1 ms），而不是把 tick 的读取在 50 Hz 下加宽到每舵机 22 字节。采样间隔和"达成速率"的测量窗口是同一个，于是一个时钟驱动两者。

- **voltage 跨舵机求平均**：15 个都在一块电池上，单点读数只是同一测量加了更多噪声；答 0 的设备被过滤掉，而不是当电池半亏地平均进去。
- **temperature 不求平均**——调用者拿到每个关节，并**按名字**报最热的那个，因为一个撑着下蹲的膝盖温度远高于嘴，而对 15 个舵机求均值会藏住那个正逼近过热关断的。

**一个沉默的舵机不会产生短应答**：`rustypot` 的 `sync_read` 等每个 id，只要有一个不回就整笔失败。所以两次读都是全有或全无，调用者保留上一次采样，而不把一次错过当新闻。

**板温是第三个来源，且根本不在总线上。** SoC 热区里最热的那个，在同一个每秒采样里从 `sysfs` 读（`robotd/src/soc.rs`）。它在 `robotd` 而非 `duck-control`，因为它是 Linux 板子的属性、不是机器人的属性——这也是为什么它在电机总线不应答时仍能应答，而那正是它挣得一席之地的时刻：一块在堵住的通风口后发烫的板子，和一台舵机死掉的机器人，在你能同时看到两个数字之前是同一个症状。取跨热区的最大值而非某个具名热区，好让接线不同的板子无法悄悄漏掉那个正在爬升的。

**IMU 陈旧(staleness)被永久追踪。** "读成功了但板子交回同一个采样"会把死的方向喂给策略，除非有人去数否则不可见，而且已知会发生。总线层记住上一个块、数相同的后继者，并在一段跑得够久（半秒，正是 SFLP 解码器愿意把芯片输出称作一次测量之前等待的跨度）后在 journal 里说一次。第一次重复块就警告，是教会所有人忽略这条消息的办法；超过阈值后限速，因为一块停止刷新的板子每 tick 产一条，50 Hz 的相同警告会把 journal 挤爆。

抽象接缝（seam）：

```rust
trait RobotIo {
    fn read(&mut self) -> Result<Sensors>;                 // joints + IMU，一次事务
    fn write(&mut self, targets: &JointTargets) -> Result<()>;
    fn set_gain(&mut self, kp: u16) -> Result<()>;         // 每关节一次写，不是每 tick
    fn set_torque(&mut self, on: bool) -> Result<()>;      // 同上
    fn slow_sensors(&mut self) -> Result<SlowSensors>;     // voltage + 每关节 temperature
    fn imu_stale(&self) -> ImuStale;                       // 有默认实现
    fn imu_ready(&self) -> bool;                           // 有默认实现
}
```

`set_gain` 和 `set_torque` 是**每关节一次事务**，这就是为什么两者都不是每 tick 调用，也是为什么 bring-up 是个状态机而不是循环反复施加的标志（§3.3）。两个固有方法在 trait 之外，因为只有 `init` 用它们：`present_positions`（更轻的读，启动时用一次以采纳机器人已处的姿态）和 `interpolate_to`（阻塞式线性 ramp——刻意阻塞，因为运行期间别的东西都不该跟总线说话）。

两个实现：`DynamixelIo` 和 `FakeIo`（脚本化采样，可按需冻结或失败）。`FakeIo` 是测试套件跑的对象，也是为什么 `cargo test` 不需要硬件、网络、Docker。

**两者都没有 `cfg` 从 macOS 上关掉。** 那道门本意是把 `serialport` 挡在笔记本依赖树外，但 `rustypot` 和 `serialport` 在那儿都能干净构建，所以它啥也没买到，却代价了"没有板子也能对总线层做类型检查"的能力——而那恰恰是最可能被没有板子的人编辑的代码。只有真正开物理端口的入口点被 gate，所以 Mac 构建仍然拒绝假装它有机器人：`robotd --fake` 是笔记本路径，且必须显式要求，而非退而求其次地自动回退。

#### 2.1.1 一次事务读 16 个设备：12 字节公约数、时序预算、三层对齐

§1.4 说"一次 `sync_read` 读全部 16 个设备"，但没展开它凭什么成立、以及"对齐"到底对的是什么。

**12 字节是舵机和 IMU 的公约数。** tick 读的连续块是地址 124 起的 12 字节（`bus.rs` 的 `READ_ADDR=124`、`READ_LEN=12`）。对舵机，这 12 字节是控制表里相邻的四个量——`present_pwm`(124)、`present_current`(126)、`present_velocity`(128)、`present_position`(132)；解析时 `[0..2]` pwm（没用）、`[2..4]` 电流、`[4..8]` 速度、`[8..12]` 位置。对 IMU 板，**同一个地址 124 也恰好吐 12 字节**——`[0..6]` 陀螺 x/y/z（i16）、`[6..12]` SFLP 四元数 x/y/z（半精度）。两种设备、同一形状、不同含义；`bus.rs` 的注释一句话点破："Twelve bytes … happens to be exactly what the IMU board serves at the same address"，测试里还专门断言 `READ_LEN == IMU_BLOCK_LEN`。**这才是"IMU 能和舵机共用一次 `sync_read`"的物理前提**：不是协议碰巧兼容，是 IMU 板刻意在地址 124 镜像了舵机那 12 字节的形状。

**半双工：不是同时采样，是一问十六答。** 串口当然不能"一次同时"读 16 个设备。Dynamixel Protocol 2.0 的 Sync Read 是主机发**一条广播指令**（起始地址 124、长度 12、ID 列表 `[200, 20…24, 30…34, 10…14]`），16 个设备在**同一条线**上**按序逐个应答**。1 Mbps、8N1 ≈ 10 µs/字节：请求包 ~0.2 ms，16 个应答各 ~24 字节 ≈ 3.8 ms——`main.rs` 给的实测就是 "a real bus read costing 3–8 ms"。写回是 Sync Write，**广播、无应答**，15×4 字节 + 开销 ≈ 0.7 ms。`READ_TIMEOUT=30 ms` 把"少一个设备"封成一次有界的小卡顿，而不是按串口驱动默认值把循环卡死。`return_delay_time=0`（§2.1）正是把每设备 500 µs 的应答周转去掉，否则 16 个就是 8 ms、占 20 ms 预算的 40%。

**"对齐"是三层，都不是时间戳同步。**

| 层 | 对的是什么 | 怎么做到 |
|---|---|---|
| 空间对齐 | 16 个设备同一时刻 | 一次 `sync_read` = **一个原子快照**。`io.rs` 里 `Sensors` 的注释就叫 "One atomic sample of the robot"，关节和 IMU 装在一起，因为它们本就来自同一笔事务，关节之间、关节与 IMU 之间没有采样偏移 |
| 位置对齐 | 哪个 block 是哪个设备 | `rustypot` 返回的 block 顺序 = 请求 ID 顺序（IMU 第一，然后 `JOINT_IDS`），解析**纯按下标**：`blocks[0]`→IMU、`blocks[1+j]`→第 j 个关节，解析时不再匹配 ID |
| 时间对齐 | 读/写与 tick | 每 tick **恰好一次 read + 一次 write**；观测"天然老一个 tick"，而 50 Hz 下策略**就是按这个延迟训练的**（`main.rs` 的 `COAST_TICKS` 注释），所以对齐是设计出来的、不是凑的 |

**这不是"传感器时间同步"。** 严格的 sensor time sync 是多个**各有时钟的异步源**对齐到共同时间基（PTP、时间戳缓冲、插值）。这里 16 个设备**没有独立时钟要对齐**——它们被一笔事务一次性问完。整套机制的准确名字是**总线复用 + 原子采样 + 固定速率调度**；唯一带"时间"语义的是新鲜度（IMU 陈旧追踪，§2.1）和 30 ms 超时，都是"够不够新 / 会不会卡死"的有界判断，不是"对齐到同一时刻"。

### 2.2 一个 observation builder

每个 alpha 策略都是 `obs[1,61] → actions[1,14]`——在 walking、standing、ground pick、ball kick、sit 上都验证过。所以恰好只有一种布局：

```
[ gyro(3) | projected_gravity(3) | joint_pos(14) | joint_vel(14) | last_action(14) | command(13) ]
                                                                    command = vel(3) + head(4) + body(6)
```

关节全程排除嘴(mouth)；actions 映射回 15 个电机槽，索引 9 留零。51/54D 遗留、49D 轮式、85D 追踪布局随变体一起消失。

command 块（唯一曾存疑的部分）已定案——是从原型的 `control_step` 里**读出来**的，不是猜的：

```text
48..51   vx, vy, vyaw
51..55   neck_pitch, head_pitch, head_yaw, head_roll
55..57   body x, y      —— 硬编码为零，训练时未绑定
57..60   body z, roll, pitch
60       body yaw       —— 硬编码为零，训练时未绑定
```

关于它有三件事**各自看似合理却是错的**：

1. **全零的 body 是 nominal 编码，不是占位符**——x、y、yaw 字面硬编码为零表示"未绑定"，z/roll/pitch 除非 body-pose 模式激活否则为零。
2. **head 目标搭载在 command 里，不是叠加在策略输出之上。** 原型在不同模式下两者都做，并把事后叠加 gate 在 `if !new_cmd_obs` 后面，注释写着 "head_offsets are a COMMAND fed via the obs vector instead — don't double-add it here"。两者都做会把头**弯两次**。
3. **body 块顺序是 `z, roll, pitch`**——不是 `z, pitch, roll`。把最后两个换过来，会让机器人被要求前倾时**侧倾**。

### 2.3 策略

形状刻意像原型的。`robotd/src/control.rs` 持有优先级链和来自 `control_step`（它替换的对象）的每个数值默认：

```text
skill windows ← advance / expire (roulade 窗口, kick 计时, ground-pick 阶段, sit↔stand 起身)
command       ← 调用者平滑后的 command，为活动 skill 重新编码
net           ← roulade > kick > ground pick > sit/rise > stand-按幅值(或强制) > walk
action        ← ONNX
targets       ← home pose + action_scale × action
filters       ← head 和 legs 上的一阶低通
```

原型的两个微妙处值得点名，因为它们很容易被"顺手修好"：

- **一个 kick 窗口以 standing 调参运行**——kick 的 observation 携带全零 command，而 standing 转换恰好在"全零 command"上触发，所以一次 kick 跑在 `standing_action_scale` 和软化的 standing 增益下。**保留**，因为 kicks 就是对着它调出来的。
- **sitstand 的*起身(rise)*也以 standing 增益运行**（它的 command 全零），而*坐下(sit)*不是（它的 posture 标志让 twist 幅值为 1）。同一机制、同一理由。

一处刻意的分歧：原型通过跨转换保存/恢复 `action_scale` 来追踪 standing action scale，这会在一次 sit→stand 循环后留下一个陈旧值，直到下一次 walk 才纠正；这里 scale 和 gain 每 tick 从活动状态**重算**。

策略文件来自 params 文件里的路径，默认指向 release 目录——所以一次普通更新携带的是"对着这个二进制训练出来的策略"，而开发者把路径指向自己的 `.onnx` 就能迭代，不必切一个 release。

一切在**加载时**校验，不在推理时：observation 宽度、action 数量、以及 ONNX Runtime 到底在不在。每个网络都必须是 61-输入、14-输出，在加载时检查，而不是走到一半才发现。原型还有一个用遗留 3 值 command 的 51-D 家族；那些只在它的 `--new-cmd-obs=false` 路径下加载，而 `robotd` 用 `observation width is 51, expected 61` 拒绝它们。循环开始前跑一次 warm-up 推理，既把首次调用成本从热路径上挪走（在热路径上它看起来和一次错过 deadline 一模一样），又证明了 dylib 解析成功。

**`ort` 在 ONNX Runtime 缺失时 panic。** 它在 `setup_api` 里 `expect`，那是一条从任何 API 调用都可达的 lazy 路径，所以无法作为 error 捕获。放着不管会杀死控制线程：没有一个 tick 落地，健康永远报"循环还没完成一个周期"，于是守护进程看起来是卡死而非点明原因——比这个设计所拒绝的 crashloop 更糟。所以 `policy::ensure_runtime` 在碰 `ort` 之前，用 `ort` 自己那套 loader 和搜索规则探测 dylib，让"库缺失"变成一个普通 error。

**`policy.enabled` 区分"不想要策略"和"策略坏了"。** 前者是健康的，是 bench 更新测试的正确配置；后者是不健康的，于是 updater 会回滚这个 release。把两者合并，要么让一台 bench 机器人看起来坏了，要么让一个不可用的 bundle 通过 gate。`robotd --no-policy` 设置它，gate 测试也用它，因为 CI 和笔记本都没装 ONNX Runtime。

ONNX Runtime 是**板子前置条件**，由 `scripts/install.sh` 安装，**不随 release 发**。它变得比守护进程慢得多，而每个产物里塞 ~20 MB 会白白放大每次更新。代价是：一块缺它的板子能安装、能启动，然后**不能走**——这正是为什么健康要报出被搜索的路径。

**没做、且刻意不做**：预绑定 ONNX 输入/输出张量。当前路径每次推理分配一个 61-float 向量，50 Hz 下 ~244 字节。值得先在板子上测量再优化。

**从原型原样搬来，因为它管用**——head 和 leg 低通滤波器、action scale、电压自适应缩放、standing 转换的增益变化。这些是可调参（§4.2），不是要重议的决策。尤其低通 alphas 是 alpha 策略**训练时用的**值，所以必须和训练匹配，否则迁移会退化。规则是：不要回退已经跑起来的东西。

### 2.4 安全

`safety` 拥有唯一的 `RobotIo` 写句柄。没有策略、没有客户端有它，所以它之上没有任何东西*能*命令电机——这条不变量是结构性的、不是靠记住的。和 updater 对它恢复路径的论证同理：只在别的东西已经出错后才运行的代码，是最可能被悄悄弄坏的代码，所以干脆让"坏状态无法表示"。

两条规则，无条件：

- **拒绝非有限值。** 一个 `NaN` 目标不是被 clamp，是被直接拒绝。
- **范围 clamp。** 目标每 tick 都被约束在执行器行程内，无论策略要什么。这是*执行器的*行程，不是每关节的解剖学限制——它能抓住 `NaN`、荒谬的 action scale、垃圾张量；但它**拦不住**一个关节被驱动到机械上不智的地方（§9.3）。

外加 command 上的 deadman：若意图停止到达，速度归零。**Stop 不是 limp（松劲）**，这个区别很要紧——失联让机器人*站定*，因为对双足而言站立是安全态；失去平衡是另一件事，这一层不回答它。

**跌倒判定是报告，不是规则。** 它每 tick 算——trunk 系里的 projected gravity，去抖 0.2 s 好让一次坚实的落脚不算跌倒——然后发布，且**不 gate 任何东西**：一个倒地的机器人和直立的机器人一样被 enable、init、驱动、发 skill。这是刻意的。躺在地上恰恰是有人需要这些调用能用的时候，而一个"重力一误读倾斜就让步"的机器人，是那种"被人拎着时还一直想坐下"的机器人。这也是为什么"对跌倒的应对"能活在这一层之上、无需任何特例豁免：早先修订版这里有 `fall_limp` gate 和 `fall_recover` 自动站起，两者都被删了，因为**一条"恢复必须绕过它才能工作"的安全规则，根本不算安全规则**。

#### 2.4.1 跌倒是第三种事件

上面的跌倒判定回答"机器人倒了吗"。这对报告一台躺在侧面的机器人是对的问题，但对"软化一次落地"是错的问题：gravity 越过 `fall_gravity_z` 并持续 200 ms *就是*机器人已经在地上了，值得行动的窗口那时已经关上。

所以 `limp_fall`（自在一台真机上验证后默认开启）跑第二个、独立的检测器——`duck_control::fall`——基于**速率**而非位置。projected gravity 随 trunk 旋转，所以 `ġ = −ω × g` 是精确的，直接来自同一 12 字节 IMU 块里的 gyro；把它外推 ~0.3 s 就能说 gravity 正往哪去。它在机器人**已经倾斜**（≈26°）、**仍在倒**（而非在恢复）、且**预测会越过跌倒阈值**时触发——去抖三个 tick。对 SFLP 四元数求导会把滤波器的延迟加进"那个全部价值就在于提前"的数字里。

它买到的不是落地本身，而是落地后的**站起**。standing 策略能把一台"静止、姿态已知"的机器人干净地扶起，但对一台乱蹬的机器人只能在几次"以 walking 增益顶着地板"的尝试后才扶起——电机负载就来自那里。所以这个序列把跌倒从策略手里拿走：以 `gain_limp` 松劲、跟着关节下去，等 gyro 安静下来，用 ~1 s ramp 回到 standing 姿态，交还。交还不过是如此——twist 全程被保持在零，所以 command 幅值选中 standing 网络，那就是站起。

它按构造跑在策略路径之外：整个序列期间 `driving` 为假（§1.4），所以目标来自序列而非控制器，且它们和其它任何 tick 一样经 `apply` 到达电机——没有豁免、没有后门。`safety` 里什么都不用被告知，正因为判定不 gate 任何东西：姿态 ramp 移动一台躺在地上的机器人，和移动任何其它 tick 一样。

调参即特性，而且是**非对称**的：一次误报是一场机器人*自己造成*的跌倒，比它本想避免的僵硬落地更糟。默认值刻意偏"晚"一侧。

一块耗尽的电池是另一件"不被要求也会移动机器人"的事：`safety.battery_empty_shutdown`（默认开），在平滑电压上触及 empty 下限会让机器人坐下并关掉板子。EMA 在 ~10 s 上移动，所以一次负载骤降(sag)无法触发它——到达 6.6 V 需要一块真正耗尽的电池。

### 2.5 机器人模型

alpha 专用的 Rust 常量：15 关节、Dynamixel ID `20–24 / 30–34 / 10–14`、名字、`DEFAULT_POSITION`、执行器行程。和原型 `motor.rs` 同样的表，减去三个死变体。**恰好只有一台机器人**；第二个修订版可以是第二张表。

其中两张表在自身 crate 之外也承重：

- `JOINT_NAMES` **来自** `duck-ipc-proto`，因为 wire 按位置索引 `joints` 和 `targets`，两个顺序绝不允许漂移——一个 `const` 断言让"绝不允许"成真。
- `DEFAULT_POSITION` 必须匹配训练环境里的 `HOME_FRAME`：策略观测的是**相对于 home 姿态**的关节位置，所以这里一个不一致就是 14 个 observation 槽上的一个恒定偏移。

---

## 3. API

### 3.1 意图进（intents in）

两套词汇——意图进、状态出——恰好映射到 JSON-RPC 的两个消息族：

```jsonc
// 连续量：notifications，无 id，无应答，last-writer-wins
{"jsonrpc":"2.0","method":"robot.move","params":{"vx":0.2,"vy":0.0,"vyaw":0.4}}
{"jsonrpc":"2.0","method":"robot.head","params":{"neck_pitch":0.35,"head_pitch":0.35,
                                                 "head_yaw":0.0,"head_roll":0.0}}
// 离散量：requests，有应答
{"jsonrpc":"2.0","id":7,"method":"robot.stop"}
{"jsonrpc":"2.0","id":8,"method":"robot.enable","params":{"on":true}}
```

一切都是**弧度、trunk 系、右手系，符号在协议定义里钉死**。原型带着 `--laser-track-yaw-sign`、`--laser-track-pitch-sign`、`--laser-fk-pitch-sign`、`--laser-fk-neck-sign`、`--imu-z-rotation-deg`，因为那套约定从没被写下来，每个消费者都得经验性地重新发现它。把它写进协议就**删掉了这一整类问题**。

50 Hz 下，连续意图作为 notification 意味着没有应答流量。当它们日后走 WebRTC 时，notification 路由到不可靠的 `teleop` 通道、request 路由到可靠的 `control` 通道——这正是 `architecture.md` §5.2 要的，而且它是从消息族里**自然掉出来**的，不是谁要记住的规则。

**twist 和 head 刻意分成两个槽。** 单个合并槽更新一个字段需要 read-modify-write，而两个客户端（一个手柄驱动身体、另一个驱动头）会悄悄丢掉彼此的更新。分开的槽让每个在实践中都是单写者，于是 last-writer-wins 名副其实。**每个槽都盖时间戳**，因为循环真正的问题从来不是"值是多少"而是"它多旧"——那正是 deadman 读的东西。

`look`（注视方向）推迟；两种 gaze 形式都会暴露，它们之间的仲裁是无混合的 last-writer-wins。

### 3.2 状态出（state out）

一条流，可订阅，按订阅者抽稀(decimated)。它必须报告**被拒绝了什么**，不只是发生了什么——一个遥操作 UI 显示摇杆向前、机器人却静止、还没有任何解释，是没法的，而 safety 一直在 clamp：

```jsonc
{"method":"robot.state","params":{
  "t":1234.567,
  "move":{"requested":[0.4,0,0],"applied":[0.15,0,0],"limited_by":["max_velocity"]},
  "policy":"walk", "safety":{"fallen":false,"limp":false},
  "loop":{"hz":49.8,"missed":0},
  "battery":{"volts":7.62,"percent":64}
}}
```

**电池同时带 volts 和 percent**，这里和 `robot.health` 里都是。映射——6.6 V 空、8.2 V 负载下满、一块 NP-F550——住在 `duck_control::model::battery_percent` 里，且**已经应用好再传出去**。原型只传 volts，App 用自己那套常量重新推百分比，这就是同一块电池在两块屏幕上显示两个数的由来。一个画电池 pill 的客户端不该知道这台机器人出厂配哪块电池。**没有燃料计(fuel gauge)**：测量就是舵机自己的供电电压（§2.1），所以它在负载下骤降、静置时回升。

`robotctl monitor` 和日后的 App 用同一个 payload。这就是替换掉原型那一堆东西的：9870 上的 180 字节帧、9871 上的 JPEG 流、9872 上的 UDP 命令 socket、9874/9875 上的 maploc 端口、web hub 的 `/state.json`。今天加一个字段意味着改四个可能悄悄不一致的地方；这里意味着一个 struct，而老客户端忽略它们不认识的。

`robot.subscribe` 把一个连接变成一条流；循环发布进一个**有界 broadcast**，从不等待订阅者，所以慢客户端得到一个"间隙(gap)"而非施加背压——这正是 updater 对进度用的规则。抽稀是服务端、按订阅者的，所以一个 10 Hz 仪表盘真的比一个 50 Hz 数字孪生花机器人更少。

**应答(acknowledgement)点名策略。** `robot.subscribe` 用 `SubscribeResult` 应答：这个进程被配置了哪些网络（按文件名），外加"没有任何东西在驱动时"的一句话（params 里 disabled，或想要却加载不了）。这属于握手而非帧，因为它在进程存活期间不会变，而帧上的 `policy` 回答的是另一个问题：**哪个网络驱动了*这一 tick***。两个 gait 不同的 release 都报 `walk`，而"这是哪个网络？"是任何比较它们的人问的第一件事。放到帧上反而会在控制线程上每 tick 分配两个字符串，去回答一个从不不同的问题。

两个容易搞错的细节：**没人订阅时什么都不组装**——那是一台机器人的常态——因为构建帧会在"没有理由就不该拜访分配器"的线程上分配。以及**限制名是为 wire 拼写出来的**，不是从 Rust enum 派生的，所以重命名一个变体无法悄悄弄坏一个在 `limited_by` 上分支的客户端。

### 3.3 bring-up：`enable`、`init`、`relax`

**`robotd` 绝不自己移动机器人。** 启动时它读当前位置、采纳为目标、且不碰 torque。Dynamixel 在进程死掉期间保持它上次命令的目标，所以一次重启让姿态不变、没有间隙——机器人**穿过一次更新而站着、毫无察觉**。启动时插值到默认姿态会让每次更新重启都移动一台站着的机器人：一个跌倒风险，而且当被测对象正是 updater 时是个混淆因素。

`robot.enable` 过去只翻一个标志、别的啥也不做。torque 来自 `robotd init`——一个自己打开电机总线的独立子命令——所以它需要守护进程停着、它不出现在任何文档里，而在一台新机器人上按 Start 没有任何可见效果：策略跑了、循环写了位置、舵机却无视它们。于是循环有了一个 bring-up 状态，而**显式的 `robot.enable` 才是推进它的东西**：

```
Limp ──enable(策略已加载, 有新采样)──▶ Homing(torque on, 2 s ramp)──▶ Ready ──▶ 策略驱动
```

**旧规则保护的那条不变量没变：这里没有任何事因为"一个进程启动了"而发生。** 一次被更新重启的 `robotd` 发现 `Limp`、不要 torque、让一台站着的机器人继续站着——`a_restart_asks_for_no_torque` 断言的正是这个，断言的是**任何写入的缺席**而非写入一个 `false`。改变的是："绝不碰 torque"是一条比它要保护的属性更宽的规则，而且它在每次驱动前放了一个手动步骤。

两个条件 gate bring-up，各有理由：

- **一个已加载的策略。** `enable` 意思是"启用策略"；给关节上电去跑一个被禁用或加载不了的策略，会让机器人在一个坏 release 上站起来然后held住它。
- **一个新采样。** ramp 从关节当前所在处开始。从一个没人读过的位置开始，正是 ramp 要避免的那次猛冲。

**"倒在地上"不是其中一个条件。** 早先修订版在那里拒绝过，那时 `apply` 会把一台倒地的机器人held在 limp 增益、而一次 ramp 等于在写一个无法发生的站起；它随判定不再 gate 任何东西（§2.4）而一起去掉了。在一台躺在地上的机器人上启动，恰恰是有人请它重新站起来的办法——ramp 照常跑，standing 策略接手。

策略再次被禁用时 **torque 不撤**：机器人保持姿态，这才是"一台站着的机器人继续站着"在这一侧的含义。

`robot.init` 和 `robot.relax` 是同样两个转换，直接请求——因为"站起来"和"松手"是它们自己的决定，而在此之前前者是一个自己打开电机总线的子命令（§1.1）、后者根本不存在。两者都由 `robotd` 服务，所以都不需要守护进程停着，也都无法在控制循环正做同一件事时写总线。`init` 刻意不需要策略：站起来是可以向一台没有 walking 网络的机器人要求的，而且它正是让 bring-up 完全可测的东西，因为 CI 没有 ONNX Runtime。

它们作为循环**每 tick 取一次的 request** 到达，而非它反复施加的 flag：一次 `set_torque` 是每关节一笔总线事务，所以一个 level 会把 16 次写塞进每个 tick。后到的 request 替换一个未被读走的早先 request——被要求站起来、然后在 20 ms 内又被要求松手，第二个才是本意。而 `relax` 会清掉 `enabled`，否则下一 tick 会看到一台"被要求驱动"的机器人并把它straight back up。

**两者都不能经 BLE 到达。** 一个会把机器人摔到地上的手机按钮不值得提供，而站起来会一次移动每个关节，那要求发起者正看着机器人。

### 3.4 健康，以及什么可以进入判定

| 方法 | 回答 |
|---|---|
| `robot.health` | **循环在满足它的 deadline** ——从达成速率和错过 deadline 计数算出——外加一段判定从不咨询的机器人描述：loop、bus、IMU、battery、servo 和 board 温度 |
| `robot.safeToRestart` | 策略启用且机器人在动时为 false |
| `robot.modelApi` | 常量 |
| `robot.remoteSessionActive` | `false` —— `mediad` 拥有真正的答案 |

健康由 IPC 侧从循环发布的 atomics 算出——一个 last-tick 时间戳加若干计数器——**从不通过问循环**。这正是让一个*卡死的*循环把自己报成不健康、而不是挂住调用者的东西。

一个跑在目标 60% 的循环是活着的、应答每个请求的、且严重坏掉的。让这个区分成真，正是控制循环在任何会走的东西之前先建的原因（§5.1）。

**什么可以、什么不可以进入判定。** `healthy` 和 `degraded` 是更新系统的输入，所以只有**一个 release 能被归咎的**条件才可以设置它们——那正是 `degraded` 已经为一块没上电的 bench 板而存在所要强制的。答案上的其它一切都是**描述**，任何自动决策都不得读它：battery、motor 温度、loop/bus/IMU 计数。

- gate 在电池上，意味着一台在低电量时被更新的机器人会回滚这个 release，然后在同一块低电量电池上评判它的替代者，于是**根本没法被更新**，直到有人搞清楚为什么。
- motor 温度在一个炎热下午会做同样的事。

**为什么它们还是一起传。** 一个方法，因为问题只到一次：一台行为古怪的机器人被问"怎么回事"，而一个没有背后数字的判定只会开启第二轮提问。loop 部分携带的正是判定据以算出的那些数字，所以 `unhealthy: control loop at 43.9 Hz` 可以挨着 `missed = 0` 读——这区分了"一个被晚唤醒的循环"和"一个做太多事的循环"，而它们有不同的修法。`robotctl health` 从 `updaterd` 加上软件那一半，两个都打印。

`safeToRestart` 在策略启用且机器人在动时为 false：在走到一半时重启电机控制是机器人摔倒的方式（`updater-design.md` §7.2）。

### 3.5 维护是独立命名空间

`init`、紧急 torque-off、标定、原始关节写入**不是意图**。它们住在自己的命名空间里，好让中继的"按传输 allow-list"能把它们挡在远程传输之外。信令 gating 决定*谁连接*；它不说一个遥操作者同时也是个机械师——而 `update.*` 到达一个 DataChannel 会意味着一个远端 peer 能触发回滚。

---

## 4. 循环周边

### 4.1 循环读快照，绝不等待

意图和 params 由 IPC 线程发布、由循环作为单个原子加载读取。没有任何东西能对循环施加背压，也没有请求同步进入它。遥测经一个**有界 broadcast** 出去，慢订阅者得到一个间隙——updater 的 IPC 层已经用并记录了这个模式。

```text
   IPC 任务 (tokio, 多线程)             控制线程 (自己的 runtime, 50 Hz)
     robot.move  ──► ┌────────────┐
     robot.head  ──► │intent 槽   │ ──原子加载, 每 tick 一次──►  read
                     │ twist│head │     ArcSwap, 盖时间戳
                     └────────────┘
     robot.init  ──► ┌────────────┐
     robot.relax ──► │power 请求  │ ──每 tick 取一次──────────►  bring-up
     skills      ──► │skill 标志  │     last request wins
                     └────────────┘
     robot.health ◄── ┌──────────┐ ◄──── publish ───────────  atomics
     safeToRestart    │ atomics  │       ticks, hz, missed, fallen, moving
                      └──────────┘
     robot.state  ◄── ┌──────────┐ ◄── send, 仅当订阅 ──────  frame
                      │broadcast │       有界, 滞后即丢
                      └──────────┘
```

**没有任何通道朝反方向跑。** 健康是*被发布的*、从不被问，这正是让卡死的循环把自己报成不健康、而不是挂住调用者的东西。

skill 标志是布尔而非队列：在一个 20 ms tick 内，同一个按钮的第二次按没有额外含义，而两个*不同*的请求都值得被看到——单个 last-writer 槽会丢掉一个。

### 4.2 params

一个启动时读的 TOML 文件，**不被 watch**——实时 reload 以后再说。它住在 `releases/<ver>/` 之外（`/etc/robot/robotd.toml`），所以挺过更新*和*回滚。

属于板子而非 release，正是让"手改的策略路径"能粘住的原因：默认指向 `releases/<ver>/` 内，所以一次普通更新把策略和"对着它训练的二进制"放在一起，而删掉 override 就回到那个。**文件可以完全缺席**——一块没配好的板子以内建默认起来、而不是拒绝启动，这比一个不肯跑的守护进程好远程诊断得多。反面教训（吃过硬亏学到的）：一个*未注释*的值会在那块板上被**永久冻结**、而 release 继续前进，这就是一整个 fleet 曾以 kP 120 站着、而 release 默认写着 160 的由来。

大约十个值，不是 142 个：control rate、gains、action scale、低通 alphas、max velocities、deadman timeout、policy paths、update-gate 阈值。原型里的 flag 爆炸大多是变体、死 skill、死传感器，全没了。

有一个开关比其余都粗：`policy.mode`——`walk` 或 `roller`——既选加载哪些策略*又*选调参默认，所以每个未设字段按 mode 解析，把一台机器人挪到轮子上就是"一行 + 一次重启"。它是一个预设(preset)、不是变体：roller 线是原型的，rebase 在 alpha 默认之上。

### 4.3 手柄是一个客户端

`padd` 读 `gilrs`、经 `robotd` 的 socket 发意图。它自成一个 crate，好让手柄栈留在 `robotctl`（那个必须在坏机器人上能用的工具）之外。

一跳 socket，几十微秒。它买到的：App、SDK 和任何远程客户端用的输入路径，正是一个开发者每天在用的路径，所以它无法悄悄烂掉。开发时 `ssh -L /tmp/robotd.sock:/run/robotd.sock` 给你"手柄在笔记本、机器人在板子"，无需代码。

在机器人上它是 `padd.service`，从 boot 起运行、驱动任何连上的手柄——没手柄也安全，因为它什么都不发、deadman held 住机器人。它保持**非特权**，这是"手柄是客户端"里承重的那部分：它的 `input` 和 `robot` 组成员身份就是它全部的所有物。所以配对手柄属于 `configd`（`architecture.md` §1）、不属于这里——绑定一个设备需要 root 和 BlueZ，而一个持有二者的 `padd` 就不再是在演练 App 将用的那套 API 了。

### 4.4 里程计，跑在循环已经取的那个采样上

`odometry::Odometry::alpha()` 每 tick 从"循环刚读的关节位置 + IMU 四元数"步进一次，估计值作为 `odom: { position, yaw }` 出去到状态流。`robotctl monitor` 把它画成 3D 视图下的路径图(path map)。

**基于接触**，从原型搬来：一个鞋底角是地面锚点，trunk 的世界姿态经 `kinematics` crate 的模型由它正向运动学推出，而当另一个角掉到它下面时锚点移动——所以一步永不让估计跳变。heading 是 IMU 积分的 yaw，所以世界系就是机器人 boot 时朝的方向。**没有磁力计、没有东西修正漂移**；这是相对运动，每个消费者都必须这样对待它。

它花循环每 tick 两次链求值、零额外总线流量，这就是为什么它跑在循环自己的速率、而非原型的独立 100 Hz。这个成本正是 §7 里"不做里程计"的决定被**推翻**而非重新争论的原因：反对的从来不是算术，而是"没有东西读这个答案"。

### 4.5 循环还驱动什么，以及为什么它们都没有设计页

`robotd` 在 slice 2 之后长了四个"既不是控制也不是安全"的子系统。它们共享一个形状：每个都挂在 tick 上、没有一个可以阻塞它、没有一个能到达总线（除了经循环已经仲裁的意图）。

| 进程内 | 是什么 | 推理住在哪 |
|---|---|---|
| `sound.rs` | 播放时的嗓音——一个 `aplay` 子进程，新声音杀掉旧的，因为 codec 的 PCM 是独占的 | 模块头注释 |
| `theremin.rs` | 15 Hz 从 `tofd` 取深度 → 一个音符、一次张嘴、一行状态，被 50 Hz 循环采样、绝不被等待 | 模块头注释 |
| `chorale.rs` | 几只鸭子唱同一首：id 最低者指挥，指挥拥有座位安排，`btd` 搬运信标且不做思考 | 模块头注释 |
| `pet-detect/` | 一个 ~20 KB CNN，跑在 onboard mic 的 40-band log-mel 窗口上，在自己的 worker 里 | crate 头注释 |

外加 `soc.rs`，从 `sysfs` 读板子自己的热区——不在 `RobotIo` 后面，因为它必须在电机总线不应答时继续应答。

**它们没有一个有设计页，而那是规则在起作用、不是缺口。** 一个服务挣得一页，是当"第二个读者否则得从代码里推导它的契约"时。这些各只有一个实现、一个消费者，它们的决策够局部，可以住在模块头注释里、紧挨着它们所约束的东西。它们真正需要的是*可操作*，那是 cheatsheet 的活。

### 4.6 三个速率的汇合：theremin（15 Hz × 50 Hz × 48 kHz）

§4.5 表里 theremin 那行的"被 50 Hz 循环采样、绝不被等待"，展开就是三个速率的汇合——这是全仓库最接近"多速率传感器对齐"的地方，而它**刻意不做时间同步**。三个速率各跑各的，靠**两级无锁"最新值槽"**串起来，每跨一级用**采样保持 + 平滑**过渡：

```text
tofd（15 Hz，另一个守护进程）
   │  Unix socket · 换行 JSON 帧（64 zone：distance_mm + status）
   ▼
[tof-reader 线程]  每来一帧就 store，旧帧直接覆盖丢弃
   │  ArcSwapOption<Frame>{ distance, status, at = 到达时刻 }   ← 最新值槽 ①
   ▼
[50 Hz 控制回路]  Theremin::tick(now)：load_full（原子，永不阻塞）
   │   ├─ 新鲜度门：at.elapsed() < 500ms ?
   │   ├─ hand::Tracker.track → hold 桥接逐帧抖动 → closeness（0..1）
   │   └─ 同一个数 closeness 分三路：
   │        ├─ mouth ───────────────────────► targets[MOUTH_INDEX] ─► sync_write（50 Hz 舵机总线）
   │        ├─ hz = theremin_hz_at(closeness)┐
   │        └─ level = 1.0 / 0.0 ────────────┤ theremin_set(hz, level, open=mouth)
   │                                         ▼
   │                             Live 原子量 {hz, level, open}（f64 bits）  ← 最新值槽 ②
   ▼                                         │
[48 kHz 音频写线程]  每 10ms 块：read Live → Stream::set(target) → block() 逐样本 slew
   │   freq += (freq_target − freq) × a_pitch   （一阶极点：把 50 Hz 阶梯变成连续 glide）
   │   配速：领先播放 30ms 就 sleep
   ▼
aplay 管道 → 扬声器
```

**边界 1（15 Hz → 50 Hz）：最新值槽 + 采样保持 + Tracker hold。** 专用线程 `tof-reader`（`theremin.rs` 的 `read_frames`/`stream_frames`）停在 socket 上订阅 `TofStream`，每帧到达就 `slot.store(...)`，**旧帧被直接覆盖**——不排队、不积压。帧上的 `at` 是**到达时刻**（不是传感器采样时刻），且**只用来判断新鲜度**（`FRAME_STALE = 500 ms`），**不参与任何时间轴对齐**——这是与"时间同步"最本质的区别。50 Hz 的 `tick` 做一次原子 `load_full`：15 Hz 帧间隔 ~66 ms、tick 间隔 20 ms，所以**一帧深度会被连续约 3.3 个 tick 读到同一个值**，这就是采样保持。`kinematics::hand::Tracker` 再补一层 **hold**，把传感器在量程边缘的逐帧抖动（某 zone 这帧可用、下帧不可用）桥接住。**速率失配 = 淡入淡出，不是门控**：帧超过 500 ms 没更新 → `closeness = None` → `level = 0`（淡出到静音），但乐器仍"在手"，帧一回来自动复奏（模块头："Rate mismatch is a fade, not a gate"）。

**边界 2（50 Hz → 48 kHz）：Live 原子量 + 逐样本 slew。** 50 Hz 算出 `closeness` 后（`main.rs`），`theremin_hz_at(closeness)` 给音高、`theremin_set(hz, level, mouth)` 写 `sound.rs` 的 `Live { hz, level, open }` 三个 `AtomicU64`（存 f64 bits）——注释："the writer wants the newest value, not a queue of every value it slept through"。**一个手势、三个输出、同一个数**：`closeness` 同时驱动音高、音量、嘴张开度，且传给音频的 `open` **和传给嘴舵机的 `mouth` 是同一个值**（`stream.rs`："pass the same value the mouth servo is being sent, so what is seen and what is heard are one gesture"）——否则就成了"一段嘴部动画盖在一段声音上"，而不是"一只动物在发一个声"。48 kHz 写线程每 `SYNTH_BLOCK = SR/100 = 480 样本 = 10 ms` 读一次 `Live`、`Stream::set` 设 target，然后 `block()` **逐样本 slew**（`freq += (freq_target − freq) × a_pitch`，level/open 同理），于是 50 Hz 的阶梯式更新被平滑成连续 glide。写线程自己配速：领先播放 `SYNTH_LEAD_S = 30 ms` 就 sleep，使"参数被设置 → 被听到"的延迟有界（~30–40 ms）——这是延迟选择，不是同步。

**为什么不用时间戳同步。** 三个源**没有共同时钟**：`tofd` 是另一个守护进程、控制回路是本地 tokio interval、音频是 `aplay` 管道。建共同时间基（PTP / 时间戳缓冲 / 插值）成本极高、收益极低——**人手移动很慢**（15 Hz 绰绰有余），**音频要的是平滑不是精确对齐**。所以全程用"最新值 + 采样保持 + slew"，每个边界都无锁（`ArcSwapOption` / 原子量），**生产者永不被消费者阻塞、消费者永不被生产者阻塞，旧值直接丢**。

**三个线程的隔离（硬约束：50 Hz 回路永不阻塞）。** `tof-reader` 把阻塞的 socket read 隔离在回路外（回路只做一次原子 load）；音频写线程把 48 kHz 渲染 + `aplay` 管道写隔离在回路外（回路只写三个原子量）；`tick()` 注释明说 "Never blocks"。回路一旦阻塞，舵机 `sync_write` 就会迟到、机器人会抽——所以所有跨速率的等待都被推到专用线程里，回路本身永远只是"原子读最新值 → 算 → 原子写最新值"。

---

## 5. 它为什么存在，从哪来

### 5.1 目标，它不是"一个好的 robotd"

想要两件事，而第二件重排了工作顺序：

1. **在控制核心上快速迭代。**
2. **真正测试 updater。**

更新引擎已经完成，却从没在硬件上跑过。它最重要的路径因此未被证明：`on_apply` 里的 `systemctl restart` 从没遇过真 systemd，30 s 健康门 timeout 是个自认的猜测，而最糟的是——**auto-rollback 只有在 `robot.health` 有意义时才有意义。** 它过去意味着"控制循环 tick 过一次"，所以到目前为止每次被测的回滚都是对着一个占位符测的。

这就是为什么第一个增量**不会走**。它存在的意义是在一块真板上成为一个真实的健康信号（§3.4）。

### 5.2 robotd 替换什么

`robotd` 替换原型，但不是一步到位——原型做五件可分离的活，只有一件是 `robotd` 的：

| 原型的活 | 去向 | 何时 |
|---|---|---|
| 控制循环、策略、电机、IMU | `robotd` | 已完成 |
| 手柄 | 一个意图客户端（`padd`） | 已完成 |
| 相机、ball/laser/pet 检测、JPEG | `mediad` | M5 |
| web hub、PWA、brain 命令 socket | `mediad` / App | M5+ |
| maploc —— 建图、MCL、规划 | 无归属 | — |

所以两者并行一段时间。它们不能*同时*运行——一条串行总线、一个拥有者（§1.1）——所以一块板跑这个或那个，systemd 单元用 `Conflicts=` 声明。

远程/App 层不在这里设计。它是 `reachy_mini` 架构——媒体用 WebRTC、控制用 DataChannel 上的 JSON-RPC 2.0——移植到 Rust，超出本文范围。它对 `robotd` 的唯一要求在 §3.5。

### 5.3 两个 slice，以及"完成"意味着什么

**Slice 1 —— held 住姿态。** tick、总线、模型、`RobotIo`、诚实的健康。什么都不算：`held_pose` 是启动时采纳的一个常量。**那正是重点**——它把真实负载以真实速率放上总线，所以循环计时和健康是诚实的，而且当一个刻意做坏的 release 落地时没有东西垮掉。你可以在 bench 上一整天猛锤 install / rollback / power-cut 循环。完成判据：`robotd` held 住姿态一小时无总线错误；`robotctl update apply` 装一个 release、重启它、通过 gate、机器人不动；一个"被构建成起来就不健康"的 release 被自动回滚；一次更新中途断电经 boot counter 恢复。

**Slice 2 —— 走和站。** observations、ONNX 策略、safety 层、意图、手柄客户端。完成判据：它在一块板上走，经意图 API 驱动；一次用 `robotctl` apply 的更新干净地重启它、gate 通过；`--unhealthy` 仍然回滚。

此后的一切——skill 链、bring-up 状态机、roller 预设、limp-fall——都是**在那个形状之上到来**，而非改变它。

### 5.4 "不回退"是验收判据

测量已经存在：`bench_dynamixel_bus` 报达成速率、抖动、读时间、总线时间、利用率、错误、IMU 采样新鲜度，在 50 和 100 Hz 下。把今天的数字记为基线；`robotd` 必须匹配它们。这**刻意不是一个 RT 工程项目**——没有 `SCHED_FIFO`、没有 pinning、没有 `mlockall`。循环今天是可靠的，工作是"在它周围的代码变简单的同时保持它可靠"。

---

## 6. 测试

`FakeIo` 加脚本化采样，无硬件：

- 错过 deadline 时健康变 false；
- 启动采纳当前姿态、绝不命令运动；
- safety 拒绝非有限目标、clamp 越过执行器行程的目标，且一次跌倒既不抢占策略也不抢占调用者的增益；
- limp-fall 预测器在一次跌倒时触发、在一次落脚或静态倾斜时不触发，且它的姿态 ramp 结束于 standing 姿态；
- 意图停止时 deadman 把速度归零；
- **golden observation 向量**——从 mjlab 导出并提交的 `(inputs, expected 61-float array)` 对。observation 里一个错误的索引不会大声失败；它产生一个看似合理、然后摔倒的机器人。依赖一个来自 `microduck_brain` 的导出（§9.2）。

每个测试的注释说它存在是为了防止哪种失败，遵循仓库约定。

---

## 7. 已记录的决策

| | |
|---|---|
| `duck-control` 作为 workspace crate | 编译器强制边界，不搞第二个仓库 |
| 总线层新写、常量借用 | 薄代码，但调好的数字不重新推导 |
| IMU 放进电机的 `sync_read` | 它就是同一总线上的一个设备；无 IMU 抽象 |
| 模型用 Rust 常量 | 只存在一台机器人 |
| params 文件，不 watch | 先确立文件及其位置；watcher 以后再说 |
| 策略路径在 params、默认=release 目录 | 更新携带策略；开发者 override 它 |
| 启动采纳当前姿态 | 一次更新绝不能移动一台站着的机器人 |
| bring-up 做成状态机而非 flag | `set_torque` 是每关节一笔事务 |
| 跌倒判定只报告、不 gate | 对跌倒怎么办是一个控制决策（§2.4） |
| ~~不做里程计~~ —— 已推翻 | `monitor` 的路径图读它，而且它是循环已取采样上的一次 `kinematics` pass（§4.4） |
| 优先级链保持原型的形状 | skills 是对着它的怪癖调出来的 |
| 手柄自成一个 crate | 把 `gilrs` 挡在恢复 CLI 之外 |
| slice 2 之后再做 sim | 硬件是验证路径；`FakeIo` 覆盖笔记本开发 |
| 三速率汇合用"最新值槽 + slew"，不做时间戳同步 | 三个源无共同时钟；手移动慢、音频要平滑（§4.6） |

---

## 8. 刻意推迟

MuJoCo backend 与 `RemoteIo` 协议 · skill 抽象 · 策略 bundle manifest 与 `model_api` gating · `look`/`pose`/`do` 意图 · gaze IK · params 实时 reload 与 config store · 热限制 · 速率限制 · 每设备 IMU 标定。

**里程计已离开这个列表。** 它曾被推迟因为没有东西读它；`robotctl monitor` 的路径图现在读了。§4.4。

---

## 9. 开放问题

1. **Radxa 上的 control rate。** 50 Hz 是从 Pi Zero 2W 继承的。现在有板子了可测——`bench_dynamixel_bus` 和循环自己的五分钟摘要报同样的数字。
2. **golden 向量**值得从 `microduck_brain` 拿到——作为对训练环境的回归检查，而非它们本要充当的真相源。不是任何已发布东西的前置条件。
3. **每关节限制不存在。** safety clamp 到*执行器的*行程，能抓 `NaN`、坏 action scale、垃圾张量——它拦不住一个关节被驱动到机械上不智的地方。真正的限制在 alpha MJCF（31 KB）里，没 vendor 进来。一个"看着像解剖学、其实不是"的限制会暗示一种没人拥有的保护。
4. **alpha MJCF 住哪**（若哪天想要一个 sim/real 一致性测试）——31 KB XML、19 MB meshes，当前在原型的 `scripts/alpha_assets/`。
5. **一个 C 依赖的常驻成本。** `gilrs` 在 Linux 上无条件拉 `libudev-sys`，所以 CI 和板子交叉构建都装它。同样的开销会为下一个"必须到达板子"的 C 依赖重现，所以在那条路径上**优先纯 Rust crate**。*在 macOS 上未验证*：交叉构建需要一个 aarch64 sysroot，Mac 提供不了，所以 `cargo board --bins` 在那里本地失败——用 `-p updater -p robotd -p robotctl` 构建出厂集，或在 Linux 上构建。


