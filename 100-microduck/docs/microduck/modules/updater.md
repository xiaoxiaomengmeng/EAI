# updater / updaterd —— 更新引擎与回滚网（梳理）

> 忠实梳理自官方三份设计文档：
> [`microduck/docs/design/updater-design.md`](../../../microduck/docs/design/updater-design.md)（1468 行）、
> [`restart-order.md`](../../../microduck/docs/design/restart-order.md)（442 行）、
> [`boot-recovery-net.md`](../../../microduck/docs/design/boot-recovery-net.md)（232 行）。
> 保留原文的技术例子、问题背景、推理过程、实践细节与观点分歧；去掉作者身份与推广信息。
> 代码锚点：更新引擎在 `updater/src/`（`engine.rs` 约 3829 行是核心）；`updaterd` 守护进程是它外面的一层进程。

**一句话**：updaterd 是一个**可复用的、配置驱动的更新引擎**——它验证签名、把新版本解压到侧面、用一次 `rename(2)` 原子切换 `current` 符号链接、跑健康门、失败就自动回滚，并用三层回滚网保证**永不把机器人变砖**。适配一个新机器人 = 一份新配置 + 一把新签名密钥 +（也许）一个新健康探针，引擎本身不变。

**范围收窄**：只支持**单一硬件目标**（Radxa Zero 3 / RK3566 / aarch64），没有板型/变体/IMU/摄像头矩阵（§5.6）；OS/内核/驱动不做 OTA（§11）；外设/舵机固件在产线烧录、不做现场更新（§11.1）；没有 fleet 后端——模型是"最新版本，所有机器人"。

---

## 1. 它为什么存在，解决什么问题

机器人跑一块树莓派级别的板子上，发货给**非开发者客户**，更新必须能从配套手机 App 点几下就触发。核心诉求（§2 Goals）：

- daemon 二进制 与 ML 步态模型 + 配置**各自独立更新**（它们几乎永远不会一起发）。
- 支持**多个模型**（walk/jump/stand/ground-pick），各自独立版本、同时全部加载，每个受 daemon 兼容性 gate。
- 从 App 触发并监控更新。
- 用**最低版本地板（min_supported）**强制把机器人从已知坏的 release 上赶走。
- **永不变砖**：原子 apply、自动回滚、签名产物。
- 保持简单，无 fleet 后端。
- **可在团队其它机器人上复用**，改动最小。
- 支持每组件的 **post-install hook** 做迁移/一次性步骤。

**刻意不做**（Non-goals）：OS/内核/驱动的 OTA（走带外，§11）；fleet 管理（分批灰度、逐设备遥测、仪表盘）；增量/diff 更新（当前节奏下全量产物够小）。

### 1.1 关键决策与理由（§3）

| 决策 | 选择 | 为什么 |
|---|---|---|
| 更新粒度 | **应用级**（不是 A/B 镜像） | 只有 daemon + model 变，OS 是静态的。A/B（RAUC/Mender）是过度工程。 |
| 传输 | 机器人**自己从 CDN 拉**，走它自己的 wifi | 机器人有网；手机只是触发器 + 进度显示。大载荷**永不过 BLE**。 |
| 托管 | **GitHub Releases**（daemon）+ **HF Hub**（model） | 零后端；各自是其自然归属。引擎把"source"做成可插拔。 |
| 签名 | **minisign**（CI 签，机器人验） | 久经考验的格式，极小的 verify crate，简单的密钥管理。 |
| 通道 | 每个独立版本线一条：`daemon` + 每个 model 一条 | 节奏不同；模型 reload 无需完整重启。 |
| 回滚 | **版本化目录 + 原子符号链接交换 + 健康门** | 简单、健壮、无需分区。 |
| 复用 | 配置驱动的通用引擎 | 到处是同一个引擎；每个机器人声明自己的组件。 |

**跳过 A/B 分区**有一个真实代价：**不 OTA 内核/系统库**（§11）。缓解办法：只针对 Debian **security** pocket 开 `unattended-upgrades` 打 CVE 补丁（设好即忘）；更大的变动（升内核、daemon 需要的新系统库）= **维修时重刷**；尽量把运行时依赖打进 tarball、静态链接，让一个原装 OS 永远不挡住 daemon 更新。

### 1.2 节奏：两条流（§3、§16.3）

面向客户的 release 节奏：头几个月每 2–3 周一个 daemon release，之后每 2–3 个月一个；模型走它自己独立的节奏。但**内部**迭代快得多（原型见过 3 天 8 个 tag），所以系统必须同时服务两种节奏：一条快的内部 `staging` 流和一条慢的、精选的 `stable` 流。**这正是让 staging 通道和"可脚本化回滚"成为承重件、而非可选打磨的原因。**

### 1.3 前置条件：机器人需要自己的网络（§3.1）

传输决策有个值得直说的后果：**更新要求机器人在它自己的网络接口上有可用的互联网。** wifi provisioning 因此是更新的前置条件，不是独立功能——引导顺序是 *BLE 配对 → 配 wifi → 更新*——一台从没配过网的机器人**根本无法更新**。

手机自己的连接**不能替代**。现实里持续的 BLE 应用层吞吐在算上连接间隔和 iOS 的 pacing 后大约是 **10–30 kB/s**，这让一个 daemon 产物要几十分钟、一个 model bundle 要几小时。这就是载荷永不过 BLE 的原因（§13）：**限制在于链路，不在协议**，再多的 framing 工作也救不回来。

**完全没网时仍然可用的**：`rollback` 和 `reset-to-golden`（§8.2）纯粹操作盘上已有的 release，不需要 manifest、不需要 source、不移动任何字节，所以轻松塞进 BLE，且 `btd` 即使在 `robotd` 死掉时也服务它们（§4.1）。**一台没配网或离线的机器人因此是可恢复的，只是不可升级**——这是这笔交易该失败的那一侧。

---

## 2. 核心不变量：updaterd 必须活着

这是整个系统"永不变砖"承诺的结构性根基。

### 2.1 updater 是与它更新的东西分离的独立单元（§4）

**结构规则**：updater 是一个**独立于它所更新对象的 systemd 单元**。一个进程无法干净地替换自己正在运行的二进制，而且 updater 必须在 daemon 崩溃后存活下来才能执行回滚。

```text
systemd
├── robotd     电机控制 · 运动学 · 步态 ML · 传感器循环   [RT-ish core]
├── mediad     视频/音频采集 · WebRTC/HTTP               [隔离: 媒体崩溃 ≠ 电机死]
├── btd        BLE GATT server: wifi 配置、命名、更新触发/进度
└── updaterd   更新引擎（本文档）
                    ↕ 本地 IPC (unix socket) 连 robotd + btd
```

**推论——updaterd 必须常驻，且必须把自己排除出重启集。** `updaterd` 和 `btd` 都**装在 daemon 产物内部**，所以一个朴素的"重启一切"会在 swap 中途或健康门中途杀掉执行者。`on_apply` 因此重启 release 带的**除这两个之外**的一切。

### 2.2 "排除出 in-flight 重启" ≠ "跳过"（§4，一个真实 bug）

把它读成"跳过"正是当初的 bug。把它们推迟到"下次 boot 或某个显式的稍后重启"，会让两者**无限期跑着旧二进制**：一个常驻的 `updaterd` 用 "client speaks API v4, daemon speaks v3" 拒绝一个更新的 `robotctl`（那个拒绝现在已经没了——陈旧的二进制才是故障，不是版本分歧），而 `btd` 的修复是拿"从没真正跑起来过"的二进制测的。

所以 [`RESTART_AFTER_REPLYING`](file:///Users/damon/Documents/10_code/openSources/100-microduck/microduck/updater/src/engine.rs#L2401) 在结果上了线 **5 秒后**，用一个 systemd transient timer 调度每一个重启——久到够一次写入，短到没人需要等。每个排除的理由恰好在那一刻过期：更新完成了，`btd` 一直在搬运的那个应答也送达了。

这里没有任何东西等到 reboot，而 `robotctl health` 的 unit 块是确认这一点的去处：它打印每个进程**运行自**哪个 release，那是一个失败的延迟重启唯一会显形的地方。

两个列表 [`NEVER_RESTART`](file:///Users/damon/Documents/10_code/openSources/100-microduck/microduck/updater/src/engine.rs#L2806) 和 `RESTART_AFTER_REPLYING` 都是 `["updaterd", "btd"]`，一个测试（`every_unit_held_back_is_restarted_once_the_answer_is_out`）断言它们是同一个集合——**两半是一个决策**。

**排除集写在代码里（`NEVER_RESTART`）而非配置里**：它们是"这两个 daemon 是什么"的属性，不是运维该有机会搞错的选择。早先的设计把这个列表放在 `/etc/robot/updater.toml`，而 `install.sh` 会保留它——于是一个在某 daemon 存在之前 provision 的板子从不重启它，而更新照样报成功（`install-path-gap.md` §4）。

### 2.3 不变量：btd 和 updaterd 要在 robotd 死掉时可用（§4.1）

**`btd` 和 `updaterd` 必须能启动、运行、并完全可用，即使 `robotd` 崩溃、崩溃循环、或根本不存在。** 它们是**恢复路径**——如果它们只能在机器人健康时工作，那么客户真正需要它们的那个情形，恰恰是它们不工作的情形。这是"现场永不变砖"最重要的单一结构属性。

具体：
- **对 `robotd` 无 systemd 依赖。** 没有 `Requires=`/`After=robotd`，不共享命运。`btd` 和 `updaterd` 独立启动并保持在线。
- **无硬 IPC 依赖。** 每个进 `robotd` 的调用（"safe to restart?" 检查、健康探针、模型 reload）都是*可选且超时受限*的。一个死掉的 socket 是一个正常、预期的回答——绝不挂起、绝不 panic、绝不拒绝服务手机。
- **降级模式语义。** `robotd` 挂时，safe-to-restart 检查平凡通过（没东西在动），updaterd 继续。健康 gate 退回"新 `robotd` 起没起来、报没报健康"——这恰是从坏 release 恢复时最要紧的检查。
- **`btd` 报告真相。** App 必须能看到 "daemon unhealthy, version X, last update failed"，并从那个状态提供 *update* / *rollback* / *reset to golden*。一台不能走的机器人仍必须能解释自己、并接受修复。
- **最小依赖面。** 这两个是最后防线，所以保持它们**无聊**：少依赖、无 ML runtime、无媒体栈、无 GPU/摄像头访问。有趣代码里的一个 bug 绝不能有能力打倒恢复路径。

**推论**：更新逻辑不能住在 `btd` 里——作为更新的一个客户端，`btd` 不能是执行更新的那个东西，否则它会在中途杀掉自己。`btd` 保持一个瘦传输，常驻的 `updaterd` 拥有状态机、单飞锁、boot 恢复、以及任何"根本没有客户端在场"的周期/强制检查（§8.1）。updaterd 也**永远不应要求 `btd`**：一个本地 `robotctl`（§15）经 unix socket 必须能驱动恢复，即使 BLE 自己坏了。

---

## 3. 磁盘布局与原子交换（`store.rs`）

### 3.1 布局（§7.1）

```text
/opt/robot/daemon/
├── releases/1.4.1/     ← previous（为回滚保留）
├── releases/1.4.2/     ← new
├── current → releases/1.4.2     ← systemd 单元指向 current/
└── golden  → releases/1.4.1     ← 永不 prune；rescue 读的就是它
```

每个版本完整躺在自己的目录里，互不覆盖；"切换版本"只是改一个符号链接指向哪个目录。原子性来自对符号链接的**单次 `rename(2)`**（同一文件系统上）——**永远不会有半成品状态处于激活态**。

### 3.2 为什么是 rename 覆盖，不是删除+重建（`link_to`）

[`link_to`](file:///Users/damon/Documents/10_code/openSources/100-microduck/microduck/updater/src/store.rs#L165-L203) 先在真链接旁边写一个临时符号链接 `.{name}.tmp`，再 `rename` 盖上去：同目录的 `rename(2)` 是原子的，所以并发读者要么看到旧目标、要么看到新目标，**绝不会看到一个缺失的链接**。删除再重建恰好会打开那个窗口。

目标存的是**相对路径**（`releases/<ver>`），这样即使挂载点移动，树仍然有效（而且在 shell 里读起来也合理）。

**持久性，不只是原子性**：swap 后调 `fsync_parent(&link)`。对 `current`：boot counter 是在 swap **之前** arm 的，所以如果 swap 落了盘而 pending 记录没落盘，一次掉电会留下一个坏 release 处于激活态、却没有 trial 可回退——正是 §7 说"不可能发生"的那个状态。对 `golden`：一个没挺过掉电的链接，是一次"没有目标可瞄"的救援，发生在一台刚在更新中途断电的板子上。

### 3.3 机器人专有状态必须挺过更新（§5.7）

有些文件属于**这台**机器人，一次更新或回滚绝不能摧毁它们。三类，区别对待：

| 类别 | 例子 | 生命周期 |
|---|---|---|
| **Shipped**（被替换） | 二进制、策略 bundle、默认配置、静态 systemd 单元 | 在 `releases/<ver>/` 下；原子交换 |
| **Robot-specific**（永不碰） | 标定数据、每设备生成资产、学习/持久状态（地图、人格、习惯） | 在 release 目录**之外**；跨更新*和*回滚保留 |
| **User preferences**（保留、迁移） | 机器人名字、wifi 凭据、活动模型选择、App 设的调参 | 自己的配置文件；由 hook 在 `schema_version` bump 时迁移（§9） |

两条规则：
1. release 目录是**一次性**的。任何必须存活的东西住在 `/etc/robot/` 或 `/var/lib/robot/`。一次回滚绝不能丢用户数据或逼客户端重标定。
2. **运行时配置住在一个配置文件里，不在 systemd 单元里。** 单元是 shipped 且静态的；它们读一个 updater 从不覆写的配置文件。（原型在 install 时把 CLI flag 烘进一个生成的单元，所以重装会悄悄重置设置——一个值得点名的陷阱。）

推论：重新生成每设备资产必须以稳定 seed 为键、只在真正需要时做，绝不在每次更新时无条件做。

### 3.4 staging、prune、clean_staging

- **staging 目录**叫 `.staging-<ver>`（[`staging_dir`](file:///Users/damon/Documents/10_code/openSources/100-microduck/microduck/updater/src/store.rs#L78-L81)）。前缀被**刻意选成永远无法 parse 成 semver**，这正是让 [`list`](file:///Users/damon/Documents/10_code/openSources/100-microduck/microduck/updater/src/store.rs#L120-L143) 永远不会把一个 staging 目录看成真 release 的东西。它是最终路径的**同级**，好让后面的 `rename` 留在一个文件系统上，且明显标记，让一次崩溃的运行留下"显然是垃圾"而非"可能被误认成真 release"的东西。
- **`prune(keep_previous, golden)`**：删旧 release，保留活动的、`keep_previous` 个最近的其他版本（默认 1，那正是最可能的回滚目标）、以及**无条件保留 golden**。golden 豁免于**计数**，不只是被计数优先——这样 never-brick 保证不会随版本累积而悄悄过期（§8.2）。
- **`clean_staging`**：启动时清理一次被中断运行的 staging 残留（一个 `kill -9` 中途解压绝不能永久泄漏磁盘）。只碰 `.staging-` 前缀，所以真 release 永远不会被它抓到。
- **`current()` / `golden()`**：链接缺失（新机器人）或悬空时返回 `Ok(None)`——两者都是**可恢复状态，不是错误**。

---

## 4. 更新模型：component 与 source

### 4.1 通道（§5.1）

两条逻辑通道，各有自己的 manifest 和版本线，住在**不同主机**上，引擎用每组件一个可插拔的 `source` 处理：

- `daemon` → **GitHub Releases**，tag `daemon-v1.4.2`，assets = artifact + `.minisig` + manifest。
- `model` → **HF Hub** repo，按 git revision/tag 版本化；文件在 `https://huggingface.co/ORG/MODEL/resolve/<rev>/<file>` 解析。**我们自己仍 minisign 签模型产物**并把 `.minisig` 存在旁边——HF 不替我们签。

### 4.2 产物与 manifest（§5.2、§5.3）

一个 release 一个压缩 tarball，如 `daemon-1.4.2.tar.zst`：

```text
bin/robotd  bin/mediad  bin/btd        # (daemon 通道)
model/gait.onnx  config/…              # (model 通道)
version.toml                           # semver, min_hw_rev, schema_version
hooks/postinstall                      # 可选、可执行、带版本+签名
```

manifest 是一个作为 release asset 发布的小 JSON，用 minisign 签名：

```json
{
  "channel": "daemon",
  "version": "1.4.2",
  "url": "https://github.com/ORG/REPO/releases/download/daemon-v1.4.2/daemon-1.4.2.tar.zst",
  "sha256": "…",
  "sig_url": "…/daemon-1.4.2.tar.zst.minisig",
  "min_hw_rev": 3,
  "schema_version": 2,
  "min_supported": "1.5.1",
  "changelog": "…"
}
```

- **`schema_version` 是迁移上下文，不是兼容性 gate。** gate 它会自我拆台：判断一个 manifest 的引擎**永远是上一个 release 的引擎**，所以拒绝 `schema_version > supported` 会让每一次 schema bump 都无法投递——包括携带那个"能理解它的引擎"的 release 本身。迁移由 post-install hook 做，引擎只把数字透传（§9）。
- **`min_supported` 是最低版本地板**（§8.1）。从 v1 就进 schema，即使一开始不用——它**无法被后补**到那些从没学会去看它的机器人上。
- **刻意还没有 `not_valid_after` 字段**：manifest 过期是防冻结攻击的手段，承担其运营成本的决策推迟到 §8.4.2。后加它是叠加式的（缺失 = 无过期），不像 `min_supported`，所以不必现在预留。
- `model` manifest 额外带 `model_api`（§5.5）。

### 4.3 每个模型是一个组件（§5.5）

有几个模型——walk、jump、stand、ground-pick——**各自独立版本化**，且**全部同时加载**而非从多个里选一个。这正是一个 *component* 的定义：一个有自己版本线的东西。所以每个模型是 `updater.toml` 里自己的一个条目，不需要特殊 store 布局：

```toml
[component.model-walk]
install_dir = "/opt/robot/model/walk"
source      = { type = "hf_hub", repo = "ORG/gait-walk", revision = "main" }
on_apply    = { action = "reload", unit = "robotd", signal = "SIGHUP" }
```

每个于是免费且独立地得到：自己的回滚目标、golden release、pin、boot-counter trial、known-bad 历史。`robotctl update apply model-walk` 更新一个模型而不碰其它。试某个的旧版本是 `robotctl update select model-walk 1.1.0`——`select` 重指向任何已安装的 release，不下载。

**模型用 `reload`（SIGHUP → 重新 mmap 权重）而非 `restart`**，所以一次权重交换**永不丢掉电机控制**。每组件的 `on_apply` 正是让这一点自然的东西，也是模型和 daemon 保持分离组件的核心原因。

**一个 "library layout" 被试过又删了。** 早先草稿有个 `Layout::Library` 模式（`library/<ver>/` + 一个 `active` 符号链接），本意是装几个具名 bundle、选一个激活。它其实做不到那件事——它只改了两个目录名，所以在能力上和普通布局**完全相同**，却*看起来*像个特性。更糟的是，"多个已装、一个激活"根本不是模型的形状：它们是**全部同时激活**的。删掉。（如果哪天单个 slot 真需要竞争的候选——两个不同的 walk 模型、用户选一个——那是一个真正的 `(name, version)` store key 和一次真实改动，见 §17。当前设计没有任何东西假装支持它。）

**兼容性（保持轻量）**：daemon 广播一个 **model API version**（`model_api`，如 `1`），描述它实现的传感器输入/执行器输出契约。每个 model manifest 声明它需要的 `model_api`。规则：一个模型只有在 `model.model_api <= daemon.model_api` 时才加载。API version 在接口变化时 bump（如模型开始消费一个**新传感器输入**或产出**新输出**）。一个需要 API `2` 的模型在只会说 `1` 的 daemon 上被拒 → App 提示先更新 daemon；旧模型在更新的 daemon 上继续工作（向后兼容）。**一个够不到的 `robotd` 让这变成 *unknown* 而非 incompatible**——这是等一个 model 的理由，但不是等 daemon 的理由。

### 4.4 硬件：单一目标（§5.6）

**当前目标：Radxa Zero 3**（RK3566，Cortex-A55 → `aarch64`），跑 **Armbian 26.2.x** + **Debian 13 (Trixie)** userland，Rockchip BSP 内核 6.1.115。临时性的——手边最接近的板子而非最终选择。

构建/验证：`scripts/board-test.sh` 用 `cargo-zigbuild` 交叉编译（`zig cc` 提供 cross sysroot 和 `zstd` 需要的 C 编译器），在容器里的真 ARM64 Linux 上跑二进制。glibc floor 钉在 **2.31**（实际 floor 2.30）而非 Trixie 的 2.41，脚本**只在 Trixie 上**跑二进制。这个 pin 防的是*构建主机*：不 pin 的话，构建会链接开发者机器或 CI runner 上的 glibc（比板子的新），于是链接干净、然后在机器人上加载失败，而构建输出里没有任何东西指向原因。内核版本不是约束：`flock`、`SO_PEERCRED`、`statvfs` 都远早于 6.1。

容器覆盖不了、必须靠硬件的：一次真的 `systemctl restart`（`on_apply` 从没对真 systemd 跑过）、eMMC 写入行为与时序、对真文件系统的 `statvfs`、以及健康门 timeout 是否适合一台要几十秒才站起来的机器人。

保留的东西刻意最小：manifest 里的 `min_hw_rev`（§5.3）作为单一前向兼容守卫（一个整数，无矩阵）；一个**稳定设备 ID** 供更新日志用——SoC serial（`/proc/device-tree/serial-number`）可用且挺过重刷，无需 provisioning 步骤去获取。

---

## 5. 签名与信任（§5.4）

- CI 用 minisign 私钥（放在 CI secrets，理想情况离线母钥）签**产物**和 **manifest** 两者。
- 机器人出厂带**一组受信任的 minisign 公钥**（`/etc/robot/trusted_keys/`），**不是单把**。签名只要对*任一*受信任密钥验证通过即有效。**这就是信任锚——不需要 PKI。**
- **密钥轮换**：从第一天就发一*组*意味着一把丢失/泄露的密钥是可存活的——发一个由现有密钥签名的更新，加入新密钥、（稍后）退役旧的。一把烘死的单钥会让丢失变成一条不可恢复的死更新路径。
- **机器人上的验证顺序**：验 manifest 签名 → 下载产物 → 验 sha256 → 验产物签名。**没有未签名的字节会被执行或解压到 live 路径。**

### 5.1 密钥保管（§5.4）

用 `cargo xtask keygen` 生成，它拒绝写进仓库、并把私钥写成 `0600`：

| 密钥 | 加密 | 私钥半住在 | 在谁的受信任集里 |
|---|---|---|---|
| `release-1.pub/.key` | 是 | 密码管理器**和** CI secrets | 每台机器人 |
| `release-2.pub/.key` | 是，**不同 passphrase** | 只在密码管理器——**绝不进 CI** | 每台机器人 |
| `release-3.pub/.key` | 是，**不同 passphrase** | 离线；理想情况永不上联网机器 | 每台机器人 |
| `team.dev.pub/.key` | 否 | 团队库 + CI | 仅开发者板子 |

这张表编码了三件事，每件都容易搞错：
1. **备用密钥必须从第一张镜像就发。** 机器人对着烘进它的密钥集验证，所以一把替代密钥无法稍后经空中引入——一把丢失的唯一密钥意味着手动重刷每台机器人。
2. **备用密钥必须在 passphrase *和* 暴露面上都不同。** 它的目的是在第一把被泄露时存活；`release.key` 出于必要在 CI 里，所以 `release-2.key` 绝不能也在。共享一个 passphrase 会把两者塌成一个。
3. **passphrase 是生成的，不是选的。** 它们从不被键入——住在密码管理器和 secret store 里——所以好记毫无价值，而 minisign 的 scrypt 参数（32 MiB，opslimit 2²⁰）对一个泄露密钥文件的离线攻击只是个小刹车。≥128 bits，如 `openssl rand -base64 24`。

release 在 **CI 里签**，不在本地——一个刻意选择，其补偿性的审批 gate 记在 `ci-setup.md`。dev key 刻意不加密：CI 非交互签名，一个和保护它的密钥存在一起的 passphrase 加不了多少。它真正的保护是**结构性**的——它不在客户机器人的受信任集里，并被那里的 `allow_dev_keys = false` gate 住。

---

## 6. apply 流程（升级）——完整时序

入口 [`apply`](file:///Users/damon/Documents/10_code/openSources/100-microduck/microduck/updater/src/engine.rs#L594-L641) → [`apply_inner`](file:///Users/damon/Documents/10_code/openSources/100-microduck/microduck/updater/src/engine.rs#L644-L821) → [`stage_and_swap`](file:///Users/damon/Documents/10_code/openSources/100-microduck/microduck/updater/src/engine.rs#L824-L1098) → [`post_swap`](file:///Users/damon/Documents/10_code/openSources/100-microduck/microduck/updater/src/engine.rs#L1102-L1138)。**step 8 之前没有任何东西碰 live 路径。** 下面是 [`restart-order.md` §2](../../../microduck/docs/design/restart-order.md) 里从代码逐行读出来的权威 18 步表：

| # | 步骤 | 重启什么? |
|---|---|---|
| 1 | preflight：时钟、robot stopped、无 live session | 否 |
| 2 | 拉 manifest、验它的签名、channel/pin/downgrade/兼容性检查 | 否（但见下） |
| 3 | 再 preflight，这次查磁盘空间（需求来自 manifest） | 否 |
| 4 | 下载到 `releases/.staging-<ver>/dl/` | 否 |
| 5 | 验 sha256，再验产物签名 | 否 |
| 6 | 解压到 `releases/.staging-<ver>/root/`，写 `.updater-manifest.json` | 否 |
| 7 | **`hooks/preinstall`**（cwd = staged tree）——若低于 floor 则装 ONNX Runtime，再跑 release 的 `setup-gstreamer.sh` 装 mediad 的栈 | 否 |
| 8 | `rename` staged tree 到 `releases/<ver>/` | 否 |
| 9 | **arm boot counter**（`pending.json`），在 swap **之前** | 否 |
| 10 | swap `current` → `releases/<ver>` | 否 |
| 11 | **`hooks/postinstall`**（cwd = `releases/<ver>`） | **启动新 shipped 的单元** |
| 12 | `on_apply`——对 §7 的每个单元 `systemctl restart`，一次一个 | **configd, mediad, padd, robotd, tofd** |
| 13 | `releases/<ver>/bin/updaterd --self-test` | 否 |
| 14 | 健康门：经 socket 轮询 `robotd`，每 500 ms，至多 30 s | 否 |
| 15 | confirm boot counter，prune 旧 release | 否 |
| 16 | 释放更新锁，调度延迟重启 | **调度 updaterd, btd** |
| 17 | 应答发到 `robotctl` / App | — |
| 18 | step 16 后约 5 s | **updaterd, btd** |

**从 step 11 起的任何失败都回滚**：swap `current` 回去、对 previous release 重跑 `on_apply`、confirm boot counter、记日志。step 1–10 失败时旧 release 仍 live、没有东西要撤销。

### 6.1 单飞锁（single-flight，`journal.rs`）

[`apply`](file:///Users/damon/Documents/10_code/openSources/100-microduck/microduck/updater/src/engine.rs#L601-L602) 第一件事是拿锁：
```rust
let lock = UpdateLock::try_acquire(&self.config.state_dir)?.ok_or(Error::Busy)?;
```
[`UpdateLock`](file:///Users/damon/Documents/10_code/openSources/100-microduck/microduck/updater/src/journal.rs#L163-L201) 用 std 的 advisory 文件锁（`File::try_lock`，Rust 1.89 稳定，无需依赖）。**用 OS 锁而非 PID 文件**，这样内核在 `updaterd` 被 kill 时释放它——一个陈旧 PID 文件会让机器人**永久无法更新**，那是比一次竞态更糟的失败。`Ok(None)`（已被持有）报 `Busy`，是"要报给 App 的正常回答"，不是错误。

**锁在任何东西被 spawn 之前释放，这个顺序承重**：一次 `fork` 复制进程里每个打开的描述符，所以持锁时 spawn 会把锁的一个副本交给子进程——在一个并行跑多个 engine 的测试二进制里，还会复制*其它* engine 的锁。它表现为无关操作以 `Busy` 失败。`drop(lock)` 之后没有任何东西再碰 store。

### 6.2 preflight 为什么调两次，为什么先查时钟（§7.2）

[`preflight`](file:///Users/damon/Documents/10_code/openSources/100-microduck/microduck/updater/src/engine.rs#L2187-L2200) 每次 apply 调**两次**：一次带 `None` 在任何网络访问之前（时钟、robot stopped、无 live session），再带验证过的 manifest 查磁盘空间（其需求只有从 `size` 才知道）。**拆开它正是让一个未同步的时钟不显现为一个晦涩 TLS 错误、而显现为为它写的那句诊断的东西**：manifest 拉取是 HTTPS，在一块没有电池 RTC 的板子上，错误的时钟会在任何下载之前让证书日期校验失败。minisign 本身与时间无关。

preflight 检查项（任一失败都干净中止、无副作用）：单飞锁；手机无关（机器人自己拉，即使 BLE 断了更新也跑到完成，状态持久化并在重连时回放给手机）；时钟 sane（要求 NTP 同步或 sane-clock 检查 + 有界重试）；robot stopped（向 `robotd` 做一个轻量 "safe to restart?" 查询，不安全就拒绝——对"在运动中重启电机控制"的廉价保险）；无活动远程会话（在 telepresence 中途重启 `robotd`/`mediad` 是个坏惊喜）；磁盘空间（eMMC 有限，为 download + extract + `keep_previous` 验证空闲空间）。

### 6.3 step 2 的守卫（`apply_inner`）

拉到 manifest 并 `verify_manifest` 验签后（**签名通过前什么都不发生**），依次是若干拒绝：

- **channel 检查**（`check_channel`）；**pin 检查**（若被 pin 到别的版本则拒绝）。
- **`AlreadyCurrent`**：若 manifest 版本 == 已装版本，停止并报 `already_current`，什么都不下载/交换。但这里回答的是**错误的问题**在一个情形下：release 装了、而某个 daemon 在从**另一个** release 服务。那正是运维伸手去 `apply` 通常想修的，而回答"已是最新"告诉他们没什么可修。所以它调 [`stale_units`](../../../microduck/docs/design/restart-order.md) 点名那些跑着别的版本的单元、在应答后重启它们（见 §7.6）。
- **回滚攻击守卫**（downgrade attack，§8.4.1）：**只守卫 `Target::Latest`**。一个签名证明产物是*我们的*；它对产物是否*最新*什么都没说。一个陈旧或被回退的镜像能提供一个旧的、仍有效签名的 manifest，把整个 fleet 悄悄倒退回一个我们撤回的版本——这是签名产物方案上的经典降级攻击。`Exact` 豁免（那是定向回退怎么做到的，是运维的刻意动作，镜像诱导不了）；`Ref` *永远*看起来像降级（dev 构建是 semver prerelease，排序低于它前驱的 release），守卫它会拒绝每一次分支安装；`from_dir` 豁免（一个 root 在命令行点名的目录不是一个倒退回去了的镜像，而且本地构建是个 prerelease、排序低于板子正在跑的，守卫它会拒绝每一次笔记本推送）。`rollback` 和 `reset-to-golden` 刻意向后走、且不经过这里。
- **`staging_has_nothing_newer`**：`--staging` 用在一台已经领先候选通道的板子上（一个 release 直接晋升到 stable、没发 staging 候选时可达）。**拒绝而非安装**：这里以下每一层在被要求时都行为正确（产物验证了、swap 发生了、一个旧 release 不含的单元起不来、更新回退了），但运维要的是"正在被测的那个"而它不存在，所以答案是一句话而非一次回滚。
- **兼容性检查**（`manifest.compatibility`）：`Ok` 放行；`Refused` 返回 `Incompatible`；`Unknown`（通常因 `robotd` 够不到）——对 daemon 通道放行（那次更新正是修一个死掉的 `robotd` 的办法），但对声明了 `model_api` 的 model manifest 则等待。

### 6.4 step 4–7：验证、解压到侧面、orphan、pre-hook（`stage_and_swap`）

- **下载**带进度合并（`DownloadProgress`）：source 每 HTTP chunk 说一次，没人需要那么多，所以 coalesce 而非逐条转发。drop sender 让 pump 看到 end-of-stream 并转发它拥有的一切——`abort()` 会丢掉最后几个更新，让一次下载可见地卡在 97%。
- **step 4 完整性、然后真实性**，都在任何东西被解压之前：`verify_sha256`（对着 manifest 的 `sha256`），然后 `keys.verify_file`（minisign）。哈希和验签流式处理几百 MB、在这类板子上要几秒，跑在 async worker 上会卡住那些本该继续回答 `status`/`subscribe` 的 IPC 任务，所以都进 `spawn_blocking`。
- **step 5 解压到侧面**（`extract_dir`），绝不盖在 live 路径上；也是 CPU-bound（zstd），进 `spawn_blocking`。把验证过的 manifest 作为 `.updater-manifest.json`（[`EMBEDDED_MANIFEST`](file:///Users/damon/Documents/10_code/openSources/100-microduck/microduck/updater/src/engine.rs#L166)）写进 release，供 `select` 和溯源用。
- **step 5b orphan 检查**（§7.3，见本文 §13）：`would_orphan` 检查这个 release 会不会让一个已装单元无可执行文件。在这里而非 preflight，因为候选的文件列表此刻才存在；在 dry run 返回之前，因为"这次降级会不会工作"正是 dry run 被问的。此刻什么都没动：staging 是一次性的，boot counter 未 arm，`current` 仍在原处。
- **dry_run** 在此返回 `DryRunPassed`。
- **step 6 pre_install hook**（600 s 上限，见 §12），在 release 变 live 之前。
- **step 7 发布**：一次 `rename(extract_dir, release_dir)`，然后 **arm boot counter（在 swap 之前）**，然后 `store.swap_to`。（arm 的顺序见 §9.2。）

### 6.5 step 8：post_swap——此后一切失败都回滚

[`post_swap`](file:///Users/damon/Documents/10_code/openSources/100-microduck/microduck/updater/src/engine.rs#L1102-L1138) 是 swap 之后、因此能触发回滚的三件事：
1. **post_install hook**（120 s）；
2. **run_apply_action**（restart daemon / reload+SIGHUP model）；
3. 若是 Restart 动作，`self_test_updaterd`——**在 commit 前用新二进制自测 updaterd 本身能起来**（§7.4）；
4. **health_gate**（§8）。

结果处理（[`stage_and_swap` 尾部](file:///Users/damon/Documents/10_code/openSources/100-microduck/microduck/updater/src/engine.rs#L1049-L1097)）：
- **Ok** → `boot_counter.confirm` + `store.prune(keep_previous, golden)` + 返回 `Applied`。prune 是 best-effort（更新已成功），但一次失败必须可见，否则一台慢慢填满 eMMC 的机器人看起来完全健康。
- **Err** → `rollback_to(previous)`；成功返回 `RolledBack`，无 previous 可退返回 `Stuck`（"什么都没被回退，所以说 rolled back 是撒谎"）。

### 6.6 step 16–18：延迟重启（`schedule_deferred_restarts`）

对 `updaterd` 然后 `btd`：
```text
systemd-run --on-active=5s --timer-property=AccuracySec=100ms -- systemctl restart <unit>
```
四个承重细节：
- **一个 transient unit，不是子进程。** `systemctl restart updaterd` 杀掉 `updaterd` 的整个 cgroup，而 `updaterd` 的一个子进程*在*那个 cgroup 里——它会在重启自己父进程的途中被杀。一个 `systemd-run` timer 活在它外面。
- **更新锁先 drop**（step 16，在 spawn 之前，理由见 §6.1）。
- **在 `Applied` 上、以及在带 stale 单元的 `already_current` 上**——[`restarts_owed`](file:///Users/damon/Documents/10_code/openSources/100-microduck/microduck/updater/src/engine.rs#L2683-L2685) 决定是哪个，且它是唯一决定它的东西。前者欠固定的那对；后者欠 §7.6 发现的跑着错误 release 的那些（通常就是 `updaterd` 自己）。**一次回滚什么都不欠**：它让常驻的 `updaterd` 已经匹配 `current`，所以在那重启只是无意义的搅动。
- **每个单元都走 timer**，包括那些更新会在 step 12 就地重启的。给 `configd`/`robotd` 开第二条立即路径会买到五秒、代价一个机制。
- **失败被记录、从不返回。** 一次成功的更新不会因为一个重启无法调度而被报成失败；代价是一个 daemon 跑在旧二进制上直到下次 boot。

timer 在应答被写之前*就已调度*（引擎跑在 `update.apply` 调用内部），所以那 5 s 从 step 16 起算，而非从客户端看到答案起算。客户端看到它的结果、然后是一个断开的连接——对 BLE，一次普通的重连。

**被重启的 `updaterd` 随后做什么**：它是一次正常启动，所以完整跑 §9.1 的启动序列（`clean_staging`、`record_boot`、§7.6 的 reconciliation——一次没发生的 `btd` 重启正是在那里被抓到——然后服务）。两个容易错过的后果：**boot counter 数的是 `updaterd` 启动次数，不是字面的 boot**（一次成功的 apply 在 step 15 confirm 了自己的 trial，所以没有悬而未决的可推进——但一个被*更早*中断的更新留下 armed 的 trial 会被这次重启推进）；**周期检查的时钟重置**（`INITIAL_CHECK_DELAY` 是进程启动后 60 s，`check_interval` 从那里起算）。


---

## 7. 重启次序（`restart-order.md`）

这一页是"每个移动 `current` 符号链接的路径、以及 boot 时"的权威次序。写它是因为"哪个 daemon 重启了、在什么点"曾被三份文档以三种不同方式回答，而这个答案决定了一次 skew 怎么被诊断。**每一步都从代码读出，且每步点名拥有它的函数**；当别处的叙述性注释和这页不一致时，**注释才是 bug**。

### 7.1 七个 daemon，和一个不是 daemon 的单元

一个 release 发七个 daemon——`robotd`、`configd`、`btd`、`padd`、`mediad`、`tofd`、`updaterd`——每个 `ExecStart` 一个 `/opt/robot/daemon/current/bin/` 下的路径，所以每个都在符号链接移动的瞬间变陈旧，且每个要么被更新重启、要么在更新后重启。

它还发一个 `robot-boot-check.service`，**它不是它们之一，且一次更新绝不能重启它**：它问"boot 起来的那个 release 上来了没"、没上来就交给 `robot-rescue`，所以在更新中途跑它等于把一个回滚检查指向一群合法地正在重启中途的 daemon。它不带 `[Install]` 段，而那正是让它出局的东西（§7.2）。

| unit | 更新中途重启 | 应答后约 5 s 重启 |
|---|---|---|
| `robotd` / `configd` / `padd` / `mediad` / `tofd` | 是 | — |
| `updaterd` | **从不**——它是执行更新的那个进程 | 是 |
| `btd` | **从不**——它可能是更新经由到达的那个传输 | 是 |

**没有任何东西被 held 到 reboot。** 这在 `RESTART_AFTER_REPLYING` 存在之前是真的，也是关于这个系统最常见的陈旧说法。因为"调度一次重启"不是"它发生了"的证据，下一次 `updaterd` 启动会拿每个单元对着活动 release 检查、重启任何陈旧的（§7.6）。

### 7.2 什么算"一个更新会启动的单元"（`has_install_section`）

**一个带 `[Install]` 段的 `systemd/*.service` 文件。** 没有的那个由别的东西触发（一个 timer，或另一个单元把它拉进来），所以它的生命周期不是更新该驱动的，而 [`engine.rs::has_install_section`](file:///Users/damon/Documents/10_code/openSources/100-microduck/microduck/updater/src/engine.rs#L2862-L2864) 是决定这一点的地方。

**用规则而非一串名字**，因为已经有两个地方在应用它、而且它们不一致：`hooks/postinstall` 拒绝 `enable --now` 一个无 `[Install]` 的单元、并明说理由——*"对它 `enable --now` 会在安装它的那次更新中途跑一个回滚检查，daemon 合法地正在重启中途"*——而引擎读每个 `*.service` 并在片刻后照样重启它。以这个段为键，两者**按构造一致**，下一个这样的单元无需谁记得。一个**读不了**的单元文件算作"要重启的一个"：那更可能是权限问题而非刻意无触发的单元，而一次响亮失败的重启胜过一次被悄悄跳过的。

### 7.3 集合怎么算（`units_shipped` vs `units_to_restart`）

```text
units_shipped(release)  = (release 自己的 systemd/*.service stems) ∪ (updater.toml 的 on_apply.units)   排序去重
units_to_restart        = units_shipped(…) − NEVER_RESTART{updaterd, btd}
```

一次更新重启 `units_to_restart`；§7.6 的启动检查读 `units_shipped`，因为一次更新碰不了的那两个单元，恰是它存在去看的那两个。今天的 release 上 `units_to_restart` 恰好是 `configd, mediad, padd, robotd, tofd`（字母序，所以在每台板子和每个测试里次序相同）。

**从 release 而非从板子派生**的两个后果：`padd` 被重启，尽管 `deploy/updater.toml` 从没提它——配置列表是**叠加式**的、不是权威的，它只需点名 release *不*发的单元；一个 `updater.toml` 早于某 daemon 的板子仍重启那个 daemon——那个文件属于运维、`install.sh` 保留它，这正是 `configd` 曾有一阵子没被重启的由来（`install-path-gap.md` §4）。

**每个单元由它自己的 `systemctl restart <unit>` 重启，从不批量**——单个 `systemctl restart a b` 在任一单元未知时**整体失败**、且*不*重启存在的那个。一个 systemd 报 `LoadState=not-found` 的单元被跳过并警告；一个存在但重启失败的单元**让更新失败并回滚它**。

### 7.4 step 13 —— self-test（`self_test_updaterd`）

`updaterd` 在更新期间不重启自己，所以一个**起不来的替换二进制**否则会在下次 boot、commit 之后才被发现，而恢复活在正在失败的那个进程内部。取而代之：新的 `bin/updaterd` 用 `--self-test` 跑——它加载 `/etc/robot/updater.toml`、构造引擎、在碰任何状态**之前**退出（10 s 上限）。

一次失败是 `Error::SelfTest`，携带那个二进制的**最后一行 stderr**（所以 "config error: unknown field `foo`" 到达回滚理由，而非被压平成 "unreachable"），并回滚。一个没有 `bin/updaterd` 的 release（一个 model 组件）平凡通过。**除非 `on_apply` 是 restart 否则整个跳过**——这正是让它不挡 bootstrap install 路的东西。

它抓到的：一个错架构的二进制、一个缺失的库、一个立即 panic，以及最可能的那个——**一个新 `updaterd` 拒绝板子现有的 `updater.toml`**（那是运维的文件、跨 install 保留，而一个 release 可以自由改变它期望什么）。

### 7.5 其它转换（`transition_to`）

`select`、`rollback`、`reset-to-golden` 共享 [`engine.rs::transition_to`](../../../microduck/docs/design/restart-order.md)：arm、swap、`on_apply`、健康门、失败则回退。它们**不跑 hook、不 self-test**——没有产物被解压，所以没有新东西要装或证明。

| | hooks | `on_apply` | self-test | 延迟 updaterd/btd 重启 |
|---|---|---|---|---|
| `update apply` | 是 | 是 | 是 | **是** |
| `update select <ver>` | 否 | 是 | 否 | **是** |
| `update rollback` | 否 | 是 | 否 | **否** |
| `update reset-to-golden` | 否 | 是 | 否 | **否** |
| 失败 apply 内的自动回滚 | 否 | 是 | 否 | **否** |
| 启动时 boot-counter 回退 | 否 | 是 | 否 | **否** |

底部两个 `否` 按构造正确：在一次失败的 apply 和一次 boot-counter 回退里，`updaterd` 和 `btd` 从没被重启，所以它们仍是属于"被退回到的那个 release"的二进制。显式 `rollback`/`reset-to-golden` 的 `否` 是另一种情形、值得知道：如果被离开的那个 release 曾在更早某点成功 apply 过，`updaterd` 和 `btd` *确实*被重启到了它上面，而一次显式回退把它们留在那——`robotd`/`configd`/`padd` 退回去了，那两个**留在活动 release 的前面**。这在下次 `updaterd` 启动（而非 reboot）时自行解决：§7.6 双向比较，所以一个跑着比 `current` 更新的 release 的 `btd` 按同一测试是陈旧的、会被重启。`updaterd` 是那个唯一不自愈的进程，**刻意如此**。

因为 `select` 和回退跳过 `hooks/postinstall`，它们也从不*移除*一个单元文件。一次降级到早于某 daemon 的 release 会把那个 daemon 的单元留着、指向一个旧 release 不含的二进制；单元失败，而它在重启集里，失败的重启让转换失败（`install-path-gap.md`，见 §13）。

### 7.6 启动 reconciliation —— 重启到底发生没（`reconcile.rs`）

`systemd-run` 返回成功只意味着一个 transient timer 被创建，不意味着重启跑了、也不意味着新二进制起来了；§6.6 的 step 16 刻意吞掉那些失败。这留下**一条让机器人跑在一个它没安装的 release 上的静默路径**，且它恰好落在没有别的东西看守的那两个单元上。[`reconcile.rs`](file:///Users/damon/Documents/10_code/openSources/100-microduck/microduck/updater/src/reconcile.rs) 关闭它。

**它读什么**：每个 daemon 启动时发布自己的身份——`duck_ipc_proto::publish_identity`，由每个 daemon 开篇的 `log_startup_identity!` 宏调用：
```text
/run/<service>/identity.json    { service, version, revision, built_at, exe, pid }
```
`exe` 字段来自 `/proc/self/exe`，所以它**穿过 `current` 符号链接解析**、点名进程实际被启动自的那个 release 目录；`Identity::release()` 从中 parse 出 `releases/<ver>`。**自己发布而非从外面读**：一个进程知道自己的版本和 revision、能无特权读自己的 `exe`，而读另一个用户的需要 root。单元里的 `RuntimeDirectory=<service>` 创建那个由单元 `User=` 拥有的目录、挺过 `ProtectSystem=strict`、且**在单元停止时被 systemd 移除**——所以一个停止的 daemon 无法留下一个自称在跑的身份。

**它决定什么**：每组件、对 `units_shipped` 里每个单元（所以 `updaterd` 和 `btd` 都包括，这正是重点），比较发布的 release 和组件的活动 release。`verdict_for` 是个纯函数，每个 syscall 都留在它外面。

| verdict | 何时 | 动作 |
|---|---|---|
| `Current` | 发布 == 活动 | 什么都不做，静默 |
| `Restarted` | 两者不同 | `systemctl restart <unit>`，warn |
| `ReportedOnly` | 不同**且单元是 `updaterd`** | 记录，从不行动 |
| `RestartFailed` | 尝试了重启但失败 | error 记录 |
| `Unknown` | 没有身份文件 | 什么都不做 |

三个刻意的不行动：
- **`updaterd` 在这里从不重启自己。** 一个对"哪个 release 活动"持不同意见的后继会重启、再次不同意、循环——在那个拥有恢复的进程里，所以没有任何东西能打破循环。**只报告是安全的**：进程刚启动，所以任何关于它的陈旧都是在这段代码跑之前决定的。它从另一侧被修（§6.6 的 `restarts_owed`）。
- **一个停止的单元被留着停止。** 没有身份文件意味着要么停止了、要么太老不会发布，两者在这里读起来一样。启动它会在每次 `updaterd` 启动时覆盖"谁停止了它"。
- **一个什么都没发布的 daemon 不算陈旧。** 为"旧"而重启一台机器人的 daemon 是个没人要的决定；下一次更新让它们能回答。

比较是**方向无关**的——任何差异都是陈旧——这正是让它覆盖"一个被显式回滚留在 `current` *前面*的 release"、而不只是留在后面的东西。它在每次 `updaterd` 启动跑，所以在 boot 时是个 no-op（一切都从同一符号链接启动）；成本是每单元一次文件读。

**从 `apply` 做同样的读**：[`reconcile::stale_units`](../../../microduck/docs/design/restart-order.md) 是对同一批身份文件跑 `verdict_for` 但把行动拿掉，`apply` 和 `select` 在它们的 already-current 路径上调它。它存在正是因为上面那个例外：一个陈旧的 `updaterd` 是这个模块拒绝修的唯一 skew，而事后唯一会看它的人是"因为某 daemon 看起来不对而跑 `apply`"的人。那过去回答 `already_current` 且什么都不调度，读起来像"确认没什么可修"。两个属性让它不成为 §7.6 守的那个循环：它在**请求时**触发而非每次启动，且经 `systemd-run` 调度而非从被重启的进程内部驱动 `systemctl`。它也**报告**：单元被点名在 `already_current` 结果里，所以客户端看到哪些 daemon 没跑着机器人已装的 release。

### 7.7 首次安装（`scripts/install.sh`，§6）

裸板子上按序：
1. 写 `/etc/robot/updater.toml` 和受信任密钥。
2. `bootstrap_first_release`：取一个独立的 `updaterd` 二进制、跑 `updaterd install [--from <dir>]`。那是普通的 `Engine::apply` 但强制两个设置，因为在一块没有 release 的板子上它们是**事实而非策略**：`on_apply = none`（单元住在正被装的 release 内）和 `health = none`（没有 `robotd` 可探）。所以 §6 跳过 step 12/13/14——但 **step 11 仍跑**，`hooks/postinstall` 正是装单元、`enable --now` 每个带 `[Install]` 段的那个的地方，daemon 第一次启动就在那里、从 hook 内部。step 16 也不跳过：bootstrap install 在 `install.sh` 还在跑时就调度 updaterd/btd 5 s 后重启。
3. `install_units`：再拷一遍单元、`daemon-reload`，然后按序 `enable --now` updaterd、robotd、configd、btd、padd、mediad——`configd` 在 `btd` 前因为 `btd` 向 `configd` 要配对 PIN，`mediad` 最后因为它的单元 `After=` 另外三个。`btd`/`padd`/`mediad` 可能失败而不让安装失败（一台没无线电、没手柄、没摄像头的机器人仍能更新和走）。
4. `install_token_dropin`：写 `GITHUB_TOKEN` drop-in、`daemon-reload`、`systemctl try-restart updaterd`（光 `daemon-reload` 会让*正在跑的* `updaterd` 没有 token，那是每台板子）。

### 7.8 诊断一次 skew（§7）

两个版本号可以合法地同时不同，是哪一对决定诊断：

| 观察 | 意味着 |
|---|---|
| 更新后几秒内 `updaterd`/`btd` 落后于已装 release | 预期——延迟重启还没触发 |
| `btd`/`robotd`/`configd`/`padd`/`mediad`/`tofd` 持续与 `current` 不一致 | 重启没生效*且* §7.6 没修好它——要么 `updaterd` 之后没重启，要么它的重启失败（journal，error） |
| `updaterd` 持续落后 | 延迟重启从没落地，且 §7.6 不修这个。`robotctl update apply daemon` 修它——报 `already_current` 且 `updaterd` 在 `stale` 里、并调度重启 |
| 任何 daemon 报 `restarting` | 它退出了、systemd 在拉回来。更新后几秒内那是重启本身；持续则是它根本起不来（摄像头 flex 拔了的 `mediad` 就是这行为它存在的）——§7.6 不管它，因为没在跑的 daemon 没有身份可比 |
| 任何 daemon 报 `failed` | 它起不来、systemd 停止重试了。同一个 journal，且不像 `stopped`，没人选择它 |
| 任何 daemon 报 `build unknown` | 它在跑、但没发布身份：一个早于此机制的构建，或 `exec` 到第一行之间的瞬间。§7.6 不管它，一次更新让旧的能回答 |

**问机器人而非推断**：`robotctl health` 的 `units` 块给每个 daemon 打印一行——它进程被启动自的 release，与已装不一致时警告。手动同样答案：`cat /run/configd/identity.json; readlink /opt/robot/daemon/current`。

---

## 8. 健康门与即时回滚（§8）

apply 之后，新 `robotd` 必须在一个每组件 timeout 内清除一个**健康标志**（电机 ack、模型加载、媒体管线 init）。

### 8.1 health gate 的三态（`health_gate`）

[`health_gate`](file:///Users/damon/Documents/10_code/openSources/100-microduck/microduck/updater/src/engine.rs#L2095-L2185) 对 `HealthCheck::Socket` 轮询 `robotd`（每 `HEALTH_POLL_INTERVAL`=500 ms，到 timeout，如 30 s）：

- **`Healthy`** → `GatePassed::Healthy`，通过、commit。
- **`Degraded(reason)`** → **warn 但照样 commit**。理由（和 §10 的 boot net 不能修硬件的理由同一个）：一个没有舵机供电的 `robotd` 在 golden 上以同样方式失败。在那里回退会把一个硬件故障藏到一次软件变更后面、也回退下一个 release，而最糟的是——**在没说一声的情况下换掉正拿着机器人的人脚下的代码**。这不是假设：一块装了分支构建、在上面配对了手柄的板子，两次 boot 后回来跑着 stable release，之后每个命令都跑在没人要过的代码上。
- **`Unhealthy` / `Unreachable` / `Incompatible`** → 轮询到 timeout 后 `Err`，触发回滚。`Incompatible`（"以一个这个 updaterd 读不了的形状回答"）失败得像 `Unreachable`、但读起来完全不像它——对一个正在服务它 socket 的机器人说 "unreachable" 会把读者送到系统的错误半边一小时（见 `install-path-gap.md`）。

`HealthCheck::Command`（一个 exec probe）只有二态：成功退出 = `Healthy`，否则 `Err`——**一个 exec probe 报 pass 或 fail，没法说"降级但不是我的错"**，那个区分是 `robot.health` 经 socket 才有的。`HealthCheck::None` 没有可 gate 的东西，平凡返回 `Healthy`（这是设计最弱的状态：swap 一成功就 commit，根本没有自动回滚，boot counter 是唯一剩下的恢复机制——离开那个状态正是 M1 的意义，§16.5）。

**什么可以、什么不可以进入判定**（和 robotd.md §3.4 呼应）：只有**一个 release 能被归咎的**条件才可以设 healthy/degraded，因为它们是更新系统的输入。电池、温度是**描述**，任何自动决策都不得读它——gate 在电池上意味着一台低电量时被更新的机器人会回滚这个 release、然后在同一块低电量电池上评判它的替代者，于是**根本没法被更新**。

### 8.2 即时回滚（`rollback_to`）

post_swap 失败走 [`rollback_to(previous)`](file:///Users/damon/Documents/10_code/openSources/100-microduck/microduck/updater/src/engine.rs#L1193-L1268)：

```rust
store.swap_to(previous)?;              // swap 回旧版本
self.boot_counter.confirm(component)?; // 清除 trial
// ↑ 这两个失败 = Error::RollbackFailed（最严重级别，刻意与普通过失区分，好让支持立刻看到）
// run_apply_action 失败 → 只记为 apply_error，带进 outcome
```

**swap 和 trial 就是恢复**：过了这里机器人*已经*回到它来自的 release，只有这两个失败意味着它没法被放回去。**apply action（重启 unit）失败不再报 `RollbackFailed`**——它可被一个普通坏 release 触发：`hooks/postinstall` 覆写 unit 文件且刻意不放回，所以一个*因某 unit 起不来*而失败的 release 会留下那个文件，而如果被退回的 release 也发一个同名 unit，这次重启就继承同样的失败（`scripts/systemd-test.sh` 精确复现这个）。把它报成回滚失败是在断言唯一不成立的那件事（swap 已发生）、埋没唯一成立的那件事（有个 daemon  down）。所以它被带进 outcome，两者都说。`RollbackFailed` 保持它的原义：**机器人根本没法被放回去**。

若 `previous` 为 `None`（一个首装、gate 失败、又没配 golden）：坏 release 留着 linked，但 **trial 照样清除**——留着它 armed 会让每次后续 boot 都"恢复"同一个不可恢复的更新、每次追加一条假回滚、永不收敛。调用者报 `Stuck`，如实说什么都没被回退。

### 8.3 回滚目标怎么选（`rollback_target`）

[`rollback_target`](file:///Users/damon/Documents/10_code/openSources/100-microduck/microduck/updater/src/engine.rs#L1145-L1157) 选**严格*更旧*于 `current` 的最高已装 release，跳过任何最近记录 outcome 是回滚的**。**两个约束，都是硬学来的**：

- **严格更旧。** 一个朴素的"最新的非 current"会在一次自动回滚后向*前*走，因为刚失败的那个 release 还在盘上（失败路径刻意不 prune）。那会让 `rollback`——支持工程师在一次坏更新后第一个伸手去敲的命令——**把坏更新重装回去**。
- **非 known-bad。** 一个 journal 记录为回滚过的 release 不是一个安全落脚点，即使它是最新的那个更旧的。`known_bad` 从 journal 的*每版本最新 outcome* 派生，所以它在一个 release 确实成功后**自清**（latest-outcome 而非 ever-failed，这样一个失败过一次、后来成功的版本不会被永久拉黑）。

### 8.4 keep_previous（§8）

`keep_previous`（默认 1）保留的 release 目录**约束磁盘占用，同时总是留一个已知可用的回滚目标**。config 校验拒绝 `keep_previous = 0 且无 golden`（那会不留任何回滚目标）。

---

## 9. boot counter：崩溃/掉电恢复（`journal.rs` + `recover_on_start`）

健康门的前提是 updaterd 进程活着并跑完了 gate。但 boot counter 处理一种**不同的**失败：**一个 boot 起来、跑起来了、却从不报健康的 release**，以及**一个根本起不来的 release**（后者最终由 §10 兜）。

### 9.1 数据结构（`BootCounter`）

[`BootCounter`](file:///Users/damon/Documents/10_code/openSources/100-microduck/microduck/updater/src/journal.rs#L319-L423) 的状态存在 `pending.json`，每条是一个 [`PendingUpdate`](file:///Users/damon/Documents/10_code/openSources/100-microduck/microduck/updater/src/journal.rs#L329-L336)：
```rust
struct PendingUpdate { component: String, version: Version, previous: Option<Version>, boots: u32 }
```
一条 `PendingUpdate` 就是该版本正处于 **"trial（试用期）"**。`arm` 写入 `boots = 0`；`record_boot` 给每条 `boots += 1`（`saturating_add`）；`exhausted` 是 `boots >= MAX_BOOT_ATTEMPTS`（= **2**）；`confirm` 移除该 component 的 trial（全空时删文件而非留 `{}`，好让"有没有悬而未决"可由存在性回答）。

**trial 按 component 为键**：一个 model 选择绝不能覆盖或清掉一个 daemon 更新的 trial，否则一个 swap 后崩溃的 daemon release 会悄悄丢掉 never-brick 保证所依赖的那条记录。每个 component 的 trial 独立推进。

### 9.2 arm 为什么在 swap 之前（`stage_and_swap` step 9）

```rust
std::fs::rename(extract_dir, &release_dir)?;         // step 8 发布目录
self.boot_counter.arm(&PendingUpdate{ boots: 0 })?;  // step 9 ← 先 arm
store.swap_to(&manifest.version)?;                    // step 10 ← 再 swap
```
[`arm`](file:///Users/damon/Documents/10_code/openSources/100-microduck/microduck/updater/src/journal.rs#L345-L358) 的注释：**在 swap 之前写，所以一次在 swap 和健康检查之间的崩溃仍可恢复**——相反的顺序会留下一个**未记录的坏 release 处于激活态**。（再叠加 §3.2 的 `fsync_parent`：swap 落盘而 pending 没落盘的掉电窗口也被堵上。）代码里有个 `faults.abort_after_swap` 故障注入，模拟 swap 后立即 `kill -9`——符号链接指向新 release、boot counter 仍 armed，恢复是 `recover_on_start` 的活，一个测试随后演练它。

### 9.3 recover_on_start：启动恢复序列

[`recover_on_start`](file:///Users/damon/Documents/10_code/openSources/100-microduck/microduck/updater/src/engine.rs#L1648-L1845) 在 socket 被服务**之前**跑（[`main.rs::serve`](../../../microduck/docs/design/restart-order.md) step 5），所以一台 boot 进坏 release 的机器人在任何东西能请它做别的事之前就已经开始回退。顺序（每个都 load-bearing）：

1. **`clean_staging`**（每个组件）：删 `releases/.staging-*`。
2. **`record_rescue`**：若 `<state_dir>/rescued` 存在，说明 boot 恢复网（§10）在没有任何东西运行时把 `current` swap 到了 golden。把它作为一次 rollback 拷进更新日志、**清除该 component 的 trial**、删文件。**在 `record_boot` 之前是刻意的**：救援已经走得比 trial 会走的更远（到 golden 而非 previous），一个留着 armed 的 trial 会在一两次 boot 后把 `current` 从救援刚选的 golden 上挪走——恢复网和 never-brick 保证互相拆台，只差一次 boot。删文件也是**释放救援的循环守卫**的东西（§10.5）。
3. **`refresh_golden_links`**：把每个组件配置的 golden 发布成 `current` 旁边的 `golden` 符号链接，好让 rescue 用一次 `readlink`、无解析器就能找到（§10.4）。
4. **`record_boot`**：给 `pending.json` 里每个 armed trial 的 `boots` +1。
5. **对每个 `boots >= 2` 的 trial**：回退到 `previous`，当 `previous` 缺失、盘上不见、或本身被记录为回滚过时升级到 `golden`（§8.2 的链）。这意味着 swap 符号链接、confirm trial、重跑 `on_apply`——**所以一次 boot-counter 回退会重启 `configd`/`padd`/`robotd`**。它不跑 hook、不调度任何东西，所以 systemd 刚启动的 `updaterd`/`btd` 进程被留着从"正被放弃的那个 release"跑。§7.6 随后抓 `btd`（它在这之后跑、对着回退使之活动的 release 比较）；`updaterd` 被报告并留着，陈旧到下次 boot。
6. **journal 结果**。这里一次失败被记录、服务继续：拒绝服务会移除修它的唯一办法。

**"预算决定何时问；机器人决定是否回退"**（§8）：trial 到这一步意味着没有 apply 确认过它——通常原因不是坏 release，而是一次 apply 在 gate 跑之前就被杀（正是 release 自己的 `hooks/postinstall` 重启 `updaterd` 时发生的）。所以 `recover_on_start` 重问健康门那**同一个三态问题**（仅对 socket gate；`HealthCheck::None` 没什么可问，`Command` 只答二态）：healthy → confirm（预算白花在一个一直好好的 release 上数 boot）；degraded → confirm（回滚修不好的硬件原因）；其它 → 回退（`robotd` 不回答、或回答不健康，正是这个预算存在的目的）。

**trial 只在一次 apply 于 §6 step 9–15 之间被中断时才存在**——一次掉电、一个 `kill -9`、或一次 swap 后被取消的 RPC。一个 committed 的更新什么都不留 armed。因为 `exhausted` 是 `boots >= 2` 而 `arm` 写 `boots = 0`，中断后*第一次* `updaterd` 启动记 "update still on trial"、*第二次*回退。回退结果若是 `Some` → `RolledBack`；若无可退目标（无 previous 无 golden）→ `Stuck`（"需要运维介入"），且 `rollback_to` 已清 trial，所以只报一次而非每次后续 boot。

---

## 10. boot recovery net：updaterd 自己起不来（`boot-recovery-net.md`）

这一页拥有 **boot 截止时间及其谓词、它看守的单元的成员规则、`robot-rescue`、`golden` 符号链接、以及循环守卫**。

### 10.1 它关闭的缺口

`updaterd` 在更新期间从不重启自己，因为它是执行更新的那个进程。三层收窄了这代价，**且三层都跑在一台完全起来的板子上**：`self_test_updaterd` 在 commit 前跑 release 自己的 `updaterd --self-test`（§7.4）；延迟重启在更新应答后 5 s 把 `updaterd`/`btd` 放到新二进制上（§6.6）；启动 reconciliation 从后继那一侧检查那些重启发生了（§7.6）。

**但一个在运行的板子上能起来的二进制，仍可能在 *boot* 时失败**——那时网络没起来、某外设没枚举、单元次序不同。而 reconciliation 跑在 *`updaterd` 启动时*，所以它**预设了一个能启动的 `updaterd`**。这最后一点正是这整套存在的理由：`BootCounter::record_boot` 从 `Engine::recover_on_start` 跑、在 `updaterd` 内部、数的是 `updaterd` **启动次数**而非字面 boot——所以**一个不启动的 `updaterd` 不计数，never-brick 保证恰好经由那个可能是牺牲品的进程关闭**。留在那里的话，一台板子有一个盘上的好 release、一个 armed 着要退回它的 trial、却没有任何在跑的东西能对二者采取行动。

### 10.2 部件

| | |
|---|---|
| `robot-boot-check.timer` | 每次 boot 180 s 时触发 |
| `robot-boot-check.service` | oneshot：问 release 上来没、没上来就交接 |
| `scripts/robot-boot-check` | 谓词。装在 `/usr/local/sbin/robot-boot-check` |
| `scripts/robot-rescue` | swap `current` 到 golden、记录、reboot。`/usr/local/sbin/robot-rescue` |
| `<install_dir>/golden` | 指向那个有常驻保证的 release 的符号链接，由 `updaterd` 发布 |
| `<state_dir>/rescued` | 面包屑：上一次尝试，也是循环守卫 |

两个脚本都是 `/bin/sh`，且都被 `scripts/install.sh` 和 `hooks/postinstall` **拷贝**进 `/usr/local/sbin`——和那两个拷贝（而非穿过 `current` 读）的单元文件是同一个决定、同一个理由：**一个作为 release 载荷里的二进制发布的 rescue，会被它存在去挺过的那些 release 恰恰弄坏**（错架构、缺共享库、启动 panic）；一个穿过 `current` 读的 rescue 会把恢复路由经过正被恢复的那个东西。

### 10.3 触发器：一个截止时间，不是 `OnFailure=`

`OnFailure=` 看着像天然契合——systemd 恰在一个单元耗尽重启、进入 `failed` 时触发它，无轮询、无常驻 watchdog。**它不适合*这些*单元**，它们被配置成不会到达那个状态：

| unit | 策略 | `RestartSec` |
|---|---|---|
| `btd` / `padd` | `Restart=always` | 5 s |
| `configd` / `robotd` | `Restart=always` | 2 s |
| `updaterd` | `Restart=on-failure` | — |

`Restart=always` 在每个情形都刻意、且在单元文件里有论证。没有单元覆盖 `StartLimitIntervalSec`/`StartLimitBurst`，所以 systemd 默认的"10 秒 5 次启动"适用——而一个每 5 s 重启的 `btd` 从不接近它。**它无限期崩溃循环、从不进入 `failed`**，而那正是 `OnFailure=` 等的状态。让 `OnFailure=` 触发意味着重调这些单元好让 daemon *放弃*：**为了启用救援情形而降级正常情形**。

一个截止时间反转了问题、不需要单元改动、且抓得住那个永远重启的家伙。**`OnBootSec=180`** 因为板子比看起来慢——`hci0` 直到上电后约 73 s 才存在（其中 `bluetooth.service` 花 26 s 阻塞在 `dbus` 后面）——而 daemon 等它们的硬件而非退出，所以一次慢 boot 不是一次失败 boot。

**谓词**：两个条件，都是"systemd 试过跑一个 daemon 却留不住它"的正面证据，任一成立、任一成员：`ActiveState` 是 `failed`；`NRestarts` ≥ 3。其它一切被留着不管，**而这个非对称正是重点**：一个运维停掉的单元是 `inactive` 无重启，绝不能让板子丢掉它的 release（和 §7.6 留一个停止单元停止是同一个判断）；一个 release 早于的单元没加载、什么都不答，读起来一样；一个还在等硬件的 daemon 是 `active`；一次恢复了的崩溃留一两次重启、不是三次。**一个假阴性花掉运维一次诊断；一个假阳性花掉一个好 release。** 三次而非一次，因为 `Restart=always` + 2–5 s `RestartSec` 意味着一个真循环在截止时间内轻松过三次，而一个死于瞬时原因的 daemon 不会。

**三种它拒绝触发的方式**：oneshot 上的 `Conflicts=shutdown.target`（一个关机路上被杀的 daemon 可被记为 `failed`；没有它，一个错误时刻的 `systemctl poweroff` 最终把机器人 reboot 进 golden）；一个 **uptime 守卫**（过 10 分钟，是别的东西而非 timer 启动了它）；oneshot 上**无 `[Install]` 段**（两个安装器都读作"别 enable 这个"，否则 `hooks/postinstall` 对刚写的单元跑 `enable --now` 会在安装它的更新中途跑一个回滚检查）。

### 10.4 成员规则与 rescue

对"任何 daemon 失败 → 回滚"的反对是：一个 daemon 可能因回滚修不好的原因失败（缺无线电、没上电的电机总线），而为一个硬件故障回退一个好 release 比什么都不做更糟。这化解成一条**准入规则：一个单元只有当它*等它的硬件*（缺失或行为异常）而非退出时才可加入。** 这样一个 down 的成员意味着*二进制*坏了（错架构、缺库、启动 panic、它拒绝的一个配置）——那正是回滚修的。

| daemon | 在集合里 | 依赖缺失时 |
|---|---|---|
| `robotd` | 是 | 等电机总线，"等着，不命令任何东西" |
| `updaterd` | 是 | 无论如何服务；它是恢复路径，必须在东西坏时可达 |
| `btd` | 是 | 每 5 s 重试一个适配器；若出现又不答就重跑整个 bring-up |
| `configd` | 是 | 服务 `system.*`/`pad.*`，报 `net.state=unavailable` |
| `padd` | 否 | 无 robotd socket 时干净退出，刻意如此 |

`btd` 凭本事（而非对称）挣得一席：没网时 BLE 是唯一的进入方式，所以一台没有 `btd`、又没有已知 wifi 的机器人**根本无法被够到**。`configd` 同样：`system.pin` 是 `btd` 拿到手机用来认证的 PIN 的地方，所以一个 down 的 `configd` 把手机路径一起带走。若未来某 daemon 在缺硬件时退出，要么修那个 daemon、要么把它留在集合外，**别削弱规则**。

**rescue**：`robot-rescue` 读两个符号链接、**什么都不 parse**。这不是省字节——它最可能救援的是一个**其 `updaterd` 拒绝板子 `updater.toml`** 的 release（那是运维的文件、跨 install 保留，而一个 release 可以自由改变它期望什么）。**一个 parse 同一个文件的 rescue 会死于它正在治的病。** golden 是*那个文件里*的一个版本（`ComponentConfig::golden`），所以 `refresh_golden_links` 在每次 `updaterd` 启动把它发布成 `golden` 符号链接：一次 `readlink`，没有可拒绝的东西。**它是缓存而非第二真相源，且失败安全**——若 `updaterd` 停止启动，链接仍点名上次写入时是 golden 的那个 release，那是一个已知可用的 release。一个配置了但*未安装*的 golden 根本不发布链接，因为一个悬空链接会告诉 rescue 它有目标而其实没有。

**用 golden 而非 previous**：当坏掉的是恢复路径本身时，previous 可能也坏。golden 是那个有常驻保证的 release，且 `prune` 从不移除它。它在三种情况拒绝、并说是哪种：没有 golden 链接、golden 点名的 release 未安装、以及 **`current` 已经是 golden**——最后这个最要紧：daemon 那时正在那个带常驻保证的 release 上失败，所以这不是一次 release 故障，而 swap 除了多加一次 reboot 什么都改不了，**那正是一个硬件故障变成 reboot 循环的方式**。swap 把一个临时符号链接 rename 盖到 `current` 上（匹配 `Store::link_to`），因为 `ln -sfn` 会 unlink 再重建、留下一个 daemon 无法 exec 的窗口。`--reboot` 是可选的：每个单元经 `current` exec，所以 swap 在它们重启前什么都不做，而一台 daemon 死掉时正站着的机器人是一台会倒的机器人——谁拿着它谁决定。boot check 传这个 flag；控制台前的一个人得到打印出来的命令。

### 10.5 循环守卫，以及什么打开它

两件独立的事阻止"swap 并 reboot"自己发生两次：① `current` 已经是 golden 时的拒绝（覆盖普通情形，因为一次成功救援恰好留下那个状态）；② **面包屑**（覆盖①漏掉的——一次更新把 `current` 从 golden 挪走、又失败了）。

`Engine::record_rescue` 在下次 `updaterd` 启动清面包屑（先把它拷进更新日志）。所以**守卫恰好在板子证明它能再跑它的更新 daemon 时打开、在它不能时保持关闭**：一台静坐、journal 干净的机器人胜过一个永远 reboot 的。`robot-rescue --force` 是手动越过它的办法。那个清除跑在 **boot counter 之前**，且顺序 load-bearing（§9.3 step 2）。

面包屑是 `key=value` 而非 JSON，尽管 `updaterd` parse 它：
```text
at=1786453421
install_dir=/opt/robot/daemon
from=1.2.0
to=1.0.0
because=boot check: robotd.service (failed, 7 restarts)
```
写者是一个在"东西已经出错"的板子上跑的 shell 脚本，而在 `sh` 里正确引用 JSON 是产出一条什么都读不了的记录的方式。[`journal::Breadcrumb`](file:///Users/damon/Documents/10_code/openSources/100-microduck/microduck/updater/src/journal.rs#L281-L317) 出于同一理由**宽容**——每个字段都可选——因为一条 parse 不了的面包屑是一个永不打开的守卫。`install_dir` 是把这条 entry 归属到一个组件、而不需要第二个地方和 `updater.toml` 就"daemon"这个名字达成一致的方式。

**证据**：一次静默回滚是"一天怎么溜进 'it works on my board'"的方式，所以 swap 留三条痕迹——journal（两个脚本，经 `logger` 若有、且总是 stderr）；更新日志（作为 `Outcome::RolledBack`、失败单元点名在其 reason 里——这是 `robotctl update log` 显示的，且它永久，不像被清的面包屑）；`robotctl version`（间接但有用：一台跑着比已装更旧 release 的板子正是它报告的）。

**什么没被测**：决策被测——`xtask/tests/rescue.rs` 对一棵临时树问两个脚本每个问题，用一个 stub `systemctl` 供应单元状态并记录是否被要求 reboot，`scripts/board-test.sh` 在真 ARM64 Linux 上覆盖安装器那半。**systemd 没被测**：timer 是否在 `OnBootSec=180` 触发、`Conflicts=shutdown.target` 是否在关机路上让 oneshot 让开、`NRestarts` 是否按谓词假设的那样在一个崩溃循环单元上读——全都未验证。除了真 systemd（`systemd-nspawn`，或一个以它为 pid 1 的特权容器）没有什么能回答它们。**这是"为它论证"的第二个理由：恢复网是那段当其它一切都不工作时必须工作的代码，也是最不适合测试的代码。** 在那之前，这个机制在一块 bench 板子上验证、否则不验证，而做那件事的办法是给一块板子一个 `robotd` 起不来的 release。


---

## 11. 强制升级地板 / kill switch（§8.1）

**已实现**，包括让它工作的那部分：config 里的 `check_interval` 让常驻 `updaterd` 定时轮询每个 source，`auto_apply` 决定那个定时器**在没有客户端等待时**可以装什么。一次调度 pass 在另一个更新 in-flight 时整个跳过，且第一次检查在 boot 后延迟（网络常常没起来，而一起重启的 fleet 会成群到达）。

**不设 `check_interval` 会禁用轮询——连同地板一起**，因为一台没人点更新的机器人永远学不到地板存在。`updaterd` 在这种情况下启动时警告，且在 `auto_apply` 设了但没有定时器跑它时**另行**警告。

### 11.1 auto_apply —— 无人值守更新策略

| | |
|---|---|
| `off` | 从不；可用性被记录，一个 mandatory release 被大声记录 |
| `mandatory` | **默认**——只有地板标记为 mandatory 的 release。普通的等客户端 |
| `all` | 每个可用的 release |

**一个有序设置，而非每紧急度一个 flag**，这样一个 config 无法表达"自动 apply 普通更新但不 apply mandatory 的"——即自动更新除了"为救援一个坏 fleet 而发布的那些 release"之外的一切。`mandatory` 是默认，因为替代方案是一个 fleet 卡在我们已撤回的 release 上；普通 release 等待，因为*一台机器人何时重启是它主人的决定*，那正是整个 App 驱动更新流存在要给他们的决定。`all` 是 canary 和 bench 设置，正是它让 §16.2 的 Tier 2 可配置（跟踪 `staging`、装每个候选的实验室机器人）；发给客户机器人会**从他们手里夺走重启决定**。

**无论策略如何，一次无人值守 apply 是一次*普通* apply**：它跑同样的 preflight（所以一台正在走或有远程会话的机器人拒绝、并在下个 interval 重试，而非在某人手下重启）、同样的健康门（所以一个起不来的 release 被回滚）。这也是**为什么不需要维护窗口**：`safeToRestart` 是对"现在是不是坏时机"比时钟更好的回答，且它跑在任何网络访问*之前*，所以一台忙的机器人跳过它零成本。

**这属于 daemon，不属于 cron。** 一个调 `robotctl update apply` 的 systemd timer 或 crontab 看着等价、其实不是：显式 apply **刻意绕过下面的 known-bad 守卫**（因为一个重试某 release 的运维可能已经修好了原因）。一个外部 timer 会继承那个绕过、丢掉保护——拿了各自的错误一半——并重新引入下一段存在去防的那个循环。

### 11.2 known_bad —— 无人值守路径拒绝一个已回滚过的候选

**无人值守路径拒绝一个这台机器人已经回滚过的候选**，无论策略如何、即使 release 是 mandatory。没有那个守卫，一个坏 release 是一个**没有出口的 fleet 级陷阱**：check 说可用 → apply → gate 失败 → 回滚 → 等 `check_interval` → 重复。每个循环重新下载产物、重写 eMMC、重启 `robotd`，所以每台机器人既不可用、又在磨损，靠电池，没有客户端介入去注意，**循环里没有任何东西收敛**。

守卫是 `known_bad`（从 journal 的*每版本最新 outcome* 派生，所以在一个 release 确实成功后自清，§8.3）。它**只**适用于无人值守路径：一次显式 `robotctl update apply` 仍重试，因为运维可能已修好原因、拒绝他们会移除那个明显的检查办法。守卫触发时以 `error` 记录——一台卡在 mandatory 地板下的机器人需要一个修好的 release，那该响亮。这个测试**数 journal 里的尝试次数**而非检查 live 版本：每个循环后符号链接都回到好 release 上，所以机器人的*状态*看着正确、即使循环在跑。

### 11.3 min_supported 地板（§8.1）

manifest 的 `min_supported` 让我们即使在 "latest-for-all"（一台机器人否则只在被点时才更新）下也能把机器人赶下一个已知坏 release：check/启动时，若运行版本 `< min_supported`，更新是 **mandatory**——`updaterd` 不等点就 apply（且运行中的坏版本可被拒绝）。一个更钝的 **kill switch**（一个签名的 "version X revoked" 声明）是同一机制的升级。

**它是签名的事实**意味着攻击者无法*伪造*一个——但见 §15：一个签名说一个产物是**我们的**、不说它是**最新的**，所以光签名不防一个旧真 manifest 的重放。字段现在就进 schema（§4.2），因为它无法后补。**`min_supported` 只在一台机器人成功拉到一个带它的 manifest 后才生效**，所以它是一个*补救*工具、不是防御：一台被切断更新的机器人永远学不到地板存在。

---

## 12. Post-install hooks（§9）

和 dpkg `postinst` 同思路，但 hook 装在**（签名的）tarball 内部**，所以没有未签名代码会跑。Rust 二进制或 shell 脚本——引擎只是 `exec` 它。

**契约**：
- 在符号链接 swap *之后*、`apply` *之前*跑（`pre_install` hook 若存在，在 swap *之前*跑）。
- **pre-install 比 post-install 上限长——10 分钟对 2 分钟**（[`PRE_INSTALL_HOOK_TIMEOUT`](file:///Users/damon/Documents/10_code/openSources/100-microduck/microduck/updater/src/engine.rs#L65-L66) = `UPDATE_MAX_SILENCE_SECONDS`=600 s，[`HOOK_TIMEOUT`](file:///Users/damon/Documents/10_code/openSources/100-microduck/microduck/updater/src/engine.rs#L48)=120 s）。pre-install 是那个装"release 需要、板子可能没有"的东西的 hook（ONNX Runtime；mediad 的 GStreamer 栈，在一块从没有过它的板子上约 100 MB apt），它负担得起这些分钟**恰恰因为什么都没被 swap**：旧 release 仍 live 且服务着，所以一个长 pre-install 是一次慢更新、而非一台状态不明的机器人。
- **非零退出 ⇒ 更新失败 ⇒ 回滚。**
- 提供的环境变量（缺失的值被**省略**而非设为空，好让 hook 能区分"首装"和"未知"）：`UPDATE_COMPONENT`/`UPDATE_CHANNEL`（都设、同值）、`UPDATE_NEW_VERSION`、`UPDATE_OLD_VERSION`（有 previous 时）、`UPDATE_RELEASE_DIR`（正被装的 release，也是 hook 的工作目录——**这通常是一次迁移想要的**）、`UPDATE_INSTALL_DIR`（组件根）、`UPDATE_NEW_SCHEMA_VERSION`/`UPDATE_OLD_SCHEMA_VERSION`。
- hook 以一个**清空的环境**加一个固定 `PATH` 跑，所以行为不取决于 systemd 恰好怎么调用了 `updaterd`。
- 输出被捕获、截断，且**无论成败都到达更新日志**（失败时在 error 内，成功时逐行）：因为 pre-install hook 的报告——哪些插件、哪个 runtime、这块板子有没有 NPU——是"这台机器人在这个 release 到底能做什么"的答案，而一次 `journalctl -u updaterd` grep 曾在一块 hook 明明跑过的板子上回来空的。每 hook 输出上限 `MAX_OUTPUT` = 8 KiB。

### 12.1 "如果一个 fresh install 做它，hook 就做它"（§9.1）

**一个 release 直到它需要的一切都在板子上才算装了。** 把一个文件发进 release 目录不是安装它；把一个脚本发进 release 目录不是运行它。hook 是唯一在每台板子、每次更新都跑的东西，所以它是"这台板子在这个 release 工作前必须有 X"归属的地方——不在一个 X 存在之前跑过一次的 provisioning 脚本里，也不在某个人的记忆里。

**问题不是"release 需不需要它"，而是"一个 fresh install 做不做它"。** "需要"是个判断，而它已经放过一个情形：一段把机器人名字放进 shell 提示符的片段不是一个 release *需要*的东西，所以它进了 `install.sh`、别处没有，而任何只更新过的板子都没有它。改问那个机械的问题——**`scripts/install.sh` 写不写这个文件、跑不跑这个命令？**——因为那个能靠 grep 一个文件回答，也因为一台不是全新的板子得到的恰好是 hook 做的、别无其它。

**这已被搞错四次，四种不同形状**：
1. **单元。** 一个加了 daemon 的 release 把它的 `.service` 放进产物、却没放进 systemd 看的地方。`btd` 在一块 release 完整正确的板子上以 `203/EXEC` 失败，而 `on_apply` 无法重启一个还不存在的单元。（`install-path-gap.md` 是记录；`hooks/postinstall` 是修复。）
2. **GStreamer 栈和 3A 引擎。** provisioning 装它们，所以在它们存在之前 provision 的板子没有，一个插件比 release 构建所对的更旧的板子也没有。`hooks/preinstall` 在每次更新跑 release 自己的 `setup-gstreamer.sh` 和 `setup-rkaiq.sh`，那是关闭它的东西。
3. **NPU。** 加鸭子检测器的分支写了 `setup-npu.sh`、把它打进 release 挨着模型——**却从不调用它**。每台板子都会发一个够不到 NPU 的检测器，直到有人 SSH 进去，而发现这个的办法是 `rknn_init` 返回一个数。
4. **登录 shell。** `install.sh` 长出一段 `/etc/profile.d` 片段把机器人名字放在 hostname 旁边——`microduck@radxa-zero3 (coincoin):~$`——好让三个 ssh 窗口对三只鸭子不是三个相同提示符。没有别的东西装它，而 `install.sh` 不被打进 release，所以没有 hook *能*。一个在它被写之前 provision 的板子吃了此后每次更新却从没得到它；发现的办法是在一个 release 含该特性的板子上 `ls: cannot access '/etc/profile.d/robot-name-prompt.sh'`。

**形状每次都一样：工作*做了*，而让工作到达板子的那个东西被漏了。** 它在 review 里不可见，因为那个加脚本的 diff 看着完整。第四次还加了一种它*看着对*的方式：它在 `install.sh` 里的两个邻居（登录 banner 和 `robotctl` 补全）有同样的缺口且更老，所以周围的代码读起来像该跟随的模式——**连续三个只有 fresh install 跑的函数不是一个先例，是三个实例**。

**推论**：
- **`install.sh` 对板子做的任何事，一个 hook 也做。** 其它三个是这一条的实例，而它的方向正是错误实际被犯的方向：一步被加进 `install.sh`（在那里容易在正被 provision 的板子上测），而更新路径从不听说它。`install.sh` 不在产物里、hook 无法调它，所以共享的那步属于一个两者都跑的 `scripts/setup-*.sh`——那正是 `setup-gstreamer.sh` 和 `setup-login.sh`。测试 `every_install_sh_step_reaches_an_updated_board` 读 `install.sh` 的调用点，除非每个都被一个 hook 也做、或被写下来（带理由）作为"只有 fresh install 做的事"，否则失败。它是个**强制函数而非证明**——逃生舱是表里的一行——但四个实例都得为那行辩护，而没有一个能。
- **一个 hook 步骤必须幂等，且在无事可做时必须便宜。** 前者因为它在每台板子、每次更新跑（"已完成"是常态而非例外）；后者因为成本在每次更新被支付到永远：一个在已应有尽有板子上要 10 秒的步骤是给每个未来 release 加 10 秒，而 hook 阶段每特性长一步、预算却不动。可行的形状是一个 **stamp**：`setup-npu.sh` 把 runtime 版本写到 `/usr/lib/librknnrt.version` 并比较，`setup-gstreamer.sh` 比较一个 stamp 和两个 `dpkg -s` 调用，两者在一块已配好的板子上都不做可测量的事。
- **hook 跑的一个脚本必须被打包。** `every_script_the_hooks_run_is_packaged` 从两个 hook 里读 `script=scripts/…` 赋值，任一打包点漏一个就构建失败。
- **一个脚本需要的必须和它一起旅行。** `setup-rkaiq.sh` 从挨着它的 C 文件编译一个 LD_PRELOAD shim；`setup-npu.sh` 从挨着它的 `.dts` 编译一个 device-tree overlay。各有一个测试说这一点，因为失败是静默的：脚本跑了、找不到它的源、更新成功、日志里一条警告。
- **可选硬件永不致命。** hook 的非零会让更新失败并回滚，所以一个 hook 只可为"意味着 release 不可安装"的事失败。一台没摄像头、没蓝牙适配器、没 NPU 的板子仍是一台机器人；那些 install 步骤说出丢了什么、点名重试的命令、并返回成功。

**这正是 provisioning 和 updating 之间那个分工的意义**：`provision.sh` 和 `install.sh` 装配一块新板子、跑一次。在某物存在之前 provision 的每台板子由一次普通更新修——这意味着一个 release 不可对"它的板子何时被 provision"做任何假设，而一个只由 provisioning 执行的 setup 步骤是一半 fleet 永远得不到的步骤。

---

## 13. orphan 检查：这个 release 会让一个已装单元变孤儿吗（§7.3）

`hooks/postinstall` 装一个 release 发的单元、并在回滚时把它们留在后面（§12）。对回滚那是对的——下次成功更新重装它发的一切。但对一次**降级到早于引入某 daemon 的那个 release** 就不对：单元留着，它的 `ExecStart` 点名一个旧 release 不含的二进制，systemd 以 `203/EXEC` 失败它，而因为那个 daemon 在派生的重启集里，失败的重启让更新失败、进而回退。**在一块解析到 stable `0.2.0`（早于 `configd`）的板子上观察到过。**

所以一个缺某已装单元所 exec 二进制的候选被拒——[`orphan.rs`](file:///Users/damon/Documents/10_code/openSources/100-microduck/microduck/updater/src/orphan.rs) 的 `would_orphan`，`Error::WouldOrphanUnit`。它读 `/etc/systemd/system/*.service`、过滤到那些 `Exec*=` 指向组件 `current` 符号链接的单元：live 目录是孤儿唯一出现的地方（因为它比安装它的 release 活得久），而这个过滤正是把"我们的 release 从没发过的单元"挡在外面的东西。**任何它 parse 不了的东西产生 no finding**——为一个解析器只是没看懂的单元文件拒绝一次更新是更糟的失败。

三个值得保留的放置决定：**不在 preflight**（两趟 preflight 都在产物下载前跑，候选文件列表还不存在；这在解压后、swap 前跑，仍是"无副作用"——staging 一次性、boot counter 未 arm、`current` 没动，代价是一次下载才知道）；**在 dry run 返回之前**（"这次降级会不会工作"正是 dry run 被问的）；**没有 target 豁免**，不像只在 `Latest` 上触发的 `WouldDowngrade`（那个关于一个镜像提供陈旧 manifest，这个关于一个起不来的单元，它不在乎 target 怎么被点名——而 `Ref` 正是这个情形被观察到的方式）。

不在 rollback、reset-to-golden、`select` 上：那些刻意向后走、正是一台板子下一个坏 release 的办法，所以任何能拒绝的东西都不属于恢复路径（§2.3）。拒绝点名那个单元、那个缺失的二进制、以及越过它的办法——移除单元（`systemctl disable --now configd.service && rm /etc/systemd/system/configd.service`）。**刻意不是一个 `--force` flag**：移除单元正是运维的本意（一台低于引入某 daemon 的 release 的板子本就不该跑那个 daemon），它让情形成真、而非覆盖一个说它不为真的检查，而下一次发那个单元的更新会重装它。

---

## 14. 更新日志与 run transcript（§8.3）

把最近 N 次更新尝试（时间戳、channel、from→to 版本、outcome、error）持久化到盘，可经 BLE/HTTP 取回。**这是客户报"更新失败"时支持要的第一件东西**，所以它也记录被拒和被回滚的尝试，不只成功的。

### 14.1 update-log.jsonl（`Journal`）

[`Journal`](file:///Users/damon/Documents/10_code/openSources/100-microduck/microduck/updater/src/journal.rs#L26-L161) 是 append-only：newline-delimited JSON，然后 `sync_data`——**durable 到足以挺过一次失败更新自己可能引发的掉电**。一行被撕碎的尾行在**读时被容忍**而非当损坏（丢掉最后一条 entry 可接受，因它拒绝读整个日志不可接受）。`trim` 用 write-and-rename 保持有界（`max_entries`），所以一次 trim 中途崩溃不会留下截断的日志。[`known_bad`](file:///Users/damon/Documents/10_code/openSources/100-microduck/microduck/updater/src/journal.rs#L140-L160) 从它派生（§8.3、§11.2）。

### 14.2 run transcript（`runs/<id>.jsonl`）

日志回答*一次尝试发生了、怎么结束的*；它回答不了一次尝试**做了什么**，而那个问题也有一个诚实的家。

**为什么不用 journal（systemd 的）**：`journalctl -f` 是看一次更新发生的对地方、无需机械；但它是*保存*它们的错地方。这块板子的 `/var/log` 是一个 zram 设备，所以 `Storage=persistent` drop-in 买到的是一次干净 reboot 的存活、**不是一次掉电的存活**——而任何人需要这个的更新恰恰不成比例地是那些以掉电结束的。更糟，phase 时间线**从来就不在 journal 里**：phase 只作为 `update.progress` notification 发出，所以状态机对自己的叙述只在"一个客户端持有 socket 开着"期间存在，而一次没人看的 `robotctl update apply` 留下一个中间缺失的轮廓。

**一次 run 记录什么**：一 run 一文件，`runs/<id>.jsonl` 在 `state_dir`，每行 `fsync`（和日志同规则），在每个 `install_dir` 之外（所以一次 swap 或回滚无法摧毁关于那次 swap 或回滚的叙述，§3.3）：run 的开场（caller 点名的 target、source、当时 live 的是什么、以及从 `SO_PEERCRED` 得到的"谁请的"）；通过签名检查的 manifest（版本、hash、size、URL、**哪把**受信任密钥放行了它、它构建自的 revision）；每个 phase 边界（带 detail、因此带各自耗时）；hook 输出逐字（带退出码，成败都记）；每个被重启/reload/延迟的单元及 systemd 说了什么；健康门的裁决（那是决定 commit 对回滚的）；怎么结束的。

一条日志 entry 带 run 号，所以两者像 `git log` 和 `git show` 那样组合：`robotctl update log` 是索引、`robotctl update show` 是详情。

**记录从不让一次更新失败**：每次写是 best-effort、失败报给 journal。一次完成但丢了日记的更新，严格胜过一次为保日记诚实而被放弃的更新。

**在别处结束的 run**：一次 daemon 更新重启 `updaterd` 自己，所以它的 transcript 按设计停在中途，而裁决在几分钟和一次 reboot 后、从 boot-counter trial 到达。那次回退**开它自己的一个 run** 而非重开第一个（第一个进程没了、无法 append）。配对反而清晰：一个没有 `ended` 的 transcript 是一个裁决在别处的 run，`show` 在裁决本该在的地方这么说，而下一个 run 就是那个裁决。rescue-to-golden 路径（由一个 shell 脚本在 `updaterd` 启动前做）得到一个从它留的面包屑重建的薄 run、标记为二手。

**界限**：留 20 个 run，每 run 4000 events 和 2 MiB，每 event 64 KiB。超上限时 writer 停止并记录丢了多少，且**结尾无论如何被写**——一个缺尾的 transcript 绝不能读成一个停在那里的 run。`hooks::MAX_OUTPUT`（8 KiB/hook）已经把最大贡献者限住，所以一次普通 daemon run 是几 KB。

**journal 由客户端拼接、不由 daemon**：`robotctl update show` 对 run 的窗口跑 `journalctl --utc`、scoped 到 run 自己说它碰过的那些单元、并追加。读系统 journal 是 `updaterd` 有、却不该经一个"读侧刻意不设 gate"的 socket 外借的特权——而且它是覆盖*其它* daemon（transcript 从来看不到）的那一半。一个没有 journal 访问权的运维得到 transcript 加上"要跑的命令"去取其余。

---

## 15. 签名买到什么、不买什么：降级与冻结（§8.4）

一个 minisign 签名证明一个产物**来自我们、且没被改**。它对*何时*发布、是否仍最新**什么都没说**。两种攻击在一个完全有效的签名下存活，它们是任何签名产物方案的标准对：

| | 是什么 | 状态 |
|---|---|---|
| **降级 / 回滚** | 提供一个*更旧、真签名*的 manifest，让机器人倒退到我们撤回的版本上 | **已修**（§6.3） |
| **冻结** | 永远提供*当前*的 manifest，让机器人永远学不到有修复存在 | **开放**（§15.1） |

两者都可被任何控制"机器人拉到什么"的人达到：一个陈旧或被回退的 CDN/镜像、一个缓存代理、DNS 拦截、或一个敌对的本地网络。**两者都不需要一把被偷的密钥。**

### 15.1 冻结 —— 开放，需一个决定（§8.4.2）

当前没有东西区分"没有更新"和"有人在阻止我看到一个更新"。一台被无限期钉在一个脆弱版本上的机器人是现实的伤害——同一个它已在跑的版本，所以没有完整性检查触发。选项（从最便宜）：① **签名 manifest 过期**（加 `not_valid_after`，拒绝更旧的；代价是发布变得时限绑定——CI 必须按计划重签 `stable` manifest 即使无 release 变化，否则每台机器人开始警告；且依赖机器人时钟，那正是 §6.2 的时钟检查存在去不信任的）；② **单调 manifest 计数器**（一个只增的 `sequence`，机器人记见过的最高值、拒绝更低的；稳健地检测元数据*回滚*、无时钟依赖，但**不检测冻结**）；③ **报告陈旧而非强制**（记"上次成功拉 manifest"、在 `status`/App 显示"上次检查 47 天前"；密码学上什么都不检测，但让一台被冻结的机器人对它主人和支持*可见*；无发布负担、无时钟依赖）；④ **phone-home**（§17，一台上报的机器人让*我们*看到哪些停止了检查，唯一能中心化检测冻结的选项，需后端）。

**推荐：现在做 (3)，等有发布节奏可挂时做 (1)。** 陈旧报告几乎免费、没有自己的失败模式、把一次静默攻击变成一次可见的；过期是真防御但运营成本（一个若错过就警告整个 fleet 的重签计划）在发布管线成为例行之前不值得付。

**为 v1 显式接受**：一台网络敌对的机器人可被阻止更新。它**不能**被弄得*降级*、装一个我们没签的产物、或装一个通不过健康门的产物。**那些才是我们真正依赖的属性。**

---

## 16. 托管、CI 与晋升（§6、§16）

### 16.1 托管在 GitHub Releases（§6、§6.1）

artifact + `.minisig` + `manifest.json` 是每个 release 的 asset。"最新是什么"经 GitHub API 查匹配 channel 前缀（`daemon-v*`）的最新 tag（偏好，若两通道留一个 repo）。

**一个私有 repo 无法服务 fleet——所以这个 repo 不会保持私有**（决定 2026-08-26：公开 `pollen-robotics/microduck`）。理由值得留着，因为它不显然：一个私有 repo 的 `releases/download/<tag>/<asset>` URL **带不带 token 都返回 404**（直接验证过）；而 `api.github.com/repos/<repo>/releases/assets/<id>` + `Accept: application/octet-stream` 带 token 返回 200。所以引擎经 API 端点解析每个 asset——这对**开发者的板子**（环境里有 `GITHUB_TOKEN`、`--ref` 装分支构建）和对一个公开 repo 都行，但对一台**没有 token、也不该有**的客户机器人不行：一个烘进镜像的 fleet 级凭据会泄露、且无法在不重刷的情况下轮换，正是签名密钥被分层去避免的同一个问题。

**公开不改任何代码**（API 路径对私有和公开都对，正因如此才那样写）。它改的是一个**预算**：未认证的 GitHub API 请求限每 IP 每小时 60 次，一台别人家里没 token 的机器人从"那个地址后面一切共享的匿名池"里花。`check_interval="6h"` 下一只鸭子远够不着，但一个房间二十只同 wifi 一起更新可能。这不是正确性问题（`http.rs` 已把 403/429 读成"稍后再来"），而是一个理由去偏好 `browser_download_url` 取字节（从 `objects.githubusercontent.com` 的对象下载不花那个池）、并保留 API 路径给私有 repo 和 dev 板子。

### 16.2 通道与晋升（§16.3）

实现为三个 GitHub Actions workflow 加一个 `cargo xtask` 发布器：

| | |
|---|---|
| `ci.yml` | fmt、clippy、tests，加 `board-test.sh`——**唯一证明二进制在 aarch64 Linux 上跑得起来的 job** |
| `release.yml` | 在一个 `daemon-staging-v*` tag 上：交叉构建、打包、签名、**用机器人自己的代码路径验证**、发一个 prerelease |
| `promote.yml` | 手动：重签一个 *stable* manifest、把验证过的 artifact 拷到 stable release、退役 staging |

发布器是个 Rust `xtask` 而非 shell 脚本，就为一个理由：**它复用 updater 用来验证的那套确切的 `minisign`/`tar`/`zstd`/`sha2` crate**。一个 shell 版会依赖分别安装的二进制，其行为可能从"机器人接受什么"漂移——那是差异最不该能藏身的地方。它也链接*完整*的 `minisign` crate（能签），而 daemon 只链接 `minisign-verify`（不能签）。

三个值得陈述的属性，每个都是被断言而非被假设：
- **晋升从不重构建。** stable manifest 携带 staging 的 `sha256`，晋升在检查那个 digest 后把 staging artifact 拷到 stable release，所以客户端收到的字节正是 canary 验证过的字节；一个测试断言 digest 未变。（manifest 过去指回 *staging* release 而非拷贝，理由是"一套字节无法发散而两套能"。它忽略的是机器人安装前验 `sha256`，所以一个发散的拷贝永远无法静默安装——而一个 artifact 活在一个像脚手架的 tag 下的 stable 通道，离坏掉只差一次清理。**它坏了**：删掉 `daemon-staging-v0.1.x` release 让 `daemon-v0.1.0/v0.1.1/v0.1.4` 正确签名却指向一个 404。stable release 现在自包含，staging 由 `promote.yml` 自己退役。）
- **artifact 可复现。** tar 里固定 mtime 意味着同样输入产同样归档，所以一次重构建能和发出的比较；一个测试断言同输入的两次打包 hash 相同。
- **`release.yml` 发布前验证。** 它经真引擎（`updaterd install --from`，一个 `LocalDir` source）安装 release 并断言二进制落地可执行。**若 updater 不能接受一个 release，没人能下载它。**

两个守卫存在因为两种错都容易：`package` 拒绝一个不匹配 `Cargo.toml` 的 `--version`（tag 了却没 bump），`promote` 拒绝一个不匹配 staging manifest 的版本。

**canary 用一个 flag、每命令拿一个候选**：`sudo robotctl update apply daemon --staging`。早先草稿说 canary"自动拉 staging"，那按写的无法实现：`newest_version` 跳过任何 GitHub 标记为 prerelease 的*和*任何带 semver prerelease 分量的——那个过滤恰是把客户机器人挡在候选构建外的东西，所以它留着，而 `--staging` 是它唯一的 opt-in（在 `staging_tag_prefix` 下的第二次扫描，允许 *GitHub* flag 但仍排除 semver prerelease，所以一个分支构建永不会被误认成候选）。`latest_manifest` 不动，所以 `auto_apply` 和周期检查继续解析 stable：**没有人和 root，什么都不会漂到一个候选上。** 第一次尝试是靠改 `tag_prefix` 把一块板子指向 staging，它对一个就在那里的候选报 `no releases … with tag prefix "daemon-staging-v"`——前缀说了去哪看，而 prerelease 过滤拒绝看。`--staging` 在通道落后于板子时拒绝（§6.3），`--staging --version X` 无守卫、是越过它的办法。绿了则 **promote**：把 `stable` 重指向*已验证的同一批字节*（重签 stable manifest 引用同一个 tarball + hash），无重构建、无重刷、无手动拷文件。坏 `stable` → 同样的动作反向（重指向 prior known-good manifest；`min_supported` 随后在修复落地时把机器人往前拉）。

### 16.3 为可测性设计 + 两个测试层（§16.1、§16.2）

**动机**：在上一台机器人上，release 靠手验证——手动回退到已知版本、手动 apply 新的、手动检查，太多人为错误空间。这里的设计目标：**让更新路径可脚本化、自动测试，好让发货给客户是一个*决定*、不是一套手动流程。** 关键框定：那大部分人为错误面（回退 → apply → 验证）活在更新*机制*里，那是纯软件、可在 CI 里**无硬件**测；只有真正依赖硬件的行为需要真板子。

**为可测性设计**：真代码路径对着测试产物（`updaterd` 从 config 读它的 source/manifest，所以测试把它指向一个本地目录或 `staging` 通道、驱动*确切的生产代码*，没有一个会从现实漂移的 mock updater）；可脚本化、幂等的原语（经 `robotctl`：`apply --version X`、`rollback`、`pin`/`unpin`、`model select`、`reset-to-golden`，全**非交互**——无提示、没有运维能答错的东西）；`--dry-run`（跑 fetch/verify/compat/space、在 swap 前停）；**故障注入 hook**（强制一个失败的健康探针、一个损坏的产物、一个非零 hook、或一个 swap 中途 kill——所以回滚是*被测的*、不是被假设的）。

**Tier 1 —— 机制测试（CI，无硬件，快）**，每个 PR 跑，仅这一层就移除大部分手动回退风险。自动覆盖：签名有效/无效/错钥 → 接受/拒绝；hash 不匹配 → 拒绝；`min_hw_rev`/`model_api` 不兼容 → 带清晰状态拒绝；一个更旧但真签名的 manifest 冒充 "latest" → 作为降级拒绝，而一个显式 `--version` 降级仍允许；`rollback` 从不落在刚失败的那个、也不落在日志记为回滚过的那个上；boot-counter trial 是每组件的（一个组件的转换绝不消耗另一个的预算）；**机器人专有状态跨更新*和*回滚保留**（标定、生成资产、学习状态、用户偏好全完好）；健康更新 → commit + prune；失败健康探针 → 自动回滚、机器人在*旧*版本上健康；hook 非零退出 → 回滚；**`kill -9` 中途 swap / 模拟掉电 → 重启后状态一致、绝无半 live release**；磁盘满 → 任何变更前中止；并发触发 → 单飞、第二个报 busy；**`robotd` 缺失/崩溃循环 → `btd` 和 `updaterd` 仍服务，update/rollback/reset-to-golden 全完成**（含一个"从一个刻意做坏的 release 恢复"测试——它演练那条最要紧的路径）；`robotd` IPC 挂起（socket 开、无应答）→ 超时而非卡死；版本矩阵（升级 A→B、跳 A→C、回滚 B→A、跨 `schema_version` bump 的迁移）。Tier 1 用一个 in-process `FakeRobot`（实现 `RobotClient`）驱动引擎——那是测引擎*决策*的对办法（一个 fake 可按需不健康、挂起或缺席，都不易真实上演），但它从不序列化任何东西，所以**根本看不到 wire**。

**Tier 1b —— 对一个真 `robotd` 进程**（`robotd/tests/updater_gate.rs`，仍 CI、仍无硬件）关闭那个缺口：它 spawn 真二进制、让真 `SocketRobotClient` 经一个真 unix socket 和它说话——引擎调的每个方法必须以它 parse 的形状回来，一次更新必须 gate 并 commit，而 `robotd --unhealthy` 必须回退 `current` 后面的*内容*、不只是符号链接。没有别的东西覆盖这个：protocol crate 自己的 round-trip 测试检测不到 skew（两侧共享 struct），`robotd` 里一个 `dispatch` 级单元测试便宜地抓一个改了的应答形状，但只有一个真进程演练 socket 本身。

**Tier 2 —— 设备上验收（canary 机器人）**：依赖硬件的行为（步态、电机、媒体）需要真板子。保留几台 lab/canary 机器人跟踪 `staging`、每个候选自动更新、跑一个脚本化验收测试（电机扫描、步态冒烟、媒体 init、健康自测）、报 pass/fail + 更新日志。可重复，因为 reset-to-golden 和 `apply --version` 可脚本化。

### 16.4 Provenance 与 bootstrap 状态（§16.4、§16.5）

`version.toml` 和更新日志记录每个 artifact 的 git SHA + hash，所以"一个客户在跑的确切版本"总在实验室可复现。这关闭了上次伤到的那个环：**人决定*是否*晋升；机器每次都同样地做回退/apply/验证。**

`daemon` artifact 仍发得比它最终会发的少：`bin/updaterd`、`bin/robotctl`、`bin/robotd`、`version.toml`，还没有 `mediad` 或 `btd`。**那不成循环**：§2.3 确立 `updaterd` 和 `btd` 装在 daemon artifact *内部*，所以"用 updater 更新 updater"是真实的最终流程——也是最冒险的那个，先演练它正是 §12 build 顺序的意义。`updater.example.toml` 里两个设置刻意惰性到有东西可 gate 之前，现在都 live 了：`on_apply` 从 `none` → `restart ["robotd","configd"]`；`health` 从 `none` → `socket`, 30 s（timeout 是个猜测，直到 M4 测一次真 boot）。`mediad` 需要在旧权威列表下有一个条目、现在不需要：**重启集从 release 发的单元派生**、跳过任何无 `[Install]` 段的，而 `mediad.service` 有一个——所以规则是单元文件、不是谁维护的一个名字。那个把示例 config 钉在惰性值上的单元测试现在断言相反：`on_apply` 必须重启 `robotd`、必须**不**重启 `updaterd` 或 `btd`（§2.1），`health` 必须是一个 socket 探针——一次退回 `probe="none"` 会静默禁用自动回滚、却看着像一个词的 diff，正是那种需要一个测试站在它前面的改动。**仍未测的**：`on_apply` 里的 `systemctl restart` 从没对真 systemd 跑过（dev 笔记本上没有，stub 它会测那个 stub）；健康门本身*确实*对一个真 `robotd` 进程经真 socket 测了，所以剩下未证的恰是重启那步和 30 s timeout，两者都落在 M4 的 Radxa 上。

---

## 17. 范围外与备选（§11、§12）

- **OS 更新刻意不在 OTA 范围内**（§11）：只开 Debian security pocket 的 `unattended-upgrades`；更大的变动 = 维修时重刷；最小化对基础 OS 的运行时依赖。
- **外设 / MCU 固件不在范围内**（§11.1）：子板和舵机固件在**产线**烧，不在现场。不是 OTA 组件、不为它做任何准备。（若哪天变了，它会是一个 `on_apply` 经电机总线烧录的组件——实质上更冒险，因为回滚意味着重烧而非一次符号链接交换。）
- **备选：apt / dpkg（自己的签名 repo）**（§12）：仍值得留在桌上，尤其对 Debian 上的 daemon 通道。**Pros**：签名（GPG）、原子安装、版本化、maintainer 脚本（`preinst`/`postinst`/`prerm`/`postrm`——我们想要的 post-install hook）全免费且久经考验；标准 Debian 工具；`apt` 处理依赖。**Cons（对我们的情形）**：没有开箱的健康门回滚、也没有干净的"回退到上一版本"（我们反正要在上面建）；把 ML 模型作为一个普通版本化 asset 发很别扭；不带手机触发/进度 IPC（仍需包一层）；**可移植性**——在一个未来非 Debian 机器人（Yocto 等）上不会存在，这与"跨机器人的通用工具"目标相悖。**何时 apt 合理**：若 fleet 可预见的未来是 Debian-only、且愿意放弃自动健康门回滚（或用 `apt install pkg=old` pin 驱动回滚），一个自己的签名 apt repo 对 daemon 二进制是一条合法更简单的路径（模型仍想要一条侧通道）。**推荐**：从自定义 Rust 引擎起步（为可移植性 + 回滚），若 Rust 引擎的维护成本哪天超过收益、且 fleet 保持 Debian-only，再重访 apt。

---

## 18. 已记录的决策

| | |
|---|---|
| 应用级更新，不做 A/B 镜像 | 只有 daemon + model 变，OS 静态；A/B 是过度工程 |
| 版本化目录 + 原子符号链接交换 | 简单、健壮、无需分区；一次 `rename(2)` 保证无半 live release |
| updater 是独立于它更新对象的单元 | 一个进程无法干净替换自己的二进制；须在 daemon 崩溃后存活以回滚 |
| updaterd/btd 排除出 in-flight 重启，5 s 后延迟重启 | 它们装在 daemon 产物内；"排除" ≠ "跳过"（§2.2 的 bug） |
| 排除集写代码里（`NEVER_RESTART`）非配置里 | 是"这两个是什么"的属性，不是运维该搞错的选择 |
| 重启集从 release 的单元派生、以 `[Install]` 段为键 | 让 postinstall 和引擎按构造一致；从 release 派生让旧 config 板子也重启新 daemon |
| 单飞锁用 OS 文件锁非 PID 文件 | 进程被 kill 时内核释放；陈旧 PID 会永久禁更新 |
| arm boot counter 在 swap 之前 + `fsync_parent` | swap 与健康检查之间崩溃仍可恢复；堵住"swap 落盘 pending 没落盘"的掉电窗口 |
| 健康门三态，degraded 照样 commit | 硬件故障回滚修不好；回退会在没说一声下换掉正被拿着的机器人的代码 |
| `rollback_target` 严格更旧 + 非 known_bad | 否则 `rollback` 会把刚失败的坏更新重装回去 |
| known_bad 只 gate 无人值守路径 | 显式 apply 仍重试（运维可能已修好原因）；否则坏 release 是无出口的 fleet 陷阱 |
| 降级守卫只 gate `Latest` | 挡攻击者能造成的、留运维能选择的（Exact/Ref/from_dir 豁免） |
| 三层回滚网（健康门 / boot counter / boot recovery net） | 各抓不同失败；第三层预设"updaterd 起不来"，活在进程和 release 之外 |
| rescue 读两个符号链接、什么都不 parse | 最可能救援的正是"updaterd 拒绝 updater.toml"的 release |
| boot check 用截止时间非 `OnFailure=` | `Restart=always` 的单元从不进 `failed`；截止时间反转问题、抓住永远重启的家伙 |
| golden 豁免于 prune 计数（非仅优先） | never-brick 保证不随版本累积而悄悄过期 |
| 每个模型是独立组件 | 各自版本线；reload+SIGHUP 让权重交换不丢电机控制 |
| `schema_version` 是迁移上下文非 gate | 判断 manifest 的引擎永远是上一个 release 的；gate 它会让每次 bump 无法投递 |
| 机器人状态在 install_dir 之外 | 一次 swap 或回滚无法摧毁它，也无法摧毁关于它的日志/transcript |
| hook 装在签名 tarball 内 | 没有未签名代码会跑；"fresh install 做的 hook 也做"（四次教训） |
| 发布器是 Rust `xtask` 非 shell | 复用 updater 验证用的确切 crate，行为不漂移 |
| 晋升从不重构建、拷贝已验证字节 | 客户端收到的正是 canary 验证过的；stable 自包含（404 教训） |
| 仓库公开 | 私有 repo 的 download URL 带不带 token 都 404；公开不改代码、只改一个 API 预算 |

---

## 19. 刻意推迟 / 开放问题（§17）

**已知、刻意开放的实现缺口**：
- **单个 model slot 内的竞争候选不支持。** 每个模型是一个组件、一条版本线（覆盖"walk/jump/stand 各自独立更新"），但*不*覆盖"两个不同 walk 模型装着、用户选一个"。那需要一个 `(name, version)` store key，波及约 14 个文件（`Store` 的 keyed 方法、`known_bad`、`PendingUpdate`、`Pins`、`golden`、`pinned`、以及 wire 类型——所以一次 `API_VERSION` bump），并把四件事变成 per-bundle 而非全局：prune 计数、回滚目标、"latest"、golden。推迟到确知需要；猜那四件正是被删的 `Layout::Library` 的由来（§4.3）。
- **没法发现可安装的模型。** App 能列*已装*的，但没有 `Source` 能答"什么可被装"——加一个模型今天要改 config。想要一个 `Source::list_*` 操作。
- **没有 recovery mode。** §8.2 的链是 `current → previous → golden`，三个都实现了（含越过一个缺失或 known-bad previous 的升级）。最后那个"仍能重新拉取的最小 recovery mode"不存在。
- **一个 `RobotClient` 服务每个组件。** `HealthCheck::Socket { path }` 被尊重为*跑哪个探针*，但 socket 路径本身来自 `main` 构造的东西。两个组件都探 `robotd` 时没问题；若某组件需要一个不同的 peer 就是个陷阱。想要一个每组件 client。

**仍开放**：手机投递 artifact 作为 fallback（给一台没有可用互联网的机器人；引擎侧几乎免费因为 `LocalDir` 已经真验证路径 apply 一个目录，成本全在那条必须是 wifi 或 USB 的链路上，是 v1 的备选而非 v1）；manifest 陈旧报告（§15.1 的 (3)，便宜、推荐）；config 归属（config 随 model bundle、随 daemon、还是成自己一条小通道？hook 无论如何处理迁移）；最小成功/失败 phone-home（每次更新一个 ping 能让我们早点抓到一个坏 release，是让 `min_supported` 实际可操作的东西）；行为/brain 层（若那个更高层的行为层哪天作为自己的 artifact 发，它成第三条通道、有自己的兼容性约束，其学习状态也妥妥是 §5.7 材料）。

**显式不做**：硬件变体矩阵（§5.6）、外设固件 OTA（§11.1）、分批灰度/遥测、增量更新。

**自初稿以来已决定**：分离托管（GitHub + HF Hub）；schema 里的 `min_supported` 地板；带具名 slot 的模型 *bundle* + `model_api` 兼容；多把受信任签名密钥；单一硬件目标；机器人专有状态保留（§5.7）；config 在一个文件里而非 systemd 单元里。
