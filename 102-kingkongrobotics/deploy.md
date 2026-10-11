# Jumper 本地训练环境搭建与部署测试

本文记录在本机为 [jumper](jumper/) 搭建训练环境、跑通「训练 → 回放 → 导出 → 打包」链路的步骤与实测结果。
官方完整说明见 [jumper/docs/AGENT_SETUP.md](jumper/docs/AGENT_SETUP.md) 与 [jumper/deploy/README.zh.md](jumper/deploy/README.zh.md)。

所有命令都在仓库根目录执行：

```bash
cd /home/creator/ws/102-kingkongrobotics/jumper
source .venv/bin/activate
```

## 进度总览

| 步骤 | 内容 | 状态 |
|---|---|---|
| 1 | 创建 `.venv`（Python 3.12） | ✅ 完成 |
| 2 | 安装 PyTorch（cu130，清华源） | ✅ 完成 |
| 3 | `pip install -e ".[dev]"` 安装 jumper | ✅ 完成 |
| 4 | 验证门 A–D | ✅ 门 C 已确认，其余按下文核对 |
| 5 | 冒烟训练 `jumper.flat` 5 轮 | ✅ 完成 |
| 6 | pytest 测试套件 | ⬜ 待执行 |
| 7 | 回放并录制 MP4 | ⬜ 待执行 |
| 8 | 导出部署用 ONNX + `layout.json` | ⬜ 待执行 |
| 9 | 正式训练 `jumper.tripod` 10000 轮 | ✅ 完成（约 4 h） |
| 10 | Rust 工具链 + 打包 bundle（`.app`） | ⬜ 可选 |
| 11 | 上板（RK3576） | ⬜ 需要实体机器人 |

## 本机环境（实测）

| 项目 | 值 |
|---|---|
| OS | Ubuntu 22.04，x86_64，24 核，93 GB 内存 |
| GPU | NVIDIA GeForce RTX 5090（32 GB，sm_120），驱动 580.178.04 |
| Python | 3.12.13（`/usr/bin/python3.12` 创建的 `jumper/.venv`） |
| torch / torchvision | 2.11.0+cu130 / 0.26.0 |
| mujoco / mujoco-warp | 3.11.0 / 3.11.0 |
| warp-lang | 1.18.0（CUDA Toolkit 13.4，Driver 13.0） |
| onnx / onnxruntime | 1.23.2 / 1.31.0 |
| Rust | rustc 1.96.0（缺 wasm32 target、wasm-bindgen-cli） |
| CycloneDDS / Docker | 未安装 |

---

## 1. 创建虚拟环境

官方要求使用仓库内 `.venv`，不要用 conda。

```bash
/usr/bin/python3.12 -m venv .venv
source .venv/bin/activate
python -c "import sys; print(sys.prefix, sys.version.split()[0])"
```

预期：`/home/creator/ws/102-kingkongrobotics/jumper/.venv 3.12.13`。若显示 miniconda 路径说明未激活，**后续 pip 会装错位置**。

## 2. 安装 PyTorch

阿里云 `pytorch-wheels` 镜像下载 cu128 wheel 只有约 100 kB/s（820 MB 需 2 小时以上），改用清华 PyPI 源的默认构建 cu130（驱动 ≥ 580 即可，jumper 要求 Blackwell 用 "cu128 or newer"）。

任选其一：

```bash
pip install torch==2.11.0 torchvision==0.26.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

```bash
bash /home/creator/ws/scripts/setup_py312_cu130_rtx5090_torch.sh --python .venv/bin/python
```

验证：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

预期：`2.11.0+cu130 True NVIDIA GeForce RTX 5090`。

## 3. 安装 jumper

`-e` 必须加：`rl/` 下的 `mjrl`、`mjlab`、`rsl_rl` 只能通过 editable 安装导入。

```bash
pip install -e ".[dev]" -i https://pypi.tuna.tsinghua.edu.cn/simple --default-timeout 120
pip uninstall -y mjlab rsl-rl-lib
```

第二条确保没有 PyPI 版 mjlab / rsl_rl 遮蔽仓库内副本，提示 "not installed" 属正常。

## 4. 验证门 A–D

```bash
python -c "import warp as wp; wp.init(); print(wp.get_devices())"
python -c "import mujoco; from mujoco import rollout; print('mujoco', mujoco.__version__)"
python -c "import mjlab, rsl_rl, os; print(os.path.dirname(mjlab.__file__)); print(os.path.dirname(rsl_rl.__file__))"
python scripts/train.py --list
```

| 门 | 预期 | 实测 |
|---|---|---|
| A | stdout 为 `['cpu', 'cuda:0']` | ✅ 训练日志显示 `cuda:0 : NVIDIA GeForce RTX 5090 (31 GiB, sm_120)` |
| B | `mujoco 3.11.x` | ✅ 3.11.0 |
| C | 两个路径都在 `jumper/rl/` 下 | ✅ `jumper/rl/mjlab`、`jumper/rl/rsl_rl` |
| D | 列出 18 个 `jumper.*` 任务 | 待核对 |

## 5. 冒烟训练（已完成）

```bash
python scripts/train.py --task jumper.flat --num_envs 256 --max-iterations 5 --headless
```

实测结果：

| 指标 | 值 |
|---|---|
| 后端 | `backend=warp device=cuda:0 num_envs=256` |
| 首次运行 | Warp 编译内核，单个模块最长约 8 s，缓存在 `~/.cache/warp/1.18.0/`，第二次运行会快很多 |
| 每轮耗时 | 约 0.35 s（采集 0.285 s + 学习 0.065 s） |
| 吞吐 | 约 17.6k steps/s |
| 第 4 轮 | Mean reward -4.12，Mean episode length 66.54，fell_over 0 |
| 产物 | `logs/jumper/jumper.flat/2026-10-10_23-29-02/`：`model_0.pt`、`model_4.pt`、`2026-10-10_23-29-02.onnx`、tensorboard 事件 |

以下输出是**正常现象**，无需处理：

- `num_envs=256 differs from this backend's default of 4096`：只是提醒 PPO 有效 batch 变了；正式训练用默认值即可。
- `Attach conflict ... timestep ... integrator`：mjlab 场景合并时的提示，`MjSpec.attach()` 不传播子模型 `<option>`。
- 前几轮 Mean reward 为负。

> 训练时自动导出的 `<时间戳>.onnx` **不能直接用于部署**：它缺少动作到关节的映射（22 个关节 vs 20 维动作），按下标配对会从第 5 个关节起错位。部署必须走第 8 步的 `scripts/export.py`。

## 6. 运行测试套件（下一步）

```bash
python -m pytest tests/ -q
```

约 30 s。未装 wasm target / CycloneDDS 时，`test_fsm_extension.py`、`test_deploy_sources.py` 等 Rust 相关测试会跳过或失败，属预期；其余应全部通过。

## 7. 回放并录制 MP4（下一步）

无显示器渲染需先设 `MUJOCO_GL=egl`。不带 `--checkpoint` 时自动取最新 checkpoint，这里写明路径以便复现：

```bash
MUJOCO_GL=egl python scripts/play.py --task jumper.flat --checkpoint logs/jumper/jumper.flat/2026-10-10_23-29-02/model_4.pt --headless --steps 500 --video out/flat_smoke.mp4
```

预期生成 `out/flat_smoke.mp4`（640×480，30 fps）。5 轮的策略基本不会走路，这一步只验证回放与录像链路。

有桌面时可去掉 `--headless`、`--video` 打开实时查看器：

```bash
python scripts/play.py --task jumper.flat --checkpoint logs/jumper/jumper.flat/2026-10-10_23-29-02/model_4.pt
```

## 8. 导出部署用策略（下一步）

```bash
python scripts/export.py --task jumper.flat --checkpoint logs/jumper/jumper.flat/2026-10-10_23-29-02/model_4.pt
```

默认输出到 `tasks/jumper/flat/out/<日期-时间>/`，包含：

| 文件 | 作用 |
|---|---|
| `actor.onnx` | `[1, obs_dim] → [1, act_dim]`，观测归一化已烘焙进图 |
| `layout.json` | 部署契约：关节顺序、动作变换、PD 增益、控制频率、观测布局 |
| `README.md` | 如何组装观测、施加动作 |
| `model_4.pt` | 导出所用 checkpoint 的副本 |

导出时会用 onnxruntime 对 ONNX 与 PyTorch 输出逐元素比对，日志中应看到误差很小的数值校验结果。

## 9. 正式训练一个可用策略（可选）

冒烟通过后，用默认 4096 环境训练一个真正能走的步态。以三足步态为例（RTX 5090 上官方实测 4096 环境约 0.82 s/轮）：

```bash
python scripts/train.py --task jumper.tripod --headless
```

查看曲线：

```bash
tensorboard --logdir logs/jumper/jumper.tripod
```

训练完成后对新的 checkpoint 重复第 7、8 步（把 `jumper.flat` 和路径换成 `jumper.tripod` 及其 `logs/jumper/jumper.tripod/<时间戳>/model_<N>.pt`）。

### 实测结果：`jumper.tripod`（2026-10-10_23-54-30）

默认 4096 环境、10000 轮，耗时约 4.0 小时，吞吐 52k–72k steps/s。每 50 轮存一个 checkpoint，最终为 `model_9999.pt`。

| 指标（第 0 → 2500 → 5000 → 7500 → 9999 轮） | 变化 |
|---|---|
| Mean reward | -0.85 → 95.9 → 99.7 → 102.3 → **104.8** |
| Mean episode length | 13.6 → 1000 → 1000 → 1000 → **1000**（跑满时长） |
| fell_over | 全程 0 |
| 课程等级 `Curriculum/command/level` | 0 → **3**（第 2500 轮前已到顶） |
| 线速度命令范围 `lin_range` | 0.25 → **0.50 m/s** |
| 角速度命令范围 `ang_range` | 0.30 → **0.75 rad/s** |
| 线速度跟踪误差 `error_vel_xy` | 0.142（2500 轮）→ **0.127** |
| 偏航跟踪误差 `error_vel_yaw` | 0.467（2500 轮）→ **0.387** |

结论：约 2500 轮已学会稳定行走并升到最高课程，之后主要是跟踪误差缓慢下降。

查看方式（命令范围不要超过 `lin_range` 0.5、`ang_range` 0.75）：

```bash
tensorboard --logdir logs/jumper/jumper.tripod --port 6006

# 录仿真视频（不需要显示器）
MUJOCO_GL=egl python scripts/play.py --task jumper.tripod --checkpoint logs/jumper/jumper.tripod/2026-10-10_23-54-30/model_9999.pt --headless --steps 1000 --command 0.4 0.0 0.0 --video out/tripod_forward.mp4

MUJOCO_GL=egl python scripts/play.py --task jumper.tripod --checkpoint logs/jumper/jumper.tripod/2026-10-10_23-54-30/model_9999.pt --headless --steps 1000 --command 0.0 0.0 0.6 --video out/tripod_turn.mp4

# 实时查看（需要桌面环境）
python scripts/play.py --task jumper.tripod --checkpoint logs/jumper/jumper.tripod/2026-10-10_23-54-30/model_9999.pt

# 导出部署包
python scripts/export.py --task jumper.tripod --checkpoint logs/jumper/jumper.tripod/2026-10-10_23-54-30/model_9999.pt

# 导出结果会放在 tasks/jumper/tripod/out/<日期-时间>/，里面有 actor.onnx、layout.json、README.md 和 model_9999.pt。之后打包成 .app，就是用这个目录。


#导出部署包，或者先录一段回放视频看看步态
python scripts/export.py --task jumper.tripod --checkpoint logs/jumper/jumper.tripod/2026-10-10_23-54-30/model_9999.pt

MUJOCO_GL=egl python scripts/play.py --task jumper.tripod --checkpoint logs/jumper/jumper.tripod/2026-10-10_23-54-30/model_9999.pt --headless --steps 1000 --command 0.4 0.0 0.0 --video out/tripod_forward.mp4

```

## 10. Rust 工具链与打包 bundle（可选）

bundle（目录 + `.app`）是板端、浏览器、`play --app` 共同加载的产物。本机还缺以下组件：

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.128 --locked
bash deploy/fsm/vendor/rknpu2/fetch.sh
```

验证门 E：

```bash
cargo check --locked --lib --manifest-path deploy/fsm/Cargo.toml --target wasm32-unknown-unknown --no-default-features --features web
cargo check --locked --lib --manifest-path deploy/fsm/Cargo.toml --no-default-features --features py
wasm-bindgen --version
```

本机没有 Docker，无法交叉编译板端二进制和转换 `.rknn`，只能先打一个不完整的 bundle 用于浏览器 / `play --app` 验证（未实测）：

```bash
python scripts/deploy.py --allow-incomplete --mode walk=tasks/jumper/tripod/out/<导出目录>
```

完整打包还需要：

- Docker：`deploy/fsm/docker-build.sh` 交叉编译板端 controller，`deploy/convert/docker-convert.sh` 把 ONNX 转成 RK3576 NPU 的 `.rknn`。
- CycloneDDS 11.0.1（需 sudo，命令见 [AGENT_SETUP.md 2.2 节](jumper/docs/AGENT_SETUP.md)）：本机 device 构建与验证门 F 需要。

## 11. 上板（需要实体机器人）

```bash
python3 .claude/skills/deploy/scripts/check_board.py --host <user>@<board> --bundle <export>
./runtime/board/controller --bundle . --check-reference
```

第一条是只读预检；第二条在板子上执行，不触碰电机总线。官方状态：RKNN 板端推理与 1 kHz 控制循环驱动电机总线**尚未在真机验证**。

---

## 下一步建议

1. 执行第 4 步的 `python scripts/train.py --list`，确认 18 个任务（门 D）。
2. 执行第 6 步 pytest，记录通过 / 跳过数量。
3. 执行第 7 步录 MP4、第 8 步导出，确认 `tasks/jumper/flat/out/<日期-时间>/` 下四个文件齐全。
4. 链路通了再做第 9 步正式训练 `jumper.tripod`，用它重新回放和导出。
5. 需要打包时再装 Rust 组件和 Docker（第 10 步）。




-- 介绍 jumper

# 从小鸭子到小螃蟹：我在本地训练出了会走路的「跳跳」

![跳跳](docs/media/official/jumper-hero-zh.png)

> 一只 22 个自由度的开源螃蟹机器人。一块 RTX 5090，4 小时，从"站都站不稳"练到"0 摔倒、0.5 m/s 稳定行走"。

## 一、从 Microduck 说起

之前我在本地研究过 Pollen Robotics 的 **Microduck**：一只约 25 cm 高、800 g 重的双足小鸭子，会走路、打滚、捡东西，摔倒了还能自己爬起来。它背后的技术路线很清晰，分成两个仓库：

- **`microduck_rl` 负责"练"**：在 MuJoCo Warp（mjlab）里用 PPO 训练策略，导出 ONNX；
- **`microduck` 负责"跑"**：Rust 写的一组守护进程跑在瑞芯微 RK3566 上，以 50 Hz 驱动 15 个舵机执行策略。

最近看到金钢科技开源的螃蟹机器人 **跳跳（Jumper）**，第一反应是：这是 Microduck 的"同门师弟"。同样是"GPU 仿真里练策略 → 导出 ONNX → Rust 运行时在瑞芯微芯片上跑"，只不过两条腿的鸭子变成了六条腿、带两只钳子的螃蟹，难度和玩法都上了一个台阶。

## 二、认识跳跳

官网口号是"**跳跳，让世界继续好玩**"。它能灵活跑动，也能完成高动态跳跃，官方的长期愿景是成为帮忙做家务的"家庭小精灵"。

### 能走、能跳、能跳舞、能抓

| 走路 | 跳跃 |
|---|---|
| ![走路](docs/media/official/walk.gif) | ![跳跃](docs/media/official/jump.gif) |
| **跳舞** | **抓取** |
| ![跳舞](docs/media/official/dance.gif) | ![抓取](docs/media/official/claw.gif) |

*以上四个动作都是仓库里对应任务训练出的策略在仿真中的表现。*

### 和 Microduck 放在一起看

| 参数 | 跳跳 Jumper | Microduck |
|---|---|---|
| 形态 | 类螃蟹六足，前腿带夹爪 | 双足小鸭 |
| 尺寸 / 重量 | 400×400×200 mm / 1.8 kg | 约 25 cm 高 / 约 800 g |
| 舵机 | 22 个触觉舵机 | 15 个舵机 |
| 主控 | 瑞芯微 RK3576（6 TOPS NPU） | 瑞芯微 RK3566 |
| 传感器 | 6 轴 IMU、54×42 dToF、500 万像素广角摄像头 | IMU、ToF、摄像头 |
| 能力 | 移动 ≥ 0.5 m/s、跳高 ≥ 400 mm、抓取 1 kg、续航约 2 小时 | 行走、打滚、轮滑、捡东西、自主站起 |

22 个自由度的分配：两条前腿各 5 个关节，末端是钳子；中腿、后腿各 3 个关节。走路类策略控制其中 20 个（两只钳子不参与），跳舞类策略 22 个全上。

### 不用装环境，先在浏览器里玩

官方提供了一个**网页版交互模拟器**：点击地面，跳跳会绕开障碍走过去；也可以用键盘遥控，或点上方按钮触发迷宫舞、鞠躬、挥手、递爪、敬礼、跳跃、巴西舞、螃蟹舞、梦之翼等动作。

![网页版交互模拟器](docs/media/official/simulator.png)

*网页模拟器：上方是动作按钮，下方三栏分别切换外观、场景、APP。*

这个模拟器跑的就是仓库里那份 Rust 控制器，编译成 WebAssembly 放进了网页。下方的**外观、场景、APP** 三栏，正好对应跳跳主打的三个关键词：好看、好玩、有用。

## 三、好看、好玩、有用：一句话交给 AI

跳跳最有意思的地方，是把机器人开发拆成了三个入口，每一个都可以**用一句话交给 AI**：

| 入口 | 做什么 | 在哪个仓库 |
|---|---|---|
| 好看 · 换外观 | 设计外壳（`.skin`），渲染校验能不能装上 | `jumper-design` |
| 好玩 · 训动作 | 走路、跳跃、跳舞、打招呼等强化学习任务 | `jumper` |
| 有用 · 变智能 | 抓取、巡检、分拣等应用玩法 | 两者结合 |

比如对 AI 说："**理解 KingKongRobotics/jumper，给跳跳换上暖沙游侠外观**"。AI 会先读仓库里的说明文档和 `.claude/skills`（环境搭建、新建任务、遥控映射、部署打包等技能），再按既定流程完成外观校验和渲染。

![暖沙游侠外观](docs/media/official/design-warm-sand.png)

*官方示例中的「暖沙游侠」外观。*

同一副螃蟹骨架，换个壳就是另一个角色。社区里已经出现了史努比、皮卡蟹、瓦力、足球运动员、霓虹螃蟹、蝎子等不少脑洞作品：

![社区外观作品](docs/media/official/1-image.png)

*社区里网友创作的跳跳外观。*

配合这次开源，官方还办了**螃蟹机器人开源动作大赛**（2026-09-30 至 2026-11-30）：改外观、编动作、做智能，发 15 秒以上的作品视频参赛，最终选出 100 个优质开源作品，**每个送一台跳跳**。

## 四、技术拆解

### 1. 训练：mjlab + rsl_rl，GPU / CPU 双后端

- **物理仿真**：MuJoCo 3.11。主力后端是 **MuJoCo Warp**，在 GPU 上一次并行几千个环境；另一个后端是原生 MuJoCo，在 CPU 上多线程 rollout。两个后端共用同一份任务代码，切换时只替换底层的 `Simulation` 类。
- **训练框架**：mjlab 1.6.0 + rsl_rl 5.4.2（PPO）。两个库以源码形式内置在仓库的 `rl/` 目录里，项目自己的改动统一标注 `[mjrl]`，方便和上游对比。这一点和 Microduck 一样，都站在 mjlab 的肩膀上。
- **任务即目录**：`tasks/jumper/tripod/` 就是任务 `jumper.tripod`，主要由 `env_cfg.py`、`rl_cfg.py` 组成。目前共 18 个任务：
  - 走路：`flat`、`tripod`（三角步态，最快）、`tetrapod`、`ripple`（最稳）、`posture`、`five_foot`（收起左前腿当夹爪用）；
  - 跳跃：`jump`、`ref_free_jump`；
  - 模仿：5 个 dance、4 个 gesture；
  - 特技：`swing`（站上秋千把自己荡起来）。
- **严格的对照实验**：`flat`、`tripod`、`ripple`、`tetrapod` 四个走路任务之间只允许步态和速度上限不同，其余参数必须一致，由测试强制检查。哪种步态更好，可以直接拿数据说话。

### 2. 部署：一份 Rust 状态机，三个运行端

```
model_*.pt ──export.py──▶ actor.onnx + layout.json ──convert──▶ actor.rknn ──deploy.py──▶ bundle（.app）
```

- **`layout.json` 是唯一的参数来源**。`export.py` 导出 ONNX 时一并生成它，记录观测和动作的布局、关节名、增益和缩放。下游所有数字都只从这里读，关节按名字对应，不在别处重复声明，从根上避免"仿真里好好的，上机关节顺序错了"。
- **NPU 推理**：ONNX 再转换成 RK3576 NPU 能跑的 `.rknn`。
- **同一个 Rust crate，三种形态**：控制器（`deploy/fsm`）分别编译成板端二进制、浏览器 wasm（也就是前面的网页模拟器）和 `play --app` 的仿真扩展。仿真里看到的和真机上跑的，是同一套状态机逻辑。

Microduck 用的也是 Rust 运行时，但它是"多个守护进程 + JSON-RPC"的完整机器人系统；跳跳目前更聚焦于"一个状态机控制器 + 多模式策略包"。

### 3. 设计：外观和场景独立成仓

外观（`.skin`）和场景（`.map`）放在独立仓库 `jumper-design` 里，用 `shellflow.py` 做装配校验和多视角渲染；训练仓库只管动作，两边职责分明。目前的边界是：外观只用于展示，场景只能用于回放，还不能拿来训练。

## 五、我在本地跑了一遍

**环境**：Ubuntu 22.04 + RTX 5090（32 GB）+ Python 3.12 + PyTorch 2.11.0（cu130，清华源安装）+ MuJoCo / MuJoCo Warp 3.11。

### 第一步：冒烟测试

256 个环境跑 5 轮，单轮约 0.35 秒，约 1.76 万 steps/s，确认安装和训练链路都通了。

```bash
python scripts/train.py --task jumper.flat --num_envs 256 --max-iterations 5 --headless
```

### 第二步：正式训练 `jumper.tripod`

默认 4096 个并行环境、10000 轮，5090 跑了约 **4 小时**，吞吐 5.2 万到 7.2 万 steps/s。

```bash
python scripts/train.py --task jumper.tripod --headless
```

不加 `--headless` 时会打开 MuJoCo 窗口，可以直接看到一大群跳跳在同一块地板上同时练走路，这就是 GPU 并行仿真：

<video src="docs/media/official/train.mp4" controls width="100%"></video>

*训练画面：同一块地板上并行着一大群跳跳，每一只都是一个独立的仿真环境。*

训练曲线里的关键指标：

| 指标 | 开始 | 第 2500 轮 | 第 10000 轮 |
|---|---|---|---|
| 平均奖励 | -0.85 | 95.9 | **104.8** |
| 平均回合长度 | 13.6 | 1000 | **1000**（每回合都撑满时长） |
| 摔倒次数 | 0 | 0 | **0** |
| 课程等级 | 0 | 3 | **3**（最高级） |
| 速度指令范围 | 0.25 m/s / 0.3 rad/s | 0.5 / 0.75 | **0.5 m/s / 0.75 rad/s** |
| 线速度 / 转向跟踪误差 | — | 0.142 / 0.467 | **0.127 / 0.387** |

读这张表：一开始回合长度只有 13.6 步，几乎刚开局就结束；**到第 2500 轮左右，跳跳已经能每回合都走满时长**，课程升到最高级，速度范围扩到 0.5 m/s，正好对上官网"最高移动速度 ≥ 0.5 m/s"的参数。后面 7500 轮主要在打磨速度跟踪精度。

### 第三步：回放训练好的策略

用 `scripts/play.py` 加载最终的 `model_9999.pt`：

```bash
python scripts/play.py --task jumper.tripod --checkpoint logs/jumper/jumper.tripod/2026-10-10_23-54-30/model_9999.pt
```

<video src="docs/media/official/play.mp4" controls width="100%"></video>

*回放画面：跳跳用三角步态行走，右侧面板实时显示 22 个关节（`LF/RF_J0–J4`、`LM/RM/LR/RR_J0–J2`）的控制量。*

从训练画面里的"一大群"，到回放时的"这一只"，同一个策略从并行仿真走到了单机可视化，下一站就是导出部署包、上真机。

### 几点体会

1. **上手门槛比想象中低**。按 `docs/AGENT_SETUP.md` 一步步装即可。最大的坑是 PyTorch 下载慢（换清华源解决），以及 5090 需要 cu128 及以上的 PyTorch。
2. **5090 跑 4096 个环境很轻松**。一晚上就能练出一个能稳定行走的六足策略。
3. **checkpoint 很占空间**。默认每 50 轮存一个，一次训练约 2 GB。我只保留了第 2500、5000、7500、9999 轮，压到 73 MB，TensorBoard 曲线不受影响。
4. **还没做的**：导出部署包、Rust/wasm 打包（本机缺 wasm-bindgen 和 Docker）、上真机。官方也说明，RKNN 板端推理和 1 kHz 电机总线控制循环**尚未在真机上完成验证**。所以现阶段最适合的玩法，是先在仿真里把动作和外观做出来。

## 六、小结

如果说 Microduck 证明了"小型机器人 + mjlab 强化学习 + Rust 运行时"这条路走得通，那么跳跳就是在这条路上又往前迈了一步：

- **更复杂的本体**：22 个自由度、六足加夹爪，能走、能跳、能抓、能跳舞；
- **更工程化的训练**：双物理后端、任务即目录、对照实验由测试强制保证；
- **更统一的部署**：一份 Rust 状态机，板端、网页、仿真三处共用；
- **更低的创作门槛**：外观、动作、场景都能一句话交给 AI，还有一场送 100 台机器人的开源大赛。

下一步，我打算把 tripod 策略导出成部署包，再试试一句话给跳跳换上暖沙游侠外观。

> 项目地址：<https://github.com/KingKongRobotics/jumper> · 官网：<https://kingkong.tech/jumper> · 大赛：<https://beunlimited.me/zh/events/crab-robot-challenge-2026>

--