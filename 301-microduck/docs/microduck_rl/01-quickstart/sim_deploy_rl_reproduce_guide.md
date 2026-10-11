# Microduck 仿真部署与 RL 训练复现指南（5090 机 · py3.12 · uv · mjlab / MuJoCo Warp）

> 版本 v1.0 · 2026-09-02 · 目标硬件：RTX 5090 32GB（sm_120），Ubuntu 22.04，驱动 580
>
> 本文指导在**没有真机**的情况下，在本地完成 Microduck 机器鸭的 MuJoCo 仿真查看、官方策略推理回放，以及基于 PPO 的本地仿真训练与 ONNX 导出。所有命令均在 `301-microduck/` 目录下操作，**请按顺序执行并自行验证**。

> **解说文案**：最近机器人圈最火的，不是动辄几十万的人形机器人，而是一只399美元的小鸭子。8月27日，Hugging Face旗下Pollen Robotics发布开源双足机器人 Microduck：开订24小时订单额突破260万美元、卖出5000多台，高峰期平均4秒一台；4天狂揽10500台订单、销售额超400万美元，新订单排期已拉长到4~6个月。它凭什么这么风靡？一是便宜，399美元把具身智能的门槛从几十万直接打到两千多块；二是开源，可以用RL教它新把戏，再把训练好的ONNX一键部署回真机；三是够萌，25厘米、不到800克，走路一摇一摆，摔倒了能自己爬起来，还会穿轮滑鞋。我在本地MuJoCo仿真里跑了官方策略，还是相当好玩的，憨态可掬，两个脚蹼倒腾得有模有样。一句话总结：它既是极客手里能跑 RL、折腾新把戏的严肃开发平台，也是能走进家庭、逗娃解闷的萌系新玩具——两种身份，它都接得住。


---

## 〇、先搞清楚两个仓库的分工

Microduck 项目由两个仓库组成，职责完全不同：

| 仓库 | 角色 | 语言 | 内容 |
|------|------|------|------|
| `microduck`（已在 `301-microduck/microduck`） | **鸭子的大脑**：机载运行时 | Rust | 50Hz 控制环、15 个舵机驱动、蓝牙/游戏手柄/相机守护进程；`policies/` 里自带 **10 个官方训好的 ONNX 策略** |
| `microduck_rl`（需拉取，见步骤 1） | **仿真 + 训练** | Python | mjlab（MuJoCo Warp GPU 并行仿真）+ PPO（rsl_rl）、完整 MJCF 模型与 mesh、域随机化、BAM 舵机物理、ONNX 导出、键盘遥控推理脚本 |

**数据流闭环**：

```text
microduck_rl 训练（MuJoCo Warp + PPO, 50Hz）
      │  scripts/export.py（观测归一化器会烤进 ONNX 图）
      ▼
  *.onnx  （obs[1,61] → actions[1,14]）
      │  仿真验证：scripts/infer_policy.py（CPU MuJoCo + 键盘遥控）
      │  真机部署：microduck 仓库 policies/ 或 deploy/robotd.toml 的 [policy] 绝对路径
      ▼
  robotd 加载运行
```

注意：`microduck` 仓库里的 `kinematics/assets/alpha/robot_walk.xml` 只是 Rust 运动学单元测试的夹具，**没有 mesh、没有地面、不能用来观看**。好看的完整模型（mesh + 场景 + 关键帧）只存在于 `microduck_rl`。

---

## 一、环境要求

| 项 | 要求 | 本机情况 |
|----|------|----------|
| GPU | CUDA 显卡，训练走 MuJoCo Warp GPU 并行 | RTX 5090 32GB ✅ |
| 驱动 | 较新 NVIDIA 驱动（warp 1.12 自带 CUDA 12.9 runtime，需驱动兼容） | 580.173.02 ✅ |
| Python | **3.12.x**（`>=3.12,<3.13`，bam 依赖不支持 3.13） | `/usr/bin/python3.12` ✅ |
| 包管理 | **uv**（microduck_rl 官方方式，按 uv.lock 复现） | 需安装（步骤 3） |
| 磁盘 | ~10GB（uv 虚拟环境含 CUDA 版 torch/warp 约 2-3GB） | 充足 ✅ |
| 网络 | GitHub（clone 仓库 + bam 是 git 依赖）、PyPI（可走国内镜像） | 已配 SSH insteadOf + aliyun pip 镜像 |

microduck_rl 的关键锁定依赖（来自其 pyproject.toml）：`mjlab==1.3.0`、`warp-lang==1.12.0`、`torch==2.9.1`、`onnxruntime>=1.24.4`、`better-actuator-models`（BAM 舵机模型，git 依赖 `Rhoban/bam` 的 `mjlab_frictionloss` 分支）。

---

## 二、步骤 1：拉取 microduck_rl（作为 submodule）

按本项目硬约束，第三方仓库一律以 **git submodule + HTTPS URL** 管理。

```bash
cd /home/creator/ws

# 若之前尝试过、残留了空目录，先清理（目录非空则不要删）
rmdir 301-microduck/microduck_rl 2>/dev/null || true

git submodule add https://github.com/pollen-robotics/microduck_rl.git 301-microduck/microduck_rl
cd 301-microduck/microduck_rl

# microduck_rl 的文档/Quickstart 基于 develop 分支，确认切过去
git checkout develop

# 验证
ls src/mjlab_microduck/robot/microduck/   # 应能看到 robot_walk.xml、scene*.xml、mesh 等
```

随后在根仓库提交 submodule 指针：

```bash
cd /home/creator/ws
git add .gitmodules 301-microduck/microduck_rl
git commit -m "chore: add microduck_rl as submodule"
```

> 别人 clone 本仓库后，用 `git submodule update --init --recursive 301-microduck/microduck_rl` 拉取。

---

## 三、步骤 2：GPU 环境说明（关于 setup_py312 脚本的重要提醒）

`scripts/setup_py312_cu128_rtx5090_torch.sh` **文件名带 py312，但脚本内容实际是 cp310 的**，直接在 Python 3.12 下运行必然报错：

- 第 8 行：`WHEEL_DIR="$HOME/Downloads/py310-cu128-torch"`（指向 py310 轮子目录）
- 第 15-18 行：四个轮子文件名全部是 `cp310`
- 第 75-77 行：`if [ "$PY_VERSION" != "3.10" ]; then fail ...`（非 3.10 直接退出）

而且 microduck_rl **锁定的是 torch 2.9.1**（不是脚本里的 2.11.0），由 uv 按 `uv.lock` 自动安装：在 x86_64 Linux 上，PyPI 的 torch 2.9.1 wheel 本身就捆绑了 `nvidia-*-cu12` CUDA 库，warp-lang 1.12.0 也自带 CUDA 12.9 runtime，5090（sm_120 / Blackwell）在 CUDA 12.9 + 驱动 580 下受支持。

**因此推荐做法：microduck_rl 环境完全交给 `uv sync`（步骤 3），不要把 setup 脚本的 torch 装进它的虚拟环境**——硬装 2.11.0 会破坏 uv.lock 的一致性，下次 `uv sync` 还会被改回 2.9.1。

setup 脚本仍有两个可复用价值，按需选择：

1. **驱动/GPU 基线自检**：先确认显卡和驱动正常（与 Python 版本无关）：

   ```bash
   nvidia-smi   # 应看到 RTX 5090、驱动 580、CUDA Version 13.0
   ```

2. **若想在独立的 Python 3.12 环境里用本地 cu128 轮子验证 torch**（与 microduck_rl 无关，纯 GPU 冒烟），需先把脚本复制一份并改三处：

   ```bash
   cp /home/creator/ws/scripts/setup_py312_cu128_rtx5090_torch.sh \
      /home/creator/ws/301-microduck/scripts/setup_py312_cu128_local.sh
   ```

   修改点（共 3 处）：
   - `WHEEL_DIR="$HOME/Downloads/py310-cu128-torch"` → `"$HOME/Downloads/py312-cu128-torch"`
   - 四个 `cp310` 轮子文件名 → 对应的 `cp312` 文件名
   - `if [ "$PY_VERSION" != "3.10" ]` → `!= "3.12"`，报错文案同步改

   注意本机 `~/Downloads/py312-cu128-torch/` 目录目前是**空的**，`~/Downloads/` 根目录下只有一个 `torch-2.11.0+cu128-cp312-*.whl`，torchvision/torchaudio/triton 的 cp312 轮子需自行下载补齐（脚本启动时会逐个检查缺失并报错退出）。**microduck_rl 本身用不到 torchvision/torchaudio**，所以这条路径对复现训练不是必需的。

---

## 四、步骤 3：安装 uv 并同步依赖

```bash
# 0) 退出 conda base，避免 base 环境与 venv 混淆
conda deactivate

# 1) 在 301-microduck 顶层自建项目虚拟环境并激活（系统自带 /usr/bin/python3.12）
cd /home/creator/ws/301-microduck
python3.12 -m venv .venv
source .venv/bin/activate

# 2) 在 .venv 内安装 uv（本机 pip 已配 aliyun 镜像，实测装到 0.12.9）
pip install uv
uv --version    # 验证

# 3) 环境变量（每次新开 shell 都需要；建议追加到 ~/.bashrc）
#    uv 默认会把环境建到 microduck_rl/.venv（子模块内部）；用 UV_PROJECT_ENVIRONMENT
#    把 uv sync/run 的目标指回步骤 1) 自建的 301-microduck/.venv，全程只用这一套环境，
#    保持第三方 submodule 目录纯净（同 101-pi-serial-models 把 .venv312 放顶层的惯例）
export UV_PROJECT_ENVIRONMENT="$HOME/ws/301-microduck/.venv"
export UV_HTTP_TIMEOUT=600
#    注意：UV_INDEX_URL **不要常驻 export**！uv 会把"索引配置变化"视为 lockfile 过期，
#    每条 uv 命令（包括 uv run）都会自动重新解析，把 uv.lock 的源 URL 全部改写成镜像，
#    导致 submodule 变脏（M uv.lock）、官方测试 test_x86_64_resolution_stays_on_pypi 失败。
#    仅在需要下载新包时按命令临时携带，用完恢复 lock：
#    UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ uv sync
#    git -C /home/creator/ws/301-microduck/microduck_rl checkout -- uv.lock

# 4) 按 uv.lock 把全部依赖同步进 301-microduck/.venv（Python 3.12）
#    首次会下载约 2-3GB（torch 2.9.1 CUDA 版、warp、mjlab、bam 等），耐心等
cd /home/creator/ws/301-microduck/microduck_rl
uv sync
```

说明与排障：

- **bam 是 git 依赖**（`https://github.com/Rhoban/bam.git`），uv 会直接 clone；本机已配置 `url.git@github.com:.insteadOf https://github.com/`，有 SSH key 即可走 SSH。若 GitHub 不通，需自行配置代理。
- 虚拟环境由第 1) 步自建在 `301-microduck/.venv`（项目顶层），`UV_PROJECT_ENVIRONMENT` 把 `uv sync`/`uv run` 的目标指到它，因此**不会**在 `microduck_rl` 子模块内另建环境——第三方 submodule 目录保持纯净，与 `101-pi-serial-models` 把 `.venv312` 放在顶层而非 `repos/lerobot-main` 内是同一惯例。也**不要**复用 `101-pi-serial-models/.venv312`（那里是 torch 2.11.0+cu130，版本不符）。
- 后续所有 `uv run`（查看、推理、训练、导出）都必须在已 `export UV_PROJECT_ENVIRONMENT` 的 shell 里执行，否则 uv 会在 `microduck_rl/.venv` 重建一套环境并重复下载 2-3GB。建议把第 3) 步的两个 export（`UV_PROJECT_ENVIRONMENT`、`UV_HTTP_TIMEOUT`）追加到 `~/.bashrc`；**`UV_INDEX_URL` 不要追加**（原因见上一条）。
- `uv sync` 默认"精确同步"，会移除 lockfile 之外的包（刚 pip 装进去的 uv、pip 本身都可能被移除）。若 sync 后 `uv` 命令消失，在激活的 `.venv` 里执行 `python3.12 -m ensurepip --upgrade && pip install uv` 恢复；之后往环境补装包一律用 `uv pip install <包名>`。
- 若 uv 警告找不到 Python 3.12：`uv python install 3.12`（uv 可自行下载托管版 Python）。

安装完成后做三项验证：

```bash
cd /home/creator/ws/301-microduck/microduck_rl

# ① torch 能看到 GPU（关键！若 False 先解决驱动/容器问题再往下走）
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 期望输出类似：2.9.1+cu12... True NVIDIA GeForce RTX 5090

# ② 任务注册表能列出全部 Microduck 任务
uv run list-envs | grep -i microduck

# ③ 跑官方 CPU 测试（配置不变量、关节索引、奖励符号、NaN 防护）
#    注意：pytest 不在项目依赖里，先补装；本机 ~/.bashrc source 了 ROS2 Humble，
#    其自带旧版 launch_testing pytest 插件与 pytest 9 不兼容，须临时剥离 PYTHONPATH
uv pip install pytest
unset UV_INDEX_URL   # 曾临时 export 过镜像变量的话先去掉（否则 uv 运行前会重写 uv.lock，见 FAQ）
env -u PYTHONPATH uv run pytest tests/ -q
#    实测预期：196 passed, 1 skipped。若 test_x86_64_resolution_stays_on_pypi 失败，
#    说明 shell 里还 export 着 UV_INDEX_URL，恢复 lock 后重跑，见 FAQ
```

---

## 五、步骤 4：零训练先看鸭子

### 5.1 静态查看完整模型（MuJoCo Viewer）

microduck_rl 的 MJCF 模型带 mesh、地面和 STAND/SIT/FOLD 关键帧：

```bash
cd /home/creator/ws/301-microduck/microduck_rl

# 先看有哪些场景文件
ls src/mjlab_microduck/robot/microduck/scene*.xml

# 用 MuJoCo 交互查看器打开（本机已实测可正常启动画面）
uv run python -m mujoco.viewer --mjcf=src/mjlab_microduck/robot/microduck/scene.xml

# 其他场景按需查看（把 scene.xml 换成对应文件）：
#   scene_backlash.xml        齿隙版基础场景
#   scene_walk.xml            行走训练场景（scene_walk_backlash.xml 为齿隙版）
#   scene_ball.xml            踢球任务（含小球）
#   scene_rollers.xml         装滑轮版
#   scene_allcollisions.xml   全碰撞体自检场景
```

查看器里可拖动关节、按空格暂停、在 Control 面板直接给 14 个舵机发指令。需要本机图形桌面；纯 SSH 无显示环境请跳过本步，改用 5.2 的 `--record` 录视频后回看。

### 5.2 键盘遥控官方 ONNX 策略（CPU MuJoCo 推理）

`microduck` 仓库自带的 10 个官方 ONNX 可以直接在 microduck_rl 的 CPU 仿真里回放——这是验证整套 sim2real 契约最省事的方式：

```bash
cd /home/creator/ws/301-microduck/microduck_rl

# 单策略：走路（--new-cmd-obs 必加：官方 ONNX 为 61 维新观测契约，缺失会报 Got 51 Expected 61）
uv run scripts/infer_policy.py --walking ../microduck/policies/alpha_walking.onnx --new-cmd-obs

# 多策略热切换彩排（walk/stand/sitstand/roulade 运行时可随时接管）
uv run scripts/infer_policy.py \
  --walking  ../microduck/policies/alpha_walking.onnx \
  --standing ../microduck/policies/alpha_stand.onnx \
  --sitstand ../microduck/policies/alpha_sitstand.onnx \
  --roulade  ../microduck/policies/roulade.onnx \
  --new-cmd-obs
```

按键说明（以脚本实测 help 为准，**在运行脚本的终端输入**，viewer 窗口不捕获按键）：

- 默认速度模式：`↑/↓` 前进加速/减速倒退、`←/→` 横移、`A`/`E` 左/右转向、`SPACE` 停止、`T` 暂停/恢复推理、`P` 随机推搡（抗扰测试）、`Q` 退出；
- `B` 切换身体姿态模式（`↑/↓` Δz、`←/→` Δpitch、`A`/`E` Δroll，`SPACE` 复位）、`H` 切换头部模式（neck/head 的 pitch/yaw/roll）；
- 彩排按键需加载对应策略才生效：`G` 地面捡物（`--ground-pick`）、`Y` 坐/站（`--sit` 或 `--sitstand`）、`R` 前滚翻（`--roulade`）、`K`/`L` 左/右踢球（`--kick-left`/`--kick-right`）。

常用可选参数：

| 参数 | 作用 |
|------|------|
| `--record` | 录制回放视频（无图形界面时用） |
| `--save-csv` | 保存观测/动作 CSV，用于 sim2real 数据对比 |
| `--debug` | 打印调试信息 |

可用的官方 ONNX（都在 `301-microduck/microduck/policies/`）：`alpha_walking`、`alpha_stand`、`alpha_sitstand`、`alpha_ground_pick`、`ball_kick_left`、`ball_kick_right`、`roller`、`roller_crouch`、`roulade`。输入统一为 61 维观测（48 维本体感知 + 13 维指令 `[twist(3), head_pose(4), body_pose(6)]`），输出 14 维关节动作。

---

## 六、步骤 5：本地仿真训练（PPO）

### 6.1 先配置 wandb

训练日志由 mjlab/rsl_rl 走 wandb（与 LeRobot 一样没有 tensorboard）。**必须在启动 `train` 之前设置好**，否则进程会卡在交互式三选一提示（Create account / Use existing / Don't visualize）等待输入：

```bash
export WANDB_MODE=offline     # 离线模式，run 存本地 wandb/ 目录，事后可 wandb sync 上传（推荐）
# 或：uv run --with wandb wandb login   # 有 API key 时在线记录
```

> 若已经卡在提示处：输入 `3` 回车等价于本次禁用 wandb，训练会继续，checkpoint 照常保存到本地 `logs/rsl_rl/...`，只是这次 run 不会写入 `wandb/`（之后 `play`/`export.py` 无法用 wandb-run-path 引用它）。

### 6.2 先 smoke 再正式训练

```bash
cd /home/creator/ws/301-microduck/microduck_rl

# Smoke：512 并行环境跑通管线（几分钟内看到 reward 曲线开始动即可 Ctrl-C）
uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 512

# 正式：走路主任务，4096 并行环境，官方参考耗时约 1-2 小时出可用步态
uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096
```

显存不够就降 `--env.scene.num-envs`（2048 / 1024），5090 32GB 跑 4096 官方配置无压力。

**首次启动的正常现象（本机 5090 实测）**：

- 开头有十几行 `Module ... load on device 'cuda:0' took xxx ms (compiled)`：这是 MuJoCo Warp 首次为 sm_120 JIT 编译 CUDA kernel，约 1-2 分钟，编译缓存在 `~/.cache/warp/1.12.0`，**之后启动不再编译**；
- `[mdp] Patches 1-2 active: NaN-safe reward/advantage`、`Patch 4 active: ONNX export filters passive_* joints`：上游打在 rsl_rl 上的补丁，正常；
- tyro 的 `UserWarning`（list/tuple 类型注解）和 mjlab 的 actuator/site 匹配提示均无害，可忽略；
- 环境配置表应打印：512 envs、physics step 0.005s、env step 0.02s（即 50Hz 控制）、**actor obs 61 维**（base_ang_vel 3 + projected_gravity 3 + joint_pos 14 + joint_vel 14 + actions 14 + command 3 + head_command 4 + body_command 6）、**critic 76 维**（多特权信息：base_lin_vel、foot_height/air_time/contact/contact_forces）、16 个 reward term、7 个 curriculum；
- 策略网络为 MLP：actor `61→512→256→128→14`（ELU），critic `76→512→256→128→1`；
- 之后开始刷 `Learning iteration N/...`，smoke 看到 mean_reward 不为负且 episode length 正常、无 NaN 即算跑通。

正式训练建议挂后台、日志落盘（train 的本地日志目录形如 `logs/rsl_rl/velocity/<时间戳>_velocity/`，checkpoint 也保存在这里）：

```bash
mkdir -p logs
WANDB_MODE=offline nohup uv run train Mjlab-Velocity-Flat-MicroDuck \
  --env.scene.num-envs 4096 > logs/train_velocity_$(date +%Y%m%d_%H%M%S).log 2>&1 &
tail -f logs/train_velocity_*.log   # 观察，Ctrl-C 只退出 tail 不影响训练
```

断点续训：

```bash
uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096 \
  --agent.run-name resume --agent.load-checkpoint model_29999.pt --agent.resume True
```

### 6.3 可训练任务清单（13 个）

`uv run list-envs` 可看实时注册表。主任务都有 Flat/Rough（平地/崎岖地形）两种变体：

| 任务 id | 说明 |
|--------|------|
| `Mjlab-Velocity-{Flat,Rough}-MicroDuck` | **主任务**：速度指令 + 头部姿态指令行走 |
| `Mjlab-VelStand-{Flat,Rough}-MicroDuck` | 行走 + 跌倒恢复合一策略 |
| `Mjlab-StandUp-{Flat,Rough}-MicroDuck` | 从俯卧/仰卧/坐姿站起并保持 |
| `Mjlab-SitStand-{Flat,Rough}-MicroDuck` | 坐姿 ↔ 站姿轻柔切换 |
| `Mjlab-GroundPick-{Flat,Rough}-MicroDuck` | 下蹲用嘴尖触地再站起 |
| `Mjlab-BallKick-Flat-MicroDuck` | 向前踢 70mm/15g 小球（策略看不到球） |
| `Mjlab-Roulade-Flat-MicroDuck` | 前滚翻落地站起 |
| `Mjlab-Velocity-Flat-MicroDuck-Rollers` | 装滑轮后的滑轮速度跟踪 |
| `Mjlab-Velocity-Swizzle-MicroDuck` | 经典对称滑步 |
| `Mjlab-RollerCrouch-Flat-MicroDuck` | 滑行中下蹲 |
| `Mjlab-RollerSlope-Flat-MicroDuck` | 滑坡滑行 |
| `Mjlab-RollerStandUp-Flat-MicroDuck` | 从地面站到滑轮上 |
| `Mjlab-Spin-Flat-MicroDuck` | 穿滑轮原地快速旋转 |

**Backlash（齿隙）变体**：每个主任务在任务 id 中 `MicroDuck` 前插入 `-Backlash`，如 `Mjlab-Velocity-Flat-Backlash-MicroDuck`，会在 14 个舵机上串联 ±1° 齿轮间隙模型（观测和动作维度不变，ONNX/运行时无需改动），是 sim2real 配方的一部分。

### 6.4 sim2real 配方已内置（了解即可）

- **BAM 舵机模型**：XL330 舵机按电压控制律建模（反电动势、库仑/Stribeck/负载摩擦），而非理想 PD——这个尺度下舵机保真度决定 sim2real 差距；
- **域随机化**：电池电压、负载压降、指令延迟、摩擦幅值逐环境随机（各 env cfg 文件顶部的 `ENABLE_*` 开关）；
- **关节布局**：14 个舵机 = 0-4 左腿（hip_yaw/roll/pitch、knee、ankle）+ 5-8 颈/头（neck_pitch、head_pitch/yaw/roll）+ 9-13 右腿。

---

## 七、步骤 6：查看训练结果、导出 ONNX、回片验证

```bash
cd /home/creator/ws/301-microduck/microduck_rl

# ① 在查看器里观看训练好的策略（在线 wandb）
uv run play Mjlab-Velocity-Flat-MicroDuck \
  --wandb-run-path <entity>/<project>/<run_id>

# ② 导出 ONNX（观测归一化器会被烤进计算图）
uv run scripts/export.py Mjlab-Velocity-Flat-MicroDuck \
  --wandb-run-path <entity>/<project>/<run_id>
# 产物形如 output.onnx

# ③ 用导出的 ONNX 在 CPU MuJoCo 里键盘遥控验证
uv run scripts/infer_policy.py --walking output.onnx
```

> **重要**：部署用的 ONNX **必须**由 `scripts/export.py` 产出——导出器会把观测归一化器烘焙进 ONNX 图。手工从 checkpoint 转换的 ONNX 在运行时会吃到未归一化观测，行为完全不可控。
>
> 离线 wandb（`WANDB_MODE=offline`）训练时，run 数据在本地 `wandb/` 目录，`play`/`export` 的 run-path 按 wandb 本地 run 目录填写，或先 `wandb sync` 上传后用在线路径。

---

## 八、将来有真机时：上机部署（备忘）

1. 把导出的 ONNX 放到板子上，编辑 `microduck/deploy/robotd.toml`（或板子上的 `/etc/robot/robotd.toml`）：

   ```toml
   [policy]
   walk  = "/home/radxa/my_walk.onnx"
   stand = "/home/radxa/my_stand.onnx"
   ```

2. `sudo systemctl restart robotd`。加载失败的策略会通过 `robot.health` 报 `policy unavailable: <原因>`，控制环继续保持姿态，不会让鸭子摔地上。
3. 所有策略共享 61 维观测契约，运行时可在 walk/recover/trick 策略间热切换——这就是 `infer_policy.py --new-cmd-obs` 在仿真里彩排的那套逻辑。

---

## 九、常见问题

| 现象 | 排查 |
|------|------|
| `git submodule add` 报 already exists / 路径已被跟踪 | 之前中断的尝试残留：`git rm --cached 301-microduck/microduck_rl 2>/dev/null; rmdir 301-microduck/microduck_rl 2>/dev/null`，并删掉 `.gitmodules` 里对应的 `[submodule "301-microduck/microduck_rl"]` 段落后重试 |
| `torch.cuda.is_available()` 为 False | 先 `nvidia-smi` 确认驱动；确认用的是 `uv run`（.venv 内 torch 2.9.1 CUDA 版）而非系统 Python；容器内需 `--gpus all` |
| `uv sync` 卡在 bam 下载 | bam 是 GitHub git 依赖，检查 SSH key / 代理；可手动 `git clone https://github.com/Rhoban/bam` 测试连通性 |
| PyPI 下载慢/超时 | `export UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/`、`UV_HTTP_TIMEOUT=600` |
| warp 首次训练报错或编译慢 | warp 首次运行会 JIT 编译 CUDA kernel（5090 上约 1-2 分钟，十几行 `Module ... (compiled)`），缓存在 `~/.cache/warp/`，属正常；保持驱动为新版 |
| `train` 启动后停在 `wandb: Enter your choice:` | 没预设 wandb 模式：输入 `3` 继续（本次不记录），或 Ctrl-C 后 `export WANDB_MODE=offline` 再启动（run 存本地，供 `play`/`export.py` 引用） |
| 训练 OOM | 降 `--env.scene.num-envs`（4096→2048→1024） |
| `list-envs` 找不到 Microduck | 确认在 `microduck_rl` 目录下、`uv sync` 成功（任务通过 `mjlab.tasks` entry point 注册） |
| 想用 setup_py312 脚本装 GPU 环境 | 该脚本已修正为 cp312（装 torch 2.11.0+cu128，见 `scripts/setup_py312_cu128_rtx5090_torch.sh`），但 microduck_rl 的 torch 由 uv.lock 锁定 2.9.1，两套不要混用：训练环境只走步骤 3 的 `uv sync` |
| 发现 `microduck_rl/.venv` 被创建 | 执行 `uv sync`/`uv run` 时忘了 `export UV_PROJECT_ENVIRONMENT`，uv 在子模块内重建了环境；确认 `echo $UV_PROJECT_ENVIRONMENT` 指向 `301-microduck/.venv` 后，删除 `microduck_rl/.venv` 即可 |
| `uv sync` 后 `uv` 或 `pip` 命令消失 | `uv sync` 精确同步会移除 lockfile 外的包；激活 `.venv` 后 `python3.12 -m ensurepip --upgrade && pip install uv` 恢复，补装包改用 `uv pip install` |
| pytest 报 `PluginValidationError: Plugin 'launch_testing'` 或先报 `No module named 'lark'` | `~/.bashrc` source 的 ROS2 Humble 经 PYTHONPATH 泄漏进 venv，其 py3.10 旧版 launch_testing 插件被 pytest 自动加载且钩子签名过旧；用 `env -u PYTHONPATH uv run pytest tests/ -q` 剥离后跑即可，**不要**为此安装 lark |
| 唯一失败 `test_x86_64_resolution_stays_on_pypi`（断言 torch 应来自 pypi.org，实际是 aliyun） | uv 把"索引配置变化"视为 lockfile 过期：只要 shell 里 export 着 `UV_INDEX_URL`，**每条** `uv sync`/`uv run` 都会自动重锁并把 uv.lock 的源 URL 全部改写成镜像（实测 `uv run python -c pass` 也会触发）。因此 `git checkout -- uv.lock` 后紧跟的 `uv run` 会立即改回去——必须先 `unset UV_INDEX_URL` 再恢复 lock。该测试是上游 CI 保护，与本地训练无关；已装好的环境不受影响 |
| `infer_policy.py` 报 `Got: 51 Expected: 61`（obs 维度不匹配） | 官方 ONNX 用 61 维新观测契约（48 本体 + 13 指令），脚本默认构造 51 维旧格式，加 `--new-cmd-obs` 即可。注意按键在终端输入而非 viewer 窗口；`G`/`Y`/`R`/`K`/`L` 需配合 `--ground-pick`/`--sit`/`--roulade`/`--kick-left` 等参数加载对应策略才生效 |

---

## 十、参考链接

- microduck_rl 仓库：<https://github.com/pollen-robotics/microduck_rl>（develop 分支 README 为最新 Quickstart）
- microduck 运行时：<https://github.com/pollen-robotics/microduck>
- mjlab 训练框架：<https://github.com/mujocolab/mjlab>（MuJoCo Warp + rsl_rl）
- BAM 舵机模型：<https://github.com/Rhoban/bam>
- onshape-to-robot（MJCF 导出工具）：<https://github.com/Rhoban/onshape-to-robot>
