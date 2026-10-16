# microduck_rl 入口地图（Entry Points Map）

> 本文梳理 [`microduck_rl/`](../../../microduck_rl/) 仓库的**所有程序入口**——它没有单一的 `main`，而是"一个 mjlab 插件包 + 一组工具脚本"，入口分散在三处。本文回答："我想跑某个功能，该从哪个文件、哪个命令进？"
>
> 配套文档：[`architecture_and_glossary.md`](../02-architecture/architecture_and_glossary.md)（整体理解与概念详解，其 [§D.3](../02-architecture/architecture_and_glossary.md) 展开 `train` 调用链路）、[`desktop_quickstart.md`](../01-quickstart/desktop_quickstart.md)（部署与首次训练）。

---

## 核心结论：没有单一 main

microduck_rl 的**主身份是"被 mjlab 加载的库"**，而非"自己启动的应用"。它的业务逻辑（奖励、观测、课程）大多是**被框架在训练循环中回调的零件**，不是一个从头跑到尾的 `main`。因此它有**三类入口**：

```text
① Entry Point（插件入口）—— 最本质，无 main，靠 import 副作用
   pyproject.toml [project.entry-points."mjlab.tasks"]
        └─ mjlab_microduck.tasks  ← mjlab 启动时自动加载，注册 27 个任务

② Console Scripts（可执行命令）—— 3 个，用 `uv run <name>`
   pyproject.toml [project.scripts]
        ├─ train      → train_cli:main      （训练，转发给 mjlab）
        ├─ duck-body  → body_server:main     （反向 sim2real 服务器）
        └─ publish    → publish.cli:main     （发布到 HF Hub）

③ 独立脚本（scripts/ 与 src 内工具）—— 用 `uv run scripts/xxx.py`
        ├─ 有 main() 的工具：infer_policy / export / play_latest / testbench …
        ├─ thin wrapper：scripts/export.py、scripts/hf/train_hf.py（转发到 src 包）
        ├─ 共享库（非入口）：wandb_utils.py
        └─ 顶层执行脚本：crouch_pose_editor.py
```

**一句话**：概念上的"主入口"是 ① `mjlab_microduck.tasks`（它作为插件被 mjlab 加载）；能敲的"主命令"是 ② `train`（但它的 `main` 只是转发壳）。

---

## 前置知识：uv 与 uv run（第一次读必看）

上面反复出现 `uv run xxx`。如果你第一次接触这个仓库，先搞懂这个前缀——它是**运行本项目一切命令的统一方式**。

### uv 是什么

**uv** 是 [Astral](https://astral.sh/) 公司（也是 `ruff` 的作者）用 **Rust** 写的**超快 Python 包与项目管理器**。它一个工具替代了过去一整套：

| 传统工具 | 干什么 | uv 对应 |
|---|---|---|
| `pip` | 装包 | `uv pip` / `uv add` |
| `venv`/`virtualenv` | 建虚拟环境 | uv 自动管理 `.venv` |
| `pip-tools`/`poetry` | 锁依赖版本 | [`uv.lock`](../../../microduck_rl/uv.lock)（跨平台确定性锁文件） |
| `pyenv` | 管 Python 版本 | `uv python` |

**特点**：快（Rust 实现，比 pip 快 10–100×）、单一 lockfile 保证"你这台和我这台装出来的依赖完全一致"、自动创建和同步 `.venv`。

**本项目就是 uv 项目**：有 [`uv.lock`](../../../microduck_rl/uv.lock)、[`pyproject.toml`](../../../microduck_rl/pyproject.toml) 里的 `[tool.uv]` 段（override-dependencies、sources、CUDA 索引路由），README 也要求先装 uv。

> 类比：**uv 之于 Python ≈ cargo 之于 Rust ≈ npm 之于 Node**——一个工具搞定"依赖 + 环境 + 运行"。

### uv run 是干什么的

`uv run <命令>` = **"在本项目的虚拟环境里运行这个命令，运行前自动确保依赖已按 `uv.lock` 装好"**。它替你做了三件烦人的事：

1. 向上找到项目根（含 `pyproject.toml` 的目录）；
2. 检查 `.venv` 是否与 `uv.lock` 一致，不一致就**自动 sync**（装/更新依赖）；
3. 在 `.venv` 环境里执行命令。

所以你**不需要**手动 `source .venv/bin/activate`、也不需要先 `pip install`——`uv run` 一条命令保证"环境正确 + 运行"。三种常见形态：

```bash
uv run train …                 # ① 跑已安装的 console script（train/publish/duck-body，或 mjlab 的 play/list-envs）
uv run scripts/export.py …     # ② 跑脚本（等价 uv run python scripts/export.py …）
uv run --with pytest pytest tests/   # ③ 临时加一个依赖再跑（pytest 不在主依赖里，--with 临时拉进来）
```

> 相关命令 `uv sync`：只装依赖、不运行任何东西。首次 `git clone` 后常先跑它。**ARM 机器（DGX Spark / Jetson）上首次 sync 要 `export UV_HTTP_TIMEOUT=600`**——它会拉 ~2 GB 的 CUDA wheel，uv 默认 30s HTTP 超时会在下载途中 abort（README 明确提醒）。

### uv run train 支持哪些指令

`train` 命令的结构（参数来自 [`AGENTS.md`](../../../microduck_rl/AGENTS.md) "Commands" 段与 README 的真实用例）：

```text
uv run train <TASK_ID> [--env.<字段> 值 …] [--agent.<字段> 值 …] [--hf-jobs]
```

| 部分 | 含义 | 例子 |
|---|---|---|
| `<TASK_ID>` | **位置参数**，要训练的任务名 | `Mjlab-Velocity-Flat-MicroDuck`（用 `uv run list-envs` 列出全部） |
| `--env.*` | 覆盖**环境 cfg**（`ManagerBasedRlEnvCfg`）的任意字段 | `--env.scene.num-envs 4096`（并行环境数）； terrain、reward 权重、DR 开关等凡是 env_cfg 里的字段都能覆盖 |
| `--agent.*` | 覆盖**RL/训练 cfg**（`RslRlOnPolicyRunnerCfg`）的字段 | `--agent.max_iterations 5`、`--agent.run-name resume`、`--agent.load-checkpoint model_29999.pt`、`--agent.resume True` |
| `--hf-jobs` | **本项目特有**（非 mjlab），被 entry-point 拦截，提交到 HF 托管 GPU | 见 [§一](#一entry-point插件入口最本质) 与 [§五](#五train-的调用链路最核心入口展开) |

**关键认知**：`--env.*` / `--agent.*` 不是固定清单，而是 **tyro 从所选 task 的 cfg dataclass 自动生成的**——每个 task 的 env_cfg 不同，可覆盖的字段就不同。**完整列表随时用 `uv run train <TASK_ID> --help` 查看**。

**常用组合**（照抄即可，来自项目实际约定）：

```bash
# ① 冒烟测试 —— 永远先跑这个！5 迭代 + 64 环境，几分钱就能 catch ~95% 的配置错误
uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 64 --agent.max_iterations 5

# ② 正式训练 —— 4096 环境，GPU 上约 1-2 小时出可用步态
uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096

# ③ 从 checkpoint 续训
uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096 \
    --agent.run-name resume --agent.load-checkpoint model_29999.pt --agent.resume True

# ④ 没有 GPU？加 --hf-jobs 提交到 Hugging Face Jobs 远程跑
uv run train Mjlab-Velocity-Flat-MicroDuck --hf-jobs
```

> AGENTS.md 的原话："A 5-iteration smoke test at 64 envs catches ~95% of config errors for cents. Never launch a long run without one."（先冒烟，再长跑。）

**`train` 的兄弟命令**（同样 `uv run` 前缀，但由 **mjlab 框架**提供，不在本项目 `[project.scripts]` 里）：

```bash
uv run list-envs                 # 打印实时任务注册表（看有哪些 TASK_ID 可选）
uv run play <TASK_ID> --wandb-run-path <entity/project/run_id>   # 在 viewer 里回放训练好的策略
```

（`train` 本身也是 mjlab 的命令，本项目只是声明了一个同名壳，见 [§二](#二console-scripts三个可执行命令)。）

### uv run train 的调用过程（从敲命令到训练开始）

整个过程分**两层**：**uv 层**（找到并启动 `train`）+ **mjlab 层**（`train` 内部怎么跑起来，[§五](#五train-的调用链路最核心入口展开) 已详解）。

```text
你敲：uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096
  │
  │  ── uv 层（uv 这个工具做的事）──
  ▼ ① uv 向上找到项目根（含 pyproject.toml 的目录）
  ▼ ② 读 pyproject.toml + uv.lock，检查 .venv 是否与 lock 一致
  │      不一致 → 自动 sync（把依赖装进 .venv）
  ▼ ③ 在 .venv/bin/ 里找到名为 train 的可执行入口（console script）
  │      （安装时由 [project.scripts] / mjlab 生成）
  ▼ ④ exec .venv/bin/train，把 <TASK_ID> 和 --env.* 等参数原样传进去
  │      ≈ 你手动 activate .venv 再敲 train …，但 uv 自动保证了环境正确
  │
  │  ── mjlab 层（进入 train 之后，详见 §五）──
  ▼ ⑤ .venv/bin/train → mjlab.scripts.train:main → import mjlab
  ▼ ⑥ mjlab 加载 entry-point → import mjlab_microduck.tasks（注册 27 任务 / 拦截 --hf-jobs）
  ▼ ⑦ tyro 两阶段解析 argv：先认 <TASK_ID>，再按该 task 的 cfg 解析 --env.* / --agent.*
  ▼ ⑧ 构建 ManagerBasedRlEnv → MicroduckOnPolicyRunner → runner.learn()（PPO 训练循环）
```

一句话：**`uv run` 负责"把正确的环境和正确的 `train` 可执行文件准备好并启动"，之后接力给 mjlab 的 `train` 去做真正的解析与训练。** ⑤–⑧ 的完整推导（含命名冲突、`SystemExit` 穿透）见 [§五](#五train-的调用链路最核心入口展开)。

---

## 入口总览（速查表）

| 类 | 入口 | 调用方式 | main / 目标位置 | 作用 |
|---|---|---|---|---|
| ① 插件 | `mjlab.tasks` entry-point | 被 mjlab 自动 import | [`tasks/__init__.py`](../../../microduck_rl/src/mjlab_microduck/tasks/__init__.py) | 注册 27 个任务 + 拦截 `--hf-jobs` |
| ② 命令 | `train` | `uv run train <task> …` | [`train_cli.py:25`](../../../microduck_rl/src/mjlab_microduck/train_cli.py#L25) | 训练（转发 mjlab） |
| ② 命令 | `duck-body` | `uv run duck-body …` | [`body_server.py:494`](../../../microduck_rl/src/mjlab_microduck/sim/body_server.py#L494) | 反向 sim2real 服务器 |
| ② 命令 | `publish` | `uv run publish …` | [`publish/cli.py:221`](../../../microduck_rl/src/mjlab_microduck/publish/cli.py#L221) | 导出→gate→上传 HF Hub |
| ③ 脚本 | export | `uv run scripts/export.py …` | [`export.py:280`](../../../microduck_rl/src/mjlab_microduck/export.py#L280) | checkpoint → ONNX |
| ③ 脚本 | infer_policy | `uv run scripts/infer_policy.py …` | [`infer_policy.py:896`](../../../microduck_rl/scripts/infer_policy.py#L896) | 在 MuJoCo 跑 ONNX 策略 |
| ③ 脚本 | play_latest | `uv run scripts/play_latest.py …` | [`play_latest.py:26`](../../../microduck_rl/scripts/play_latest.py#L26) | 回放最新 wandb run |
| ③ 脚本 | testbench_sim2real | `uv run scripts/testbench_sim2real.py …` | [`testbench_sim2real.py:541`](../../../microduck_rl/scripts/testbench_sim2real.py#L541) | sim2real 测试台 |
| ③ 脚本 | validate_bam_testbench | `uv run scripts/validate_bam_testbench.py` | [`validate_bam_testbench.py:181`](../../../microduck_rl/scripts/validate_bam_testbench.py#L181) | 验证 BAM 执行器参数 |
| ③ 脚本 | view_slope_terrain | `uv run scripts/view_slope_terrain.py` | [`view_slope_terrain.py:59`](../../../microduck_rl/scripts/view_slope_terrain.py#L59) | 可视化斜坡地形 |
| ③ 脚本 | plot_observations | `uv run scripts/plot_observations_comparison_plotly.py` | [`plot_observations_comparison_plotly.py:267`](../../../microduck_rl/scripts/plot_observations_comparison_plotly.py#L267) | 画真机 vs 仿真 obs 对比 |
| ③ 脚本 | crouch_pose_editor | `uv run python scripts/crouch_pose_editor.py` | 顶层执行（无 main） | 交互式编辑轮上蹲姿 |
| ③ 脚本 | hf/train_hf | `uv run scripts/hf/train_hf.py <task> …` | [`train_hf.py:14`](../../../microduck_rl/scripts/hf/train_hf.py#L14) | `--hf-jobs` 逃生舱 |
| ③ 脚本 | hf/uploader | HF job 内后台运行 | [`uploader.py:18`](../../../microduck_rl/scripts/hf/uploader.py#L18) | job 内上传 checkpoint |
| ③ 工具 | add_backlash | `python …/add_backlash.py` | [`add_backlash.py:60`](../../../microduck_rl/src/mjlab_microduck/robot/microduck/add_backlash.py#L60) | 给 XML 模型串齿隙铰 |
| ③ 工具 | microduck_constants | `python …/microduck_constants.py` | [`microduck_constants.py:264`](../../../microduck_rl/src/mjlab_microduck/robot/microduck_constants.py#L264) | 开 viewer 看机器人模型 |
| — 库 | wandb_utils | 被 import（非入口） | [`wandb_utils.py`](../../../microduck_rl/scripts/wandb_utils.py) | play_latest/export 共享的 wandb helper |

---

## 一、Entry Point：插件入口（最本质）

[`pyproject.toml`](../../../microduck_rl/pyproject.toml#L40-L41) 声明：

```toml
[project.entry-points."mjlab.tasks"]
mjlab_microduck = "mjlab_microduck.tasks"
```

- **没有 `main` 函数**，靠 **import 副作用**工作：`import mjlab` 时，mjlab 的 `__init__.py` 在模块作用域调 `_import_registered_packages()`，遍历 `mjlab.tasks` 这个 entry-point 组，自动 import [`mjlab_microduck.tasks`](../../../microduck_rl/src/mjlab_microduck/tasks/__init__.py)。
- 该模块**顶层**依次做两件事：
  1. `maybe_submit_to_hf_jobs()`——若命令行有 `--hf-jobs`，在此提交到 HF 托管 GPU 并 `SystemExit`（本地训练器永不启动）。
  2. `register_mjlab_task()` × 27——把 task_id → (env_cfg, rl_cfg, runner) 建进 mjlab 全局任务表。
- **为什么这是"最本质"的入口**：它让 microduck 的任务对 mjlab 可见。没有它，`uv run train Mjlab-Velocity-Flat-MicroDuck` 会因为"查无此任务"而失败。而且这条路径是 **mjlab 自己的导入路径**，任何安装顺序都保证它执行——`--hf-jobs` 的拦截正是依赖这一点（见 [§六](#六设计要点为什么这样组织)）。

---

## 二、Console Scripts：三个可执行命令

[`pyproject.toml`](../../../microduck_rl/pyproject.toml#L43-L56) 的 `[project.scripts]`，安装后在 `.venv/bin/` 生成可执行入口，用 `uv run <name>` 调用：

### `train` → [`train_cli.py:main`](../../../microduck_rl/src/mjlab_microduck/train_cli.py#L25)

```python
def main() -> int | None:
    from mjlab.scripts.train import main as mjlab_train_main
    return mjlab_train_main()
```

- **只是个转发壳**——真正的训练逻辑在 mjlab。
- **故意与 mjlab 的 `train` 撞名**（last-writer-wins，mjlab 赢，所以 `.venv/bin/train` 实际指向 mjlab 的实现，这个壳通常根本不被调用）。
- **但必须保留声明**：删掉它不会把名字还给 mjlab，反而让下次 `uv sync` 按本仓库 RECORD 卸载 `.venv/bin/train` 而不重装 mjlab 的，`train` 从 venv 消失，`uv run train` 落到 PATH 上任意叫 `train` 的程序——2026-08-31 就落到 liblinear 的 `train`，回答 `can't open input file Mjlab-Velocity-Flat-MicroDuck`。由 `tests/test_hf_jobs_flag.py` 锁住。

### `duck-body` → [`body_server.py:main`](../../../microduck_rl/src/mjlab_microduck/sim/body_server.py#L494)

- 反向 sim2real：启动一个 TCP 服务器，把 MuJoCo 里仿真的鸭子身体（关节位置/速度/电流/IMU）喂给真机的 `robotd --sim`，让真机的完整软件栈驱动一个仿真身体。
- 用法：`uv run duck-body --ducks 4 --port 7801`。

### `publish` → [`publish/cli.py:main`](../../../microduck_rl/src/mjlab_microduck/publish/cli.py#L221)

```python
def main() -> int:
    cfg = tyro.cli(PublishConfig)
    ...
```

- 发布链路：解析权重（从 checkpoint 走 export，或直接用现成 onnx）→ `check_onnx`（shape gate）→ `smoke_run_onnx`（跑 50 步验非 NaN/非常数）→ build+validate manifest → 上传 HF Hub。
- 用法：`uv run publish --task … --repo user/microduck-my-walk --kind perpetual --slot walk`。

---

## 三、独立脚本：`scripts/` 与 `src` 内工具

这些不在 `[project.scripts]` 里，用 `uv run scripts/xxx.py`（或 `uv run python scripts/xxx.py`）调用。按入口形态分四种：

### 3.1 有 `main()` 的工具脚本（自带完整逻辑）

| 脚本 | main | 作用 |
|---|---|---|
| [`infer_policy.py`](../../../microduck_rl/scripts/infer_policy.py#L896)（1524 行） | L896 | 在经典 MuJoCo（CPU 单环境）里加载 ONNX 策略跑起来，部署演练/看效果。镜像了 BAM 参数（不 import，避免拖入 mjlab/torch/warp 的 ~16s 导入） |
| [`play_latest.py`](../../../microduck_rl/scripts/play_latest.py#L26) | L26 | 找某用户最新的 wandb run 并回放 |
| [`testbench_sim2real.py`](../../../microduck_rl/scripts/testbench_sim2real.py#L541) | L541 | sim2real 测试台，`--mode sim/real` 对比两侧 rollout |
| [`validate_bam_testbench.py`](../../../microduck_rl/scripts/validate_bam_testbench.py#L181) | L181 | 验证 BAM 执行器模型参数（`--plot` 看图） |
| [`view_slope_terrain.py`](../../../microduck_rl/scripts/view_slope_terrain.py#L59) | L59 | 可视化斜坡地形（`--rows` 控制显示的刚度数） |
| [`plot_observations_comparison_plotly.py`](../../../microduck_rl/scripts/plot_observations_comparison_plotly.py#L267) | L267 | 用 Plotly 画真机 vs 仿真观测对比 |
| [`hf/uploader.py`](../../../microduck_rl/scripts/hf/uploader.py#L18) | L18 | 在 HF job 容器内**后台运行**，读 `CKPT_REPO` 环境变量，把 checkpoint 持续上传到 HF Hub |

### 3.2 Thin wrapper（转发到 `src` 包里的 main）

这两个脚本本身几乎没逻辑，只是把调用转发进包——**目的是让逻辑住在包里，好被 `uv run publish` 等复用**：

- [`scripts/export.py`](../../../microduck_rl/scripts/export.py)（11 行）：`from mjlab_microduck.export import main`，真正逻辑在 [`src/mjlab_microduck/export.py:280`](../../../microduck_rl/src/mjlab_microduck/export.py#L280)。docstring 明说"The logic lives in the package so `uv run publish` shares it"。
- [`scripts/hf/train_hf.py`](../../../microduck_rl/scripts/hf/train_hf.py)（15 行）：`from mjlab_microduck.hf_jobs import submit`，是 `--hf-jobs` 集成标志的**back-compat 逃生舱**——万一 entry-point 拦截失效，可用它直接提交。

### 3.3 共享库与顶层执行脚本（非命令入口）

- [`scripts/wandb_utils.py`](../../../microduck_rl/scripts/wandb_utils.py)：**不是入口**，是 `play_latest`/`export_latest` 共享的 wandb helper 库（`find_latest_run`、`find_latest_runs`，常量 `WANDB_PROJECT = "pollen-robotics/mjlab_microduck"`）。被 import，不单独运行。
- [`scripts/crouch_pose_editor.py`](../../../microduck_rl/scripts/crouch_pose_editor.py)：**无 `main` 封装**，是顶层直接执行的交互式脚本——开 MuJoCo viewer（roller 机器人 standing），拖滑块摆出蹲姿，关重力、保持 base 直立下降，关窗时打印 `CROUCH_POSE` dict 供粘贴。用法 `uv run python scripts/crouch_pose_editor.py`。

### 3.4 `src` 内的模型工具（直接 `python` 运行）

- [`robot/microduck/add_backlash.py:60`](../../../microduck_rl/src/mjlab_microduck/robot/microduck/add_backlash.py#L60)：后处理器，给每个 servo 关节串一个 `passive_<joint>_backlash` ±1° 齿隙铰，生成 `*_backlash.xml`。
- [`robot/microduck_constants.py:264`](../../../microduck_rl/src/mjlab_microduck/robot/microduck_constants.py#L264)：直接运行会开 `mujoco.viewer` 展示机器人模型（调试用）。

---

## 四、按任务选入口（实用指引）

```text
训练一只鸭子      → uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096
                    （train_cli:main → 转发 mjlab → 经 entry-point 加载本包任务）
远程 GPU 训练     → uv run train <task> … --hf-jobs
                    （被 tasks/__init__.py 顶层拦截 → hf_jobs.submit）
导出 ONNX        → uv run scripts/export.py <TASK> --wandb-run-path <entity/project/run> [--checkpoint 3000]
发布到 HF Hub     → uv run publish --task … --repo … --kind perpetual --slot walk
部署演练（看效果） → uv run scripts/infer_policy.py <policy.onnx>
回放最新 run      → uv run scripts/play_latest.py <user_filter>
反向 sim2real     → uv run duck-body --ducks 4 --port 7801   +   真机 robotd --sim 127.0.0.1:7801
编辑蹲姿          → uv run python scripts/crouch_pose_editor.py
给模型加齿隙       → python src/mjlab_microduck/robot/microduck/add_backlash.py …
```

---

## 五、`train` 的调用链路（最核心入口展开）

`uv run train` 表面是普通命令，背后是三层机制协作（完整推导见 [`architecture_and_glossary.md` §D.3](../02-architecture/architecture_and_glossary.md)）：

```text
uv run train <task> [--hf-jobs]
   │
   ▼  .venv/bin/train（mjlab 赢了命名冲突）→ mjlab.scripts.train:main
   │     执行 `from mjlab.scripts.train import main` 时触发 `import mjlab`
   ▼  mjlab/__init__.py 模块作用域 → _import_registered_packages()
   │     遍历 "mjlab.tasks" entry-point 组
   ▼  import mjlab_microduck.tasks        ← ① 插件入口被自动加载
   │     顶层第一件事：maybe_submit_to_hf_jobs()
   │        ├─ 有 --hf-jobs → hf_jobs.submit() → sys.exit()（SystemExit 是
   │        │                 BaseException，穿过 loader 的 except Exception）
   │        └─ 无 → no-op，继续 register_mjlab_task() × 27
   ▼  回到 mjlab.scripts.train:main → tyro 两阶段解析 argv → 取 env_cfg
   ▼  构建 ManagerBasedRlEnv → MicroduckOnPolicyRunner → runner.learn()（PPO 循环）
```

---

## 六、设计要点（为什么这样组织）

1. **库优先于命令**：核心逻辑（export/submit）住在 `src` 包里，`scripts/` 下只放 thin wrapper。这样 `uv run publish` 能直接复用 `mjlab_microduck.export`，不必 subprocess 调脚本，也避免逻辑两份漂移。
2. **entry-point 是"夺不走"的钩子**：`--hf-jobs` 不放在 console script 里处理（那个 script 可能因命名冲突根本不被调用），而放在 `mjlab.tasks` entry-point——因为"mjlab 启动必 import 本插件"是 mjlab 自己的导入路径，任何安装顺序都保证它先于 tyro 解析 argv 执行。
3. **撞名保留而非删除**：`train` 与 mjlab 撞名后，删除声明的危害（venv 里没有 `train`）比保留（一个用不上的壳）更大，于是选择"保留 + 做成无害"，并用测试锁住。
4. **`SystemExit` 穿透 `except Exception`**：插件 loader 用 `except Exception` 防单个插件坏掉拖垮全部，而 `SystemExit` 继承自 `BaseException`，能穿透出去终止进程——这是 `--hf-jobs` 提交后能干净退出、不启动本地训练器的关键。
5. **逃生舱冗余**：`scripts/hf/train_hf.py` 是 entry-point 拦截的后备；`MICRODUCK_IN_HF_JOB=1` 环境变量防止 job 容器内递归提交。关键路径都留了 plan B。

---

> 本文为学习梳理笔记，基于 [`pyproject.toml`](../../../microduck_rl/pyproject.toml) 与源码入口核实整理。配套：[`architecture_and_glossary.md`](../02-architecture/architecture_and_glossary.md)、[`desktop_quickstart.md`](../01-quickstart/desktop_quickstart.md)、[仓库 README](../../../microduck_rl/README.md)。
