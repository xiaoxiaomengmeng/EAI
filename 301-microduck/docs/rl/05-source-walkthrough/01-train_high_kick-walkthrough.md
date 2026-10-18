# 01 · train_high_kick.py 逐行解读：自定义任务如何挂进 mjlab 训练管线

> 05-source-walkthrough 系列第 ① 篇。标本 = 本仓库实测跑通的 high kick 任务入口
> [`practice/tasks/ball_kick/train_high_kick.py`](../../../practice/tasks/ball_kick/train_high_kick.py)（68 行），
> 上游依赖 [`mjlab/tasks/registry.py`](../../../.venv/lib/python3.12/site-packages/mjlab/tasks/registry.py)（72 行）、
> [`mjlab/scripts/train.py`](../../../.venv/lib/python3.12/site-packages/mjlab/scripts/train.py)。
> 跟着这条命令读到底：`WANDB_MODE=offline MUJOCO_GL=egl ../../../.venv/bin/python train_high_kick.py --env.scene.num-envs 64 --agent.max-iterations 1`
>
> 行号以当前仓库版本为准；上游为 site-packages 只读源码，引用行号便于跳转。

---

## 0. 一张图看全脚本的执行顺序

```text
python train_high_kick.py --env.scene.num-envs 512 --agent.max-iterations 300
│
├─ L21  os.environ.setdefault("MUJOCO_GL", "egl")   # headless 渲染开关，必须在 import mujoco 之前
├─ L25  sys.path.insert(0, 脚本所在目录)             # 让 high_kick_cfg 可被 import
├─ L30  from mjlab.scripts.train import TrainConfig, launch_training
├─ L31  from mjlab.tasks.registry import list_tasks, register_mjlab_task
│
├─ L56  register_high_kick_task()
│   ├─ L39  import mjlab_microduck.tasks            # 副作用：官方 17 个 task 进注册表
│   ├─ L40  from mjlab_microduck.tasks import MicroduckOnPolicyRunner
│   ├─ L44  if TASK_ID in list_tasks(): return      # 幂等守卫
│   └─ L46  register_mjlab_task(task_id, env_cfg, play_env_cfg, rl_cfg, runner_cls)
│                                                   # 写进全局 _REGISTRY（一个 dict 赋值）
├─ L57  tyro.cli(TrainConfig, default=TrainConfig.from_task(TASK_ID), ...)
│       └─ train.py L41-43  from_task = load_env_cfg + load_rl_cfg（均 deepcopy）
│          命令行 flag（--env.scene.num-envs …）逐字段覆盖这份默认配置
└─ L63  launch_training(task_id=TASK_ID, args=args)
    ├─ train.py L108  ManagerBasedRlEnv(cfg=cfg.env, device, render_mode)  # 7 个 manager 在此诞生
    ├─ train.py L151  load_runner_cls → MicroduckOnPolicyRunner（None 才回退默认 runner）
    ├─ train.py L165  runner = runner_cls(env, agent_cfg, log_dir, device)
    └─ train.py L173  runner.learn(...)             # 进入 rsl_rl PPO 主循环（另篇走读）
```

全脚本**没有一行训练逻辑**：它只干两件事——**把任务写进注册表**、**把 CLI 参数交给官方管线**。这就是「不改子模块也能加任务」的全部秘密。

---

## 1. import 之前的三行 preamble（L17-31）：每行都是一个坑

| 行 | 代码 | 为什么必须这样 |
|---|---|---|
| L21 | `os.environ.setdefault("MUJOCO_GL", "egl")` | MuJoCo 的 GL 后端在 `import mujoco` 那一刻就定型；headless 机器没有 DISPLAY，不提前设 egl 后面渲染必崩。EGL（Embedded-System GL）是 Khronos 的「渲染 API ↔ 原生平台」接口层，能建**无窗口 GPU OpenGL 上下文**做离屏渲染（对照：`glfw` 需窗口系统、`osmesa` 为 CPU 软件渲染）。用 `setdefault` 是尊重外部已显式指定的值 |
| L25 | `sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))` | 本脚本所在目录不是安装包；插进 sys.path 后 L42 的 `from high_kick_cfg import ...` 才找得到（mjlab/mjlab_microduck 靠 editable 安装可见，practice 侧模块靠这一行） |
| L27-31 | 各 import 带 `# noqa: E402` | import 出现在上面两条副作用语句之后，违反 PEP8「import 置顶」，逐行抑制告警 |
| L33 | `TASK_ID = "Mjlab-BallKickHigh-Flat-MicroDuck"` | 注册表 key、`from_task`、`launch_training`、play 入口全靠这一个字符串认任务；命名仿官方风格（框架-任务-地形-机器人） |

---

## 2. register_high_kick_task()：到底在「注册」什么

### 2.1 五个参数各自被谁消费

| 参数 | 含义 | 下游消费者（代码位置） |
|---|---|---|
| `task_id` | 任务唯一 ID | 注册表 key；`TrainConfig.from_task` / `launch_training` / `run_play` 都按它查表 |
| `env_cfg` | **训练**环境配方：obs/actions/rewards/terminations/events/DR/场景全在里面 | `run_train` L108 拿它建 `ManagerBasedRlEnv`；**reward 权重也在这里**（`cfg.rewards`） |
| `play_env_cfg` | **回放/评估**环境配方（`play=True` 分支：关随机化类事件/课程等） | `run_play` 里 `load_env_cfg(task, play=True)` |
| `rl_cfg` | RL 训练配置：PPO 超参、seed、`experiment_name`/`run_name`（**决定日志目录名**） | `run_train` 拿它建 runner；注意它**不含** reward 权重与 obs 配置 |
| `runner_cls` | 驱动训练循环的 runner 类 | `run_train` L151-153：非 None 就用它，None 回退 `MjlabOnPolicyRunner`；这里复用 `MicroduckOnPolicyRunner`，所以不用自己写 PPO 循环 |

### 2.2 `_REGISTRY` 的底层数据结构（草稿末尾那个问题的答案）

[registry.py](../../../.venv/lib/python3.12/site-packages/mjlab/tasks/registry.py#L10-L40) 全貌就三样东西：

```python
@dataclass
class _TaskCfg:                      # L10-15：一条注册记录 = 4 个字段
    env_cfg: ManagerBasedRlEnvCfg
    play_env_cfg: ManagerBasedRlEnvCfg
    rl_cfg: RslRlBaseRunnerCfg
    runner_cls: type | None

_REGISTRY: dict[str, _TaskCfg] = {}  # L19：模块级全局 dict，task_id -> _TaskCfg

def register_mjlab_task(...):        # L38-40
    if task_id in _REGISTRY:
        raise ValueError(f"Task '{task_id}' is already registered")   # 重复 = 抛异常，不覆盖
    _REGISTRY[task_id] = _TaskCfg(env_cfg, play_env_cfg, rl_cfg, runner_cls)
```

即「注册」= **往一个模块级 dict 里塞一条 4 字段记录**。模块级 = 进程内全局、import 后一直活着，所以任何脚本在任何时机调用 `register_mjlab_task` 都生效。

三个读函数（`load_env_cfg` L48-55 / `load_rl_cfg` L58-63 / `load_runner_cls` L66-71）里前两个返回 **deepcopy**——训练/play 每次拿到的是副本，运行时怎么改 cfg 都不会污染表里原件。这是「组装层拿官方 cfg 逐项覆盖」手法的安全阀：`make_high_kick_env_cfg()` 里 pop/赋值随便改，改的永远是副本或新建对象。

### 2.3 为什么函数开头要 `import mjlab_microduck.tasks`（L39-40）

两个目的，缺一不可：

1. **触发官方任务族注册**：官方 17 个 task 的注册写在 `mjlab_microduck/tasks/__init__.py` 的 import 副作用里（走的也是 `register_mjlab_task`）。不 import 它，`list_tasks()` 里只有你一个人，官方管线里的其他引用也会缺位。
2. **拿 runner 类**：`MicroduckOnPolicyRunner` 从同一个包导出，高踢任务复用它驱动训练。

### 2.4 幂等检查 L44-45 的真实理由

```python
if TASK_ID in list_tasks():
    return
```

`register_high_kick_task()` 被 **train / play 两类入口复用**（play_high_kick.py 直接 import 它），同一进程里可能被调用多次；而 registry 对重复 ID 是**抛 ValueError**（L38-39）不是覆盖——所以这个守卫防的不是「覆盖」，是「第二次调用直接炸」。`list_tasks()` 本身只是 `sorted(_REGISTRY.keys())`（L43-45）。

---

## 3. main() 的 tyro 层（L57-62）：CLI 参数从哪来

```python
args = tyro.cli(
    TrainConfig,
    default=TrainConfig.from_task(TASK_ID),   # train.py L41-43
    prog=f"train_high_kick {TASK_ID}",
    config=mjlab.TYRO_FLAGS,
)
```

- `TrainConfig` 是两层 dataclass：`env: ManagerBasedRlEnvCfg` + `agent: RslRlBaseRunnerCfg`（train.py L26-27）。
- `from_task(TASK_ID)` = `load_env_cfg(task_id)` + `load_rl_cfg(task_id)`（两次 deepcopy）拼出**默认配置**——默认值全部来自注册表，也就是来自你的 `high_kick_cfg.py`。
- `tyro.cli` 把 dataclass 的每个字段递归展开成 CLI flag：`--env.scene.num-envs 512` 的语义 = 「把默认 env_cfg 副本的 `scene.num_envs` 字段改成 512」；`--agent.max-iterations 300` 同理改 rl cfg。没写的字段保持注册表默认。
- 官方 `train.py` 的 main 是「tyro 位置参数选 task_id」；我们把它换成**固定 TASK_ID + 先注册**，其余原样——差异仅此一处。

---

## 4. launch_training → run_train：注册表被消费的后半场

[train.py](../../../.venv/lib/python3.12/site-packages/mjlab/scripts/train.py) 里与注册表相关的四步：

| 行 | 代码 | 说明 |
|---|---|---|
| L108-109 | `ManagerBasedRlEnv(cfg=cfg.env, device=device, render_mode=...)` | env 在此构建；冒烟日志里那串 `[INFO] <RewardManager> contains N active terms` 就是这一步各 manager 初始化打印的 |
| L151-153 | `runner_cls = load_runner_cls(task_id)`；None → `MjlabOnPolicyRunner` | 我们注册时给了 `MicroduckOnPolicyRunner`，走自定义分支 |
| L159-165 | 先写 params yaml 再 `runner_cls(env, agent_cfg, str(log_dir), device)` | 注释写明原因：runner 会**原地改** agent_cfg，所以配置文件要先落盘 |
| L173 | `runner.learn(...)` | 进入 rsl_rl 的 PPO 主循环（`on_policy_runner.py::learn`，见 ball_kick README 7.5 与后续走读篇） |

device/seed 在 L50-60 处理（多 GPU 时按 local_rank 分卡、seed 加 rank 偏移）；日志目录 = `rl_cfg.experiment_name/run_name` + 时间戳，**相对 cwd** 落在 `./logs/rsl_rl/<实验名>/<ts>/`——所以 v4/v5 靠实验名分目录互不覆盖。

---

## 5. 误区清单（对着初稿自核出的三处修正）

| # | 初稿说法 | 代码真相 |
|---|---|---|
| 1 | 「重复注册可能会**覆盖**已有条目」 | registry L38-39 对重复 ID **直接 raise ValueError**；幂等检查防的是异常，不是覆盖 |
| 2 | 「rl_cfg 包括 PPO、**奖励权重、观测配置**等」 | reward 权重在 `env_cfg.rewards`、观测在 `env_cfg.observations`；rl_cfg 只有 PPO 超参/seed/实验名等 |
| 3 | 「play 时通常**减少并行**」 | 并行数 `num_envs` 是 CLI/play 调用方覆盖的字段；`play_env_cfg` 的本质差异是 `play=True` 进官方工厂的 play 分支（关随机化类配置）。练习：自己 `print` 两份 cfg 的 events/curriculum 键差异验证 |

初稿说对的部分：五参数职责表、`import 触发注册 → list_tasks → register → from_task → launch_training` 的五步流程、import 官方包的两个目的——均与源码一致。

---

## 6. 一条命令自测本篇结论

```bash
cd practice/tasks/ball_kick && ../../../.venv/bin/python -c "
import mjlab_microduck.tasks, train_high_kick as T
from mjlab.tasks.registry import list_tasks, _REGISTRY
T.register_high_kick_task(); T.register_high_kick_task()   # 连调两次不报错 = 幂等成立
print('registered:', T.TASK_ID in list_tasks())
e = _REGISTRY[T.TASK_ID]
print('record:', type(e).__name__, '| runner:', e.runner_cls.__name__, '| rewards:', len(e.env_cfg.rewards))
"
```

预期输出：`registered: True` / `record: _TaskCfg | runner: MicroduckOnPolicyRunner | rewards: 12`（v5 模式 12 个 reward 项）。

---

## 7. 下一篇预告

- **02 · high_kick_cfg.py 组装层**：`make_high_kick_env_cfg()` 如何「拿官方 662 行配方逐项覆盖」，`REWARD_MODE` 双模式分支与实验名隔离。
- **03 · high_kick_rewards.py × reward_manager**：一个 reward 函数从 `cfg.rewards` 挂进去，到 `Episode_Reward/<term>` 出现在日志的完整数据流（含 ×dt 时间平均语义）。
- 认知层配套：流程与为什么见 [`03-walkthrough`](../03-walkthrough/training_walkthrough.md)；本任务实战判据见 [ball_kick README](../../../practice/tasks/ball_kick/README.md)。
