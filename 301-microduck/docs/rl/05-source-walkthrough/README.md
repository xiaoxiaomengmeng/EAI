# 05-source-walkthrough · 源码端到端走读

> 认知阶梯第 ⑥ 层：选一条端到端路径，从入口逐函数跟到底，把 `04-modules` 里散落的模块**串成一条线**。

## 这个目录将放什么

一篇 **velocity end-to-end 源码走读**：跟着一条真实命令——

```bash
uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096
```

从入口一路读到产出 checkpoint，逐个文件、逐个关键函数地跟：

```text
entry point (tasks/__init__.py 的 import 副作用)
  → train_cli.py → mjlab runner 构建
  → make_microduck_velocity_env_cfg() 组装 env（往各 manager 塞 term）
  → 7 个 manager：obs / action / reward / termination / event / command / curriculum
  → mdp.py 里对应的 term 函数
  → rsl_rl 的 PPO 训练循环
  → checkpoint 落盘 output/velocity/<时间戳>/model_<iter>.pt
```

## 和 03-walkthrough 的区别（别混淆）

| | 讲什么 | 视角 |
|---|---|---|
| [`03-walkthrough`](../03-walkthrough/training_walkthrough.md) | **流程和为什么**（8 个阶段） | 偏认知，命令细节指向 quickstart |
| `05-source-walkthrough`（本目录） | **源码怎么串**（逐函数调用链） | 偏代码，跟着一条命令读到底 |

## 现状

- ✅ [01-train_high_kick-walkthrough.md](01-train_high_kick-walkthrough.md)：以 practice 侧实测跑通的 high kick 入口为标本，从 `python train_high_kick.py` 逐行跟到注册表/tyro/launch_training（含 `_REGISTRY` 数据结构与误区清单）。
- 🚧 待写：velocity end-to-end 走读（上文规划）；02 high_kick_cfg 组装层、03 reward 数据流（见 01 篇末预告）。

返回 [总导航 README](../README.md)。
