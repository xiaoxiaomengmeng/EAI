# 06-topics · 专题深挖

> 认知阶梯第 ⑦ 层：挑几个难点专题深入，并串联仓库自带的**一手案例资料**。

## 这个目录将放什么

- **sim2real 专题**：域随机化（DR）、BAM 执行器拟合、61-D 契约对齐、为什么"仿真训得好、真机会崩"
- **reward 设计专题**：term 分类、两种惩罚风格、符号约定（双重否定 bug）、curriculum 权重调度
- **案例复盘**：把仓库自带的真实调参与踩坑记录串成"一个策略是怎么炼成的"

## 仓库自带的一手案例（配合看）

- [`roller_standup_policy_summary.md`](../../../microduck_rl/docs/roller_standup_policy_summary.md) — 一个策略从设计到调试的完整复盘（含符号 bug）
- [`docs/superpowers/specs/`](../../../microduck_rl/docs/superpowers/specs/) — 每个任务的设计文档
- [`docs/superpowers/plans/`](../../../microduck_rl/docs/superpowers/plans/) — 实现计划（含 iteration 数和验收标准）

## 原则

保留原始资料里的**真实 run 编号、调参推理、踩坑复盘**（如 velocity 的 push 从 ±0.5 改成 ±0.3、roller_standup 的 `gentle_rise` 符号 bug），**不抽象成空泛提纲**——这些"为什么"比"是什么"更值钱。

## 现状

🚧 **待写。** 返回 [总导航 README](../README.md)。
