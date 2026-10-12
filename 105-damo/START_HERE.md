# 105-damo

一句话：阿里达摩院（DAMO Academy）Rynn 系列具身模型复现合集——RynnBrain 具身大脑与 RynnWorld-4D 4D 世界模型；两者均为 gitlink 子模块，当前仅克隆源码。

## 内容

| 子项目 | 定位 | 状态 |
|---|---|---|
| `RynnBrain/` | 具身大脑（视觉-语言-动作），自带 cookbooks 教程与在线 Demo | 源码已克隆；权重已在 `~/.cache/modelscope`（RynnBrain-8B / CoP-8B / Nav-8B / Plan-8B 四个 8B），未运行验证 |
| `RynnWorld-4D/` | 面向机器人操作的 4D 具身世界模型（含微调脚本） | 仅源码，权重未下载、未运行验证 |

## 从这里开始

- 上游说明：[RynnBrain/README.md](RynnBrain/README.md)（Project Page / HuggingFace / ModelScope / arXiv 链接）、[RynnWorld-4D/README.md](RynnWorld-4D/README.md)
- 权重：RynnBrain 四个 8B 模型已在 `~/.cache/modelscope/models/DAMO_Academy--RynnBrain-*/`（未入 `~/ws/models/`）；RynnWorld-4D 权重未下载，按其 README 的 ModelScope / HuggingFace 入口获取，统一放 `~/ws/models/`（命名与仓库名一致），项目内软链引用
