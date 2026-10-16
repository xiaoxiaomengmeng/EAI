# 103-lingbot

一句话：Robbyant LingBot 系列（深度补全、视频生成、世界模型、VLA）本地部署；四个子目录均为独立子模块，各用各的环境。

## 启动环境

| 子目录 | 环境 | 激活命令 | 常用入口 |
| --- | --- | --- | --- |
| `lingbot-depth` | conda `lingbot-depth`（Python 3.10，torch 2.11+cu128，mdm 为 editable） | `conda activate lingbot-depth` | `python example.py --example 0` |
| `lingbot-video` | 待定（原 `.venv` 已按用户要求删除，重建步骤见 `README_PRACTICE.md`） | — | `bash scripts/run_demo_5090.sh --low-vram` |
| `lingbot-world-v2` | 待定（原 `.venv` 已按用户要求删除） | — | `run_fast.sh`（默认 8 卡 torchrun） |
| `lingbot-vla-v2` | 待定（本机尚未建环境） | — | 见其 README |

- 权重放在各子目录内：`lingbot-depth/models/`、`lingbot-video/models/`、`lingbot-world-v2/lingbot-world-v2-14b-causal-fast/`
- 激活前先 `cd ~/ws/103-lingbot/<子目录>`

## 从这里开始

- 架构笔记：[docs/01-LingBot-VLA-v2-架构与运行逻辑详解.md](docs/01-LingBot-VLA-v2-架构与运行逻辑详解.md)
- 各子目录 README；lingbot-video 的 5090 实操见 `lingbot-video/README_PRACTICE.md`
