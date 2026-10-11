# 102-cosmos

一句话：NVIDIA Cosmos 世界模型 / 策略模型（Cosmos-Predict2.5-2B、Cosmos3-Edge、Cosmos3-Edge-Policy-DROID）的本地部署与验证。

## 启动环境

两套环境，按用途选择：

```bash
# 1) conda：本目录脚本（diffusers 加载模型）
conda activate cosmos                 # Python 3.12，torch 2.9.1+cu128，diffusers 0.40，transformers 5.x
cd ~/ws/102-cosmos

# 2) uv venv：cosmos-framework 官方脚本（policy server 等）
cd ~/ws/102-cosmos/3rd/cosmos-framework
source .venv/bin/activate             # Python 3.13，torch 2.10.0+cu128，cosmos-framework 为 editable
```

- 模型权重：`~/ws/models/{Cosmos-Predict2.5-2B,Cosmos3-Edge,Cosmos3-Edge-Policy-DROID}`
- 源码：`3rd/cosmos-framework`（server）、`3rd/RoboLab`（仿真评测 client）

## 从这里开始

- 详细说明：[README.md](README.md)（模型、仓库、环境配置、输入输出规范）
- 常用入口：
  - `python scripts/verify_models.py`（conda `cosmos`，验证模型文件与加载）
  - `python -m cosmos_framework.scripts.action_policy_server_robolab --help`（uv venv）
