# 100-openVLA

一句话：OpenVLA 7B 视觉-语言-动作模型，按「单步推理 → LIBERO 评测 → Franka MuJoCo 闭环」三阶验证。

## 启动环境

```bash
cd ~/ws/100-openVLA
source .venv/bin/activate                       # Python 3.10
export LIBERO_CONFIG_PATH=$PWD/.libero          # 用项目自己的 LIBERO 配置，不用 ~/.libero
```

## 从这里开始

- 详细说明：[README.md](README.md)、[docs/](docs/)
- 常用入口：`scripts/01_openvla_verify.py`（单步）→ `scripts/02_libero_eval.py`（LIBERO）→ `python -m sim`（闭环）
