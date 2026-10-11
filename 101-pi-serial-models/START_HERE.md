# 101-pi-serial-models

一句话：基于 lerobot v0.6.2 的 Pi0 / Pi0.5 / ACT 策略训练与仿真评测（ALOHA 双臂 + LIBERO）。

## 启动环境

```bash
cd ~/ws/101-pi-serial-models
source .venv312/bin/activate          # Python 3.12，torch 2.11.0+cu130，lerobot 为 editable（repos/lerobot-main）
export MUJOCO_GL=egl                  # 无显示器渲染（评测脚本默认已设置）
```

- 基座权重：`~/ws/models/{pi0,pi05}_{base,libero_base}`
- LIBERO 读取全局配置 `~/.libero/config.yaml`，assets 在 `~/.libero/assets`

## 从这里开始

- 详细说明：[README.md](README.md)（脚本导航、实测结果、已知坑），[docs/](docs/)（17~25 号指南）
- 常用入口：`bash scripts/31_eval_act.sh`（ACT）→ `bash scripts/33_eval_pi05_libero.sh`（pi0.5 LIBERO）
