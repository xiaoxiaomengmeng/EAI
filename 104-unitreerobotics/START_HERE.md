# 104-unitreerobotics

一句话：宇树 UnifoLM-WMA-0（世界模型 + 动作）在 RTX 5090 上的本地交互推理与学习笔记。

## 启动环境

```bash
cd ~/ws/104-unitreerobotics
source .venv/bin/activate      # Python 3.10，torch 2.11.0+cu128，unifolm_wma / dlimp 为 editable
```

- 源码：`repos/unifolm-world-model-action`（fork，含 `external/dlimp` 子模块）
- 权重：`~/.cache/modelscope/models/unitreerobotics--UnifoLM-WMA-0-Dual/snapshots/master/unifolm_wma_dual.ckpt`（16.7G，不在 ~/ws/models）
- 注：环境里没装 `tensorflow_datasets`，`import dlimp` 会失败；推理流程不用 dlimp，不影响使用

## 从这里开始

- 详细说明：[README.md](README.md)（环境初始化、推理参数、5090 适配记录）→ [docs/01-UnifoLM-WMA学习笔记.md](docs/01-UnifoLM-WMA学习笔记.md)（原理与输入输出）
- 常用入口（无需先激活环境）：
  - `bash scripts/run_z1_dual_arm_stackbox.sh`（另有 `run_g1_pack_camera.sh` 等，按数据集各一个）
  - `bash scripts/run_wma_interaction.sh <dataset>`，结果写入 `outputs/interaction/<dataset>/`
