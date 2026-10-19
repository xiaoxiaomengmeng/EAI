# 104-unitreerobotics

一句话：宇树双子项目 —— UnifoLM-WMA-0（世界模型+动作）交互推理 与 unitree_rl_mjlab（mjlab+PPO）RL 运动控制训练，均在 RTX 5090 上。

## 启动环境

两个子项目环境相互独立，不可混用：

```bash
# A. UnifoLM-WMA-0 世界模型推理
cd ~/ws/104-unitreerobotics
source .venv/bin/activate      # Python 3.10，torch 2.11.0+cu128，unifolm_wma / dlimp 为 editable

# B. unitree_rl_mjlab RL 训练
conda activate unitree_rl_mjlab
# Python 3.11，torch 2.9.1+cu128，mjlab 1.2.0；命令须在 repos/unitree_rl_mjlab 下执行
```

- **A（世界模型推理）**
  - 源码：`repos/unifolm-world-model-action`（fork，含 `external/dlimp` 子模块）
  - 权重：`~/.cache/modelscope/models/unitreerobotics--UnifoLM-WMA-0-Dual/snapshots/master/unifolm_wma_dual.ckpt`（16.7G，不在 ~/ws/models）
  - 注：环境里没装 `tensorflow_datasets`，`import dlimp` 会失败；推理流程不用 dlimp，不影响使用
- **B（RL 训练）**
  - 源码：`repos/unitree_rl_mjlab`（上游克隆，不改动；环境搭建/验证判据/踩坑见 [docs/02-unitree_rl_mjlab环境搭建与验证.md](docs/02-unitree_rl_mjlab环境搭建与验证.md)）

## 从这里开始

- **UnifoLM-WMA-0**：[README.md](README.md)（环境初始化、推理参数、5090 适配记录）→ [docs/01-UnifoLM-WMA学习笔记.md](docs/01-UnifoLM-WMA学习笔记.md)（原理与输入输出）
  - 常用入口（无需先激活环境）：
    - `bash scripts/run_z1_dual_arm_stackbox.sh`（另有 `run_g1_pack_camera.sh` 等，按数据集各一个）
    - `bash scripts/run_wma_interaction.sh <dataset>`，结果写入 `outputs/interaction/<dataset>/`
- **unitree_rl_mjlab**：[docs/02-unitree_rl_mjlab环境搭建与验证.md](docs/02-unitree_rl_mjlab环境搭建与验证.md)（已验证安装命令、三判据、踩坑记录、训练与 play 命令）
