# LIBERO 评测笔记

本文记录 OpenVLA 工作区中运行 LIBERO 本地 MuJoCo / robosuite 仿真的关键步骤、模型选择和常见问题。

## 1. LIBERO 是什么

LIBERO 是机器人操作仿真 benchmark，底层基于：

```text
MuJoCo + robosuite
```

它测试机器人策略能否在仿真环境中完成抓取、移动、放置、空间关系和长程组合任务。

LIBERO 评测是闭环控制：

```text
图像观测 + 指令
  -> OpenVLA action
  -> 环境 step
  -> 新观测
  -> 下一步 action
  -> 成功率统计
```

## 2. 官方入口

OpenVLA 官方 LIBERO 入口：

```text
repos/openvla/experiments/robot/libero/run_libero_eval.py
```

相关工具：

```text
repos/openvla/experiments/robot/libero/libero_utils.py
repos/openvla/experiments/robot/openvla_utils.py
repos/openvla/experiments/robot/robot_utils.py
```

## 3. 任务套件

官方脚本支持：

```text
libero_spatial
libero_object
libero_goal
libero_10
libero_90
```

常用对应关系：

```text
libero_spatial -> openvla/openvla-7b-finetuned-libero-spatial
libero_object  -> openvla/openvla-7b-finetuned-libero-object
libero_goal    -> openvla/openvla-7b-finetuned-libero-goal
libero_10      -> openvla/openvla-7b-finetuned-libero-10
```

## 4. 为什么 base 模型不能直接跑 LIBERO

`run_libero_eval.py` 会强制设置：

```python
cfg.unnorm_key = cfg.task_suite_name
```

例如跑 `libero_spatial` 时，会要求模型包含：

```text
model.norm_stats["libero_spatial"]
```

base `openvla-7b` 更适合 `bridge_orig`，通常不包含 `libero_*` key。因此直接使用 base 模型会出现：

```text
Action un-norm key libero_spatial not found in VLA norm_stats
```

即使修改代码绕过检查，也不是标准 LIBERO 评测，因为动作统计和模型微调分布不匹配。

## 5. 安装依赖

激活环境：

```bash
cd /home/creator/ws/openVLA
source .venv/bin/activate
```

安装 OpenVLA 源码依赖：

```bash
cd /home/creator/ws/openVLA/repos/openvla
python -m pip install -r requirements-min.txt
python -m pip install -e . --no-deps
```

注意必须使用 `--no-deps`，避免覆盖当前 RTX 5090 已验证的 PyTorch / CUDA 环境。

安装评测脚本依赖：

```bash
python -m pip install \
  draccus==0.8.0 \
  wandb \
  json-numpy \
  jsonlines \
  peft==0.11.1 \
  rich \
  tensorflow==2.15.0 \
  tensorflow_datasets==4.9.3 \
  tensorflow_graphics==2021.12.3
```

安装 LIBERO 额外依赖：

```bash
python -m pip install -r experiments/robot/libero/libero_requirements.txt
python -m pip install numpy==1.26.4 opencv-python==4.10.0.84
```

## 6. LIBERO 源码配置

如果 `/home/creator/ws/openVLA/repos/LIBERO` 已存在：

```bash
cd /home/creator/ws/openVLA/repos/LIBERO
source /home/creator/ws/openVLA/.venv/bin/activate
python -m pip install -e .
```

配置路径：

```bash
mkdir -p /home/creator/ws/openVLA/.libero

cat > /home/creator/ws/openVLA/.libero/config.yaml <<'YAML'
assets: /home/creator/ws/openVLA/repos/LIBERO/libero/libero/assets
bddl_files: /home/creator/ws/openVLA/repos/LIBERO/libero/libero/bddl_files
benchmark_root: /home/creator/ws/openVLA/repos/LIBERO/libero/libero
datasets: /home/creator/ws/openVLA/repos/LIBERO/libero/datasets
init_states: /home/creator/ws/openVLA/repos/LIBERO/libero/libero/init_files
YAML
```

运行前设置：

```bash
export LIBERO_CONFIG_PATH=/home/creator/ws/openVLA/.libero
export MUJOCO_GL=egl
```

## 7. 最小 smoke test

推荐先跑 1 个 trial，避免默认完整评测过久。

```bash
cd /home/creator/ws/openVLA/repos/openvla
source /home/creator/ws/openVLA/.venv/bin/activate
export LIBERO_CONFIG_PATH=/home/creator/ws/openVLA/.libero
export MUJOCO_GL=egl

python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --center_crop True \
  --num_trials_per_task 1 \
  --use_wandb False \
  --run_id_note smoke-test
```

## 8. 常用参数

```text
--model_family openvla
--pretrained_checkpoint <checkpoint>
--task_suite_name <suite>
--center_crop True
--num_trials_per_task 1
--use_wandb False
--run_id_note smoke-test
--seed 7
--load_in_8bit False
--load_in_4bit False
```

## 9. 注意事项

`center_crop True` 很重要。官方 LIBERO 微调 checkpoint 通常使用图像增强训练，评测时应保持输入分布一致。

首次运行 Hugging Face checkpoint 会自动下载。如果网络不可用，需要提前准备本地 checkpoint，并把 `--pretrained_checkpoint` 改成本地路径。

默认完整评测很慢。LIBERO-Spatial 等任务集可能会运行多个任务，每个任务默认 50 个 episode，建议先 smoke test。

如果无显示器或远程服务器运行，设置：

```bash
export MUJOCO_GL=egl
```
