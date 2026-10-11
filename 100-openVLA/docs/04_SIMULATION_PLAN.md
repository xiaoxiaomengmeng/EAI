# OpenVLA 仿真推理路线规划

本文定义当前工作区从单步推理到 MuJoCo 仿真、LIBERO 评测的学习和实现路线。

## 1. 两条路线的定位

```text
sim      自定义 MuJoCo demo，用于工程链路验证
LIBERO           标准 MuJoCo / robosuite benchmark，用于成功率评测
```

不要混淆二者：

```text
sim 跑通不等于 LIBERO 成功率
```

## 2. 阶段 1：单步推理

目标：验证模型可以加载并输出 7D action。

命令：

```bash
cd /home/creator/ws/100-openVLA
source .venv/bin/activate
python scripts/demo_predict_action.py
```

重点观察：

```text
模型是否成功加载
processor 是否正常
输入图像和 prompt 是否构造成功
action shape 是否为 7D
unnorm_key 是否为 bridge_orig
```

## 3. 阶段 2：sim dry-run

目标：不加载 7B 模型，只验证 MuJoCo、渲染、控制和输出保存。

命令：

```bash
cd /home/creator/ws/100-openVLA
source .venv/bin/activate
export MUJOCO_GL=egl

python -m sim \
  --dry-run \
  --policy-steps 2 \
  --sim-steps-per-policy 2
```

成功后应能看到：

```text
mp4 视频
json 日志
action / target_ctrl / timing 等字段
```

## 4. 阶段 3：OpenVLA base + MuJoCo

目标：加载真实 `openvla-7b` base，完成自定义 MuJoCo 闭环。

命令：

```bash
python -m sim \
  --robot franka \
  --motion-mode ik \
  --policy-steps 20 \
  --sim-steps-per-policy 30 \
  --unnorm-key bridge_orig \
  --attn-implementation sdpa
```

目标不是一次就成功抓取，而是确认：

```text
图像 -> OpenVLA action -> adapter -> MuJoCo ctrl -> 新图像
```

## 5. 阶段 4：IK 与控制适配

目标：让 action 的前三维更合理地驱动末端位置。

已完成能力：

```text
ee_site 插入
mj_jacSite 位置雅可比
阻尼最小二乘 IK
ik / demo / direct 三种 motion_mode
夹爪距离阈值
下压偏移
```

下一步可扩展：

```text
姿态 IK
末端姿态约束
接触检测
grasp success 判定
阶段式抓取策略
```

## 6. 阶段 5：抓取策略与诊断

优先看 JSON 字段：

```text
gripper_cube_distance
cube_pos
ee_site_pos
gripper_state
target_ctrl
ik_info
timing
```

建议按以下顺序优化：

```text
先确认末端能接近 cube
再确认夹爪不会过早闭合
再确认闭合后 cube 有接触或移动
最后再考虑微调模型
```

## 7. 阶段 6：LIBERO 标准评测

目标：跑标准 MuJoCo / robosuite benchmark。

推荐先跑 smoke test：

```bash
cd /home/creator/ws/100-openVLA/repos/openvla
source /home/creator/ws/100-openVLA/.venv/bin/activate
export LIBERO_CONFIG_PATH=/home/creator/ws/100-openVLA/.libero
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

注意：LIBERO 应使用 `openvla-7b-finetuned-libero-*`，不要直接用 base `openvla-7b`。

## 8. 当前推荐学习顺序

```text
1. scripts/demo_predict_action.py 单步推理
2. docs/02_ACTION_SPACE_AND_CONTROL.md 理解 action 与 unnorm_key
3. sim dry-run
4. sim + openvla-7b base
5. docs/03_MUJOCO_SIM_NOTES.md 理解 MuJoCo 和 IK
6. LIBERO smoke test
7. docs/07_TRAINING_AND_FINETUNING.md 理解训练和微调
```
