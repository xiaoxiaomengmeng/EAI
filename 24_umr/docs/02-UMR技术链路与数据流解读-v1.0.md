# UMR 技术链路与数据流解读（人体动作 → 点云对应 → 机器人 qpos）

> 版本 v1.0 · 2026-09-12 · 对应代码：[UMR](https://github.com/hanyang9/UMR)（arXiv 2609.02134）
>
> 配套文档：[01-UMR安装部署指南-v1.0.md](01-UMR安装部署指南-v1.0.md)。本文回答一个问题：**跑 `humanoid_retarget_pipeline.py` 的那几分钟里，数据到底经历了什么，最后那个 npz 里装的是什么**。以本机实测的 LAFAN1 跳舞 → Unitree G1 为例。


## 运行日志（本次实测）

```text
python scripts/humanoid_retarget_pipeline.py \
  --config robot_configs/humanoid_retarget_unitree_g1_example.json

  cd ~/ws/24_umr/UMR
python scripts/humanoid_retarget_pipeline.py \
  --config robot_configs/humanoid_retarget_unitree_g1_example.json
[SMPLXModel] loading format=pkl path=/home/creator/ws/24_umr/UMR/smpl/SMPLX_NEUTRAL.pkl
[Dataset] Added smplx_neutral_betas_c4eb7419e9: verts=10475 faces=20908 center=spine1 joint_id=3 offset=[7.578963413834572e-05, -0.24949052929878235, -0.019167549908161163]
[Dataset] Added unitree_g1: verts=192938 faces=385626 exterior_surface=True exterior_method=first_hit
[HumanoidPipeline] saved correspondence dataset: /home/creator/ws/24_umr/UMR/data/correspondence_unitree_g1_smplx_neutral_betas_c4eb7419e9.npz names=['smplx_neutral_betas_c4eb7419e9', 'unitree_g1'] points=(2, 4096, 3)
[HumanoidPipeline] reuse correspondence dataset: /home/creator/ws/24_umr/UMR/data/correspondence_unitree_g1_smplx_neutral_betas_c4eb7419e9.npz
[HumanoidPipeline] /home/creator/miniconda3/envs/umr/bin/python /home/creator/ws/24_umr/UMR/scripts/train_correspondence_template_residual_ae.py --data /home/creator/ws/24_umr/UMR/data/correspondence_unitree_g1_smplx_neutral_betas_c4eb7419e9.npz --out-dir /home/creator/ws/24_umr/UMR/output/correspondence_unitree_g1_smplx_neutral_betas_c4eb7419e9 --template-name smplx_neutral_betas_c4eb7419e9 --template-sort y_z_x --num-points 4096 --normalize per_sample_height --epochs 500 --batch-size 2 --lr 0.001 --lr-scheduler cosine --min-lr 1e-05 --chamfer-weight 1.0 --repulsion-weight 0.002 --repulsion-k 8 --repulsion-radius 0.035 --residual-weight 0.0 --edge-weight 0.4 --edge-graph geodesic --edge-k 32 --noise-std 0.002 --dropout-ratio 0.0 --resample-every 0 --log-every 100 --save-every 1000 --seed 0 --device auto
[TrainResidual] data=/home/creator/ws/24_umr/UMR/data/correspondence_unitree_g1_smplx_neutral_betas_c4eb7419e9.npz samples=2 names=['smplx_neutral_betas_c4eb7419e9', 'unitree_g1']
[TrainResidual] template=smplx_neutral_betas_c4eb7419e9 sort=y_z_x
[TrainResidual] device=cuda num_points=4096 out=/home/creator/ws/24_umr/UMR/output/correspondence_unitree_g1_smplx_neutral_betas_c4eb7419e9
[TrainResidual] normalize=per_sample_height
[TrainResidual] training_mode=all_samples
[TrainResidual] normalization_scales smplx_neutral_betas_c4eb7419e9=1.8482, unitree_g1=1.3228
[TrainResidual] per_sample_heights smplx_neutral_betas_c4eb7419e9=1.848237, unitree_g1=1.322845
[TrainResidual] chamfer_weight=1.0 repulsion_weight=0.002 residual_weight=0.0 edge_weight=0.4 edge_graph=geodesic repulsion_k=8 repulsion_radius=0.035 edge_k=32 edge_count=131072
[TrainResidual] epoch=00001 loss=0.002756 chamfer=0.001105 repulsion=0.001651 residual=0.000000 edge=0.000000 lr=0.00099999
[TrainResidual] epoch=00100 loss=0.001674 chamfer=0.000107 repulsion=0.001484 residual=0.000000 edge=0.000084 lr=0.00090546
[TrainResidual] epoch=00200 loss=0.001667 chamfer=0.000111 repulsion=0.001467 residual=0.000000 edge=0.000088 lr=0.00065796
[TrainResidual] epoch=00300 loss=0.001662 chamfer=0.000111 repulsion=0.001471 residual=0.000000 edge=0.000081 lr=0.00035204
[TrainResidual] epoch=00400 loss=0.001660 chamfer=0.000110 repulsion=0.001465 residual=0.000000 edge=0.000085 lr=0.00010454
[TrainResidual] epoch=00500 loss=0.001660 chamfer=0.000111 repulsion=0.001465 residual=0.000000 edge=0.000084 lr=0.00001000
[TrainResidual] Saved final checkpoint and correspondence slots to /home/creator/ws/24_umr/UMR/output/correspondence_unitree_g1_smplx_neutral_betas_c4eb7419e9
[HumanoidPipeline] retarget motion seq_key='dance1_subject2' seq_index=0 frames=0:-1:1 max_frames=0
[HumanoidPipeline] /home/creator/miniconda3/envs/umr/bin/python /home/creator/ws/24_umr/UMR/scripts/retarget_smpl_to_humanoid_surface_vector.py --config /home/creator/ws/24_umr/UMR/robot_configs/humanoid_retarget_unitree_g1_example.json --data /home/creator/ws/24_umr/UMR/sample_data/lafan1_smplx --seq-index 0 --start 0 --end -1 --stride 1 --max-frames 0 --seq-key dance1_subject2 --ground-contact-anchor-cost 5.0 --ground-penetration-hard-constraint --ground-penetration-hard-constraint-mode surface_slots_all --ground-penetration-margin 0.0 --ground-penetration-hard-slack-cost 1000.0 --ground-penetration-threshold 0.05 --ground-penetration-max-points 128 --robot-self-penetration-margin -0.001 --robot-self-penetration-hard-slack --robot-self-penetration-hard-slack-cost 1000.0 --step-limit-mode l2 --max-dq 0.15 --global-step-size 0.4 --root-step-limit-mode off --root-max-translation-dq 0.3 --root-max-rotation-dq 0.5 --root-global-translation-step-size 0.3 --root-global-rotation-step-size 0.5 --trajectory-filter-mode lqr --trajectory-filter-data-cost 1.0 --trajectory-filter-velocity-cost 0.0 --trajectory-filter-acceleration-cost 0.1 --trajectory-filter-jerk-cost 0.01 --trajectory-filter-root-translation --trajectory-filter-anchor-start-frames 1 --trajectory-filter-anchor-end-frames 1 --trajectory-warm-start-mode sequential --trajectory-warm-start-bidirectional-continuity-cost 0.1 --trajectory-warm-start-bidirectional-switch-cost 0.01 --slots /home/creator/ws/24_umr/UMR/output/correspondence_unitree_g1_smplx_neutral_betas_c4eb7419e9/correspondence_slots_final.npz --out /home/creator/ws/24_umr/UMR/output/unitree_g1_retarget/dance1_subject2_smplx_unitree_g1.npz --slots-field reconstructed_slots
[HumanoidRetarget] body_segment_module=retarget_body_segment_surface schema=smplx_55_to_19_upper_arm_v1 segments=19
[HumanoidRetarget] source=/home/creator/ws/24_umr/UMR/sample_data/lafan1_smplx format=smplx_npz seq=dance1_subject2 frames=3945 fps=30.000
[SMPLXModel] loading format=pkl path=/home/creator/ws/24_umr/UMR/smpl/SMPLX_NEUTRAL.pkl
[SurfaceRetarget][SMPLX] device=cuda:0 batch_size=3945 frames=3945
[SMPLXModel] loading format=pkl path=/home/creator/ws/24_umr/UMR/smpl/SMPLX_NEUTRAL.pkl
[SurfaceRetarget][SMPLXCache] build gender=neutral batch_size=3945 device=cuda:0
[HumanoidRetarget] inferred robot_height=1.32230 from slots sample=unitree_g1 field=target_points mode=y_extent
[HumanoidRetarget] source slots=smplx_neutral_betas_c4eb7419e9 center_mode=spine1 template_center=[7.578963413834572e-05, -0.24949052929878235, -0.019167549908161163]
[HumanoidRetarget] bound SMPL slots: slots=4096, error_mean=0.01023, p95=0.02638
[HumanoidRetarget] smplx_55_to_19_upper_arm_v1 upper-arm split ratio={'leftArm': {'target': 'leftUpperArm', 'ratio_start': 0.0, 'ratio_end': 0.3}, 'rightArm': {'target': 'rightUpperArm', 'ratio_start': 0.0, 'ratio_end': 0.3}}; leftUpperArm=54, leftArm=102, rightUpperArm=57, rightArm=102
[HumanoidRetarget] SMPLX_55_TO_17_FACE slot part labels from smplx_55_to_19_upper_arm_v1: {'rightHand': 84, 'rightUpLeg': 383, 'leftArm': 102, 'head': 328, 'leftLeg': 287, 'leftFoot': 120, 'torso': 723, 'rightFoot': 120, 'rightArm': 102, 'leftHand': 87, 'rightLeg': 305, 'leftForeArm': 140, 'rightForeArm': 137, 'leftUpLeg': 397, 'hips': 509, 'leftShoulder': 74, 'rightShoulder': 87, 'leftUpperArm': 54, 'rightUpperArm': 57}; map=embedded_55_to_17_map; segm=/home/creator/ws/24_umr/UMR/assets/smplx_parts_segm.pkl
[HumanoidRetarget] human_height=1.84824 robot_height=1.32230 smpl_scale=0.715439 ground_z=0.00000 source_ground_align=none source_template=smplx_neutral_betas_c4eb7419e9
[SurfaceRetarget][GroundContactMap] slots=4096, frames=3945, min=0.0000, p5=0.0927, p50=0.6970, max=1.6369, snap<0.0050m=96580/16158720
[HumanoidRetarget] SMPLX smplx_55_to_19_upper_arm_v1 body segment slots: rightHand=84, rightUpLeg=383, leftArm=102, head=328, leftLeg=287, leftFoot=120, torso=723, rightFoot=120, rightArm=102, leftHand=87, rightLeg=305, leftForeArm=140, rightForeArm=137, leftUpLeg=397, hips=509, leftShoulder=74, rightShoulder=87, leftUpperArm=54, rightUpperArm=57
[HumanoidRetarget] selected segment slots: total=456, rightHand=15/15, rightUpLeg=25/25, leftArm=15/15, head=20/20, leftLeg=20/20, leftFoot=30/30, torso=50/50, rightFoot=30/30, rightArm=15/15, leftHand=15/15, rightLeg=20/20, leftForeArm=8/8, rightForeArm=8/8, leftUpLeg=25/25, hips=40/40, leftShoulder=30/30, rightShoulder=30/30, leftUpperArm=30/30, rightUpperArm=30/30
[HumanoidRetarget][GroundPenetration] hard_constraint=true mode=surface_slots_all margin=0.0000 slack=False slack_cost=1000.0000 threshold=0.0500 max_points=128 candidate_slots=4096
[HumanoidRetarget][GroundContactMap] active selected slots under threshold=0.1000: min=0, mean=49.75, max=67, active_frames=3940/3945, selected_slots=456, max_points=128
[HumanoidRetarget][GroundContactAnchor] snapped-to-zero selected slots: min=0, mean=5.34, max=24, active_frames=2889/3945, cost=5.0000
[HumanoidRetarget][SurfacePoint] segment costs rightHand=10.0000, rightUpLeg=10.0000, leftArm=10.0000, head=1.0000, leftLeg=10.0000, leftFoot=10.0000, torso=20.0000, rightFoot=10.0000, rightArm=10.0000, leftHand=10.0000, rightLeg=10.0000, leftForeArm=10.0000, rightForeArm=10.0000, leftUpLeg=10.0000, hips=20.0000, leftShoulder=0.0000, rightShoulder=0.0000, leftUpperArm=0.0000, rightUpperArm=0.0000; active_slots=3824/4096
[HumanoidRetarget][SurfaceNormal] segment costs rightHand=1.0000, rightUpLeg=5.0000, leftArm=5.0000, head=1.0000, leftLeg=5.0000, leftFoot=1.0000, torso=5.0000, rightFoot=1.0000, rightArm=5.0000, leftHand=1.0000, rightLeg=5.0000, leftForeArm=5.0000, rightForeArm=5.0000, leftUpLeg=5.0000, hips=0.0000, leftShoulder=0.5000, rightShoulder=0.5000, leftUpperArm=0.1000, rightUpperArm=0.1000; active_slots=3587/4096
[HumanoidRetarget][SelfContactMap] selected_slots=456, candidate_pairs=83199, threshold=0.1000, active_pairs min=147, mean=248.75, max=256, max_pairs=256
[HumanoidRetarget][SelfContactMap] mode=source_clearance_body_pair_topk, selected_slots=456, candidate_pairs=83199, body_pairs=151, active_pairs min=151, mean=151.00, max=151, body_topk={"default": 1}, default_body_topk=1, rank_weight_range=[1.0000, 0.1000]
[HumanoidRetarget][SelfContactMap] merged modes=['threshold_global_topk', 'source_clearance_body_pair_topk'] pairs min=0, mean=389.12, max=401
[HumanoidRetarget][JointLimit] configured=8 matched_same_name=8 scalar_joints=29
[HumanoidRetarget][StepLimit] dof_max_dq_box configured=0 matched_same_name=0
[HumanoidRetarget] robot slot bind pose=tpose qpos_joints=4 source_model=smplx
[HumanoidRetarget] bound robot slots: slots=4096, visual_geoms=37, point_cloud_center=body:waist_yaw_link
[HumanoidRetarget][SurfaceNormal] reference-pose normal offset magnitude: mean=0.71032, p95=1.91816, max=1.99992
[HumanoidRetarget] scalar_joints=29 robot_xml=/home/creator/ws/24_umr/UMR/assets/g1/dance1_subject2_smplx_unitree_g1.floating_mjcf.xml
[HumanoidRetarget] retarget forward: 100%|█████████████████████████████████████████████████████████████████| 3945/3945 [00:30<00:00, 128.17it/s]
[HumanoidRetarget][TrajectoryFilter] mode=lqr data=1 vel=0 acc=0.1 jerk=0.01 root_translation=True anchor_start=1 anchor_end=1 step_p95 0.149144->0.144822 accel_p95 0.0807837->0.0699183 jerk_p95 0.0888602->0.0604294
[HumanoidRetarget] saved /home/creator/ws/24_umr/UMR/output/unitree_g1_retarget/dance1_subject2_smplx_unitree_g1.npz qpos=(3945, 36) cost mean=0.065865 max=0.231606
[HumanoidPipeline] /home/creator/miniconda3/envs/umr/bin/python /home/creator/ws/24_umr/UMR/scripts/visualize_robot_retarget_result.py --result /home/creator/ws/24_umr/UMR/output/unitree_g1_retarget/dance1_subject2_smplx_unitree_g1.npz --play --rate-limit --viewer-backend glfw-ui --ghost-interval 1.0 --source-slot-offset -2.0 0.0 0.0 --no-show-source-slots --no-show-robot-slots --no-show-ground-contact-map
[RetargetVis] collision geoms for rendering: hidden, count=48
[RetargetVis] result=/home/creator/ws/24_umr/UMR/output/unitree_g1_retarget/dance1_subject2_smplx_unitree_g1.npz, frames=3945/3945, fps=30.000
[RetargetVis] xml=/home/creator/ws/24_umr/UMR/assets/g1/dance1_subject2_smplx_unitree_g1.floating_mjcf.xml, model.nq=36, qpos_width=36
[RetargetVis] seq=dance1_subject2
[RetargetVis] GLFW UI viewer: panel_width=320, target_fps=30.000 wall-clock playback; ghost_interval=1.0s; drag frame slider; Space play/pause; G ghost trail; C clear ghosts; A/D step; Q/E seek; [/] speed.
[RetargetVis] started recording /home/creator/ws/24_umr/UMR/visualizations/viewer_captures/recording_20260912_205824_853.mp4 (2560x1440, fps=60.000)
[RetargetVis] finishing recording /home/creator/ws/24_umr/UMR/visualizations/viewer_captures/recording_20260912_205824_853.mp4 (17726 frames queued, 2560x1440)
[RetargetVis] saved recording /home/creator/ws/24_umr/UMR/visualizations/viewer_captures/recording_20260912_205824_853.mp4 (17726/17726 frames written, dropped=105, 2560x1440)
[RetargetVis] started recording /home/creator/ws/24_umr/UMR/visualizations/viewer_captures/recording_20260912_210034_830.mp4 (2560x1440, fps=60.000)
[RetargetVis] finishing recording /home/creator/ws/24_umr/UMR/visualizations/viewer_captures/recording_20260912_210034_830.mp4 (18093 frames queued, 2560x1440)
[RetargetVis] saved recording /home/creator/ws/24_umr/UMR/visualizations/viewer_captures/recording_20260912_210034_830.mp4 (18093/18093 frames written, dropped=387, 2560x1440)
```

---

## 〇·零、按安装指南的五个阶段逐项追踪这次运行

安装指南把完整数据链路概括为 ① adapter → ② canonical 采样 → ③ 对应学习 → ④ 逐帧重定向 → ⑤ 平滑/保存/回放。需要先区分：**这条 Quick Start 命令在进程内调度的是 `build → train → retarget → view` 四个阶段；安装指南的阶段①是它的前置数据准备，阶段⑤则由 `retarget` 的尾部平滑/保存与 `view` 共同完成。**

### 0. 运行前已经就位的四类输入

| 输入 | 本次实际位置与格式 | 从哪里来 | 在何时使用 |
|------|-------------------|----------|------------|
| 人体动作 | `sample_data/lafan1_smplx/dance1_subject2.npz`；NumPy `.npz` 压缩归档 | 原始 LAFAN1 是 BVH 动捕；仓库按 `lafan_to_smplx` 工作流预转换并随示例提供这一个 `.npz` | 阶段①用它的体型选择人体模板；阶段④读取整段动作 |
| 人体模型 | `smpl/SMPLX_NEUTRAL.pkl` | SMPL-X 官方 body model，安装时由用户放入 | 阶段②产生 canonical 人体网格；阶段④按每帧姿态生成运动人体网格 |
| 机器人模型 | `assets/g1/g1_29dof.xml`；MJCF/XML，引用 G1 mesh | 仓库 LFS 资产 | 阶段②采样 G1 外表面，阶段④求解，阶段⑤回放 |
| 配置 | 默认 `humanoid_retarget_defaults.json` + 覆盖文件 `robot_configs/humanoid_retarget_unitree_g1_example.json`；JSON | 仓库自带 | 选择动作、采样/训练/优化参数，以及 G1 的 T-pose 和关节限位 |

配置加载器会深度合并两个 JSON。默认配置指定 `motion.data=sample_data/lafan1_smplx`、`motion.seq_key=dance1_subject2`、4096 个采样点、500 个训练 epoch 与 viewer 参数；G1 文件只补充机器人名称、MJCF、点云中心、4 个 T-pose 关节值和 8 条额外关节限位。因此日志中的长命令并不是用户手写出来的，而是 pipeline 将合并后的配置展开后传给子脚本。

### ① 数据源 adapter：本次没有在 pipeline 内执行

原始 LAFAN1 数据是 BVH 骨骼动画，UMR 的这个入口**不会直接读取 BVH**。它只接受能产生 SMPL-X 表面的动作序列；原始 BVH 必须先在命令外经 `lafan_to_smplx` 等 adapter 转换。示例仓库已经准备好了转换结果，所以本次日志从人体模型/点云构建开始，没有出现 BVH 解析或转换命令。

本机实际读取的 `dance1_subject2.npz` 内部字段如下：

| 字段 | 实际形状与类型 | 语义 | loader 中的标准化结果 |
|------|----------------|------|------------------------|
| `poses` | `(3945, 55, 3)`，`float32` | 3945 帧、55 个 SMPL-X 关节的轴角旋转；单位 rad | 展平为 `pose_aa=(3945, 165)` |
| `trans` | `(3945, 3)`，`float32` | 每帧根节点的世界平移；单位 m | `trans_orig=(3945, 3)` |
| `betas` | `(16,)`，`float32` | 整段动作共享的人体体型系数 | `beta=(16,)`，决定 canonical 模板 |
| `gender` | 标量字符串 `neutral` | 选择 SMPL-X 中性模型 | `gender=neutral` |
| `mocap_frame_rate` | 标量 `int64`，值 `30` | 原始动作帧率 | `fps=30.0` |

#### 如何理解一帧的 SMPL-X 关节姿态

可以把这段数据理解为共有 **3945 帧**。第 `t` 帧含有：

```text
poses[t]：55 个关节 × 每关节 3 个轴角旋转值 = 165 个数
trans[t]：人体根节点在世界坐标中的 x、y、z 平移 = 3 个数
合计：168 个随时间变化的运动参数
```

这里每个关节的 3 个旋转数值是**轴角（axis-angle）**，而不是 `roll / pitch / yaw` 欧拉角。若轴角向量为 `r=[rx, ry, rz]`，其长度 `||r||` 是旋转角度（rad），归一化方向 `r / ||r||` 是旋转轴。因此，一个三元组表达的是“绕某个三维方向旋转多少”，可避免直接使用欧拉角时的万向节锁问题。

`poses` 的首个三元组是根节点的全局朝向；其余关节存储相对于父关节的**局部旋转**。非根关节的相对平移并不逐帧写入动作 `.npz`，而是由 SMPL-X 模型文件中的静态骨架偏移（rest offset / 骨长）给出，且会随 `betas` 指定的体型而改变。于是：

```text
根节点世界位姿 = trans[t]（世界 xyz 平移）+ poses[t, 0]（全局轴角旋转）
子关节世界变换 = 父关节世界变换 × 静态骨架偏移 × 本关节局部轴角旋转
```

SMPL-X 的前向运动学（FK）按骨架层级递推这些变换，得到所有关节的世界位置和姿态；再结合蒙皮权重，把 10475 个模板顶点变形成该帧的人体网格。也就是说，跳舞动作不是“每个关节都记录 xyz 坐标”，而是“根部世界平移 + 关节旋转 + 固定/体型相关骨架”共同推导出来的。

#### 10475 个网格顶点与 55 个骨架关节的关系

可以把 SMPL-X 看成“骨架驱动皮肤”的模型，但两者职责不同，不能把它们当作同一种点：

| 部分 | 固定数量 | 作用 | 每帧是否直接存储 |
|------|----------|------|------------------|
| 网格（皮肤/外表面） | 10475 个顶点、20908 个三角面 | 构成可见的人体表面；每个顶点最终有 `(x,y,z)` 坐标 | 不直接存储；由模型计算出 `vertices=(10475,3)` |
| 骨架 | 55 个关节 | 构成控制人体姿态的层级结构 | `poses=(55,3)` 存每关节轴角旋转 |
| 根部平移 | 1 个根节点 | 把整个人体放到世界坐标位置 | `trans=(3,)` 直接存储 |

单帧的生成顺序是：

```text
55 个关节轴角旋转 + 根节点世界平移
        │ 前向运动学
        ▼
55 个关节的世界位置与姿态
        │ 线性混合蒙皮（每个顶点受附近一个或多个关节加权影响）
        ▼
10475 个顶点的新坐标：vertices=(10475, 3)
        ▼
20908 个三角面按既有顶点索引连接，形成当前帧人体网格
```

因此，10475 个点是人体的“皮肤/外表面”，55 个关节是人体的“骨架”。关节旋转不会改变网格顶点或三角面的数量，只会通过蒙皮改变这 10475 个点的位置；`betas` 还能改变体型、骨架比例和静态顶点形状，但同样不改变网格拓扑。

也就是说，`poses[t] + trans[t] + betas` 并不是已经烘焙好的顶点动画，而是第 `t` 帧的**人体参数**。稍后 SMPL-X 前向运动学会把它变成 10475 顶点的人体网格。日志中的 `frames=3945 fps=30.000` 正是由这些数组读出，整段跳舞约为 131.5 秒。

### ② canonical 表面采样：为“人体模板”和“G1 模板”各建立 4096 个表面点

这一阶段对应日志开头的 `[SMPLXModel] loading`、两条 `[Dataset] Added ...` 以及 `saved correspondence dataset`。它由 pipeline 的 `build_correspondence_dataset()` 调用 `build_correspondence_ae_dataset.py` 完成。

1. pipeline 从动作 `.npz` 取出 `gender=neutral` 和 `betas=(16,)`，据此选择 `SMPLX_NEUTRAL.pkl` 并生成一个静态 canonical 人体模板；名字 `smplx_neutral_betas_c4eb7419e9` 中的哈希来自该体型。这里**不会逐帧播放跳舞动作**。
2. SMPL-X 前向计算得到 canonical 人体 mesh：`10475` 个顶点、`20908` 个三角面，并以 `spine1` 为中心消除平移差。
3. MuJoCo 载入 `g1_29dof.xml`，把配置中的四个 `tpose_qpos` 写入双肩/双肘：肩 roll 为 `±π/2`、肘为 `π/2`，从而得到 G1 的 T-pose 参考外形；以 `body:waist_yaw_link` 为中心。
4. 人体和 G1 各从三角网格表面采样 4096 点。G1 原始外观网格为 `192938` 顶点、`385626` 三角面；`exterior_surface=true` 与 `first_hit` 射线法只保留可见外壳，避免采到内部结构或碰撞几何。

输出为 `data/correspondence_unitree_g1_smplx_neutral_betas_c4eb7419e9.npz`。其核心数组是 `points=(2, 4096, 3)`：第 0 份为人体点坐标，第 1 份为 G1 点坐标；另外还保存 `mesh_vertices_0/1`、`mesh_faces_0/1`、采样到的三角面 id、中心偏移、T-pose 关节名和值等可复现元数据。日志先显示 `saved`，随后训练阶段再次调用 build 时显示 `reuse`，说明缓存校验通过、没有第二次采样。

#### 三角面、网格顶点与 4096 个采样点

三角面（triangle face）由三个顶点索引组成，例如 `face=[12,89,225]` 表示把 `v[12]`、`v[89]`、`v[225]` 连成一个表面小片。大量相邻三角面拼接后才构成完整的人体皮肤或机器人外壳；顶点随姿态变化而移动，但三角面引用的顶点编号保持不变。

因此，人体的 4096 点**不是从 10475 个顶点中直接挑出 4096 个**，而是在完整的 20908 个三角面内部重新生成的表面点。实际采样流程是：

```text
完整三角网格（人体：10475 顶点 / 20908 面；G1：192938 顶点 / 385626 面）
        │ 计算每个三角面的面积
        ▼
按面积比例随机选取三角面
        │ 在每个选中三角面内用重心坐标生成点
        ▼
候选点：人体默认 4096 × 8 = 32768 个
        │ 最远点采样（FPS），使点尽量均匀覆盖表面
        ▼
最终 4096 个表面点
```

重心坐标的含义是：若采样点位于三角面顶点 `A、B、C` 内，则 `P=wA×A+wB×B+wC×C`，并且 `wA+wB+wC=1`。所以采样点通常不是原始网格顶点，而是三角面中的任意位置。G1 还会先以 `first_hit` 射线从外部检测可见三角面，过滤内部壳层/碰撞几何，再做 FPS；这就是它虽然拥有高密度 mesh，最终仍统一压缩为 4096 点的原因。

G1 的 192938 顶点和 385626 三角面不是 UMR 训练生成的，而是 `g1_29dof.xml` 引用的预建模视觉 mesh 资产。UMR 在 T-pose 下加载所有视觉 geom，应用各零件的空间变换并拼接成一个网格。其他示例机器人也采用同样的“预建 MJCF/XML + mesh 资产 → 加载拼接 → 统一采样 4096 点”方式；按本机示例配置统计的视觉网格规模如下：

| 机器人 | 顶点数 | 三角面数 |
|---|---:|---:|
| Unitree G1 | 192,938 | 385,626 |
| Booster K1 | 87,943 | 176,095 |
| EngineAI T800 | 202,134 | 355,387 |
| HighTorque PiPlusPro | 753,804 | 1,596,739 |
| Unitree H2 | 123,828 | 248,194 |

这些数量来自当前仓库的视觉资产和配置，可能随 XML 或 mesh 资产版本而变化；它们不影响输入点数统一为 4096 的设计。

### ③ 点云对应学习：把两组“无序点”变成同编号的 surface slot

这一阶段对应日志中的 `train_correspondence_template_residual_ae.py`、`epoch=...` 与 `Saved final checkpoint...`。训练输入不是跳舞的 3945 帧，而是阶段②的两份静态点云：一个当前 `betas` 对应的人体 canonical/T-pose 点云，以及一个 G1 T-pose 点云。因此本次日志的 `samples=2、batch_size=2` 表示一个 batch 内有两个独立样本：

```text
batch_points=(2,4096,3)
├── batch_points[0]：人体无序点云 (4096,3)
└── batch_points[1]：G1 无序点云   (4096,3)
```

它们不会拼接成 `(8192,3)` 的混合点云，而是分别经过**同一套网络权重**。所谓“共享 Encoder”指参数共享，不是把人和机器人点混在一起编码。每个训练 epoch 还会对每个样本独立施加轻微绕 z 轴旋转、0.002 高斯噪声和随机点顺序打乱；这些是 T-pose 的数据增强，不是额外的动作帧。

#### 对应网络是什么：PointNet 风格的模板残差自编码器

它不是 RNN、LSTM 或 Transformer，也不学习时间序列；其 Encoder 是用于无序点云的 PointNet 风格 `1×1 Conv1d`，Decoder 是 MLP。对应类是 [PointNetTemplateResidualAE](../UMR/scripts/train_correspondence_template_residual_ae.py)：

```text
单个无序点云：(4096,3)
        │ 转置为 (3,4096)
        ▼
共享 Encoder：1×1 Conv  3→64→128→512→1024
        │ 对 4096 个点做 global max pooling
        ▼
该样本的全局形状描述 latent=(1024,)
        │
        ▼
Decoder MLP：1024→512→512→(4096×3)
        │ reshape
        ▼
residual_slots=(4096,3)
        + 固定的人体 template_points=(4096,3)
        ▼
reconstructed_slots=(4096,3)
```

`1024` 是 Encoder 的全局 latent 宽度，是 PointNet 常见的工程超参数，而非由 4096 自动推导的必然数；脚本可通过 `--latent-dim` 修改，默认值为 1024。它把一个点云的整体形状压缩成固定长度描述，再由 Decoder 解码为有固定下标顺序的 4096 个 slot。默认 `hidden_dim=512`，也可配置。

人体和 G1 分别产生一个 latent：`latent[0]=(1024,)`、`latent[1]=(1024,)`，再分别生成两组 `reconstructed_slots`。网络并没有人工标注“人的左膝必须对应 G1 左膝”；同编号 slot 的语义是在以下损失共同约束下自动形成的：

| loss 项 | 本次权重 | 约束作用 |
|---------|---------:|----------|
| Chamfer | 1.0 | 每组输出点都要覆盖、贴近自己的输入表面 |
| repulsion | 0.002 | slot 相互排斥，避免聚在小区域 |
| geodesic edge | 0.4 | 人体模板上邻近的 slot，在另一形状上也应尽量邻近，保留表面拓扑 |
| residual | 0.0 | 本次不额外惩罚偏离人体 template 的残差大小 |

初始时 Decoder 最后一层被置零，故两种输入起初都会输出固定的人体 template；训练再学习各形状相对模板的残差。最终可以按编号理解为 `human_slot[i] ↔ g1_slot[i]`（`i=0...4095`），但这是**学习到的软语义表面对应**，不是人体原始第 i 个顶点与 G1 原始第 i 个顶点的硬映射。

#### 4096 个 slot 如何重新贴回各自网格

训练所得 `reconstructed_slots` 的坐标可能与初始采样点略有偏移。因此，重定向开始前不会简单沿用采样时的 `sample_face_ids`；UMR 会将训练后的每个 slot 重新投影/绑定到各自 canonical 网格的最近三角面，记录三角面 id 和重心坐标。

```text
人体/G1 reconstructed_slots[i]
        │ 最近顶点搜索 → 邻近候选三角面 → 最近三角面投影
        ▼
slot #i 的网格绑定：face_id + barycentric weights
```

- **人体侧**：SMPL-X 每帧计算新的 10475 顶点坐标；用同一 `face_id` 中的三个新顶点和保存的重心坐标插值，得到该帧人体 `slot[i]` 的位置；该三角面的叉积同时给出当前法向。可把 slot 看成贴在可变形皮肤上的标记点。
- **G1 侧**：每个 slot 绑定到一个视觉 geom，并记录相对这个刚性零件的 `local_pos/local_normal`。每次候选 `qpos` 改变后，MuJoCo FK 更新该 geom 的旋转和平移，slot 随所属零件刚性移动。可把它看成贴在机器人外壳上的标记点。

这两种绑定都不是“4096 点与原始网格顶点一一对应”；它们是“4096 个表面点分别绑定到三角面/刚体零件”。正式优化时仍保留全部 slot 的绑定信息，之后根据 19 个身体段权重选择或激活其中一部分参与不同约束。

#### 训练产物与跳舞动作如何使用

训练结束会写出两类产物：

| 文件 | 内容 | Quick Start 后续是否直接使用 |
|------|------|------------------------------|
| `checkpoint-final.pt` | 网络权重、优化器状态、训练参数等 | 否；用于继续训练或复现实验 |
| `correspondence_slots_final.npz` | 两组 `reconstructed_slots=(2,4096,3)`、`target_points`、归一化数据、template 边图、latent 等 | 是；pipeline 以 `--slots-field reconstructed_slots` 读取 |

跳舞时不会再次把 3945 帧动作送进这个 AE，也没有 RNN 负责时间连续性。流程是：

```text
第 t 帧 poses[t] + trans[t]
        │ SMPL-X FK / 蒙皮
        ▼
人体 4096 个 slot 的当前位置、法向
        │ 以同编号 slot 为目标
        ▼
优化 G1 的 qpos[t]，使 G1 slot[i] 靠近人体 slot[i]
        │
        ▼
qpos_seq=(3945,36)
```

时间连续性来自 `qpos[t-1]` 初始化 `qpos[t]` 的 warm start、相邻帧步长限制，以及全部帧求解后执行的 LQR 平滑，而不是来自这个离线点云网络。

这里的 slot 文件存的是**对应点坐标及其训练元数据，不直接存表面法向或 19 段标签**。后续逐帧重定向会依托网格重新计算法向，并用 `assets/smplx_parts_segm.pkl` 和 `retarget_body_segment_surface.py` 给人体 slot 分段。

### ④ 对应引导的逐帧重定向：从每帧人体参数求每帧 G1 `qpos`

这一阶段从日志的 `retarget motion seq_key='dance1_subject2'` 开始，到 `retarget forward: 100%` 结束。pipeline 启动 `retarget_smpl_to_humanoid_surface_vector.py`，将动作路径、slot 路径和求解器参数作为命令行参数传入。内部数据变化如下：

```text
SMPL-X 参数：pose_aa[t]=(165,) + trans[t]=(3,) + beta=(16,)
        │  SMPL-X FK（GPU，3945 帧批量）
        ▼
每帧人体 mesh / 对应 source slot 的世界坐标与法向
        │  按身体段选取 456 个代表 slot，计算接触与约束
        ▼
每帧非线性优化：qpos[t]=(36,)
        │  顺序 warm start + MuJoCo 正运动学/几何查询
        ▼
qpos_seq=(3945,36)
```

具体来说：

1. **生成运动人体表面。** SMPL-X 用 3945 帧 `pose_aa` 与 `trans_orig` 前向计算；日志的 `[SurfaceRetarget][SMPLX] ... batch_size=3945` 表示本次全部帧可一次放入 GPU。阶段③的 source slot 被绑定回人体网格，日志给出绑定误差 `error_mean=0.01023m`。
2. **统一尺寸和语义。** 从 slot 推得 G1 高度 `1.32230m`，人体高度为 `1.84824m`，故将人体目标缩放为 `smpl_scale=0.715439`。再把 4096 个 source slot 按 SMPL-X 分割映射到 19 个身体段，并从中选 456 个代表点；这解释了日志中的 `segments=19` 和 `selected segment slots: total=456`。
3. **准备可自由运动的机器人。** 为了让根部也能平移/旋转，脚本从 `g1_29dof.xml` 生成 `assets/g1/dance1_subject2_smplx_unitree_g1.floating_mjcf.xml`，在根部加入 free joint。于是 MuJoCo 的位置向量从 29 个关节扩展为 `7` 个浮动基座值（xyz + 四元数）加 `29` 个关节值，即每帧 `qpos=(36,)`。
4. **逐帧求解。** 对第 `t` 帧，以第 `t-1` 帧解作 warm start；MuJoCo 根据候选 `qpos` 做正运动学，取得机器人绑定 slot 的位置/法向。优化器让它们靠近同编号、缩放后的人体 slot，同时施加日志中显示的地面接触锚定、防穿地、身体段自穿约束、MJCF+配置的关节限位、`max_dq=0.15` 的相邻帧步长限制等。这里运行的是“带 MuJoCo 几何计算的运动学优化”，不是让机器人靠控制器实时追随人体。
5. **形成原始轨迹。** 3945 帧求解完成后得到 `qpos_seq=(3945,36)`；日志的 `128.17it/s` 表明本机约 30 秒完成逐帧优化。

### ⑤ 轨迹平滑、保存并在 MuJoCo 中显示：预计算轨迹的回放，而非再次物理仿真

这个阶段对应日志的 `[TrajectoryFilter]`、`saved ... npz` 和全部 `[RetargetVis]` 行。

1. **整段 LQR 平滑。** 阶段④逐帧得到的解先组成完整序列，再通过 LQR 在不大幅偏离原轨迹的前提下压低加速度与 jerk。日志显示 `step_p95 0.149144→0.144822`、`accel_p95 0.0807837→0.0699183`、`jerk_p95 0.0888602→0.0604294`，说明平滑后的轨迹更连续。
2. **保存自描述结果。** `np.savez_compressed()` 写出 `output/unitree_g1_retarget/dance1_subject2_smplx_unitree_g1.npz`。本机实际字段包括：`qpos=(3945,36)`、`frame_ids=(3945,)`、`fps=(1,)`、`robot_xml`、`robot_name`、`robot_joint_names=(29,)`、`source_data`、`source_sequence_key`、`source_format`、`smpl_scale`、`ground_z`。因此结果文件既有轨迹值，也保留了应加载哪个 floating MJCF、每列关节含义和源动作出处。
3. **读取并构建 viewer。** pipeline 立刻调用 `visualize_robot_retarget_result.py --result ... --play --rate-limit --viewer-backend glfw-ui`。viewer 先从结果 `.npz` 读取 `qpos`、`fps=30` 和 `robot_xml`，再用 `MjModel.from_xml_path()` 加载刚才生成的 floating MJCF，创建 `MjData`；日志验证 `model.nq=36` 与 `qpos_width=36` 一致，并默认隐藏 48 个纯碰撞 geom。
4. **逐帧渲染。** 每个播放时刻，viewer 将 `playback_qpos[frame]` 写入 `data.qpos[:]`，调用 `mujoco.mj_forward(model, data)` 更新 FK、body/geom 位姿，然后以 GLFW 渲染外观 mesh；`--rate-limit` 使帧索引按 30fps 的墙钟时间推进。它没有调用 `mj_step()` 来让物理引擎重新积分，因此屏幕看到的是**已经优化完成的 qpos 运动学回放**，不是执行控制策略后的新物理 rollout。
5. **可选录制。** 用户触发录制后，viewer 对渲染帧编码，才出现日志末尾两个 `recording_...mp4`；这两个 mp4 是显示层输出，不会改变重定向 `.npz`。

### 一条数据在这次运行中的完整“身份变化”

```text
LAFAN1 原始 BVH（命令外已转换）
  → dance1_subject2.npz：3945 帧 SMPL-X 参数（poses/trans/betas/fps）
  → canonical 人体模板 + G1 T-pose 模板：两个三角网格
  → correspondence_*.npz：两组无序 points=(2,4096,3)
  → correspondence_slots_final.npz：同编号 reconstructed_slots=(2,4096,3)
  → 每帧人体 slot 目标 + 约束优化
  → dance1_subject2_smplx_unitree_g1.npz：qpos=(3945,36) + 回放元数据
  → floating_mjcf.xml + MjData.qpos + mj_forward()
  → GLFW MuJoCo 窗口 / 可选 mp4
```

---

## 〇、一句话总览

UMR 不做"人体关节 → 机器人关节"的直接映射，而是把**运动着的人体外表曲面**当作中间语言：

1. **离线学一次**：在标准站姿（canonical pose）下，学习人体表面 4096 个点与机器人表面 4096 个点的**有序对应**（哪个点对应哪个点）；
2. **每条动作复用**：人体做动作时，这 4096 个点跟着运动；逐帧求解机器人关节角，让机器人身上对应点的**位置/朝向/接触关系**跟上人体，同时满足不穿地、不自穿、关节限位等物理约束。

学到的对应关系对"同一人体模板 + 同一机器人"只学一次，之后所有动作直接复用——这就是它能批量生产训练数据的关键。

---

## 关键词与缩写速查（建议先读）

- **Retargeting（动作重定向）**：把一个骨架（人）的动作搬到另一个骨架（机器人）上的技术，核心难点是两边关节数量、肢体比例、自由度都不一样。
- **SMPL-X**（Skinned Multi-Person Linear - eXpressive）：马普所提出的参数化人体模型，输入体型和姿态参数就能生成带手部、表情细节的人体网格；UMR 把它作为所有人体动作的统一"中间格式"。
- **SMPL / SMPL-H**：SMPL-X 的前身，H 版增加了手部；部分数据源仍是旧格式，由 adapter 负责兼容。
- **betas**：SMPL-X 的体型参数（一组形状系数），决定人体的高矮胖瘦；模板名里那串哈希就来自它。
- **FK（Forward Kinematics，正运动学）**：已知所有关节角，推算身体和网格顶点在世界坐标系中的位置；SMPL-X 生成人体、MuJoCo 更新机器人姿态都靠 FK。
- **canonical pose（规范姿态）**：学对应之前让人体和机器人共同摆出的对齐基准姿态（人体用标准站姿、机器人用 T-pose），对应关系只在这个姿态下学一次。
- **T-pose**：双臂向两侧平伸、身体站直呈"T"字的标准姿态；机器人的 T-pose 由配置文件里的 `tpose_qpos` 给出。
- **点云（Point Cloud）**：用三维空间中一堆离散坐标点来表示物体表面；UMR 在人体和机器人表面各采样 4096 个点。
- **点云对应（Point Cloud Correspondence）**：确定人身上每个点在机器人表面上"语义对应"的是哪个点（人的膝附近 ↔ 机器人的膝附近）。
- **slot（槽位）**：自编码器学出的共享编号；同一个 slot 编号在人体点云和机器人点云里各取一个点，这两个点就是一对对应。
- **AE（Autoencoder，自编码器）**：一种"编码—解码"结构的神经网络，这里被用来从两套无序点云中发现一套共享、有序的 slot 结构。
- **Chamfer Distance（倒角距离）**：两组点云间双向最近点距离的平均值，衡量两堆点整体形状差多少，是阶段①的主对齐损失。
- **repulsion（排斥正则项）**：让各 slot 对应的点在表面上互相"散开"，防止所有点挤在一小块区域、保证覆盖均匀。
- **geodesic（测地距离 / 测地图）**：沿曲面表面走的最短距离（不是两点间的空间直线），用于约束"人身上相邻的点在机器人上也要相邻"，保住拓扑结构。
- **normal（表面法向）**：垂直于表面的方向向量；优化位置的同时对齐法向，才能保证"掌心朝前、脚底板朝下"这类朝向语义。
- **qpos**：MuJoCo 的广义坐标位置向量，即机器人某一时刻的完整姿态；本例 36 维 = 浮动基座 7 维 + 29 个关节角。
- **浮动基座（floating base）**：把机器人躯干视为可在空间中自由平移旋转的根（3 个平移 + 四元数 4 个值 = 7 维），而不是固定在地面上；这样才能复现行走、跳跃、摔倒。
- **DOF（Degree of Freedom，自由度）**：`g1_29dof` 表示 G1 有 29 个可驱动关节（`scalar_joints=29`）。
- **MJCF（MuJoCo XML 模型格式）**：MuJoCo 描述机器人的文件，定义刚体、关节、mesh、碰撞体、关节限位等；`*.floating_mjcf.xml` 是临时生成的浮动基座版本。
- **MuJoCo**：Google DeepMind 的物理仿真器，UMR 用它做运动学计算、碰撞检测和最终动作回放。
- **硬约束 / slack（松弛变量）**：数学优化中"绝对不许违反"的条件；工程上常用"违反一点但重罚（cost 1000）"的 slack 变量近似实现，比如不许穿地。
- **warm start（热启动）**：用上一帧的解当作下一帧优化的初值；双向 warm start 还会从动作末尾反向再解一遍，用来救回卡住的奇异帧。
- **LQR（Linear-Quadratic Regulator，线性二次调节器）**：此处借用它的二次型代价思想做整条轨迹的事后平滑，在"贴合原解"和"限制加速度/抖动"之间折中。
- **jerk（加加速度）**：加速度的变化率；jerk 越小动作越柔顺不突兀。
- **LAFAN1**：Ubisoft La Forge 开源的游戏级动捕数据集（BVH 格式），Quick Start 的跳舞动作来自它，需先用 `lafan_to_smplx` 转成 SMPL-X。
- **BONES-SEED / SOMA**：BONES 工作室发布的人形机器人动作数据集；SOMA（SOMA-X）是其基于扫描人体表面的动作表示。
- **GRAIL**：NVIDIA 发布的大规模人-场景交互数据集（locomanipulation），包含走路、坐下、与家具交互等。
- **OMOMO**：人体操作物体（human-object manipulation）动捕数据集，含女性 SMPL-X 序列，所以需要 `SMPLX_FEMALE` 模型。
- **OmniContact**：接触丰富的人形操作研究（论文+数据集），其公开发布的 G1 重定向数据就是 UMR 内部版本生产的。
- **MimicKit**：物理角色模仿工具包，提供人形"角色动画"作为一类输入源（与真人动捕相对）。
- **BVH**（Biovision Hierarchy）：经典骨骼动捕文件格式；**FBX** 是 Autodesk 的通用三维绑定/动画格式——UMR 的 NR 数据源用这两种格式，FBX 解析需 Node.js。
- **AdaPT**：人形机器人打网球的研究项目（身体 + 球拍的对应与重定向），UMR 有专门的 adapt 流程。
- **NR 源**：仓库作者团队（Noitom Robotics）的内部 FBX/BVH 人体动作源。
- **adapter（适配器）**：`sample_data/<source>/README.md` 与对应 pipeline 变体的统称，作用是把上述各种异构数据统一整理成 SMPL-X 参数序列。
- **npz**：NumPy 的压缩存档格式，UMR 的中间数据、对应关系、最终 qpos 结果全部存成 `.npz`。
- **sim2real（仿真到真机）**：让仿真中得到的策略/动作能迁移到真实机器人；满足接触、限位、不穿透的重定向结果更利于 sim2real。

---

## 〇·一、先看 Quick Start 命令：两个文件各做什么

安装指南中的命令是：

```bash
python scripts/humanoid_retarget_pipeline.py \
  --config robot_configs/humanoid_retarget_unitree_g1_example.json
```

可以把它理解为：**用 `humanoid_retarget_pipeline.py` 这位“总调度员”，读取 G1 的专属配置，再补齐仓库的通用默认配置，完成一次从示例人体动作到 G1 `qpos` 的流水线执行。**

### 1. `humanoid_retarget_pipeline.py`：流水线总调度员

这个脚本本身不把所有算法塞在一起实现；它负责读取配置、检查缓存、组装参数并依次调度下游脚本。未显式传入 `--stage` 时，参数默认是 `all`，因此成功运行一次会按以下顺序执行：

| 阶段 | 总调度脚本的动作 | 实际执行者 | 结果 |
|------|------------------|------------|------|
| `build` | 构建或复用 canonical 人体/G1 表面采样数据 | `build_correspondence_ae_dataset.py`（由 pipeline 直接调用） | `data/correspondence_unitree_g1_*.npz` |
| `train` | 训练或复用人体—G1 的有序表面 slot | `train_correspondence_template_residual_ae.py`（子进程） | `output/correspondence_unitree_g1_*/correspondence_slots_final.npz` |
| `retarget` | 用 slot 对每帧 SMPL-X 人体姿态求 G1 的浮动基座和关节角 | `retarget_smpl_to_humanoid_surface_vector.py`（子进程） | `output/unitree_g1_retarget/dance1_subject2_smplx_unitree_g1.npz` |
| `view` | 读取结果并打开 MuJoCo GLFW viewer | `visualize_robot_retarget_result.py`（子进程） | 屏幕上的回放窗口 |

因此，终端里出现的 `[HumanoidPipeline] ...` 命令行并不是多余日志，而是总调度脚本将当前配置展开后实际启动的下游任务。任一子任务失败时，`subprocess.run(..., check=True)` 会让总流程立即报错退出，避免生成不完整结果。

脚本还负责缓存校验：已有且兼容的采样数据、slot 文件或最终结果会被复用；配置不兼容时才重建。常用控制参数如下：

```bash
# 只执行其中一个阶段；后续阶段会检查其前置产物是否存在/兼容
python scripts/humanoid_retarget_pipeline.py --config robot_configs/humanoid_retarget_unitree_g1_example.json --stage retarget

# 强制重建对应数据、强制重训 slot，或强制覆盖最终轨迹
# --force-build / --force-train / --force-retarget

# 不打开图形窗口；适合 SSH 或只想得到 npz 的情形
# --skip-view
```

> `--stage retarget` 在 slot 缺失或不兼容时会自动补做 build/train；`--stage view` 只读取已有结果。`--dry-run` 可只打印计划执行的下游命令而不运行。

### 2. 配置不是只有 24 行：自动合并的默认参数

传入的 [humanoid_retarget_unitree_g1_example.json](../UMR/robot_configs/humanoid_retarget_unitree_g1_example.json) 看起来只定义了 `robot`，但这不是一份不完整配置。配置加载器 `load_config()` 发现它没有 `extends` 后，会自动将仓库根目录的 [humanoid_retarget_defaults.json](../UMR/humanoid_retarget_defaults.json) 作为父配置，并让 G1 文件中的同名字段覆盖默认值。

```text
humanoid_retarget_defaults.json              # 通用：SMPL-X、示例动作、4096 点、500 epoch、优化器、viewer
                    +
robot_configs/humanoid_retarget_unitree_g1_example.json  # 特化：G1 模型、中心、T-pose、额外限位
                    ↓ 深度合并
本次运行的完整运行时配置
```

所以本例默认读取 `sample_data/lafan1_smplx` 中键名为 `dance1_subject2` 的动作，使用 `smpl/` 下的 neutral SMPL-X 模型；采样 4096 点、训练 500 epoch、使用 CUDA SMPL-X 前向计算，并在完成后以 30fps 打开 viewer。这些内容来自父配置，而不是写在 24 行的 G1 文件里。

### 3. `humanoid_retarget_unitree_g1_example.json`：把“通用流程”绑定到 G1

该文件的 `robot` 段只回答机器人侧的四类问题：用哪个模型、在哪里对齐、canonical 姿态怎么摆、哪些关节范围必须额外收紧。

| 字段 | 本例值 | 对流水线的影响 |
|------|--------|----------------|
| `name` | `unitree_g1` | 缓存与输出目录命名的一部分，例如 `output/unitree_g1_retarget/`。 |
| `xml` | `../assets/g1/g1_29dof.xml` | G1 的 MuJoCo MJCF。路径相对于 `robot_configs/` 解析，实际指向 `UMR/assets/g1/g1_29dof.xml`；它提供 29 个可驱动关节、几何 mesh、原始关节范围和碰撞信息。 |
| `point_cloud_center` | `body:waist_yaw_link` | 用腰部 `body` 作为机器人表面点云的对齐中心，使人体和机器人比较时消除全局平移差异；也定义重定向时的根部参考。 |
| `tpose_qpos` | 左/右 shoulder roll 为 `±1.57079632679`，两侧 elbow 为 `1.57079632679` | 在**阶段①采样前**把 G1 摆成 T-pose。数值单位是弧度，`1.57079632679 ≈ π/2 = 90°`；这决定 canonical 表面形状，因而会影响之后学出的对应关系。 |
| `joint_limits` | 膝、腕 pitch、肘、肩 pitch 共 8 项 | 在 MJCF 自带范围之上给本次求解附加/收紧范围：膝最小 `0.1` rad 防反折，腕 pitch 限于 `[-0.3, 0.8]`，肘上限为 `π/2`，肩 pitch 限于 `[-3.0, 1.4]`。字符串 `"inf"` / `"-inf"` 会被脚本解析成正/负无穷，表示该侧不额外设限。 |

其中最容易误解的是 `tpose_qpos`：它**不是**让最终跳舞轨迹永远保持举臂；它只定义“学人体和 G1 表面对应时，G1 的标准参考姿态”。最终每帧的 29 个关节值仍由阶段②优化器求解。

### 4. 这次命令实际消耗与产出什么

```text
配置合并后输入
├── smpl/SMPLX_NEUTRAL.pkl 或 .npz                 # 参数化人体模板
├── sample_data/lafan1_smplx/dance1_subject2.npz   # 3945 帧人体动作
├── assets/g1/g1_29dof.xml                          # G1 29-DOF MuJoCo 模型
└── G1 专属的 T-pose / 额外关节限位

可缓存的中间产物
├── data/correspondence_unitree_g1_*.npz            # 两边 canonical 曲面采样
└── output/correspondence_unitree_g1_*/correspondence_slots_final.npz
                                                     # 4096 对有序表面点坐标（法向在重定向时依托网格重新计算）

最终产物
└── output/unitree_g1_retarget/dance1_subject2_smplx_unitree_g1.npz
                                                     # qpos=(3945, 36)
```

换言之，`--config` 选择的是“**把哪台机器人、以什么参考姿态和限位条件接入这套通用流水线**”；`humanoid_retarget_pipeline.py` 则执行“**构建对应 → 学对应 → 按动作求解 → 回放**”。后文依次展开这两个核心计算阶段。

---

## 一、全链路总览图

```text
运动源（异构）
 LAFAN1/SMPL-X · BONES-SEED/SOMA · GRAIL · OMOMO · BVH/FBX · MimicKit 角色
        │  各 adapter 统一转换成 SMPL-X 参数序列
        ▼
sample_data/.../*.npz  （body_pose/global_orient/transl/betas，3945 帧 @30fps）
        │  SMPL-X 前向运动学（smpl/SMPLX_NEUTRAL.pkl，10475 顶点人体网格）
        ▼
┌──────────────────── 阶段 ① 点云对应学习（离线，一次性，可缓存）────────────────────┐
│  人体 canonical 网格 ──采样 4096 点──┐                                            │
│                                      ├─ 残差自编码器：共享 slot 排序             │
│  机器人 T-pose MJCF ──采样 4096 点──┘  （chamfer + 排斥 + 测地边一致性）         │
└──────────────────────────────────────────────┬───────────────────────────────────┘
                                               ▼
              correspondence_slots_final.npz（4096 对有序表面点坐标 + 训练元数据）
                                               │
┌──────────────────── 阶段 ② 对应引导的逐帧优化（每条动作一次）────────────────────┐
│  对每一帧：决策变量 = 机器人 qpos（浮动基座 7 + 29 关节 = 36）                   │
│  最小化：表面点位置误差 + 法向误差（按 19 个身体段加权）                         │
│  约束/惩罚：地面接触锚定、地面穿透（硬）、自穿透（硬 slack）、                    │
│            关节限位、单帧步长限幅、根部位移/旋转限幅                              │
│  顺序 warm start（双向）→ 逐帧求解（128 帧/秒）→ LQR 轨迹后滤波                  │
└──────────────────────────────────────────────┬───────────────────────────────────┘
                                               ▼
        output/unitree_g1_retarget/dance1_subject2_smplx_unitree_g1.npz
        qpos=(3945, 36)，可直接喂 MuJoCo viewer / 作为机器人模仿学习训练数据
```

---

## 二、输入：统一到 SMPL-X 表示

无论原始动作来自哪里，adapter 都先转成 **SMPL-X 参数序列**，这是 UMR 的统一入口：

- `global_orient`：根骨朝向（1×3×3 旋转或轴角）
- `body_pose`：身体关节姿态
- `transl`：根部位移
- `betas`：体型参数（决定胖瘦高矮，canonical 模板的形状也由它定）
- 帧率/帧 id：本例 3945 帧 @30fps（约 131 秒跳舞）

SMPL-X 模型（`smpl/SMPLX_NEUTRAL.pkl`，约 519MB；同目录的 `.npz` 版约 104MB）是一个**参数化人体**：给定姿态+体型，前向运动学输出 **10475 个顶点、20908 个三角面**的人体网格。这是所有后续计算的"人体真值曲面"。

```text
[Dataset] Added smplx_neutral_betas_c4eb7419e9: verts=10475 faces=20908 center=spine1
```

名字里的 `c4eb7419e9` 是体型 betas 的哈希——换了体型就会生成新模板、需要新对应；neutral + 默认 betas 对应这一个模板。

---

## 三、阶段 ① 点云对应学习：把两套曲面"对齐编序"

### 3.1 为什么需要它

传统 retargeting 要人手工指定"人的肩对应机器人的肩、人的肘对应机器人的肘"。G1 没有脖子、没有手指、腿臂比例和人完全不同，这种关节级映射既费力又必然丢信息。

UMR 的思路：**机器人的 mesh 本来就定义了它的全部外表形状**，直接在表面点之间找对应即可，不需要关节语义。

### 3.2 表面采样（canonical pose）

- **人体侧**：用 SMPL-X 的 neutral 标准姿态网格采样 4096 点，中心对齐到 `spine1`（脊柱）；
- **机器人侧**：把 G1 的 MJCF 摆成配置文件里的 **T-pose**（`tpose_qpos`：两肩外展 90°、肘弯 90°），从**外表面**射线求交采样 4096 点——`exterior_method=first_hit` 保证只取看得到的外壳（滤掉内部结构/碰撞体），中心对齐到 `body:waist_yaw_link`。

```text
[Dataset] Added unitree_g1: verts=192938 faces=385626 exterior_surface=True exterior_method=first_hit
saved correspondence dataset: .../correspondence_unitree_g1_smplx_neutral_betas_c4eb7419e9.npz
points=(2, 4096, 3)
```

G1 原始 mesh 有 19 万顶点，被抽稀到 4096 个代表点。这一步的中间产物是 `data/correspondence_<robot>_<template>.npz`，里面同时存了采样点、原始 mesh、中心偏移、采样参数，便于复现。

### 3.3 残差自编码器：学一套"共享的点序"

两个点云各有 4096 个点，但初始顺序互不相干。脚本 [train_correspondence_template_residual_ae.py](../UMR/scripts/train_correspondence_template_residual_ae.py) 训练一个小型自编码器，让两个点云通过同一套隐层 **slot（槽位）** 重排：同一个 slot 索引在人身上和在机器人身上应当落在**语义对应的表面位置**。

训练信号（日志可见）：

| loss 项 | 权重 | 作用 |
|---------|------|------|
| `chamfer` | 1.0 | 两组点整体形状对齐（双向最近点距离） |
| `repulsion` | 0.002 | slot 之间互相排斥，避免点挤成一团、保证覆盖均匀（k=8，半径 0.035） |
| `edge` | 0.4 | **测地邻接图一致性**（k=32）：人身上相邻的两个点，在机器人上的对应点也应相邻——这是对应保持拓扑结构而不是乱配的关键 |

先按各自身高归一化（人 1.8482m、G1 1.3228m，`normalize=per_sample_height`），500 epoch、batch 2、cosine 学习率衰减。本机 5090 上这一步很快：

```text
epoch=00001 chamfer=0.001105 ...
epoch=00500 loss=0.001660 chamfer=0.000111 edge=0.000084
Saved final checkpoint and correspondence slots
```

### 3.4 产物与复用

```text
output/correspondence_unitree_g1_smplx_neutral_betas_c4eb7419e9/correspondence_slots_final.npz
```

这个文件是阶段 ① 的唯一“成果”：4096 对有序的人体/机器人表面点坐标及训练元数据。法向会在后续逐帧重定向时依托网格重新计算，19 段身体归属也在那时依据人体分割生成。**再跑任何动作都不会重训**，日志里能看到：

```text
[HumanoidPipeline] reuse correspondence dataset: ...
```

只有以下情况才重建：换机器人、换人体模板（性别/betas）、改采样点数或 T-pose。

---

## 四、阶段 ② 对应引导的逐帧优化：让 G1 跟着动

对应脚本：[retarget_smpl_to_humanoid_surface_vector.py](../UMR/scripts/retarget_smpl_to_humanoid_surface_vector.py)。这一步才是真正"算动作"。

### 4.1 决策变量

G1 模型是 `g1_29dof.xml`（29 个驱动关节），求解时生成一个**浮动基座**版本 MJCF（日志里的 `*.floating_mjcf.xml`），所以每帧待求 `qpos` 为：

```text
7（浮动基座：3 平移 + 4 四元数）+ 29（关节）= 36 维
scalar_joints=29，model.nq=36
```

3945 帧顺序求解，最终 `qpos=(3945, 36)`。

### 4.2 表面点绑定到身体段

4096 个 slot 不是等权使用的。先把 SMPL-X 的 55 类身体部件映射成 **19 个功能段**（`smplx_55_to_19_upper_arm_v1`）：头、躯干、髋、大臂/小臂/手、大腿/小腿/脚、肩等，然后每个段挑出少量代表点（共 456 个）：

```text
selected segment slots: total=456, head=20, torso=50, hips=40, leftFoot=30 ...
```

这样做的好处：可以**按段给不同权重**，表达"哪里更像更重要"。

### 4.3 目标函数（要最小化的）

每帧把 4096（实际激活 3824）个对应点的两类误差加权求和，权重按段配置（见 `retarget_body_segment_surface.py`）：

- **表面位置误差 SurfacePoint**：机器人对应点应到达人体点的位置。躯干/髋权重 20，四肢 10，头 1；肩/大臂为 0（这些段允许 G1 自由补偿，因为它没有对应自由度）；
- **表面法向误差 SurfaceNormal**：表面朝向也要一致（比如掌心朝前），小腿/大臂权重 5，躯干 5，头/脚 1。

```text
[SurfacePoint] active_slots=3824/4096
[SurfaceNormal] active_slots=3587/4096
```

### 4.4 接触与物理约束（决定动作"能落地"的关键）

光拟合表面会得到穿地、打滑、反关节的动作，所以还加了：

| 约束 | 实现方式 | 日志依据 |
|------|----------|----------|
| **地面接触锚定** | 人体表面贴近地面的点（<5mm）强制机器人对应点 z=0，cost 5.0 | `GroundContactAnchor ... cost=5.0`，2889/3945 帧有锚点 |
| **防穿地（硬）** | 表面点不允许低于地面，阈值 5cm，最多取 128 点，违约重罚 1000 | `GroundPenetration hard_constraint=true` |
| **防自穿（硬 slack）** | 身体段对之间留 1mm 间隙，违约用 slack 变量 + 罚 1000 | `SelfContactMap`、`robot-self-penetration-margin -0.001` |
| **关节限位** | MJCF 自带限位 + 配置文件额外收紧（膝≥0.1、腕/肘/肩范围） | `JointLimit configured=8 ... scalar_joints=29` |
| **步长限幅** | 相邻两帧每个关节变化 ≤0.15 rad（l2 盒约束），全局步长 0.4，防跳变 | `--max-dq 0.15 --global-step-size 0.4` |
| **根部位移限幅** | 根平移 ≤0.3/帧、旋转 ≤0.5/帧 | `--root-max-translation-dq 0.3 ...` |

> 人体与机器人身高不同：人 1.848m、G1 1.322m，动作整体按 `smpl_scale=0.7154` 缩放后再优化，脚始终锚在 z=0，避免"缩放后悬空/入土"。

### 4.5 求解方式

逐帧非线性优化（对 qpos 做雅可比线性化 + 限步长迭代，硬约束靠高罚 slack），关键工程优化：

- **顺序 warm start**：用上一帧的解初始化下一帧（动作天然连续）；批量模式还开双向 warm start + 动态规划，应对偶发奇异点；
- 全量 batch 在 GPU 上跑，实测 **128 帧/秒，3945 帧约 30 秒**：

```text
retarget forward: 100%|██| 3945/3945 [00:30<00:00, 128.17it/s]
```

### 4.6 LQR 轨迹后滤波

逐帧独立求解的结果拼起来会有轻微抖动。最后做一次**整段 LQR 平滑**，在"贴合原始解（data cost 1.0）"和"限制加速度（0.1）/加加速度 jerk（0.01）"之间取平衡，并锚定首尾帧、包含根平移。效果直接打印：

```text
step_p95 0.1491→0.1448  accel_p95 0.0808→0.0699  jerk_p95 0.0889→0.0604
```

步长、加速度、抖动的 p95 都下降——动作更顺，又没偏离原解。

### 4.7 输出

```text
saved .../dance1_subject2_smplx_unitree_g1.npz qpos=(3945, 36) cost mean=0.065865 max=0.231606
```

`cost mean=0.066` 是表面拟合的残差水平（越小越贴合，作为质量指标）。

---

## 五、npz 里到底有什么

输出文件不只是一串关节角，还带齐了回放所需的全部元数据（代码 `RETARGET_VISUALIZATION_FIELDS`）：

| 字段 | 内容 |
|------|------|
| `qpos` | **核心产物**，(3945, 36)：每帧的浮动基座 + 29 关节角度 |
| `fps` / `frame_ids` | 30fps、源帧索引（支持 stride/裁剪后对齐） |
| `robot_xml` / `robot_name` / `robot_joint_names` | 用的哪个 MJCF、关节名顺序（保证 qpos 列含义可追溯） |
| `source_data` / `source_sequence_key` / `source_format` | 来源动作文件、序列名、格式（如 `smplx_npz`） |
| `smpl_scale` / `ground_z` | 人体→机器人缩放比 0.7154、地面高度 |

这意味着 npz 是**自描述**的：拿到文件就能用 `visualize_robot_retarget_result.py` 独立回放，也能写脚本把 `qpos` 转成训练用的关节轨迹/动作 chunk。

---

## 六、可视化阶段

pipeline 最后自动调用 viewer（等价于手动执行）：

```bash
python scripts/visualize_robot_retarget_result.py \
  --result output/unitree_g1_retarget/dance1_subject2_smplx_unitree_g1.npz \
  --play --rate-limit --viewer-backend glfw-ui
```

- `--rate-limit`：按墙钟 30fps 播放（不跑满）；`--ghost-interval 1.0`：拖 1 秒残影，便于看运动趋势；
- viewer 按键：`Space` 播放/暂停、`A/D` 单帧、`Q/E` 快进退、`[/]` 变速、`G` 残影开关、拖动进度条跳帧；
- 碰撞体默认隐藏（48 个 collision geom hidden），只看外观 mesh。

---

## 七、数据链路在"生产训练数据"语境下的位置

UMR 本身**不训练策略**，它处在更上游——为人形机器人的模仿学习/RL 批量造数据：

```text
互联网/动捕库里的海量人体动作（BVH、SMPL-X、角色动画……）
        │  UMR：免关节映射、批量（--workers 并行）、自带物理约束
        ▼
成百上千条机器人可行 qpos 轨迹（output/batch_retarget/<robot>/）
        │  转换成模仿学习数据集（state/action chunk）
        ▼
PPO / ACT / Diffusion Policy 等策略训练（如本机 microduck、lerobot 工作流）
        ▼
机器人真机部署
```

相比"真机遥操采集"，这条链路几乎零成本扩数据；相比"关节映射 IK"，表面约束保证了接触和姿态语义不丢，生成的动作天然满足限位/不穿地，sim2real 更友好。论文提到 OmniContact 发布的 G1 重定向数据就是内部版 UMR 产出的。

---

## 八、缓存与复用规则（什么时候重算什么）

| 改了什么 | 重算阶段 ① 对应？ | 重算阶段 ② 优化？ |
|----------|------------------|------------------|
| 换一条动作（同体型、同机器人） | 否，直接复用 slots | 是 |
| 换机器人 / 改 MJCF / 改 T-pose | 是 | 是 |
| 换人体性别或 betas（体型） | 是（模板哈希变了） | 是 |
| 只改优化权重/步长/滤波参数 | 否 | 是（可加 `--force-retarget` 覆盖已有结果） |
| 重跑 pipeline | 自动检测，存在即复用 | 已有结果默认跳过 |

相关产物路径：

```text
data/correspondence_<robot>_<template>.npz                    # 阶段①采样数据（含原始mesh）
output/correspondence_<robot>_<template>/correspondence_slots_final.npz  # 阶段①成果
output/<robot>_retarget/<seq>_<robot>.npz                     # 阶段②成果（qpos）
output/batch_retarget/<robot>/                                # 批量产物 + batch_summary.json
```

---

## 九、用本次日志做一遍逐段对照

```text
1. [SMPLXModel] loading ... SMPLX_NEUTRAL.pkl                 加载参数化人体
2. [Dataset] Added smplx_neutral_betas_... verts=10475        人体 canonical 网格
3. [Dataset] Added unitree_g1 verts=192938 first_hit          G1 T-pose 外壳采样
4. saved correspondence dataset ... points=(2, 4096, 3)       阶段①采样落盘
5. train_correspondence_template_residual_ae.py ... 500 ep    训练共享 slot 排序
6. Saved final checkpoint and correspondence slots            阶段①完成（之后永久复用）
7. frames=3945 fps=30                                         读取跳舞序列
8. human_height=1.848 robot_height=1.322 smpl_scale=0.7154    身高归一化
9. segments=19, selected slots=456                            表面点分段加权
10. GroundContact / Penetration / SelfContact / JointLimit    装配物理约束
11. retarget forward 3945/3945 [00:30, 128it/s]               逐帧 GPU 优化
12. TrajectoryFilter LQR accel/jerk 下降                      轨迹平滑
13. saved ... .npz qpos=(3945,36) cost mean=0.0659            输出机器人轨迹
14. GLFW UI viewer ... Space/A/D/Q/E                          可视化验收
```

---

## 十、关键代码文件地图

| 文件 | 职责 |
|------|------|
| [humanoid_retarget_pipeline.py](../UMR/scripts/humanoid_retarget_pipeline.py) | 总编排：串起采样→训练→优化→可视化，管理缓存与子进程 |
| [build_correspondence_ae_dataset.py](../UMR/scripts/build_correspondence_ae_dataset.py) | 阶段①：人体/机器人 canonical 表面采样与数据集构建 |
| [train_correspondence_template_residual_ae.py](../UMR/scripts/train_correspondence_template_residual_ae.py) | 阶段①：残差自编码器学共享 slot 排序 |
| [retarget_smpl_to_humanoid_surface_vector.py](../UMR/scripts/retarget_smpl_to_humanoid_surface_vector.py) | 阶段②：逐帧表面优化 + 接触/约束 + LQR 滤波（SMPL-X 主线） |
| [retarget_body_segment_surface.py](../UMR/scripts/retarget_body_segment_surface.py) | 19 个身体段的划分、每段采样数与位置/法向权重 |
| [surface_sampling.py](../UMR/scripts/surface_sampling.py) | 表面点采样（first-hit 射线、曲率加权等） |
| [mujoco_geom_surface.py](../UMR/scripts/mujoco_geom_surface.py) | 从 MuJoCo geom/mesh 提表面 |
| [smplx_model_loader.py](../UMR/scripts/smplx_model_loader.py) / [smpl_surface_retarget_common.py](../UMR/scripts/smpl_surface_retarget_common.py) | SMPL-X 加载/前向、各动作源 npz/bvh 的统一读取 |
| [visualize_robot_retarget_result.py](../UMR/scripts/visualize_robot_retarget_result.py) | 读结果 npz，GLFW viewer 回放 |
| `humanoid_retarget_pipeline_{batch,hsi_hoi,character,nr,adapt}.py` | 批量、人-场景/物交互、角色动画、FBX/BVH、AdaPT 等变体入口 |
| [robot_configs/humanoid_retarget_unitree_g1_example.json](../UMR/robot_configs/humanoid_retarget_unitree_g1_example.json) | 机器人侧唯一配置：MJCF、中心点、T-pose、关节限位 |

---

## 十一、局限与注意事项

- **依赖 canonical 对齐质量**：T-pose 摆得准不准直接决定对应质量，这是换机器人时唯一需要人工做的事（UMR Studio 辅助复制 `tpose_qpos`）；
- **体型相关**：对应是绑人体模板的，儿童/特殊体型（不同 betas）要另建模板，BONES-SEED 为每个体型 Axxx 各复用一套；
- **优化非解析全局最优**：逐帧 warm start 可能被奇异帧带偏，所以批量模式默认开双向 warm start + 动态规划；`cost mean` 异常大的片段应在 `batch_summary.json` 里排查；
- **手指/面部等细节不保留**：slot 权重里 hand 权重低、G1 无手指，精细操作要靠 HOI 变体的末端高权重 + 物体接触约束；
- **输入必须是 SMPL-X**：OmniContact 的 BVH 因官方转换器未开源暂不直接支持；LAFAN1 需先用 `lafan_to_smplx` 转换。

---

## 十二、参考

- 论文：<https://arxiv.org/abs/2609.02134>
- 项目主页（含效果视频）：<https://hanyang9.github.io/UMR/>
- UMR Studio（T-pose 配置）：<https://hanyang9.github.io/UMR/umr_studio.html>
- 安装部署见同目录 [01-UMR安装部署指南-v1.0.md](01-UMR安装部署指南-v1.0.md)
