# LIBERO 官方评测记录（第二阶段）

对 OpenVLA LIBERO 微调模型在 **LIBERO 官方 benchmark** 上的成功率评测记录。
评测循环、成功率统计、rollout 视频全部走官方 `repos/openvla/.../run_libero_eval.py`，
由薄封装脚本 [`scripts/02_libero_eval.py`](../../../scripts/02_libero_eval.py) 复用调用。

## 评测配置

| 项 | 值 |
|---|---|
| 模型 | `models/openvla-7b-finetuned-libero-spatial`（ModelScope 微调版软链） |
| 任务套件 | `libero_spatial`（10 个任务） |
| 每任务 rollout | 1 trial（smoke test 规模） |
| unnorm_key | `libero_spatial`（官方脚本按 task_suite_name 自动设置） |
| attn 后端 | `sdpa` |
| 加载方式 | 离线：`trust_remote_code=False` + `local_files_only=True` |

## 本 checkpoint 可测试的任务与场景

> 该微调权重的 `dataset_statistics.json` **仅含 `libero_spatial` 一个 `unnorm_key`**，脚本又自动
> 令 `unnorm_key = task_suite_name`。因此**它只能有效评测 `libero_spatial` 这一套**；换成
> `libero_object / libero_goal / libero_10 / libero_90` 会因反归一化找不到 key 而报错或输出乱动作，
> 需改用对应套件的专用微调权重。

### 场景特征（libero_spatial = 空间推理）

同一物体（黑碗 black bowl）、同一目标（放到盘子 plate 上），**只改变黑碗与周围参照物的空间关系**
（在某物旁边 / 上面 / 抽屉里 / 桌面中央等），专门考察模型的**空间定位与指代理解**能力。
场景固定为一张桌面，含盘子、碗、饼干盒、酥皮模、木柜（含抽屉）、灶台等参照物。

### 10 个任务（脚本一次跑完整套，无单任务开关）

| # | 任务指令 | 空间关系 |
|---|---|---|
| 0 | pick up the black bowl between the plate and the ramekin and place it on the plate | 盘子与酥皮模之间 |
| 1 | pick up the black bowl next to the ramekin and place it on the plate | 酥皮模旁边 |
| 2 | pick up the black bowl from table center and place it on the plate | 桌面中央（本次唯一失败项） |
| 3 | pick up the black bowl on the cookie box and place it on the plate | 饼干盒上方 |
| 4 | pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate | 木柜顶层抽屉内 |
| 5 | pick up the black bowl on the ramekin and place it on the plate | 酥皮模上方 |
| 6 | pick up the black bowl next to the cookie box and place it on the plate | 饼干盒旁边 |
| 7 | pick up the black bowl on the stove and place it on the plate | 灶台上方 |
| 8 | pick up the black bowl next to the plate and place it on the plate | 盘子旁边 |
| 9 | pick up the black bowl on the wooden cabinet and place it on the plate | 木柜上方 |

## 结果

**成功率 9/10 = 90.0%**（详见 `EVAL-libero_spatial-2026_09_22-1trial-90pct.txt`）。
唯一失败任务：`pick up the black bowl from table center and place it on the plate`。
与官方公布水平（libero_spatial 50 trials 约 84%）量级一致；本次为 1 trial 抽样，仅作链路验证。

## 复现命令

```bash
cd /home/creator/ws/100-openVLA && source .venv/bin/activate
python scripts/02_libero_eval.py --num-trials-per-task 1 --run-id-note smoke-test
# 正式评测（每任务 50 trials，贴近官方口径）：
# python scripts/02_libero_eval.py --num-trials-per-task 50
```

## 跑通所需的环境/兼容修复（均在薄封装或环境中完成，未改子模块源码）

1. **LIBERO editable 失效**：`libero_source_path.pth` 残留改名前旧路径，改为
   `/home/creator/ws/100-openVLA/repos/LIBERO` 后 `import libero` 恢复。
2. **微调 checkpoint auto_map 联网卡死**：改为离线加载（`trust_remote_code=False` +
   `local_files_only=True`），复用第一阶段修复。
3. **wandb 与 protobuf 版本不匹配**：`--use-wandb` 关闭时注入空 stub 跳过顶层 `import wandb`。
4. **PyTorch>=2.6 `torch.load` 默认 `weights_only=True`**：LIBERO init_states 含 numpy 对象，
   将默认改回 `weights_only=False`。
5. **robosuite 1.4.x × mujoco>=3 的 `mj_fullM` 签名变更**：注入 numpy 展开稀疏惯性矩阵的兼容
   shim（不降级 mujoco，避免破坏第一阶段 sim 的 mink IK）。

## 可视化查看（有显示屏时）

官方评测是「离屏渲染 → 存 MP4」，本身没有实时窗口。有显示屏时直接**回放已生成的 MP4**
即可（评测已渲染好，零改动）。需在**有图形界面的桌面终端**里执行（`echo $DISPLAY` 应非空；
纯 SSH/headless 会话开不了窗口）。

```bash
cd /home/creator/ws/100-openVLA/repos/openvla/rollouts/<日期>
# 单个回放（256×256 较小，放大到 512 更清晰）
ffplay -x 512 -y 512 "<episode文件>.mp4"
# 或用 mpv 连续循环播放当天全部
mpv --autofit=512x512 --loop-playlist *.mp4
```

- 文件名自带 `success=True/False` 与任务名，可直接定位失败任务重点看。
- MP4 内容即**喂给模型的 agentview 画面**（256×256，官方固定分辨率），非第三视角高清图。

## 文件说明

- `EVAL-libero_spatial-2026_09_22-1trial-90pct.txt`：官方格式评测日志（逐任务成败 + 总成功率）。
- `console_smoke_run.txt`：控制台输出，含各兼容补丁 banner 与逐集 rollout 记录（诊断证据）。
- rollout 视频（`.mp4`）按仓库 `.gitignore` 策略默认不入库，保留在本地
  `repos/openvla/rollouts/`；如需归档演示片段可 `git add -f` 单独强制添加。
