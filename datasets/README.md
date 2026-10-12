# 数据集统一索引

实体数据留在各项目目录内（用户确认，避免迁移带来回归风险），本目录只做**位置索引**；新数据集下载后仍放项目内，并在此登记一行。

## 索引

| 数据集 | 位置 | 格式 / 规模 | 引用项目 |
|---|---|---|---|
| ALOHA sim（insertion / transfer_cube × human / scripted，共 4 个） | `101-pi-serial-models/data/aloha_sim_*` | lerobot v2（HF parquet：`data/chunk-000` + `meta` + `videos`），各 68~88M | `101-pi-serial-models`（ACT / pi0 训练） |
| LIBERO（10 任务系列） | `101-pi-serial-models/data/libero` | lerobot v2，1.9G | `101-pi-serial-models`（pi0 / pi0.5 评估） |
| pusht | `101-pi-serial-models/data/pusht` | lerobot v2，9.1M | `101-pi-serial-models` |
| 人体 / 手势演示视频 | `501-human-dexterous-teleop/data/videos` | 视频，6.9M | `501-human-dexterous-teleop`（demo 输入） |

## 工作区外的数据依赖

- LIBERO 运行时：`100-openVLA/.libero/config.yaml`（`LIBERO_CONFIG_PATH`）；全局 `~/.libero/`（工作区外）另有配置与任务数据
- 模型权重索引见 [models/README.md](../models/README.md)；仿真资源见 `assets/`
- 各独立克隆仓库（UMR、4D-Humans 等）自带的数据不在此登记
