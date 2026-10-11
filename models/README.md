# 模型权重统一管理目录

本目录集中存放工作区所有项目的模型权重，通过项目内软链或脚本引用。**不纳入 git 版本管理**（见根级 `.gitignore`）。

## 模型索引

| 模型 | 来源 | 用途 | 引用项目 |
|------|------|------|----------|
| `Cosmos-Predict2.5-2B` | NVIDIA / ModelScope `nv-community/Cosmos-Predict2.5-2B` | 扩散世界基础模型（图像/视频/世界状态生成） | `102-cosmos` |
| `Cosmos3-Edge` | NVIDIA / ModelScope `nv-community/Cosmos3-Edge` | 边缘多模态 VLM（理解 + 图像/视频生成） | `102-cosmos` |
| `Cosmos3-Edge-Policy-DROID` | NVIDIA / ModelScope `nv-community/Cosmos3-Edge-Policy-DROID` | 机器人边缘策略模型 | `102-cosmos` |
| `pi0_base` | Physical-Intelligence / lerobot | π0 基础策略权重 | `101-pi-serial-models` |
| `pi0_libero_base` | Physical-Intelligence / lerobot | π0 LIBERO 任务基础权重 | `101-pi-serial-models` |
| `pi05_libero_base` | Physical-Intelligence / lerobot | π0.5 LIBERO 任务基础权重 | `101-pi-serial-models` |
| `openvla-7b` | ModelScope `zixiaoBios/openvla-7b` | OpenVLA base 模型（unnorm_key=bridge_orig） | `100-openVLA` |
| `openvla-7b-finetuned-libero-spatial` | ModelScope `superpeach/openvla-7b-finetuned-libero-spatial` | OpenVLA LIBERO-spatial 微调 checkpoint（unnorm_key=libero_spatial） | `100-openVLA` |

## 约定

- 新模型下载后统一放入本目录，命名与 ModelScope / HuggingFace 仓库名一致
- 各项目通过**相对路径软链**或**脚本变量**引用本目录，避免模型文件重复存储
- 模型下载命令记录在对应项目的 README 中