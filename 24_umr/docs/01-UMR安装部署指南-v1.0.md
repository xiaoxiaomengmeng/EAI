# UMR 安装部署指南（5090 机 · py3.12 · conda · SMPL-X → 人形机器人动作重定向）

> 版本 v1.0 · 2026-09-10 · 目标硬件：RTX 5090 32GB（sm_120），Ubuntu 22.04，驱动 580
>
> 对应仓库：[UMR](https://github.com/hanyang9/UMR)（*Unified Motion Retargeting for Humanoids with Learned Point Cloud Correspondence*，arXiv 2609.02134）。本文在官方 README 安装步骤基础上，针对 RTX 5090 + 国内网络环境给出可直接执行的版本，**请按顺序执行并自行验证**。

---

## 〇、UMR 是干什么的

UMR 把**运动人体的外表曲面（点云）**当作人体动作与人形机器人之间的共享接口。它不是直接把“人的关节角”翻译成“机器人的关节角”，而是先学习两套身体表面的语义对应，再在物理与运动学约束下求解机器人动作：

```text
异构人体动作源
SMPL-X / LAFAN1 / BONES-SEED(SOMA) / GRAIL / OMOMO / BVH/FBX / MimicKit
        │ ① 数据源 adapter：统一为 SMPL-X 参数序列（姿态、根位移、体型、帧率）
        ▼
SMPL-X 人体网格 + 机器人 MJCF、T-pose、关节限位
        │ ② canonical 表面采样：两侧各取固定数量的外表面点及法向
        ▼
阶段①：点云对应学习（离线、一次性、可缓存）
        │ ③ 残差自编码器为人体/机器人点云学习共享 slot 顺序
        ▼
有序 source↔robot 对应关系（同一人体模板 + 同一机器人可复用）
        │ ④ 按每帧人体姿态更新人体表面；以对应关系为目标优化机器人 qpos
        ▼
阶段②：对应引导的逐帧重定向（接触、防穿、限位、连续性约束）
        │ ⑤ 轨迹平滑、保存元数据、MuJoCo 回放
        ▼
机器人 qpos 序列（仿真回放 / 模仿学习或控制训练数据）
```

### 模块组成与职责

| 模块 | 主要脚本/目录 | 做什么 | 关键输入与产物 |
|------|---------------|--------|----------------|
| **流水线编排与缓存** | [`scripts/humanoid_retarget_pipeline.py`](../UMR/scripts/humanoid_retarget_pipeline.py) | 总入口；加载并合并配置，按 `build → train → retarget → view` 调度下游模块，并复用已兼容的中间结果。 | 输入机器人配置与动作源 defaults；输出整条流水线的缓存和结果。 |
| **数据源适配与人体模型** | `sample_data/<source>/`、`smpl/`、`scripts/smplx_model_loader.py` | 将不同来源的动作整理为统一的 SMPL-X/SMPL 参数序列，并通过正运动学生成随帧变化的人体网格。 | 输入 BVH、SOMA、SMPL-X 等；统一为姿态、`transl`、`betas` 等人体参数。 |
| **机器人配置** | [`robot_configs/`](../UMR/robot_configs) 与 `humanoid_retarget_defaults*.json` | 定义目标机器人的 MJCF 路径、点云对齐中心、T-pose、额外关节限位，以及不同运动源的默认参数。 | 输入 MJCF 和配置；确定机器人形状、参考姿态和求解边界。 |
| **canonical 表面采样** | [`scripts/build_correspondence_ae_dataset.py`](../UMR/scripts/build_correspondence_ae_dataset.py)、`mujoco_geom_surface.py`、`surface_sampling.py` | 在人体标准姿态和机器人 T-pose 下提取外表面，采样点云、法向和 mesh；机器人侧会过滤内部几何。 | 输出 `data/correspondence_*.npz`，供对应学习和可视化使用。 |
| **点云对应学习** | [`scripts/train_correspondence_template_residual_ae.py`](../UMR/scripts/train_correspondence_template_residual_ae.py) | 训练模板残差自编码器，让两套无序点云获得同一套有序 slot；利用 Chamfer、点间排斥和测地邻接约束保持覆盖与拓扑。 | 输出 `correspondence_slots_final.npz`：人体—机器人对应点、法向等；换动作时直接复用。 |
| **身体分段与权重** | [`scripts/retarget_body_segment_surface.py`](../UMR/scripts/retarget_body_segment_surface.py) | 将人体表面划分为躯干、头、四肢等功能段，为各段分配采样数、位置误差权重和法向误差权重。 | 让优化重点保留脚部、躯干、手臂等对动作语义重要的区域。 |
| **对应引导的重定向求解** | [`scripts/retarget_smpl_to_humanoid_surface_vector.py`](../UMR/scripts/retarget_smpl_to_humanoid_surface_vector.py) | 逐帧优化浮动基座和机器人关节角，使机器人对应点的位置、表面朝向与人体匹配。 | 输出初始 `qpos` 轨迹；G1 示例为浮动基座 7 维 + 29 关节维。 |
| **接触、安全与连续性约束** | 主求解器中的 GroundContact、GroundPenetration、SelfContact、JointLimit 等项 | 约束脚部接地、防止穿地/自穿、遵守关节限位，并限制相邻帧关节与根部变化；以上一帧结果 warm start。 | 使轨迹既贴近人体动作，又满足机器人可执行性与时序连续性。 |
| **后处理、批处理与可视化** | `humanoid_retarget_pipeline_batch.py`、[`scripts/visualize_robot_retarget_result.py`](../UMR/scripts/visualize_robot_retarget_result.py) | 对轨迹做 LQR 平滑；批量处理多条动作；用 MuJoCo 回放并保存可追溯元数据。 | 输出 `output/<robot>_retarget/*.npz`，可独立回放或转换为训练数据。 |

**核心复用边界**：只要人体模板（例如性别/`betas`）、目标机器人、T-pose 和采样配置不变，阶段①的对应关系就只需学习一次；之后替换动作序列时，通常只运行阶段②。**接入新机器人**时，准备其 MJCF，并在 UMR Studio 中调出合理的 T-pose 后写入机器人配置即可，无需手工标注人体—机器人关节映射。支持的运动源与机器人配置见仓库 README 表格（G1、H2、Booster K1、EngineAI T800、MimicKit 等配置已自带）。

---

## 一、目录约定（重要）

按本项目惯例，第三方仓库一律以 git submodule 管理，Python 环境放在 submodule 外部（conda 环境 `umr`，与 submodule 目录解耦）：

```text
24_umr/
├── UMR/          # 第三方 submodule（保持纯净，不要在里面建 .venv）
└── docs/         # 本文所在目录

conda env: umr    # miniconda3/envs/umr，Python 3.12
```

---

## 二、环境要求

| 项 | 要求 | 本机情况 |
|----|------|----------|
| GPU | CUDA 显卡（点云对应学习走 torch CUDA） | RTX 5090 32GB（sm_120 / Blackwell）✅ |
| 驱动 | NVIDIA ≥ 570（torch 2.9 wheel 自带 CUDA 12.8 运行库） | 580 ✅ |
| Python | **3.12.x** | 3.12.13 ✅ |
| 磁盘 | ~6GB（torch CUDA wheel + 依赖；数据/输出另计） | 充足 |
| 构建工具 | cmake、g++（个别包无预编译 wheel 时源码编译用） | 已装 ✅ |
| 网络 | GitHub（smplx 是 git 依赖）、PyPI（本机已配 aliyun pip 镜像）、SMPL-X 官网 | 已配 SSH insteadOf |
| 可选 | Node.js ≥ 18（仅 NR FBX/BVH 数据源解析需要，Quick Start 用不到） | 按需安装 |

关键依赖（来自 [requirements-umr.txt](file:///home/creator/ws/24_umr/UMR/requirements-umr.txt)）：mujoco 3.2~3.3、warp-lang≥1.6、trimesh + embreex（Embree 射线求交）、cholespy（稀疏 Cholesky 求解）、coacd==1.0.5（物体凸分解，仅 HSI/HOI 预处理需要）、clarabel、obj2mjcf、smplx（git 依赖）、py-soma-x。

---

## 三、步骤 1：创建 conda 环境

本机实测采用 conda（与官方 README 一致），环境名 `umr`：

```bash
conda create -n umr python=3.12 pip -y
conda activate umr
which python    # 应指向 /home/creator/miniconda3/envs/umr/bin/python
```

> 每次新开 shell 都要先 `conda activate umr` 再执行后续命令。
>
> 备选：也可用 venv（`python3.12 -m venv .venv && source .venv/bin/activate`，建在 `24_umr/` 顶层、不进 submodule）。两种方式二选一即可，下文统一假设 conda 环境已激活，直接用 `python`/`pip`。

---

## 四、步骤 2：安装 PyTorch（与官方 README 的关键差异）

官方 README 写的是：

```bash
# 官方原命令 —— 5090 上不要用！
python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.4.1
```

`torch 2.4.1 + cu121` 是 2024 年的构建，最高只覆盖到 sm_90（Hopper），**不包含 5090 的 sm_120（Blackwell）kernel**，装上去一跑 CUDA 就报：

```text
CUDA error: no kernel image is available for execution on the device
```

PyTorch 从 2.7 起正式支持 Blackwell。本机 microduck 环境已验证 **PyPI 默认的 torch 2.9.1（wheel 自带 nvidia-*-cu12 系列 CUDA 12.8 运行库）在 5090 上可用**，requirements 未锁定 torch 版本，因此直接装：

### 推荐：复用本机 uv 缓存（免下载 899MB）

本仓库的 microduck 环境（`100-microduck/.venv`）此前 `uv sync` 时已下载过 torch 2.9.1 及全部 nvidia-*-cu12 依赖，内容缓存在 `~/.cache/uv`。pip 与 uv 是两套独立缓存，pip 看不到这些文件、会重新从镜像下载（wheel 899.7MB，aliyun 节点实测只有 ~130kB/s，要近 2 小时）；用 **uv pip** 安装则直接命中缓存、硬链接秒装：

```bash
# 确保 conda 环境 umr 已激活：which python 指向 miniconda3/envs/umr

# uv 二进制在 microduck 的 venv 里（也可自行 pip install uv）
# uv pip 会自动检测当前激活的 conda 环境（CONDA_PREFIX），包装进 umr
~/ws/100-microduck/.venv/bin/uv pip install "torch==2.9.1"
```

uv 只在解析元数据时走少量网络，包本体与 CUDA 依赖全部本地硬链接，不重复下载。前提是 `~/.cache/uv` 未被清理；可用下面命令确认缓存里有 torch：

```bash
ls ~/.cache/uv/archive-v0/ | head; find ~/.cache/uv -maxdepth 3 -name "torch-2.9.1.dist-info"
```

### 备选：普通 pip 安装（uv 缓存不存在时）

```bash
pip install "torch==2.9.1"
```

> 本机 pip 默认走 aliyun 镜像，PyPI 版 torch wheel 本身就捆绑 CUDA 库，不需要再加 `--index-url`，也不需要系统装 CUDA Toolkit。大包下载慢时换清华源通常更快：
>
> ```bash
> pip install "torch==2.9.1" -i https://pypi.tuna.tsinghua.edu.cn/simple
> ```

安装后**必须先验证 GPU 可见**，再装其余依赖：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 期望输出：2.9.1+cu128 True NVIDIA GeForce RTX 5090
# 若同时打印 Failed to initialize NumPy 告警属正常：numpy 尚未安装，
# 步骤 3 装 requirements-umr.txt 时会带上，不影响 GPU 验证
```

若输出 `False`：先 `nvidia-smi` 确认驱动正常；`which python` 确认指向 `miniconda3/envs/umr/`（conda 环境已激活），而不是系统/base Python。

---

## 五、步骤 3：安装其余依赖

继续用 uv pip（torch 之外的包 uv 缓存里没有，仍走 aliyun，但都是小包）：

```bash
cd ~/ws/24_umr
~/ws/100-microduck/.venv/bin/uv pip install -r UMR/requirements-umr.txt
```

不想依赖 microduck 环境里的 uv 时，`pip install uv` 后直接用 `uv`，或退回普通 `pip install -r UMR/requirements-umr.txt`，效果一致（只是大包下载无缓存加速）。

### 已知坑：coacd==1.0.5 没有 cp312 预编译 wheel

requirements 锁定的 `coacd==1.0.5` 只有较老 Python 的二进制包，pip 会转而**源码编译**（需要 cmake/g++/网络拉取其子模块，耗时几分钟）。

- **编译成功**：什么都不用做。
- **编译失败**：coacd 只用于 HSI/HOI 任务中物体的凸分解预处理，Quick Start（LAFAN1→G1）完全用不到。先装其余包、再装新版（新版有 abi3 wheel，免编译）：

  ```bash
  grep -v '^coacd' UMR/requirements-umr.txt | pip install -r /dev/stdin
  pip install "coacd==1.0.14"
  ```

- 之后真正处理 GRAIL/OMOMO 数据前，再用 `python -c "import coacd"` 验证即可。

### smplx 是 git 依赖

requirements 中的 `smplx @ git+https://github.com/ZhengyiLuo/smplx.git` 会现场 clone。本机已配置 `git@github.com:` 的 insteadOf，有 SSH key 即可；若 GitHub 不通需先配代理，或手动 `git clone` 验证连通性。

### Node.js（仅 NR FBX/BVH 数据源需要）

使用 `humanoid_retarget_pipeline_nr.py` 前需 Node.js ≥ 18（仅用于 Three.js 解析 FBX，与可视化无关）：

```bash
# Ubuntu 22.04 源里是 node 18，直接装即可
sudo apt install nodejs
node --version
```

---

## 六、步骤 4：放置 SMPL-X 人体模型（必须，仓库不提供）

UMR 的人体动作以 SMPL-X 表示，模型文件受许可限制**不随仓库分发**，需自行下载：

1. 注册并接受协议：<https://smpl-x.is.tue.mpg.de/>
2. 下载 SMPL-X 模型（`.npz` 或 `.pkl` 均可），**至少放 neutral 模型**到：

```text
~/ws/24_umr/UMR/smpl/SMPLX_NEUTRAL.npz   # 或 SMPLX_NEUTRAL.pkl
```

处理特定性别序列（如 OMOMO 女性动作）时再补 `SMPLX_FEMALE.*`、`SMPLX_MALE.*`。

> 仓库 `smpl/` 目录下目前只有一个 README.md，属正常，模型要自己放进去。

---

## 七、步骤 5：安装后自检

```bash
cd ~/ws/24_umr/UMR
python - <<'PY'
import torch, mujoco, trimesh, cholespy, clarabel, smplx, warp
print("torch :", torch.__version__, torch.cuda.is_available())
print("mujoco:", mujoco.__version__)
print("warp  :", warp.__version__)
warp.init()                      # 首次会编译少量 kernel，正常
import os
print("SMPLX neutral exists:",
      os.path.exists("smpl/SMPLX_NEUTRAL.npz") or os.path.exists("smpl/SMPLX_NEUTRAL.pkl"))
PY
```

全部打印且 SMPLX 行是 `True` 即环境就绪。embree/embreeex 首次 import 可能打印告警，不影响使用。

---

## 八、步骤 6：Quick Start（LAFAN1 跳舞 → Unitree G1）

> **前置：先拉 Git LFS 数据**。UMR 的示例动作、机器人 mesh、数值数组全部在 LFS 上，clone 时若未装 git-lfs，工作区里只是指针文件，运行必报 `invalid load key, 'v'`。首次运行前执行一次：
>
> ```bash
> conda install -n umr -c conda-forge git-lfs -y   # 已装可跳过
> cd ~/ws/24_umr/UMR
> git lfs install
> git lfs pull
> git lfs ls-files    # 无 * 标记即为完整
> ```

仓库自带示例序列 `sample_data/lafan1_smplx/dance1_subject2.npz` 和 G1 模型 `assets/g1/g1_29dof.xml`，无需额外下载数据：

```bash
cd ~/ws/24_umr/UMR
python scripts/humanoid_retarget_pipeline.py \
  --config robot_configs/humanoid_retarget_unitree_g1_example.json
```

流程：构建/复用学到的点云对应 → 对应引导优化 → 自动打开 MuJoCo viewer 播放重定向结果（需要本机图形桌面；纯 SSH 用下文的离线方式）。

结果文件保存在：

```text
output/unitree_g1_retarget/dance1_subject2_smplx_unitree_g1.npz
```

### 单独回放结果

```bash
python scripts/visualize_robot_retarget_result.py \
  --result output/unitree_g1_retarget/dance1_subject2_smplx_unitree_g1.npz \
  --play
```

无显示环境时先确认脚本是否支持离屏渲染参数（以 `--help` 为准），或在有桌面的机器上回放该 npz。

### 其他运动源（按需，均需先按 adapter 指南备数据）

除 LAFAN1 Quick Start 外，其他数据源的动作/资产都因或许可原因**不随仓库分发**，运行前必须先读 `sample_data/<source>/README.md` 把数据摆好，否则会在模板构建或读序列阶段报 `FileNotFoundError`。

```bash
# BONES-SEED / SOMA
# 前置（两类数据，详见 sample_data/bones-seed/README.md）：
#  ① 公开 SOMA-X 资产（无需审批）：
#     hf download nvidia/SOMA-X SOMA_neutral.npz SOMA_template_rig.usda \
#       SOMA_procedural_transforms.json MHR/SOMA_wrap_lod1.obj \
#       MHR/base_body_lod1.obj MHR/mhr_model_lod1.pt \
#       --local-dir sample_data/bones-seed/soma_assets
#     国内可加 HF_ENDPOINT=https://hf-mirror.com
#  ② gated BONES-SEED 数据集（HF 页面接受许可 + hf auth login）：
#     下载 soma_uniform.tar.gz 和 soma_shapes/**，解压摆放到
#     motions_uniform/bvh/ 与 shapes/ 下；缺任一部分分别会在
#     阶段①（缺 SOMA_neutral.npz）或阶段②（缺 flip_090_002__A304_M.bvh）报错
#  ③ 必做：把 USDA rig 合并回 npz（见下方"SOMA-X 版本错位坑"小节，
#     直接跑会报 Complete SOMA bind/T-pose assets not found）
python scripts/humanoid_retarget_pipeline.py \
  --config robot_configs/humanoid_retarget_unitree_g1_example.json \
  --defaults humanoid_retarget_defaults_bones_seed.json

# GRAIL 人-场景交互（前置：按 sample_data/grail/README.md 下载数据集，物体 mesh 需先用 CoACD 凸分解）
python scripts/humanoid_retarget_pipeline_hsi_hoi.py \
  --config robot_configs/humanoid_retarget_unitree_g1_example.json \
  --defaults humanoid_retarget_defaults_hsi_hoi_grail.json

# AdaPT 身体+球拍（无需额外参数）
python scripts/humanoid_retarget_pipeline_adapt.py
```

各数据源的本地目录布局要求见 `sample_data/<source>/README.md`。

### SOMA-X 版本错位坑（BONES-SEED 必做，2026-09-12 实测）

**现象**：7 个 SOMA-X 文件都下载好了，跑 BONES-SEED 仍报：

```
FileNotFoundError: Complete SOMA bind/T-pose assets not found. Expected SOMA_neutral.npz
with bind_shape, bind_pose_world, t_pose_world, and skinning weights ...
```

**根因**：HF `nvidia/SOMA-X` main（v0.3.0，2026-09-02）把 rig 数据（`bind_shape`、`bind_pose_world`、`t_pose_world`、关节层级、CSC 蒙皮权重等 11 个键）从 `SOMA_neutral.npz` 迁到了 `SOMA_template_rig.usda`（UsdSkel），npz 只剩形状模型/拓扑等约 29 个键。而 UMR 首版（2026-09-07）的 `scripts/soma_source.py` 仍按旧格式直读 npz。HF 上旧版 tag（v0.2.2/v0.2.4）的 npz 实测同样不含这些键，**换 revision 解决不了**。`UMR_SOMA_ASSETS_PATH` 环境变量也无效——问题是文件内容不是搜索路径。

**修复（官方迁移路径）**：py-soma-x 0.3.0 的 `soma.io.load_lod_rig_from_usd` 注释明确说明它返回的就是"过去存在 npz 里的 rig 键，用于和 slim core asset 合并"。下载完成后、首次跑 pipeline 前，在 **UMR 仓库根目录、umr 环境**执行一次：

```bash
# 先备份原始 slim npz（出问题可直接重下，备份非必需）
cp sample_data/bones-seed/soma_assets/SOMA_neutral.npz \
   sample_data/bones-seed/soma_assets/SOMA_neutral.npz.slim.bak

python - <<'PY'
import numpy as np
from soma.io import SOMA_NEUTRAL_RIG_KEYS, load_lod_rig_from_usd

A = "sample_data/bones-seed/soma_assets"
core = dict(np.load(f"{A}/SOMA_neutral.npz", allow_pickle=False))
rig = load_lod_rig_from_usd(f"{A}/SOMA_template_rig.usda", "mid")  # UMR 用 mid LOD
merged = dict(core)
for k in SOMA_NEUTRAL_RIG_KEYS:                      # 11 个 rig 键
    if k not in merged:
        merged[k] = np.asarray(rig[k])
np.savez_compressed(f"{A}/SOMA_neutral.npz", **merged)
print("keys:", len(merged))
PY
# 期望：keys: 40 或 41（slim 约 29 键 + 11 rig 键，具体随 HF 文件版本略有出入）
```

**验证**（合并后应看到 bind_shape=(18056,3)、110 关节、权重 (18056,110)）：

```bash
python - <<'PY'
import numpy as np
d = np.load("sample_data/bones-seed/soma_assets/SOMA_neutral.npz", allow_pickle=False)
req = {"bind_shape", "bind_pose_world", "t_pose_world", "joint_names",
       "skinning_weights_data", "skinning_weights_indices",
       "skinning_weights_indptr", "skinning_weights_shape"}
assert req.issubset(d.files), req - set(d.files)
print("ok:", d["bind_shape"].shape, d["joint_names"].shape, len(d.files), "keys")
PY
```

注意：合并后的 npz 只在本机 UMR 目录使用，不要提交/再分发（SOMA-X 资产本身按 NVIDIA 许可使用）。若以后 UMR 更新适配了新格式，以其官方说明为准。

---

## 九、批量重定向

```bash
python scripts/humanoid_retarget_pipeline_batch.py \
  --config robot_configs/humanoid_retarget_unitree_g1_example.json \
  --batch-config humanoid_retarget_defaults_batch.json
# BONES-SEED 用 humanoid_retarget_defaults_batch_bones_seed.json
```

- 结果在 `output/batch_retarget/<robot-name>/`，`batch_summary.json` 记录每个片段状态；
- 批量模式默认启用**双向 warm start + 动态规划**，对奇异点更鲁棒，官方推荐大规模重定向使用；
- 常用参数：`--workers N`（重定向并行）、`--correspondence-workers N`（对应准备并行）、`--retarget-gpus`（分配 GPU）、`--force-retarget`（重建已有结果）、`--recursive/--pattern`（发现输入文件）。

---

## 十、换成自己的机器人

1. 复制一份配置，如 `robot_configs/humanoid_retarget_unitree_g1_example.json`；
2. 修改 `"name"` 和 `"xml"`（指向机器人 MJCF），按需调整 `"joint_limits"`；
3. 打开 [UMR Studio](https://hanyang9.github.io/UMR/umr_studio.html)：载入机器人 asset 文件夹 → 选 MJCF → 调整成 T-pose → 点 **Copy T-pose Config**，把 `tpose_qpos` 粘进配置；
4. 用新配置跑 pipeline 即可，无需手工关节映射。

---

## 十一、常见问题

| 现象 | 排查 |
|------|------|
| `CUDA error: no kernel image is available` | 装成了旧 torch（如 README 的 2.4.1+cu121），不含 sm_120 kernel；`pip install --force-reinstall "torch==2.9.1"`（PyPI 版，cu128 运行库） |
| `torch.cuda.is_available()` 为 False | `nvidia-smi` 确认驱动；`which python` 确认在 `miniconda3/envs/umr`（`conda activate umr`）；驱动版本过旧时升级到 ≥570 |
| torch 报 `Error 804: forward compatibility was attempted on non supported HW`（GeForce 上） | 驱动升级后未重启：用户态 libcuda（新）与内核 nvidia 模块（旧）不匹配。实锤：`cat /proc/driver/nvidia/version` 与 `modinfo nvidia | grep ^version` 版本不一致。`sudo reboot` 后复测即恢复（本机 2026-09-10 实测踩坑） |
| pip 安装 coacd 卡很久或编译失败 | 1.0.5 无 cp312 wheel 走源码编译；失败就改装 `coacd==1.0.14`（见步骤 3），HSI/HOI 之外用不到它 |
| `smplx` 安装时 git clone 失败 | GitHub 网络问题：检查 SSH key / 代理，或手动 clone 后 `pip install <本地路径>` |
| 运行时报找不到 `SMPLX_NEUTRAL` | 模型没放：去 smpl-x.is.tue.mpg.de 下载后放到 `UMR/smpl/SMPLX_NEUTRAL.npz`（或 .pkl） |
| MuJoCo viewer 报 `GLXBadDrawable` / 无法打开窗口 | 需要本机图形桌面；检查 `echo $DISPLAY`；SSH 会话请加 `ssh -X` 或改为在桌面机回放 npz；崩溃后打印的 GLX 错误是窗口被拆的次生错误，可忽略 |
| import 时混进 ROS2 的 py3.10 包 | `~/.bashrc` 的 ROS2 会泄漏 PYTHONPATH：`env -u PYTHONPATH python scripts/...` 运行 |
| pip 下载慢（torch 899MB 只有百来 kB/s） | 首选复用本机 uv 缓存：`~/ws/100-microduck/.venv/bin/uv pip install "torch==2.9.1"`（见步骤 2，硬链接秒装）；缓存不在了就换清华源 `-i https://pypi.tuna.tsinghua.edu.cn/simple` |
| `nvidia-smi` 报 `Driver/library version mismatch` | 驱动刚升级过、内核模块未重载：重启机器，或`sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia`后重新加载（重启最省事） |
| NR FBX 流程报找不到 node | 需 Node.js ≥ 18，`sudo apt install nodejs` |
| 加载示例 npz 报 `UnpicklingError: invalid load key, 'v'`（或文件只有 100 多字节） | UMR 的 `.npz/.npy/.pkl/mesh/.bvh` 全部走 **Git LFS**，clone 时没装 git-lfs 就只拿到指针文件（内容以 `version https://git-lfs...` 开头）。修复：`conda install -n umr -c conda-forge git-lfs -y`，进仓库 `git lfs install && git lfs pull`，用 `git lfs ls-files` 确认无 `*` 未下载标记；LFS 对象在 GitHub 上，国内直连失败需挂代理后重跑 pull |
| BONES-SEED 报 `Complete SOMA bind/T-pose assets not found` 且 `soma_assets/` 目录为空/不存在 | SOMA-X 外部数据未下载：① 公开 7 个文件用 `hf download nvidia/SOMA-X ... --local-dir sample_data/bones-seed/soma_assets`（国内加 `HF_ENDPOINT=https://hf-mirror.com`）；② gated 动作/体型需在 HF 接受 bones-studio/seed 许可后下载并按 `sample_data/bones-seed/README.md` 摆放，缺 BVH 会在阶段②接着报错 |
| BONES-SEED 报 `Complete SOMA bind/T-pose assets not found`，但 7 个 SOMA-X 文件**都已下载** | SOMA-X 版本错位：新版 npz 不含 rig 键（v0.3.0 起迁到 `SOMA_template_rig.usda`），换 HF revision 无用。用 `soma.io.load_lod_rig_from_usd` 把 11 个 rig 键合并回 npz，完整命令与验证见第八节"SOMA-X 版本错位坑"小节（2026-09-12 实测修复） |

---

## 十二、参考链接

- 项目主页：<https://hanyang9.github.io/UMR/>
- 论文：<https://arxiv.org/abs/2609.02134>
- 代码仓库：<https://github.com/hanyang9/UMR>
- SMPL-X 模型：<https://smpl-x.is.tue.mpg.de/>
- UMR Studio（T-pose 配置工具）：<https://hanyang9.github.io/UMR/umr_studio.html>
- 物体凸分解 CoACD：<https://github.com/SarahWeiii/CoACD>
- 支持的数据集：[BONES-SEED](https://huggingface.co/datasets/bones-studio/seed)、[GRAIL](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL)、[LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset)、[OMOMO](https://github.com/lijiaman/omomo_release)





---口播（视频讲解文案）

本节记录定稿的讲解提示词与对应生成的文案。文案用于自行组合的演示视频配音（teaser 图、人体回放录屏、重定向回放录像等画面由剪辑时自行呈现，文案中不出现）。

### 提示词

```text
你是一名技术视频讲解词撰稿人。请为 UMR（Unified Motion Retargeting）项目撰写一段讲解文案。要求：
1. 纯文字段落，不嵌 Markdown 图片、不放链接、不用列表；
2. 不使用"大家好""这段视频"等口播开场，不提录屏、剪辑、截取等制作过程；
3. 内容顺序：先正常介绍 UMR 项目——定位（人体动作统一重定向到人形机器人的开源框架）、解决的问题（传统做法需为每种机器人手工编写关节映射）、核心方法（离线学习人体与机器人的点云对应关系，再经 MuJoCo 优化把 SMPL-X 人体动作平滑迁移到任意人形机器人）、官方总览图含义（纵轴四种输入身体表示、横轴五种目标机器人，同一段动作统一重定向到每个组合）；
4. 再介绍两个演示作为链路验证：第一个是人体动作回放（可视化播放一段 SMPL-X 人体舞蹈动作，验证人体模型、动作数据与图形环境就绪）；第二个是重定向全链路运行——一条命令的流水线依次完成四个阶段：构建人体与机器人的点云对应数据集、训练点云对应网络、以对应关系为约束逐帧物理优化求解机器人关节轨迹（含地面接触与自碰撞规避等约束及轨迹平滑）、最后由 MuJoCo 查看器自动回放并录制结果；要强调这是"计算与求解"而非简单回放，约五分钟的录像是整条链路的最终产物；
5. 语言流畅自然、适合朗读配音，正文不出现文件路径与文件名，结尾点明"输入端+输出端打通"的验收逻辑。
```

### 文案

**标题**：UMR 统一动作重定向：一条命令，让人形机器人学会跳人类的舞

**关键词 / 话题词**：UMR、Unified Motion Retargeting、动作重定向、人形机器人、SMPL-X、MuJoCo、点云对应、LAFAN1、Unitree G1、具身智能

UMR，全称 Unified Motion Retargeting，是一个把人体动作统一重定向到人形机器人的开源框架。它要解决的问题是：如何把同一段人体动作数据，迁移到体型不同、关节数量不同、自由度也不同的各种人形机器人上。传统做法需要为每种机器人手工编写关节映射规则，换一台机器人就要重新调一遍；UMR 的思路是先离线学习一套人体与机器人之间的点云对应关系，再基于这套对应关系，通过 MuJoCo 物理优化把 SMPL-X 格式的人体动作平滑地"穿"到任意人形机器人身上，无需任何手工关节映射。官方总览图展示的正是这种统一性：纵轴是四种输入身体表示——Humanoid Character、SOMA、SMPL-X 和 Human Mesh，横轴是五种目标机器人——HighTorque PiPlus、Booster K1、Unitree G1、EngineAI T800 和 Unitree H2，同一段动作被重定向到每一个组合上，姿态都保持自然协调。

**简版（短视频配音用）**：

环境部署完成后，我用两个直观的演示来验证整条链路。第一个可视化播放一段人体舞蹈动作，这一步验证了人体模型、动作数据和图形环境都已就绪。第二个是重定向全链路运行：先从人体和目标机器人的表面采样点云、构建对应数据集；再在这份数据上训练点云对应网络，经过数百轮训练让损失稳定收敛；然后以学到的对应关系为约束，逐帧物理优化求解机器人的关节轨迹；求解完成后，查看机器人端的运动。画面里不再是人体在跳舞，而是机器人"学会"了这段舞。输入的人体动作能正确回放，输出的机器人动作流畅自然，就说明从数据到仿真的完整链路已经打通。

**详版（长视频配音用）**：

环境部署完成后，我们用两个直观的演示来验证整条链路。第一个是人体动作回放：可视化播放一段 SMPL-X 人体舞蹈动作，这一步验证了人体模型、动作数据和图形环境都已就绪。第二个是重定向全链路运行：一条命令启动的流水线会依次完成四个阶段——先从人体和目标机器人的表面采样点云、构建对应数据集；再在这份数据上训练点云对应网络，经过数百轮训练让损失稳定收敛；然后以学到的对应关系为约束，逐帧物理优化求解机器人的关节轨迹，求解中同时施加地面接触、防穿透、防自碰撞等硬约束，并用轨迹滤波把动作磨平滑；求解完成后，MuJoCo 查看器自动回放机器人端的运动并录制画面。所以这段约五分钟的录像并不是一次简单的回放，而是整条重定向链路跑完后的最终产物：画面里不再是人体在跳舞，而是机器人"学会"了这段舞。输入的人体动作能正确回放，输出的机器人动作流畅自然，就说明从数据到仿真的完整链路已经打通。

---

