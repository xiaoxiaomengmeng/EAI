# 503-do-as-i-do · 入口

> Do as I Do 本地复现（arXiv 2606.19333，UPenn malik-group）：**固定管线**从日常人类操作视频生成灵巧操作数据——视频 → 手物重建与 6DoF 跟踪 → 重定向到机器人手（MuJoCo Warp MPC）→ UR3e + Sharpa Wave 真机回放。
> 上游：<https://github.com/malik-group/do-as-i-do>

**状态：🟡 基础可用/部分验证**（2026-10-19 实测：上游已克隆含 5 个二级子模块；doasido_retargeting / sam3 / tapnet 三套 env 已建且 torch cu128 全绿；官方 whisking demo 回放通过。待建：sam3d、hawor、deployment + 权重/MANO 落盘）

## 怎么跑

### 1. 克隆上游（子模块，注意二级子模块与 LFS）

```bash
cd ~/ws
export GIT_LFS_SKIP_SMUDGE=1                      # 先跳过 LFS 大文件，按需再拉
git submodule add https://github.com/malik-group/do-as-i-do.git 503-do-as-i-do/repos/do-as-i-do
cd 503-do-as-i-do/repos/do-as-i-do
./reconstruction/setup/00_init_submodules.sh      # 补齐 modules/ 下 5 个 fork 子模块（sam3/sam3d/Fast-SAM3D/HaWoR/tapnet）
```

### 2. 建环境（reconstruction 4 套 + retargeting 1 套 + deployment 1 套）——5090 定制路线

> **动手前先提交两处异步申请**（审批要等，期间正好把环境建完）：① HF gated：`facebook/sam3` 与 `facebook/sam-3d-objects` 各自页面点申请，之后 `hf auth login`；② MANO 注册（mano.is.tue.mpg.de，HaWoR 手部模型用，纯手动下载、不走任何脚本）。

⚠️ **不要一键 `./setup/01_create_envs.sh`（默认 all）**：本地脚本核实，mk_sam3d 走 cu121/torch~2.5、mk_hawor 装 torch 1.13+cu117，在 5090（sm_120）上均不可用（脚本注释自己建议改用 env/*.yml；fork 自带 setup 同样是 cu117/cu121 也不可用——env/README.md L11 明说 50xx 走 fallback）。正确顺序（从轻到重，风险最高的 hawor 放最后）：

```bash
conda deactivate   # 别串环境
cd ~/ws/503-do-as-i-do/repos/do-as-i-do/reconstruction

# ① doasido_retargeting：最轻最先建——建完立刻能看官方 demo（验证阶梯阶段 1 只依赖它）
#    env 名自定防撞名：retargeting 是纯 pip 包入口，env 名自由（已验证无硬编码）；
#    reconstruction 的 4 个 env 名被 config/paths.sh 锁定，保持默认不动
conda create -y -n doasido_retargeting python=3.12 && conda activate doasido_retargeting
cd ../retargeting && pip install -e . && cd ../reconstruction
conda deactivate

# ② sam3 + tapnet：脚本配方本身就是 cu128，支持单 env 参数
./setup/01_create_envs.sh sam3
./setup/01_create_envs.sh tapnet
conda run -n sam3   python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # 期望 2.8.0+cu128, True
conda run -n tapnet python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # 期望 2.7.0+cu128, True

# 浏览器开 localhost:8081 看 whisking demo 三层叠加——验证阶梯阶段 1 达成。
bash ~/ws/503-do-as-i-do/scripts/01_replay_demo.sh

# ③ sam3d：不用脚本（cu121 会废），用 known-working 的 yml（torch 2.8+cu128）
conda env create -f env/sam3d.yml
conda activate sam3d
pip uninstall -y notebook                                          # 解遮蔽（yml 路线也要做）
pip install "geocalib @ git+https://github.com/cvg/GeoCalib.git"   # yml 早于 geocalib 加入，需补（env/README L27）
pip install viser
pip install -e modules/sam-3d-objects                             # yml 里 -e 是原机器绝对路径，重装
conda deactivate

# ④ hawor：同样用 yml（torch 2.9+cu128），风险最高放最后；yml 不含本地编译包，要补编译
conda env create -f env/hawor.yml
conda activate hawor
# 先给 lietorch 打 1 行 patch（torch>=2.x 编译必撞，env/README L38-45 给了 diff）：
#   modules/HaWoR/thirdparty/DROID-SLAM/thirdparty/lietorch/lietorch/include/dispatch.h
#   ::detail::scalar_type(the_type)  →  the_type.scalarType()
cd modules/HaWoR/thirdparty/DROID-SLAM && python setup.py install && cd -
conda env config vars set TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 -n hawor   # checkpoint 含 DictConfig，必须
conda deactivate

# ⑤ 权重下载（gated 已获批 + hf auth login 后跑；须先激活任一项目 env——
#    脚本会把 huggingface-hub CLI 和 gdown pip 装进当前激活 env，裸跑会污染 base）
conda activate sam3d
./setup/02_fetch_weights.sh --download
conda deactivate
#    SAM3（Stage 1）权重是运行时自动下载的，同样 gated：跑管线前确保 hf auth login 过
#    MANO 不走此脚本（只 echo 提示）：官网手动下载后放 modules/HaWoR/_DATA/data/mano/MANO_RIGHT.pkl
#    与 modules/HaWoR/_DATA/data_left/mano_left/MANO_LEFT.pkl
```

⑥ deployment（**按需**，验证到阶段 5 前再建即可；本机无真机，只用到 mujoco_replay 仿真侧）：

```bash
conda env create -f ~/ws/503-do-as-i-do/repos/do-as-i-do/deployment/env/deployment.yml
```

sam3d 若在 GLB/纹理烘焙时报 `NameError: GaussianRasterizationSettings`，按 env/README L29-36 装 mip-splatting 的 diff_gaussian_rasterization（`TORCH_CUDA_ARCH_LIST=12.0` 正是 5090）。

### 3. 零 GPU 成本起步：官方 whisking demo 可视化（开箱即用）

```bash
conda activate doasido_retargeting
bash ~/ws/503-do-as-i-do/scripts/01_replay_demo.sh
# viser 网页（默认 :8081）三层叠加：MANO 参考（橙）→ IK 幽灵（蓝透明）→ 物理优化轨迹（实体），Frame 滑条可逐帧播
```

## 官方样例：whisking demo（随仓库分发，克隆即得）

上游只带这一个 demo（打蛋器搅拌，右手），但是**视频 + 全链产物**的完整包，且均为普通 git 文件（非 LFS 指针，`GIT_LFS_SKIP_SMUDGE=1` 不影响）：

| 位置 | 内容 |
|---|---|
| `reconstruction/whisking/whisking.mp4` | 样例视频（0.9MB） |
| `reconstruction/whisking/video_segmentation/` | SAM3 分割 masks |
| `reconstruction/whisking/obj_tracking_out/` | 物体 6DoF 跟踪 layout（管线 A 产物） |
| `reconstruction/whisking/whisking/all_hand_meshes.npz` | HaWoR 手网格（2.7MB） |
| `retargeting/outputs/sharpa/right/whisking/0/` | `scene.xml` + `trajectory_kinematic.npz` + `trajectory_mjwp.npz` + `config.yaml`（管线 B 产物） |
| `retargeting/outputs/assets/` | whisk 物体 mesh + CoACD 凸分解 + sharpa 手 meshes |

用途：① 阶段 1 回放开箱即播；② 阶段 2/3 复跑 whisking 时有官方产物逐项对照——复现判据从「跑出文件」升级为「结果与官方可比」。

## 输入 / 输出 / 控制方式

| | |
|---|---|
| 输入 | 固定相机的人类手物操作视频（自定义视频放 `data/`，输出落在视频同目录，不进子模块）；reconstruction 另需参考帧号、物体名、锚定手 |
| 输出 | `layout_camera_frame_optimized.json`（物体 6DoF 轨迹）→ `outputs/sharpa/<hand>/<task>/0/`：`scene.xml`、`trajectory_kinematic.npz`（IK）、`trajectory_mjwp.npz`（MPC 优化，含逐步跟踪误差） |
| 控制 | 三段式固定管线（无 LLM、不烧 API）：reconstruction（run_pipeline.sh）→ retargeting（launch.py，`--no-show-viewer` 可 headless）→ deployment（replay_retarget.py 调工作空间位姿 → 50Hz 流式真机） |

## 验证阶梯

| 阶段 | 目的 | 命令 | 判据 | 状态 |
|---|---|---|---|---|
| 0 | 克隆 + 5 envs + 权重就位 | 见上「怎么跑」 | `conda env list` 5 套齐（sam3/sam3d/hawor/tapnet/doasido_retargeting；deployment 按需，阶段 5 前建）；MANO 两 pkl 已落盘 `_DATA/`；`02_fetch_weights.sh` 无 gated 报错 | 🟨 3/5：doasido_retargeting ✅、sam3 ✅（2.8.0+cu128）、tapnet ✅（2.7.0+cu128）；sam3d/hawor 待建，权重/MANO 待落盘 |
| 1 | 官方 demo 可视化（零 GPU） | `scripts/01_replay_demo.sh` | viser 网页三图层可播放 | ✅ 2026-10-19 实测：viser :8081 起服务，scene.xml + mjwp/kinematic/keypoints 四层资产加载成功，播 1000 帧（跳过 600 warmup） |
| 2 | reconstruction 官方 demo | 拷视频到 data/ 后跑：`cp repos/do-as-i-do/reconstruction/whisking/whisking.mp4 data/ && ./run_pipeline.sh ../../../data/whisking.mp4 125 whisk right` | data/ 下生成 `layout_camera_frame_optimized.json`，与官方 `obj_tracking_out/` 对照合理 | ⬜ |
| 3 | retargeting 官方 demo | `python launch.py --task whisking_repro --raw-dir <data/ 下阶段 2 输出目录>` | `trajectory_mjwp.npz` 生成且跟踪误差与官方 `whisking/0/` 可比 | ⬜ |
| 4 | 自定义视频全链 | 视频放 `data/` 跑阶段 2→3 | 全链产物 + `replay_viser.py --run-dir` 可回放 | ⬜ |
| 5 | deployment 仿真侧（无真机） | `replay_retarget.py --side right --traj .../trajectory_mjwp.npz --solve-only` | `trajectory_dual_ur3e.npz` 生成、IK 残差/碰撞间隙可查 | ⬜ |

## 已知坑

- **显存压线**：reconstruction 要求 ≥32GB VRAM，5090 正好 32GB——跑前关掉其他占显存进程；retargeting 的 MuJoCo Warp 优化也吃 GPU。
- **权重脚本会 pip 装进当前激活 env**（huggingface-hub CLI、gdown）：`02_fetch_weights.sh` 必须在某个项目 env 里跑，别裸跑污染 base；MANO 两 pkl 纯手动下载落盘（路径见「怎么跑」⑤），脚本不代下。
- **LFS 按需拉取**：克隆时 `GIT_LFS_SKIP_SMUDGE=1` 跳过的 LFS 大文件（demo 资产不受影响，均为普通 git 文件），需要时进 `repos/do-as-i-do` 执行 `git lfs pull`。
- **SAM3 分割是点击式 GUI，需要 X display**（`config/paths.sh` 设 `SAM3_DISPLAY=:1`）：SSH 远程时要转发 X 或试 text prompting。
- **5090 建环境必须分路线**（详见「怎么跑」第 2 步）：`01_create_envs.sh` 的 sam3/tapnet 配方是 cu128 可直接用，但 sam3d（cu121）/ hawor（cu117）会装废——这两个必须用 `env/sam3d.yml`、`env/hawor.yml`（known-working cu128 导出），fork 自带 setup 在 50xx 上同样不可用。
- **hawor 的 lietorch 编译必撞 dispatch.h 错**：torch>=2.x 下要把 `::detail::scalar_type(the_type)` 改为 `the_type.scalarType()`（上游 env/README 明示的官方 patch）；hawor env 还需 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`（checkpoint 含 omegaconf DictConfig）。
- **复跑管线别对着 whisking.mp4 原地跑**：`run_pipeline.sh` 输出落视频同目录，会**覆盖子模块内官方 demo 产物**——先把视频拷到 `data/` 再跑；复跑 retargeting 用 `--task whisking_repro` 换名，官方 `outputs/sharpa/right/whisking/0/` 保持只读对照。
- **deployment 缺件**：Sharpa Wave SDK 私有不随仓库发布（要放 `robot_replay/Sharpa/`），且需双 UR3e 真机——本机只验证到 mujoco_replay 仿真侧。
- 唯一要改的路径配置：`reconstruction/config/paths.sh`。

## 目录

```
503-do-as-i-do/
├── repos/do-as-i-do/      # 上游子模块（其 modules/ 下还有 5 个 fork 子模块；不改源码）
│   ├── reconstruction/    # 管线 A：分割→mesh→MoGe→HaWoR→TAPIR→引导扩散跟踪（4 env）
│   ├── retargeting/       # 管线 B：清洗对齐→CoACD 凸分解→MJCF 生成→mink IK→MuJoCo Warp MPC（1 env）
│   └── deployment/        # 管线 C：dual UR3e IK 回放 + 50Hz 真机流式（1 env，本机仅 sim）
├── docs/                  # 专题笔记（01-架构与关键技术解析）
├── scripts/               # 验证脚本
├── data/                  # 自定义视频与其管线产物（gitignore）
├── START_HERE.md          # 本文件
└── README.md
```

## 关联

- **502-DexAgent**：agentic 对照路线。两处实质交集：① **retargeting 底座同为 SPIDER**（facebookresearch/spider）——DexAgent Stage 3 兜底方法 `physics_informed_retargeting` 也依赖 SPIDER，环境经验可互通；② **同一双 Sharpa Wave 灵巧手**（22 指关节），MJCF 资产与关节命名约定两边对齐后可互喂。
- **500-umr / 501**：同属 5xx 人类数据→机器人重定向家族；501 的手部姿态估计链路与本管线 HaWoR 环节重叠。
- 结合实验（do-as-i-do 确定性重建 → DexAgent Stage 3 agentic 轨迹搜索）待两边都过官方 demo 后另立文档。
