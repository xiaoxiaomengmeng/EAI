# ws 工作区目录重组方案与执行记录

> 本文档记录 ws 工作区目录重组的需求、方案、执行进度与变更日志，**边改边记**，防止遗忘。
> 需求有变化时直接修改对应章节，并在文末「变更日志」追加一条。

## 1. 需求背景

工作区内容分四大块 + 公共底座：

1. **具身智能模型系列**：VLA、世界模型、具身基础模型
2. **非具身模型**：LLM 等基模测试、MiniMax + ComfyUI 等生成类
3. **真机部署测试**：机械臂/人形机器人真机，以及好玩的本地复现（跳跳、Microduck 等）
4. **具身数据 infra**：UMR、SAM、人体动捕/遥操作数据采集等
5. **公共底座**：模型权重软链、数据集索引、仿真资产、部署脚本、文档

模型和数据集大部分下载在 `~/.cache/modelscope/models`（当前 447G / 18 个模型），另有 `~/.cache/huggingface`（3.8G）。

## 2. 编号规范（已确认）

| 段 | 用途 |
|---|---|
| **1xx** | 具身智能模型系列 |
| **2xx** | 非具身模型（LLM 基模、生成类） |
| **3xx** | 真机部署测试、人形机器人真机、趣味本地复现 |
| **4xx** | 预留，暂时留空 |
| **5xx** | 具身数据 infra |
| 无编号 | 公共底座：`models/`、`datasets/`、`assets/`、`scripts/`、`docs/` |

- 编号与名称之间统一用 `-` 分隔（如 `100-openVLA`）；顶层目录名内部的 `_` 也统一改 `-`（如 `101-pi-serial-models`），子目录名不动
- 选三位数是因为每段有 100 个号位，两位数每段只有 10 个
- `23_video` 中 ComfyUI 与 MiniMax 配合使用才有意义，作为整体不拆分
- `human_dexterous_teleop` 属于数据 infra，归入 5xx

## 3. 目录映射表

| 现目录 | 新目录 | 段 | 备注 |
|---|---|---|---|
| `10_openVLA` | `100-openVLA` | 1xx | |
| `11_pi_serial_models` | `101-pi-serial-models` | 1xx | 名称内 `_` 一并改 `-` |
| `12_cosmos` | `102-cosmos` | 1xx | |
| `13_lingbot` | `103-lingbot` | 1xx | depth/video/world/vla 四子项目不拆 |
| `20_unitreerobotics` | `104-unitreerobotics` | 1xx | UnifoLM-WMA 世界模型 |
| `21_demo` | `105-damo` | 1xx | RynnBrain + RynnWorld-4D；原名 demo 为笔误，应为 damo（阿里达摩院） |
| `23_video` | `200-video` | 2xx | ComfyUI + MiniMax，不拆 |
| `colibri` | `201-colibri` | 2xx | 纯 C GLM MoE 运行时 |
| `101-robots` | `300-robots` | 3xx | RynnMotion + sim_robot |
| `100-microduck` | `301-microduck` | 3xx | |
| `102-kingkongrobotics` | `302-kingkongrobotics` | 3xx | 保留原名 |
| `24_umr` | `500-umr` | 5xx | |
| `200_human_dexterous_teleop` | `501-human-dexterous-teleop` | 5xx | |
| `modules` | `models` | 底座 | 模型权重软链统一目录 |
| （新建） | `datasets` | 底座 | 数据集软链 + 索引 |
| `assets` / `scripts` / `docs` | 不变 | 底座 | |

**改名顺序约束**：`100-` / `101-` / `102-` / `200_` 占用了 1xx/2xx 的目标号位，必须先迁到 3xx/5xx，再改 10~24 系列，否则撞名。

> 第 2 步已按此表完成改名（2026-10-11）。第 4、5 节是改名前的 review 快照，保留旧目录名不改；第 5.1 节环境清单已更新为新路径。

## 4. 现状问题清单（2026-10 review）

### 4.1 git 配置不一致
- `.gitmodules` 仍登记 `22_lingbot/*`（实际为 `13_lingbot/*`）与根目录 `mujoco_menagerie`（实际为 `assets/mujoco/mujoco_menagerie`）
- 3 个 gitlink 未登记，导致 `git submodule status` 报 `fatal: no submodule mapping found`：

  | 路径 | 本地 remote | 登记时转 https |
  |---|---|---|
  | `12_cosmos/3rd/RoboLab` | `git@github.com:NVlabs/RoboLab.git` | `https://github.com/NVlabs/RoboLab.git` |
  | `12_cosmos/3rd/cosmos-framework` | `git@github.com:NVIDIA/cosmos-framework.git` | `https://github.com/NVIDIA/cosmos-framework.git` |
  | `13_lingbot/lingbot-vla-v2` | `git@github.com:Robbyant/lingbot-vla-v2.git` | `https://github.com/Robbyant/lingbot-vla-v2.git` |

- `.gitignore` 注释仍引用 `22_lingbot/.gitignore`

### 4.2 根 README 过时
- 仍列出已不存在的 `09_human_dexterous_teleop`、`22_lingbot`、`Wan-Dancer`、根目录 `mujoco_menagerie`
- 未收录 `100/101/102/12/13/23/24/200`、`assets`、`modules`、`docs`

### 4.3 模型权重未统一管理
- `modules/` 只软链 9 个，缺：RynnBrain ×4（8B / Plan / Nav / CoP）、lingbot-vla-v2-6b、lingbot-depth-pretrain-vitl-14-v0.5、UnifoLM-WMA-0-Dual、MiniMax-H3、Wan-Dancer-14B
- 项目内直接存放约 267G 权重，未走缓存：
  - `13_lingbot/lingbot-video/models` 185G（含 Qwen3.6-27B）
  - `13_lingbot/lingbot-world-v2/lingbot-world-v2-14b-causal-fast` 81G
  - `13_lingbot/lingbot-depth/models` 1.2G（疑与 modelscope 缓存重复）
- `21_demo/RynnBrain/models` 直接软链整个 `~/.cache/modelscope/models`，与 `modules/` 约定不一致

### 4.4 可清理空间（待决定）
| 项 | 大小 | 说明 |
|---|---|---|
| `MiniMax--MiniMax-H3` | 97G | |
| `Wan-AI--Wan-Dancer-14B` | 80G | 需 8 卡，单卡 5090 无法运行 |
| `11_pi_serial_models/outputs/pi0_aloha_cube_full` | 48G | 可只留最终 checkpoint |
| `101-robots/RynnMotion.zip` | — | 与已解压目录重复 |

### 4.5 文档散落
- `20_unitreerobotics/` 根目录：`error.log`、`fix_readme.md`、`STUDY_NOTES.md`
- `13_lingbot/` 根目录：`LingBot-VLA-v2_架构与运行逻辑详解.md`
- `102-kingkongrobotics/deploy.md` 不在 `docs/` 下
- `11_pi_serial_models/docs/` 下 `墨奇.txt`、`数据.txt` 不符合 `NN-主题.md` 命名

### 4.6 数据 infra 缺口
- 无专门的数据集目录与索引；数据散在 `11_pi_serial_models/data`（aloha ×4、libero、pusht，2.1G）与 `~/.cache/huggingface/datasets`（parquet）

### 4.7 改名前已存在的环境问题（第 0 步基线发现）

以下 editable 包在重组前就已 import 失败，原因是更早一次改名留下的旧路径，**不是本次重组引起的**。是否顺手修复待用户决定（见「待确认事项」）。完整输出见 [ws-restructure-baseline.txt](ws-restructure-baseline.txt)。

| 环境 | editable 包 | 记录的源码路径 | 状态 |
|---|---|---|---|
| `10_openVLA/.venv` | openvla | `/home/creator/ws/openVLA/openvla`（旧路径） | `prismatic` import 失败 |
| `13_lingbot/lingbot-video/.venv` | lingbot-video | `/home/creator/ws/lingbot-video`（旧路径） | import 失败 |
| `200_human_dexterous_teleop/frankmocap_demo/.venv` | lingbot-video | 同上 | import 失败，疑似误装 |
| conda `4D-humans` | lingbot-video | 同上 | import 失败，疑似误装 |
| conda `lingbot-depth` | lingbot-video | 同上 | import 失败，疑似误装 |
| `20_unitreerobotics/.venv` | dlimp | `repos/unifolm-world-model-action/external/dlimp` | import 失败（第 4 步查明：路径没问题，是环境缺 `tensorflow_datasets`；推理不用 dlimp，未补装） |

另：conda `lingbot-depth` 中 `mdm` 记录的路径 `/home/creator/ws/lingbot-depth` 已不存在，但 import 仍成功（可能由其它 `.pth` 提供），第 4 步需复核。

## 5. 改名影响面

| 影响 | 说明 | 处理方式 |
|---|---|---|
| 13 个虚拟环境 | 写死绝对路径；editable 安装的 finder/`.pth` 静默失效（`pip show` 正常但 import 报错） | 批量替换 venv 内旧路径 + `pip install -e . --no-deps` 重装，逐个验证 import |
| 子模块 | 路径登记在 `.gitmodules` 与 `.git/modules` | 先修 `.gitmodules`，再用 `git mv` 迁移 |
| tracked 文件中的路径 | 约 577 个文件含 `/home/creator/ws/...` 或 `modules/`，多为文档；运行相关的是 `10_openVLA/scripts/*.py`、`10_openVLA/sim/config.py` | 批量替换 + 人工复核 |
| gitignored 本地配置 | 如 `10_openVLA/.libero/config.yaml`、`libero_source_path.pth`，全局 sed 覆盖不到 | 单独 grep 后手工修 |
| 软链 | 项目内软链为相对路径；`modules/` 软链指向缓存绝对路径 | 不受影响，改名后复核 |

### 5.1 环境清单

项目内 venv 共 13 个（改名后需修复）：

| venv 路径 | Python |
|---|---|
| `100-openVLA/.venv` | 3.10.12 |
| `101-pi-serial-models/.venv312` | 3.12.13 |
| `102-cosmos/3rd/cosmos-framework/.venv` | 3.13 |
| `103-lingbot/lingbot-video/.venv` | 3.10.12（已按用户要求删除） |
| `103-lingbot/lingbot-world-v2/.venv` | 3.10.12（已按用户要求删除） |
| `104-unitreerobotics/.venv` | 3.10.12 |
| `200-video/ComfyUI/.venv` | 3.10.12 |
| `301-microduck/.venv` | 3.12.13 |
| `301-microduck/microduck_rl/.venv` | 3.12.13 |
| `302-kingkongrobotics/jumper/.venv` | 3.12.13 |
| `501-human-dexterous-teleop/hand_gesture_recognition/.venv` | 3.10.12 |
| `501-human-dexterous-teleop/frankmocap_demo/.venv` | 3.10.12 |
| `501-human-dexterous-teleop/mediapipe_holistic_demo/.venv` | 3.10.12 |

conda 环境（位于 `~/miniconda3/envs`，不受目录改名影响，但 editable 安装的包仍需检查）：

| conda env | 对应项目 |
|---|---|
| `4D-humans` | `501-human-dexterous-teleop/4d-humans` |
| `cosmos` | `102-cosmos`（与 cosmos-framework/.venv 二选一，待确认） |
| `lingbot-depth` | `103-lingbot/lingbot-depth` |
| `umr` | `500-umr` |
| `env_isaaclab` | 暂无对应项目目录 |
| `py312` | 通用 |

文档中提到但本机不存在的环境：`conda activate unifolm-wma`（20_unitreerobotics，实际用 `.venv`）、`conda activate rynnworld4d`（21_demo）、`conda activate ros_noetic` / `venv/`（101-robots/RynnMotion）、`conda activate openvla`（10_openVLA，实际用 `.venv`）、`conda activate venv_frankmocap`（200_human，实际用 `frankmocap_demo/.venv`）。

## 6. 项目入口文档 `START_HERE.md`

每个顶层项目目录新增一个显眼的入口文档，内容极简：这个目录干嘛的 + 启动哪个环境 + 从哪开始看。

- **文件名**：`START_HERE.md`（已确认）
  - 名字即用途，不会和上游自带的 `README.md` 冲突
  - 全大写，在文件列表里醒目
- **作为规范**：今后新建的顶层项目目录都必须带一个 `START_HERE.md`
- **存放位置**：项目根目录（用户明确要求显眼，作为「新文档放 docs/」约定的例外）
- **写入时机**：第 2 步改名完成后写，目录名直接用新名
  - ⚠️ 提醒：第 2 步完成后漏写，用户提醒后补上。现改为**第 4 步每修好一个项目的环境，就立即写该项目的 `START_HERE.md`**（环境命令实测通过后再写）；没有 venv 的项目（`105-damo`、`201-colibri`、`300-robots`、`500-umr`）在第 4 步结束时补齐
  - 进度：`100-openVLA` ✅、`101-pi-serial-models` ✅、`102-cosmos` ✅、`103-lingbot` ✅、`104-unitreerobotics` ✅、`200-video` ✅；其余 7 个待写
- **子模块目录**：`501-human-dexterous-teleop`、`201-colibri` 本身是子模块，文件会落在子模块仓库里（显示为子模块 untracked），处理方式待确认

模板（控制在 15 行左右）：

````markdown
# 100-openVLA

一句话：OpenVLA 7B 视觉-语言-动作模型，MuJoCo（WX250S/Franka）仿真推理与 LIBERO 评测。

## 启动环境

```bash
cd ~/ws/100-openVLA
source .venv/bin/activate   # Python 3.10
```

## 从这里开始

- 详细说明：[README.md](README.md)、[docs/](docs/)
- 常用入口：`scripts/01_openvla_verify.py`
````

各项目环境对照（「待定」项由用户逐一运行后确定，写入口文档时如实标注「待定」）：

| 新目录 | 环境 | 备注 |
|---|---|---|
| `100-openVLA` | `.venv`（3.10） | |
| `101-pi-serial-models` | `.venv312`（3.12） | |
| `102-cosmos` | conda `cosmos`（3.12，本目录脚本）+ `3rd/cosmos-framework/.venv`（3.13，官方脚本） | 两套按用途分 |
| `103-lingbot` | 按子项目：depth 用 conda `lingbot-depth`；video/world-v2 的 `.venv` 已删除，与 vla-v2 均待定 | 多环境 |
| `104-unitreerobotics` | `.venv`（3.10） | |
| `105-damo` | 待定 | 未找到可用环境 |
| `200-video` | `ComfyUI/.venv`（3.10） | |
| `201-colibri` | 无（纯 C，`make` 编译） | |
| `300-robots` | 待定 | RynnMotion 为 C++；sim_robot 环境待定 |
| `301-microduck` | 待定（候选：根 `.venv` 与 `microduck_rl/.venv`，均 3.12） | 两个环境分工待定 |
| `302-kingkongrobotics` | `jumper/.venv`（3.12） | |
| `500-umr` | conda `umr` | |
| `501-human-dexterous-teleop` | 按子项目：三个 `.venv`（3.10）+ conda `4D-humans` | 多环境 |

## 7. 根目录整理、命名规范与 Skill 管理（已确认）

### 7.1 根目录只留一个入口

根目录仅保留 `README.md`（重写为纯索引）、`LICENSE`、`.gitignore`、`.gitmodules`、`.qoder/` 与各目录。

| 原文件 | 处理 |
|---|---|
| `user_profile.md` | → `.qoder/rules/user-profile.md`，规则类型「始终生效」（frontmatter `trigger: always_on`），每次对话自动加载 |
| `embodied_ai_model_test_catalog.md` | → `docs/embodied-ai-model-test-catalog.md` |
| `embodied_ai_model_test_catalog.html`、`readme.html` | 删除（内容已过时/与 md 重复，git 历史可找回） |

### 7.2 命名规范（第 5 步写入 `docs/conventions.md`）

| 对象 | 规则 | 例 |
|---|---|---|
| 顶层目录 | `NNN-kebab-case`；专有名词保留原大小写 | `101-pi-serial-models`、`100-openVLA` |
| 固定文件 | 全大写 | `README.md`、`START_HERE.md`、`LICENSE` |
| 项目文档 | 放 `docs/`，新文档用 `NN-主题.md`；已有文档不改名（避免大量断链） | `docs/01-环境搭建.md` |
| 脚本 | `NN_动词_对象.py/.sh` | `scripts/02_libero_eval.py` |
| Skill / Rule | 全小写 kebab-case，动词开头 | `new-ws-project` |

- 已执行：父仓库内 5 个小写 `readme.md` 统一改为 `README.md`（`102-cosmos`、`104-unitreerobotics`、`300-robots`、`assets`、`models`）
- 每个项目文档分三层：`START_HERE.md`（入口）→ `README.md`（详细）→ `docs/`（专题）

### 7.3 Skill 管理

- 存放：项目级 `ws/.qoder/skills/<name>/SKILL.md`，纳入 git；需要跨项目用的从 `~/.qoder/skills/<name>` 软链过来，源文件只维护一份（同名时项目级优先）
- 索引：`.qoder/skills/README.md` 登记名字 / 用途 / 触发说法
- 脚本：通用的复用 `ws/scripts/`，专用的放 skill 自己的 `scripts/`
- 新建：先用 `/create-skill` 生成模板，再登记索引
- git：`.gitignore` 已加 `.qoder/*`、`!.qoder/rules/`、`!.qoder/skills/`，已用 `git check-ignore` 验证 rules/skills 被跟踪、其余忽略
- 候选首批：`new-ws-project`（分配编号 + START_HERE + 登记根 README）、`link-modelscope-model`（下载 → `models/` 软链 → 索引）、`fix-venv-after-rename`（`check_venv_editable.py` 检查与修复）

## 8. 执行步骤与进度

每步完成后验证，验证通过再单独 commit。

- [x] **第 0 步：准备**（2026-10-11 完成）
  - [x] 未提交改动原样随迁移、不加入 git（已确认）：`102-kingkongrobotics/docs/`（untracked）、5 个子模块 modified、2 个子模块含 untracked；快照已记入 baseline，迁移后对照复核
  - [x] 记录各项目改名前 import 基线：新增 [scripts/check_venv_editable.py](../scripts/check_venv_editable.py)，对 13 个 venv + 4 个 conda env 运行，结果存 [ws-restructure-baseline.txt](ws-restructure-baseline.txt)；发现的既有问题见 4.7
- [x] **第 1 步：修 `.gitmodules`（改名前必须完成）**（2026-10-11 完成，待提交）
  - [x] 修 `22_lingbot/*` → `13_lingbot/*`（depth / world-v2 / video，条目名与 path 一起改）
  - [x] 修 `mujoco_menagerie` → `assets/mujoco/mujoco_menagerie`
  - [x] 补登记 RoboLab、cosmos-framework、lingbot-vla-v2（https URL）
  - [x] `git submodule init` 登记上述 8 个子模块到 `.git/config`；删除 `.git/config` 中 5 条失效条目（`09_human_dexterous_teleop`、`22_lingbot/*` ×3、`mujoco_menagerie`）
  - [x] 验证：`git submodule status` 无 fatal、无未初始化项；21 个 gitlink 与 `.gitmodules` 一一对应；子模块自身 remote（ssh）未被改动（未执行 `git submodule sync`）
  - 备注：除 `100-microduck/microduck_rl` 外，其余子模块的 `.git` 都是内嵌目录（未 absorb 到 `.git/modules`），第 2 步 `git mv` 时会随目录整体移动
- [x] **第 2 步：`git mv` 改名**（2026-10-11 完成，已暂存未提交）
  - [x] 前置：确认无进程以 ws 内目录为工作目录；`git mv` 要求 `.gitmodules` 无未暂存改动，已 `git add .gitmodules`（仅暂存）
  - [x] 2.1 先迁 3xx/5xx：`100-microduck`、`101-robots`、`102-kingkongrobotics`、`200_human_dexterous_teleop`
  - [x] 2.2 再迁 1xx/2xx：`10_`~`13_`、`20_`、`21_demo` → `105-damo`、`23_`、`colibri`
  - [x] 2.3 最后：`24_umr` → `500-umr`，`modules` → `models`
  - [x] 验证：`git submodule status` 21 项全部正常；gitlink 与 `.gitmodules` path 一一对应（`git mv` 自动更新 path）；21 个 gitlink SHA 与 HEAD 一致；absorb 的 `microduck_rl` 的 `core.worktree` 已自动指向新路径；未提交改动（5 个子模块 modified、2 个含 untracked、`302-kingkongrobotics/docs/`）原样随迁
  - [x] 2.4 根目录整理与 README 大写（见第 7 节）：`user_profile.md` → `.qoder/rules/user-profile.md`；catalog md → `docs/`；删除 2 个 html；5 个 `readme.md` → `README.md`；`.gitignore` 加 `.qoder` 规则
  - [x] 2.5 统一子模块 name（`git mv` 只改 path 不改 name，改名后 section 名仍是旧路径）：
    - 改前备份到 `.git/restructure-backup/`（`gitmodules`、`config`、`submodule status` 各一份）
    - 19 个 section 用 `git config [-f .gitmodules] --rename-section` 同时改 `.gitmodules` 与 `.git/config`，name 统一等于 path；`assets/mujoco/mujoco_menagerie`、`docs/every-embodied` 本来一致，跳过
    - `microduck_rl`（吸收式）：`.git/modules/100-microduck/microduck_rl` → `.git/modules/301-microduck/microduck_rl`，子模块 `.git` 文件的 gitdir 同步改；`core.worktree` 是相对路径且层级不变，无需改
    - 验证：name≠path 0 条；`.git/config` 无旧名残留，且与 `.gitmodules` 名称一一对应；`git submodule status` 21 项与改前逐字一致；url 未变；未执行 `git submodule sync`（避免覆写子模块本地 ssh remote）；`.gitmodules` 已重新暂存
- [x] **第 3 步：修路径引用**
  - [x] tracked 文件批量替换旧路径：60 个文件 sed 替换，无残留；排除本重组文档、`.gitmodules`、历史运行日志 `100-openVLA/docs/reports/libero_eval/console_smoke_run.txt`；`103-lingbot/.gitignore` 的历史注释（合并自原 13_lingbot 与 22_lingbot）手工恢复；根 `.gitignore` 注释改为列出 6 个子 `.gitignore`
  - [x] `modules` → `models`：仅 `100-openVLA` 涉及（`ws/modules`、`"modules"`、`modules/`），变量 `MODULES_DIR` → `MODELS_DIR`；`sys.modules` 不动
  - [x] 语法检查：`py_compile` 15 个、`bash -n` 9 个通过（yaml 仅改路径字符串，未做解析校验）
  - [x] gitignored 本地配置手工修复：`100-openVLA/.libero/config.yaml` 5 行；`302-kingkongrobotics/jumper/tasks/jumper/{flat,tripod}/out/*/README.md` 的 Source checkpoint
  - [x] 子模块内引用：`104-unitreerobotics/repos/unifolm-world-model-action/configs/inference/world_model_interaction.yaml:225` 的 `data_dir` 原写死 `/home/creator/ws/20_unitreerobotics/...`（fork 提交 9e82896 引入；`data.setup()` 会读该目录下 csv，`--prompt_dir` 不覆盖），按用户决定改为相对路径 `examples/world_model_interaction_prompts`（`run_wma_interaction.sh` 先 cd 到仓库根）；子模块内新增 1 处未提交改动
  - [x] 软链复核：`301-microduck/microduck_rl/wandb/offline-run-*/files/` 下 `model_0.pt`、`ws.diff`、`*.onnx` 已重建指向 `/home/creator/ws/301-microduck/...`，均可解析；`501-human-dexterous-teleop/4d-humans/4D-Humans/detectron2/detectron2/model_zoo/configs` 指向更早的 `/home/creator/ws/human_dexterous_teleop/...`，改名前就已断，未处理
  - 保留不改：`console_smoke_run.txt`（历史日志）；101 系列文档中原有的 `file:///Users/damon/...` macOS 路径
- [ ] **第 4 步：修虚拟环境**（用户要求先修一个试点，没问题再推其余）
  - 只读扫描（13 个 venv）：
    - A 类（本次改名弄坏，8 个）：`100-openVLA`、`101-pi-serial-models`、`102-cosmos/3rd/cosmos-framework`、`104-unitreerobotics`、`200-video/ComfyUI`、`301-microduck`、`301-microduck/microduck_rl`、`302-kingkongrobotics/jumper`
    - B 类（改名前就已失效，5 个）：`103-lingbot/lingbot-video`、`103-lingbot/lingbot-world-v2`、`501-human-dexterous-teleop/{hand_gesture_recognition,frankmocap_demo,mediapipe_holistic_demo}`，`activate` / shebang 指向更早的 `/ws/lingbot-*`、`/ws/human_dexterous_teleop/...`
  - 每个 venv 的做法：备份到 `.git/restructure-backup/venv-<项目>/` → sed 替换 `bin/` 下旧路径 → 手工改 `.pth` → `python -m pip install -e <源码> --no-deps --no-build-isolation` → 对比 `pip freeze` + import + `check_venv_editable.py`
  - [x] 4.1 试点 `100-openVLA/.venv`：
    - bin 67 个文件（57 个 shebang 指向更早的 `/ws/openVLA/.venv`，10 个指向 `/ws/10_openVLA/`）已改，残留 0；`source activate` 后 `which python/pip` 正确
    - `libero_source_path.pth` → `/home/creator/ws/100-openVLA/repos/LIBERO`
    - 重装 libero、openvla（源码 `repos/openvla`），finder 旧路径残留 0；`pip freeze` 304 个包版本不变
    - 验证：`check_venv_editable.py` 显示 libero ok、openvla 全部 ok（基线里 `prismatic` 是 FAIL，4.7 中这一项一并修好）；设 `LIBERO_CONFIG_PATH=.libero` 后 assets / bddl / init_files 均存在，libero_spatial 10 个任务可加载；`02_libero_eval.py --help` 正常；torch 2.11.0+cu128 CUDA 可用；`repos/openvla`、`repos/LIBERO` 无新增改动
    - 已写 `100-openVLA/START_HERE.md`，文中 `source activate` + `LIBERO_CONFIG_PATH` + 入口脚本命令均实测可用
    - 发现：LIBERO 默认读 `~/.libero/config.yaml`（全局，工作区外），其中 4 行指向 `/home/creator/ws/11_pi_serial_models/.venv312/...`，被本次改名弄断，归到 101 项目处理；100-openVLA 脚本自己设 `LIBERO_CONFIG_PATH`，不受影响
  - [ ] 4.2 其余 A 类 7 个（已完成 4/7）
    - [x] `101-pi-serial-models/.venv312`：
      - bin 80 个文件旧路径已改，残留 0；`pyvenv.cfg` 的 command 行同步改（仅说明性字段）；重装 lerobot editable（`repos/lerobot-main`），`pip freeze` 297 个包版本不变
      - `~/.libero/config.yaml`（工作区外）：4 行 `11_pi_serial_models` → `101-pi-serial-models`，改前副本在备份目录 `home-libero-config.yaml.before`
      - **顺带修改名前的遗留**（提交 `0482e02` 把 `11_pi_serial_models/models/*` 搬到根 `modules/` 后未同步，用户确认修法）：
        - 脚本 `22/23/24_train_*.sh` 的 `MODEL_DIR` → `$HOME/ws/models/...`；`11_download_act_assets.sh` 的 `MODEL_DIR` → `$HOME/ws/models`；`31_eval_act.sh` 注释同步
        - `outputs/` 下 29 个 checkpoint JSON + 2 个 PEFT README：`.../11_pi_serial_models/models/` → `/home/creator/ws/models/`，其余 `11_pi_serial_models/` → `101-pi-serial-models/`（备份 `outputs-json-before.tgz`，31 项）；wandb `debug.log` 历史日志不改
        - docs/17、19、21 中可执行命令里的 `~/ws/101-pi-serial-models/models/` → `~/ws/models/`；docs/22 的日志摘录、目录树示意不改
      - README 修正：删去不存在的 `models/` 目录树项、cu128 → cu130、`22_lingbot` → `103-lingbot`、补实测成功率与 `~/.libero` 依赖、`32_eval_pi0.sh` 默认 POLICY_DIR 不存在的坑
      - 验证：`check_venv_editable.py` lerobot ok；libero 3 个路径存在、libero_spatial 10 任务可加载；PI0/PI05 Policy import ok；torch 2.11.0+cu130 CUDA 可用；**实跑** pi0.5 LoRA `lerobot-eval`（libero_spatial 10×1）从 `~/ws/models/pi05_libero_base` 加载基座，成功率 90%；`31_eval_act.sh` 1 轮跑通；产物在 `outputs/_verify_after_rename*`；`repos/lerobot-main` 无新增改动
      - 已写 `101-pi-serial-models/START_HERE.md`
      - 注意：GPU / `/tmp` 在 sandbox 内不可用，实跑评测需在 sandbox 外执行
    - [x] `102-cosmos`：有两套环境
      - conda `cosmos`（`~/miniconda3/envs/cosmos`，工作区外）：没有任何工作区路径引用，不受改名影响，无需修
      - uv venv `3rd/cosmos-framework/.venv`（Python 3.13，torch 2.10.0+cu128，环境内没有 pip）：bin 151 个文件（含 activate）已 sed，`pyvenv.cfg` 同步；`uv pip install -e --no-deps --no-build-isolation` 因缺 `editables` 失败（未改动环境），为了不往环境里新增包，改为直接 sed `_editable_impl_cosmos_framework.pth` 和 `direct_url.json`（hatchling 的 editable 只是一行路径的 .pth，结果等价）；整个 `.venv` 旧路径残留 0；`uv pip freeze` 425 行，只有 editable 那 1 行不同；备份在 `venv-102-cosmos-framework/`
      - 顺带修 `0482e02` 遗留：`scripts/verify_models.py` 的 `MODELS_DIR` 改为 `~/ws/models`（原 `102-cosmos/models` 已不存在）
      - README 修正：模型改为 3 个，软链路径改为 `~/ws/models/`；补「两套 Python 环境」表；注明 `NVIDIA/cosmos` 没有克隆、verify 脚本不覆盖 Cosmos3-Edge
      - 验证：uv venv 中 `import cosmos_framework` 指向新路径，CUDA 可用，`action_policy_server_robolab --help` 正常；conda 中 `verify_models.py` 完整跑通（Predict2.5 与 DROID 文件级全部 PASS，DROID 以 `Cosmos3OmniPipeline` 加载成功）；子模块 `3rd/*` 无新增改动
      - 已写 `102-cosmos/START_HERE.md`
  - [ ] 4.3 B 类 5 个及 4.7 其余项（用户改为按项目编号顺序推进，已完成 2/5）
    - [x] `103-lingbot`（用户确认「三个全修」）：
      - `lingbot-video/.venv`：bin 33 个文件（指向更早的 `/ws/lingbot-video/.venv`）已 sed；环境里 setuptools 只有 59.6，低于构建要求的 ≥69，所以用默认隔离构建 `pip install -e . --no-deps`（清华源）重装；finder MAPPING、direct_url 已是新路径，`.venv` 残留 0；freeze 只有 editable 那行的写法变了（改成 git+ssh 形式）；`check_venv_editable` 显示 lingbot_video ok，CUDA 可用
      - `lingbot-world-v2/.venv`：bin 36 个文件已 sed；没有 editable；freeze 不变；`which python/pip` 和 shebang 正确，torch 2.11.0+cu128 CUDA 可用
      - conda `lingbot-depth`（工作区外）：mdm 的 editable 指向 `/ws/lingbot-depth`（基线里写的 ok 是误判：mdm 是命名空间包，路径不存在也能 import 成功）；`pip install -e . --no-deps --no-build-isolation` 重装后 MAPPING 已是新路径；**实跑** `example.py --example 0` 通过（输出到 `/tmp/lingbot_depth_verify`）；子模块 git status 不变
      - 未处理（已记录）：lingbot-depth 环境里装错的 lingbot-video editable（仍指向 `/ws/lingbot-video`）；子模块 lingbot-video 已提交的 `scripts/download_models/*.sh` 有 4 处 `MODEL_DIR` 写死 `/home/creator/ws/lingbot-video/models/...`（运行脚本 `run_demo_5090.sh` 用的是相对路径，不受影响）；`README_PRACTICE.md`、lingbot-depth `fix_readme.md`（历史日志）、`LingBot-VLA-v2_架构与运行逻辑详解.md` 里的 macOS 路径都不改
      - lingbot-vla-v2 本机还没有环境，START_HERE 里写「待定」
      - 备份：`venv-103-lingbot-video/`、`venv-103-lingbot-world-v2/`、`conda-lingbot-depth/`；已写 `103-lingbot/START_HERE.md`
      - **后续（用户要求）**：`lingbot-video/.venv`（7.1G）、`lingbot-world-v2/.venv`（7.9G）已删除（删前确认没有进程在用；子模块 git status 不受影响），共释放约 15G；START_HERE 中这两项改为「待定」；conda `lingbot-depth` 不是 venv，保留
    - [x] `104-unitreerobotics/.venv`（A 类，按编号在此处理）：
      - bin 61 个文件已 sed，残留 0；`pyvenv.cfg` 里没有路径
      - 两个 editable（unifolm_wma 的 `.pth`、dlimp 的 finder MAPPING）和两个 `direct_url.json` 直接 sed。环境里 setuptools 是 59.6，低于 unifolm 构建要求的 ≥65.6.3，重装需要隔离构建，还会重写仓库里的 egg-info；setuptools 的 editable 产物只记路径，sed 结果等价
      - freeze 309 行，版本全部不变；只有两行 editable 的写法变了（新路径下能找到 git 仓库，显示为 git+https 形式）
      - dlimp 在基线里是 FAIL，原因查明：路径没问题，是环境缺 `tensorflow_datasets`（改名前就这样）。现在 finder 已能解析到新路径；推理代码 src/、scripts/ 都不用 dlimp，按最小改动原则没补装
      - 验证：unifolm_wma ok，torch 2.11.0+cu128 CUDA 可用；**实跑** `world_model_interaction.py`（unitree_z1_dual_arm_stackbox，n_iter 1、ddim 5 步，输出到 `/tmp/wma_verify`）exit 0，生成 wm/dm 视频；子模块 git status 与改前一致（仍只有第 3 步改的那个 yaml）
      - README、STUDY_NOTES 的路径第 3 步已改过，无残留；权重在 `~/.cache/modelscope`（16.7G），不属于 `0482e02` 遗留
      - 备份 `venv-104-unitreerobotics/`；已写 `104-unitreerobotics/START_HERE.md`
    - [x] `200-video/ComfyUI/.venv`：
      - bin 32 个文件 `23_video` → `200-video`，残留 0；没有 editable；freeze 264 行，前后一致
      - 模型软链（6 条，指向 `~/.cache/modelscope/models/MiniMax--MiniMax-H3/`）全部能解析；`miniMax/README.md` 的路径第 3 步已改过
      - 验证：`python main.py --quick-test-for-ci` exit 0，torch 2.11.0+cu128，RTX 5090 识别正常，custom_nodes 导入没有报错（日志里的 ERROR 是 ComfyUI-Manager 连不上 GitHub，属于网络问题）；ComfyUI 子模块 git status 不变
      - 发现（未处理）：`miniMax/README.md` 第 4、47 行引用的 `100-openVLA/DAMO_README.md` 早在提交 `1cb6bbd` 就已删除，是改名前就存在的死链
      - 备份 `venv-200-video-ComfyUI/`；已写 `200-video/START_HERE.md`
- [ ] **第 5 步：文档与索引**
  - [ ] 根 README 按 1xx/2xx/3xx/5xx/底座重写为纯索引（每目录一行，链接各自 `START_HERE.md`）
  - [ ] 新建 `docs/conventions.md`（第 7.2 节命名规范）
  - [ ] 新建 `.qoder/skills/README.md` 索引（第 7.3 节）
  - [ ] `models/README.md` 补齐 18 个 ModelScope + 3 个 HF 模型
  - [ ] 新建 `datasets/` 及索引（aloha、libero、pusht、HF parquet）
  - [ ] 散落文档归入各项目 `docs/`
  - [ ] 更新 `.gitignore` 注释
  - [ ] 各项目根目录新增 `START_HERE.md`（见第 6 节；已提前到第 4 步逐个写，此处只做最终核对：13 个顶层项目目录都有）
- [ ] **第 6 步（可选，单独决定）：空间清理**
  - [ ] MiniMax-H3 / Wan-Dancer-14B / pi0 中间 checkpoint / RynnMotion.zip
  - [ ] lingbot 项目内权重迁到 modelscope 缓存后软链回去

## 9. 待确认事项

- [x] 名称内部 `_` 也统一改 `-`，只改顶层目录名
- [x] 第 0 步未提交改动：原样随迁移，不加入 git
- [x] `21_demo` → `105-damo`（demo 为笔误，应为达摩院 damo）；`102-kingkongrobotics` → `302-kingkongrobotics` 保留原名
- [x] `.gitmodules` section 名是否改成与新 path 一致 → 用户决定先修，已完成（见 2.5）
- [x] 入口文档文件名用 `START_HERE.md`，并作为新建顶层目录的规范
- [ ] 子模块目录（`501-human-dexterous-teleop`、`201-colibri`）的入口文档：放子模块里不提交，还是提交到各自 fork
- [x] 第 6 节环境对照表中未确定项先写「待定」，由用户逐一运行后确定
- [ ] 4.7 中改名前就已失效的 editable 包（openvla、lingbot-video ×4、dlimp）：第 4 步顺手修，还是保持原样；误装在 frankmocap / 4D-humans / lingbot-depth 里的 lingbot-video 是否卸载
- [ ] 提交时机：用户决定第 0/1 步暂不单独提交，与改名一起之后再提交（当前第 1、2 步改动均为已暂存未提交）
- [x] 根目录整理：`user_profile.md` 放 `.qoder/rules/` 自动生效；两个 html 删除；README 统一大写；顶层目录专有名词保留大小写

## 10. 变更日志

| 日期 | 内容 |
|---|---|
| 2026-10-11 | 完成工作区 review，确定四大块 + 公共底座的需求理解 |
| 2026-10-11 | 确定编号规范：1xx 具身模型 / 2xx 非具身模型 / 3xx 真机与趣味复现 / 4xx 留空 / 5xx 数据 infra，统一用 `-`；`modules` 改名 `models` |
| 2026-10-11 | 确定 `23_video` 不拆、`human_dexterous_teleop` 归 5xx |
| 2026-10-11 | 创建本文档，尚未执行任何目录改动 |
| 2026-10-11 | 确定第 0 步未提交改动原样随迁移、不加入 git |
| 2026-10-11 | 确定入口文档名为 `START_HERE.md` 并作为新建顶层目录规范；未确定的环境标「待定」 |
| 2026-10-11 | **执行第 0 步**：新增 `scripts/check_venv_editable.py`，生成 `docs/ws-restructure-baseline.txt`（editable 包 import 基线 + git/子模块状态快照）；发现 6 处改名前就已失效的 editable 包（记入 4.7） |
| 2026-10-11 | **执行第 1 步**：`.gitmodules` 修 4 条失效条目、补 3 条；`.git/config` init 8 条、删 5 条失效条目；`git submodule status` 恢复正常。尚未提交 |
| 2026-10-11 | 确定名称内部 `_` 一并改 `-`；`21_demo` 改名 `105-damo`（达摩院）；第 0/1 步暂不单独提交 |
| 2026-10-11 | **执行第 2 步**：`git mv` 完成 14 个目录改名（13 个项目 + `modules` → `models`），子模块状态与 SHA 验证通过；已暂存未提交。路径引用、venv 尚未修复（第 3、4 步） |
| 2026-10-11 | `user_profile.md` 译为中文并加入 mjlab；新增第 7 节（根目录整理、命名规范、Skill 管理），原第 7~9 节顺延为第 8~10 节；**执行 2.4**：根目录文件归位、删 2 个 html、5 个 `readme.md` 改大写、`.gitignore` 加 `.qoder` 规则（已暂存未提交） |
| 2026-10-11 | 新增第 6 节「项目入口文档」需求：每个项目根目录放一个极简入口文档（暂定 `START_HERE.md`）；补全环境清单为 13 个 venv + 6 个 conda env；原第 6~8 节顺延为第 7~9 节 |
| 2026-10-11 | 复核 2.4：根 README「顶层文件」表中 catalog 路径改为 `docs/embodied-ai-model-test-catalog.md`，并补登 `.qoder/rules/user-profile.md` |
| 2026-10-11 | **执行第 3 步（除子模块内引用）**：tracked 60 个文件旧名替换、`modules` → `models`、ignored 本地配置（`.libero/config.yaml`、jumper out README）、wandb 3 个软链重建；unifolm yaml 的 `data_dir` 待确认。均未提交 |
| 2026-10-11 | 第 3 步完成：unifolm yaml 的 `data_dir` 改为相对路径（子模块内未提交改动） |
| 2026-10-11 | **执行 2.5**：`.gitmodules` / `.git/config` 共 19 个子模块 name 改为与新 path 一致，`microduck_rl` 的 `.git/modules` 目录与 gitdir 同步迁移；`submodule status` 与改前一致，已暂存未提交 |
| 2026-10-11 | **执行 4.1 试点**：`100-openVLA/.venv` 修复完成（bin 67 个文件、`libero_source_path.pth`、libero/openvla editable 重装），import 全部 ok，4.7 的 openvla/prismatic 一并修好；扫描出 B 类 5 个改名前就已失效的 venv；发现 `~/.libero/config.yaml` 的 101 旧路径 |
| 2026-10-11 | 用户提醒 `START_HERE.md` 漏写：补写 `100-openVLA/START_HERE.md`（命令实测通过）；写入时机改为第 4 步每修好一个项目环境就写该项目的入口文档 |
| 2026-10-11 | 100-openVLA README 按「怎么跑 → 模型输入输出/EE 控制 → 各阶段」重写，并修正夹爪极性与 LIBERO 动作量纲两处错误（docs/02 同步） |
| 2026-10-11 | **执行 4.2 之 101-pi-serial-models**：venv + `~/.libero/config.yaml` 修复；顺带修 `0482e02` 遗留的权重路径（脚本 5 个、checkpoint JSON/README 31 个、docs 3 个）；README 修正；实跑 pi0.5 LoRA / ACT 评测通过；写 START_HERE.md。均未提交 |
| 2026-10-11 | **执行 4.2 之 102-cosmos**：conda `cosmos` 无需修；uv venv `3rd/cosmos-framework/.venv` 修复（bin 151 个 + .pth/direct_url）；`verify_models.py` 的 MODELS_DIR 改为 `~/ws/models`；README 修正，写 START_HERE.md；验证通过，均未提交 |
| 2026-10-11 | 按用户要求改为按编号顺序推进：**执行 103-lingbot**（B 类）——lingbot-video / lingbot-world-v2 两个 .venv 与 conda lingbot-depth 修复，lingbot-video、mdm 重装为 editable；lingbot-depth example 实跑通过；写 START_HERE.md；均未提交 |
| 2026-10-11 | **执行 104-unitreerobotics**：.venv bin 61 个文件 + 2 个 editable 的 .pth/finder/direct_url 用 sed 修复；查明基线中 dlimp FAIL 是缺 `tensorflow_datasets`，不是路径问题，未补装；WMA 推理短跑通过；写 START_HERE.md；均未提交 |
| 2026-10-12 | 按用户要求删除 `103-lingbot` 下两个 `.venv`（共约 15G），START_HERE 同步改为「待定」。**执行 200-video/ComfyUI**：bin 32 个文件 sed，quick-test 通过；写 START_HERE.md；均未提交 |
