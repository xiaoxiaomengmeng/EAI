# AGENTS.md — ws 工作区结构与约定

> 受众：在本工作区工作的 AI 编码代理。人类阅读入口是根 [README.md](README.md)（纯索引）。
> 本文件是工作区约定的**单一来源**（2026-10-20 起并入原 docs/conventions.md，该文件已删除）；目录重组的历史沿革见 [docs/ws-restructure-plan.md](docs/ws-restructure-plan.md)。
> 本文件**不复制**项目索引与状态（那两份信息只维护在根 README 与各 START_HERE，多份必漂移）。

## 1. 工作区规划

- 环境：Ubuntu 22.04 + NVIDIA RTX 5090（sm_120），CUDA 12.8/13.0，Python 3.10/3.12；torch 安装用根 `scripts/` 统一脚本。
- 结构：**编号项目目录 + 公共底座** 两层；每个顶层目录是一个独立项目（多为父仓库的子模块或含子模块）。

| 编号段 | 用途 |
|---|---|
| 1xx | 具身智能模型（VLA / 世界模型 / RL 框架） |
| 2xx | 非具身模型与工具 |
| 3xx | 真机部署、人形机器人、趣味复现 |
| 4xx | 预留，留空 |
| 5xx | 具身数据 infra（采集 / 重定向 / 遥操作 / 数据合成） |
| 无编号 | 公共底座：`models/`（权重软链）、`datasets/`（数据索引）、`assets/`（仿真资产）、`scripts/`（统一部署脚本）、`docs/`（工作区文档） |

- 目录命名 `NNN-kebab-case`，专有名词保留原大小写（`100-openVLA`、`502-DexAgent`）。
- 各项目用途 / 环境 / 验证状态：查根 README 索引表；怎么跑：查项目 START_HERE.md。

## 2. 子项目标准结构

新建项目**必须**包含以下骨架（`[ ]` 为可选）：

```
NNN-project/
├── START_HERE.md    # 必需：入口。先讲怎么跑，再讲输入输出与控制方式；状态行用 ✅🟡⚠️
├── README.md        # 必需：详细说明（模型 IO、部署链路、技术细节）
├── .gitignore       # 必需：项目特有忽略（视频/权重/运行产物不入库）
├── docs/            # 必需：专题文档，命名 NN-主题.md
├── repos/           # 新建项目必需：上游仓库以子模块形式放这里
├── [scripts/]       # 项目脚本，命名 NN_动词_对象.sh/.py
├── [pics/]          # 实测截图 / 录屏
└── [config/]        # 本地配置；含密钥时必须 gitignore
```

**2026-10-20 实际盘点**（新项目按标准执行，老项目缺口按待补处理、不强制迁移）：

| 现状 | 项目 |
|---|---|
| 全达标 | 100、101、104、502、503 |
| 缺 `README.md`（START_HERE 已有） | 102、103、105、301、302、500 —— 补齐时从 START_HERE 拆分详细内容 |
| 缺 `README.md` + `docs/` | 200-video |
| `repos/` 的历史兼容 | 老项目上游代码位置不一：102 用 `3rd/`、103 用 `lingbot-*` 系列目录、105/300/301/302/500 直接放根。**不迁移**；agent 找上游代码时按此定位，别假设一定在 `repos/` |
| 整目录型子模块特例 | `201-colibri`、`501-human-dexterous-teleop`：子模块 path 即项目目录本身，写在其内部的约定文件（README/START_HERE）**不进父仓库提交**，仅本地存在 |

## 3. 文档体系

- 三层：`START_HERE.md`（入口 + 当前状态）→ `README.md`（详细）→ `docs/`（专题深读）。
- 固定文件名全大写：`README.md`、`START_HERE.md`、`LICENSE`；不使用小写 readme.md。
- 根目录只保留 `README.md`、`AGENTS.md`、`LICENSE`、`.gitignore`、`.gitmodules`、`.qoder/`、编号项目目录与公共底座；其余工作区级文档一律进 `docs/`。
- 项目 `docs/` 内命名 `NN-主题.md`（01、02…顺延）；已有文档不改名以免断链。
- **新增专题文档后的挂载动作**：①挂入该项目 `START_HERE.md` 的文档索引；②建立相关文档交叉引用；③AGENTS.md 与根 README 均**不登记**（前者只管规则，后者只索引到 START_HERE 层）。
- 技术深读文档统一体裁：**《XX架构与关键技术解析》**（2026-10-20 起，502/503/104 已统一）。架构与关键技术写在一篇里，不再分设两篇。
- 状态徽标三档：✅ 已验证/可运行；🟡 基础可用/部分验证；⚠️ 暂未验证/环境待定。状态反映**项目自身成熟度**，换机复现需重建环境。
- 状态信息**只收敛两处**：项目 START_HERE 状态行 + 根 README 状态列。项目自有 README 只描述用途，不写验证状态（三处同步必漂移）。
- 每个验证阶段写清：目的、命令、场景、判据、坑总结。

## 4. 资产与环境约定

- **模型权重**：统一软链到 `models/`（指向 `~/.cache/modelscope`、`~/.cache/huggingface`），本地源码离线加载，不依赖 `trust_remote_code` 联网。
- **数据集**：`datasets/` 只放位置索引；实体数据（lerobot v2 格式等）留各项目内。
- **仿真资产**：`assets/mujoco/mujoco_menagerie` 机器人模型库，各项目引用不复制。
- **验证阶梯**（模型复现类工作）：单步推理 → 官方 benchmark → 自定义仿真闭环，前一阶通过才进下一阶。
- **复现对齐红线**：坐标系、四元数顺序、归一化统计、动作空间（EE/关节）、单位与夹爪极性——每一处约定都要与源实现核对。

## 5. Agent 行为约定

- **上游子模块不改源码**；兼容问题在封装层（shim）解决；装源码包用 `--no-deps`，保护已验证的 torch/CUDA 栈。
- 外部仓库一律不向其内部写文件、不动其 git 状态。
- **不可逆操作**（删除、覆盖、强推）先确认；改重要文件前留副本（可放 `.git/` 下如 `.git/restructure-backup/`，不污染工作区）。
- **git 提交**：工作区未配置提交身份，用一次性 `git -c user.name=... -c user.email=...`（身份由用户确认）；提交用 pathspec 限定范围（注意 CWD 在子目录时 pathspec 相对解析）；**提交后必须核对新 hash 与暂存区状态**，不能只看 `git log` 回显（身份缺失时 commit 会静默中止，log 显示的是旧提交）。
- **大文件**：视频、权重、数据集、运行产物不入库；训练 checkpoint 只保留关键节点（如首/峰/末）。
- **文档与实测同步**：验证完成后立即更新 START_HERE 状态行与根 README 状态列；实测数据标注时间与来源，与推测明确区分（标「实测」/「待核」）。

## 6. git 与子模块细则

- 子模块 URL 统一使用 https；`.gitmodules` 里 submodule section 改名与目录改名**同步进行**。
- `.gitignore` 分层：根 `.gitignore` 管全局规则，各项目目录一份管项目特有规则；**带目录名前缀的规则在目录改名时必须同步**（全量清单见根 `.gitignore` 注释）；历史性注释不改。

## 7. Qoder 扩展（.qoder/）

- 规则放 `.qoder/rules/`（`user-profile.md` 为 `trigger: always_on` 身份设定，每次对话自动加载）；只跟踪 rules 与 skills，其余 `.qoder/` 本地状态忽略。
- skill 放 `.qoder/skills/<name>/SKILL.md`，名字全小写 kebab-case、动词开头；索引登记在 `.qoder/skills/README.md`。
- 跨项目复用的 skill 从 `~/.qoder/skills/<name>` 软链到本仓库源文件，只维护一份；同名时项目级优先。

## 8. 快速索引

| 要做什么 | 去哪看 |
|---|---|
| 找项目、看状态 | 根 [README.md](README.md) |
| 跑某个项目 | 该项目 `START_HERE.md` |
| 建新项目 | 本文 §1 编号规则 + §2 骨架 |
| 写技术深读文档 | 本文 §3（统一体裁《XX架构与关键技术解析》） |
| 重组方案与执行记录 | [docs/ws-restructure-plan.md](docs/ws-restructure-plan.md) |
| RTX 5090 环境 | 根 [README.md](README.md) 环境段 + `scripts/` |
