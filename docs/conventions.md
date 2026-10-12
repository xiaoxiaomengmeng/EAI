# ws 工作区约定（conventions）

> 2026-10 目录重组时与用户确认的约定汇总。变更约定时同步更新本文件与 [ws-restructure-plan.md](ws-restructure-plan.md)。

## 目录编号

顶层目录命名 `NNN-kebab-case`，专有名词保留原大小写（如 `100-openVLA`，不改小写）：

| 段 | 用途 | 现有 |
|---|---|---|
| 1xx | 具身基础模型（VLA / 世界模型 / RL 框架） | 100~105 |
| 2xx | 非具身模型与工具 | 200-video、201-colibri |
| 3xx | 真机项目与趣味复现 | 300-robots、301-microduck、302-kingkongrobotics |
| 4xx | 留空（预留） | — |
| 5xx | 数据、遥操作与 infra | 500-umr、501-human-dexterous-teleop |
| 底座 | 共享资源 | assets（仿真资源）、models（模型权重）、scripts（统一部署脚本）、docs（工作区级文档） |

## 文件与文档

- 固定文件名全大写：`README.md`、`START_HERE.md`、`LICENSE`；不使用小写 readme.md
- 每个顶层项目目录必须有 `START_HERE.md`（入口）→ 项目 `README.md`（详细）→ `docs/`（专题）三层
- **新文档一律放项目 `docs/` 下**，命名 `NN-主题.md`；已有文档不改名以免断链
- 根目录只保留 `README.md`（纯索引，按编号段分组链接各 START_HERE.md）、`LICENSE`、`.gitignore`、`.gitmodules`、`.qoder/` 与各项目目录；其余工作区级文档进 `docs/`
- 脚本命名 `NN_动词_对象.py/.sh`；工作区级统一脚本放根 `scripts/`，项目专用脚本放项目 `scripts/`

## git 与子模块

- 子模块 URL 统一使用 https；submodule section 改名与目录改名同步进行
- `.gitignore` 分层：根 `.gitignore` 管全局规则，各项目目录一份管项目特有规则；**带目录名前缀的规则在目录改名时必须同步**（全量清单见根 `.gitignore` 注释）；历史性注释不改
- 上游子模块不改源码，兼容问题在封装层（shim）解决；装源码包用 `--no-deps`
- 外部仓库一律不向其内部写文件、不动其 git 状态；其中 `201-colibri`、`501-human-dexterous-teleop` 为**整目录型子模块**（子模块 path 即顶层项目目录），写在其内部的文件（如 START_HERE）不会进父仓库提交，只作为本地未跟踪文件存在

## Qoder 扩展

- 规则放 `.qoder/rules/`（`user-profile.md` 为 `trigger: always_on` 身份设定，每次对话自动加载）；只跟踪 rules 与 skills，其余 `.qoder/` 本地状态忽略
- skill 放 `.qoder/skills/<name>/SKILL.md`，名字全小写 kebab-case、动词开头；索引登记在 [.qoder/skills/README.md](../.qoder/skills/README.md)
- 跨项目复用的 skill 从 `~/.qoder/skills/<name>` 软链到本仓库源文件，只维护一份；同名时项目级优先
