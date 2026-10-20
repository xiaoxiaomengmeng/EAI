# 502-DexAgent · 入口

> DexAgent 本地复现：**Agentic Human2Sim2Robot 框架**（arXiv 2609.35318）——人类操作视频进，物理验证的灵巧手机器人轨迹 + 随机化多视角训练视频出。
> 上游：<https://github.com/dexagent1/DexAgent>

**状态：⚠️ 暂未验证**（目录已建，上游未克隆、环境未建；按下面验证阶梯推进）

## 怎么跑

### 1. 克隆上游（子模块）

```bash
cd ~/ws
git submodule add https://github.com/dexagent1/DexAgent.git 502-DexAgent/repos/DexAgent
```

### 2. 建环境（独立 conda，勿与其他项目共用）

上游锁 **MuJoCo 3.10 + NumPy 1.x**（其他版本接触仿真不同，勿升级）：

```bash
conda create -y -n dexagent python=3.11
conda activate dexagent
pip install -r ~/ws/502-DexAgent/repos/DexAgent/requirements.txt   # 依赖冲突时用 --no-deps 保护已验证包
```

### 3. 零成本验证（不花一分钱 API）

```bash
conda activate dexagent
bash ~/ws/502-DexAgent/scripts/01_render_prompt.sh
# 产出 out_prompt/{system.md, tools.json, user.txt}：stage1 发给 LLM 的完整 prompt 包
```

## 输入 / 输出 / 控制方式

| | |
|---|---|
| 输入 | 固定相机拍摄的人类第一人称操作视频 + 一句任务 prompt（`config/dexagent.yaml`，从上游拷贝修改，含 key 不入库） |
| 输出 | `runs/<name>/stage{1..4}/`：语义 JSON → MuJoCo 场景 → 物理验证轨迹 `trajectory.npz` + `sim_video.mp4` → 随机化渲染多视角视频 |
| 控制 | LLM agent 经 function calling 驱动工具库（skills 产出 / verifiers 判决），生成-验证循环；四阶段可单跑可串跑 |

## 验证阶梯

| 阶段 | 目的 | 命令 | 判据 | 状态 |
|---|---|---|---|---|
| 0 | 看 prompt 包与工具 schema，理解 harness | `scripts/01_render_prompt.sh` | 三个文件生成 | ⬜ |
| 1 | mock 干跑全流程（不调 API） | `run_dexagent.py <yaml> --llm mock:turns.json` | 流程走完不崩 | ⬜ |
| 2 | Stage 1 单阶段实跑（budget 设 5–10 USD） | `stage1_semantic_understanding/run_stage1.py <yaml>` | `scene_semantics.json` 过 verifier | ⬜ |
| 3 | 全链路 stage1→3 | `run_dexagent.py <yaml> --to stage3` | `trajectory.npz` + `sim_video.mp4` | ⬜ |
| 4 | Stage 4 数据渲染 | `run_stage4.py <stage4.json> --steps randomize,export,render,composite` | 多视角视频 + 轨迹 | ⬜ |

## 已知坑

- **`agent.budget_usd` 必填**：不设直接拒跑（BudgetError），第一次先给小额度。
- **Stage 4 `plate` 步需 40GB 显存**（VOID 5B 峰值 38GB），5090 32GB 会 OOM——用 `--steps` 跳过 plate 先跑其余四步。
- **SAM 3.1 是 gated 模型**：HF 上申请访问 + `hf auth login` 后才能下权重。
- 感知模型（SAM3/SAM2/HaMeR/深度）各自独立环境，`dexagent.yaml` 的 `environments:` 按解释器路径路由，缺包时工具自报、agent 换路，主 harness 保持 CPU 轻量。

## 目录

```
502-DexAgent/
├── repos/DexAgent/    # 上游子模块（不改源码，兼容问题在封装层解决）
├── docs/              # 专题笔记（01-架构与关键技术解析）
├── scripts/           # 验证脚本
├── config/            # 本地 dexagent.yaml（含 key，gitignore；从 repos/DexAgent/dexagent.yaml 拷贝）
├── runs/              # 运行输出（gitignore，yaml 的 run.output_dir 指向这里）
├── out_prompt/        # 阶段 0 产物（gitignore）
├── START_HERE.md      # 本文件
└── README.md
```

## 关联

- **503-do-as-i-do**（规划中）：malik-group 固定管线对照路线（重建→重定向→UR3e 部署）。两边感知栈（SAM3/SAM3D/HaWoR/MoGe）可共享 conda env、**同一双 Sharpa Wave 灵巧手**的 MJCF 资产与关节约定可统一；结合思路：用 do-as-i-do 的确定性重建喂 DexAgent Stage 3 的 agentic 轨迹搜索。待 503 启动后落文档。
