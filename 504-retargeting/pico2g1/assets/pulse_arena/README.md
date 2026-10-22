# PULSE 运行资源

这是精简的 MuJoCo 回放资源目录，不包含 Blender 源文件、生成工具、演示及预览图。

- `scene.xml`：可直接用 MuJoCo 加载的独立场景入口。
- `environment.xml`、`interactive.xml`：固定环境及互动道具。
- `assets/*.msh`：11 个二进制网格，保留原顶点和三角面。
- `design.json`：ego2sim 的出生点、目标位置等运行配置。

在 ego2sim 的 replay/live 命令后添加 `--scene pulse` 使用。加载与计分代码位于仓库 `sim_scenes/`，无需安装 Blender 或运行生成脚本。详细说明见 `../../docs/pulse_scene.md`。

资源以 MuJoCo 3.12.0 验证。二进制网格是 MuJoCo MSH 格式：[官方格式说明](https://mujoco.readthedocs.io/en/stable/XMLreference.html#asset-mesh)。此转换减少磁盘体积，不减少三角面数量或承诺运行内存降低。
