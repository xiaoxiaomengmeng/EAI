# ORBIT 人形机器人互动场

沿桌侧走动，把青、紫、橙三枚滑盘分别推入对应 DOCK。整枚盘位于目标区、速度低于 0.08 并脱离机器人接触持续 0.5 秒，每盘 30 分，共 90 分。桌面高度 0.82 m。

在 ego2sim 的 SONIC 或轻量 IK 的 replay/live 命令后添加 `--scene orbit`。计分自动接入状态栏；SONIC 窗口的重置按钮同时还原机器人、道具和分数。轻量 IK 可重启当前回放重开一轮。

独立入口：`scene.xml`。`environment.xml` 为固定环境，`interactive.xml` 为可动道具，`design.json` 为任务和出生点，`assets/*.msh` 为实际引用的视觉网格。除此之外无需 Blender、贴图、生成脚本或预览图。物理地面为 z=0 无限平面，视觉场地 14 × 12 m。

视觉 group 1 无碰撞；碰撞 group 3 使用基本几何。框架内自动兼容 G1 身体与 Wuji 指尖碰撞位，默认隐藏碰撞层并关闭阴影。独立 MuJoCo 查看器需隐藏 group 3，方能看到正常视觉层。

规则实现于 `sim_scenes/task_game.py`，集成于 `sim_scenes/__init__.py`。更改目标区时应同步修改场景标记和 design.json。这里是接触驱动的被动物体，没有新增机器人控制器或道具 actuator。SONIC 手部目前直接写姿态；IK 保持机器人姿态推进道具，均不代表经过训练验证的力控抓取。等待输入时只渲染，不推进物理或计分。

详细运行与开发说明：仓库 `docs/play_worlds.md`。
