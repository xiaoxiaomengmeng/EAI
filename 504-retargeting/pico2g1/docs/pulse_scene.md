# PULSE 可选互动场景接入

PULSE 使用 `--scene pulse`；默认 empty，兼容 `--xixi-scene`。两个模拟器的 replay / live 子命令均支持。新增的 strike / orbit / switchyard 见 [互动场景说明](play_worlds.md)。不自动打开 GUI，不需要重新生成 Blender。

## 运行

在仓库根目录，使用现有 SONIC 环境：

```sh
g1_sonic/.venv/bin/python g1_sonic/launch_g1_sonic.py live DEVICE_IP --scene pulse
g1_sonic/.venv/bin/python g1_sonic/launch_g1_sonic.py replay /path/to/session --scene pulse
```

轻量 IK 同样追加 `--scene pulse`。独立加载可使用 `mujoco.MjModel.from_xml_path("assets/pulse_arena/scene.xml")`；资源目录已移除生成脚本和专用查看器。

## 接入细节

`sim_scenes` 只在选择 PULSE 时导入其 `environment.xml` 和 `interactive.xml`，并解析仓库内的绝对网格路径；不会引用原 Codex 输出目录。原机器人 floor 被场景地面替代，保留模拟器原有照明。场景 visual group 1 不参与碰撞，碰撞 group 3 在 UI 中隐藏但仍然生效。场景碰撞 affinity 兼容 G1 / Wuji 当前激活的碰撞位。

可动物件追加在 G1 + Wuji 后面：机器人保持 76 qpos，场景追加 31 qpos / 27 个速度自由度，总计 107 qpos；29 个机器人 actuator 不变。按钮不继承机器人关节的摩擦和 armature。合并后的积分器及步长沿用对应模拟器，不用独立场景的全局选项覆盖机器人配置。

SONIC 每个物理子步更新计分，重置同时重置物件和分数。IK 在每个输入帧期间保持机器人姿态，物件运行约 20 ms 的物理步进，提供近似运动学交互。按键 / 托盘 / 球门 / 路点计分由原场景 `game.py` 提供。尚未以真实遥操作数据或实体硬件验证全身平衡与抓取成功率。

资源目录仅保留 3 个 MJCF、11 个二进制 MSH 网格、运行配置 design.json 和简短说明。计分逻辑位于 `sim_scenes/pulse_game.py`。Blender、OBJ、预览、演示和生成 / 验证脚本均已从此目录移除。原始制作包仍保存在制作工作区中。

转换未简化几何，顶点 / 三角面 / 物理参数一致，实际渲染像素差为 0。目录由 15.52 MB 降至约 6.74 MB，节省约 57% 磁盘空间；不声称运行内存同比降低。

## 验证

```sh
g1_sonic/.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

第一组测试实际编译两个加载器的 G1 + Wuji + PULSE，验证原机器人索引、单一地面、出生点、按钮摩擦、道具步进 / 重置，并用离屏 Qt 渲染及真实 SONIC ONNX 推理验证源帧处理。精简后的独立 scene.xml 另作加载及步进检查。空场景和西溪加载另作回归加载检查。
