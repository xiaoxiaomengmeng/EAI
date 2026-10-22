# G1 三视角 LeRobot 录制

轻量 IK 和 Mac SONIC 窗口均有可展开/收起的数据采集区域，包含三路相机画面及录制控件。点击主画面右上角的 26×26 小箭头图标切换，图标悬浮在主画面内，不占用布局高度。默认展开，三路采集画面按可用宽度保留 16:9 显示高度；收起后整个采集区域让出空间，主画面自动铺满。

收起不会中断录制或导出。图标在录制中变红、导出中变黄，悬停可查看完整状态，重新展开时立即显示当前视角。主画面随可用区域调整渲染宽高，保持透视比例；极大窗口的预览最长边限制为 1600 像素，录制视频仍固定为 1280×720。沿用现有 replay / live 启动方式，无需连接天机 IK 或手部在线服务。

## 操作

1. 填写任务描述，例如“拿起桌上的物块”。
2. 点击“保存位置”选择父目录，默认是仓库根目录下的 `recordings/`。
3. 点击“开始录制”。成功处理的机器人动作会保存到磁盘；等待输入、拖动主相机不会产生数据帧。
4. 点击“停止并保存”。后台根据录下的轨迹生成三路高清视频，界面显示导出进度。
5. 显示“已保存”后点击“打开目录”。每次开始录制均创建一个带时间戳的独立数据集，其中包含一个 episode。

回放结束、输入错误、进入标定/姿态调节或 SONIC 重置会自动结束当前录制并保存已有帧。标定期间不可开始录制。关闭窗口会先完成当前导出再关闭；未录到动作的空录制不会生成有效数据集。

## 输出

输出遵循示例中的 LeRobot **v2.1** 布局：

```text
g1_YYYYMMDD_HHMMSS_xxxxxx/
  data/chunk-000/episode_000000.parquet
  videos/chunk-000/
    observation.images.head/episode_000000.mp4
    observation.images.left_wrist/episode_000000.mp4
    observation.images.right_wrist/episode_000000.mp4
  meta/
    info.json
    tasks.jsonl
    episodes.jsonl
    episodes_stats.jsonl
    stats.json
  raw/
    model.mjb
    capture.json
    frames.bin
```

相机均为 1280 × 720、30 fps、H.264 编码、yuv420p、MP4，无音频。高清画面从保存的仿真模型和轨迹重新渲染，预览小图不参与编码。相机固定位置与界面一致：左右手腕相机位于手腕局部下方 12 cm、前方 1.5 cm，向上倾斜 40°，垂直视场 55°，以覆盖张手和握拳时的五指细节。相机保留手腕滚转；主画面旋转/缩放不会改变输出视角。

Parquet 中每行对应三路视频的同一个帧号和时间戳。`info.json` 给出每个 feature 的类型、维数及关节名称：

| 字段 | 内容 |
| --- | --- |
| `observation.state` | G1 29 个本体关节 + Wuji 40 个手指关节，共 69 维实际角度，单位 rad |
| `action` | 同一关节顺序的 69 维目标角度，单位 rad |
| `observation.base_pose` | 浮动基座世界位姿 `[x, y, z, qw, qx, qy, qz]`，位置单位 m |
| `timestamp` | episode 内的秒数，从 0 开始，每帧增加 1/30 |
| `frame_index`、`index` | 从 0 开始的连续帧索引 |
| `episode_index`、`task_index` | 当前单 episode 数据集内均为 0 |

关节顺序取模型中 pelvis 子树内标量关节的顺序，并写入 feature names；不包含场景道具的自由度。使用关节名匹配 URDF，不要按天机示例的 54 维索引读取。浮动基座单独保存，因此站位和整体旋转不会丢失。这是机器人轨迹数据，不是重新生成一份 URDF 模型文件。

SONIC 的 `action` 为限位后的策略目标以及手指重定向目标，`observation.state` 为仿真实际关节角度，两者可以不同。轻量 IK 直接将解出的目标赋给运动学模型，源采样点上 state 与 action 相同。

## 时间同步

录制使用**已处理仿真步的时间**，不是墙上时钟或原始设备采样时间。每个成功处理的输入对应 1/50 秒；源数据等待、标定和录制前的时间不计入。现有实时预览会优先处理最新输入，导出记录的是实际执行的仿真轨迹，不能用它恢复被预览丢弃的设备帧。

停止时以 30 Hz 重采样：关节和场景姿态在相邻仿真状态之间插值，四元数采用旋转插值；目标动作采用零阶保持。三路相机从该行对应的同一姿态渲染。共有 N 个源样本时，输出 `ceil(N × 30 / 50)` 帧，末尾不足一个视频周期时保持最后姿态。所以显示/编码较慢时数据不会发生视频与轨迹错位，但导出的视频时长不等同于操作的实际耗时。

## 依赖与恢复

录制需要系统 PATH 中的 `ffmpeg` 和 `ffprobe`，FFmpeg 必须支持 `libx264` 编码。普通预览不需要这些程序。macOS 可通过 `brew install ffmpeg` 安装；Python 侧使用工程已有 MuJoCo、NumPy 和 PyArrow。

原始轨迹和冻结的 MuJoCo 编译模型保留在数据集目录中。导出出错时，不会发布表示完成的 `meta/info.json`；界面会给出失败原因和原始轨迹位置。确认原录制已停止，排除磁盘空间或 FFmpeg 问题后，在仓库根目录执行：

```sh
g1_sonic/.venv/bin/python -m sim_scenes.lerobot_recording /absolute/path/to/g1_recording
```

恢复依赖兼容的 MuJoCo 版本；编译模型用于本地恢复，不代替机器人 URDF 资产的版本管理。已完成的数据集和已有目录不会被新录制覆盖。

格式字段与统计数组约定参考用户提供的 v2.1 示例，以及 [LeRobot v0.3.3 数据集工具](https://github.com/huggingface/lerobot/blob/v0.3.3/src/lerobot/datasets/utils.py)和[统计实现](https://github.com/huggingface/lerobot/blob/v0.3.3/src/lerobot/datasets/compute_stats.py)。不要求安装完整 LeRobot 训练环境即可录制。
