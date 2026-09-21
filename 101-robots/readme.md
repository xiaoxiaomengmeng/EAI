# 101-robots

本目录用于收纳机器人相关项目。当前包含 **RynnMotion** —— 一个面向快速原型验证、原语运动控制、异构遥操作、数据采集与多平台部署的机器人运动框架。

## RynnMotion 支持的机器人

### 1. 真实硬件驱动（`RynnMotion/robots/`，10 个厂商模块）

| 模块 | 厂商 / 机器人 | 形态 |
| --- | --- | --- |
| RynnFranka | Franka FR3（+ Robotiq 夹爪） | 7DOF 机械臂 |
| RynnFlexiv | 非夕 Flexiv Rizon 4s（集成 RDK/TDK/DDK） | 7DOF 力控臂 |
| RynnRealman | 睿尔曼 RM75 | 7DOF 机械臂 |
| RynnPiper | AgileX Piper | 6/7DOF 机械臂 |
| RynnDamiao | 达妙 DM6（SocketCAN / 串口电机） | 6DOF 机械臂 |
| RynnLeRobot | LeRobot 生态：SO-101 / SO-101 6DOF / LeKiwi | 桌面臂 + 移动操作 |
| RynnTianji | 天机双臂（HTC Vive 遥操作 + Manus 手套） | 双臂 |
| RynnWuji | 无己灵巧手 ×2 + 天机臂（DiffIK 遥操作） | 灵巧手 + 双臂 |
| RynnSharpa | Sharpa（触觉感知 / 灵巧手 SDK） | 末端执行器 |
| RynnRokae | 珞石（目前仅 README 占位，驱动未落地） | 规划中 |

### 2. MuJoCo 仿真机械臂（`RynnMotion/models/3.robot_arm/`，12 款）

FR3、UR5e、Piper、RM75、SO-101、Rizon4s、ECO65（大象机器人）、DM6-6DOF（达妙）、OpenArm、SO-101-6DOF、SR5-6DOF、JAKA S5（节卡）。

其中有对应真机驱动的：FR3、Piper、RM75、Rizon4s、DM6-6DOF、SO-101、SO-101-6DOF；**仅仿真模型、暂无真机驱动包的 5 款**：UR5e、ECO65、OpenArm、SR5-6DOF、JAKA S5。

### 3. 其他形态

- **移动操作**：LeKiwi（SO-101 + 底盘）、XLeRobot
- **教学低自由度模型**：oneLink、twoLink
- **末端执行器**：UMI 夹爪
- Python 侧 mink 示例还内置 iiwa、Panda、宇树 G1 人形的运动学模型

## 总结

- 真机驱动层：**9 家厂商已落地，珞石规划中**
- 仿真层：**12 款机械臂**，其中 5 款暂无真机驱动
- 框架核心卖点：异构硬件统一抽象 —— 官方演示过一台 SO-101 主手同时遥操作 FR3、UR5e、Piper、RM75 四台从臂

## 附：UR5e 仿真模型能力详解（`RynnMotion/models/3.robot_arm/21.ur5e/`）

### 文件分工

| 文件 | 作用 |
| --- | --- |
| `mjcf/ur5e_2f85_links.xml` | 核心：UR5e 6 轴 + Robotiq 2F85 夹爪，23 个 body / 14 个关节，含真实惯量参数 |
| `mjcf/ur5e_actuators.xml` | 7 个驱动器 + split 腱传动 + 夹爪四连杆等式约束 + 自接触排除 |
| `mjcf/ur5e_sensors.xml` | **25 个传感器**（见下） |
| `mjcf/ur5e_defaults.xml` | 伺服增益、力限、摩擦、关节限位等物理参数类 |
| `mjcf/ur5e_assets.xml` | 20 个臂部 OBJ 网格 + 引用 `2.end_effector/11.robotiq` 的 8 个夹爪 STL |
| `mjcf/ur5e_contacts.xml` | 6 对夹爪内部自碰撞排除 |
| `mjcf/ur5e_pinocchio.xml` | **纯运动学精简版**（无夹爪/无碰撞体），供 Pinocchio 解析雅可比和动力学 |
| `mjcf/ur5e_robot.xml` | 独立完整模型，含 3 个预设位 `home` / `standby1` / `standby2` |
| `scene/scene.xml` | 主场景：地面、灯光、1920×1080 固定相机、RGB 基坐标轴 |
| `scene/scene_pickplace.xml` | 抓取任务场景：6 个 2cm 彩色方块（freejoint 可抓）+ 1 个固定马克杯道具 |

### 具体能力

1. **真实尺寸 6DOF 运动学** — 连杆按 UR5e 实际 DH（基座高 0.163、大臂 0.425、小臂 0.392 等），末端 `EE` site 位于法兰外 0.25m 并做工具坐标系翻转；另有 shoulder/elbow/wrist 三个辅助 site。关节限位：肘关节 ±π，其余 5 轴 ±2π。
2. **高保真动力学** — 总质量 22.05 kg（真机 20.7kg + 夹爪），每个连杆有实测惯量，臂关节 armature=0.1。伺服分级：大臂三轴 kp=2000/kd=400、±150Nm，腕部三轴 kp=500/kd=100、±28Nm；ctrl 直接给目标关节角即位置控制。
3. **Robotiq 2F85 机构级仿真** — 完整四连杆（左右各 driver/coupler/spring_link/follower 共 8 铰链，含被动弹簧关节），单个 `grip` actuator 经 split 腱同步两指，控制量范围 0–255。实测语义：**0 = 闭合（弹簧回位，指垫间距 14.9mm），255 = 张开（98.4mm，行程 83.4mm）**——方向与真机 Modbus 协议（255=夹紧）相反，写控制代码时需注意；指尖硅胶垫摩擦 0.6–0.7，可稳定抓物。
4. **25 路传感器** — 末端位姿四元数/线速度/角速度/陀螺仪/加速度计/接触力（7 路）+ 6 关节位置、速度、实际输出力矩（18 路）。
5. **碰撞与渲染分离** — 外观 mesh 走 group 2 不参与碰撞，臂部用胶囊体（group 3，仿真快），夹爪用 mesh 碰撞 + 高摩擦指垫；6 对内部机构接触已排除。
6. **任务场景** — pickplace 场景含 6 个 freejoint 彩色方块供抓取（马克杯为固定道具）；主场景预留 1080p 固定机位可离屏渲染。
7. **多臂扩展** — scene 用 `<replicate count="1">` 包裹机器人，改 count 即可复制多台（故 actuator 名带 `_0` 后缀）。

### 不具备 / 需注意

- 无 RGB/depth 相机传感器（camera 仅用于渲染）；夹爪无触觉 sensor（只有摩擦几何）
- 物理步长 1ms + implicitfast；`scene.xml` 无 keyframe（预设位仅在 ur5e_robot.xml）
- 工作空间半径 1.23m（`RynnMotion/config/robot.yaml`）

### 资产官方来源

这些模型不是 RynnMotion 从零建模，上游均可在官方渠道找到：

| 资产 | 官方来源 |
| --- | --- |
| UR5e 臂 | Google DeepMind **MuJoCo Menagerie** [`universal_robots_ur5e`](https://github.com/google-deepmind/mujoco_menagerie/tree/main/universal_robots_ur5e)；URDF 见 Universal Robots 官方 [`Universal_Robots_ROS2_Description`](https://github.com/UniversalRobots/Universal_Robots_ROS2_Description) |
| Robotiq 2F85 夹爪 | Menagerie [`robotiq_2f85`](https://github.com/google-deepmind/mujoco_menagerie/tree/main/robotiq_2f85)；官方 ROS 2 描述包 [`robotiq/ros`](https://github.com/robotiq/ros)（BSD-3，2026 年起 Robotiq 官方维护） |
| 组装后的整套工程 | 阿里巴巴达摩院 **RynnMotion** 仓库 [`alibaba-damo-academy/RynnMotion`](https://github.com/alibaba-damo-academy/RynnMotion)（即本目录的 `RynnMotion/`） |

### 本地逐项验证

无窗口自动核查（37 项断言，系统 python3 + 自带 mujoco 即可，无需编译 C++）：

```bash
cd ~/ws/101-robots/sim_robot
python3 verify_ur5e.py
```

覆盖：4 个 MJCF 入口加载、14 关节/7 驱动拓扑、质量惯量、关节限位、伺服增益力限、25 路传感器、3 个 keyframe、夹爪 0–255 全行程数值实验、pickplace 自由体统计、接触排除、28 个网格文件完整性、Pinocchio 版结构、home 位 1000 步稳定性（实测末端漂移 0.002mm）、可选 EGL 离屏渲染。

图形界面手动验证：

```bash
python3 view_ur5e.py                  # 交互模式，Control 面板拖滑条控关节/夹爪
python3 view_ur5e.py --demo           # 6 关节正弦摆动
python3 view_ur5e.py --drag           # 拖红球 IK 跟随（mocap + 阻尼最小二乘 IK + 位置伺服）
python3 view_ur5e.py --scene pickplace --drag   # 抓取场景中拖臂到方块上方，滑条调 grip 抓物
```
