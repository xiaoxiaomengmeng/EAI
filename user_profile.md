## User Preferences
- Communication language: Chinese
- Code style: minimal edits over full rewrites
- Debugging approach: prefers detailed logging and step-by-step verification
- Visualization: requests real-time image display with action annotations during inference
- Writing style: incorporates casual terms like '捯饬', '哟' while maintaining professionalism

## Role & Positioning
- 具身智能全栈算法工程师：横跨「算法研究 → 数据管线 → 工程部署」三层，不局限于单点

## Tech Stack
- Python, ROS2 (proficient)
- MediaPipe, MuJoCo (proficient)
- Computer vision, robotics simulation (domain expertise)
- 策略学习：π0/π0.5、VLA（LingBot-VLA、OpenVLA、Cosmos）
- 感知算法：SAM 分割、深度估计（LingBot-Depth、MoGe）、光流
- 控制：IK / FK / OSC / DLS-IK / 位置伺服
- 框架：lerobot、robosuite、PyTorch、transformers

## Focus Areas
1. **算法研究**：策略学习、感知算法应用、Sim-to-Real 泛化边界验证
2. **数据管线**：多模态数据预处理、归一化统计、感知模型串联（SAM→深度→光流→标注）
3. **工程部署**：MuJoCo 环境封装、策略推理链路部署、真机/仿真一致性验证、多仓库工程化管理
4. **开源复现**：关注实际可运行性，不仅是论文复现

## Workflows
- Likes to reference existing implementations (e.g., aero-hand-open/ros2) for problem-solving
- Requests clear visualizations and dual-window displays for debugging
- 重视复现链路的每一处约定对齐（坐标系、四元数顺序、归一化统计）
- 关注零样本泛化验证，而非仅微调/采数据
