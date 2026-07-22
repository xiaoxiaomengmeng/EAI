"""Entrypoint for the OpenVLA simulation demo.

核心实现已经拆分到 /home/creator/ws/openVLA/openvla_sim/：
- config.py: 默认路径和参数配置
- mujoco_env.py: MuJoCo 场景、渲染、step、距离诊断
- openvla_policy.py: OpenVLA base 模型加载和 action 推理
- action_adapter.py: OpenVLA 7DoF action 到 actuator ctrl 的映射
- runner.py: rollout 主循环、视频和 JSON 日志保存
- cli.py: 命令行参数入口
- robots/: 机器人配置（franka, wx250s）

运行方式：
    python run_openvla_sim.py
    python -m openvla_sim

指定机器人（默认 franka）：
    python run_openvla_sim.py --robot franka
    python run_openvla_sim.py --robot wx250s
"""

from openvla_sim.cli import main


if __name__ == "__main__":
    main()