"""高踢任务的自定义 reward 函数（practice 侧，**不改子模块 mdp.py**）。

为什么放这里而不是塞进 `microduck_rl/src/mjlab_microduck/tasks/mdp.py`：
  - microduck_rl 是上游子模块，红线是「不改上游」；
  - `RewardTermCfg.func` 只是一个可调用对象，运行时解析，practice 侧定义的
    函数照样能挂进 env cfg（见 high_kick_cfg.py）。

设计目标（用户拍板）：**高度为主、射程为辅**——主奖励用「dense 离地高度」，
即球心离开地面的瞬时高度，逐步给分：球飞得越高、滞空越久，累积奖励越多，
天然鼓励「把球挑高」而不是「轻推滚地」。

三种候选塑形（本文件实现前两种，第三种留思路）：
  1. `ball_height_above_ground`（默认采用）：dense 离地高度，最直观、梯度最密。
  2. `ball_vertical_velocity`（备用）：奖励球向上的速度分量，只在起球瞬间给分，
     更强调「挑射爆发力」，但滞空阶段无梯度，探索更难。
  3. 峰值高度（peak height，暂不实现）：需要一条 per-env 的 running-max buffer，
     在 reset event 里清零、每步取 max(z)。它奖励「历史最高点」，能避免
     「球一直悬在高处刷分」的 exploit，但要额外维护状态。若后续发现策略用
     滞空刷分，再按此思路补一个 `reset_ball_peak` event + `ball_peak_height` reward。
"""

from __future__ import annotations

import torch

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv


def ball_height_above_ground(
    env: ManagerBasedRlEnv,
    asset_name: str = "ball",
    ball_radius: float = 0.035,
    max_height: float = 0.5,
) -> torch.Tensor:
    """球心离地高度（dense、线性、clamp 到 [0, max_height]）。

    离地高度 = 球心世界坐标 z − 球半径（球静止在地面时离地高度为 0）。
    clamp 上界 `max_height` 防止无限追求高度导致数值爆炸，也给「够高就行」
    一个明确天花板（microduck 本体仅 25cm，0.5m 已是很高的挑射）。

    返回形状 [num_envs]，与其它 reward 项一致，由 RewardManager 乘权重后累加。
    """
    ball: Entity = env.scene[asset_name]
    z = ball.data.root_link_pos_w[:, 2]
    height = (z - ball_radius).clamp(min=0.0)
    return torch.nan_to_num(height, nan=0.0).clamp(0.0, max_height)


def ball_vertical_velocity(
    env: ManagerBasedRlEnv,
    asset_name: str = "ball",
    max_speed: float = 3.0,
) -> torch.Tensor:
    """【备用塑形】球向上的速度分量，clamp 到 [0, max_speed]。

    只奖励 vz > 0（向上），下落与水平运动记 0 而非惩罚——避免策略因为怕
    「球终将落下」而不敢起球。默认不挂载，保留给后续对比实验。
    """
    ball: Entity = env.scene[asset_name]
    vz = ball.data.root_link_lin_vel_w[:, 2]
    return torch.nan_to_num(vz, nan=0.0).clamp(0.0, max_speed)


def ball_lift_above_start(
    env: ManagerBasedRlEnv,
    asset_name: str = "ball",
    start_height: float = 0.08,
    max_lift: float = 0.5,
) -> torch.Tensor:
    """【垫高版主奖励】球心超过初始垫高高度 `start_height` 的增量，clamp[0, max_lift]。

    为什么基准是 `start_height`（= 基座顶 0.045 + 球半径 0.035 = 0.08）而非地面：
    球静止在基座顶时 z = start_height → lift = 0，**不白给分**；只有真把球挑离
    基座、升得比初始更高，lift 才 > 0。若沿用「离地高度」(z − ball_radius)，球
    一垫高就白拿 0.045 分，策略会退化成「什么都不做也得分」。这是垫高版必须改
    基准的根本原因。

    返回形状 [num_envs]，由 RewardManager 乘权重后累加。
    """
    ball: Entity = env.scene[asset_name]
    z = ball.data.root_link_pos_w[:, 2]
    lift = (z - start_height).clamp(min=0.0)
    return torch.nan_to_num(lift, nan=0.0).clamp(0.0, max_lift)


def ball_drop_penalty(
    env: ManagerBasedRlEnv,
    asset_name: str = "ball",
    drop_z: float = 0.06,
    max_speed: float = 3.0,
) -> torch.Tensor:
    """【垫高版盲区修复】球被向下击落的那一刻扣分（事件型，非状态型）。

    v1 的盲区：`ball_lift = clamp(z − start, min=0)`，把球从桩顶拍落地面
    （z=0.035 < start）得 0，与「啥都不干」同分 → 策略无动力区分「敲落」与
    「不动」。但不能直接改成「状态型负分」（球在地面就一直扣）：那会整回合
    累积到 −70 级、压过站姿的 +18，反而把策略逼成「永远躲球」。

    所以只在 vz<0（向下）且球心已低于 `drop_z`（=已被击落）时，取 −vz 的正值
    作为惩罚幅度（cfg 里乘负权重）。球落地静止后 vz≈0 → 惩罚归零，不累积。

    为什么阈值是 `drop_z`（默认 0.06）而非 start_height（0.08）：球静坐细桩顶
    会因接触下沉/微晃，实测 resting z≈0.075~0.08；若直接用 z<0.08 作阈值，静止时
    也会误触惩罚（假阳性，整回合累积成噪声、冤枉策略）。预留 ~2cm 容差带，只有
    真掉落（z<0.06，向地面 0.035 坠落）才罚。
    返回形状 [num_envs] 的非负幅度，与官方 overshoot 惩罚同约定。
    """
    ball: Entity = env.scene[asset_name]
    vz = ball.data.root_link_lin_vel_w[:, 2]
    z = ball.data.root_link_pos_w[:, 2]
    below = (z < drop_z).to(vz.dtype)           # 真掉落（低于容差带）才罚
    down = (-vz).clamp(min=0.0)                 # 只罚向下速度
    return torch.nan_to_num(down * below, nan=0.0).clamp(0.0, max_speed)
