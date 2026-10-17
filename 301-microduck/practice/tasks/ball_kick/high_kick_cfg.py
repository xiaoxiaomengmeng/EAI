"""高踢 env cfg —— 官方 ball_kick 配方之上覆盖「场景 + reset + reward」（垫高版）。

核心手法（工程点）：调用官方 `make_microduck_ball_kick_env_cfg()` 拿到那份
662 行的完整配方（场景 / 传感器 / 域随机化 / curriculum / 非对称 actor-critic
全部现成），然后只覆盖三处：entities（加垫高基座）、events（球摆到基座顶）、
rewards（换成挑高目标）。cfg 是可变 dataclass，逐项增删改即可，不改子模块。

为什么要垫高（贴地版实测 ball_height≈0 的根因）：microduck 的脚是扁平 sole，
球贴地时球底与地面零缝隙，脚插不进球下方 → 无法「从下往上铲」，只能水平推球
滚动。把球垫到桩顶（球心离地 0.08m），脚才能做「斜向上挑」。

为什么支撑是「细桩 tee」而非实体 box（v1 踩坑实录）：v1 用 3cm 见方实体 box 垫球，
300 iter 训练后 rollout 实测球 z 硬卡在起始高 0.08、一次都没被挑高，只会被脚
拍落下桩（drop 181/4800、lift 0/4800）——实体 box 顶面堵死球正下方，和贴地版被
地板挡住是同一个几何陷阱。v2 改用半径 0.010m 的细圆柱桩，球赤道下方空出 ~18mm
的槽让 toe 扫进去，挑高才几何可达。详见 pedestal.xml 注释。

与官方「轻触版」的差异：
  1. 场景加 `pedestal`（5kg 重块细圆柱桩 + freejoint，几乎不动，只作球座）；
  2. reset 换成 `reset_ball_on_pedestal`：细桩摆到右脚正前方、球同心摆到桩顶；
  3. 删除 `ball_speed_overshoot`（−4.0，压制猛踢）与 `ball_forward_velocity`
     （12.0，主奖励却是「滚地球」歧途的元凶）；
  4. 新增 `ball_lift`（主奖励，基准=桩顶初始高 0.08，不白给分）
     + `ball_vertical_velocity`（辅奖励，只奖向上分量，作稠密 bootstrap）。

v3→v4「运动员式猛踢」重平衡（300 iter 实测 ball_lift 仍≈ 0 后）：
  v3 曾加过事件型 `ball_drop`（只罚向下击落）想堵「拍落球与不动同得 0 分」
  的盲区，但 v1/v2/v3 三版 300 iter 下来 ball_lift 始终贴 0（峰值≤ 0.5mm），
  而 mean_reward 早饱和——证明病根不是几何也不是 iter，而是站姿奖励（~20）
  压倒一切、且 fell_over 终止 + upright 把冒险挥腿的期望收益压成负，策略理性
  地选「躲球站着刷分」。用户定调「踢高是唯一硬目标、摔倒其次」后，v4 照
  官方 roulade 任务的同款先例重平衡：ball_lift 权重 8→30 压倒站姿；删 upright
  与 fell_over 终止（允许前倾/摔倒）；support/pose/height 站姿项降为软偏好（×~0.25）；
  action_rate_l2 改轻且删 curriculum ramp；删 ball_drop（接触自由化）。详见
  make_high_kick_env_cfg 末尾的重平衡块。
"""

from __future__ import annotations

import os
import sys
from copy import deepcopy

import mujoco
import torch

# 让本文件无论从哪个 cwd 被 import，都能找到同目录的 high_kick_rewards。
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import high_kick_rewards as hk  # noqa: E402

from mjlab.entity import Entity, EntityCfg  # noqa: E402
from mjlab.managers import EventTermCfg, RewardTermCfg  # noqa: E402

from mjlab_microduck.tasks import mdp as microduck_mdp  # noqa: E402
from mjlab_microduck.tasks.microduck_ball_kick_env_cfg import (  # noqa: E402
    BALL_OFFSET_ABS_Y,
    BALL_OFFSET_X,
    BALL_POS_NOISE_XY,
    BALL_RADIUS,
    MicroduckBallKickRlCfg,
    make_microduck_ball_kick_env_cfg,
)

# ── 任务常量 ─────────────────────────────────────────────────────────────────
KICK_FOOT = "right"          # 复用官方右脚踢；改 "left" 即镜像

PEDESTAL_HEIGHT = 0.045      # 细桩顶面离地高度（圆柱半高 0.0225，见 pedestal.xml）
BALL_START_Z = PEDESTAL_HEIGHT + BALL_RADIUS   # 球心初始高度 = 0.045 + 0.035 = 0.08

# ── reward 权重（v4「运动员式猛踢」重平衡）────────────────────────────────────
# 用户定调：把球踢得更高是唯一硬目标；摔倒其次（可接受）；站稳只是软偏好，
# 绝不能压过踢高。故 v4 把 lift 抬成主奖励、站姿降权、删掉一切对抗猛踢的约束。
# 先例：官方 roulade 任务为让翻滚不被姿态压力对抗，同样 `del upright` + `del fell_over`。
W_BALL_LIFT = 30.0           # 主奖励：球心超过初始垫高高的增量，压倒站姿
W_BALL_VERT_VEL = 10.0       # 辅奖励：球向上的速度分量（只奖 vz>0，稠密 bootstrap）
MAX_LIFT = 0.5               # lift 天花板（m）；microduck 仅 25cm，0.5m 已很高
VERT_VEL_MAX_SPEED = 3.0     # 向上速度 clamp 上界（m/s）

# 站姿软偏好（原官方值 → v4 值，约 ×0.25）：仍能加分，但压不过踢高。
W_SUPPORT_FOOT = 0.5         # support_foot_grounded 原 2.0
W_POSE_LEGS = 0.5            # pose_stand_legs 原 2.0
W_POSE_NECK = 0.3            # pose_stand_neck 原 1.0
W_HEIGHT_STAND = 0.5         # height_stand 原 1.0
# 放开动作/角速度约束，允许爆发性挥腿（upright 直接删、fell_over 终止直接删）。
W_ACTION_RATE = -0.05        # action_rate_l2 原 -0.1（且 curriculum 会 ramp 到 -1.0，v4 删 ramp）
W_BODY_ANG_VEL = -0.01       # body_ang_vel 原 -0.05
W_ANG_MOMENTUM = -0.005      # angular_momentum 原 -0.02

# pedestal.xml 与本文件同目录。
_PEDESTAL_XML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pedestal.xml")


def get_pedestal_spec() -> mujoco.MjSpec:
    """加载垫高基座 MJCF（复刻官方 get_ball_spec 的 from_file 写法）。"""
    return mujoco.MjSpec.from_file(_PEDESTAL_XML)


# 复用 ball 已验证可行的 dynamic entity 机制：freejoint 重块，位置由 reset event
# 每回合写；init pos 只影响首次 reset 前的 pristine 状态。
PEDESTAL_CFG = EntityCfg(
    spec_fn=get_pedestal_spec,
    init_state=EntityCfg.InitialStateCfg(pos=(0.3, 0.0, PEDESTAL_HEIGHT * 0.5)),
)


def reset_ball_on_pedestal(
    env,
    env_ids: torch.Tensor,
    offset: tuple = (BALL_OFFSET_X, -BALL_OFFSET_ABS_Y),
    noise_xy: float = BALL_POS_NOISE_XY,
    ball_radius: float = BALL_RADIUS,
    pedestal_height: float = PEDESTAL_HEIGHT,
    asset_name: str = "ball",
    pedestal_name: str = "pedestal",
):
    """把基座摆到右脚正前方、球摆到基座顶（垫高版的 reset_ball）。

    复刻官方 `reset_ball_in_front_of_foot` 的 yaw-frame 逻辑（从 qpos 直接读机器人
    根，因为 root_link_pos_w 要下一次 forward() 才刷新），差异只有两点：
      1. 额外摆一个 pedestal（box 中心在 origin_z + 半高，无 noise，与球同心）；
      2. 球的 z 从「贴地 origin_z + r」抬到「基座顶 origin_z + pedestal_height + r」。
    球的 XY 仍带 noise（策略对球盲，这是逼出鲁棒挥腿的 DR），基座跟随球的 XY，
    保证球稳稳落在基座顶。

    必须注册在 reset_base / set_ground_state 之后（events 按 dict 插入顺序执行），
    这样机器人根位姿已定稿。kick_dir 一并写入，与官方保持一致。
    """
    if env_ids is None or len(env_ids) == 0:
        return
    env_ids = env_ids.to(env.device)
    robot: Entity = env.scene["robot"]
    ball: Entity = env.scene[asset_name]
    pedestal: Entity = env.scene[pedestal_name]

    root = env.sim.data.qpos[env_ids][:, robot.indexing.free_joint_q_adr]
    qw, qx, qy, qz = root[:, 3], root[:, 4], root[:, 5], root[:, 6]
    yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)

    n = len(env_ids)
    off = torch.tensor(offset, device=env.device, dtype=torch.float).repeat(n, 1)
    off += (torch.rand(n, 2, device=env.device) * 2.0 - 1.0) * noise_xy
    x = root[:, 0] + cos_y * off[:, 0] - sin_y * off[:, 1]
    y = root[:, 1] + sin_y * off[:, 0] + cos_y * off[:, 1]
    origin_z = env.scene.terrain.env_origins[env_ids, 2]
    zeros6 = torch.zeros(n, 6, device=env.device)

    # 基座：box 中心在半高处，底面贴地、顶面到 origin_z + pedestal_height。
    ped_pose = torch.zeros(n, 7, device=env.device)
    ped_pose[:, 0] = x
    ped_pose[:, 1] = y
    ped_pose[:, 2] = origin_z + pedestal_height * 0.5
    ped_pose[:, 3] = 1.0  # identity quat
    pedestal.write_root_link_pose_to_sim(ped_pose, env_ids)
    pedestal.write_root_link_velocity_to_sim(zeros6, env_ids)

    # 球：坐在基座顶。
    ball_pose = torch.zeros(n, 7, device=env.device)
    ball_pose[:, 0] = x
    ball_pose[:, 1] = y
    ball_pose[:, 2] = origin_z + pedestal_height + ball_radius
    ball_pose[:, 3] = 1.0  # identity quat
    ball.write_root_link_pose_to_sim(ball_pose, env_ids)
    ball.write_root_link_velocity_to_sim(zeros6, env_ids)

    # 与官方一致：冻结本回合的踢腿方向（世界系 XY 单位向量）。
    kick_dir = microduck_mdp._ball_kick_dir(env)
    kick_dir[env_ids, 0] = cos_y
    kick_dir[env_ids, 1] = sin_y


def make_high_kick_env_cfg(play: bool = False, kick_foot: str = KICK_FOOT):
    """造垫高高踢 env cfg：官方 ball_kick cfg 之上覆盖 entities / events / rewards。"""
    cfg = make_microduck_ball_kick_env_cfg(play=play, kick_foot=kick_foot)

    # 1) 场景加垫高基座。robot 必须保持第一个 entity（set_random_ground_state /
    #    reset 事件按 qpos[:, 0:7] 写机器人根），dict 追加不改变已有顺序。
    cfg.scene.entities["pedestal"] = PEDESTAL_CFG

    # 2) 覆盖 reset_ball：改成把球摆到基座顶。沿用官方算好的右脚 offset（含镜像）。
    ball_offset_y = -BALL_OFFSET_ABS_Y if kick_foot == "right" else BALL_OFFSET_ABS_Y
    cfg.events["reset_ball"] = EventTermCfg(
        func=reset_ball_on_pedestal,
        mode="reset",
        params={
            "offset":          (BALL_OFFSET_X, ball_offset_y),
            "noise_xy":        BALL_POS_NOISE_XY,
            "ball_radius":     BALL_RADIUS,
            "pedestal_height": PEDESTAL_HEIGHT,
            "asset_name":      "ball",
            "pedestal_name":   "pedestal",
        },
    )

    # 3) reward：删掉「滚地球」的两个官方项，换成挑高目标。
    cfg.rewards.pop("ball_speed_overshoot", None)    # −4.0，压制猛踢，与踢高冲突
    cfg.rewards.pop("ball_forward_velocity", None)   # 12.0，主动引导水平推滚，是元凶

    # 主奖励：球心超过初始垫高高的增量（基准 BALL_START_Z，静止在基座顶时 = 0，不白给）。
    cfg.rewards["ball_lift"] = RewardTermCfg(
        func=hk.ball_lift_above_start,
        weight=W_BALL_LIFT,
        params={
            "asset_name":   "ball",
            "start_height": BALL_START_Z,
            "max_lift":     MAX_LIFT,
        },
    )

    # 辅奖励：球向上的速度分量（只奖 vz>0，起球瞬间给梯度，是最早的稠密 bootstrap）。
    cfg.rewards["ball_vertical_velocity"] = RewardTermCfg(
        func=hk.ball_vertical_velocity,
        weight=W_BALL_VERT_VEL,
        params={
            "asset_name": "ball",
            "max_speed":  VERT_VEL_MAX_SPEED,
        },
    )

    # 盲区修复：把球向下击落那一刻扣分（事件型，球落地静止后不累积）。
    # v4 删除：接触自由化——不再罚「碰球」，方向完全交给 lift / vertvel 塑造，
    # 避免惩罚把策略逼成「躲球站着刷分」（v3 实测的失败模式）。

    # ── v4「运动员式猛踢」重平衡（踢高为唯一硬目标，摔倒其次，站稳软偏好）──
    # 删掉一切对抗猛踢/前倾的强约束，站姿降为软偏好。
    cfg.rewards.pop("upright", None)   # 高斯强力对抗前倾摆腿，删（roulade 同款先例）
    cfg.rewards["support_foot_grounded"].weight = W_SUPPORT_FOOT
    cfg.rewards["pose_stand_legs"].weight = W_POSE_LEGS
    cfg.rewards["pose_stand_neck"].weight = W_POSE_NECK
    cfg.rewards["height_stand"].weight = W_HEIGHT_STAND
    # 放开动作/角速度约束，允许爆发性挥腿。
    cfg.rewards["action_rate_l2"].weight = W_ACTION_RATE
    cfg.rewards["body_ang_vel"].weight = W_BODY_ANG_VEL
    cfg.rewards["angular_momentum"].weight = W_ANG_MOMENTUM
    # 摔倒不再中断回合（用户：摔倒是其次）。躺平拿不到 lift＝隐式惩罚，无需显式项。
    cfg.terminations.pop("fell_over", None)
    # 删 action_rate 的 curriculum ramp（-0.1→-1.0），否则长跑会把猛踢重新压死。
    cfg.curriculum.pop("action_rate_weight", None)

    return cfg


# ── RL runner cfg：deepcopy 官方 PPO 超参，只改 experiment/run/迭代数 ─────────
# experiment_name 决定日志目录 logs/rsl_rl/<experiment_name>/，务必与贴地版区分。
# max_iterations 默认给小规模验证值，可被 CLI --agent.max-iterations 覆盖。
HIGH_KICK_RL_CFG = deepcopy(MicroduckBallKickRlCfg)
HIGH_KICK_RL_CFG.experiment_name = f"ball_kick_aggr_{KICK_FOOT}"
HIGH_KICK_RL_CFG.run_name = f"ball_kick_aggr_{KICK_FOOT}"
HIGH_KICK_RL_CFG.max_iterations = 300
