"""
session_02 · 第 6–8 周（主产出 / 阶段 1 硬检查点）：从零实现 DQN 打通 CartPole-v1
==================================================================================

对应 learning_roadmap.md 阶段 1 第 6–8 周：
  第 6 周「学 DQN 原理（experience replay、target network）」
  第 7 周「在 CartPole-v1 上从零实现 DQN，调通训练循环」
  第 8 周「调参让 CartPole 稳定撑到 500 步」

为什么需要 DQN？—— 承接 02 文件
  FrozenLake 只有 16 个离散状态，一张 16×4 的 Q 表就装得下。
  但 CartPole 的状态是【连续】的 4 维向量（小车位置/速度、杆角度/角速度），
  状态有无穷多个，表格根本存不下。DQN 的核心思想就是：
      用一个神经网络 Q(s,a;θ) 去【逼近】那张存不下的 Q 表。

DQN 相比「朴素地拿神经网络套 Q-learning」多两件关键武器（本文件都实现，且可用开关关掉做对照）：
  ① Experience Replay（经验回放）：把经历存进 buffer，训练时随机抽样。
     - 为什么：相邻样本高度相关，直接拿去训练会破坏 SGD 的 i.i.d. 假设 → 震荡/发散；
       随机抽样打破相关性，还能让一条经验被反复利用（提高样本效率）。
  ② Target Network（目标网络）：算 TD 目标时用一份「冻结的旧网络」，每隔若干步才同步。
     - 为什么：TD 目标 r+γ·maxQ(s') 里也含网络参数，若用正在更新的同一网络，
       就成了「追一个自己在动的靶子」→ 自举震荡。冻结目标 = 把靶子固定一阵子。

    python 03_dqn_cartpole.py --smoke                 # 冒烟：极少步数验证流程通不通
    python 03_dqn_cartpole.py                         # 正式训练（默认 12 万步，冲 500）
    python 03_dqn_cartpole.py --no-replay             # 消融：关掉经验回放，看它多不稳
    python 03_dqn_cartpole.py --no-target-net         # 消融：关掉目标网络，看自举震荡

训练曲线会存到 dqn_cartpole_curve.csv；评估回报刷新最高时存 dqn_cartpole_best.pt。
"""

from __future__ import annotations

import argparse
import csv
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym


# --------------------------------------------------------------------------- #
# 设备选择（与 session_01 一致：cuda / Apple mps / cpu）
# --------------------------------------------------------------------------- #
def get_device(pref: str) -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# --------------------------------------------------------------------------- #
# Q 网络：输入 4 维状态，输出 2 个动作各自的 Q 值
# --------------------------------------------------------------------------- #
class QNetwork(nn.Module):
    """MLP 4 → 128 → 128 → 2。输出 Q(s,·)，即每个动作的估计价值。

    注意：输出是 Q 值（实数，可正可负），不是概率，所以末尾【不加 softmax】——
    这跟 session_01 分类网络不同：那里要预测类别概率，这里要回归「价值」。
    """

    def __init__(self, obs_dim: int = 4, hidden: int = 128, n_actions: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# --------------------------------------------------------------------------- #
# 经验回放池：存历史 transition，训练时随机抽样
# --------------------------------------------------------------------------- #
class ReplayBuffer:
    """用 deque 实现的固定容量经验池。每条经验 = (s, a, r, s_next, terminated)。

    随机抽样打破样本间的时间相关性（见文件头「为什么 replay」）。
    """

    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)         # 满了自动丢最旧的

    def push(self, s, a, r, s_next, terminated) -> None:
        self.buf.append((s, a, r, s_next, terminated))

    def sample(self, batch_size: int, device: torch.device):
        batch = random.sample(self.buf, batch_size)          # 随机抽 batch_size 条
        s, a, r, s_next, term = zip(*batch)                  # 解包成 5 个元组
        # 转成张量并搬到 device；注意 dtype：状态/奖励 float，动作 int（gather 要用）
        return (torch.tensor(np.array(s), dtype=torch.float32, device=device),
                torch.tensor(a, dtype=torch.int64, device=device),
                torch.tensor(r, dtype=torch.float32, device=device),
                torch.tensor(np.array(s_next), dtype=torch.float32, device=device),
                torch.tensor(term, dtype=torch.float32, device=device))

    def __len__(self) -> int:
        return len(self.buf)


# --------------------------------------------------------------------------- #
# ε 线性衰减调度：前 exploration_fraction 的训练步内，ε 从 start 降到 end
# --------------------------------------------------------------------------- #
def epsilon_at(step: int, total: int, frac: float, start: float, end: float) -> float:
    decay_steps = max(1, int(total * frac))
    if step >= decay_steps:
        return end
    return start + (end - start) * (step / decay_steps)


# --------------------------------------------------------------------------- #
# 评估：纯贪心（不探索）跑若干局，返回平均回报（CartPole 每步 +1，最高 500）
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(q: QNetwork, device: torch.device, n_episodes: int, seed: int) -> float:
    env = gym.make("CartPole-v1")
    q.eval()                                       # 评估模式（本网络无 BN/dropout，但养成习惯）
    returns = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        total, done = 0.0, False
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            action = int(q(obs_t).argmax(dim=1).item())   # 贪心：选 Q 最大的动作
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
        returns.append(total)
    q.train()
    env.close()
    return float(np.mean(returns))


# --------------------------------------------------------------------------- #
# 单步 DQN 更新（四步循环，复用 session_01 的骨架）
# --------------------------------------------------------------------------- #
def dqn_update(q_online, q_target, optimizer, buffer, batch, args, device,
               no_replay: bool) -> float:
    """执行一次梯度更新，返回 loss（标量 float）。

    batch 非 None 时用传入的单条 transition（消融 --no-replay 场景）；
    否则从 buffer 随机抽样。
    """
    if no_replay:
        s, a, r, s_next, term = batch                # 单条经验，各自已是 (1,) 张量
    else:
        s, a, r, s_next, term = buffer.sample(args.batch_size, device)

    # ── 算 TD 目标（不需要梯度）──
    with torch.no_grad():
        # 消融：--no-target-net 时用「正在训练的在线网络」自己算目标（错误的自举靶子）
        target_net = q_online if args.no_target_net else q_target
        next_q = target_net(s_next).max(dim=1).values          # max_a' Q(s',a')
        # terminated=1 时是真正终态，无未来价值；truncated（超时）不是终态，仍要 bootstrap
        td_target = r + args.gamma * next_q * (1.0 - term)

    # ── 算当前预测 Q(s,a) ──
    # gather：从 2 个动作的 Q 里挑出「实际采取的那个动作 a」对应的 Q 值
    pred_q = q_online(s).gather(1, a.unsqueeze(1)).squeeze(1)

    # ── 四步循环：清零 → 算 loss → 反传 → 更新 ──
    loss = F.mse_loss(pred_q, td_target)             # 让预测 Q 逼近 TD 目标（回归）
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.item())


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="session_02 从零 DQN × CartPole-v1")
    ap.add_argument("--total-timesteps", type=int, default=120000, help="总环境交互步数")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--buffer-size", type=int, default=50000)
    ap.add_argument("--learning-starts", type=int, default=2000, help="先随机攒够这么多步再开始训练")
    ap.add_argument("--train-freq", type=int, default=1, help="每隔几步环境交互做一次梯度更新")
    ap.add_argument("--target-update-interval", type=int, default=500, help="每隔几步硬同步目标网络")
    ap.add_argument("--epsilon-start", type=float, default=1.0)
    ap.add_argument("--epsilon-end", type=float, default=0.05)
    ap.add_argument("--exploration-fraction", type=float, default=0.4, help="前多少比例步数内衰减完 ε")
    ap.add_argument("--eval-interval", type=int, default=5000, help="每隔多少步评估一次")
    ap.add_argument("--eval-episodes", type=int, default=20)
    ap.add_argument("--device", default="auto", help="auto / cpu / cuda / mps")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--smoke", action="store_true", help="冒烟：极少步数验证流程")
    ap.add_argument("--no-replay", action="store_true", help="消融：关掉经验回放")
    ap.add_argument("--no-target-net", action="store_true", help="消融：关掉目标网络")
    ap.add_argument("--out", default=str(Path(__file__).parent / "dqn_cartpole_best.pt"))
    ap.add_argument("--curve", default=str(Path(__file__).parent / "dqn_cartpole_curve.csv"))
    args = ap.parse_args()

    if args.smoke:
        args.total_timesteps = 2000
        args.learning_starts = 200
        args.eval_interval = 500
        args.eval_episodes = 3
        args.target_update_interval = 100

    device = get_device(args.device)
    set_seed(args.seed)
    tag = []
    if args.no_replay:
        tag.append("NO-REPLAY")
    if args.no_target_net:
        tag.append("NO-TARGET-NET")
    tagstr = (" [" + ",".join(tag) + "]") if tag else ""
    print(f"PyTorch {torch.__version__} | 设备：{device}{tagstr} | 总步数={args.total_timesteps}")

    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]         # 4
    n_actions = env.action_space.n                   # 2

    q_online = QNetwork(obs_dim, 128, n_actions).to(device)     # 在线网络（实时训练）
    q_target = QNetwork(obs_dim, 128, n_actions).to(device)     # 目标网络（周期性冻结副本）
    q_target.load_state_dict(q_online.state_dict())             # 初始与在线网络一致
    q_target.eval()                                             # 目标网络只做前向，不训练
    optimizer = torch.optim.Adam(q_online.parameters(), lr=args.lr)
    buffer = ReplayBuffer(args.buffer_size)

    # ── 主循环：与环境交互收集经验 + 训练 ──
    obs, _ = env.reset(seed=args.seed)
    episode_return, episode_cnt = 0.0, 0
    best_eval, t0 = 0.0, time.time()
    curve_rows = []

    for step in range(1, args.total_timesteps + 1):
        epsilon = epsilon_at(step, args.total_timesteps, args.exploration_fraction,
                             args.epsilon_start, args.epsilon_end)

        # ε-greedy 选动作（探索）
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        if random.random() < epsilon:
            action = env.action_space.sample()                  # 探索：随机
        else:
            with torch.no_grad():
                action = int(q_online(obs_t).argmax(dim=1).item())  # 利用：贪心

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        # 存经验：注意存的是 terminated（真终态），不是 done（含超时 truncated）——
        # 超时截断并非真终态，TD 目标仍应对它 bootstrap，这是初学者极易搞错的点。
        single = None
        if args.no_replay:
            single = (obs_t,
                      torch.tensor([action], dtype=torch.int64, device=device),
                      torch.tensor([reward], dtype=torch.float32, device=device),
                      torch.tensor(np.array(next_obs), dtype=torch.float32, device=device).unsqueeze(0),
                      torch.tensor([float(terminated)], dtype=torch.float32, device=device))
        else:
            buffer.push(obs, action, reward, next_obs, float(terminated))

        # 攒够经验后按 train_freq 更新
        can_train = (len(buffer) >= args.learning_starts) if not args.no_replay \
            else (step >= args.learning_starts)
        if can_train and step % args.train_freq == 0:
            loss = dqn_update(q_online, q_target, optimizer, buffer, single, args, device,
                              args.no_replay)

        # 周期性硬同步目标网络（把在线网络的权重拷给目标网络）
        if step % args.target_update_interval == 0:
            q_target.load_state_dict(q_online.state_dict())

        obs = next_obs
        episode_return += reward
        if done:
            episode_cnt += 1
            obs, _ = env.reset()
            episode_return = 0.0

        # 周期性评估 + 存最佳 checkpoint + 记曲线
        if step % args.eval_interval == 0 or step == args.total_timesteps:
            ev = evaluate(q_online, device, args.eval_episodes, seed=9000)
            cur_loss = loss if can_train else float("nan")
            print(f"  step {step:7d}/{args.total_timesteps} | ε={epsilon:.3f} | "
                  f"评估平均回报={ev:6.1f} | loss={cur_loss:.4f} | {time.time()-t0:5.1f}s")
            curve_rows.append({"step": step, "epsilon": round(epsilon, 4),
                               "eval_return": round(ev, 2), "loss": round(cur_loss, 5)})
            if ev > best_eval:
                best_eval = ev
                torch.save({"model_state": q_online.state_dict(),
                            "arch": f"MLP-{obs_dim}-128-128-{n_actions}",
                            "eval_return": ev, "step": step}, args.out)

    env.close()
    with open(args.curve, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["step", "epsilon", "eval_return", "loss"])
        w.writeheader()
        w.writerows(curve_rows)

    print("\n" + "=" * 72)
    print(f"训练完成，用时 {time.time()-t0:.1f}s | 最高评估回报 = {best_eval:.1f} / 500")
    print(f"最佳权重已存 {args.out}；训练曲线已存 {args.curve}")
    if args.smoke:
        print("ℹ️  冒烟测试只验证流程（replay/target/训练循环不报错），不追求回报。")
        print("    正式训练：python 03_dqn_cartpole.py")
    elif tag:
        print(f"⚠️  这是消融实验{tagstr}：对比默认版的回报曲线，体会被关掉的组件为何重要。")
    elif best_eval >= 475:
        print("✅ 阶段 1 硬检查点达成：DQN 让 CartPole-v1 稳定撑到 ~500 步（≥475 即视为 solved）。")
    else:
        print("❌ 还没稳定到 475+。可加大 --total-timesteps 到 200000，或换 --seed 重试（DQN 对随机种子敏感）。")
    print("=" * 72)


if __name__ == "__main__":
    main()
