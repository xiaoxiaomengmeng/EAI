"""
session_02 · 第 5 周：表格 Q-learning 打通 FrozenLake（你的第一个「会学习」的 RL 算法）
========================================================================================

对应 learning_roadmap.md 阶段 1 第 5 周：
  「实现 Q-learning 打通 FrozenLake；理解 exploration（ε-greedy）」

01 文件里我们用 Bellman 最优方程「上帝视角」算出了 V*（因为整个 MDP 的 P、R 都已知）。
但真实 RL 里**环境的转移和奖励是未知的**——智能体只能靠「试错」去学。
Q-learning 就是最经典的试错算法：它不预先知道 P/R，而是**一边走一边用采样到的
(s, a, r, s') 去逼近 Bellman 最优方程**，最终学出一张 Q 表。

    python 02_q_learning_frozenlake.py                 # 默认 3000 episode
    python 02_q_learning_frozenlake.py --smoke         # 冒烟：少量 episode 验证流程
    python 02_q_learning_frozenlake.py --episodes 5000 # 训更久，成功率更高

核心更新式（off-policy TD control）——就是 Bellman 最优方程的「采样增量版」：
    Q[s,a] ← Q[s,a] + α · [ r + γ · max_a' Q[s',a'] − Q[s,a] ]
                        └────────── TD target ──────────┘  └当前估计┘
                         └────────────── TD error（时序差分误差）──────────────┘
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import gymnasium as gym

# FrozenLake 的动作含义（4x4 地图固定）：0=左, 1=下, 2=右, 3=上
ACTION_NAMES = ["左", "下", "右", "上"]


def set_seed(seed: int) -> None:
    """固定所有随机源，让结果可复现（RL 调试的第一纪律）。"""
    random.seed(seed)
    np.random.seed(seed)


def epsilon_greedy(Q: np.ndarray, s: int, epsilon: float, nA: int) -> int:
    """ε-greedy 策略：以 ε 概率随机探索，以 1−ε 概率选当前最优动作。

    这是「探索 vs 利用（exploration vs exploitation）」权衡的最经典实现：
      - 利用(exploitation)：选 Q 最大的动作 → 用已知最好的知识拿奖励；
      - 探索(exploration)：随机试别的动作 → 万一有更好或还没发现的路径。
    训练早期 ε 大（多探索），后期 ε 衰减到很小（多利用）。
    """
    if random.random() < epsilon:
        return random.randrange(nA)          # 探索：随机动作
    return int(np.argmax(Q[s]))              # 利用：贪心选 Q 最大的动作


def greedy_eval(env, Q: np.ndarray, n_episodes: int, seed: int) -> float:
    """纯贪心（不探索）评估：跑 n_episodes 局，返回到达目标的成功率。"""
    successes = 0
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        s = int(obs)
        done = False
        while not done:
            a = int(np.argmax(Q[s]))                 # 评估时 ε=0，只利用
            obs, reward, terminated, truncated, _ = env.step(a)
            s = int(obs)
            done = terminated or truncated
            if terminated and reward == 1.0:         # FrozenLake 到达目标才给 +1 且 terminated
                successes += 1
    return successes / n_episodes


def train(args) -> tuple[np.ndarray, list[float]]:
    """Q-learning 训练主循环。返回 (学到的 Q 表, 评估成功率历史)。"""
    # is_slippery=False → 关掉冰面打滑，转移动态确定，表格法能可靠收敛（利于聚焦算法本身）
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False)
    nS = env.observation_space.n                     # 状态数：4x4 = 16
    nA = env.action_space.n                          # 动作数：4
    print(f"环境 FrozenLake-v1(4x4, 不打滑) | 状态数 nS={nS} | 动作数 nA={nA}")
    print(f"超参：episodes={args.episodes} α(学习率)={args.alpha} γ(折扣)={args.gamma} "
          f"ε:{args.epsilon_start}→{args.epsilon_end} 衰减率={args.epsilon_decay}")

    Q = np.zeros((nS, nA))                           # Q 表：每个 (状态,动作) 的价值，初始全 0
    eval_every = max(1, args.episodes // 10)         # 每 10% 训练进度评估一次
    eval_history: list[float] = []
    epsilon = args.epsilon_start

    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset(seed=args.seed + ep)
        s = int(obs)
        done = False
        ep_reward = 0.0
        while not done:
            a = epsilon_greedy(Q, s, epsilon, nA)    # 用 ε-greedy 选动作
            obs, reward, terminated, truncated, _ = env.step(a)
            s_next = int(obs)
            done = terminated or truncated

            # ── Q-learning 核心更新（Bellman 最优方程的采样增量版）──
            best_next = np.max(Q[s_next])            # max_a' Q[s',a']：下一状态的最优价值
            # 若 episode 结束(terminated)，下一状态无未来价值，bootstrap 项置 0
            td_target = reward + (0.0 if terminated else args.gamma * best_next)
            td_error = td_target - Q[s, a]           # 预测与目标的差
            Q[s, a] += args.alpha * td_error         # 朝目标迈一小步（α 控制步长）

            s = s_next
            ep_reward += reward

        # ε 指数衰减：越训越少探索（floor 在 epsilon_end）
        epsilon = max(args.epsilon_end, epsilon * args.epsilon_decay)

        if ep % eval_every == 0 or ep == args.episodes:
            sr = greedy_eval(env, Q, args.eval_episodes, seed=9000)
            eval_history.append(sr)
            print(f"  episode {ep:5d}/{args.episodes} | ε={epsilon:.3f} | "
                  f"贪心成功率={sr*100:5.1f}% (共{args.eval_episodes}局)")

    env.close()
    return Q, eval_history


def print_policy(Q: np.ndarray) -> None:
    """把学到的 Q 表打印成一张「每个格子该往哪走」的策略地图，直观看结果。"""
    arrows = ["←", "↓", "→", "↑"]
    n = int(round(Q.shape[0] ** 0.5))                # 4x4 → n=4
    print(f"\n学到的最优策略（{n}x{n} 地图，箭头=该状态贪心动作；S=起点 G=目标 H=冰窟）：")
    for r in range(n):
        row = []
        for c in range(n):
            s = r * n + c
            if s == 0:
                row.append("S")
            elif s == n * n - 1:
                row.append("G")
            else:
                row.append(arrows[int(np.argmax(Q[s]))])
        print("   " + " ".join(row))


def main() -> None:
    ap = argparse.ArgumentParser(description="session_02 表格 Q-learning × FrozenLake")
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--alpha", type=float, default=0.5, help="学习率 α")
    ap.add_argument("--gamma", type=float, default=0.99, help="折扣因子 γ")
    ap.add_argument("--epsilon-start", type=float, default=1.0)
    ap.add_argument("--epsilon-end", type=float, default=0.01)
    ap.add_argument("--epsilon-decay", type=float, default=0.999, help="每 episode ε 乘以此值衰减")
    ap.add_argument("--eval-episodes", type=int, default=100, help="每次评估跑多少局")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true", help="冒烟：少量 episode 验证流程")
    ap.add_argument("--out", default=str(Path(__file__).parent / "qtable_frozenlake.npy"))
    args = ap.parse_args()

    if args.smoke:
        args.episodes = 300
        args.eval_episodes = 20

    set_seed(args.seed)
    Q, hist = train(args)

    print("\n" + "=" * 72)
    final_sr = hist[-1] if hist else 0.0
    print(f"训练结束，最终贪心成功率 = {final_sr*100:.1f}%")
    print_policy(Q)

    if args.smoke:
        print("\nℹ️  这是冒烟测试（少量 episode），只验证流程；跑完整训练看真实成功率：")
        print("     python 02_q_learning_frozenlake.py --episodes 3000")
    elif final_sr >= 0.9:
        np.save(args.out, Q)
        print(f"\n✅ 检查点达成：FrozenLake 贪心成功率 ≥ 90%（Q 表已存 {args.out}）")
        print("   你已实现第一个「会学习」的 RL 算法。下一步：03_dqn_cartpole.py")
        print("   —— 当状态多到无法用表格存（如 CartPole 的连续状态），就该请神经网络登场了。")
    else:
        print("\n❌ 成功率还没到 90%。多为训练不足或探索衰减太快：")
        print("     加大 --episodes 到 5000，或调 --epsilon-decay 到 0.9995（探索更久），或 --alpha 0.8")
    print("=" * 72)


if __name__ == "__main__":
    main()
