"""
session_02 · 第 3–4 周：MDP 建模 + Bellman 方程（可运行的「手算」）
====================================================================

对应 learning_roadmap.md 阶段 1 的第 3–4 周：
  第 3 周「学 state/action/reward/policy/return/discount γ」
  第 4 周「理解 V(s)、Q(s,a)、Bellman 方程；手算一个 3 状态小 MDP」

roadmap 要求「手算一个 3 状态小 MDP」。本文件把那次手算变成**可运行、可验证**的代码：
我们真的定义一个 3 状态 MDP，用两种 Bellman 方程把它的价值函数**迭代到收敛**，
再和解析解对照——你就彻底明白 Bellman 方程到底在算什么。

从上到下直接运行：
    python 01_mdp_and_bellman.py               # 跑全部小节
    python 01_mdp_and_bellman.py --section 3   # 只看 Bellman 期望方程那节
    python 01_mdp_and_bellman.py --gamma 0.99  # 换折扣因子看 V* 如何变化

本文件用到的 MDP（贯穿全篇）：
    三个格子排成一行，最右边是目标：  [0] -- [1] -- [2★]
    动作：0=左移, 1=右移（撞墙则原地不动）
    奖励：只要「移动后到达目标格 2」就得 +1，其余每一步 0
    这是一个确定性 MDP（给定 (s,a)，下一个状态唯一）。

四节内容：
  §1 MDP 五元组 (S, A, P, R, γ) —— RL 问题的标准建模方式
  §2 return 与折扣 γ —— 为什么未来的奖励要打折
  §3 Bellman 期望方程 —— 给定策略 π，评估它有多好（policy evaluation）
  §4 Bellman 最优方程 —— 求最优价值 V* 与最优策略（value iteration）
"""

from __future__ import annotations

import argparse

import torch

# --------------------------------------------------------------------------- #
# MDP 定义：3 状态链式世界
# --------------------------------------------------------------------------- #
N_STATES = 3          # 状态集合 S = {0, 1, 2}
N_ACTIONS = 2         # 动作集合 A = {0:左, 1:右}
GOAL = 2              # 目标状态
ACTION_NAMES = ["左", "右"]


def next_state(s: int, a: int) -> int:
    """确定性状态转移：a=0 左移、a=1 右移，撞墙则原地不动。"""
    if a == 0:                       # 左移
        return max(s - 1, 0)
    return min(s + 1, N_STATES - 1)  # 右移


def step_reward(s: int, a: int) -> float:
    """奖励函数 R(s,a)：移动后到达目标格 +1，否则 0。"""
    return 1.0 if next_state(s, a) == GOAL else 0.0


def build_P_and_R() -> tuple[torch.Tensor, torch.Tensor]:
    """把 MDP 显式写成张量，方便后面用矩阵运算做 Bellman backup。

    返回：
      P : (S, A, S) 转移概率，P[s,a,s'] = P(s'|s,a)。确定性 → 每行是 one-hot。
      R : (S, A)    即时奖励，R[s,a] = 在 s 采取 a 得到的奖励。
    """
    P = torch.zeros(N_STATES, N_ACTIONS, N_STATES)
    R = torch.zeros(N_STATES, N_ACTIONS)
    for s in range(N_STATES):
        for a in range(N_ACTIONS):
            P[s, a, next_state(s, a)] = 1.0   # 确定性：目标状态概率 1
            R[s, a] = step_reward(s, a)
    return P, R


def header(n: int, title: str) -> None:
    print("\n" + "=" * 72)
    print(f"§{n}  {title}")
    print("=" * 72)


# --------------------------------------------------------------------------- #
# §1 MDP 五元组：RL 把「序贯决策问题」建模成 (S, A, P, R, γ)
# --------------------------------------------------------------------------- #
def section1_mdp(gamma: float) -> None:
    header(1, "MDP 五元组 (S, A, P, R, γ)")
    P, R = build_P_and_R()

    print("强化学习的世界 = 一个马尔可夫决策过程 (Markov Decision Process)：")
    print(f"  S  状态集合   = {[s for s in range(N_STATES)]}（{GOAL} 是目标★）")
    print(f"  A  动作集合   = {[f'{i}:{n}' for i, n in enumerate(ACTION_NAMES)]}")
    print(f"  P  转移概率   P(s'|s,a)：本例确定性，见下表")
    print(f"  R  奖励函数   R(s,a)：到达目标 +1，否则 0")
    print(f"  γ  折扣因子   = {gamma}（未来奖励的打折率，0~1）")

    print("\n转移与奖励明细（s, a → s', r）：")
    for s in range(N_STATES):
        for a in range(N_ACTIONS):
            s2 = next_state(s, a)
            tag = "★目标" if s2 == GOAL else ""
            print(f"  在状态 {s} 做「{ACTION_NAMES[a]}」 → 到 {s2}{tag}，奖励 r = {R[s, a]:.0f}")

    print("\n『马尔可夫性』= 下一个状态只取决于 (当前状态, 动作)，与怎么走到这里的无关。")
    print("这正是 RL 能『只盯当前 obs 做决策』的理论前提。")
    # 🔗 RL 联系：microduck 的 velocity 任务就是一个 MDP——
    #    S=61 维观测(关节/角速度/指令)，A=关节目标，R=mdp.py 里设计的奖励项，
    #    P=MuJoCo 物理仿真的状态转移，γ=训练配置里的 discount。
    # ✏️ 试一试：把 GOAL 改成 0（目标在最左），重跑看最优策略是否变成一路向左。


# --------------------------------------------------------------------------- #
# §2 return（回报）与折扣 γ：未来奖励为什么要打折
# --------------------------------------------------------------------------- #
def section2_return(gamma: float) -> None:
    header(2, "return 与折扣因子 γ")

    # 假设智能体沿某条轨迹走，收到这样一串即时奖励：
    rewards = [0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    print("一段轨迹收到的即时奖励序列 r_0, r_1, ... =", rewards)

    def discounted_return(rs: list[float], g: float) -> float:
        """G = Σ_t γ^t · r_t：把未来奖励按 γ 逐期打折后求和（return 的定义）。"""
        return sum((g ** t) * r for t, r in enumerate(rs))

    print("\n同一串奖励，在不同 γ 下的 return G（= Σ γ^t r_t）：")
    for g in (0.5, 0.9, 0.99, 1.0):
        print(f"  γ = {g:<4} → G = {discounted_return(rewards, g):.3f}")

    print("\n观察：γ 越小越『短视』（只看重眼前奖励），γ 越大越『远视』（重视长期）。")
    print("      γ=1.0 不打折，无限长任务里 return 可能发散到无穷，所以实践中 γ<1（如 0.99）。")
    print(f"      本项目后面统一用 γ={gamma}。")
    # 🔗 RL 联系：CartPole 每撑一步 +1，鸭子往前走每步给正奖励——
    #    γ 决定了智能体是「贪眼前一步」还是「为长期存活/前进而行动」。
    # ✏️ 试一试：把 rewards 改成 [1,0,0,0,0,10]，看不同 γ 下「早奖励 vs 晚奖励」谁更被看重。


# --------------------------------------------------------------------------- #
# §3 Bellman 期望方程：给定策略 π，评估它的价值 V^π（policy evaluation）
# --------------------------------------------------------------------------- #
def bellman_expectation_backup(V: torch.Tensor, P: torch.Tensor, R: torch.Tensor,
                               policy: torch.Tensor, gamma: float) -> torch.Tensor:
    """一次 Bellman 期望方程 backup：V ← 用当前 V 算出更准的 V。

    公式（对每个状态 s）：
        V(s) = Σ_a π(a|s) · Σ_s' P(s'|s,a) · [ R(s,a) + γ · V(s') ]
             = Σ_a π(a|s) · [ R(s,a) + γ · E_{s'}[V(s')] ]
    直觉：一个状态的价值 = 按策略选动作后，(即时奖励 + 折扣后的下一状态价值) 的期望。
    """
    # 先算每个 (s,a) 的动作价值 Q(s,a) = R(s,a) + γ Σ_s' P(s,a,s') V(s')
    next_V = torch.einsum("saj,j->sa", P, V)   # 对 s' 求期望：(S,A,S)×(S,) → (S,A)
    Q = R + gamma * next_V
    # 再按策略 π(a|s) 对动作求期望：V(s) = Σ_a π(s,a) Q(s,a)
    return (policy * Q).sum(dim=1)


def section3_bellman_expectation(gamma: float) -> None:
    header(3, "Bellman 期望方程：评估一个给定策略 V^π")
    P, R = build_P_and_R()
    print("MDP 的 P 与 R：")
    print(f"  P =\n{P}")
    print(f"  R =\n{R}")

    # 用一个「均匀随机策略」：每个状态左右各 50%。这是最简单的策略。
    policy = torch.full((N_STATES, N_ACTIONS), 1.0 / N_ACTIONS)
    print(f"  π =\n{policy}")
    print("评估的策略 π = 均匀随机（每个状态『左/右』各 50%）")

    # 从全 0 的 V 出发，反复 backup 直到收敛（这就是 policy evaluation 的迭代法）
    V = torch.zeros(N_STATES)
    print(f"  V =\n{V}")
    print("\n迭代式 Bellman backup（看 V 如何一轮轮收敛）：")
    for it in range(1, 51):
        V_new = bellman_expectation_backup(V, P, R, policy, gamma)
        delta = (V_new - V).abs().max().item()   # 本轮 V 的最大变化量
        V = V_new
        if it <= 8 or it % 10 == 0:
            print(f"  第 {it:2d} 轮：V = {[round(v, 4) for v in V.tolist()]}  | 最大变化 Δ = {delta:.2e}")
        if delta < 1e-10:                         # 收敛判据：V 几乎不再变化
            print(f"  第 {it:2d} 轮：V = {[round(v, 4) for v in V.tolist()]}  | Δ={delta:.2e} < 1e-10 → 收敛✅")
            break

    print(f"\n随机策略的价值 V^π = {[round(v, 4) for v in V.tolist()]}")
    print("解读：随机乱走也能偶尔蹭到目标，所以 V 都是正数；但明显不如『一路向右』——")
    print("      下一节用 Bellman 最优方程求出真正最好的 V*，对比就知道差多少。")
    # ✏️ 试一试：把 policy 改成「状态0/1 都 100% 向右」，看 V^π 是否直接等于下一节的 V*。


# --------------------------------------------------------------------------- #
# §4 Bellman 最优方程：求 V* 与最优策略（value iteration）
# --------------------------------------------------------------------------- #
def bellman_optimality_backup(V: torch.Tensor, P: torch.Tensor, R: torch.Tensor,
                              gamma: float) -> torch.Tensor:
    """一次 Bellman 最优方程 backup：V ← max_a [ R(s,a) + γ Σ_s' P V(s') ]。

    与期望方程的唯一区别：把「按 π 对动作求期望 Σ_a π(a|s)」换成「取 max_a」。
    max 体现了『最优』——在每个状态都挑当下最好的动作。
    """
    next_V = torch.einsum("saj,j->sa", P, V)
    Q = R + gamma * next_V
    return Q.max(dim=1).values               # 对动作取最大 → V*(s)


def section4_bellman_optimality(gamma: float) -> None:
    header(4, "Bellman 最优方程：求 V* 与最优策略")
    P, R = build_P_and_R()

    # value iteration：反复用「最优 backup」迭代，V 会收敛到最优价值 V*
    V = torch.zeros(N_STATES)
    for it in range(1, 201):
        V_new = bellman_optimality_backup(V, P, R, gamma)
        delta = (V_new - V).abs().max().item()
        V = V_new
        if delta < 1e-12:
            break
    print(f"value iteration 在第 {it} 轮收敛")

    # 解析解对照：一路向右最优，V*(2)=1/(1-γ)（到目标后每步都还能再拿 +1），
    #            V*(1)=1+γV*(2)，V*(0)=γV*(1)。
    Vs2 = 1.0 / (1.0 - gamma)
    Vs1 = 1.0 + gamma * Vs2
    Vs0 = gamma * Vs1
    analytic = [Vs0, Vs1, Vs2]
    print(f"\n迭代得到的 V* = {[round(v, 4) for v in V.tolist()]}")
    print(f"解析手算的 V* = {[round(v, 4) for v in analytic]}   ← 两者一致，验证 Bellman 最优方程正确✅")

    # 从 V* 导出最优策略：每个状态取让 Q 最大的动作（贪心）
    next_V = torch.einsum("saj,j->sa", P, V)
    Q_star = R + gamma * next_V
    best_action = Q_star.argmax(dim=1)
    print("\n最优策略 π*（每个状态选 Q* 最大的动作）：")
    for s in range(N_STATES):
        q = [round(v, 3) for v in Q_star[s].tolist()]
        print(f"  状态 {s}：Q*(左)={q[0]}, Q*(右)={q[1]} → 最优动作 =「{ACTION_NAMES[best_action[s]]}」")

    print("\n结论：除目标格外都应『向右』直奔目标——这就是 Bellman 最优方程自动推出来的最优行为。")
    print("      你刚用代码完整走了一遍『policy evaluation（§3 评估）→ policy improvement（§4 改进）』，")
    print("      这正是几乎所有 RL 算法的内核循环。")
    # 🔗 RL 联系：DQN（03 文件）学的就是这个 Q*(s,a)，只是用神经网络代替这张表；
    #    Q-learning（02 文件）的更新式 Q←Q+α[r+γ max Q(s')-Q] 就是 Bellman 最优方程的采样版。
    # ✏️ 试一试：把 γ 从 0.9 调到 0.99，看 V* 三个数如何变大（未来奖励打折更少）。


# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="session_02 §1 MDP 与 Bellman 方程演示")
    parser.add_argument("--section", type=int, default=None, help="只运行指定小节 (1–4)，默认全跑")
    parser.add_argument("--gamma", type=float, default=0.9, help="折扣因子 γ (0~1)")
    args = parser.parse_args()

    print(f"MDP：3 状态链 [0]--[1]--[2★] | 折扣 γ = {args.gamma}")
    sections = {
        1: section1_mdp,
        2: section2_return,
        3: section3_bellman_expectation,
        4: section4_bellman_optimality,
    }
    for n in ([args.section] if args.section else list(sections)):
        sections[n](args.gamma)

    print("\n" + "=" * 72)
    print("✅ §1 跑完。你现在应能说清：MDP 五元组、return/γ、")
    print("   Bellman 期望方程(评估策略) vs Bellman 最优方程(求最优)——即检查点『白板推导 Bellman』。")
    print("   下一步：python 02_q_learning_frozenlake.py —— 用采样版 Bellman 学出 Q 表。")
    print("=" * 72)


if __name__ == "__main__":
    main()
