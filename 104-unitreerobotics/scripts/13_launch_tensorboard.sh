#!/usr/bin/env bash
# 13_launch_tensorboard.sh —— 一键启动训练曲线 dashboard（TensorBoard）
#
# 【这是什么】
#   训练时 logger=tensorboard，事件文件落在 repos/unitree_rl_mjlab/logs/rsl_rl/<task>/<run>/。
#   本脚本把它起成本地 web 服务，浏览器交互看全部 36 个 scalar tag。
#   与 07 脚本分工：07 = 一键出带标注的结论图（PNG 落盘）；本脚本 = 交互探索、自己截图。
#
# 【用法】
#   ./scripts/13_launch_tensorboard.sh          # 默认端口 6006
#   ./scripts/13_launch_tensorboard.sh 6007     # 端口被占时换一个
#   启动后浏览器打开 http://localhost:<端口>（Ctrl+C 停止）
#
# 【怎么看（操作路径）】
#   1. 左上角 runs 列表 = 勾选要看的 run，多选可叠加对比：
#        g1_velocity/2026-10-19_17-23-46   正式 velocity run（口播稿主角）
#        g1_tracking/2026-10-19_20-01-54   正式 tracking run（跳舞）
#        g1_velocity/2026-10-19_17-19-02   废弃 smoke run（只到 model_4，忽略）
#   2. SCALAR 面板 = 标量曲线；勾「Show data download links」可导出 CSV
#   3. Smoothing 滑块：0 = 原始锯齿，0.95 = 只看趋势；截图建议 0.6~0.9
#   4. 塌陷段放大：鼠标框选 x 轴 4500~6500 区域即可局部放大
#
# 【看哪些窗口 / 关键参数（velocity 正式 run 实测口径）】
#   ★ Train/mean_reward
#       曲线故事：峰值 39.37 @ 4965 → 5000~6000 塌陷（谷底 21.41，跌 18 分）
#                 → 后半程缓慢恢复，终值 27.45（仅峰值的 70%）
#       判读：断崖下跌 + 部分恢复 = 策略震荡失稳（该任务无 curriculum，训练范围全程固定）；
#             缓慢台阶下行 = 课程切换。两种成因应对完全相反，先查配置再下结论。
#   ★ Train/error_vel_xy
#       速度跟踪误差：塌陷同窗口 0.54 → 0.94 m/s（恶化 75%）——reward 跌的根因在这
#   ★ Train/error_vel_yaw
#       转向误差：终态 ~1.42 rad/s，贴合度一般（play 里能看出来走偏）
#     Train/episode_length   拉满 1000 = 无提前终止（正常）
#     Train/fell_over        iter 2000 起归零 = 不再摔倒（正常）
#   tracking run 重点：模仿误差类 tag（ee_body_pos 等）应收敛
#       （raw 末段 ~29500 有 -4.3 小回撤，EMA 口径基本单调；best checkpoint = model_29000，
#        峰值 26.22 @ 29079）
#
# 【截图清单（对外分享建议）】
#   1. mean_reward 全程（Smoothing 0.6~0.9）
#   2. 塌陷段局部放大（4500~6500）
#   3. error_vel_xy 同窗口对照（证明 reward 跌 = 跟踪变差）
#   4. 要带峰值/塌陷区间/checkpoint 竖线标注的成品图 → 跑 scripts/14_plot_velocity_curve.py

set -euo pipefail

PORT="${1:-6006}"
WS_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOGDIR="$WS_DIR/repos/unitree_rl_mjlab/logs/rsl_rl"
TB=~/miniconda3/envs/unitree_rl_mjlab/bin/tensorboard

echo "[i] logdir : $LOGDIR"
echo "[i] 浏览器打开 http://localhost:$PORT （Ctrl+C 停止）"
exec "$TB" --logdir "$LOGDIR" --port "$PORT"
