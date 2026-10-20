#!/usr/bin/env bash
# 03_launch_tensorboard.sh —— 一键启动 microduck 训练曲线 dashboard（TensorBoard，双源）
#
# 【这是什么】
#   同时挂载两条任务线的日志（TensorBoard --logdir_spec 多目录模式）：
#   ① 官方任务（microduck_rl/logs/rsl_rl）——现存数据：
#      ball_kick_right ×3 run（Mjlab-BallKick-Flat-MicroDuck 轻触踢球，2026-10-12，
#        含 17-41-31 那个 3250 iter 的 run：iter 500 即平台、3250 为过训的实证线）
#      velocity ×1 run（Mjlab-Velocity-Flat-MicroDuck）
#   ② practice 高踢（practice/tasks/ball_kick/logs/rsl_rl）——2026-10-20 已清理，
#      当前为空、目录存在才挂载；用 scripts/01 重训后自动出现：
#      v4 = ball_kick_aggr_right（机制盲，脖子夹球负教材）、v5 = ball_kick_clean_right（机制感知）
#
# 【用法】
#   ./scripts/03_launch_tensorboard.sh          # 默认端口 6006
#   ./scripts/03_launch_tensorboard.sh 6007     # 端口被占时换一个
#
# 【怎么看】
#   浏览器开 http://localhost:6006；左侧按组名（官方任务 / 高踢v4v5）折叠区分两条线；
#   Smoothing 拉 0.6~0.9；双曲线可按住 Shift 点选多条叠图对比。
#
# 【看哪些 tag（高踢线，与 104 的 G1 任务判读不同！本任务站姿被主动砍）】
#   ★ Episode_Reward/ball_lift          高踢主指标：持续往上爬 = 干净踢射在长高（v5 实测 0→1.35）
#   ★ Episode_Reward/ball_neck_contact  v5 专属：绝对值要往 0 走；持续很负 = 又在脖子夹球（exploit 复活）
#   ★ Policy/mean_std                   探索结晶度：卡高位不降 = 策略没结晶（v5 遗留短板，卡过 1.57）
#   ★ Episode_Termination/nan_state     >0 要警惕数值爆炸（v4 曾现 -1.99e8 负尖峰）
#   官方轻触线的判读参考 docs/microduck_rl/01-quickstart/desktop_quickstart.md §7.5；
#   高踢线每个控制台字段的含义与代码出处：practice/tasks/ball_kick/README.md §7.5

set -euo pipefail

PORT="${1:-6006}"
WS_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TB="$WS_DIR/microduck_rl/.venv/bin/tensorboard"
OFFICIAL="$WS_DIR/microduck_rl/logs/rsl_rl"                  # 官方任务线（现存数据）
PRACTICE="$WS_DIR/practice/tasks/ball_kick/logs/rsl_rl"     # practice 高踢线（重训后出现）

[[ -d "$OFFICIAL" ]] || { echo "[x] $OFFICIAL 不存在"; exit 1; }

SPEC="官方任务:$OFFICIAL"
if [[ -d "$PRACTICE" ]]; then
  SPEC="$SPEC,高踢v4v5:$PRACTICE"
else
  echo "[i] practice 高踢日志目录不存在（已清理），本次只挂官方线；跑 01 脚本重训后自动挂上"
fi

echo "[i] 双源挂载: $SPEC"
echo "[i] 浏览器打开 http://localhost:$PORT（Ctrl+C 停止）"
exec "$TB" --logdir_spec "$SPEC" --port "$PORT"
