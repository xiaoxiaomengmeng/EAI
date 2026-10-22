#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# 03_replay_g1_sonic_bridge.sh — SONIC 数据桥: 录制数据转 ZMQ pose 流
#
# 链路: 无界面纯转换转发——录制/设备姿态重采样 50 Hz、5 帧滑窗打包,
#       ZMQ PUB 发布 tcp://*:5556 (topic: pose), 供下游 SONIC 进程订阅。
#       手部走 Dex3 重定向 (assets/unitree_dex3_*.urdf), 不需要 G1 模型和 wuji。
# 注意: 仓库内没有 ZMQ 消费端, 单跑本桥只会发流无接收方; 消费进程需另行准备
#       (消息字段与二进制布局见 pico2g1/g1_sonic_bridge/README.md)。
# 入参: replay 只接受文件名含 vr_full_body 的 parquet (不接受 session 目录)。
# 前置: cd pico2g1/g1_sonic_bridge && bash setup.sh   (建 .venv, 装 torch(CPU)/pyzmq/dex-retargeting)
# 用法: scripts/03_replay_g1_sonic_bridge.sh <session名|session路径> [透传参数...]
# 例:   scripts/03_replay_g1_sonic_bridge.sh session_20260828_161538
#       DRY_RUN=1 scripts/03_replay_g1_sonic_bridge.sh session_20260828_161538
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS504="$(dirname "$SCRIPTS_DIR")"
APP_DIR="$WS504/pico2g1/g1_sonic_bridge"
DATA_DIR="$WS504/data"

SESSION="${1:-}"
if [ -z "$SESSION" ]; then
    echo "用法: $0 <session名|session路径> [透传参数...]"
    echo "data/ 下可用 session:"
    for d in "$DATA_DIR"/session_*/; do [ -d "$d" ] && echo "  $(basename "$d")"; done
    exit 1
fi

case "$SESSION" in
    */*) SESSION_DIR="$SESSION" ;;
    *)   SESSION_DIR="$DATA_DIR/$SESSION" ;;
esac
[ -d "$SESSION_DIR" ] || { echo "!! session 目录不存在: $SESSION_DIR"; exit 1; }

PARQUET="$(ls -1 "$SESSION_DIR"/data/chunk-*/observation.data.vr_full_body_*.parquet 2>/dev/null | head -n1 || true)"
[ -n "$PARQUET" ] || { echo "!! 未找到 vr_full_body parquet: $SESSION_DIR/data/chunk-*/"; exit 1; }

[ -x "$APP_DIR/.venv/bin/python" ] || {
    echo "!! $APP_DIR/.venv 不存在, 先建环境:"
    echo "   cd $APP_DIR && bash setup.sh    # 建 .venv, 装 torch(CPU) / pyzmq / dex-retargeting"
    exit 1
}

echo "[session] $SESSION_DIR"
echo "[parquet] $PARQUET"
echo "[zmq]     发布 ${SONIC_ENDPOINT:-tcp://*:5556} (topic: pose) — 确认下游 SONIC 消费进程已就绪"
cd "$APP_DIR"
if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "[dry-run] cd $APP_DIR && .venv/bin/python launch_g1_sonic_bridge.py replay $PARQUET ${*:2}"
    exit 0
fi
exec .venv/bin/python launch_g1_sonic_bridge.py replay "$PARQUET" "${@:2}"
