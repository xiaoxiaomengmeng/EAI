#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# 01_replay_g1_ik.sh — g1_ik 轻量 IK 回放录制 session
#
# 链路: MuJoCo 雅可比 + 带限位最小二乘直接解 G1 29 DoF, 不跑策略网络;
#       PySide6 界面实时调参 (滑杆写 g1_ik/tuning.json, 不入库)。
# 入参: replay 只接受 observation.data.vr_full_body_*.parquet 文件 (不接受 session 目录),
#       手套数据按文件名替换 vr_full_body -> glove_left/right 自动找到。
# 前置: cd pico2g1/g1_ik && bash create_env.sh   (本机 .venv 已建于 2026-10-20, import 冒烟全通)
# 用法: scripts/01_replay_g1_ik.sh <session名|session路径> [透传参数...]
# 例:   scripts/01_replay_g1_ik.sh session_20260828_161538
#       scripts/01_replay_g1_ik.sh session_20260828_161538 --scene pulse --render-shadows
#       DRY_RUN=1 scripts/01_replay_g1_ik.sh session_20260828_161538   # 只打印命令不执行
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS504="$(dirname "$SCRIPTS_DIR")"
APP_DIR="$WS504/pico2g1/g1_ik"
DATA_DIR="$WS504/data"

SESSION="${1:-}"
if [ -z "$SESSION" ]; then
    echo "用法: $0 <session名|session路径> [透传参数...]  (如 --scene pulse --render-shadows)"
    echo "data/ 下可用 session:"
    for d in "$DATA_DIR"/session_*/; do [ -d "$d" ] && echo "  $(basename "$d")"; done
    exit 1
fi

# session 名 -> 目录; 传入含 / 的参数视为路径直接用
case "$SESSION" in
    */*) SESSION_DIR="$SESSION" ;;
    *)   SESSION_DIR="$DATA_DIR/$SESSION" ;;
esac
[ -d "$SESSION_DIR" ] || { echo "!! session 目录不存在: $SESSION_DIR"; exit 1; }

# replay 入参是 vr_full_body parquet; 多 episode 时取第一个
PARQUET="$(ls -1 "$SESSION_DIR"/data/chunk-*/observation.data.vr_full_body_*.parquet 2>/dev/null | head -n1 || true)"
[ -n "$PARQUET" ] || { echo "!! 未找到 vr_full_body parquet: $SESSION_DIR/data/chunk-*/observation.data.vr_full_body_*.parquet"; exit 1; }

[ -x "$APP_DIR/.venv/bin/python" ] || {
    echo "!! $APP_DIR/.venv 不存在, 先建环境:"
    echo "   cd $APP_DIR && bash create_env.sh"
    exit 1
}

echo "[session] $SESSION_DIR"
echo "[parquet] $PARQUET"
cd "$APP_DIR"
if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "[dry-run] cd $APP_DIR && .venv/bin/python launch_g1_ik.py replay $PARQUET ${*:2}"
    exit 0
fi

# GUI 前置: DISPLAY 自动探测 (IDE/SSH 远程会话常无 DISPLAY; 物理桌面 Xorg 挂在 :0/:1)
if [ -z "${DISPLAY:-}" ]; then
    uid_="$(id -u)"
    for sock in /tmp/.X11-unix/X*; do
        [ -S "$sock" ] || continue
        guess=":${sock##*/X}"
        auth="/run/user/$uid_/gdm/Xauthority"
        [ -r "$auth" ] || auth="$HOME/.Xauthority"
        if DISPLAY="$guess" XAUTHORITY="$auth" timeout 3 xset q >/dev/null 2>&1; then
            export DISPLAY="$guess" XAUTHORITY="$auth"
            echo "[display] 自动探测: DISPLAY=$DISPLAY (XAUTHORITY=$auth)"
            break
        fi
    done
fi
if [ -z "${DISPLAY:-}" ]; then
    echo "!! 未探测到可用 X display: PySide6 界面需要图形会话。"
    echo "   物理桌面在跑时可手动指定: export DISPLAY=:1 XAUTHORITY=/run/user/$(id -u)/gdm/Xauthority"
    echo "   或到物理机前 / 远程桌面 (AnyDesk) 里跑; headless 验证可 QT_QPA_PLATFORM=offscreen (看不到画面)。"
    exit 1
fi
exec .venv/bin/python launch_g1_ik.py replay "$PARQUET" "${@:2}"
