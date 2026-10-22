#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# 02_replay_g1_sonic.sh — SONIC v1.1 策略回放录制 session
#
# 链路: PICO 24 关节 -> SMPL -> SONIC v1.1 ONNX 策略 (CPU) 驱动 G1 29 电机;
#       手套 21 关键点 -> wuji-retargeting -> Wuji Hand 40 关节。固定 50 Hz,
#       PySide6 双画面 (左 G1, 右原始人体+双手), 界面「重置机器人」清 SONIC 10 帧历史。
# 入参: replay 接受 session 目录 (也可传 parquet, 与 01 脚本同规则); 本脚本按上游
#       文档姿势传 session 目录, 内部含 vr/glove/videos 全套。
# 前置: cd pico2g1/g1_sonic && bash setup.sh
#       (建 .venv + 拉 wuji 到 3rd/ + 从 HF nvidia/GEAR-SONIC 下 ~191MB onnx 到 models/)
# 用法: scripts/02_replay_g1_sonic.sh <session名|session路径> [透传参数...]
# 例:   scripts/02_replay_g1_sonic.sh session_20260828_161538
#       scripts/02_replay_g1_sonic.sh session_20260828_161538 --scene pulse
#       DRY_RUN=1 scripts/02_replay_g1_sonic.sh session_20260828_161538
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS504="$(dirname "$SCRIPTS_DIR")"
APP_DIR="$WS504/pico2g1/g1_sonic"
DATA_DIR="$WS504/data"

SESSION="${1:-}"
if [ -z "$SESSION" ]; then
    echo "用法: $0 <session名|session路径> [透传参数...]  (如 --scene pulse --render-shadows)"
    echo "data/ 下可用 session:"
    for d in "$DATA_DIR"/session_*/; do [ -d "$d" ] && echo "  $(basename "$d")"; done
    exit 1
fi

case "$SESSION" in
    */*) SESSION_DIR="$SESSION" ;;
    *)   SESSION_DIR="$DATA_DIR/$SESSION" ;;
esac
[ -d "$SESSION_DIR" ] || { echo "!! session 目录不存在: $SESSION_DIR"; exit 1; }

# 数据自检: replay 至少要能找到 vr_full_body parquet
PARQUET="$(ls -1 "$SESSION_DIR"/data/chunk-*/observation.data.vr_full_body_*.parquet 2>/dev/null | head -n1 || true)"
[ -n "$PARQUET" ] || { echo "!! 未找到 vr_full_body parquet: $SESSION_DIR/data/chunk-*/"; exit 1; }

[ -x "$APP_DIR/.venv/bin/python" ] || {
    echo "!! $APP_DIR/.venv 不存在, 先建环境:"
    echo "   cd $APP_DIR && bash setup.sh    # 建 .venv + 拉 wuji + 下 ~191MB SONIC onnx"
    exit 1
}
if [ -z "$(find "$APP_DIR/models" -name '*.onnx' -print -quit 2>/dev/null)" ]; then
    echo "!! $APP_DIR/models/ 下没有 onnx 策略权重, 先补下载:"
    echo "   cd $APP_DIR && bash setup.sh    # 或按 setup.sh 内 HF 地址手动下"
    exit 1
fi

echo "[session] $SESSION_DIR"
cd "$APP_DIR"
if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "[dry-run] cd $APP_DIR && .venv/bin/python launch_g1_sonic.py replay $SESSION_DIR ${*:2}"
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
exec .venv/bin/python launch_g1_sonic.py replay "$SESSION_DIR" "${@:2}"
