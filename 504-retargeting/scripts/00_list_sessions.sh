#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# 00_list_sessions.sh — 概览 data/ 下已解压的录制 session
#
# 用途: 下载数据后先看一眼有什么——每个 session 的任务名 / 时长 / 采集时间,
#       以及回放所需的四类 parquet 是否齐全 (vr_full_body / vr / glove_left / glove_right)。
# 用法: scripts/00_list_sessions.sh
# 后续: 看单文件 schema 细节用 pico2g1/tools/inspect_parquet.py (可借任意链路 .venv 跑)。
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$(dirname "$SCRIPTS_DIR")/data"

shopt -s nullglob
sessions=("$DATA_DIR"/session_*/)
shopt -u nullglob

if [ ${#sessions[@]} -eq 0 ]; then
    echo "data/ 下没有已解压的 session 目录 (zip 需先解压): $DATA_DIR"
    ls -lh "$DATA_DIR" 2>/dev/null || true
    exit 1
fi

echo "data/ 下共 ${#sessions[@]} 个 session:"
for d in "${sessions[@]}"; do
    d="${d%/}"
    name="$(basename "$d")"
    echo
    echo "── $name"
    if [ -f "$d/metadata.json" ]; then
        python3 - "$d/metadata.json" <<'EOF'
import json, sys
m = json.load(open(sys.argv[1]))
print("   任务: %s    时长: %ss    采集: %s" %
      (m.get("task_name", "?"), m.get("duration_sec", "?"), m.get("collect_time", "?")))
EOF
    else
        echo "   (缺 metadata.json)"
    fi
    chunk="$(ls -1d "$d"/data/chunk-* 2>/dev/null | head -n1 || true)"
    if [ -n "$chunk" ]; then
        missing=""
        for kind in vr_full_body vr glove_left glove_right; do
            ls "$chunk"/observation.data.${kind}_*.parquet >/dev/null 2>&1 || missing="$missing $kind"
        done
        if [ -z "$missing" ]; then
            echo "   parquet: 四类齐全 ✅  ($chunk/)"
        else
            echo "   parquet: 缺$missing ⚠️  (回放需要 vr_full_body / vr / glove_left / glove_right)"
        fi
    else
        echo "   parquet: 未找到 data/chunk-*/ ⚠️"
    fi
    if [ -d "$d/videos" ]; then
        echo "   videos: $(find "$d/videos" -type f | wc -l) 个文件"
    fi
done

echo
echo "提示: schema 细节 -> pico2g1/tools/inspect_parquet.py <session目录或parquet>"
echo "      回放 -> scripts/01_replay_g1_ik.sh <session名>"
