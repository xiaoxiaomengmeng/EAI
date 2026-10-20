#!/usr/bin/env bash
# 验证阶梯 · 阶段 0（零成本）：导出 DexAgent stage1 将发给 LLM 的完整 prompt 包
# 前置：repos/DexAgent 已克隆、conda env dexagent 已建并激活（见 START_HERE.md）
# 产出：502-DexAgent/out_prompt/{system.md, tools.json, user.txt}
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$ROOT/repos/DexAgent"
OUT="$ROOT/out_prompt"

if [ ! -d "$REPO" ]; then
    echo "缺 $REPO：先按 START_HERE.md 第 1 步克隆上游子模块" >&2
    exit 1
fi

mkdir -p "$OUT"
cd "$REPO"
python -m agent.render_prompt stage1 dexagent.yaml "$OUT"
echo "prompt 包已导出到 $OUT"
