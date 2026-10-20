#!/usr/bin/env bash
# 验证阶梯 · 阶段 1（零 GPU 成本）：官方 whisking demo 开箱回放
# 前置：repos/do-as-i-do 已克隆（含二级子模块）+ conda env retargeting 已建（pip install -e retargeting/）
# 产出：viser 网页（默认 http://localhost:8081）三层叠加回放，浏览器打开即可逐帧播放
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$ROOT/repos/do-as-i-do/retargeting"

if [ ! -d "$REPO" ]; then
    echo "缺 $REPO：先按 START_HERE.md 第 1 步克隆上游子模块" >&2
    exit 1
fi
if ! conda env list | grep -q '^doasido_retargeting '; then
    echo "缺 conda env doasido_retargeting：见 START_HERE.md 第 2 步（python=3.12 + pip install -e .）" >&2
    exit 1
fi

# 官方 demo 全资产随仓库分发：scene.xml + trajectory_mjwp.npz + kinematic + MANO 关键点 + meshes
cd "$REPO"
conda run -n doasido_retargeting --no-capture-output python replay_viser.py
