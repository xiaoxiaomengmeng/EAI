#!/usr/bin/env python3
"""列出当前 Python 环境中的 editable 安装包，并逐个尝试 import 其顶层模块。

用途：目录改名前后对比，发现 editable finder / .pth 中的绝对路径失效
（pip show 正常但 import 失败）。

用法：<venv>/bin/python scripts/check_venv_editable.py
"""
import importlib
import importlib.metadata as md
import json
import sys

editable = []
for d in md.distributions():
    raw = d.read_text("direct_url.json")
    if not raw:
        continue
    try:
        info = json.loads(raw)
    except ValueError:
        continue
    if info.get("dir_info", {}).get("editable"):
        name = d.metadata["Name"]
        tops = (d.read_text("top_level.txt") or "").split()
        if not tops:
            tops = [name.replace("-", "_").lower()]
        editable.append((name, info.get("url", ""), tops))

lines = []
for name, url, tops in sorted(editable):
    status = []
    for top in tops:
        if top.startswith("_"):
            continue
        try:
            importlib.import_module(top)
            status.append(f"{top}:ok")
        except Exception as e:  # noqa: BLE001 - 只记录失败类型
            status.append(f"{top}:FAIL({type(e).__name__})")
    lines.append(f"  {name} <- {url.replace('file://', '')} [{', '.join(status)}]")

print(f"python {sys.version.split()[0]}, editable={len(editable)}")
print("\n".join(lines) if lines else "  (no editable packages)")
