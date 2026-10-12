# 200-video

一句话：ComfyUI + MiniMax H3 音视频生成模型在 RTX 5090（32G）上的本地部署；两者配合使用，作为一个整体目录。

## 启动环境

```bash
cd ~/ws/200-video/ComfyUI
source .venv/bin/activate      # Python 3.10，torch 2.11.0+cu128
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

- 源码：`ComfyUI/`（子模块，已内置 MiniMax H3 节点；custom_nodes 有 ComfyUI-Manager、ComfyUI_MiniMaxH3_Director）
- 权重：MiniMax-H3 已于 2026-10-12 清理（modelscope 缓存 97G + ComfyUI 侧 6 条软链，`--quick-test-for-ci` 不依赖它）；需要时按 [miniMax/README.md](miniMax/README.md) §4 重新下载并重建软链

## 从这里开始

- 部署指南：[miniMax/README.md](miniMax/README.md)（模型说明、环境变量、工作流模板）
- 启动服务：`python main.py --listen 0.0.0.0 --port 8188`，浏览器打开 `http://<本机IP>:8188`
- 自检：`python main.py --quick-test-for-ci`（初始化完自动退出）
