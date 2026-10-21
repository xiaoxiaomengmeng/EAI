---
name: convert-webm-to-mp4
description: 在 Ubuntu 上把 webm（VP8/VP9）录屏转换为 H.264 mp4，含奇数分辨率抹偶、实测压缩参数与视频不入库规则。当用户要求 webm 转 mp4、转换录屏或演示视频格式、或 ffmpeg 报 width not divisible by 2 时使用。平台限定 Ubuntu。
---

# webm 转 mp4（Ubuntu）

## 前置检查

1. 确认 ffmpeg 可用：`command -v ffmpeg`；缺失则提示用户 `sudo apt install ffmpeg`（apt 安装由用户亲自执行）。
2. 探查源文件流信息（分辨率、编码、有无音频、时长）：

```bash
ffprobe -v error -show_entries format=duration,size \
  -show_entries stream=codec_name,codec_type,width,height \
  -of default=noprint_wrappers=1 in.webm
```

## 转换命令（标准配方）

```bash
ffmpeg -y -i in.webm -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" \
  -c:v libx264 -crf 20 -preset medium -movflags +faststart out.mp4
```

- `scale=trunc(iw/2)*2:trunc(ih/2)*2`：宽高抹到偶数。录屏分辨率常为奇数（实测踩坑 1851×1093），libx264/yuv420p 直接报 `width not divisible by 2` 失败；对偶数输入无副作用，无条件带上。
- `-crf 20 -preset medium`：实测压缩率——28M VP8 → 8.7M，193M → 64M。
- `-movflags +faststart`：moov 前置，浏览器与预览面板可流式播放。
- 录屏通常无音频流；若 ffprobe 显示有 opus/vorbis 音轨，追加 `-c:a aac`（保留）或 `-an`（丢弃）。

## 执行注意

- 输出文件已存在时先向用户确认再覆盖（`-y` 会静默覆盖，违反不可逆操作先确认约定）。
- 源 >100M 转换约需 2 分钟级：放后台执行或拉长 timeout。
- 转换后验证：ffprobe 确认 codec=h264、宽高为偶数、duration 与源一致；`ls -lh` 对比大小并向用户报告数字。

## 工作区入库约定

视频一律不入库：.gitignore 用通配 `pics/*.mp4` + `pics/*.webm`（不用具体文件名，避免重命名后条目静默失效、大文件被误提交）。转换完成后 webm 源文件是否删除由用户确认，不擅自删。
