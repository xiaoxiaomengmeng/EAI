# 八路实时推流

在 G1 IK 或 SONIC 窗口右侧开启「实时推流」，点击「打开预览」。将旁边显示的局域网地址复制到另一台机器的浏览器即可同时观看。两台机器需要能互相访问；如系统询问网络权限，允许仿真进程接收入站连接。

默认端口 **8876**，推流默认关闭。它共享当前主相机、头部、左腕部、右腕部，以及身体正视、身体侧视、左手原始关键点、右手原始关键点画面，拖动/缩放本机主视角也会同步。收起数据采集面板不影响推流。关闭推流或退出窗口会关闭服务；本机回放暂停或结束时保留最后一帧，网页提示画面未更新。

## 命令行启动

在已有 `replay` 或 `live` 命令末尾加 `--stream` 即可，两套程序均支持：

```bash
# 在仓库根目录执行，将路径替换成实际回放会话目录
g1_sonic/.venv/bin/python g1_ik/launch_g1_ik.py replay /path/to/session --stream
g1_sonic/.venv/bin/python g1_sonic/launch_g1_sonic.py replay /path/to/session --stream

# 设备实时输入；本机同时显示，其他机器访问本机的推流地址
g1_sonic/.venv/bin/python g1_ik/launch_g1_ik.py live DEVICE_IP --stream

# Wi-Fi 带宽不足时降低帧率与 JPEG 质量
g1_sonic/.venv/bin/python g1_ik/launch_g1_ik.py replay /path/to/session \
  --stream --stream-port 8876 --stream-fps 20 --stream-quality 65
```

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--stream` | 关闭 | 启动时开启；也可在桌面勾选 |
| `--stream-host` | `0.0.0.0` | 监听所有 IPv4 网卡；`127.0.0.1` 限本机 |
| `--stream-port` | `8876` | HTTP 端口；两套仿真同时运行时使用不同端口 |
| `--stream-fps` | `30` | 编码帧率上限，范围 1–60；实际受本机渲染速度限制 |
| `--stream-quality` | `80` | JPEG 质量，范围 1–95 |

这是面向可信局域网的只读 HTTP 预览服务，不带登录功能；有网络可达性的机器能够观看。页面无外部依赖，无需额外运行前端开发服务器。

## 接口

原有四路相机接口保持原路径，新增四路原始关键点接口。以下路径均相对于 `http://仿真机器IP:8876`。

| 画面 | MJPEG 连续流 | 单帧 JPEG |
| --- | --- | --- |
| 主画面 | `/stream/main.mjpg` | `/snapshot/main.jpg` |
| 头部 | `/stream/robot_head.mjpg` | `/snapshot/robot_head.jpg` |
| 左腕 | `/stream/robot_left_wrist.mjpg` | `/snapshot/robot_left_wrist.jpg` |
| 右腕 | `/stream/robot_right_wrist.mjpg` | `/snapshot/robot_right_wrist.jpg` |
| 身体正视 | `/stream/source_body_front.mjpg` | `/snapshot/source_body_front.jpg` |
| 身体侧视 | `/stream/source_body_side.mjpg` | `/snapshot/source_body_side.jpg` |
| 左手原始关键点 | `/stream/source_left_hand.mjpg` | `/snapshot/source_left_hand.jpg` |
| 右手原始关键点 | `/stream/source_right_hand.mjpg` | `/snapshot/source_right_hand.jpg` |

- `/`：响应式八路预览，左侧主画面独占第一行，头部、左腕、右腕等宽排列在第二行；右侧上排是身体正视和侧视，下排是左右手原始关键点，与 PySide6 侧栏相同。窄屏时原始关键点区域移到仿真区域下方；支持单路放大、全屏、断线重连。
- `/api/streams`：JSON 接口列表、帧率上限、编码耗时及编码错误状态。
- `/api/frames?after=SEQUENCE`：八路同帧二进制数据包；`after=0` 或省略时取最新帧。传入上一帧序号时等待更新，最多 1 秒；无更新返回 HTTP 204。页面使用此接口，确保八路一起显示。

二进制包结构为 **4 字节大端 JSON 长度 + UTF-8 JSON + 八张连续 JPEG**。JSON 示例：

```json
{
  "sequence": 1789441912738559,
  "captured_at": 1789441912.71,
  "age_ms": 45.2,
  "encode_ms": 5.8,
  "frames": [
    {"id": "main", "size": 50000},
    {"id": "robot_head", "size": 10000},
    {"id": "robot_left_wrist", "size": 9000},
    {"id": "robot_right_wrist", "size": 9000},
    {"id": "source_body_front", "size": 6000},
    {"id": "source_body_side", "size": 6000},
    {"id": "source_left_hand", "size": 4000},
    {"id": "source_right_hand", "size": 4000}
  ]
}
```

`size` 是该张 JPEG 的字节长度；按 `frames` 顺序切片。`sequence` 标识同一次渲染，各独立 JPEG 和 MJPEG 分片也包含 `X-Sequence`、`X-Captured-At`。`captured_at` 是主画面开始渲染时的 Unix 秒。客户端重连时从 `after=0` 开始。

## 低延迟策略与边界

图像来自本机正在显示的 MuJoCo 状态，不另建仿真。主画面分辨率随桌面可用区域变化，最长边不超过 1600；三路机器人预览沿用 320×180。四路原始关键点图像使用 PySide6 当前面板尺寸，在 GUI 线程立即绘制后裁出；复用相同投影、骨架连线、颜色和标定/旋转辅助线。尚未收到输入时显示与桌面一致的空面板。LeRobot 的 1280×720 H.264 录制独立于该预览服务。

编码在单独线程中完成，每组八路图像只编码一次，多客户端复用。服务仅保留一组待编码 RGB 和一组最新 JPEG；新帧覆盖旧的待处理帧。网络发送不占用 Qt 渲染线程，也不持有图像共享锁。MJPEG 写入有 2 秒超时，HTTP 并发上限为 32 个连接。

网页每次只请求、解码并显示一组图像，然后请求下一组最新帧；下载速度跟不上时跳过中间帧，避免持续播放积压内容。后台标签暂停拉取，恢复可见后重新获取最新画面。网页请求有 4 秒超时，失败后自动重连。对于低延迟和八路同步，优先使用网页或 `/api/frames`；通用 MJPEG 播放器可能自行缓冲，需关闭其缓存。

所有接口禁用 HTTP 缓存。如果经过反向代理，也需要关闭代理缓冲。浏览器采用 Fetch 的 `cache: 'no-store'`，行为见 [MDN Request.cache](https://developer.mozilla.org/en-US/docs/Web/API/Request/cache)。

页面的「帧龄估计」= 服务端处理该响应时的帧龄 + 收到响应头后下载、解码所用时间；不依赖两台机器时钟一致，但不包含响应头单程网络传播与屏幕扫描时间，不能等同于精确的端到端延迟。「本次下载」仅统计响应体下载，不包含等待下一帧。窄带网络仍可能让一张大图下载变慢，降低质量/帧率可减少带宽占用；不保证任意网络下的固定延迟。

从四路版本升级后，需要重启仿真程序并刷新预览页。`/api/frames` 现在返回八路，接入方应按 `frames` 中的 ID 和 size 解析，不要写死四路数量。
