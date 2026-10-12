# 302-kingkongrobotics

一句话：金钢科技（KingKongRobotics）Jumper 六足机器人"跳跳"——22 自由度异构六足，RK3576 + 6 TOPS NPU 板载推理；mjlab/rsl_rl 训练速度跟踪与舞蹈/手势模仿策略，sim2real 上真机。

## 启动环境

```bash
cd ~/ws/302-kingkongrobotics/jumper
source .venv/bin/activate     # Python 3.12 · torch 2.11.0+cu130 · mjrl_lab 为 editable（聚合 controller / mjlab / mjrl / rsl_rl / scenes / tasks 六个包）
python scripts/train.py --task jumper.flat --num_envs 16 --max-iterations 1 --headless --no-tensorboard   # 冒烟
```

- 统一入口 `scripts/train.py`：`--list` 看全部任务（jumper.flat / five_foot 速度跟踪、dance_* 舞蹈模仿、gesture_* 招手作揖等）；`play.py` 回放、`export.py` 导出、`deploy.py` 部署。训练默认自动起 tensorboard，冒烟记得 `--no-tensorboard`。
- 环境已于 2026-10-12 目录改名后修复：editable finder 映射与 import 全链路验证通过。

## 从这里开始

- **部署与真机**：[deploy.md](../deploy.md)（RKNN 转换、板端部署全流程）
- **宣传文案**：[docs/media/official/](docs/media/official)（intro 长文 / intro2 精简版 / voiceover 口播稿）
- 上游说明：[jumper/README.md](jumper/README.md)
