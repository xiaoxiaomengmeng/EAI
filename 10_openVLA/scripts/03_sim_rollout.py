"""阶段③ · 自定义 MuJoCo 闭环 rollout（Franka Panda，不依赖 repos/LIBERO）。

与 scripts/ 下另外两个脚本相互独立、各成入口：
- 01_openvla_verify.py：单步 7D action，验证 repos/openvla 源码可加载；
- 02_libero_eval.py   ：LIBERO 官方 benchmark 成功率评测（走 run_libero_eval）；
- 03_sim_rollout.py（本脚本）：在本项目自定义 tabletop_scene.xml 上跑
  「模型→action→IK→MuJoCo」的闭环 rollout，采集视频 + 逐步 action 数据。

关于「模型加载」的取舍（本脚本刻意与 sim 现状不同）：
- sim/openvla_policy.py 走裸 transformers + trust_remote_code=True（等同
  demo_predict_action.py），加载的是 checkpoint 内嵌远程代码；base 能跑，但微调
  checkpoint 的 auto_map 指向跨仓库远程代码会联网卡死。
- 本脚本改用「调 repos/openvla 源码」的方式（对齐 01_openvla_verify.py）：把
  repos/openvla 注入 sys.path，用 prismatic 源码把 OpenVLA 注册到 HF Auto 类，再以
  trust_remote_code=False + local_files_only=True 离线加载。优点：全离线零联网挂起、
  base/微调 checkpoint 通吃、闭环里跑的确是本仓库 repos/openvla 源码实现，且与阶段①
  同一条加载路径，①→③ 阶梯自洽。一次性多拉起 tensorflow 的约 6s 噪音对多步闭环可忽略。

闭环本身（MuJoCoEnv + IK + 抓取状态机 + 视频/JSON 落盘）直接复用本项目 sim 包，不重写；
本脚本只把「策略加载」这一层换成源码方式，通过子类覆盖 RolloutRunner._load_policy 注入。

用法示例：
    cd /home/creator/ws/10_openVLA && source .venv/bin/activate
    # 先 dry-run 冒烟（不加载 7B，只验证 MuJoCo/IK/落盘链路）
    python scripts/03_sim_rollout.py --dry-run --policy-steps 2 --sim-steps-per-policy 2
    # base 模型 + bridge_orig 跑真实模型驱动的闭环（自定义场景标准组合）
    python scripts/03_sim_rollout.py --instruction "pick up the red cube" --policy-steps 50
    # 也可换成任意本地 checkpoint（源码加载对微调权重同样离线可用）
    python scripts/03_sim_rollout.py --preset libero-spatial --unnorm-key libero_spatial
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

# headless 渲染必须在 import mujoco 前设置（sim.mujoco_env 也会 setdefault，这里再兜底一次）
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPENVLA_REPO = PROJECT_ROOT / "repos" / "openvla"
MODULES_DIR = PROJECT_ROOT.parent / "modules"

if not OPENVLA_REPO.is_dir():
    raise SystemExit(f"未找到 repos/openvla 源码目录：{OPENVLA_REPO}")

# 注入 repos/openvla（导入 prismatic 源码）与项目根（导入 sim 包）。
sys.path.insert(0, str(OPENVLA_REPO))
sys.path.insert(0, str(PROJECT_ROOT))

# 从 repos/openvla 的 prismatic 源码包导入 OpenVLA 的 HF 适配类（源码加载的核心）。
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import (
    PrismaticImageProcessor,
    PrismaticProcessor,
)

# 复用 sim 包的闭环基础设施（环境/控制/落盘全部走 sim，不重写）。
from sim.config import (
    ControlConfig,
    PolicyConfig,
    RenderConfig,
    RolloutConfig,
    SceneConfig,
)
from sim.openvla_policy import DryRunPolicy
from sim.robots import DEFAULT_ROBOT, ROBOT_CONFIGS
from sim.runner import RolloutRunner

# 预置本地 checkpoint（对齐 01_openvla_verify.py）。自定义场景标准组合是 base + bridge_orig。
PRESETS = {
    "base": {
        "checkpoint": str(MODULES_DIR / "openvla-7b"),
        "unnorm_key": "bridge_orig",
    },
    "libero-spatial": {
        "checkpoint": str(MODULES_DIR / "openvla-7b-finetuned-libero-spatial"),
        "unnorm_key": "libero_spatial",
    },
}

ATTN_CHOICES = ["eager", "sdpa", "flash_attention_2"]


def log_step(message: str):
    print(f"[Sim Rollout] {message}", flush=True)


def register_openvla_auto_classes():
    """用 prismatic 源码把 OpenVLA 注册到 HF Auto 类（对齐 openvla_utils.get_vla）。

    注册后以 trust_remote_code=False + local_files_only=True 加载，使用的就是
    repos/openvla 源码实现，而非 checkpoint 内嵌/远程代码，从而彻底避免联网挂起。
    """
    from transformers import (
        AutoConfig,
        AutoImageProcessor,
        AutoModelForVision2Seq,
        AutoProcessor,
    )

    AutoConfig.register("openvla", OpenVLAConfig, exist_ok=True)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor, exist_ok=True)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor, exist_ok=True)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction, exist_ok=True)


class SourceOpenVLAPolicy:
    """基于 repos/openvla 源码的 OpenVLA 策略，接口与 sim.openvla_policy.OpenVLAPolicy 对齐。

    对齐点（供 RolloutRunner 复用）：predict(image, instruction) -> np.ndarray(shape=(7,))，
    并暴露 load_timing / last_timing 供 runner 汇总耗时。
    差异点：加载走源码方式（trust_remote_code=False + local_files_only=True），
    并读取 checkpoint 内 dataset_statistics.json 覆盖 norm_stats（对齐 get_vla）。
    """

    def __init__(self, model_path: str, attn_implementation: str, unnorm_key: str):
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        self.model_path = model_path
        self.unnorm_key = unnorm_key
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.load_timing = {}
        self.last_timing = {}

        log_step(f"源码加载 processor：{model_path}")
        processor_start = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=False, local_files_only=True
        )
        self.load_timing["processor_load_sec"] = time.perf_counter() - processor_start

        model_kwargs = {
            "attn_implementation": attn_implementation,
            "torch_dtype": self.dtype,
            "low_cpu_mem_usage": True,
            "trust_remote_code": False,
            "local_files_only": True,
        }
        if torch.cuda.is_available():
            model_kwargs["device_map"] = {"": self.device}

        log_step(f"源码加载模型：attn={attn_implementation}, dtype={self.dtype}, device={self.device}")
        model_start = time.perf_counter()
        self.model = AutoModelForVision2Seq.from_pretrained(model_path, **model_kwargs)
        self.load_timing["model_load_sec"] = time.perf_counter() - model_start

        prepare_start = time.perf_counter()
        if not torch.cuda.is_available():
            self.model = self.model.to(self.device)
        self.model.eval()
        self.load_timing["model_prepare_sec"] = time.perf_counter() - prepare_start
        self.load_timing["policy_load_sec"] = sum(self.load_timing.values())

        self._maybe_load_dataset_statistics(model_path)
        self._assert_unnorm_key()
        log_step(
            f"模型就绪：{type(self.model).__name__}（模块 {type(self.model).__module__}），"
            f"device={next(self.model.parameters()).device}"
        )

    def _maybe_load_dataset_statistics(self, checkpoint: str):
        stats_path = os.path.join(checkpoint, "dataset_statistics.json")
        if os.path.isfile(stats_path):
            with open(stats_path, "r", encoding="utf-8") as handle:
                self.model.norm_stats = json.load(handle)
            log_step(f"已加载 dataset_statistics.json：{stats_path}")

    def _assert_unnorm_key(self):
        available = list(getattr(self.model, "norm_stats", {}) or {})
        if self.unnorm_key not in available:
            raise SystemExit(
                f"unnorm_key='{self.unnorm_key}' 不在 norm_stats 中。\n"
                f"可用 key（前 20 个）：{available[:20]}\n"
                f"base 模型请用 bridge_orig；LIBERO 微调 checkpoint 用对应 libero_* key。"
            )

    def predict(self, image: Image.Image, instruction: str) -> np.ndarray:
        """执行一次源码实现的 OpenVLA 推理，返回 shape=(7,) 的连续 action。"""
        import torch

        # OpenVLA 官方 prompt 模板（与 01_openvla_verify.py / demo_predict_action.py 一致）。
        prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"

        preprocess_start = time.perf_counter()
        inputs = self.processor(prompt, image).to(self.device, dtype=self.dtype)
        preprocess_sec = time.perf_counter() - preprocess_start

        forward_start = time.perf_counter()
        with torch.inference_mode():
            action = self.model.predict_action(
                **inputs, unnorm_key=self.unnorm_key, do_sample=False
            )
        model_forward_sec = time.perf_counter() - forward_start

        postprocess_start = time.perf_counter()
        action_array = np.asarray(action, dtype=np.float64).reshape(-1)
        assert action_array.shape == (7,), f"期望 7D action，实际 shape={action_array.shape}"
        postprocess_sec = time.perf_counter() - postprocess_start

        self.last_timing = {
            "preprocess_sec": preprocess_sec,
            "model_forward_sec": model_forward_sec,
            "postprocess_sec": postprocess_sec,
        }
        return action_array


class SourcePackageRunner(RolloutRunner):
    """复用 sim 的闭环逻辑，仅把策略加载换成 repos/openvla 源码方式。"""

    def _load_policy(self):
        if self.rollout_config.dry_run:
            log_step("dry-run：使用确定性测试动作，不加载 OpenVLA。")
            return DryRunPolicy()
        return SourceOpenVLAPolicy(
            model_path=self.policy_config.model_path,
            attn_implementation=self.policy_config.attn_implementation,
            unnorm_key=self.policy_config.unnorm_key,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="自定义 MuJoCo 闭环 rollout（Franka Panda）；模型走 repos/openvla 源码离线加载。"
    )
    parser.add_argument(
        "--preset",
        default="base",
        choices=sorted(PRESETS),
        help="预置本地 checkpoint。自定义场景标准组合是 base + bridge_orig。",
    )
    parser.add_argument("--checkpoint", default=None, help="本地 OpenVLA checkpoint 目录（覆盖 --preset）。")
    parser.add_argument("--unnorm-key", default=None, help="action 反归一化 key（覆盖 --preset）。")
    parser.add_argument(
        "--robot",
        default=DEFAULT_ROBOT,
        choices=list(ROBOT_CONFIGS.keys()),
        help="机器人类型（仅 Franka Panda）。",
    )
    parser.add_argument("--instruction", default="pick up the red cube", help="自然语言指令。")
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        choices=ATTN_CHOICES,
        help="注意力后端；RTX 5090 上默认 sdpa。",
    )
    parser.add_argument(
        "--motion-mode",
        default="ik",
        choices=["ik", "demo", "direct"],
        help="运动模式。ik：ee_site Jacobian IK（默认）。",
    )
    parser.add_argument("--policy-steps", type=int, default=50, help="OpenVLA 策略查询次数。")
    parser.add_argument("--sim-steps-per-policy", type=int, default=30, help="每个策略 action 后的 MuJoCo 步数。")
    parser.add_argument(
        "--grasp-assist",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="靠近方块时锁定闭爪，让抓取完成可见。",
    )
    parser.add_argument("--output-dir", default=None, help="视频/帧/日志输出目录；省略时用 sim 默认。")
    parser.add_argument("--width", type=int, default=640, help="渲染宽度。")
    parser.add_argument("--height", type=int, default=480, help="渲染高度。")
    parser.add_argument("--dry-run", action="store_true", help="跳过 OpenVLA 加载，使用确定性测试动作。")
    parser.add_argument(
        "--save-frames", action="store_true", help="额外保存逐帧调试/视频 RGB 图。"
    )
    parser.add_argument(
        "--save-policy-inputs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="保存送入 OpenVLA 的策略输入图。",
    )
    return parser.parse_args()


def resolve_checkpoint_and_key(args):
    """按 CLI > preset 解析 checkpoint 与 unnorm_key（dry-run 时不校验目录存在）。"""
    preset = PRESETS[args.preset]
    checkpoint = args.checkpoint or preset["checkpoint"]
    unnorm_key = args.unnorm_key or preset["unnorm_key"]
    if not args.dry_run:
        ckpt_path = Path(checkpoint).expanduser()
        if not ckpt_path.is_dir():
            raise SystemExit(
                f"checkpoint 目录不存在：{ckpt_path}\n"
                f"可用 --preset 切换，或用 --checkpoint 指定本地路径。"
            )
        checkpoint = str(ckpt_path)
    return checkpoint, unnorm_key


def build_runner(args) -> SourcePackageRunner:
    robot_config = ROBOT_CONFIGS[args.robot]
    checkpoint, unnorm_key = resolve_checkpoint_and_key(args)

    scene = SceneConfig()
    render = RenderConfig(width=args.width, height=args.height)
    policy = PolicyConfig(
        model_path=checkpoint,
        instruction=args.instruction,
        unnorm_key=unnorm_key,
        attn_implementation=args.attn_implementation,
    )
    control = ControlConfig(motion_mode=args.motion_mode, grasp_assist=args.grasp_assist)
    rollout = RolloutConfig(
        policy_steps=args.policy_steps,
        sim_steps_per_policy=args.sim_steps_per_policy,
        output_dir=args.output_dir or RolloutConfig().output_dir,
        save_frames=args.save_frames,
        save_policy_inputs=args.save_policy_inputs,
        dry_run=args.dry_run,
    )
    log_step(f"prismatic 源码来自：{OPENVLA_REPO}")
    log_step(f"robot={robot_config.name} | checkpoint={checkpoint} | unnorm_key={unnorm_key}")
    log_step(f"policy_steps={rollout.policy_steps} | sim_steps_per_policy={rollout.sim_steps_per_policy} | motion_mode={control.motion_mode}")
    return SourcePackageRunner(
        robot_config=robot_config,
        scene=scene,
        render=render,
        policy=policy,
        control=control,
        rollout=rollout,
    )


def main():
    args = parse_args()
    if not args.dry_run:
        log_step("注册 OpenVLA 到 HF Auto 类（使用 repos/openvla 源码实现）...")
        register_openvla_auto_classes()
    runner = build_runner(args)
    video_path, json_log_path, log_file_path = runner.run()
    print("\n===== 闭环 rollout 完成 =====")
    print("video   :", video_path)
    print("json log:", json_log_path)
    print("text log:", log_file_path)


if __name__ == "__main__":
    main()
