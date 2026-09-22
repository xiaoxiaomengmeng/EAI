"""仅依赖 repos/openvla 源码包的 OpenVLA 验证脚本（不依赖本项目的 sim 包）。

与 scripts/demo_predict_action.py 的区别：
- demo_predict_action.py 走裸 transformers + trust_remote_code，模型代码来自 checkpoint 目录，
  并且依赖 sim.config 提供默认模型路径；
- 本脚本刻意只使用 repos/openvla 的 prismatic 源码包：把 repos/openvla 注入 sys.path，
  用 prismatic.extern.hf.* 把 OpenVLA 注册到 HF Auto 类（等价于 experiments/robot/openvla_utils.get_vla
  的注册流程），从而真正验证 repos/openvla 源码本身是否可用。

因此本脚本可在“尚未 pip install -e repos/openvla”的情况下运行，只要该目录在磁盘上即可。

用法示例：
    cd /home/creator/ws/10_openVLA && source .venv/bin/activate
    # base 模型（默认），单步 7D action
    python scripts/01_openvla_verify.py
    # 使用已下载的 LIBERO 微调 checkpoint
    python scripts/01_openvla_verify.py --preset libero-spatial
    # 指定任意本地 checkpoint 路径 + 真实图片
    python scripts/01_openvla_verify.py \
        --checkpoint /path/to/ckpt --image /path/to/img.jpg \
        --instruction "pick up the red cube" --unnorm-key libero_spatial
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForVision2Seq,
    AutoProcessor,
)

# 关键：把 repos/openvla 注入 sys.path，使得无需 pip install 也能 import prismatic 源码包。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPENVLA_REPO = PROJECT_ROOT / "repos" / "openvla"
if not OPENVLA_REPO.is_dir():
    raise SystemExit(f"未找到 repos/openvla 源码目录：{OPENVLA_REPO}")
sys.path.insert(0, str(OPENVLA_REPO))

# 从 repos/openvla 的 prismatic 源码包导入 OpenVLA 的 HF 适配类。
# 这一步 import 成功本身就是“repos/openvla 源码可用”的核心验证点。
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import (
    PrismaticImageProcessor,
    PrismaticProcessor,
)

# 工作区模型统一目录（ws/modules）：里面是指向 ModelScope 缓存的软链。
# 用软链而非硬编码缓存路径，日后换缓存位置只需改软链。
MODULES_DIR = PROJECT_ROOT.parent / "modules"

# 预置的本地 checkpoint。base 用于路线 a 的基础链路验证；
# libero-spatial 是用户已下载的 ModelScope 微调 checkpoint。
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

DEFAULT_INSTRUCTION = "pick up the object"
ATTN_CHOICES = ["eager", "sdpa", "flash_attention_2"]
DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def log_step(message):
    print(f"[OpenVLA Verify] {message}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="仅依赖 repos/openvla 源码包的 OpenVLA 单步动作预测验证。"
    )
    parser.add_argument(
        "--preset",
        default="base",
        choices=sorted(PRESETS),
        help="预置的本地 checkpoint。--checkpoint / --unnorm-key 若显式给出则覆盖预置值。",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="本地 OpenVLA checkpoint 目录（覆盖 --preset 的路径）。",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="输入 RGB 图片路径；省略时使用 224x224 占位图，仅验证推理链路。",
    )
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION, help="机器人指令文本。")
    parser.add_argument(
        "--unnorm-key",
        default=None,
        help="action 反归一化 key（覆盖 --preset 的默认值）。",
    )
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        choices=ATTN_CHOICES,
        help="注意力后端。RTX 5090 上默认用 sdpa；eager 在 OpenVLA 生成时可能触发 causal-mask 长度不匹配。",
    )
    parser.add_argument(
        "--do-sample",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="采样解码而非贪心解码。",
    )
    parser.add_argument("--device", default=None, help="留空自动选择 cuda:0 / cpu。")
    parser.add_argument(
        "--dtype",
        default=None,
        choices=sorted(DTYPE_MAP),
        help="计算精度；留空时 GPU 用 bfloat16、CPU 用 float32。",
    )
    return parser.parse_args()


def resolve_checkpoint_and_key(args):
    """按 CLI > preset 的优先级解析 checkpoint 路径与 unnorm_key。"""
    preset = PRESETS[args.preset]
    checkpoint = args.checkpoint or preset["checkpoint"]
    unnorm_key = args.unnorm_key or preset["unnorm_key"]
    ckpt_path = Path(checkpoint).expanduser()
    if not ckpt_path.is_dir():
        raise SystemExit(
            f"checkpoint 目录不存在：{ckpt_path}\n"
            f"可用 --preset 切换，或用 --checkpoint 指定本地路径。"
        )
    return str(ckpt_path), unnorm_key


def resolve_device_dtype(args):
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.dtype:
        dtype = DTYPE_MAP[args.dtype]
    else:
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    return device, dtype


def register_openvla_auto_classes():
    """把 prismatic 的 OpenVLA 类注册到 HF Auto 类（对齐 openvla_utils.get_vla）。

    使用 exist_ok=True，避免重复注册报错；这样加载本地 checkpoint 时使用的是
    repos/openvla 源码的模型实现，而不是 checkpoint 内的远程代码。

    注意：注册后必须以 trust_remote_code=False + local_files_only=True 加载。
    否则遇到 auto_map 指向跨仓库远程代码（如微调 checkpoint 的
    'openvla/openvla-7b--modeling_prismatic...'，且本地目录无 .py 文件）时，
    trust_remote_code=True 会去 huggingface.co 拉代码而卡死。
    """
    AutoConfig.register("openvla", OpenVLAConfig, exist_ok=True)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor, exist_ok=True)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor, exist_ok=True)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction, exist_ok=True)


def maybe_load_dataset_statistics(model, checkpoint):
    """若 checkpoint 内存在 dataset_statistics.json，则覆盖为 norm_stats（对齐 get_vla）。"""
    stats_path = os.path.join(checkpoint, "dataset_statistics.json")
    if os.path.isfile(stats_path):
        with open(stats_path, "r", encoding="utf-8") as handle:
            model.norm_stats = json.load(handle)
        log_step(f"已加载本地 dataset_statistics.json：{stats_path}")
    else:
        log_step(
            "checkpoint 内无 dataset_statistics.json；base 模型可忽略（norm_stats 来自 config）。"
        )


def load_image(image_path):
    if image_path is None:
        log_step("未提供 --image；使用 224x224 灰色占位图验证推理链路。")
        return Image.new("RGB", (224, 224), color=(128, 128, 128))
    path = Path(image_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"输入图片不存在：{path}")
    image = Image.open(path).convert("RGB")
    log_step(f"已加载图片：size={image.size}, mode={image.mode}")
    return image


def print_runtime_info(checkpoint, unnorm_key, device, dtype, attn):
    log_step(f"prismatic 源码来自：{OPENVLA_REPO}")
    log_step(f"OpenVLAForActionPrediction 模块：{OpenVLAForActionPrediction.__module__}")
    log_step(f"torch: {torch.__version__} | CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        log_step(f"GPU[0]: {torch.cuda.get_device_name(0)} (sm_{cap[0]}{cap[1]})")
    log_step(f"checkpoint: {checkpoint}")
    log_step(f"device={device} | dtype={dtype} | attn={attn}")
    log_step(f"unnorm_key={unnorm_key}")


def main():
    args = parse_args()
    checkpoint, unnorm_key = resolve_checkpoint_and_key(args)
    device, dtype = resolve_device_dtype(args)
    use_cuda = str(device).startswith("cuda")
    print_runtime_info(checkpoint, unnorm_key, device, dtype, args.attn_implementation)

    log_step("注册 OpenVLA 到 HF Auto 类（使用 repos/openvla 源码实现）...")
    register_openvla_auto_classes()

    log_step("加载 processor ...")
    processor = AutoProcessor.from_pretrained(
        checkpoint, trust_remote_code=False, local_files_only=True
    )
    log_step(f"processor: {type(processor).__name__}")

    model_kwargs = {
        "attn_implementation": args.attn_implementation,
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
        "trust_remote_code": False,
        "local_files_only": True,
    }
    if use_cuda:
        model_kwargs["device_map"] = {"": device}

    log_step("加载模型分片 ...")
    t0 = time.time()
    model = AutoModelForVision2Seq.from_pretrained(checkpoint, **model_kwargs)
    if not use_cuda:
        model = model.to(device)
    model.eval()
    log_step(f"模型加载完成：{type(model).__name__}，耗时 {time.time() - t0:.1f}s")
    log_step(f"model device={next(model.parameters()).device} dtype={next(model.parameters()).dtype}")

    maybe_load_dataset_statistics(model, checkpoint)

    # 校验 unnorm_key 是否存在于 norm_stats；不存在则打印可用 key 并报错。
    available_keys = list(getattr(model, "norm_stats", {}) or {})
    if unnorm_key not in available_keys:
        raise SystemExit(
            f"unnorm_key='{unnorm_key}' 不在 norm_stats 中。\n"
            f"可用 key（前 20 个）：{available_keys[:20]}\n"
            f"请用 --unnorm-key 指定正确的 key。"
        )
    log_step(f"norm_stats 共 {len(available_keys)} 个 key，已确认包含 '{unnorm_key}'。")

    image = load_image(args.image)
    prompt = f"In: What action should the robot take to {args.instruction.lower()}?\nOut:"
    log_step(f"prompt: {prompt!r}")

    inputs = processor(prompt, image).to(device, dtype=dtype)
    for name, value in inputs.items():
        if hasattr(value, "shape"):
            log_step(f"input {name}: shape={tuple(value.shape)}, dtype={value.dtype}")

    log_step("运行 model.predict_action(...) ...")
    t0 = time.time()
    with torch.inference_mode():
        action = model.predict_action(
            **inputs,
            unnorm_key=unnorm_key,
            do_sample=args.do_sample,
        )
    log_step(f"推理完成，耗时 {time.time() - t0:.2f}s")

    # 结果断言：OpenVLA 输出应为 7 维（6 DoF 位姿增量 + 1 夹爪）。
    action = np.asarray(action).reshape(-1)
    assert action.shape == (7,), f"期望 7D action，实际 shape={action.shape}"

    print("\n===== 验证结果 =====")
    print("Instruction :", args.instruction)
    print("Action (7D) :", np.round(action, 4).tolist())
    print("  xyz  dpos :", np.round(action[:3], 4).tolist())
    print("  rpy  drot :", np.round(action[3:6], 4).tolist())
    print("  gripper   :", round(float(action[6]), 4))
    print("repos/openvla 源码验证：通过（import + 注册 + 加载 + 单步推理均成功）")


if __name__ == "__main__":
    main()
