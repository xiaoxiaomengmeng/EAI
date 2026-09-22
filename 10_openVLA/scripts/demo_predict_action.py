import argparse
import sys
from pathlib import Path

from PIL import Image
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor

# 允许从 scripts/ 目录运行时导入 sim 包，复用统一的默认模型路径，避免多处硬编码
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim.config import DEFAULT_MODEL_PATH, PROJECT_ROOT

DEFAULT_INSTRUCTION = "pick up the object"

# 默认 YAML 配置：10_openVLA/config/demo_predict_action.yaml
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "demo_predict_action.yaml"

# 可选的注意力后端
ATTN_CHOICES = ["eager", "sdpa", "flash_attention_2"]

# 可选的计算精度（用于 YAML/CLI 覆盖时的字符串到 torch.dtype 映射）
DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}

# YAML 缺字段时的兜底默认值（与原有 CLI 默认保持一致）
FALLBACK_DEFAULTS = {
    "model_path": DEFAULT_MODEL_PATH,
    "image": None,
    "instruction": DEFAULT_INSTRUCTION,
    "unnorm_key": "bridge_orig",
    "attn_implementation": "sdpa",
    "do_sample": False,
    "device": None,   # None = 自动选择
    "dtype": None,    # None = 自动选择
}


def log_step(message):
    print(f"[OpenVLA Demo] {message}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Run a minimal OpenVLA action prediction demo.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="YAML config path. CLI args below override values from this file.")
    # 以下参数默认 None，表示"CLI 未指定"；最终取值优先级：CLI > YAML > 内置兜底默认
    parser.add_argument("--model-path", default=None, help="Local OpenVLA checkpoint path or HF repo id.")
    parser.add_argument("--image", default=None, help="Input RGB image path. If omitted, uses a generated placeholder image.")
    parser.add_argument("--instruction", default=None, help="Robot instruction text.")
    parser.add_argument("--unnorm-key", default=None, help="Action un-normalization key in model norm_stats.")
    parser.add_argument(
        "--attn-implementation",
        default=None,
        choices=ATTN_CHOICES,
        help="Attention backend. Use sdpa by default on RTX 5090; eager may hit a causal-mask length mismatch in OpenVLA generation.",
    )
    parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=None, help="Sample actions instead of greedy decoding.")
    return parser.parse_args()


def load_config_file(path):
    """读取 YAML 配置，返回 (dict, Path)。文件不存在时返回空 dict（全部走内置兜底默认）。"""
    cfg_path = Path(path).expanduser()
    if not cfg_path.is_file():
        return {}, cfg_path
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("读取 YAML 配置需要 PyYAML，请先安装：pip install pyyaml") from exc
    with cfg_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"配置文件格式错误（顶层应为键值映射）：{cfg_path}")
    return data, cfg_path


def resolve_config(args):
    """按 CLI > YAML > 内置默认 的优先级合并配置，并把结果写回 args。"""
    file_cfg, cfg_path = load_config_file(args.config)
    cli_values = {
        "model_path": args.model_path,
        "image": args.image,
        "instruction": args.instruction,
        "unnorm_key": args.unnorm_key,
        "attn_implementation": args.attn_implementation,
        "do_sample": args.do_sample,
    }
    resolved = {}
    for key, fallback in FALLBACK_DEFAULTS.items():
        cli_value = cli_values.get(key)
        if cli_value is not None:
            resolved[key] = cli_value
        elif file_cfg.get(key) is not None:
            resolved[key] = file_cfg[key]
        else:
            resolved[key] = fallback

    if resolved["attn_implementation"] not in ATTN_CHOICES:
        raise SystemExit(f"attn_implementation 非法：{resolved['attn_implementation']}（可选 {ATTN_CHOICES}）")

    unknown = sorted(set(file_cfg) - set(FALLBACK_DEFAULTS))
    if unknown:
        log_step(f"忽略配置文件中的未知字段：{unknown}")

    for key, value in resolved.items():
        setattr(args, key, value)
    args.config_path = cfg_path
    return args


def resolve_device_dtype(args):
    """根据配置解析 device / dtype；留空则自动选择。"""
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.dtype:
        dtype = DTYPE_MAP.get(str(args.dtype).lower())
        if dtype is None:
            raise SystemExit(f"dtype 非法：{args.dtype}（可选 {sorted(DTYPE_MAP)}）")
    else:
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    return device, dtype


def load_image(image_path):
    if image_path is None:
        log_step("No --image provided; using a generated 224x224 placeholder image to verify the inference pipeline.")
        return Image.new("RGB", (224, 224), color=(128, 128, 128))

    path = Path(image_path).expanduser()
    log_step(f"Loading image: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Input image not found: {path}")
    image = Image.open(path).convert("RGB")
    log_step(f"Image loaded: size={image.size}, mode={image.mode}")
    return image


def print_runtime_info(args, device, dtype):
    log_step(f"torch version: {torch.__version__}")
    log_step(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log_step(f"CUDA version built with torch: {torch.version.cuda}")
        log_step(f"GPU count: {torch.cuda.device_count()}")
        log_step(f"GPU[0]: {torch.cuda.get_device_name(0)}")
        capability = torch.cuda.get_device_capability(0)
        log_step(f"GPU[0] capability: sm_{capability[0]}{capability[1]}")
    log_step(f"Selected device: {device}")
    log_step(f"Selected dtype: {dtype}")
    log_step(f"Config file: {args.config_path} (loaded={args.config_path.is_file()})")
    log_step(f"Attention implementation: {args.attn_implementation}")
    log_step(f"Model path: {args.model_path}")
    log_step(f"Unnorm key: {args.unnorm_key}")
    log_step(f"Instruction: {args.instruction}")
    log_step(f"do_sample: {args.do_sample}")


def main():
    args = resolve_config(parse_args())
    device, dtype = resolve_device_dtype(args)
    use_cuda = str(device).startswith("cuda")
    print_runtime_info(args, device, dtype)

    log_step("Loading processor...")
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    log_step(f"Processor loaded: {type(processor).__name__}")

    model_kwargs = {
        "attn_implementation": args.attn_implementation,
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    if use_cuda:
        model_kwargs["device_map"] = {"": device}

    log_step("Loading model checkpoint shards...")
    model = AutoModelForVision2Seq.from_pretrained(args.model_path, **model_kwargs)
    if not use_cuda:
        model = model.to(device)
    model.eval()
    log_step(f"Model loaded: {type(model).__name__}")
    log_step(f"Model device: {next(model.parameters()).device}")
    log_step(f"Model dtype: {next(model.parameters()).dtype}")

    image = load_image(args.image)
    prompt = f"In: What action should the robot take to {args.instruction.lower()}?\nOut:"
    log_step(f"Instruction: {args.instruction}")
    log_step(f"Prompt: {prompt!r}")

    log_step("Preparing processor inputs...")
    inputs = processor(prompt, image).to(device, dtype=dtype)
    for name, value in inputs.items():
        if hasattr(value, "shape"):
            log_step(f"Input tensor {name}: shape={tuple(value.shape)}, dtype={value.dtype}, device={value.device}")

    log_step("Running model.predict_action(...)")
    with torch.inference_mode():
        action = model.predict_action(
            **inputs,
            unnorm_key=args.unnorm_key,
            do_sample=args.do_sample,
        )

    log_step("Prediction finished.")
    print("Instruction:", args.instruction)
    print("Predicted action:", action)


if __name__ == "__main__":
    main()