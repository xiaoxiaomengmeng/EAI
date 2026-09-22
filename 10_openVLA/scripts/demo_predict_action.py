import argparse
import sys
from pathlib import Path

from PIL import Image
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor

# 允许从 scripts/ 目录运行时导入 sim 包，复用统一的默认模型路径，避免多处硬编码
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim.config import DEFAULT_MODEL_PATH

DEFAULT_INSTRUCTION = "pick up the object"


def log_step(message):
    print(f"[OpenVLA Demo] {message}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Run a minimal OpenVLA action prediction demo.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Local OpenVLA checkpoint path or HF repo id.")
    parser.add_argument("--image", default=None, help="Input RGB image path. If omitted, uses a generated placeholder image.")
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION, help="Robot instruction text.")
    parser.add_argument("--unnorm-key", default="bridge_orig", help="Action un-normalization key in model norm_stats.")
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        choices=["eager", "sdpa", "flash_attention_2"],
        help="Attention backend. Use sdpa by default on RTX 5090; eager may hit a causal-mask length mismatch in OpenVLA generation.",
    )
    return parser.parse_args()


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
    log_step(f"Attention implementation: {args.attn_implementation}")
    log_step(f"Model path: {args.model_path}")
    log_step(f"Unnorm key: {args.unnorm_key}")


def main():
    args = parse_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
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
    if torch.cuda.is_available():
        model_kwargs["device_map"] = {"": device}

    log_step("Loading model checkpoint shards...")
    model = AutoModelForVision2Seq.from_pretrained(args.model_path, **model_kwargs)
    if not torch.cuda.is_available():
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
            do_sample=False,
        )

    log_step("Prediction finished.")
    print("Instruction:", args.instruction)
    print("Predicted action:", action)


if __name__ == "__main__":
    main()