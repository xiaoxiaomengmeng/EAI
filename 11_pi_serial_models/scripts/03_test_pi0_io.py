#!/usr/bin/env python3
"""独立 Pi0 输入/输出 smoke test。

这个脚本只依赖当前目录下的 models/ 链接和通用 Python 包，不读取其他仓库源码。
它的目标不是完成真实 Pi0 模型前向，而是先把 Pi0 / LIBERO 风格的输入输出
接口固定下来，方便后续接入 openpi 官方代码时对齐字段、shape 和 dtype。

默认测试 pi0_libero_base，因为它更适合后续接 Franka Panda + LIBERO + MuJoCo。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


PI0_IMAGE_SIZE = 320
PI0_ACTION_DIM = 32
PI0_ACTION_HORIZON = 50
DEFAULT_PROMPT = "pick up the red cube"


def resize_nearest(image: np.ndarray, size: int = PI0_IMAGE_SIZE) -> np.ndarray:
    """用 numpy 做最小 nearest resize，避免强依赖 PIL / cv2。"""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image must be HWC RGB, got shape={image.shape}")

    height, width, _ = image.shape
    y_idx = np.linspace(0, height - 1, size).round().astype(np.int64)
    x_idx = np.linspace(0, width - 1, size).round().astype(np.int64)
    return image[y_idx][:, x_idx].astype(np.uint8, copy=False)


def make_uint8_image(height: int, width: int, seed: int) -> np.ndarray:
    """构造可复现 RGB 图像，模拟相机输入。"""
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    return image


def pad_state(state: np.ndarray, action_dim: int = PI0_ACTION_DIM) -> np.ndarray:
    """Pi0 配置里的 state/action 维度通常 pad 到 action_dim=32。"""
    state = np.asarray(state, dtype=np.float32).reshape(-1)
    if state.size > action_dim:
        raise ValueError(f"state dim {state.size} > action_dim {action_dim}")

    padded = np.zeros(action_dim, dtype=np.float32)
    padded[: state.size] = state
    return padded


def build_raw_libero_observation(prompt: str) -> dict[str, Any]:
    """构造 LIBERO 风格原始输入。"""
    return {
        "observation/state": np.linspace(-1.0, 1.0, 8, dtype=np.float32),
        "observation/image": make_uint8_image(224, 224, seed=0),
        "observation/left_wrist": make_uint8_image(224, 224, seed=1),
        "observation/right_wrist": make_uint8_image(224, 224, seed=2),
        "prompt": prompt,
    }


def to_pi0_model_input(raw: dict[str, Any]) -> dict[str, Any]:
    """把 LIBERO 风格输入转换成 Pi0 模型侧常用字段。"""
    base = resize_nearest(np.asarray(raw["observation/image"]))
    left_wrist = resize_nearest(np.asarray(raw["observation/left_wrist"]))
    right_wrist = resize_nearest(np.asarray(raw["observation/right_wrist"]))
    state = pad_state(np.asarray(raw["observation/state"]))

    return {
        "state": state,
        "image": {
            "base_0_rgb": base,
            "left_wrist_0_rgb": left_wrist,
            "right_wrist_0_rgb": right_wrist,
        },
        "image_mask": {
            "base_0_rgb": np.bool_(True),
            "left_wrist_0_rgb": np.bool_(True),
            "right_wrist_0_rgb": np.bool_(True),
        },
        "prompt": str(raw["prompt"]),
    }


def make_dummy_action_chunk(
    horizon: int = PI0_ACTION_HORIZON,
    action_dim: int = PI0_ACTION_DIM,
) -> np.ndarray:
    """构造一个确定性的假动作块，用来验证输出接口。"""
    actions = np.zeros((horizon, action_dim), dtype=np.float32)
    actions[:, 0] = np.linspace(0.0, 0.02, horizon, dtype=np.float32)
    actions[:, 1] = np.linspace(0.0, -0.02, horizon, dtype=np.float32)
    actions[:, 2] = np.linspace(0.0, 0.01, horizon, dtype=np.float32)
    actions[:, 6] = np.linspace(1.0, -1.0, horizon, dtype=np.float32)
    return actions


def summarize_array(array: np.ndarray) -> dict[str, Any]:
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "min": float(array.min()) if array.size else None,
        "max": float(array.max()) if array.size else None,
    }


def summarize_model_input(model_input: dict[str, Any]) -> dict[str, Any]:
    return {
        "prompt": model_input["prompt"],
        "state": summarize_array(model_input["state"]),
        "image": {
            name: summarize_array(value)
            for name, value in model_input["image"].items()
        },
        "image_mask": {
            name: bool(value)
            for name, value in model_input["image_mask"].items()
        },
    }


def resolve_model_dir(root: Path, model_name: str, explicit_model_dir: str | None) -> Path:
    if explicit_model_dir:
        return Path(explicit_model_dir).expanduser().resolve()
    return (root / "models" / model_name).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Pi0-style input/output shapes without external repo code.")
    parser.add_argument("--model-name", default="pi0_libero_base", choices=["pi0_base", "pi0_libero_base"])
    parser.add_argument("--model-dir", default=None, help="Optional model directory override.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--dump-json", default=None, help="Write summary JSON to this path.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent  # 本脚本位于 scripts/，项目根在上一级
    model_dir = resolve_model_dir(root, args.model_name, args.model_dir)

    print("=== Pi0 IO Smoke Test ===")
    print(f"workspace: {root}")
    print(f"model_name: {args.model_name}")
    print(f"model_dir: {model_dir}")
    print(f"model_dir_exists: {model_dir.exists()}")

    if not model_dir.exists():
        raise SystemExit(f"模型目录不存在: {model_dir}")

    raw = build_raw_libero_observation(args.prompt)
    model_input = to_pi0_model_input(raw)
    actions = make_dummy_action_chunk()

    summary = {
        "model_name": args.model_name,
        "model_dir": str(model_dir),
        "raw_keys": sorted(raw.keys()),
        "model_input": summarize_model_input(model_input),
        "expected_output": {
            "actions": summarize_array(actions),
            "first_action_head": actions[0, :8].round(6).tolist(),
            "last_action_head": actions[-1, :8].round(6).tolist(),
        },
        "note": "这是独立 IO smoke test；真实模型前向需后续接入 openpi policy/checkpoint loader。",
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.dump_json:
        output_path = Path(args.dump_json).expanduser()
        if not output_path.is_absolute():
            output_path = root / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"summary_json: {output_path}")


if __name__ == "__main__":
    main()
