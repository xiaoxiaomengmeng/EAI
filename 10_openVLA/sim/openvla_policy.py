from __future__ import annotations

import logging
import os
import time

import numpy as np
from PIL import Image

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


_logger = logging.getLogger(__name__)


def log(message: str):
    _logger.info(message)


class OpenVLAPolicy:
    """OpenVLA base 模型封装。

    这个类只处理模型侧逻辑：
    - 加载 AutoProcessor；
    - 加载 AutoModelForVision2Seq；
    - 将 RGB 图像和自然语言指令转换为 7DoF action。

    这里必须使用 openvla-7b base 模型已有的 unnorm_key，例如 bridge_orig，
    不要用 libero_spatial/libero_object 这类 LIBERO 微调 checkpoint 专属 key。
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

        log(f"Loading processor from {model_path}")
        processor_start = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.load_timing["processor_load_sec"] = time.perf_counter() - processor_start

        model_kwargs = {
            "attn_implementation": attn_implementation,
            "torch_dtype": self.dtype,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }
        if torch.cuda.is_available():
            model_kwargs["device_map"] = {"": self.device}

        log(f"Loading model with attn={attn_implementation}, dtype={self.dtype}, device={self.device}")
        model_start = time.perf_counter()
        self.model = AutoModelForVision2Seq.from_pretrained(model_path, **model_kwargs)
        self.load_timing["model_load_sec"] = time.perf_counter() - model_start

        prepare_start = time.perf_counter()
        if not torch.cuda.is_available():
            self.model = self.model.to(self.device)
        self.model.eval()
        self.load_timing["model_prepare_sec"] = time.perf_counter() - prepare_start
        self.load_timing["policy_load_sec"] = sum(self.load_timing.values())
        log(f"Loaded {type(self.model).__name__}, device={next(self.model.parameters()).device}")

    def predict(self, image: Image.Image, instruction: str) -> np.ndarray:
        """执行一次 OpenVLA 推理，返回 shape=(7,) 的连续 action。"""
        import torch

        prompt = f"What should the robot do to {instruction.lower()}? A:"
        preprocess_start = time.perf_counter()
        inputs = self.processor(prompt, image).to(self.device, dtype=self.dtype)
        preprocess_sec = time.perf_counter() - preprocess_start

        forward_start = time.perf_counter()
        with torch.inference_mode():
            action = self.model.predict_action(
                **inputs,
                unnorm_key=self.unnorm_key,
                do_sample=False,
            )
        model_forward_sec = time.perf_counter() - forward_start

        postprocess_start = time.perf_counter()
        action_array = np.asarray(action, dtype=np.float64)
        postprocess_sec = time.perf_counter() - postprocess_start
        self.last_timing = {
            "preprocess_sec": preprocess_sec,
            "model_forward_sec": model_forward_sec,
            "postprocess_sec": postprocess_sec,
        }
        return action_array


class DryRunPolicy:
    """不加载 7B 模型的假策略，用于验证 MuJoCo 环境、控制、视频输出。"""

    load_timing = {"policy_load_sec": 0.0}

    def __init__(self):
        self.last_timing = {"preprocess_sec": 0.0, "model_forward_sec": 0.0, "postprocess_sec": 0.0}

    def predict(self, image: Image.Image, instruction: str, step: int = 0) -> np.ndarray:
        predict_start = time.perf_counter()
        action = np.zeros(7, dtype=np.float64)
        action[0] = 0.025 * np.sin(step * 0.4)
        action[1] = 0.018 * np.cos(step * 0.3)
        action[2] = -0.012
        action[4] = 0.012 * np.sin(step * 0.2)
        action[6] = 1.0 if step % 8 < 4 else 0.0
        self.last_timing = {
            "preprocess_sec": 0.0,
            "model_forward_sec": time.perf_counter() - predict_start,
            "postprocess_sec": 0.0,
        }
        return action
