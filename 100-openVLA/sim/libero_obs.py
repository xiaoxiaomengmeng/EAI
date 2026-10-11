"""LIBERO 观测/动作契约复刻 (自建 sim 侧, 不依赖 repos/LIBERO 与 robosuite)。

对照 repos/openvla/experiments/robot/ 下三个文件逐行复刻 (M2 调研实测):

1. libero_utils.get_libero_image:
       img = obs["agentview_image"]; img = img[::-1, ::-1]   # 180° 旋转
       img = resize_image(img, (224, 224))                   # JPEG 往返 + lanczos3 antialias
2. openvla_utils.get_vla_action (center_crop=True, 微调带 image aug 时必须):
       crop_scale = 0.9 -> 边长比例 sqrt(0.9), 居中裁剪后 bilinear resize 回 (224, 224)
       注意: 面积比 0.9 对应边长比 sqrt(0.9)≈0.9487, 不是 0.9 (官方注释强调过)。
3. robot_utils.normalize_gripper_action / invert_gripper_action:
       夹爪维 [0,1] -> [-1,+1] (y = 2x-1), binarize 到 ±1, 再整体翻符号。

实现说明 (用 numpy/PIL 等价替换 TF 算子, 避免自建 sim 拉 tensorflow):
- resize_image 的 tf.image.encode_jpeg/decode + lanczos3 antialias ≈ PIL LANCZOS
  (JPEG 量化误差对策略输入可忽略);
- tf.image.crop_and_resize 默认 bilinear ≈ PIL BILINEAR。
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image

# OpenVLA 在 LIBERO 上的输入分辨率 (robot_utils.get_image_resize_size: openvla -> 224)。
LIBERO_RESIZE_SIZE = 224
# LIBERO 环境相机分辨率 (libero_utils.get_libero_env: camera_heights=resolution=256)。
LIBERO_CAMERA_SIZE = 256
# 微调带 image augmentation, 评测必须 center_crop=True, crop_scale=0.9。
LIBERO_CENTER_CROP_SCALE = 0.9
# 沉降阶段空动作 (libero_utils.get_libero_dummy_action)。
LIBERO_DUMMY_ACTION = [0, 0, 0, 0, 0, 0, -1]


def rotate180(rgb: np.ndarray) -> np.ndarray:
    """复刻 `img[::-1, ::-1]`: 上下+左右翻转 (即 180° 旋转), 对齐训练预处理。"""
    return rgb[::-1, ::-1]


def resize_lanczos(rgb: np.ndarray, size: int = LIBERO_RESIZE_SIZE) -> np.ndarray:
    """复刻 resize_image 的 lanczos3 antialias 缩放 (省略 JPEG 往返)。"""
    img = Image.fromarray(rgb.astype(np.uint8))
    return np.asarray(img.resize((size, size), Image.LANCZOS))


def center_crop(rgb: np.ndarray, crop_scale: float = LIBERO_CENTER_CROP_SCALE) -> np.ndarray:
    """复刻 openvla_utils.crop_and_resize: 居中裁出面积比 crop_scale 的区域再回原尺寸。

    边长比例 = sqrt(crop_scale); 裁剪后 bilinear resize 回 (224, 224)。
    """
    side_frac = math.sqrt(crop_scale)
    lo = (1.0 - side_frac) / 2.0
    hi = 1.0 - lo
    h, w = rgb.shape[:2]
    box = (round(lo * w), round(lo * h), round(hi * w), round(hi * h))
    img = Image.fromarray(rgb.astype(np.uint8)).crop(box)
    return np.asarray(img.resize((LIBERO_RESIZE_SIZE, LIBERO_RESIZE_SIZE), Image.BILINEAR))


def preprocess_libero_image(rgb256: np.ndarray, already_upright: bool = False) -> Image.Image:
    """完整复刻送模型前的观测预处理: 256 -> (180°旋转) -> lanczos 224 -> center_crop 0.9 -> 224。

    already_upright 语义:
    - False (默认): 输入是 LIBERO obs["agentview_image"] 那样的原始 offscreen 帧缓冲
      (倒置), 需要复刻官方的 180° 旋转;
    - True: 输入已是正立图 (本项目 mujoco.Renderer 的输出内部已处理帧缓冲方向,
      M1 渲染验证为正立), 此时再旋转反而OutOfDistribution, 必须跳过旋转。
    """
    if rgb256.shape[:2] != (LIBERO_CAMERA_SIZE, LIBERO_CAMERA_SIZE):
        raise ValueError(
            f"期望 {LIBERO_CAMERA_SIZE}x{LIBERO_CAMERA_SIZE} 的 agentview 渲染, 实际 {rgb256.shape[:2]}"
        )
    rgb = rgb256 if already_upright else rotate180(rgb256)
    rgb = resize_lanczos(rgb, LIBERO_RESIZE_SIZE)
    rgb = center_crop(rgb, LIBERO_CENTER_CROP_SCALE)
    return Image.fromarray(rgb).convert("RGB")


def normalize_gripper_action(action: np.ndarray, binarize: bool = True) -> np.ndarray:
    """复刻 robot_utils.normalize_gripper_action: 夹爪维 [0,1] -> [-1,+1], 可选二值化。"""
    action = np.asarray(action, dtype=np.float64).copy()
    action[..., -1] = 2.0 * action[..., -1] - 1.0
    if binarize:
        action[..., -1] = np.sign(action[..., -1])
    return action


def invert_gripper_action(action: np.ndarray) -> np.ndarray:
    """复刻 robot_utils.invert_gripper_action: 夹爪维翻符号 (RLDS 0=close/1=open 对齐)。"""
    action = np.asarray(action, dtype=np.float64).copy()
    action[..., -1] = action[..., -1] * -1.0
    return action


def libero_action_postprocess(action: np.ndarray) -> np.ndarray:
    """run_libero_eval 送 env.step 前的完整动作后处理: normalize(binarize) -> invert。"""
    return invert_gripper_action(normalize_gripper_action(action, binarize=True))
