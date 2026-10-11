"""阶段④(M2) · 自建 libero_spatial 场景单步打通: 观测预处理复刻 + 微调模型源码加载 + 7D action 验证。

阶梯位置:
- 01/02/03: 源码加载验证 / LIBERO 官方评测 / 自定义场景闭环;
- 04(本脚本, M2): 在 M1 手搭的 sim/scenes/libero_spatial_scene.xml 上,
  ① 渲 256x256 agentview; ② 用 sim/libero_obs.py 复刻 LIBERO 官方观测预处理
  (正立渲染 -> lanczos 224 -> center_crop 0.9 -> 224; 官方额外的 180°旋转是补
  robosuite 帧缓冲倒置, 本项目渲染已正立故跳过); ③ 以 repos/openvla 源码离线加载
  libero-spatial 微调 checkpoint; ④ 单步推理, 验证 7D action 的取值空间与语义。

本脚本只验证「观测->模型->action」单步链路, 不闭环执行动作 (执行保真属 M3/M4)。

用法:
    cd /home/creator/ws/100-openVLA && source .venv/bin/activate
    python scripts/04_libero_scene_single_step.py
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPENVLA_REPO = PROJECT_ROOT / "repos" / "openvla"
SCENE_XML = PROJECT_ROOT / "sim" / "scenes" / "libero_spatial_scene.xml"
BDDL_FILE = (
    PROJECT_ROOT
    / "repos"
    / "LIBERO"
    / "libero"
    / "libero"
    / "bddl_files"
    / "libero_spatial"
    / "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate.bddl"
)
CHECKPOINT = PROJECT_ROOT.parent / "models" / "openvla-7b-finetuned-libero-spatial"
UNNORM_KEY = "libero_spatial"
OUT_DIR = PROJECT_ROOT / "outputs" / "libero_scene_single_step"

# osc_pose.json 的 output_max: 归一化动作 [-1,1] 到物理增量的缩放 (M2 仅用于解读 action 语义)。
OSC_OUTPUT_MAX = np.array([0.05, 0.05, 0.05, 0.5, 0.5, 0.5])

sys.path.insert(0, str(OPENVLA_REPO))
sys.path.insert(0, str(PROJECT_ROOT))

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig  # noqa: E402
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction  # noqa: E402
from prismatic.extern.hf.processing_prismatic import (  # noqa: E402
    PrismaticImageProcessor,
    PrismaticProcessor,
)

from sim.libero_obs import (  # noqa: E402
    LIBERO_DUMMY_ACTION,
    libero_action_postprocess,
    preprocess_libero_image,
)


def log_step(message: str):
    print(f"[M2] {message}", flush=True)


def register_openvla_auto_classes():
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


def load_instruction_from_bddl(bddl_path: Path) -> str:
    """从 BDDL 的 (:language ...) 字段取任务指令 (即官方评测的 task.language)。"""
    text = bddl_path.read_text(encoding="utf-8")
    match = re.search(r"\(:language\s+(.+?)\)", text)
    if not match:
        raise SystemExit(f"未在 BDDL 中找到 :language 字段: {bddl_path}")
    return match.group(1).strip()


def load_source_policy(checkpoint: Path, attn_implementation: str):
    """repos/openvla 源码离线加载微调 checkpoint (对齐 01/03 的加载路径)。"""
    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    log_step(f"源码加载 processor: {checkpoint}")
    processor = AutoProcessor.from_pretrained(
        str(checkpoint), trust_remote_code=False, local_files_only=True
    )

    model_kwargs = {
        "attn_implementation": attn_implementation,
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
        "trust_remote_code": False,
        "local_files_only": True,
    }
    if torch.cuda.is_available():
        model_kwargs["device_map"] = {"": device}
    log_step(f"源码加载模型: attn={attn_implementation}, dtype={dtype}, device={device}")
    model = AutoModelForVision2Seq.from_pretrained(str(checkpoint), **model_kwargs)
    if not torch.cuda.is_available():
        model = model.to(device)
    model.eval()

    stats_path = checkpoint / "dataset_statistics.json"
    if stats_path.is_file():
        model.norm_stats = json.loads(stats_path.read_text(encoding="utf-8"))
        log_step(f"已加载 dataset_statistics.json: {stats_path}")
    available = list(getattr(model, "norm_stats", {}) or {})
    if UNNORM_KEY not in available:
        raise SystemExit(f"unnorm_key='{UNNORM_KEY}' 不在 norm_stats: {available[:20]}")
    return model, processor, device, dtype


def render_agentview_256() -> np.ndarray:
    """编译自建场景、home keyframe + num_steps_wait=10 沉降, 渲 256x256 agentview。"""
    import mujoco

    log_step(f"编译场景: {SCENE_XML}")
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, key_id)

    # 沉降: 对齐 num_steps_wait=10 (官方此阶段送 dummy action, 机器人不动)。
    log_step(f"沉降 {10} 步 (dummy action={LIBERO_DUMMY_ACTION}, 位置执行器保持 home)...")
    for _ in range(10):
        mujoco.mj_step(model, data)

    renderer = mujoco.Renderer(model, height=256, width=256)
    renderer.update_scene(data, camera="agentview")
    rgb = renderer.render().copy()
    renderer.close()
    return rgb


def main():
    parser = argparse.ArgumentParser(description="M2: 自建 libero_spatial 场景单步 7D action 验证。")
    parser.add_argument("--attn-implementation", default="sdpa", choices=["eager", "sdpa", "flash_attention_2"])
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    instruction = load_instruction_from_bddl(BDDL_FILE)
    log_step(f"BDDL 指令 (task.language): {instruction}")

    # ① 渲染 + ② 观测预处理复刻
    rgb256 = render_agentview_256()
    Image.fromarray(rgb256).save(OUT_DIR / "raw_agentview_256.png")
    # 本项目 mujoco.Renderer 输出已正立 (官方 [::-1,::-1] 是补 robosuite 帧缓冲倒置), 故 already_upright=True。
    policy_image = preprocess_libero_image(rgb256, already_upright=True)
    policy_image.save(OUT_DIR / "preprocessed_224.png")
    log_step(f"观测预处理完成: 256(正立) -> lanczos224 -> crop0.9 -> 224, 存于 {OUT_DIR}")

    # ③ 微调模型源码加载
    register_openvla_auto_classes()
    model, processor, device, dtype = load_source_policy(CHECKPOINT, args.attn_implementation)

    # ④ 单步推理 (官方 prompt 模板)
    prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
    inputs = processor(prompt, policy_image).to(device, dtype=dtype)
    t0 = time.perf_counter()
    import torch

    with torch.inference_mode():
        raw = model.predict_action(**inputs, unnorm_key=UNNORM_KEY, do_sample=False)
    forward_sec = time.perf_counter() - t0
    raw = np.asarray(raw, dtype=np.float64).reshape(-1)
    assert raw.shape == (7,), f"期望 7D action, 实际 {raw.shape}"

    final = libero_action_postprocess(raw)
    physical = np.concatenate([raw[:3] * OSC_OUTPUT_MAX[:3], raw[3:6] * OSC_OUTPUT_MAX[3:]])

    log_step(f"forward {forward_sec * 1000:.1f} ms")
    log_step(f"raw 7D action   : {np.round(raw, 4).tolist()}")
    log_step(f"物理增量解读(m/rad): {np.round(physical, 4).tolist()}")
    log_step(f"送 env 前后处理  : {np.round(final, 4).tolist()}")

    # 取值空间校验: LIBERO 归一化动作前 6 维应在 [-1,1]; 后处理夹爪应二值 ±1。
    ok_range = bool(np.all(np.abs(raw[:6]) <= 1.0 + 1e-6))
    ok_grip = bool(final[6] in (-1.0, 1.0))
    log_step(f"校验 |raw[:6]|<=1: {ok_range} | 后处理 gripper∈{{-1,+1}}: {ok_grip}")

    payload = {
        "instruction": instruction,
        "unnorm_key": UNNORM_KEY,
        "raw_action": raw.tolist(),
        "physical_delta_m_rad": physical.tolist(),
        "postprocessed_action": final.tolist(),
        "checks": {"abs_raw_pos_ori_le_1": ok_range, "gripper_binary": ok_grip},
        "forward_sec": forward_sec,
    }
    (OUT_DIR / "action.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log_step(f"结果落盘: {OUT_DIR / 'action.json'}")
    log_step("M2 单步链路打通。下一步 M3: 把 final action 接到末端增量控制 (近似 OSC) 上执行。")


if __name__ == "__main__":
    main()
