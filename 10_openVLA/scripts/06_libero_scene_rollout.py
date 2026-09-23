"""阶段⑥(M4) · 自建 libero_spatial 场景完整闭环 rollout (模型 -> 观测 -> OSC 等价控制)。

闭环链路 (全部自建, 不依赖 repos/LIBERO 与 robosuite):
    每个 policy step (= 1 个 robosuite control step = 25 物理步):
      ① 渲 256x256 agentview (LIBERO 相机分辨率/机位)
      ② sim.libero_obs 官方预处理 (正立 -> lanczos224 -> crop0.9 -> 224)
      ③ repos/openvla 源码离线加载的 libero-spatial 微调模型 -> raw 7D action
      ④ libero_obs.libero_action_postprocess (夹爪 normalize+binarize+invert)
      ⑤ sim.libero_controller.LiberoOscEquivalentController 执行 (OSC 等价)
    起始有 num_steps_wait=10 个 control step 的 dummy 沉降 (对齐官方)。

产物 (outputs/libero_scene_rollout/<ts>/):
    replay_224.mp4      官方风格回放 (送模型的预处理图序列, fps=20)
    agentview_640.mp4   人眼检查用高分辨率 agentview 回放
    rollout.json        逐步 raw/post action、EE/碗/盘位姿、夹爪、跟踪误差、goal 判定
    policy_input_*.png  (可选 --save-frames)

用法:
    cd /home/creator/ws/10_openVLA && source .venv/bin/activate
    python scripts/06_libero_scene_rollout.py --policy-steps 150
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
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
CHECKPOINT = PROJECT_ROOT.parent / "modules" / "openvla-7b-finetuned-libero-spatial"
UNNORM_KEY = "libero_spatial"

sys.path.insert(0, str(OPENVLA_REPO))
sys.path.insert(0, str(PROJECT_ROOT))

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig  # noqa: E402
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction  # noqa: E402
from prismatic.extern.hf.processing_prismatic import (  # noqa: E402
    PrismaticImageProcessor,
    PrismaticProcessor,
)

import mujoco  # noqa: E402
import imageio  # noqa: E402

from sim.libero_controller import LiberoOscEquivalentController  # noqa: E402
from sim.libero_obs import (  # noqa: E402
    LIBERO_DUMMY_ACTION,
    libero_action_postprocess,
    preprocess_libero_image,
)

NUM_STEPS_WAIT = 10  # 官方沉降 control step 数
VIDEO_FPS = 20  # 官方 control_freq, 回放 1 帧 = 1 个 policy step
GOAL_HOLD_STEPS = 10  # goal 判定连续成立则提前结束


def log_step(message: str):
    print(f"[M4] {message}", flush=True)


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
    text = bddl_path.read_text(encoding="utf-8")
    match = re.search(r"\(:language\s+(.+?)\)", text)
    if not match:
        raise SystemExit(f"未在 BDDL 中找到 :language 字段: {bddl_path}")
    return match.group(1).strip()


def load_source_policy(checkpoint: Path, attn_implementation: str):
    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    processor = AutoProcessor.from_pretrained(str(checkpoint), trust_remote_code=False, local_files_only=True)
    model_kwargs = {
        "attn_implementation": attn_implementation,
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
        "trust_remote_code": False,
        "local_files_only": True,
    }
    if torch.cuda.is_available():
        model_kwargs["device_map"] = {"": device}
    model = AutoModelForVision2Seq.from_pretrained(str(checkpoint), **model_kwargs)
    if not torch.cuda.is_available():
        model = model.to(device)
    model.eval()
    stats_path = checkpoint / "dataset_statistics.json"
    if stats_path.is_file():
        model.norm_stats = json.loads(stats_path.read_text(encoding="utf-8"))
    return model, processor, device, dtype


def goal_like(bowl: np.ndarray, plate: np.ndarray) -> bool:
    """(On bowl plate) 的定性代理: 碗在盘上方且水平对齐。"""
    horiz = float(np.linalg.norm(bowl[:2] - plate[:2]))
    dz = float(bowl[2] - plate[2])
    return horiz < 0.05 and 0.005 < dz < 0.08


def lifted(bowl: np.ndarray, table_top_z: float) -> bool:
    return float(bowl[2]) > table_top_z + 0.06


def main():
    parser = argparse.ArgumentParser(description="M4: 自建 libero_spatial 场景闭环 rollout。")
    parser.add_argument("--policy-steps", type=int, default=150)
    parser.add_argument("--attn-implementation", default="sdpa", choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--save-frames", action="store_true", help="保存每步送模型的 224 图。")
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / "outputs" / "libero_scene_rollout" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    instruction = load_instruction_from_bddl(BDDL_FILE)
    log_step(f"指令: {instruction} | policy_steps={args.policy_steps} | 产物目录: {out_dir}")

    # ---- 场景 + 控制器 ----
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home"))
    mujoco.mj_forward(model, data)
    ctrl = LiberoOscEquivalentController(model, data)
    bowl_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "akita_black_bowl_1")
    plate_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "plate_1")
    finger_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1")
    table_top_z = 0.8

    renderer256 = mujoco.Renderer(model, height=256, width=256)
    renderer640 = mujoco.Renderer(model, height=480, width=640)

    def render(camera: str, renderer) -> np.ndarray:
        renderer.update_scene(data, camera=camera)
        return renderer.render().copy()

    # ---- 沉降 (num_steps_wait=10) ----
    log_step(f"沉降 {NUM_STEPS_WAIT} 个 control step...")
    for _ in range(NUM_STEPS_WAIT):
        ctrl.set_action(np.array(LIBERO_DUMMY_ACTION, dtype=np.float64))
        ctrl.step_action()
    ctrl.reset_goal_to_current()

    # ---- 模型 ----
    register_openvla_auto_classes()
    vla, processor, device, dtype = load_source_policy(CHECKPOINT, args.attn_implementation)
    import torch

    prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"

    # ---- 闭环 ----
    replay224, replay640, records = [], [], []
    goal_hold = 0
    stop_reason = f"reached_policy_steps={args.policy_steps}"
    for step in range(args.policy_steps):
        rgb256 = render("agentview", renderer256)
        policy_image = preprocess_libero_image(rgb256, already_upright=True)
        replay224.append(np.asarray(policy_image))
        rgb640 = render("agentview", renderer640)
        replay640.append(rgb640)
        if args.save_frames:
            Image.fromarray(np.asarray(policy_image)).save(out_dir / f"policy_input_{step:04d}.png")

        t0 = time.perf_counter()
        inputs = processor(prompt, policy_image).to(device, dtype=dtype)
        with torch.inference_mode():
            raw = vla.predict_action(**inputs, unnorm_key=UNNORM_KEY, do_sample=False)
        forward_sec = time.perf_counter() - t0
        raw = np.asarray(raw, dtype=np.float64).reshape(-1)
        action = libero_action_postprocess(raw)

        ctrl.set_action(action)
        ctrl.step_action()

        bowl = data.xpos[bowl_id].copy()
        plate = data.xpos[plate_id].copy()
        rec = {
            "step": step,
            "raw_action": np.round(raw, 4).tolist(),
            "action": np.round(action, 4).tolist(),
            "ee_pos": np.round(ctrl.ee_pos, 4).tolist(),
            "bowl_pos": np.round(bowl, 4).tolist(),
            "plate_pos": np.round(plate, 4).tolist(),
            "finger1_qpos": round(float(data.qpos[finger_id]), 4),
            "tracking": {k: round(v, 4) for k, v in ctrl.tracking_error().items()},
            "forward_sec": round(forward_sec, 3),
            "lifted": lifted(bowl, table_top_z),
            "goal_like": goal_like(bowl, plate),
        }
        records.append(rec)

        if rec["goal_like"]:
            goal_hold += 1
            if goal_hold >= GOAL_HOLD_STEPS:
                stop_reason = f"goal_like_held_{GOAL_HOLD_STEPS}_steps_at_step_{step}"
                break
        else:
            goal_hold = 0
        if step % 10 == 0:
            log_step(
                f"step {step:3d} | raw={np.round(raw, 3).tolist()} | bowl_z={bowl[2]:.3f} "
                f"| lifted={rec['lifted']} | fwd={forward_sec:.2f}s"
            )

    # ---- 落盘 ----
    imageio.mimsave(out_dir / "replay_224.mp4", replay224, fps=VIDEO_FPS)
    imageio.mimsave(out_dir / "agentview_640.mp4", replay640, fps=VIDEO_FPS)
    bowl = data.xpos[bowl_id].copy()
    plate = data.xpos[plate_id].copy()
    summary = {
        "instruction": instruction,
        "stop_reason": stop_reason,
        "steps_run": len(records),
        "final_bowl_pos": bowl.tolist(),
        "final_plate_pos": plate.tolist(),
        "ever_lifted": bool(any(r["lifted"] for r in records)),
        "final_goal_like": goal_like(bowl, plate),
        "records": records,
    }
    (out_dir / "rollout.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    renderer256.close()
    renderer640.close()

    log_step(f"闭环结束: {stop_reason}")
    log_step(f"碗最终位姿={np.round(bowl, 4).tolist()} | 曾抓起={summary['ever_lifted']} | 终点 goal_like={summary['final_goal_like']}")
    log_step(f"回放视频: {out_dir / 'replay_224.mp4'} / {out_dir / 'agentview_640.mp4'}")
    log_step(f"日志: {out_dir / 'rollout.json'}")


if __name__ == "__main__":
    main()
