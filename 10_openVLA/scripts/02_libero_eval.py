"""LIBERO 官方评测薄封装（第二阶段：官方 benchmark 成功率评测）。

设计原则：**不重写评测循环**。本脚本只做两件事，其余全部复用官方
repos/openvla/experiments/robot/libero/run_libero_eval.py 的 eval_libero：

1. 环境/加载修复：
   - 把 repos/openvla 注入 sys.path，并用 prismatic 源码把 OpenVLA 注册到 HF Auto 类；
   - monkeypatch 官方 get_vla / get_processor，改为 trust_remote_code=False +
     local_files_only=True 离线加载（复用第一阶段修复，避免 ModelScope 微调 checkpoint
     的 auto_map 远程代码联网卡死）；attn 后端可选，默认 sdpa（官方硬编码 flash_attention_2）。
   - checkpoint 默认指向工作区统一软链 modules/。
2. 参数转译：把本脚本的简化参数翻译成官方 CLI 参数（sys.argv），再调用官方 eval_libero()，
   评测循环、成功率统计、rollout 视频、日志全部走官方逻辑。

前置条件（本脚本不代为执行）：
    # LIBERO editable 安装当前指向了失效路径，需先重装指向 repos/LIBERO：
    cd /home/creator/ws/10_openVLA
    source .venv/bin/activate
    pip install -e repos/LIBERO --no-deps
    python -c "import libero; print('libero OK')"

用法示例：
    cd /home/creator/ws/10_openVLA && source .venv/bin/activate
    # smoke test：只跑 1 条
    python scripts/02_libero_eval.py --num-trials-per-task 1 --run-id-note smoke-test
    # 指定任务套件与本地 checkpoint
    python scripts/02_libero_eval.py \
        --task-suite-name libero_spatial \
        --checkpoint /home/creator/ws/modules/openvla-7b-finetuned-libero-spatial \
        --num-trials-per-task 5
    # 遍历官方四个套件（每个套件须用自己的微调权重，unnorm_key 才对得上；
    # 目前 modules/ 下仅有 libero-spatial，其余套件需先下载对应权重）：
    python scripts/02_libero_eval.py --task-suite-name libero_spatial \
        --checkpoint /home/creator/ws/modules/openvla-7b-finetuned-libero-spatial --num-trials-per-task 50
    python scripts/02_libero_eval.py --task-suite-name libero_object \
        --checkpoint /home/creator/ws/modules/openvla-7b-finetuned-libero-object --num-trials-per-task 50
    python scripts/02_libero_eval.py --task-suite-name libero_goal \
        --checkpoint /home/creator/ws/modules/openvla-7b-finetuned-libero-goal --num-trials-per-task 50
    python scripts/02_libero_eval.py --task-suite-name libero_10 \
        --checkpoint /home/creator/ws/modules/openvla-7b-finetuned-libero-10 --num-trials-per-task 50
"""

import argparse
import json
import os
import sys
import types
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPENVLA_REPO = PROJECT_ROOT / "repos" / "openvla"
LIBERO_REPO = PROJECT_ROOT / "repos" / "LIBERO"
MODULES_DIR = PROJECT_ROOT.parent / "modules"
DEFAULT_LIBERO_CONFIG = PROJECT_ROOT / ".libero"

if not OPENVLA_REPO.is_dir():
    raise SystemExit(f"未找到 repos/openvla 源码目录：{OPENVLA_REPO}")

# 关键：注入 repos/openvla，使 prismatic 源码与 experiments.robot.* 可 import（无需 pip install -e）。
sys.path.insert(0, str(OPENVLA_REPO))

ATTN_CHOICES = ["eager", "sdpa", "flash_attention_2"]

# task_suite_name -> 本地 checkpoint 软链。当前仅下载了 libero_spatial 微调版。
PRESET_CHECKPOINTS = {
    "libero_spatial": MODULES_DIR / "openvla-7b-finetuned-libero-spatial",
}


def log_step(message):
    print(f"[LIBERO Eval] {message}", flush=True)


def ensure_wandb_importable(use_wandb):
    """官方 run_libero_eval.py 顶部无条件 `import wandb`。

    本环境的 wandb 与 protobuf 版本不匹配（wandb.proto.*_pb2 缺符号），直接 import 会崩，
    而重装 wandb/protobuf 有连带弄坏 tensorflow 的风险。由于评测默认不使用 wandb，
    这里在 import run_libero_eval 之前做兜底：能真正 import 就用真的；否则在关闭 wandb 时
    注入一个空 stub 模块，让那句 import 通过（wandb.init/log/save 均为 no-op，绝不会被调用）。
    """
    try:
        import wandb  # noqa: F401
        return
    except Exception as exc:  # 版本不匹配等各种 ImportError
        if use_wandb:
            raise SystemExit(
                "wandb 导入失败且已开启 --use-wandb：请先修复 wandb/protobuf 版本再重试。\n"
                f"原始错误：{exc}"
            ) from exc
        log_step(f"wandb 导入失败（{type(exc).__name__}），--use-wandb 已关闭，注入空 stub 跳过。")
        stub = types.ModuleType("wandb")

        def _noop(*args, **kwargs):
            return None

        stub.init = _noop
        stub.log = _noop
        stub.save = _noop
        stub.finish = _noop
        sys.modules["wandb"] = stub


def patch_torch_load_for_legacy():
    """PyTorch>=2.6 把 torch.load 的 weights_only 默认值改为 True。

    LIBERO benchmark/__init__.py 用 torch.load 读取 init_states（含 numpy 对象），
    在新版 PyTorch 下会报 UnpicklingError: Weights only load failed。这些是本地可信
    的仿真初始状态文件，这里把 torch.load 默认改回 weights_only=False（仅当调用方
    未显式传入时），避免修改 LIBERO 子模块源码。
    """
    original_load = torch.load
    if getattr(original_load, "_libero_patched", False):
        return

    def patched_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    patched_load._libero_patched = True
    torch.load = patched_load
    log_step("已将 torch.load 默认改为 weights_only=False（适配 LIBERO init_states 加载）。")


def patch_robosuite_mj_fullM():
    """兼容 robosuite 1.4.x 对旧版 mujoco mj_fullM 的调用。

    robosuite base_controller 调用旧签名 mj_fullM(m, dst, M_sparse)，而 mujoco>=3
    改为 mj_fullM(m, d, dst)（从 d.qM 读取稀疏惯性矩阵），直接跑会报
    TypeError: incompatible function arguments。

    不选择降级 mujoco：它被 mink 与 robosuite 共同依赖，降级会连带破坏第一阶段
    sim 包的 IK。注意：mujoco>=3 的 mj_fullM 不再从手工赋值的 d.qM 读取（实测会得
    零矩阵），因此不能用 scratch MjData 转发；这里直接按 mujoco 稀疏惯性布局
    （dof_Madr 首地址 + dof_parentid 上溯父 dof）用 numpy 展开成稠对称矩阵，仅需
    model 与 qM 向量（已对链式/分叉/free joint 三种情形与原生 mj_fullM 校对一致）。
    """
    import mujoco

    original = mujoco.mj_fullM
    if getattr(original, "_libero_compat", False):
        return

    def mj_fullM_compat(m, arg2, arg3):
        # 新签名 mj_fullM(m, d, dst)：arg2 是 MjData，直接透传。
        if isinstance(arg2, mujoco.MjData):
            return original(m, arg2, arg3)
        # 旧签名 mj_fullM(m, dst, M_sparse)：arg2=dst（nv×nv 稠矩阵），arg3=qM（稀疏向量）。
        dst = arg2
        qM = arg3
        madr = m.dof_Madr
        parent = m.dof_parentid
        dst[:] = 0.0
        for i in range(m.nv):
            adr = madr[i]
            j = i
            while j >= 0:
                value = qM[adr]
                dst[i, j] = value
                dst[j, i] = value
                adr += 1
                j = parent[j]

    mj_fullM_compat._libero_compat = True
    mujoco.mj_fullM = mj_fullM_compat
    log_step("已为 robosuite 1.4.x 注入 mj_fullM 兼容 shim（numpy 展开稀疏惯性矩阵，适配 mujoco>=3）。")


def parse_args():
    parser = argparse.ArgumentParser(description="LIBERO 官方评测薄封装（离线加载 + 本地 checkpoint）。")
    parser.add_argument(
        "--task-suite-name",
        default="libero_spatial",
        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
        help="LIBERO 任务套件；unnorm_key 会由官方脚本自动设为该值。",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="本地 LIBERO 微调 checkpoint 目录；省略时按 task-suite-name 取 modules/ 下软链。",
    )
    parser.add_argument("--num-trials-per-task", type=int, default=1, help="每个任务 rollout 次数（smoke test 用 1）。")
    parser.add_argument("--num-steps-wait", type=int, default=10, help="每个 episode 等待物体稳定的步数。")
    parser.add_argument(
        "--center-crop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否中心裁剪（微调带 image aug 时须为 True）。",
    )
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        choices=ATTN_CHOICES,
        help="注意力后端；官方硬编码 flash_attention_2，此处默认改用 sdpa。",
    )
    parser.add_argument("--use-wandb", action=argparse.BooleanOptionalAction, default=False, help="是否上报 wandb。")
    parser.add_argument("--run-id-note", default=None, help="附加到 run id 的备注。")
    parser.add_argument("--seed", type=int, default=7, help="随机种子。")
    return parser.parse_args()


def resolve_checkpoint(args):
    if args.checkpoint:
        ckpt = Path(args.checkpoint).expanduser()
    else:
        preset = PRESET_CHECKPOINTS.get(args.task_suite_name)
        if preset is None:
            raise SystemExit(
                f"task-suite-name={args.task_suite_name} 无内置本地 checkpoint，请用 --checkpoint 显式指定。"
            )
        ckpt = preset
    if not ckpt.is_dir():
        raise SystemExit(f"checkpoint 目录不存在：{ckpt}\n请确认已下载并在 modules/ 建立软链，或用 --checkpoint 指定。")
    return str(ckpt)


def register_openvla_auto_classes():
    """用 prismatic 源码把 OpenVLA 注册到 HF Auto 类（对齐 openvla_utils.get_vla）。"""
    from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

    AutoConfig.register("openvla", OpenVLAConfig, exist_ok=True)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor, exist_ok=True)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor, exist_ok=True)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction, exist_ok=True)


def build_offline_loaders(attn_implementation):
    """返回离线版 get_vla / get_processor，用于替换官方实现。

    与官方 get_vla 的唯一区别：trust_remote_code=False + local_files_only=True + attn 可配置，
    从而避免微调 checkpoint 的 auto_map 远程代码触发联网卡死。其余（bf16、device、
    dataset_statistics.json 覆盖 norm_stats）保持与官方一致。
    """
    from transformers import AutoModelForVision2Seq, AutoProcessor

    from experiments.robot.openvla_utils import DEVICE

    def offline_get_vla(cfg):
        log_step(f"离线加载 VLA：{cfg.pretrained_checkpoint}（attn={attn_implementation}）")
        vla = AutoModelForVision2Seq.from_pretrained(
            cfg.pretrained_checkpoint,
            attn_implementation=attn_implementation,
            torch_dtype=torch.bfloat16,
            load_in_8bit=cfg.load_in_8bit,
            load_in_4bit=cfg.load_in_4bit,
            low_cpu_mem_usage=True,
            trust_remote_code=False,
            local_files_only=True,
        )
        if not cfg.load_in_8bit and not cfg.load_in_4bit:
            vla = vla.to(DEVICE)
        vla.eval()

        stats_path = os.path.join(cfg.pretrained_checkpoint, "dataset_statistics.json")
        if os.path.isfile(stats_path):
            with open(stats_path, "r", encoding="utf-8") as handle:
                vla.norm_stats = json.load(handle)
            log_step(f"已加载 dataset_statistics.json：{stats_path}")
        else:
            log_step("警告：checkpoint 内无 dataset_statistics.json，predict_action 可能因缺 unnorm_key 报错。")
        return vla

    def offline_get_processor(cfg):
        return AutoProcessor.from_pretrained(
            cfg.pretrained_checkpoint, trust_remote_code=False, local_files_only=True
        )

    return offline_get_vla, offline_get_processor


def apply_monkeypatch(offline_get_vla, offline_get_processor):
    """把离线加载函数打到官方按名导入的各个命名空间上。

    robot_utils 通过 `from openvla_utils import get_vla` 绑定了自己的 get_vla，
    get_model 内部调用的是 robot_utils.get_vla；run_libero_eval 通过
    `from openvla_utils import get_processor` 绑定了自己的 get_processor。
    因此必须分别覆盖这些命名空间，仅改 openvla_utils 源模块不生效。
    """
    import experiments.robot.openvla_utils as openvla_utils
    import experiments.robot.robot_utils as robot_utils

    openvla_utils.get_vla = offline_get_vla
    openvla_utils.get_processor = offline_get_processor
    robot_utils.get_vla = offline_get_vla  # get_model 调用的是这个引用

    import experiments.robot.libero.run_libero_eval as run_libero_eval

    run_libero_eval.get_processor = offline_get_processor  # run_libero_eval 按名导入的引用
    return run_libero_eval


def main():
    args = parse_args()
    checkpoint = resolve_checkpoint(args)

    # LIBERO / MuJoCo 运行所需环境变量（不覆盖用户已显式设置的值）。
    os.environ.setdefault("LIBERO_CONFIG_PATH", str(DEFAULT_LIBERO_CONFIG))
    os.environ.setdefault("MUJOCO_GL", "egl")
    log_step(f"LIBERO_CONFIG_PATH={os.environ['LIBERO_CONFIG_PATH']} | MUJOCO_GL={os.environ['MUJOCO_GL']}")

    # 官方脚本用相对路径写日志/视频（./experiments/logs、./rollouts），
    # 切到 repos/openvla 以与官方行为一致。
    os.chdir(OPENVLA_REPO)

    log_step("注册 OpenVLA 到 HF Auto 类（prismatic 源码实现）...")
    register_openvla_auto_classes()

    ensure_wandb_importable(args.use_wandb)
    patch_torch_load_for_legacy()
    patch_robosuite_mj_fullM()

    log_step("构建离线加载器并 monkeypatch 官方 get_vla / get_processor ...")
    offline_get_vla, offline_get_processor = build_offline_loaders(args.attn_implementation)

    try:
        run_libero_eval = apply_monkeypatch(offline_get_vla, offline_get_processor)
    except ModuleNotFoundError as exc:
        if "libero" in str(exc):
            raise SystemExit(
                "import libero 失败：LIBERO 未正确安装（editable 可能指向失效路径）。\n"
                "请先执行：pip install -e repos/LIBERO --no-deps，再重跑本脚本。\n"
                f"原始错误：{exc}"
            ) from exc
        raise

    # 把简化参数翻译成官方 CLI 参数，交给 draccus 解析（评测逻辑完全走官方 eval_libero）。
    argv = [
        "run_libero_eval.py",
        "--model_family", "openvla",
        "--pretrained_checkpoint", checkpoint,
        "--task_suite_name", args.task_suite_name,
        "--center_crop", str(args.center_crop),
        "--num_trials_per_task", str(args.num_trials_per_task),
        "--num_steps_wait", str(args.num_steps_wait),
        "--use_wandb", str(args.use_wandb),
        "--seed", str(args.seed),
    ]
    if args.run_id_note:
        argv += ["--run_id_note", args.run_id_note]

    log_step("调用官方 eval_libero()，参数：")
    log_step(" ".join(argv[1:]))
    sys.argv = argv
    run_libero_eval.eval_libero()


if __name__ == "__main__":
    main()
