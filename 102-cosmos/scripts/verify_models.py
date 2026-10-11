#!/usr/bin/env python3
"""
验证工作区 ~/ws/models/ 下 Cosmos 系列模型是否可用。

验证阶段：
  1. 软链接是否有效、目标目录是否存在
  2. 关键目录 / 配置文件是否齐全
  3. JSON 配置文件是否可正常解析
  4. 权重文件完整性（非零大小 + torch 可加载）
  5. 尝试用框架加载 pipeline / checkpoint（需要对应依赖，失败时给出警告而非中断）

用法：
  python3 scripts/verify_models.py            # 验证全部模型
  python3 scripts/verify_models.py --no-load  # 只做文件级检查，不加载模型
  python3 scripts/verify_models.py --cpu      # 强制使用 CPU 加载
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# --------------------------------------------------------------------------- #
# 颜色输出
# --------------------------------------------------------------------------- #
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"
RESET = "\033[0m"
BOLD = "\033[1m"


def ok(msg: str) -> None:
    print(f"{GREEN}  PASS{RESET}  {msg}")


def warn(msg: str) -> None:
    print(f"{YELLOW}  WARN{RESET}  {msg}")


def fail(msg: str) -> None:
    print(f"{RED}  FAIL{RESET}  {msg}")


def info(msg: str) -> None:
    print(f"{CYAN}  INFO{RESET}  {msg}")


def section(title: str) -> None:
    print(f"\n{BOLD}{CYAN}═══════════════════════════════════════════════════{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}═══════════════════════════════════════════════════{RESET}")


# --------------------------------------------------------------------------- #
# 工具函数
# --------------------------------------------------------------------------- #
def human_size(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:3.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PB"


def check_symlink(link_path: Path) -> bool:
    """检查软链接是否有效。"""
    if not link_path.is_symlink():
        fail(f"软链接不存在: {link_path}")
        return False
    target = link_path.resolve()
    if not target.exists():
        fail(f"软链接指向的目标不存在: {link_path} -> {target}")
        return False
    ok(f"软链接有效: {link_path.name} -> {target}")
    return True


def check_dir(path: Path, name: str) -> bool:
    if path.is_dir():
        ok(f"目录存在: {name} ({path})")
        return True
    fail(f"目录缺失: {name} ({path})")
    return False


def check_file(path: Path, name: str, min_size: int = 1) -> bool:
    if not path.is_file():
        fail(f"文件缺失: {name} ({path})")
        return False
    size = path.stat().st_size
    if size < min_size:
        fail(f"文件为空或过小 ({human_size(size)}): {name} ({path})")
        return False
    ok(f"文件完整: {name} ({human_size(size)})")
    return True


def check_json(path: Path, name: str) -> bool:
    if not path.is_file():
        fail(f"配置文件缺失: {name} ({path})")
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            json.load(f)
        ok(f"JSON 可解析: {name}")
        return True
    except Exception as e:
        fail(f"JSON 解析失败: {name} -> {e}")
        return False


def find_files(root: Path, suffixes: tuple[str, ...]):
    """递归查找指定后缀的文件。"""
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in suffixes:
            yield p


# --------------------------------------------------------------------------- #
# 模型定义
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT.parent / "models"  # 工作区统一权重目录 ~/ws/models


class ModelSpec:
    def __init__(self, name: str, required_dirs, required_files, json_files, weight_suffixes):
        self.name = name
        self.required_dirs = required_dirs
        self.required_files = required_files
        self.json_files = json_files
        self.weight_suffixes = weight_suffixes


SPECS = {
    "Cosmos-Predict2.5-2B": ModelSpec(
        name="Cosmos-Predict2.5-2B",
        required_dirs=["auto", "base", "robot"],
        required_files=["configuration.json", "tokenizer.pth", "README.md"],
        json_files=["configuration.json"],
        weight_suffixes=(".pt", ".distcp"),
    ),
    "Cosmos3-Edge-Policy-DROID": ModelSpec(
        name="Cosmos3-Edge-Policy-DROID",
        required_dirs=["scheduler", "text_tokenizer", "transformer", "vae", "vision_encoder"],
        required_files=["model_index.json", "config.json", "README.md"],
        json_files=["model_index.json", "config.json"],
        weight_suffixes=(".safetensors", ".pt", ".bin"),
    ),
}


# --------------------------------------------------------------------------- #
# 验证逻辑
# --------------------------------------------------------------------------- #
def verify_filesystem(model_dir: Path, spec: ModelSpec) -> bool:
    section(f"[{spec.name}] 阶段 1：路径与软链接")
    passed = check_symlink(MODELS_DIR / spec.name)
    if not passed:
        return False

    section(f"[{spec.name}] 阶段 2：关键目录与文件")
    all_ok = True
    for d in spec.required_dirs:
        all_ok &= check_dir(model_dir / d, d)
    for f in spec.required_files:
        all_ok &= check_file(model_dir / f, f)

    section(f"[{spec.name}] 阶段 3：配置文件解析")
    for jf in spec.json_files:
        all_ok &= check_json(model_dir / jf, jf)

    return all_ok


def verify_pt_zip(path: Path) -> bool:
    """对 torch 新版 zip 格式 .pt 做结构校验（只读头尾中央目录，不读权重数据）。"""
    import zipfile

    try:
        if not zipfile.is_zipfile(path):
            fail(f"不是合法的 torch/zip checkpoint（可能下载截断）: {path.name}")
            return False
        with zipfile.ZipFile(path) as z:
            n_entries = len(z.namelist())
        ok(f"  ↳ zip 结构完整，{n_entries} 个内部条目")
        return True
    except Exception as e:
        fail(f"checkpoint 校验异常: {path.name} -> {e}")
        return False


def verify_safetensors(path: Path) -> bool:
    """校验 safetensors：前 8 字节为小端 header 长度，其后为 JSON header。"""
    import struct

    try:
        with open(path, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            if header_len <= 2 or header_len > 100 * 1024 * 1024:
                fail(f"safetensors header 长度异常 ({header_len}): {path.name}")
                return False
            header = f.read(min(header_len, 16))
        if not header.startswith(b"{"):
            fail(f"safetensors header 不是 JSON: {path.name}")
            return False
        return True
    except Exception as e:
        fail(f"safetensors 校验异常: {path.name} -> {e}")
        return False


def verify_weights(model_dir: Path, spec: ModelSpec) -> bool:
    section(f"[{spec.name}] 阶段 4：权重文件完整性")
    weights = list(find_files(model_dir, spec.weight_suffixes))
    if not weights:
        fail("未找到任何权重文件")
        return False

    real_weights: list[Path] = []
    empty_distcp = 0
    all_ok = True
    total_size = 0

    # 先分类：.distcp 是 torch.distributed.checkpoint 的 per-rank 分片，
    # 多 rank 保存时不持有数据的 rank 会产生合法的 0 字节分片，不算损坏。
    for w in sorted(weights):
        size = w.stat().st_size
        if w.suffix == ".distcp" and size == 0:
            empty_distcp += 1
            continue
        if size == 0:
            fail(f"空权重文件（非 DCP 分片）: {w.relative_to(model_dir)}")
            all_ok = False
            continue
        real_weights.append(w)
        total_size += size

    info(
        f"权重文件共 {len(weights)} 个：{len(real_weights)} 个非空、"
        f"{empty_distcp} 个 DCP 空分片（多 rank 保存的正常产物，已跳过）"
    )

    for w in real_weights:
        rel = w.relative_to(model_dir)
        size_ok = True
        if w.suffix == ".pt":
            size_ok = verify_pt_zip(w)
        elif w.suffix == ".safetensors":
            size_ok = verify_safetensors(w)
        if size_ok:
            ok(f"{rel} ({human_size(w.stat().st_size)})")
        else:
            all_ok = False

    info(f"权重总大小: {human_size(total_size)}")
    return all_ok


def verify_load(model_dir: Path, spec: ModelSpec, use_cpu: bool) -> None:
    """尝试加载模型（依赖缺失时仅警告，不判失败）。"""
    section(f"[{spec.name}] 阶段 5：模型加载验证")

    if spec.name == "Cosmos3-Edge-Policy-DROID":
        _load_cosmos3(model_dir, use_cpu)
    elif spec.name == "Cosmos-Predict2.5-2B":
        _load_cosmos_predict(model_dir, use_cpu)


def _load_cosmos3(model_dir: Path, use_cpu: bool) -> None:
    """Cosmos3 是标准 diffusers 格式，尝试直接加载。"""
    try:
        import diffusers
    except ImportError:
        warn("未安装 diffusers，跳过加载验证")
        return

    info(f"diffusers 版本: {diffusers.__version__}")
    try:
        import torch
        dtype = torch.float32 if use_cpu else torch.bfloat16
        device = "cpu" if use_cpu else "cuda"
        info(f"使用 device={device}, dtype={dtype}")
        pipe = diffusers.Cosmos3OmniPipeline.from_pretrained(
            str(model_dir), torch_dtype=dtype
        )
        ok(f"Pipeline 加载成功: {type(pipe).__name__}")
        info(f"  transformer: {type(pipe.transformer).__name__}")
        info(f"  vae:         {type(pipe.vae).__name__}")
        info(f"  scheduler:   {type(pipe.scheduler).__name__}")
    except Exception as e:
        warn(f"加载失败（可能是版本不兼容或缺少 CUDA）: {type(e).__name__}: {e}")


def _load_cosmos_predict(model_dir: Path, use_cpu: bool) -> None:
    """Cosmos-Predict2.5-2B 是 NVIDIA 自定义格式，用 torch 尝试加载一份 checkpoint。"""
    try:
        import torch
    except ImportError:
        warn("未安装 torch，跳过加载验证")
        return

    # 挑一份较小的 checkpoint 试加载（policy/libero/model.pt）
    ckpt_candidates = [
        model_dir / "robot" / "policy" / "libero" / "model.pt",
        model_dir / "robot" / "policy" / "robocasa" / "model.pt",
        model_dir / "robot" / "action-cond" / "38c6c645-7d41-4560-8eeb-6f4ddc0e6574_ema_bf16.pt",
    ]
    ckpt = next((c for c in ckpt_candidates if c.is_file()), None)
    if ckpt is None:
        warn("未找到可加载的 checkpoint 文件")
        return

    info(f"尝试加载 checkpoint: {ckpt.relative_to(model_dir)}")
    try:
        # mmap=True 避免一次性读入全部内存
        ckpt_data = torch.load(str(ckpt), map_location="cpu", mmap=True, weights_only=False)
        if isinstance(ckpt_data, dict):
            n_keys = len(ckpt_data)
            sample_keys = list(ckpt_data.keys())[:5]
            ok(f"Checkpoint 加载成功，包含 {n_keys} 个键")
            info(f"  前 5 个键示例: {sample_keys}")
        else:
            ok(f"Checkpoint 加载成功，类型: {type(ckpt_data).__name__}")
    except Exception as e:
        warn(f"Checkpoint 加载失败: {type(e).__name__}: {e}")


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description="验证 102-cosmos 模型可用性")
    parser.add_argument("--no-load", action="store_true", help="只做文件级检查，不加载模型")
    parser.add_argument("--cpu", action="store_true", help="强制使用 CPU 加载模型")
    parser.add_argument(
        "--model",
        choices=list(SPECS.keys()) + ["all"],
        default="all",
        help="指定验证单个模型（默认全部）",
    )
    args = parser.parse_args()

    section("Cosmos 模型可用性验证")
    info(f"项目根目录: {PROJECT_ROOT}")
    info(f"模型目录:   {MODELS_DIR}")
    if not MODELS_DIR.is_dir():
        fail(f"models 目录不存在: {MODELS_DIR}")
        return 1

    targets = list(SPECS.keys()) if args.model == "all" else [args.model]
    overall_ok = True

    for name in targets:
        spec = SPECS[name]
        model_dir = MODELS_DIR / name

        fs_ok = verify_filesystem(model_dir, spec)
        w_ok = verify_weights(model_dir, spec)
        overall_ok &= fs_ok and w_ok

        if not args.no_load:
            verify_load(model_dir, spec, use_cpu=args.cpu)

    # 总结
    section("验证总结")
    if overall_ok:
        print(f"\n{GREEN}{BOLD}  ✓ 全部文件级检查通过{RESET}")
    else:
        print(f"\n{RED}{BOLD}  ✗ 存在失败项，请检查上方 FAIL 标记{RESET}")

    if args.no_load:
        info("已使用 --no-load，未执行模型加载验证")
    print()
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
