#!/usr/bin/env python3
"""验证 pico2g1 录制产物是否构成有效的 LeRobot v2.1 数据集。

录制目录三种状态（对应 sim_scenes/lerobot_recording.py 的导出流程）：
  完成     顶层 data/ + videos/ + meta/ 齐全          → 完整 v2.1 校验
  导出中断 raw/ + .export/（缺 meta）                  → raw 交叉诊断 + 暂存产物校验 + 恢复命令
  未导出   仅 raw/                                     → raw 诊断 + 导出命令提示

用法：
  04_validate_recording.py [录制目录或名称]      # 名称自动在 pico2g1/recordings/ 下查找
  04_validate_recording.py --list                # 列出 recordings/ 下的录制
  04_validate_recording.py <dir> --skip-video    # 跳过视频逐帧计数（快，仅查元数据）
  04_validate_recording.py <dir> --deep PYTHON   # 附加 lerobot 库深度加载（如 ~/miniconda3/envs/cosmos/bin/python）

依赖：pyarrow（建议用 g1_ik/.venv 运行）、系统 ffprobe；--deep 的目标解释器需装有 lerobot。
退出码：0=导出完成且全部通过；1=存在失败项或导出未完成；2=用法或目录无效。
"""

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path

try:
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    sys.exit('需要 numpy+pyarrow：请用 g1_ik/.venv 运行（pico2g1/g1_ik/.venv/bin/python 本脚本）')

SCRIPTS_DIR = Path(__file__).resolve().parent
PICO2G1 = SCRIPTS_DIR.parent / 'pico2g1'
RECORDINGS = PICO2G1 / 'recordings'
PROJECT_VENV = PICO2G1 / 'g1_ik' / '.venv'

# 与 sim_scenes/lerobot_recording.py 对齐的发布终态要求
META_REQUIRED = ('info.json', 'tasks.jsonl', 'episodes.jsonl', 'episodes_stats.jsonl')
SCALARS = {'timestamp': 'float32', 'frame_index': 'int64',
           'episode_index': 'int64', 'index': 'int64', 'task_index': 'int64'}

report_items = []


def ok(msg):
    report_items.append(('✅', msg))
    print(f'✅ {msg}')


def warn(msg):
    report_items.append(('⚠️', msg))
    print(f'⚠️ {msg}')


def fail(msg):
    report_items.append(('❌', msg))
    print(f'❌ {msg}')


def info(msg):
    report_items.append(('  ', msg))
    print(msg)


def probe_video(path, count_frames):
    """ffprobe 视频流，返回 (属性 dict, 帧数或 None)。"""
    cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
           '-show_entries', 'stream=codec_name,pix_fmt,width,height,r_frame_rate']
    if count_frames:
        cmd += ['-count_frames', '-show_entries', 'stream=nb_read_frames']
    cmd += ['-of', 'json', str(path)]
    try:
        out = subprocess.run(cmd, capture_output=True, check=True, text=True).stdout
    except (subprocess.CalledProcessError, OSError):
        return None, None
    streams = json.loads(out).get('streams', [])
    if len(streams) != 1:
        return None, None
    s = streams[0]
    return s, s.get('nb_read_frames') and int(s['nb_read_frames'])


def parse_state(d):
    if (d / 'meta' / 'info.json').exists():
        return 'complete'
    if (d / 'raw' / 'capture.json').exists():
        return 'interrupted' if (d / '.export').exists() else 'raw-only'
    return 'invalid'


# ---------------------------------------------------------------- raw 侧诊断

def check_raw(d):
    cfg_path = d / 'raw' / 'capture.json'
    try:
        cfg = json.loads(cfg_path.read_text())
    except (OSError, ValueError) as exc:
        fail(f'raw/capture.json 无法解析: {exc}')
        return None
    nq, names, sfps = cfg.get('nq'), cfg.get('joint_names'), cfg.get('source_fps')
    fmt = cfg.get('format')
    if fmt != 1:
        fail(f'raw/capture.json format={fmt}，应为 1')
    else:
        ok(f'raw/capture.json: format=1, nq={nq}, 关节 {len(names)} 个, source_fps={sfps}')

    bin_path = d / 'raw' / 'frames.bin'
    # 帧记录 = (nq + len(names)) 个 float64：qpos 在前、action 在后
    width = (nq + len(names)) if nq and names else None
    if not width or not bin_path.exists():
        fail('raw/frames.bin 缺失或 capture.json 缺 nq/joint_names')
        return None
    size = bin_path.stat().st_size
    if size % (8 * width):
        fail(f'raw/frames.bin 大小 {size} 不是 {8*width} 的整数倍，尾部记录截断')
        return None
    raw_frames = size // (8 * width)
    expected = math.ceil(raw_frames * 30 / sfps) if sfps else None
    ok(f'raw/frames.bin: {raw_frames} 帧 @ {sfps}fps → 期望导出 {expected} 帧 @30fps'
       + ('' if cfg.get('task') else '') + (f'，任务「{cfg["task"]}」' if cfg.get('task') else ''))
    if not raw_frames:
        fail('raw/frames.bin 为空（录制 0 帧）')
    return {'raw_frames': raw_frames, 'expected': expected, 'config': cfg}


# ------------------------------------------------- LeRobot v2.1 数据集校验

def check_meta(base, require_meta=True):
    """require_meta=False（暂存态）时，meta 缺失记为警告而非失败——中断态的预期症状。"""
    for name in META_REQUIRED:
        p = base / 'meta' / name
        if not p.exists():
            (fail if require_meta else warn)(f'meta/{name} 缺失')
            continue
        try:
            if name.endswith('.jsonl'):
                [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
            else:
                json.loads(p.read_text())
            ok(f'meta/{name} 存在且可解析')
        except ValueError as exc:
            fail(f'meta/{name} 存在但 JSON 解析失败: {exc}')
    if (base / 'meta' / 'stats.json').exists():
        ok('meta/stats.json 存在（本工程附加，非 v2.1 规范必需）')


def check_info(base):
    path = base / 'meta' / 'info.json'
    try:
        info_json = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        fail(f'meta/info.json 无法解析: {exc}')
        return None

    def field(name, expect=None, cast=None):
        v = info_json.get(name)
        if v is None:
            fail(f'info.json 缺字段 {name}')
            return None
        if expect is not None and v != expect:
            fail(f'info.json {name}={v}，应为 {expect}')
            return v
        if cast and not isinstance(v, cast):
            fail(f'info.json {name}={v!r}，应为 {cast.__name__}')
            return v
        return v

    ver = field('codebase_version', expect='v2.1')
    fps = field('fps', cast=int)
    eps = field('total_episodes', cast=int)
    frames = field('total_frames', cast=int)
    field('total_videos', cast=int)
    field('total_chunks', cast=int)
    field('chunks_size', cast=int)
    if 'splits' not in info_json or 'train' not in info_json.get('splits', {}):
        fail("info.json splits 缺 'train'")
    for tpl in ('data_path', 'video_path'):
        v = info_json.get(tpl)
        if not v or '{episode_index' not in v:
            fail(f'info.json {tpl} 缺失或无 episode_index 占位符')
    if ver == 'v2.1' and fps and eps is not None and frames is not None:
        ok(f'info.json: v2.1, fps={fps}, {eps} episode / {frames} 帧 / '
           f'{info_json.get("total_videos")} 视频 / chunks_size={info_json.get("chunks_size")}')
    feats = info_json.get('features')
    if not isinstance(feats, dict) or not feats:
        fail('info.json features 缺失或为空')
        return None
    vector_bad = [k for k, f in feats.items()
                  if f.get('dtype') == 'float32' and f.get('names') is not None
                  and f.get('shape') != [len(f['names'])]]
    if vector_bad:
        fail(f'features 向量列 dtype/shape/names 不一致: {vector_bad}')
    video_keys = [k for k, f in feats.items() if f.get('dtype') == 'video']
    for k in video_keys:
        v = feats[k].get('info', {})
        # v2.1 的 video 属性是扁平键名：'video.codec'、'video.pix_fmt'…而非嵌套 dict
        for sub, expect in (('video.codec', 'h264'), ('video.pix_fmt', 'yuv420p'), ('video.fps', fps)):
            if v.get(sub) != expect:
                fail(f'features[{k}].{sub}={v.get(sub)}，应为 {expect}')
        if v.get('has_audio'):
            fail(f'features[{k}].info.has_audio 应为 False')
    if fps:
        ok(f'features: {len(feats)} 项（视频 {len(video_keys)} 路: {", ".join(video_keys)}）')
    return info_json


def check_task_episodes(base, info_json):
    tasks, eps_meta = [], []
    try:
        for line in (base / 'meta' / 'tasks.jsonl').read_text().splitlines():
            if line.strip():
                tasks.append(json.loads(line))
        for line in (base / 'meta' / 'episodes.jsonl').read_text().splitlines():
            if line.strip():
                eps_meta.append(json.loads(line))
    except (OSError, ValueError) as exc:
        fail(f'meta jsonl 读取失败: {exc}')
        return tasks, eps_meta
    idxs = [t.get('task_index') for t in tasks]
    if idxs != list(range(len(tasks))):
        fail(f'tasks.jsonl task_index 不连续: {idxs}')
    else:
        ok(f'tasks.jsonl: {len(tasks)} 个任务: {[t.get("task") for t in tasks]}')
    if info_json and tasks and info_json.get('total_tasks') != len(tasks):
        fail(f"tasks.jsonl {len(tasks)} 条 ≠ info.total_tasks={info_json['total_tasks']}")
    eidx = [e.get('episode_index') for e in eps_meta]
    if eidx != list(range(len(eps_meta))):
        fail(f'episodes.jsonl episode_index 不连续: {eidx}')
    if info_json and len(eps_meta) != info_json.get('total_episodes'):
        fail(f"episodes.jsonl {len(eps_meta)} 条 ≠ info.total_episodes={info_json['total_episodes']}")
    total_len = sum(e.get('length', 0) for e in eps_meta)
    if info_json and total_len != info_json.get('total_frames'):
        fail(f'episodes.jsonl length 合计 {total_len} ≠ info.total_frames={info_json["total_frames"]}')
    else:
        ok(f'episodes.jsonl: {len(eps_meta)} episode，length 合计 {total_len}')
    task_names = {t.get('task') for t in tasks}
    for e in eps_meta:
        for tn in e.get('tasks', []):
            if tn not in task_names:
                fail(f'episode {e.get("episode_index")} 引用了 tasks.jsonl 中不存在的任务「{tn}」')
    try:
        stats_lines = [json.loads(l) for l in
                       (base / 'meta' / 'episodes_stats.jsonl').read_text().splitlines() if l.strip()]
        if sorted(s.get('episode_index') for s in stats_lines) != sorted(eidx):
            fail('episodes_stats.jsonl 的 episode 覆盖与 episodes.jsonl 不一致')
        else:
            ok(f'episodes_stats.jsonl: 覆盖 {len(stats_lines)} episode')
    except (OSError, ValueError) as exc:
        fail(f'episodes_stats.jsonl 读取失败: {exc}')
    return tasks, eps_meta


def iter_episode_files(base, info_json):
    """按 data_path/video_path 模板核对文件布局，返回 {episode_index: parquet 路径}。"""
    eps = {}
    data_files = sorted((base / 'data').glob('**/*.parquet')) if (base / 'data').exists() else []
    if not data_files:
        fail('data/ 下没有任何 parquet')
        return eps
    pattern = info_json.get('data_path', '') if info_json else ''
    for f in data_files:
        rel = f.relative_to(base).as_posix()
        m = match_template(rel, pattern)
        if m is None:
            fail(f'data/{f.relative_to(base)} 不符合 data_path 模板 {pattern}')
            continue
        chunk, ep = m
        if info_json and chunk >= info_json.get('chunks_size', 0):
            fail(f'chunk 编号 {chunk} 超出 chunks_size={info_json.get("chunks_size")}')
        if ep in eps:
            fail(f'episode {ep} 出现多个 parquet: {eps[ep]} 与 {rel}')
        eps[ep] = f
    if info_json:
        missing = set(range(info_json['total_episodes'])) - set(eps)
        if missing:
            fail(f'缺少 episode 的 parquet: {sorted(missing)}')
        if len(data_files) > info_json['total_episodes']:
            fail(f'data/ 有 {len(data_files)} 个 parquet，多于 total_episodes={info_json["total_episodes"]}')
    if eps:
        ok(f'data/: {len(eps)} 个 episode parquet 符合模板')
    return eps


def match_template(rel, pattern):
    """把 'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet' 变成可解析匹配。
    用 lambda 替换（callable 的返回值不经过转义解析，可直接含正则转义）。"""
    rx = re.sub(r'\{([a-z_]+):?0?\d*d?\}', lambda m: f'(?P<{m.group(1)}>\\d+)', pattern or '')
    m = re.fullmatch(rx, rel)
    return (int(m['episode_chunk']), int(m['episode_index'])) if m else None


def type_matches(got, want):
    """类型相等性，忽略 fixed_size_list 子字段名（录制端用 'element'，pa.list_ 默认 'item'）。"""
    if pa.types.is_fixed_size_list(got) and pa.types.is_fixed_size_list(want):
        return got.list_size == want.list_size and got.value_type.equals(want.value_type)
    return got.equals(want)


def expected_schema(features):
    fields = []
    for k, f in features.items():
        dt, shape = f.get('dtype'), f.get('shape')
        if dt == 'video':
            continue
        if dt == 'float32' and f.get('names'):
            fields.append(pa.field(k, pa.list_(pa.float32(), shape[0])))
        elif dt in ('float32', 'int64'):
            fields.append(pa.field(k, pa.float32() if dt == 'float32' else pa.int64()))
    return pa.schema(fields)


def check_parquet(base, eps, info_json, eps_meta, count_frames):
    feats = info_json.get('features', {})
    want = expected_schema(feats)
    lengths = {e['episode_index']: e.get('length') for e in eps_meta}
    tasks_path = base / 'meta' / 'tasks.jsonl'
    n_tasks = sum(1 for line in tasks_path.read_text().splitlines() if line.strip()) if tasks_path.exists() else 0
    global_index = []
    for ep, path in sorted(eps.items()):
        try:
            pf = pq.ParquetFile(path)
            table = pf.read()
        except Exception as exc:
            fail(f'episode {ep} parquet 读取失败: {exc}')
            continue
        n = table.num_rows
        if lengths.get(ep) is not None and n != lengths[ep]:
            fail(f'episode {ep}: parquet 行数 {n} ≠ episodes.jsonl length {lengths[ep]}')
        schema_ok = True
        got_names, want_names = list(table.schema.names), list(want.names)
        if got_names != want_names:
            if set(got_names) == set(want_names):
                warn(f'episode {ep}: parquet 列顺序与 features 不同: {got_names}')
            else:
                fail(f'episode {ep}: parquet 列 {got_names} ≠ features {want_names}')
                schema_ok = False
        if schema_ok:
            for k in want_names:
                if not type_matches(table.schema.field(k).type, want.field(k).type):
                    fail(f'episode {ep}: 列 {k} 类型 {table.schema.field(k).type} ≠ 期望 {want.field(k).type}')
                    schema_ok = False
        if schema_ok:
            ok(f'episode {ep}: schema 与 features 一致（{len(got_names)} 列, {n} 行）')

        def col(name):
            return table.column(name).to_pylist() if name in got_names else None

        fi = col('frame_index')
        if fi is not None and fi != list(range(n)):
            fail(f'episode {ep}: frame_index 不是 0..{n-1} 连续')
        ei = col('episode_index')
        if ei is not None and set(ei) != {ep}:
            fail(f'episode {ep}: episode_index 取值 {set(ei)}')
        idx = col('index')
        if idx is not None:
            global_index += idx
        ti = col('task_index')
        if ti is not None and not set(ti) <= set(range(max(n_tasks, 1))):
            fail(f'episode {ep}: task_index 越界: {set(ti)}')
        ts = col('timestamp')
        fps = info_json.get('fps', 30)
        if ts is not None:
            bad = [i for i in range(1, n) if ts[i] <= ts[i - 1]]
            if bad:
                fail(f'episode {ep}: timestamp 非单调（{len(bad)} 处）')
            if ts and abs(ts[0]) > 1e-6:
                warn(f'episode {ep}: 首帧 timestamp={ts[0]}，非 0')
            step = 1.0 / fps
            jumps = [i for i in range(1, n) if abs((ts[i] - ts[i - 1]) - step) > step / 2]
            if jumps:
                warn(f'episode {ep}: timestamp 步长偏离 1/{fps} 超半步的有 {len(jumps)} 处（最大差 '
                     f'{max(abs(ts[i] - ts[i-1]) - step for i in range(1, n)):.4f}s）')
        for k, f in feats.items():
            if f.get('dtype') != 'float32' or not f.get('names'):
                continue
            rows = col(k)
            if rows is None:
                continue
            lens = {len(r) for r in rows}
            if lens != {len(f['names'])}:
                fail(f'episode {ep}: 列 {k} 维度不一 {lens} ≠ {len(f["names"])}')
                continue
            flat = np.asarray([v for r in rows for v in r], dtype=np.float64)
            if not np.isfinite(flat).all():
                fail(f'episode {ep}: 列 {k} 含非有限值')
        rows = col('observation.base_pose')
        if rows:
            quat = np.asarray([r[3:7] for r in rows], dtype=np.float64)
            norms = np.linalg.norm(quat, axis=1)
            off = int((np.abs(norms - 1) > 1e-3).sum())
            if off:
                fail(f'episode {ep}: base_pose 四元数范数偏离 1 的行有 {off}（[x,y,z,qw,qx,qy,qz]）')
            else:
                ok(f'episode {ep}: base_pose 四元数范数全部 ≈1')
        state = col('observation.state')
        action = col('action')
        if state and action:
            s, a = np.asarray(state, dtype=np.float64), np.asarray(action, dtype=np.float64)
            info(f'episode {ep}: |state-action| 中位数 {float(np.median(np.abs(s - a))):.4f} rad'
                 f'（零阶保持目标与实际位有相位差，属正常）')
    if global_index and sorted(global_index) != list(range(len(global_index))):
        fail(f'index 全局不连续（共 {len(global_index)} 行）')
    elif global_index:
        ok(f'index 全局连续 0..{len(global_index)-1}')


def check_videos(base, info_json, count_frames):
    feats = info_json.get('features', {})
    fps = info_json.get('fps', 30)
    video_keys = [k for k, f in feats.items() if f.get('dtype') == 'video']
    eps_lines = [json.loads(l) for l in
                 (base / 'meta' / 'episodes.jsonl').read_text().splitlines() if l.strip()]
    tpl = info_json.get('video_path', '')
    chunk_size = info_json.get('chunks_size', 1000)
    expected_paths = {base / tpl.format(episode_chunk=e['episode_index'] // chunk_size,
                                        episode_index=e['episode_index'], video_key=k)
                      for e in eps_lines for k in video_keys}
    for k in video_keys:
        for e in eps_lines:
            ep = e['episode_index']
            path = base / tpl.format(episode_chunk=ep // chunk_size,
                                     episode_index=ep, video_key=k)
            if not path.exists():
                fail(f'episode {ep} 视频 {k} 缺失: {path.relative_to(base)}')
                continue
            props, nframes = probe_video(path, count_frames)
            if props is None:
                fail(f'episode {ep} {k}: ffprobe 失败（文件可能截断）')
                continue
            want_h = feats[k]['shape'][0]
            want_w = feats[k]['shape'][1]
            problems = []
            if (props.get('codec_name'), props.get('pix_fmt')) != ('h264', 'yuv420p'):
                problems.append(f"编码 {props.get('codec_name')}/{props.get('pix_fmt')} ≠ h264/yuv420p")
            if (int(props.get('width', -1)), int(props.get('height', -1))) != (want_w, want_h):
                problems.append(f"分辨率 {props.get('width')}x{props.get('height')} ≠ {want_w}x{want_h}")
            rate = props.get('r_frame_rate', '0/1')
            num, _, den = rate.partition('/')
            try:
                if abs(int(num) / int(den or 1) - fps) > 1e-6:
                    problems.append(f'帧率 {rate} ≠ {fps}')
            except ValueError:
                problems.append(f'帧率解析失败: {rate}')
            if count_frames and nframes is not None and nframes != e['length']:
                problems.append(f'帧数 {nframes} ≠ episode length {e["length"]}')
            if problems:
                fail(f'episode {ep} {k}: ' + '；'.join(problems))
            else:
                ok(f'episode {ep} {k}: h264 yuv420p {want_w}x{want_h}@{fps}'
                   + (f', {nframes} 帧' if nframes is not None else ', 帧数未校验(--skip-video)'))
    stray = sorted(p.relative_to(base).as_posix() for p in (base / 'videos').glob('**/*.mp4')
                   if p not in expected_paths)
    if stray:
        warn(f'videos/ 有模板外的多余 mp4: {stray}')


def check_recording_block(info_json, raw_info):
    rec = info_json.get('recording')
    if not isinstance(rec, dict):
        warn('info.json 无 recording 块（非本工程录制器产物？）')
        return
    if raw_info and rec.get('source_frames') != raw_info['raw_frames']:
        fail(f"info.recording.source_frames={rec.get('source_frames')} ≠ raw 实际 {raw_info['raw_frames']} 帧")
    else:
        ok(f"info.recording.source_frames={rec.get('source_frames')} 与 raw 一致")


def run_deep(deep_python, base):
    code = f'''
import sys
try:
    import lerobot
    print("version=" + lerobot.__version__)
except Exception:
    pass
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
try:
    ds = LeRobotDataset(repo_id="local/validate", root={str(base)!r})
    print(f"len={{len(ds)}}")
    item = ds[0]
    print("keys=" + ",".join(sorted(k for k in item if isinstance(k, str))))
except Exception as exc:
    print("ERROR", type(exc).__name__, str(exc).splitlines()[0] if str(exc) else "")
    sys.exit(1)
'''
    try:
        r = subprocess.run([deep_python, '-c', code], capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.TimeoutExpired) as exc:
        warn(f'--deep 执行失败: {exc}')
        return
    fields = dict(l.split('=', 1) for l in (r.stdout or '').splitlines() if '=' in l)
    err = next((l for l in (r.stdout or '').splitlines() if l.startswith('ERROR ')), None)
    if 'len=' in fields:
        ok(f"lerobot {fields.get('version', '?')} 加载成功: {fields.get('len')} 帧，字段 {fields.get('keys')}")
    elif err and 'BackwardCompatibilityError' in err:
        warn(f"lerobot {fields.get('version', '?')} 不读 v2.1（该版本只支持 ≥3.0，非数据问题；"
             f"主校验已证明本数据集结构完好）。训练前需转换："
             f"python -m lerobot.datasets.v30.convert_dataset_v21_to_v30 或用 lerobot≤0.3.x")
    else:
        warn(f'lerobot {fields.get("version", "?")} 加载失败（需人工确认）: {err or r.stderr.strip()[-500:]}')


def print_recover(d, raw_info):
    py = PROJECT_VENV / 'bin' / 'python'
    cmd = f'cd {PICO2G1} && {py} -m sim_scenes.lerobot_recording {d}'
    info('恢复导出（从 raw/ 重导出，幂等可重试）:')
    info(f'  {cmd}')
    _ = raw_info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('target', nargs='?', help='录制目录，或 recordings/ 下的目录名')
    ap.add_argument('--list', action='store_true', help='列出 recordings/ 下的录制后退出')
    ap.add_argument('--skip-video', action='store_true', help='跳过视频逐帧计数（快）')
    ap.add_argument('--deep', metavar='PYTHON', help='附加用指定解释器的 lerobot 库深度加载校验')
    args = ap.parse_args()

    if args.list or not args.target:
        if not RECORDINGS.exists():
            sys.exit(f'recordings 目录不存在: {RECORDINGS}')
        rows = sorted(p for p in RECORDINGS.iterdir() if p.is_dir() and not p.name.startswith('.'))
        print('recordings/ 下的录制（传入名称校验）:')
        for p in rows:
            state = {'complete': '完成', 'interrupted': '导出中断', 'raw-only': '未导出',
                     'invalid': '无 raw'}.get(parse_state(p), '未知')
            mark = {'完成': '✅', '导出中断': '⚠️', '未导出': '⚠️'}.get(state, '❌')
            print(f'  {mark} {p.name}  [{state}]')
        if not args.target:
            print('\n用法: 04_validate_recording.py <目录名或完整路径>')
        return

    d = Path(args.target).expanduser()
    if not d.is_absolute() and ('/' not in str(d) or not d.exists()):
        d = RECORDINGS / d
    d = d.resolve()
    if not d.is_dir():
        ap.error(f'目录不存在: {d}')

    state = parse_state(d)
    print(f'== 录制目录: {d}')
    print(f'== 状态判定: ' + {'complete': '✅ 导出完成（顶层 meta/ 齐全）',
                             'interrupted': '⚠️ 导出中断（有 raw/ 与 .export/，缺顶层 meta）',
                             'raw-only': '⚠️ 未导出（仅 raw/）'}.get(state, '❌ 目录内既无 meta 也无 raw，不是录制产物'))
    if state == 'invalid':
        print('== 汇总: 无可校验内容')
        sys.exit(2)

    raw_info = None
    if state != 'complete':
        raw_info = check_raw(d)

    dataset_base = d if state == 'complete' else d / '.export'
    if state != 'complete':
        if not (dataset_base / 'data').exists() and not (dataset_base / 'videos').exists():
            info('.export/ 暂存目录为空，导出在写产物前即中断')
        else:
            print(f'-- 暂存目录 {dataset_base.name}/ 按 v2.1 校验（尚未发布到顶层）')
    if not (dataset_base / 'meta' / 'info.json').exists():
        info('无 meta/info.json，跳过结构/parquet/视频校验（恢复导出后重跑本脚本将完整校验）')
    else:
        print(f'\n-- LeRobot v2.1 结构（{dataset_base.name}/）')
        check_meta(dataset_base, require_meta=(state == 'complete'))
        info_json = check_info(dataset_base)
        if info_json:
            tasks, eps_meta = check_task_episodes(dataset_base, info_json)
            eps = iter_episode_files(dataset_base, info_json)
            if eps and eps_meta:
                print('\n-- parquet 内容')
                check_parquet(dataset_base, eps, info_json, eps_meta, not args.skip_video)
            if info_json.get('total_videos'):
                print('\n-- 视频')
                check_videos(dataset_base, info_json, not args.skip_video)
            if state == 'complete':
                print('\n-- 与 raw 交叉核对')
                check_recording_block(info_json, raw_info)
        else:
            warn('info.json 不可读，跳过 parquet/视频校验')

    if args.deep:
        print('\n-- lerobot 库深度加载')
        run_deep(args.deep, dataset_base)

    print('\n== 汇总: ' + ' / '.join(f'{sym} {sum(1 for s, _ in report_items if s == sym)}'
                                     for sym in ('✅', '⚠️', '❌') if any(s == sym for s, _ in report_items)))
    if state != 'complete':
        print('\n== 恢复导出（从 raw/ 重导出，幂等；导出完成后再跑本脚本应全绿）:')
        print(f'   cd {PICO2G1} && MUJOCO_GL=egl {PROJECT_VENV}/bin/python -m sim_scenes.lerobot_recording {d}')
        info('（MUJOCO_GL=egl：无 DISPLAY 的会话里 mujoco.Renderer 需要 EGL 上下文，照抄即可）')
    bad = sum(1 for s, _ in report_items if s == '❌')
    # 只有完整导出且无失败项才算通过；中断/未导出即使无硬伤也不算过
    sys.exit(1 if (bad or state != 'complete') else 0)


if __name__ == '__main__':
    main()
