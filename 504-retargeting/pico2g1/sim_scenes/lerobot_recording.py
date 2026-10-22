"""Recoverable single-episode LeRobot v2.1 export, independent of Qt.

Capture is logical source-step time, not wall time. The frozen MJB and full qpos
spool retain scene props; robot state/action only contain named scalar joints.
Run ``python -m sim_scenes.lerobot_recording OUTPUT_DIR`` to retry an export.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading

import mujoco
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

FPS, WIDTH, HEIGHT = 30, 1280, 720
CAMERAS = {'observation.images.head': 'robot_head',
           'observation.images.left_wrist': 'robot_left_wrist',
           'observation.images.right_wrist': 'robot_right_wrist'}


def robot_joint_layout(model):
    """Named scalar joints beneath pelvis in compiled model order."""
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'pelvis')
    if pelvis < 0:
        raise ValueError('Model has no pelvis body')
    bodies = {pelvis}
    for body in range(pelvis + 1, model.nbody):
        if int(model.body_parentid[body]) in bodies:
            bodies.add(body)
    names, addresses = [], []
    for j in range(model.njnt):
        if int(model.jnt_bodyid[j]) in bodies and int(model.jnt_type[j]) in (
                int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            if not name:
                raise ValueError('Every robot scalar joint must have a name')
            names.append(name)
            addresses.append(int(model.jnt_qposadr[j]))
    if not names:
        raise ValueError('Model has no scalar robot joints')
    return names, np.array(addresses, dtype=int)


def _dependencies(preflight=False):
    paths = [shutil.which(name) for name in ('ffmpeg', 'ffprobe')]
    if not all(paths):
        raise RuntimeError('Recording requires ffmpeg and ffprobe on PATH (brew install ffmpeg)')
    if preflight:
        try:
            encoders = subprocess.run([paths[0], '-hide_banner', '-encoders'],
                                     capture_output=True, text=True, check=True, timeout=15)
            if not any('libx264' in line.split() for line in encoders.stdout.splitlines()):
                raise RuntimeError('FFmpeg has no libx264 encoder; install a full FFmpeg build')
            subprocess.run([paths[1], '-version'], capture_output=True, check=True, timeout=15)
        except (subprocess.SubprocessError, OSError) as exc:
            raise RuntimeError(f'FFmpeg/ffprobe preflight failed: {exc}') from exc
    return paths


def _json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


class _Stats:
    """Stable bounded-memory moments, including per-channel image statistics."""
    def __init__(self):
        self.n = 0
        self.frames = 0

    def add(self, values):
        a = np.asarray(values, dtype=np.float64)
        if a.ndim == 1:
            a = a[:, None]
        n = len(a)
        mean, variance = a.mean(axis=0), a.var(axis=0)
        low, high = a.min(axis=0), a.max(axis=0)
        if self.n:
            delta = mean - self.mean
            self.m2 += variance*n + delta**2*self.n*n/(self.n+n)
            self.mean += delta*n/(self.n+n)
            self.low = np.minimum(self.low, low)
            self.high = np.maximum(self.high, high)
        else:
            self.mean, self.m2, self.low, self.high = mean, variance*n, low, high
        self.n += n
        self.frames += 1

    def result(self, image=False):
        def serialize(a):
            return (a.reshape(3, 1, 1) if image else a).tolist()
        return {'min': serialize(self.low), 'max': serialize(self.high),
                'mean': serialize(self.mean), 'std': serialize(np.sqrt(self.m2/self.n)),
                'count': [self.frames if image else self.n]}


class _Encoder:
    def __init__(self, path, executable):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.log = tempfile.TemporaryFile()
        try:
            self.process = subprocess.Popen([
                executable, '-nostdin', '-v', 'error', '-y', '-f', 'rawvideo',
                '-pix_fmt', 'rgb24', '-s', f'{WIDTH}x{HEIGHT}', '-r', str(FPS),
                '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-preset', 'fast',
                '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(path)],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.log)
        except BaseException:
            self.log.close()
            raise

    def write(self, frame):
        self.process.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())

    def close(self):
        try:
            if self.process.stdin and not self.process.stdin.closed:
                self.process.stdin.close()
            status = self.process.wait(timeout=120)
            if status:
                self.log.seek(0)
                raise RuntimeError('FFmpeg encoding failed: ' + self.log.read().decode(errors='replace')[-4000:])
        except BaseException:
            if self.process.poll() is None:
                self.process.kill()
                self.process.wait()
            raise
        finally:
            self.log.close()


class RecordingSession:
    def __init__(self, model, output_dir: Path, task: str, *, source_fps=50,
                 render_shadows=False, action_semantics: str):
        _dependencies(preflight=True)
        if not math.isfinite(source_fps) or source_fps <= 0:
            raise ValueError('source_fps must be positive and finite')
        if not task.strip() or not action_semantics.strip():
            raise ValueError('Task and action semantics must not be empty')
        names, addresses = robot_joint_layout(model)
        for camera in CAMERAS.values():
            if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera) < 0:
                raise ValueError(f'Model is missing recording camera {camera}')
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.raw_dir = self.output_dir / 'raw'
        self.raw_dir.mkdir()
        self.config = {'format': 1, 'mujoco_version': mujoco.__version__, 'nq': model.nq,
                       'joint_names': names, 'joint_addresses': addresses.tolist(),
                       'task': task.strip(), 'source_fps': source_fps,
                       'render_shadows': bool(render_shadows), 'action_semantics': action_semantics,
                       'time_semantics': 'successful consumed simulation steps; first sample t=0; pauses excluded',
                       'resampling': 'ceil(N*30/source_fps); quaternion-aware qpos interpolation; last pose clamped; action zero-order hold',
                       'state_semantics': 'actual scalar robot joint positions (radians for hinges, metres for slides)',
                       'video_stats': 'LeRobot v2.1 sampled source RGB frames, spatial stride 8, normalized by 255'}
        mujoco.mj_saveModel(model, str(self.raw_dir/'model.mjb'), None)
        _json(self.raw_dir/'capture.json', self.config)
        self._initialize()
        self._stream = (self.raw_dir/'frames.bin').open('wb', buffering=0)

    def _initialize(self):
        self.names = self.config['joint_names']
        self.addresses = np.asarray(self.config['joint_addresses'], dtype=int)
        self.source_fps = self.config['source_fps']
        self.frame_count = 0
        self._stream = None
        self._lock = threading.Lock()
        self._width = self.config['nq'] + len(self.names)

    @classmethod
    def recover(cls, output_dir):
        output_dir = Path(output_dir).expanduser().resolve()
        if (output_dir/'meta/info.json').exists():
            raise FileExistsError('Dataset is already complete')
        self = cls.__new__(cls)
        self.output_dir, self.raw_dir = output_dir, output_dir/'raw'
        self.config = json.loads((self.raw_dir/'capture.json').read_text())
        if self.config['format'] != 1 or self.config['mujoco_version'] != mujoco.__version__:
            raise ValueError('Recovery requires the original capture format and MuJoCo version')
        self._initialize()
        # An interrupted final write can leave a partial trailing record. Ignore it.
        self.frame_count = (self.raw_dir/'frames.bin').stat().st_size // (8*self._width)
        return self

    def append(self, data, action):
        if self._stream is None:
            raise RuntimeError('Capture has been sealed')
        action = np.asarray(action, dtype=np.float64)
        qpos = np.asarray(data.qpos, dtype=np.float64)
        if action.shape != (len(self.names),) or qpos.shape != (self.config['nq'],):
            raise ValueError('qpos or action shape does not match recording model')
        if not np.isfinite(action).all() or not np.isfinite(qpos).all():
            raise ValueError('qpos and action must be finite')
        payload = np.concatenate((qpos, action)).astype('<f8').tobytes()
        written = self._stream.write(payload)
        if written != len(payload):
            self.seal()
            raise OSError('Incomplete raw recording write; capture sealed')
        self.frame_count += 1

    def seal(self):
        if self._stream is not None:
            stream, self._stream = self._stream, None
            try:
                stream.flush()
                os.fsync(stream.fileno())
            finally:
                stream.close()

    def resampled_frames(self, model=None):
        """Yield (timestamp, full qpos, held target) with bounded resident memory."""
        self.seal()
        if not self.frame_count:
            return
        if model is None:
            model = mujoco.MjModel.from_binary_path(str(self.raw_dir/'model.mjb'))
        raw = np.memmap(self.raw_dir/'frames.bin', dtype='<f8', mode='r',
                        shape=(self.frame_count, self._width))
        nq = self.config['nq']
        velocity = np.empty(model.nv)
        for index in range(math.ceil(self.frame_count*FPS/self.source_fps)):
            position = index*self.source_fps/FPS
            left = min(int(math.floor(position)), self.frame_count-1)
            right = min(left+1, self.frame_count-1)
            fraction = min(position-left, 1.)
            pose = raw[left, :nq].copy()
            mujoco.mj_differentiatePos(model, velocity, 1., pose, raw[right, :nq])
            mujoco.mj_integratePos(model, pose, velocity, fraction)
            mujoco.mj_normalizeQuat(model, pose)
            yield index/FPS, pose, raw[left, nq:].copy()

    def finalize(self, progress=None):
        with self._lock:
            self.seal()
            if not self.frame_count:
                return None
            if (self.output_dir/'meta/info.json').exists():
                return self.output_dir
            return self._export(progress)

    def _export(self, progress):
        from . import PulseSession, scene_render_options, DirectionalShadows
        ffmpeg, ffprobe = _dependencies()
        model = mujoco.MjModel.from_binary_path(str(self.raw_dir/'model.mjb'))
        data = mujoco.MjData(model)
        model.vis.global_.offwidth = max(WIDTH, model.vis.global_.offwidth)
        model.vis.global_.offheight = max(HEIGHT, model.vis.global_.offheight)
        model.vis.map.znear = .01/model.stat.extent
        mujoco.mj_forward(model, data)
        shadows = DirectionalShadows(model, data, enabled=self.config['render_shadows'])
        option = scene_render_options(PulseSession.for_model(model), model)
        option.geomgroup[3] = 0
        total = math.ceil(self.frame_count*FPS/self.source_fps)
        sample_count = max(min(100, total), min(int(total**.75), 10000))
        sample_indices = set(np.round(np.linspace(0, total-1, sample_count)).astype(int).tolist())
        stage = self.output_dir / '.export'
        if stage.exists():
            shutil.rmtree(stage)
        stage.mkdir()
        vectors = {'action': self.names, 'observation.state': self.names,
                   'observation.base_pose': ['x','y','z','qw','qx','qy','qz']}
        scalar_types = {'timestamp': pa.float32(), 'frame_index': pa.int64(),
                        'episode_index': pa.int64(), 'index': pa.int64(), 'task_index': pa.int64()}
        schema = pa.schema([pa.field(k, pa.list_(pa.float32(), len(v))) for k,v in vectors.items()]
                           + [pa.field(k,v) for k,v in scalar_types.items()])
        stats = {key: _Stats() for key in list(vectors)+list(scalar_types)+list(CAMERAS)}
        parquet = stage/'data/chunk-000/episode_000000.parquet'
        parquet.parent.mkdir(parents=True)
        video_paths = {k: stage/f'videos/chunk-000/{k}/episode_000000.mp4' for k in CAMERAS}
        with ExitStack() as stack:
            renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
            stack.callback(renderer.close)
            encoders = {}
            for key,path in video_paths.items():
                encoders[key] = _Encoder(path, ffmpeg)
                stack.callback(encoders[key].close)
            writer = stack.enter_context(pq.ParquetWriter(parquet, schema))
            batch = []
            for index, (timestamp, pose, action) in enumerate(self.resampled_frames(model)):
                data.qpos[:] = pose
                data.time = timestamp
                mujoco.mj_forward(model, data)
                pelvis = data.body('pelvis')
                row = {'action': action.astype(np.float32).tolist(),
                       'observation.state': pose[self.addresses].astype(np.float32).tolist(),
                       'observation.base_pose': np.r_[pelvis.xpos, pelvis.xquat].astype(np.float32).tolist(),
                       'timestamp': float(np.float32(timestamp)), 'frame_index': index,
                       'episode_index': 0, 'index': index, 'task_index': 0}
                batch.append(row)
                for key,value in row.items():
                    stats[key].add(np.asarray(value).reshape(1,-1))
                for key,camera in CAMERAS.items():
                    renderer.update_scene(data, camera=camera, scene_option=option)
                    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = shadows.enabled
                    shadows.prepare(renderer, data)
                    frame = renderer.render()
                    encoders[key].write(frame)
                    if index in sample_indices:
                        stats[key].add(frame[::8,::8].reshape(-1,3)/255.)
                if len(batch) >= 512:
                    writer.write_table(pa.Table.from_pylist(batch, schema=schema)); batch.clear()
                if progress:
                    progress(index+1,total)
            if batch:
                writer.write_table(pa.Table.from_pylist(batch, schema=schema))
        for path in video_paths.values():
            result = subprocess.run([ffprobe, '-v','error','-count_frames','-show_streams',
                                     '-of','json',str(path)], capture_output=True, check=True, text=True)
            streams = json.loads(result.stdout)['streams']
            if len(streams) != 1:
                raise RuntimeError('Encoded video must have exactly one stream')
            s = streams[0]
            if (s['width'],s['height'],s['codec_name'],s['pix_fmt'],s['r_frame_rate'],
                int(s.get('nb_read_frames',-1))) != (WIDTH,HEIGHT,'h264','yuv420p','30/1',total):
                raise RuntimeError(f'Encoded video validation failed: {path}')
        features = {k: {'dtype':'float32', 'names':v, 'shape':[len(v)]} for k,v in vectors.items()}
        features.update({k: {'dtype':str(v), 'names':None, 'shape':[1]} for k,v in scalar_types.items()})
        features['timestamp']['dtype'] = 'float32'
        features.update({k: {'dtype':'video', 'names':['height','width','channels'],
                            'shape':[HEIGHT,WIDTH,3], 'info':{'has_audio':False,'video.channels':3,
                            'video.codec':'h264','video.fps':FPS,'video.height':HEIGHT,'video.width':WIDTH,
                            'video.pix_fmt':'yuv420p','video.is_depth_map':False}} for k in CAMERAS})
        info = {'codebase_version':'v2.1','robot_type':'g1_wuji_dual_hand', 'total_episodes':1,
                'total_frames':total,'total_tasks':1,'total_videos':3,'total_chunks':1,'chunks_size':1000,
                'fps':FPS,'splits':{'train':'0:1'},
                'data_path':'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet',
                'video_path':'videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4',
                'features':features,'recording':dict(self.config, source_frames=self.frame_count)}
        meta = stage/'meta'; meta.mkdir()
        results = {k:v.result(image=k in CAMERAS) for k,v in stats.items()}
        _json(meta/'stats.json', results)
        for filename, value in [('tasks.jsonl',{'task_index':0,'task':self.config['task']}),
                                ('episodes.jsonl',{'episode_index':0,'tasks':[self.config['task']], 'length':total}),
                                ('episodes_stats.jsonl',{'episode_index':0,'stats':results})]:
            (meta/filename).write_text(json.dumps(value,ensure_ascii=False)+'\n')
        _json(meta/'info.json', info)
        # Completion metadata is published only after all outputs passed probing.
        for name in ('data','videos','meta'):
            target = self.output_dir/name
            if target.exists():
                shutil.rmtree(target)
            (stage/name).replace(target)
        stage.rmdir()
        return self.output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_dir', type=Path)
    args = parser.parse_args()
    result = RecordingSession.recover(args.output_dir).finalize()
    print(result if result else 'Empty capture: no dataset exported')


if __name__ == '__main__':
    main()
