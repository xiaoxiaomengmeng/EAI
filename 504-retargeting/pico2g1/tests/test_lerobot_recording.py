"""Behavior checks for recoverable 50 Hz to 30 Hz dataset export."""
import importlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import mujoco
import numpy as np
import pyarrow.parquet as pq


def backend():
    try:
        return importlib.import_module('sim_scenes.lerobot_recording')
    except ModuleNotFoundError:
        raise AssertionError('Recording backend is not implemented')


def model():
    return mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
    <light pos="0 0 3"/><geom type="plane" size="10 10 .1" rgba=".3 .5 .8 1"/>
    <camera name="robot_head" pos="2 0 2" xyaxes="0 1 0 -1 0 1"/>
    <camera name="robot_left_wrist" pos="0 2 2" xyaxes="-1 0 0 0 -1 1"/>
    <camera name="robot_right_wrist" pos="-2 0 2" xyaxes="0 -1 0 1 0 1"/>
    <body name="pelvis" pos="0 0 1"><freejoint/><geom size=".2" rgba="1 .1 .1 1"/>
    <body><joint name="robot_hinge" axis="0 1 0"/><geom type="capsule" size=".06 .3"/></body></body>
    <body pos="1 0 1"><joint name="prop_slide" type="slide"/><geom size=".1"/></body>
    </worldbody></mujoco>''')


class RecordingBackendTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.tmp_path = Path(directory.name)

    def test_joint_selection_and_quaternion_resampling(self):
        tmp_path = self.tmp_path
        b = backend(); m = model(); d = mujoco.MjData(m)
        names, addresses = b.robot_joint_layout(m)
        assert names == ['robot_hinge']; assert addresses.tolist() == [7]
        s = b.RecordingSession(m, tmp_path/'capture', 'move', action_semantics='target')
        for i in range(3):
            d.qpos[0] = i
            d.qpos[3:7] = [np.cos(i*np.pi/4), 0, 0, np.sin(i*np.pi/4)]
            d.qpos[7] = i*.3; d.qpos[8] = i*2
            s.append(d, np.array([10+i]))
        s.seal(); s.seal()
        rows = list(s.resampled_frames())
        assert len(rows) == 2
        timestamp, qpos, action = rows[1]
        np.testing.assert_allclose(timestamp, 1/30, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(qpos[0], 5/3, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(qpos[7], .5, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(qpos[8], 10/3, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(qpos[3:7], [np.cos(5*np.pi/12),0,0,np.sin(5*np.pi/12)], rtol=1e-6, atol=1e-7)
        assert action == [11]
        with self.assertRaises(RuntimeError): s.append(d, [0])


    def test_empty_no_overwrite_validation(self):
        tmp_path = self.tmp_path
        b = backend(); m = model(); d = mujoco.MjData(m)
        s = b.RecordingSession(m, tmp_path/'capture', 'move', action_semantics='target')
        with self.assertRaises(FileExistsError):
            b.RecordingSession(m, s.output_dir, 'move', action_semantics='target')
        with self.assertRaises(ValueError): s.append(d, [1,2])
        with self.assertRaises(ValueError): s.append(d, [float('nan')])
        assert s.frame_count == 0
        assert s.finalize() is None
        assert not (s.output_dir/'meta/info.json').exists()


    def test_actual_export_alignment_and_recovery(self):
        tmp_path = self.tmp_path
        b = backend(); m = model(); d = mujoco.MjData(m)
        s = b.RecordingSession(m, tmp_path/'capture', 'test motion', action_semantics='joint position target')
        for i in range(5):
            d.qpos[7] = i*.1
            s.append(d, [1+i])
        s.seal()
        # A real subprocess failure must leave a recoverable capture and no completion marker.
        encoder = b.shutil.which
        mock = patch.object(b.shutil, 'which', lambda name: '/usr/bin/false' if name == 'ffmpeg' else encoder(name))
        mock.start()
        self.addCleanup(mock.stop)
        with self.assertRaises((RuntimeError, subprocess.SubprocessError, BrokenPipeError)):
            s.finalize()
        assert not (s.output_dir/'meta/info.json').exists()
        mock.stop()
        recovered = b.RecordingSession.recover(s.output_dir)
        progress = []
        assert recovered.frame_count == 5
        with ThreadPoolExecutor(max_workers=1) as worker:
            result = worker.submit(recovered.finalize, lambda done,total: progress.append((done,total))).result(timeout=60)
        assert result == s.output_dir
        assert progress[-1] == (3,3)
        info = json.loads((s.output_dir/'meta/info.json').read_text())
        assert info['total_frames'] == 3 and info['total_videos'] == 3
        table = pq.read_table(s.output_dir/'data/chunk-000/episode_000000.parquet').to_pydict()
        np.testing.assert_allclose(table['timestamp'], [0,1/30,2/30], rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(np.array(table['observation.state'])[:,0], [0,1/6,1/3], rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(np.array(table['action'])[:,0], [1,2,4], rtol=1e-6, atol=1e-7)
        assert len(table['observation.base_pose'][0]) == 7
        stats = json.loads((s.output_dir/'meta/stats.json').read_text())
        np.testing.assert_allclose(stats['action']['mean'], [7/3], rtol=1e-6, atol=1e-7)
        assert stats['observation.images.head']['count'] == [3]
        assert np.array(stats['observation.images.head']['mean']).shape == (3,1,1)
        for path in (s.output_dir/'videos').rglob('*.mp4'):
            probe = json.loads(subprocess.check_output([shutil.which('ffprobe'),'-v','error','-count_frames','-show_streams','-of','json',str(path)]))['streams'][0]
            assert (probe['width'],probe['height'],probe['codec_name'],probe['r_frame_rate'],probe['nb_read_frames']) == (1280,720,'h264','30/1','3')
            raw = subprocess.check_output([shutil.which('ffmpeg'),'-v','error','-i',str(path),'-frames:v','1','-f','rawvideo','-pix_fmt','rgb24','-'])
            assert np.frombuffer(raw,dtype=np.uint8).std() > 10
        with self.assertRaises(FileExistsError): b.RecordingSession.recover(s.output_dir)

    def test_missing_h264_encoder_rejected_before_directory_creation(self):
        b = backend()
        with patch.object(b.shutil, "which", lambda name: "/usr/bin/true"):
            with self.assertRaises(RuntimeError):
                b.RecordingSession(model(), self.tmp_path/"capture", "move", action_semantics="target")
        self.assertFalse((self.tmp_path/"capture").exists())


if __name__ == "__main__":
    unittest.main()
