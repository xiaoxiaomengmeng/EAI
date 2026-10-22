"""Main's calibration flow must coexist with develop's interactive environment."""
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import mujoco
import numpy as np
from PySide6 import QtWidgets
from test_scene_startup import WaitingSource
from test_scenes import sonic, light


def sample_pose():
    pose = np.zeros((24, 7), np.float32)
    pose[:, 6] = 1
    heights = {'waist': .95, 'spine1': 1.05, 'spine2': 1.2,
               'spine3': 1.35, 'neck': 1.5, 'head': 1.65}
    positions = {'hip': (.12, .9, 0), 'knee': (.12, .5, 0),
                 'ankle': (.12, .1, 0), 'foot': (.12, .05, .15),
                 'collar': (.1, 1.4, 0), 'shoulder': (.22, 1.4, 0),
                 'elbow': (.42, 1.2, 0), 'wrist': (.48, 1, .12), 'hand': (.55, 1, .12)}
    for i, name in enumerate(sonic.JOINT_NAMES):
        if name in heights:
            pose[i, 1] = heights[name]
        else:
            side, joint = name.split('_', 1)
            pose[i, :3] = positions[joint]
            if side == 'right':
                pose[i, 0] *= -1
    return pose


class CalibrationMergeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_calibrate_save_resume_with_pulse_for_both_windows(self):
        for module in (sonic, light):
            with self.subTest(app=module.__name__), tempfile.TemporaryDirectory() as tmp:
                source, paused = WaitingSource(), threading.Event()
                offsets = {name: np.zeros(3) for name, _ in sonic.ROTATION_OFFSET_JOINTS}
                if module is sonic:
                    window = sonic.Window(source, scene='pulse', rotation_offsets=offsets, pause_event=paused)
                else:
                    model = light.build_model(scene='pulse')
                    window = light.TuningWindow(model, mujoco.MjData(model), source,
                                                light.TUNING_DEFAULTS.copy(), paused)
                window.timer.stop()
                try:
                    pose = sample_pose()
                    trackers = pose[[20, 21]].copy()
                    gloves = {side: np.zeros((21, 3)) for side in ('left', 'right')}
                    if module is sonic:
                        control = sonic.apply_rotation_sources(pose[None], trackers[None], offsets)
                        converted = sonic.convert(control, window.rest, window.parents)
                        reference = {k: np.repeat(v, 10, axis=0) for k, v in converted.items()}
                        frame = (reference, pose, trackers, gloves, {})
                    else:
                        frame = (light.apply_hand_trackers(pose, trackers), pose, trackers, gloves, {})
                    window.latest.put(frame)
                    window.consume_frame()
                    self.assertFalse(window.failed)
                    self.assertIn('PULSE', window.statusBar().currentMessage())
                    self.assertTrue(np.isfinite(window.scene_view.data.qpos).all())
                    window.show()
                    self.app.processEvents()  # Exercise SourceView's new tracker/axis painting.
                    if module is sonic:
                        window.enter_adjustment()
                        window.change_axis(0, 12)
                    else:
                        window.enter_calibration()
                        slider = window.controls['arm_scale'][0]
                        slider.setValue(slider.value()+1)
                    self.assertTrue(paused.is_set())
                    window.consume_frame()
                    self.assertFalse(window.failed)
                    self.assertIs(window.current_frame, frame)
                    self.assertIn('PULSE', window.statusBar().currentMessage())
                    self.assertNotIn('回滚', window.statusBar().currentMessage())
                    self.app.processEvents()
                    window.orbit_camera(30, 10)
                    window.zoom_camera(1)
                    np.testing.assert_allclose(window.camera.lookat,
                        window.scene_view.data.body('pelvis').xpos+window.scene_view.robot_offset)
                    path = Path(tmp)/'settings.json'
                    if module is sonic:
                        with patch.object(sonic, 'ROTATION_OFFSETS_PATH', path):
                            window.finish_adjustment()
                        self.assertEqual(json.loads(path.read_text())[window.selected_joint][0], 12)
                    else:
                        with patch.object(light, 'TUNING_PATH', path):
                            window.finish_calibration()
                        self.assertEqual(json.loads(path.read_text())['arm_scale'], window.tuning['arm_scale'])
                    self.assertFalse(paused.is_set())
                    next_frame = tuple(x.copy() if isinstance(x, np.ndarray) else x for x in frame)
                    window.latest.put(next_frame)
                    window.consume_frame()
                    self.assertIs(window.current_frame, next_frame)
                    self.assertIn('50 Hz', window.statusBar().currentMessage())
                    self.assertTrue(np.isfinite(window.scene_view.data.qpos).all())
                finally:
                    source.release.set()
                    window.close()
