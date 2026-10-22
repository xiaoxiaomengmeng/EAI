"""Real Qt capture must save only consumed simulation frames and finish safely."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import mujoco
import numpy as np
import pyarrow.parquet as pq
from PySide6 import QtWidgets, QtTest
from test_scene_startup import WaitingSource
from test_scenes import light, sonic
from test_merge_calibration import sample_pose


def source_frame(window, module):
    pose = sample_pose()
    trackers = pose[[20, 21]].copy()
    gloves = {side: np.zeros((21, 3)) for side in ("left", "right")}
    if module is sonic:
        control = sonic.apply_rotation_sources(pose[None], trackers[None], window.rotation_offsets)
        converted = sonic.convert(control, window.rest, window.parents)
        reference = {key: np.repeat(value, 10, axis=0) for key, value in converted.items()}
        return reference, pose, trackers, gloves, {}
    return light.apply_hand_trackers(pose, trackers), pose, trackers, gloves, {}


class RecordingWindowTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def make_window(self, module):
        source = WaitingSource()
        if module is sonic:
            window = sonic.Window(source, scene="pulse")
        else:
            model = light.build_model(scene="pulse")
            window = light.TuningWindow(model, mujoco.MjData(model), source, light.TUNING_DEFAULTS.copy())
        window.timer.stop()
        return window, source

    def wait_for_export(self, controls):
        deadline = time.monotonic() + 30
        while controls.busy and time.monotonic() < deadline:
            self.app.processEvents()
            QtTest.QTest.qWait(10)
        self.assertFalse(controls.busy, "Export worker did not settle")
        self.assertFalse(controls.last_error, controls.last_error)


class RecordingControlsTests(RecordingWindowTestCase):
    def test_both_windows_export_only_consumed_frames_and_stop_on_eof(self):
        for module in (light, sonic):
            with self.subTest(module=module.__name__), tempfile.TemporaryDirectory() as tmp:
                window, source = self.make_window(module)
                try:
                    self.assertTrue(hasattr(window, "recording"), "Missing recording controls")
                    controls = window.recording
                    path = Path(tmp) / "dataset"
                    controls.task_input.setText("移动手腕")
                    self.assertTrue(controls.start_recording(path))
                    window.orbit_camera(20, 0)
                    window.consume_frame()  # No input must not invent a sample.
                    self.assertEqual(controls.session.frame_count, 0)
                    expected = None
                    for _ in range(3):
                        window.latest.put(source_frame(window, module))
                        window.consume_frame()
                        self.assertFalse(window.failed)
                        if expected is None:
                            expected = window.scene_view.data.qpos[controls.joint_addresses].copy()
                    self.assertEqual(controls.session.frame_count, 3)
                    with window.latest.lock:
                        window.latest.finished = True
                    window.consume_frame()
                    self.wait_for_export(controls)
                    info = json.loads((path / "meta/info.json").read_text())
                    self.assertEqual(info["total_frames"], 2)
                    self.assertEqual(info["features"]["observation.state"]["shape"], [69])
                    rows = pq.read_table(path / "data/chunk-000/episode_000000.parquet").to_pylist()
                    np.testing.assert_allclose(rows[0]["observation.state"], expected, atol=1e-6)
                    if module is light:
                        np.testing.assert_allclose(rows[0]["action"], rows[0]["observation.state"])
                    else:
                        self.assertFalse(np.allclose(rows[0]["action"], rows[0]["observation.state"]))
                    self.assertEqual(len(list((path / "videos").rglob("*.mp4"))), 3)
                    self.assertTrue(controls.start_button.isEnabled())
                    self.assertIn("已保存", controls.status.text())
                finally:
                    source.release.set()
                    window.close()
                    self.app.processEvents()

    def test_close_waits_for_save_and_existing_directory_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            window, source = self.make_window(light)
            try:
                self.assertTrue(hasattr(window, "recording"), "Missing recording controls")
                controls = window.recording
                path = Path(tmp) / "dataset"
                path.mkdir()
                sentinel = path / "keep.txt"
                sentinel.write_text("keep")
                self.assertFalse(controls.start_recording(path))
                self.assertEqual(sentinel.read_text(), "keep")
                path = Path(tmp) / "new"
                self.assertTrue(controls.start_recording(path))
                window.latest.put(source_frame(window, light))
                window.consume_frame()
                window.show()
                self.app.processEvents()
                window.close()
                self.assertTrue(window.isVisible(), "Closing must defer until export finishes")
                self.wait_for_export(controls)
                self.app.processEvents()
                self.assertFalse(window.isVisible())
                self.assertTrue((path / "meta/info.json").exists())
            finally:
                source.release.set()
                window.close()

    def test_sonic_records_clipped_targets_separately_from_actual_state(self):
        window, source = self.make_window(sonic)
        try:
            sim = window.sim
            self.assertTrue(hasattr(sim, "commanded_qpos"), "Command target not retained")
            target = np.full(29, 100.)
            with patch.object(sim.policy, "target", return_value=(target, 0.)):
                sim.step({}, {"left": np.full(20, 100.)})
            np.testing.assert_allclose(sim.commanded_qpos[sim.qpos], sim.upper)
            np.testing.assert_allclose(sim.commanded_qpos[sim.hand_qpos["left"]], sim.hand_limits["left"][1])
            self.assertFalse(np.allclose(sim.commanded_qpos[sim.qpos], sim.data.qpos[sim.qpos]))
            sim.reset()
            np.testing.assert_allclose(sim.commanded_qpos, sim.data.qpos)
        finally:
            source.release.set()
            window.close()


if __name__ == "__main__":
    unittest.main()
