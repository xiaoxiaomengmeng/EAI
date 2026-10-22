"""The network must show the exact local camera state even with previews collapsed."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import http.client
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import mujoco
from PySide6 import QtWidgets
from sim_scenes.live_stream import StreamConfig
from sim_scenes.robot_cameras import CAMERA_MOUNTS
from test_live_stream import unpack
from test_scene_startup import WaitingSource
from test_scenes import light, sonic
from test_recording_controls import source_frame


class StreamingWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_both_windows_stream_current_pose_while_collapsed_and_stop_cleanly(self):
        for loader in (light, sonic):
            with self.subTest(loader=loader.__name__):
                source = WaitingSource()
                if loader is sonic:
                    window = loader.Window(source)
                    data = window.sim.data
                else:
                    model = loader.build_model()
                    data = mujoco.MjData(model)
                    window = loader.TuningWindow(model, data, source, loader.TUNING_DEFAULTS.copy())
                window.timer.stop()
                try:
                    self.assertTrue(hasattr(window, 'streaming'), 'Missing streaming controls')
                    controls = window.streaming
                    controls.config = StreamConfig(host='127.0.0.1', port=0, fps=60)
                    controls.toggle.setChecked(True)
                    self.assertTrue(controls.active)
                    server = controls.server
                    def fetch(after=0):
                        connection = http.client.HTTPConnection(*server.address, timeout=3)
                        connection.request('GET', '/api/frames?after=' + str(after))
                        response = connection.getresponse()
                        self.assertEqual(response.status, 200)
                        result = unpack(response.read())
                        connection.close()
                        return result
                    first, images = fetch()
                    self.assertEqual(len(images), 8)
                    window.capture_panel.set_expanded(False)
                    before = data.time
                    data.joint('left_wrist_roll_joint').qpos[0] += .8
                    data.joint('left_finger2_joint3').qpos[0] = 1.0
                    mujoco.mj_forward(window.sim.model if loader is sonic else model, data)
                    window.orbit_camera(35, 10)
                    changed, next_images = fetch(first['sequence'])
                    self.assertNotEqual(images['main'], next_images['main'])
                    self.assertNotEqual(images['robot_left_wrist'], next_images['robot_left_wrist'])
                    self.assertEqual(data.time, before)
                    self.assertFalse(window.capture_panel.expanded)

                    # Consume a real input frame and fetch immediately, without
                    # allowing a Qt paint event to run: no previous-pose screenshots.
                    frame = source_frame(window, loader)
                    gloves = frame[3]
                    for side in ('left', 'right'):
                        for finger in range(5):
                            for joint in range(4):
                                gloves[side][1 + finger * 4 + joint] = [(finger - 2) * .02, 0, (joint + 1) * .025]
                    window.latest.put(frame)
                    window.consume_frame()
                    raw_meta, raw_images = fetch(changed['sequence'])
                    for key in ('source_body_front', 'source_body_side', 'source_left_hand', 'source_right_hand'):
                        self.assertIn(key, raw_images)
                        self.assertNotEqual(images[key], raw_images[key], key + ' did not draw current input')
                        self.assertGreater(raw_images[key].width(), 100)
                        self.assertGreater(raw_images[key].height(), 100)

                    # Changing only the raw left glove must change only that hand
                    # preview, without needing robot hand targets or another paint.
                    gloves['left'][5:9, 0] += .08
                    window.render_frame()
                    left_meta, left_changed = fetch(raw_meta['sequence'])
                    self.assertNotEqual(raw_images['source_left_hand'], left_changed['source_left_hand'])
                    for key in ('source_body_front', 'source_body_side', 'source_right_hand'):
                        self.assertEqual(raw_images[key], left_changed[key])

                    # Finishing calibration must clear streamed guides even if
                    # playback has no next frame and no resize/paint event runs.
                    if loader is light:
                        window.enter_calibration()
                    else:
                        window.enter_adjustment()
                    overlay_meta, overlay_images = fetch(left_meta['sequence'])
                    if loader is sonic:
                        window.select_joint('left_wrist')
                        selected_meta, _ = fetch(overlay_meta['sequence'])
                        window.change_axis(0, 45.)
                        axis_meta, _ = fetch(selected_meta['sequence'])
                        window.reset_all_offsets()
                        overlay_meta, overlay_images = fetch(axis_meta['sequence'])
                    with tempfile.TemporaryDirectory() as tmp:
                        field = 'TUNING_PATH' if loader is light else 'ROTATION_OFFSETS_PATH'
                        with patch.object(loader, field, Path(tmp) / 'config.json'):
                            if loader is light:
                                window.finish_calibration()
                            else:
                                window.finish_adjustment()
                    _, cleared = fetch(overlay_meta['sequence'])
                    for key in ('source_body_front', 'source_body_side'):
                        self.assertNotEqual(overlay_images[key], cleared[key], key + ' kept old guides')
                    controls.toggle.setChecked(False)
                    self.assertTrue(server.stopped)
                    controls.toggle.setChecked(True)
                    self.assertTrue(controls.active)
                    active = controls.server
                    self.assertIsNotNone(active.wait_frame(timeout=2))
                finally:
                    source.release.set()
                    window.close()
                self.assertTrue(active.stopped)

    def test_body_cameras_render_continuously_while_streaming(self):
        """Each body camera runs on its own thread: previews stay live on
        their own, without a main render request driving them."""
        source = WaitingSource()
        window = sonic.Window(source)
        window.timer.stop()
        try:
            controls = window.streaming
            controls.config = StreamConfig(host='127.0.0.1', port=0, fps=60)
            controls.toggle.setChecked(True)
            self.assertTrue(controls.active)
            name = CAMERA_MOUNTS[0][0]
            changes, last = 0, None
            deadline = time.time() + .6
            while time.time() < deadline:
                with window.camera_lock:
                    image = window.camera_images.get(name)
                if image is not None and image is not last:
                    changes += 1
                    last = image
                time.sleep(.005)
            self.assertGreaterEqual(changes, 5,
                                    'body cameras must keep rendering while the stream is live')
        finally:
            source.release.set()
            window.close()


if __name__ == '__main__':
    unittest.main()
