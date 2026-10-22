"""Collapsing capture UI frees space without interrupting a recording."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from pathlib import Path
import tempfile
import unittest

from PySide6 import QtCore, QtTest
from test_recording_controls import RecordingWindowTestCase, source_frame
from test_scenes import light, sonic


class CapturePanelTests(RecordingWindowTestCase):
    def test_toggle_both_windows_keeps_capture_running(self):
        for module in (light, sonic):
            with self.subTest(module=module.__name__), tempfile.TemporaryDirectory() as tmp:
                window, source = self.make_window(module)
                try:
                    self.assertTrue(hasattr(window, "capture_panel"), "Missing collapsible capture panel")
                    panel = window.capture_panel
                    window.show()
                    self.app.processEvents()
                    self.assertTrue(panel.expanded)
                    self.assertIs(panel.toggle.parentWidget(), window.mujoco_view,
                                  "Capture toggle must overlay the main view, not consume a header row")
                    self.assertLessEqual(panel.toggle.width(), 28)
                    self.assertLessEqual(panel.toggle.height(), 28)
                    for view in window.robot_cameras.views.values():
                        self.assertGreaterEqual(view.height(), int(view.width() * 9 / 16) - 1)
                    height = window.mujoco_view.height()
                    self.assertTrue(window.recording.start_recording(Path(tmp) / "capture"))
                    images = [view.image.cacheKey() for view in window.robot_cameras.views.values()]
                    QtTest.QTest.mouseClick(panel.toggle, QtCore.Qt.MouseButton.LeftButton)
                    self.app.processEvents()
                    self.assertFalse(panel.expanded)
                    self.assertFalse(window.robot_cameras.isVisible())
                    self.assertFalse(window.recording.isVisible())
                    self.assertTrue(window.source_view.isVisible())
                    self.assertGreater(window.mujoco_view.height(), height)
                    window.latest.put(source_frame(window, module))
                    window.consume_frame()
                    self.assertEqual(window.recording.session.frame_count, 1)
                    self.assertIn("录制中", panel.toggle.toolTip())
                    self.assertEqual(images, [v.image.cacheKey() for v in window.robot_cameras.views.values()])
                    QtTest.QTest.mouseClick(panel.toggle, QtCore.Qt.MouseButton.LeftButton)
                    self.app.processEvents()
                    self.assertTrue(panel.expanded)
                    self.assertTrue(window.robot_cameras.isVisible())
                    self.assertTrue(window.recording.isVisible())
                    self.assertNotEqual(images, [v.image.cacheKey() for v in window.robot_cameras.views.values()])
                    self.assertEqual(window.recording.session.frame_count, 1, "Expansion must not add a sample")
                    window.recording.stop_recording()
                    self.wait_for_export(window.recording)
                    self.assertIn("已保存", panel.toggle.toolTip())
                finally:
                    source.release.set()
                    window.close()
                    self.app.processEvents()


if __name__ == "__main__":
    unittest.main()
