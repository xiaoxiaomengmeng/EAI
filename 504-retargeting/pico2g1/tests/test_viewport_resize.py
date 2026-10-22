"""Main-camera rendering follows the available viewport rather than a square."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import unittest
from PySide6 import QtTest
from test_recording_controls import RecordingWindowTestCase
from test_scenes import light, sonic


class ViewportResizeTests(RecordingWindowTestCase):
    def test_main_image_fills_resized_and_collapsed_viewports(self):
        for module in (light, sonic):
            with self.subTest(module=module.__name__):
                window, source = self.make_window(module)
                try:
                    window.show()
                    for width, height, expanded in ((1500, 900, True), (1500, 900, False),
                                                     (1240, 800, False), (1240, 800, True)):
                        window.capture_panel.set_expanded(expanded)
                        window.resize(width, height)
                        QtTest.QTest.qWait(180)
                        self.app.processEvents()
                        viewport = window.mujoco_view.contentsRect()
                        image = window.mujoco_view.image
                        self.assertEqual((image.width(), image.height()),
                                         (viewport.width(), viewport.height()),
                                         "Main camera still renders a square/old size")
                        pixmap = window.mujoco_view.pixmap()
                        self.assertEqual(pixmap.size(), viewport.size())
                        self.assertEqual(window.scene_view.data.time, 0, "Resizing must not step physics")
                    renderer = window.renderer
                    window.render_frame()
                    self.assertIs(window.renderer, renderer, "Unchanged viewport should reuse its GL renderer")
                finally:
                    source.release.set()
                    window.close()
                    self.app.processEvents()


if __name__ == "__main__":
    unittest.main()
