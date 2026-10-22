"""Source capture must preserve local overlays and return owned RGB pixels."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import unittest
import numpy as np
from PySide6 import QtWidgets
from test_merge_calibration import sample_pose
from test_scenes import light, sonic


class SourceStreamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_source_overlays_refresh_without_input_and_images_survive_next_render(self):
        for loader in (light, sonic):
            with self.subTest(loader=loader.__name__):
                view = loader.SourceView()
                view.resize(471, 553)  # Exercise RGB scanline padding and fractional panel edges.
                pose = sample_pose()
                trackers = pose[[20, 21]].copy()
                gloves = {side: np.zeros((21, 3)) for side in ('left', 'right')}
                try:
                    if loader is light:
                        view.set_frame(pose, trackers, gloves, sonic.source_basis(pose))
                    else:
                        view.set_frame(pose, trackers, gloves)
                    original = view.stream_frames()
                    saved = {key: image.copy() for key, image in original.items()}
                    if loader is light:
                        view.set_calibrating(True)
                    else:
                        offsets = {name: np.array([45., 30., 10.]) for name, _ in loader.ROTATION_OFFSET_JOINTS}
                        view.set_adjustment('left_wrist', offsets)
                    adjusted = view.stream_frames()
                    for key in ('source_body_front', 'source_body_side'):
                        self.assertFalse(np.array_equal(original[key], adjusted[key]), key)
                    for key in ('source_left_hand', 'source_right_hand'):
                        np.testing.assert_array_equal(original[key], adjusted[key])
                    view.resize(703, 617)
                    view.stream_frames()
                    for key in original:
                        np.testing.assert_array_equal(original[key], saved[key])
                        self.assertTrue(original[key].flags.c_contiguous)
                finally:
                    view.close()


if __name__ == '__main__':
    unittest.main()
