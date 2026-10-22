"""Compare real UI rendering with a finely sampled near shadow reference."""
import unittest
import mujoco
import numpy as np
from PySide6 import QtWidgets
from test_scene_startup import WaitingSource, sonic, light

class ShadowQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_robot_shadow_matches_fine_reference_in_both_apps(self):
        for module in (sonic, light):
            with self.subTest(app=module.__name__):
                source = WaitingSource()
                if module is sonic:
                    window = sonic.Window(source, scene='xixi', render_shadows=True)
                else:
                    model = light.build_model(scene='xixi')
                    window = light.TuningWindow(model, mujoco.MjData(model), source,
                                                light.TUNING_DEFAULTS.copy(), render_shadows=True)
                window.timer.stop()
                try:
                    window.camera.distance = 3
                    window.camera.elevation = -40
                    window.render_frame()
                    self.assertTrue(window.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW])
                    image = window.mujoco_view.image
                    actual = np.frombuffer(image.constBits(), np.uint8).reshape(
                        image.height(), image.bytesPerLine())[:, :image.width()*3]
                    actual = actual.reshape(image.height(), image.width(), 3).copy()
                    renderer = window.renderer
                    renderer._mjr_context.shadowClip = 6
                    center = window.scene_view.data.body('pelvis').xpos
                    for lamp in renderer.scene.lights[:renderer.scene.nlight]:
                        if lamp.castshadow:
                            lamp.pos[:] = [center[0], center[1], 4.24]
                    reference = renderer.render()
                    # This region includes the hands, feet, and their floor shadows.
                    crop = np.s_[350:690, 260:530]
                    error = np.abs(actual[crop].astype(float)-reference[crop]).mean()
                    # Different grids cannot match exactly. Allow less than one
                    # 8-bit intensity level while also enforcing physical texel size.
                    self.assertLess(error, .75, f'Near shadows lose fine detail: MAE={error:.3f}')
                    window.render_frame()
                    self.assertLessEqual(2*renderer._mjr_context.shadowClip /
                                         renderer._mjr_context.shadowSize, .008)
                    # Fitting the projection must not change actual illumination.
                    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
                    fitted_light = renderer.render().copy()
                    renderer.update_scene(window.scene_view.data, camera=window.camera,
                                          scene_option=window.scene_option)
                    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
                    np.testing.assert_array_equal(fitted_light, renderer.render())
                    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
                    # Move well outside the original 1.6 m shadow box, then change
                    # overview/zoom/orbit. Neither lateral nor near-plane clipping
                    # may discard the robot or the tallest office walls.
                    data = window.scene_view.data
                    data.qpos[:2] = [7, -3]
                    mujoco.mj_forward(window.scene_view.model, data)
                    for mode in (0, 1):
                        window.scene_view.select(mode)
                        window.render_frame()
                        radius = renderer._mjr_context.shadowClip
                        bounds = window.scene_view.shadows
                        for lamp in renderer.scene.lights[:renderer.scene.nlight]:
                            if lamp.castshadow:
                                self.assertTrue(np.all(np.abs(data.body('pelvis').xpos[:2]
                                                               - lamp.pos[:2]) < radius))
                                self.assertGreater(lamp.pos[2]-bounds.high[2],
                                                   renderer.scene.camera[0].frustum_near)
                    window.scene_view.orbit(110, -30)
                    window.scene_view.zoom(-5)
                    window.render_frame()
                finally:
                    source.release.set()
                    window.close()

if __name__ == '__main__':
    unittest.main()
