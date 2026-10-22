"""Mounted views must follow their own body and refresh with the displayed state."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import unittest

import mujoco
import numpy as np
from PySide6 import QtWidgets
from test_scene_startup import WaitingSource
from test_scenes import light, sonic

CAMERAS = ("robot_head", "robot_left_wrist", "robot_right_wrist")


class RobotCameraTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_mounts_follow_wrist_and_root_rotation(self):
        for loader in (light, sonic):
            with self.subTest(loader=loader.__name__):
                model = loader.build_model(scene="pulse")
                ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, n) for n in CAMERAS]
                self.assertTrue(all(i >= 0 for i in ids), "Missing body-mounted cameras")
                data = mujoco.MjData(model)
                mujoco.mj_forward(model, data)
                before = data.cam_xmat[ids].copy()
                data.joint("left_wrist_roll_joint").qpos[0] = .7
                mujoco.mj_forward(model, data)
                np.testing.assert_allclose(data.cam_xmat[ids[0]], before[0])
                np.testing.assert_allclose(data.cam_xmat[ids[2]], before[2])
                self.assertFalse(np.allclose(data.cam_xmat[ids[1]], before[1]))
                poses, rotations = data.cam_xpos[ids].copy(), data.cam_xmat[ids].reshape(3, 3, 3).copy()
                origin = data.qpos[:3].copy()
                data.qpos[3:7] = [np.sqrt(.5), 0, 0, np.sqrt(.5)]
                mujoco.mj_forward(model, data)
                yaw = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
                np.testing.assert_allclose(data.cam_xpos[ids], (poses - origin) @ yaw.T + origin, atol=1e-7)
                np.testing.assert_allclose(data.cam_xmat[ids].reshape(3, 3, 3), yaw @ rotations, atol=1e-7)

    def test_wrist_cameras_look_up_from_below_and_frame_all_fingers(self):
        for loader in (light, sonic):
            model = loader.build_model()
            data = mujoco.MjData(model)
            for curled in (False, True):
                for side in ("left", "right"):
                    for finger in range(1, 6):
                        for joint in range(1, 5):
                            target = (.65 if finger == 1 else (0.0 if joint == 2 else 1.0)) if curled else 0.0
                            name = f"{side}_finger{finger}_joint{joint}"
                            data.joint(name).qpos[0] = np.clip(target, *model.joint(name).range)
                mujoco.mj_forward(model, data)
                for side in ("left", "right"):
                    with self.subTest(loader=loader.__name__, side=side, curled=curled):
                        wrist = data.body(side + "_wrist_yaw_link")
                        camera = data.camera("robot_" + side + "_wrist")
                        rotation = wrist.xmat.reshape(3, 3)
                        position = (camera.xpos - wrist.xpos) @ rotation
                        self.assertLess(position[2], -.03, "Camera should be under the wrist")
                        forward = -camera.xmat.reshape(3, 3)[:, 2] @ rotation
                        self.assertGreater(forward[2], .2, "Camera should look up toward fingertips")
                        fovy = model.camera("robot_" + side + "_wrist").fovy[0]
                        tangent = np.tan(np.radians(fovy / 2))
                        for finger in range(1, 6):
                            tip = data.body(f"{side}_finger{finger}_link4")
                            point = tip.xpos + tip.xmat.reshape(3, 3) @ [0, 0, .025]
                            local = (point - camera.xpos) @ camera.xmat.reshape(3, 3)
                            self.assertGreater(-local[2], .01)
                            self.assertLess(abs(local[0] / (-local[2] * tangent * 16/9)), .9)
                            self.assertLess(abs(local[1] / (-local[2] * tangent)), .9)
                            # Check actual finger mesh bounds as well as the fingertip;
                            # projecting only joint origins missed clipped thumb bases.
                            prefix = f"{side}_finger{finger}_"
                            for gid in range(model.ngeom):
                                if (not model.body(model.geom_bodyid[gid]).name.startswith(prefix)
                                        or model.geom_group[gid] != 1
                                        or model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH):
                                    continue
                                mid = model.geom_dataid[gid]
                                start, count = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
                                points = (model.mesh_vert[start:start + count]
                                          @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid])
                                projected = (points - camera.xpos) @ camera.xmat.reshape(3, 3)
                                depth = -projected[:, 2]
                                self.assertGreater(depth.min(), .01)
                                normalized = projected[:, :2] / (depth[:, None] * tangent * [16/9, 1])
                                self.assertLess(np.abs(normalized).max(), .95,
                                                "Finger geometry clipped at image edge")

    def test_windows_render_three_views_and_refresh_without_stepping(self):
        for loader in (light, sonic):
            with self.subTest(loader=loader.__name__):
                source = WaitingSource()
                if loader is sonic:
                    window = sonic.Window(source, scene="pulse")
                    model, data = window.sim.model, window.sim.data
                else:
                    model = light.build_model(scene="pulse")
                    data = mujoco.MjData(model)
                    window = light.TuningWindow(model, data, source, light.TUNING_DEFAULTS.copy())
                window.timer.stop()
                try:
                    self.assertTrue(hasattr(window, "robot_cameras"), "Missing robot camera panel")
                    panel = window.robot_cameras
                    images = [view.image.copy() for view in panel.views.values()]
                    self.assertEqual(len(images), 3)
                    self.assertTrue(all(not image.isNull() for image in images))
                    self.assertNotEqual(images[0], images[1])
                    self.assertNotEqual(images[1], images[2])
                    window.camera_selector.setCurrentIndex(0)
                    clips = (model.vis.map.znear, model.vis.map.zfar)
                    before = [view.image.copy() for view in panel.views.values()]
                    data.joint("left_wrist_roll_joint").qpos[0] += .7
                    mujoco.mj_forward(model, data)
                    state, time = data.qpos.copy(), data.time
                    window.render_frame()
                    self.assertNotEqual(before[1], panel.views["robot_left_wrist"].image)
                    self.assertEqual(data.time, time)
                    np.testing.assert_array_equal(data.qpos, state)
                    self.assertEqual(clips, (model.vis.map.znear, model.vis.map.zfar))
                    self.assertFalse(panel.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW])
                    window.resize(1120, 720)
                    window.show()
                    self.app.processEvents()
                    self.assertLessEqual(window.width(), 1120)
                    self.assertLessEqual(window.height(), 720)
                    for view in panel.views.values():
                        self.assertTrue(view.isVisible())
                        self.assertGreaterEqual(view.width(), 160)
                finally:
                    source.release.set()
                    window.close()


if __name__ == "__main__":
    unittest.main()
