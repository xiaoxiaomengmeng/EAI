import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import time
import unittest
import numpy as np
import mujoco
from PySide6 import QtCore,QtGui,QtWidgets,QtTest
from test_scene_startup import WaitingSource,sonic,light
class MouseCameraTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
 def test_mouse_controls_both_windows_without_data(self):
  for module in [sonic,light]:
   with self.subTest(app=module.__name__):
    source=WaitingSource()
    if module is sonic:w=sonic.Window(source,scene='pulse')
    else:
     m=light.build_model(scene='pulse');d=mujoco.MjData(m);w=light.TuningWindow(m,d,source,light.TUNING_DEFAULTS.copy())
    w.timer.stop()
    try:
     w.camera_selector.setCurrentIndex(0)
     v=w.mujoco_view;before=v.image.cacheKey()
     wheel=QtGui.QWheelEvent(QtCore.QPointF(100,100),QtCore.QPointF(100,100),QtCore.QPoint(),QtCore.QPoint(0,120),QtCore.Qt.MouseButton.NoButton,QtCore.Qt.KeyboardModifier.NoModifier,QtCore.Qt.ScrollPhase.NoScrollPhase,False)
     self.app.sendEvent(v,wheel)
     self.assertEqual(w.camera.type,mujoco.mjtCamera.mjCAMERA_FREE)
     self.assertNotEqual(before,v.image.cacheKey())
     distance=w.camera.distance;self.app.sendEvent(v,wheel);self.assertLess(w.camera.distance,distance)
     az,el=w.camera.azimuth,w.camera.elevation
     QtTest.QTest.mousePress(v,QtCore.Qt.MouseButton.LeftButton,pos=QtCore.QPoint(100,100))
     move=QtGui.QMouseEvent(QtCore.QEvent.Type.MouseMove,QtCore.QPointF(180,150),QtCore.QPointF(180,150),QtCore.Qt.MouseButton.NoButton,QtCore.Qt.MouseButton.LeftButton,QtCore.Qt.KeyboardModifier.NoModifier)
     self.app.sendEvent(v,move);QtTest.QTest.mouseRelease(v,QtCore.Qt.MouseButton.LeftButton,pos=QtCore.QPoint(180,150))
     self.assertNotEqual(w.camera.azimuth,az);self.assertNotEqual(w.camera.elevation,el)
     frozen=w.camera.lookat.copy();w.render_frame();np.testing.assert_allclose(w.camera.lookat,frozen)
     az,el=w.camera.azimuth,w.camera.elevation;self.app.sendEvent(v,move)
     self.assertEqual(w.camera.azimuth,az);self.assertEqual(w.camera.elevation,el)
     data=w.sim.data if module is sonic else w.data
     self.assertEqual(data.time,0)
     w.camera_selector.setCurrentIndex(1);self.assertEqual(w.scene_view.mode,1)
     w.camera_selector.setCurrentIndex(0);self.assertEqual(w.camera.type,mujoco.mjtCamera.mjCAMERA_FIXED)
    finally:source.release.set();w.close()
 def test_orbit_still_repaints_after_playback_finishes(self):
  source=WaitingSource()
  w=sonic.Window(source,scene='pulse')
  w.timer.stop()
  try:
   w.latest.finished=True
   w.consume_frame()
   v=w.mujoco_view;before=v.image.cacheKey()
   w.orbit_camera(35,10)
   deadline=time.time()+10
   while w.async_result is None and time.time()<deadline:time.sleep(.01)
   self.assertIsNotNone(w.async_result,'worker never rendered the orbit')
   w.consume_frame()
   self.assertNotEqual(before,v.image.cacheKey(),'drag after the replay must repaint the view')
  finally:source.release.set();w.close()
 def test_detaching_overview_keeps_eye_and_targets_robot(self):
  from sim_scenes import SceneView
  m=sonic.build_model(scene='pulse');d=mujoco.MjData(m);c=mujoco.MjvCamera();view=SceneView(m,d,c);view.select(0)
  sc=mujoco.MjvScene(m,maxgeom=2000)
  def pose():
   mujoco.mjv_updateScene(m,d,mujoco.MjvOption(),None,c,mujoco.mjtCatBit.mjCAT_ALL,sc)
   return np.mean([x.pos for x in sc.camera],axis=0),sc.camera[0].forward.copy()
  eye,fwd=pose();self.assertTrue(hasattr(view,'orbit'),'Orbit implementation missing');view.orbit(0,0);eye2,fwd2=pose()
  np.testing.assert_allclose(eye2,eye,atol=1e-4)
  target=d.xpos[m.body("pelvis").id]+view.robot_offset
  np.testing.assert_allclose(c.lookat,target)
  np.testing.assert_allclose(fwd2,(target-eye)/np.linalg.norm(target-eye),atol=1e-5)
  d.qpos[:2]+=[3,2];mujoco.mj_forward(m,d);view.orbit(20,10)
  np.testing.assert_allclose(c.lookat,d.xpos[m.body("pelvis").id]+view.robot_offset)
  view.orbit(1e5,1e5);self.assertLessEqual(abs(c.elevation),89)
  view.zoom(100000);self.assertGreaterEqual(c.distance,.2)
  view.zoom(-100000);self.assertTrue(np.isfinite(c.distance))
if __name__=='__main__':unittest.main()
