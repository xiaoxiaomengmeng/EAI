"""Real windows must render every scene even while the source is blocked."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import threading,unittest
import mujoco,numpy as np
from PySide6 import QtWidgets
from test_scenes import sonic,light
class WaitingSource:
 def __init__(self):self.release=threading.Event();self.entered=threading.Event()
 def __iter__(self):return self
 def __next__(self):self.entered.set();self.release.wait(15);raise StopIteration
class StartupTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
 def test_scene_is_visible_before_any_data_for_both_apps(self):
  for name in ['empty','xixi','pulse','strike','orbit','switchyard']:
   for app in ['sonic','light']:
    with self.subTest(scene=name,app=app):
     source=WaitingSource()
     if app=='sonic':w=sonic.Window(source,scene=name)
     else:
      m=light.build_model(scene=name);d=mujoco.MjData(m);mujoco.mj_forward(m,d)
      w=light.TuningWindow(m,d,source,light.TUNING_DEFAULTS.copy())
     w.timer.stop()
     try:
      self.assertTrue(source.entered.wait(2))
      self.assertIsNotNone(w.mujoco_view.image,'Scene still depends on a data frame')
      self.assertFalse(w.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW])
      self.assertEqual(w.renderer._mjr_context.shadowFBO, 0,
                       'Default rendering must not allocate a shadow buffer')
      self.assertIn('场景已就绪',w.statusBar().currentMessage())
      self.assertIn('等待',w.statusBar().currentMessage())
      self.assertEqual(w.camera.type,mujoco.mjtCamera.mjCAMERA_FREE)
      self.assertEqual(w.camera_selector.currentIndex(),1)
      w.camera_selector.setCurrentIndex(0)
      if name!='empty':self.assertEqual(w.camera.type,mujoco.mjtCamera.mjCAMERA_FIXED)
      w.camera_selector.setCurrentIndex(1)
      self.assertEqual(w.camera.type,mujoco.mjtCamera.mjCAMERA_FREE)
      w.camera_selector.setCurrentIndex(0)
      data=w.sim.data if app=='sonic' else w.data
      self.assertEqual(data.time,0,'Preview must not step simulation')
      # A connection failure must retain the rendered scene and its error status.
      import contextlib,io
      with w.latest.lock:
       w.latest.error='synthetic connection failure';w.latest.finished=True
      with contextlib.redirect_stderr(io.StringIO()):w.consume_frame()
      self.assertIn('数据流错误',w.statusBar().currentMessage())
      self.assertIsNotNone(w.mujoco_view.image)
      w.camera_selector.setCurrentIndex(1)
      self.assertEqual(w.camera.type,mujoco.mjtCamera.mjCAMERA_FREE)

     finally:source.release.set();w.close()
if __name__=='__main__':unittest.main()
