"""Offscreen Qt + actual ONNX inference + robot/arena rendering integration."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from pathlib import Path
import sys
import unittest
import numpy as np
from PySide6 import QtWidgets
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'g1_sonic'))
from g1_sonic import Window,load_skeleton,SMPL_OUTPUT_JOINTS
class PulseWindowTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
 def test_source_frame_runs_policy_with_arena_and_visible_score(self):
  w=Window(iter(()),scene='pulse');w.timer.stop()
  try:
   self.assertIsNotNone(w.mujoco_view.image)
   self.assertEqual(w.scene_option.geomgroup[3],0)
   rest,_=load_skeleton()
   reference={'smpl_joints':np.tile(rest[SMPL_OUTPUT_JOINTS],(10,1,1)).astype(np.float32),'body_quat':np.tile([1.,0,0,0],(10,1)).astype(np.float32),'joint_pos':np.zeros((10,29),dtype=np.float32)}
   pose=np.zeros((24,7));pose[:,6]=1;pose[16,:3]=[1,1,0];pose[17,:3]=[-1,1,0]
   gloves={s:np.zeros((21,3)) for s in ['left','right']}
   trackers=np.zeros((2,7));trackers[:,6]=1
   # Avoid finished-stream status obscuring the source-frame status in this test.
   with w.latest.lock:w.latest.finished=False
   w.latest.put((reference,pose,trackers,gloves,{}));w.consume_frame()
   self.assertFalse(w.failed);self.assertGreater(w.sim.data.time,0)
   self.assertIn('PULSE',w.statusBar().currentMessage())
   self.assertEqual(w.sim.model.nq,107)
   with w.latest.lock:w.latest.finished=True
   w.finished_reported=False;w.consume_frame()
   self.assertIn('PULSE',w.statusBar().currentMessage())
   w.reset();self.assertEqual(w.sim.pulse.game.score,0)
  finally:w.close()
if __name__=='__main__':unittest.main()
