import unittest
import mujoco,numpy as np
from test_scenes import sonic,light
from sim_scenes import SceneView
class RobotFramingTests(unittest.TestCase):
 def test_default_robot_frame_independent_of_scene(self):
  for loader in [sonic,light]:
   frames=[]
   for name in ['empty','xixi','pulse']:
    m=loader.build_model(scene=name);d=mujoco.MjData(m);c=mujoco.MjvCamera();view=SceneView(m,d,c)
    self.assertEqual(view.mode,1)
    self.assertEqual(c.type,mujoco.mjtCamera.mjCAMERA_FREE)
    frames.append([c.distance,* (c.lookat-d.xpos[m.body('pelvis').id]),c.azimuth,c.elevation,m.vis.global_.fovy])
   np.testing.assert_allclose(frames[0],frames[1],atol=1e-6)
   np.testing.assert_allclose(frames[0],frames[2],atol=1e-6)
 def test_robot_dimensions_scale_camera_distance(self):
  def view_for(scale):
   m=mujoco.MjModel.from_xml_string(f'<mujoco><worldbody><body name="pelvis" pos="0 0 {scale}"><geom type="box" group="1" size="{.2*scale} {.15*scale} {scale}"/></body><geom type="box" pos="1000 0 0" size="50 50 50"/></worldbody></mujoco>')
   d=mujoco.MjData(m);c=mujoco.MjvCamera();view=SceneView(m,d,c);return c.distance,c.lookat.copy()
  a,ca=view_for(1);b,cb=view_for(2)
  self.assertAlmostEqual(b/a,2,places=5);np.testing.assert_allclose(cb,ca*2)
if __name__=='__main__':unittest.main()
