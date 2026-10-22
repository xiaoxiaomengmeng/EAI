import importlib.util
import os
from pathlib import Path
import sys
import unittest
import mujoco
import numpy as np
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'g1_sonic'))
sys.path.insert(0,str(ROOT/'g1_ik'))
def module(name,rel):
 spec=importlib.util.spec_from_file_location(name,ROOT/rel);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
sonic=module('sonic_scene_test','g1_sonic/launch_g1_sonic.py')
light=module('light_scene_test','g1_ik/launch_g1_ik.py')
class SceneTests(unittest.TestCase):
 def test_both_loaders_keep_robot_joint_addresses(self):
  for loader in [sonic,light]:
   with self.subTest(loader=loader.__name__):
    base=loader.build_model();pulse=loader.build_model(scene='pulse')
    self.assertEqual(pulse.nq,base.nq+31);self.assertEqual(pulse.nv,base.nv+27);self.assertEqual(pulse.nu,base.nu)
    np.testing.assert_array_equal(pulse.jnt_qposadr[:base.njnt],base.jnt_qposadr)
    self.assertEqual(sum(pulse.geom_type==mujoco.mjtGeom.mjGEOM_PLANE),1)
    self.assertGreater(pulse.vis.map.zfar*pulse.stat.extent,60)
    self.assertGreaterEqual(pulse.vis.map.znear*pulse.stat.extent,.05)
    d=mujoco.MjData(pulse);mujoco.mj_forward(pulse,d);np.testing.assert_allclose(d.body('pelvis').xpos[:2],[0,-3.8])
    self.assertEqual(pulse.dof_frictionloss[pulse.joint('button_cyan_slide').dofadr[0]],0)
 def test_kinematic_mode_advances_props_without_moving_robot(self):
  from sim_scenes import PulseSession
  m=sonic.build_model(scene='pulse');d=mujoco.MjData(m);s=PulseSession(m)
  before=d.qpos[:76].copy()
  for _ in range(60):s.step_kinematic(d,.02)
  np.testing.assert_array_equal(d.qpos[:76],before)
  self.assertGreater(d.time,1);self.assertAlmostEqual(d.body('game_ball').xpos[2],.22,delta=.005)
 def test_sonic_reset_restores_props_and_score(self):
  sim=sonic.Simulator(scene='pulse');sim.data.joint('button_cyan_slide').qpos[0]=.025
  sim.pulse.update(sim.data);self.assertEqual(sim.pulse.game.score,10)
  sim.data.joint('game_ball_free').qpos[:3]=[0,0,2];sim.reset()
  self.assertEqual(sim.pulse.game.score,0)
  np.testing.assert_allclose(sim.data.joint('game_ball_free').qpos[:3],[-6,-.8,.23])
 def test_ik_handles_extra_prop_degrees_of_freedom(self):
  m=light.build_model(scene='pulse');d=mujoco.MjData(m)
  ik=light.FullBodyIK(m,d,light.TUNING_DEFAULTS.copy())
  rest,_=sonic.load_skeleton();pose=np.zeros((24,7));pose[:,:3]=rest[sonic.SMPL_OUTPUT_JOINTS];pose[:,6]=1
  before=d.qpos[76:].copy();self.assertTrue(np.isfinite(ik.solve(pose)))
  pose[0,0]+=.01;self.assertTrue(np.isfinite(ik.solve(pose)))
  np.testing.assert_array_equal(d.qpos[76:],before)
 def test_old_xixi_alias_and_invalid_scene(self):
  from sim_scenes import resolve_scene
  self.assertEqual(resolve_scene(True,None),'xixi')
  self.assertEqual(resolve_scene(False,None),'empty')
  with self.assertRaises(ValueError):resolve_scene(True,'pulse')
  with self.assertRaises(ValueError):sonic.build_model(scene='not-a-scene')
 def test_render_options_retire_collision_copies_of_visual_meshes(self):
  """The G1 XML pairs every visual mesh with an identical collision copy;
  options must stop drawing the copies without touching the visuals."""
  from sim_scenes import PulseSession,scene_render_options
  plain=sonic.build_model(scene='strike');pulse_plain=PulseSession.for_model(plain)
  before=sum(1 for g in range(plain.ngeom) if scene_render_options(pulse_plain).geomgroup[plain.geom_group[g]])
  m=sonic.build_model(scene='strike');pulse=PulseSession.for_model(m)
  tuned=scene_render_options(pulse,m)
  visible=lambda o:sum(1 for g in range(m.ngeom) if o.geomgroup[m.geom_group[g]])
  self.assertLessEqual(visible(tuned),before-60)
  moved=np.flatnonzero(m.geom_group==5)
  self.assertGreaterEqual(len(moved),60)
  self.assertTrue(((m.geom_contype[moved]>0)|(m.geom_conaffinity[moved]>0)).all())
  visual=np.flatnonzero((m.geom_contype==0)&(m.geom_conaffinity==0))
  for gid in moved:
   self.assertTrue(any(v!=gid and m.geom_bodyid[v]==m.geom_bodyid[gid] and m.geom_type[v]==m.geom_type[gid]
                       and np.array_equal(m.geom_pos[v],m.geom_pos[gid]) and np.array_equal(m.geom_quat[v],m.geom_quat[gid])
                       and (m.geom_dataid[v]==m.geom_dataid[gid] or (m.geom_size[v]>=m.geom_size[gid]).all()) for v in visual),
                   f'geom {gid} hidden without a covering visual')
  ankle=m.body('left_ankle_roll_link').id
  for gid in np.flatnonzero(m.geom_bodyid==ankle):
   self.assertNotEqual(m.geom_group[gid],5)  # contact spheres have no visual twin
  again=scene_render_options(pulse,m)
  self.assertEqual(visible(again),visible(tuned))
 def test_redundant_collider_cover_requires_pose_and_per_axis_size(self):
  """A cover must share placement and enclose every axis, not just match a norm.

  Both fixtures below have a same-body, same-type visual whose size norm equals
  the collider's, so the old rule retired them; each actually sticks out (one by
  an offset, one along z) and must stay visible. The third is a genuine cover.
  """
  from sim_scenes import hide_redundant_colliders
  xml=("<mujoco><worldbody>"
       "<body name='off' pos='0 0 1'>"
       "<geom name='voff' type='sphere' size='0.1' pos='0 0 0' contype='0' conaffinity='0' group='1'/>"
       "<geom name='coff' type='sphere' size='0.1' pos='0.3 0 0' contype='1' conaffinity='1' group='0'/>"
       "</body>"
       "<body name='protrude' pos='0 0 2'>"
       "<geom name='vpro' type='box' size='0.2 0.2 0.05' pos='0 0 0' contype='0' conaffinity='0' group='1'/>"
       "<geom name='cpro' type='box' size='0.1 0.1 0.25' pos='0 0 0' contype='1' conaffinity='1' group='0'/>"
       "</body>"
       "<body name='enclosed' pos='0 0 3'>"
       "<geom name='venc' type='box' size='0.2 0.2 0.2' pos='0 0 0' contype='0' conaffinity='0' group='1'/>"
       "<geom name='cenc' type='box' size='0.1 0.1 0.1' pos='0 0 0' contype='1' conaffinity='1' group='0'/>"
       "</body></worldbody></mujoco>")
  m=mujoco.MjModel.from_xml_string(xml)
  self.assertEqual(hide_redundant_colliders(m),5)
  self.assertNotEqual(m.geom_group[m.geom('coff').id],5)  # same size, different offset
  self.assertNotEqual(m.geom_group[m.geom('cpro').id],5)  # equal norm, protrudes along z
  self.assertEqual(m.geom_group[m.geom('cenc').id],5)     # genuinely enclosed duplicate
if __name__=='__main__':unittest.main()
