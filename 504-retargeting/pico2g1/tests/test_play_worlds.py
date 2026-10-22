"""Catch missing scene routing, robot-index drift and premature/duplicate rewards."""
import argparse
import importlib.util
import unittest
import mujoco
import numpy as np
from test_scenes import sonic, light, ROOT

NEW_SCENES = ('strike', 'orbit', 'switchyard')

class PlayWorldTests(unittest.TestCase):
    def test_real_wuji_fingertip_contacts_new_props(self):
        # Removing the host's affinity adaptation must break actual contact generation.
        for scene, body in [('strike','strike_ball'), ('orbit','puck_cyan'),
                            ('switchyard','switch_lever')]:
            with self.subTest(scene=scene):
                m=sonic.build_model(scene=scene);d=mujoco.MjData(m)
                mujoco.mj_forward(m,d)
                fingertip=int(np.flatnonzero(m.geom_contype==2)[0])
                bid=m.body(body).id
                props=np.flatnonzero((m.geom_bodyid==bid) & (m.geom_group==3))
                gid=int(props[-1])
                # Shallow surface overlap isolates masks without degenerate coincident
                # mesh/primitive centers in the narrow-phase collision solver.
                y={'strike':-.25,'orbit':-.14,'switchyard':-.09}[scene]
                offset=d.geom_xpos[gid]-d.geom_xpos[fingertip]+[0,y,0]
                d.qpos[:3]+=offset;mujoco.mj_forward(m,d)
                self.assertTrue(any(fingertip in (c.geom1,c.geom2)
                                    and gid in (c.geom1,c.geom2) for c in d.contact),
                                'The imported Wuji fingertip cannot contact the prop')

    def test_new_scenes_load_in_both_apps_and_keep_robot_indices(self):
        from sim_scenes import add_scene_arguments, PulseSession
        parser = argparse.ArgumentParser()
        add_scene_arguments(parser)
        choices = next(a.choices for a in parser._actions if a.dest == 'scene')
        self.assertTrue(set(NEW_SCENES) <= set(choices), 'New games are not selectable')
        for loader in (sonic, light):
            base = loader.build_model()
            for name in NEW_SCENES:
                with self.subTest(loader=loader.__name__, scene=name):
                    model = loader.build_model(scene=name)
                    session = PulseSession.for_model(model)
                    self.assertIsNotNone(session)
                    self.assertGreater(model.nq, base.nq)
                    self.assertEqual(model.nu, base.nu)
                    np.testing.assert_array_equal(model.jnt_qposadr[:base.njnt], base.jnt_qposadr)
                    np.testing.assert_array_equal(session.robot_qpos, np.arange(base.nq))
                    self.assertEqual(sum(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE), 1)
                    self.assertTrue(np.all(model.geom_conaffinity[model.geom_group == 3] & 2))
                    data = mujoco.MjData(model)
                    before = data.qpos[:base.nq].copy()
                    for _ in range(30):
                        session.step_kinematic(data, .02)
                    np.testing.assert_array_equal(data.qpos[:base.nq], before)
                    self.assertTrue(np.all(np.isfinite(data.qpos)))
                    self.assertEqual(session.game.score, 0, 'Resting props must not score')
                    self.assertIn(name.upper(), session.label())

    def game(self, tasks, ordered=False):
        self.assertIsNotNone(importlib.util.find_spec('sim_scenes.task_game'),
                             'Generic physical task evaluator is missing')
        from sim_scenes.task_game import TaskGame
        model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <geom type="plane" size="10 10 .1"/>
          <body name="block" pos="0 0 .1"><freejoint name="block_free"/>
            <geom name="block_collision" type="box" size=".1 .1 .1" mass="1"/>
          </body><body name="lever" pos="3 0 1"><joint name="lever_hinge" type="hinge"/>
            <geom type="box" size=".1 .1 .2" mass="1"/>
          </body><body name="hand" mocap="true" pos="0 0 3">
            <geom type="sphere" size=".08"/>
          </body></worldbody></mujoco>''')
        data = mujoco.MjData(model)
        game = TaskGame(model, {'title':'TEST', 'tasks':tasks, 'ordered':ordered})
        mujoco.mj_forward(model, data)
        return model, data, game

    def test_region_requires_full_containment_rest_release_and_dwell(self):
        task = dict(id='dock', label='Dock', kind='region', body='block',
                    joint='block_free', center=[0,0,.1], half_size=[.4,.4,.12],
                    points=20, dwell=.3, max_speed=.08)
        m,d,g = self.game([task])
        def update(t):
            d.time=t; mujoco.mj_forward(m,d); return g.update(d)
        d.joint('block_free').qpos[0]=.35
        update(0); update(1); self.assertEqual(g.score,0, 'Partly inside is not enough')
        d.joint('block_free').qpos[0]=0; d.joint('block_free').qvel[0]=1
        update(2); update(3); self.assertEqual(g.score,0)
        d.joint('block_free').qvel[:]=0; d.mocap_pos[0]=[0,0,.24]
        update(4); update(5); self.assertEqual(g.score,0, 'Held prop must not score')
        d.mocap_pos[0]=[0,0,3]
        update(6); update(6.2); self.assertEqual(g.score,0)
        self.assertEqual(len(update(6.31)),1); self.assertEqual(g.score,20)
        self.assertEqual(update(7),[]); self.assertEqual(g.score,20)
        update(0); self.assertEqual(g.score,0, 'Rewinding time resets the round')

    def test_ordered_joint_then_waypoint_and_explicit_reset(self):
        m,d,g=self.game([
            dict(id='lever',label='Lever',kind='joint',joint='lever_hinge',
                 threshold=.6,direction=1,dwell=.1,points=10),
            dict(id='finish',label='Finish',kind='waypoint',center=[1,1],radius=.4,
                 dwell=.1,points=30)], ordered=True)
        g.update(d,actor_xy=[1,1]); d.time=1;g.update(d,actor_xy=[1,1])
        self.assertEqual(g.score,0)
        d.joint('lever_hinge').qpos[0]=.7
        d.time=2;g.update(d,actor_xy=[0,0]);d.time=2.2;g.update(d,actor_xy=[0,0])
        self.assertEqual(g.score,10)
        d.time=3;g.update(d,actor_xy=[1,1]);d.time=3.2;g.update(d,actor_xy=[1,1])
        self.assertEqual(g.snapshot()['max_score'],40);self.assertEqual(g.score,40)
        g.reset();self.assertEqual(g.score,0);self.assertEqual(g.snapshot()['completed'],[])

    def test_topple_requires_tilt_and_dwell_not_translation(self):
        m,d,g=self.game([dict(id='pin',label='Pin',kind='topple',body='block',
                             tilt_degrees=55,max_height=.5,dwell=.2,points=10)])
        d.joint('block_free').qpos[0]=1;mujoco.mj_forward(m,d)
        g.update(d);d.time=1;g.update(d);self.assertEqual(g.score,0)
        d.joint('block_free').qpos[3:]=[.7071067812,0,.7071067812,0]
        mujoco.mj_forward(m,d);d.time=2;g.update(d);d.time=2.21;g.update(d)
        self.assertEqual(g.score,10)

if __name__ == '__main__':
    unittest.main()
