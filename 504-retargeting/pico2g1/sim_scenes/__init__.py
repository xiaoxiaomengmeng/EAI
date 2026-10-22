"""Optional environment loading shared by the two local simulators."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from .shadows import DirectionalShadows
from .registry import INTERACTIVE_SCENES, SCENE_CHOICES
PULSE = Path(__file__).resolve().parents[1] / 'assets' / 'pulse_arena'

def resolve_scene(xixi_scene: bool = False, scene: str | None = None) -> str:
    if scene is not None and scene not in SCENE_CHOICES:
        raise ValueError(f'Unknown scene: {scene}')
    if xixi_scene and scene not in (None, 'xixi'):
        raise ValueError('--xixi-scene cannot be combined with another scene')
    return scene or ('xixi' if xixi_scene else 'empty')

def add_scene_arguments(parser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--scene', choices=SCENE_CHOICES, default=None,
                       help='optional environment (default: empty)')
    group.add_argument('--xixi-scene', action='store_true', help='alias for --scene xixi')
    parser.add_argument('--render-shadows', action='store_true',
                        help='render scene shadows (default: off; allocates a shadow map)')

def add_pulse_scene(root: ET.Element, asset: ET.Element) -> None:
    """Backward-compatible entry for the original PULSE scene."""
    add_interactive_scene(root, asset, 'pulse')

def add_interactive_scene(root: ET.Element, asset: ET.Element, scene: str) -> None:
    """Append environment joints after the robot so policy indices stay unchanged."""
    directory = INTERACTIVE_SCENES[scene]
    robot_mask = 1
    for geom in root.findall('.//geom'):
        robot_mask |= int(geom.get('contype', '1'))
    world = root.find('worldbody')
    for wb in root.findall('worldbody'):
        for geom in list(wb):
            if geom.tag == 'geom' and geom.get('name') == 'floor':
                wb.remove(geom)
    for filename in ('environment.xml', 'interactive.xml'):
        document = ET.parse(directory / filename).getroot()
        for item in document.findall('./asset/*'):
            item = copy.deepcopy(item)
            if item.get('file'):
                item.set('file', str(directory / 'assets' / item.get('file')))
            asset.append(item)
        for item in document.findall('./worldbody/*'):
            if item.tag == 'light':  # keep the simulator's existing illumination
                continue
            item = copy.deepcopy(item)
            for geom in ([item] if item.tag == 'geom' else item.findall('.//geom')):
                if geom.get('group') == '3':
                    geom.set('conaffinity', str(robot_mask))
            # Robot joint defaults must not change the spring's calibrated rest behavior.
            for joint in item.findall('.//joint'):
                joint.set('frictionloss', '0')
                joint.set('armature', '0')
            world.append(item)
        for sensor in document.findall('./sensor/*'):
            target = root.find('sensor')
            if target is None:
                target = ET.SubElement(root, 'sensor')
            target.append(copy.deepcopy(sensor))
    # Robot-only extent (0.8 m) clips the far half of this 18 m environment.
    statistic = root.find('statistic')
    if statistic is None:
        statistic = ET.SubElement(root, 'statistic')
    statistic.set('extent', '20')
    statistic.set('center', '0 0 1')
    visual = root.find('visual')
    if visual is None:
        visual = ET.SubElement(root, 'visual')
    clipping = visual.find('map')
    if clipping is None:
        clipping = ET.SubElement(visual, 'map')
    clipping.set('znear', '.005')
    clipping.set('zfar', '10')
    spawn = json.loads((directory / 'design.json').read_text())['spawn']
    pelvis = next(b for b in world.findall('body') if b.get('name') == 'pelvis')
    pos = [float(x) for x in pelvis.get('pos', '0 0 0').split()]
    pelvis.set('pos', ' '.join(map(str, [spawn[0], spawn[1], pos[2] + spawn[2]])))

class PulseSession:
    """Prop physics and score for policy stepping or kinematic pose playback."""
    def __init__(self, model, scene='pulse'):
        self.model = model
        self.scene = scene
        design = json.loads((INTERACTIVE_SCENES[scene] / 'design.json').read_text())
        if scene == 'pulse':
            from .pulse_game import ArenaGame
            self.game = ArenaGame(model, design)
            joint_names = ['button_cyan_slide', 'button_orange_slide', 'button_lime_slide',
                           'cube_cyan_free', 'cube_orange_free', 'cube_lime_free', 'game_ball_free']
        else:
            from .task_game import TaskGame
            self.game = TaskGame(model, design)
            joint_names = design['prop_joints']
        qmask = np.ones(model.nq, dtype=bool)
        vmask = np.ones(model.nv, dtype=bool)
        for name in joint_names:
            joint = model.joint(name)
            kind = model.jnt_type[joint.id]
            nq, nv = {int(mujoco.mjtJoint.mjJNT_FREE): (7, 6),
                      int(mujoco.mjtJoint.mjJNT_BALL): (4, 3)}.get(int(kind), (1, 1))
            q = int(joint.qposadr[0]); v = int(joint.dofadr[0])
            qmask[q:q + nq] = False
            vmask[v:v + nv] = False
        self.robot_qpos = np.flatnonzero(qmask)
        self.robot_dofs = np.flatnonzero(vmask)

    @classmethod
    def for_model(cls, model):
        for scene in INTERACTIVE_SCENES:
            if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'play_scene_' + scene) >= 0:
                return cls(model, scene)
        return cls(model) if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'game_ball') >= 0 else None

    def update(self, data):
        return self.game.update(data, actor_xy=data.body('pelvis').xpos[:2])

    def reset(self):
        self.game.reset()

    def step_kinematic(self, data, duration):
        """Hold the robot pose while props respond; this is not balanced locomotion."""
        pose = data.qpos[self.robot_qpos].copy()
        velocity = data.qvel[self.robot_dofs].copy()
        controls = data.ctrl.copy()
        data.ctrl[:] = 0
        try:
            for _ in range(max(1, round(duration / self.model.opt.timestep))):
                data.qpos[self.robot_qpos] = pose
                data.qvel[self.robot_dofs] = 0
                mujoco.mj_step(self.model, data)
                data.qpos[self.robot_qpos] = pose
                data.qvel[self.robot_dofs] = 0
                mujoco.mj_forward(self.model, data)
                self.update(data)
        finally:
            data.qpos[self.robot_qpos] = pose
            data.qvel[self.robot_dofs] = velocity
            data.ctrl[:] = controls
            mujoco.mj_forward(self.model, data)

    def label(self):
        state = self.game.snapshot()
        if self.scene != 'pulse':
            return f"{self.scene.upper()} {state['score']}/{state['max_score']} · {state['next_task']}"
        return f"PULSE {state['score']}/135 · 按钮 {state['button_index']}/3 · 路点 {state['waypoint_index']}/5"

def hide_redundant_colliders(model):
    """Move collision shapes that only copy a same-body visual into a spare group.

    Unitree's G1 XML pairs every visual mesh (group 1, contype=0) with an
    identical collision mesh (default group 0, contype=1). Both submit draw
    calls, so every link renders twice and close-ups z-fight between the two
    coincident surfaces. A collision geom moves to the spare group - 5, or 4
    when 5 is taken - only when a pure-visual geom on the same body shares its
    type and placement and either references the same mesh or encloses it along
    every axis, so its look is guaranteed to be covered. Shapes without such a
    counterpart (ankle contact spheres, shoulder cylinders) stay visible.

    Returns the group render options should hide, or None when nothing moved.
    A stateless repeat call re-detects the spare group, so the same options
    can be built twice without moving anything further.
    """
    spare = None
    for candidate in (5, 4):
        ids = np.flatnonzero(model.geom_group == candidate)
        if not len(ids):
            spare = candidate
            break
        if ((model.geom_contype[ids] > 0) | (model.geom_conaffinity[ids] > 0)).all():
            return candidate  # an earlier call already moved the colliders here
    if spare is None:
        return None
    visual = np.flatnonzero((model.geom_contype == 0) & (model.geom_conaffinity == 0))
    collide = np.flatnonzero(((model.geom_contype > 0) | (model.geom_conaffinity > 0))
                             & np.isin(model.geom_group, (0, 1, 2, 3)))
    if not len(visual) or not len(collide):
        return None
    same_body = model.geom_bodyid[collide, None] == model.geom_bodyid[visual]
    same_type = model.geom_type[collide, None] == model.geom_type[visual]
    # Same placement, or the cover promise is false: a collider copying a visual
    # mesh at a different offset is geometry of its own, not a hidden duplicate.
    same_pose = ((model.geom_pos[collide, None] == model.geom_pos[visual]).all(axis=-1)
                 & (model.geom_quat[collide, None] == model.geom_quat[visual]).all(axis=-1))
    same_mesh = (model.geom_dataid[collide, None] == model.geom_dataid[visual])
    same_mesh &= model.geom_dataid[collide, None] >= 0
    # Per axis, not by norm: a collider box (0.1, 0.1, 0.25) and a visual box
    # (0.2, 0.2, 0.05) share a norm yet the collider protrudes along z.
    enclosed = (model.geom_size[collide, None] <= model.geom_size[visual]).all(axis=-1)
    covered = (same_body & same_type & same_pose & (same_mesh | enclosed)).any(axis=1)
    moved = collide[covered]
    if not len(moved):
        return None
    model.geom_group[moved] = spare
    return spare

def scene_render_options(pulse, model=None):
    option = mujoco.MjvOption()
    if pulse:
        option.geomgroup[3] = 0
        option.sitegroup[:] = 0
    if model is not None:
        hidden = hide_redundant_colliders(model)
        if hidden is not None:
            option.geomgroup[hidden] = 0
    return option

class LockChain:
    """Reusable context manager entering several locks in a fixed order.

    Render paths take one short snapshot per image (main view, three body
    cameras), and a generator-based context manager cannot be re-entered.
    """
    __slots__ = ('locks',)

    def __init__(self, *locks):
        self.locks = locks

    def __enter__(self):
        # SIGINT is delivered at the bytecode boundary between two acquires, and
        # `with` skips __exit__ when __enter__ raises, so the held prefix is
        # unwound here. The record stays local: one chain is shared by all threads.
        acquired = []
        try:
            for lock in self.locks:
                lock.acquire()
                acquired.append(lock)
        except BaseException:
            for lock in reversed(acquired):
                lock.release()
            raise
        return self

    def __exit__(self, *exc_info):
        for lock in reversed(self.locks):
            lock.release()
        return False

class SceneView:
    """Initial environment overview and explicit robot-follow camera, independent of input."""
    def __init__(self, model, data, camera, *, render_shadows=False):
        self.model, self.data, self.camera = model, data, camera
        self.pelvis = model.body('pelvis').id
        self.overview_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'overview')
        self.mode = 0
        self.default_fovy = float(model.vis.global_.fovy)
        mujoco.mj_forward(model, data)
        self.shadows = DirectionalShadows(model, data, enabled=render_shadows)
        # Imported environments can be much larger than the robot's authored extent.
        if self.overview_id >= 0:
            model.stat.extent = max(model.stat.extent,
                                    float(np.max(self.shadows.high - self.shadows.low)))
        self._measure_robot()
        self.select(1)

    def _measure_robot(self):
        """Fit only visual geometry under pelvis; environment bounds never enter framing."""
        model, data = self.model, self.data
        bodies = {self.pelvis}
        for body in range(self.pelvis + 1, model.nbody):
            if int(model.body_parentid[body]) in bodies:
                bodies.add(body)
        mask = np.isin(model.geom_bodyid, list(bodies))
        visual = mask & (model.geom_group == 1)
        ids = np.flatnonzero(visual if visual.any() else mask)
        if not len(ids):
            raise ValueError("Robot pelvis subtree contains no geometry to frame")
        corners = []
        signs = np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
        for gid in ids:
            local = model.geom_aabb[gid,:3] + signs * model.geom_aabb[gid,3:]
            corners.extend(local @ data.geom_xmat[gid].reshape(3,3).T + data.geom_xpos[gid])
        corners = np.asarray(corners)
        center = (corners.min(axis=0) + corners.max(axis=0)) / 2
        self.robot_offset = center - data.xpos[self.pelvis]
        self.robot_fovy, self.robot_azimuth, self.robot_elevation = 45., 135., -12.
        az, el = np.radians([self.robot_azimuth, self.robot_elevation])
        forward = np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)])
        right = np.cross(forward, [0,0,1]); right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        relative = corners - center
        # Both applications render a square texture, letterboxed by their QLabel.
        tangent = np.tan(np.radians(self.robot_fovy/2)) * .68
        distance = np.maximum(np.abs(relative @ right), np.abs(relative @ up)) / tangent - relative @ forward
        self.robot_distance = max(.2, float(distance.max()))

    def select(self, mode):
        self.mode = mode
        cam, model = self.camera, self.model
        model.vis.global_.fovy = self.default_fovy
        if mode == 0 and self.overview_id >= 0:
            cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            cam.fixedcamid = self.overview_id
            distance = float(np.linalg.norm(self.data.cam_xpos[self.overview_id] - self.data.xpos[self.pelvis]))
        else:
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.fixedcamid = -1
            cam.distance = self.robot_distance * 1.6
            cam.azimuth, cam.elevation = self.robot_azimuth, self.robot_elevation
            model.vis.global_.fovy = self.robot_fovy
            cam.lookat[:] = self.data.xpos[self.pelvis] + self.robot_offset
            distance = cam.distance
        model.vis.map.znear = max(.02, distance * .003) / model.stat.extent
        model.vis.map.zfar = max(100, distance + 4 * model.stat.extent) / model.stat.extent

    def update(self):
        if self.camera.type == mujoco.mjtCamera.mjCAMERA_FREE:
            target = self.data.xpos[self.pelvis] + self.robot_offset
            self.camera.lookat[:] = target

    def prepare(self, renderer, option):
        """Snapshot the current state into this renderer's scene (cheap, lockable)."""
        self.update()
        renderer.update_scene(self.data, camera=self.camera, scene_option=option)
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = self.shadows.enabled
        self.shadows.prepare(renderer, self.data)

    def render(self, renderer, option):
        """Prepare and draw; the returned array belongs to the caller.

        mujoco allocates the render output per call (verified on 3.12.0, the
        floor in the setup scripts), so a caller may keep this array and read it
        later from another thread without copying it: the next render writes a
        different buffer. Re-check `image.base is None` before lowering that
        floor, since a renderer that reuses an internal buffer would silently
        hand deferred consumers a newer or half-written picture.
        """
        self.prepare(renderer, option)
        return renderer.render()


    def _free_camera(self):
        cam = self.camera
        if cam.type != mujoco.mjtCamera.mjCAMERA_FIXED:
            return
        eye = self.data.cam_xpos[cam.fixedcamid].copy()
        target = self.data.xpos[self.pelvis] + self.robot_offset
        distance = max(.2, float(np.linalg.norm(target - eye)))
        forward = (target - eye) / distance
        cam.lookat[:] = target
        cam.distance = distance
        cam.azimuth = float(np.degrees(np.arctan2(forward[1], forward[0])))
        cam.elevation = float(np.degrees(np.arcsin(np.clip(forward[2], -1, 1))))
        self.model.vis.global_.fovy = self.model.cam_fovy[cam.fixedcamid]
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.fixedcamid = -1

    def _clipping(self):
        distance = self.camera.distance
        self.model.vis.map.znear = max(.005, distance * .003) / self.model.stat.extent
        self.model.vis.map.zfar = max(100, distance + 4 * self.model.stat.extent) / self.model.stat.extent

    def orbit(self, dx, dy):
        self._free_camera()
        self.update()
        self.camera.azimuth = (self.camera.azimuth - dx * .3) % 360
        self.camera.elevation = float(np.clip(self.camera.elevation - dy * .3, -89, 89))
        self._clipping()

    def zoom(self, steps):
        self._free_camera()
        self.update()
        self.camera.distance = float(np.clip(self.camera.distance * np.exp(-np.clip(steps, -50, 50) * .12),
                                            .2, max(500, self.model.stat.extent * 8)))
        self._clipping()
