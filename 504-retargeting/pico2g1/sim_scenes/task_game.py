"""Read-only, once-per-round rewards from physical state, shared by play worlds."""
import math

import mujoco
import numpy as np


class TaskGame:
    def __init__(self, model, design):
        self.model = model
        self.design = design
        self.tasks = design['tasks']
        self.reset()

    def reset(self):
        self.score = 0
        self.completed = set()
        self.since = {}
        self.last_time = None

    def snapshot(self):
        pending = [t['label'] for t in self.tasks if t['id'] not in self.completed]
        return {'score': self.score, 'max_score': sum(t['points'] for t in self.tasks),
                'completed': sorted(self.completed),
                'next_task': pending[0] if pending else '全部完成'}

    def _region(self, task, data):
        model = self.model
        body = model.body(task['body']).id
        weld = int(model.body_weldid[body])
        ids = np.flatnonzero((model.body_weldid[model.geom_bodyid] == weld)
                             & ((model.geom_contype != 0) | (model.geom_conaffinity != 0)))
        if not len(ids):
            return False
        lower, upper = [], []
        for gid in ids:
            rotation = data.geom_xmat[gid].reshape(3, 3)
            size = model.geom_size[gid]
            kind = model.geom_type[gid]
            if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
                extent = np.full(3, size[0])
            elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
                extent = np.sqrt(rotation[:, 0]**2 + rotation[:, 1]**2) * size[0]
                extent += np.abs(rotation[:, 2]) * size[1]
            elif kind == mujoco.mjtGeom.mjGEOM_BOX:
                extent = np.abs(rotation) @ size
            else:
                raise ValueError('Region tasks require box, sphere or cylinder colliders')
            lower.append(data.geom_xpos[gid] - extent)
            upper.append(data.geom_xpos[gid] + extent)
        center, half = np.asarray(task['center']), np.asarray(task['half_size'])
        if not (np.all(np.min(lower, axis=0) >= center - half)
                and np.all(np.max(upper, axis=0) <= center + half)):
            return False
        if np.linalg.norm(data.joint(task['joint']).qvel) > task.get('max_speed', .08):
            return False
        # A pad is complete only after releasing the prop, including mocap hands.
        for contact in data.contact:
            if contact.geom1 in ids:
                other = contact.geom2
            elif contact.geom2 in ids:
                other = contact.geom1
            else:
                continue
            other_weld = int(model.body_weldid[model.geom_bodyid[other]])
            if other_weld != weld and (model.body_dofnum[other_weld] > 0
                                      or model.body_mocapid[other_weld] >= 0):
                return False
        return True

    def _condition(self, task, data, actor_xy):
        kind = task['kind']
        if kind == 'region':
            return self._region(task, data)
        if kind == 'joint':
            direction = task.get('direction', 1)
            return direction * float(data.joint(task['joint']).qpos[0]) >= direction * task['threshold']
        if kind == 'topple':
            body = data.body(task['body'])
            return (body.xmat[8] < math.cos(math.radians(task['tilt_degrees']))
                    and body.xpos[2] < task['max_height'])
        if kind == 'waypoint':
            return actor_xy is not None and np.linalg.norm(np.asarray(actor_xy)[:2]
                                                          - task['center'][:2]) < task['radius']
        raise ValueError(f'Unknown task kind: {kind}')

    def update(self, data, actor_xy=None):
        now = float(data.time)
        if self.last_time is not None and now < self.last_time:
            self.reset()
        self.last_time = now
        events = []
        for task in self.tasks:
            key = task['id']
            if key in self.completed:
                continue
            if self._condition(task, data, actor_xy):
                self.since.setdefault(key, now)
                if now - self.since[key] >= task.get('dwell', .2):
                    self.completed.add(key)
                    self.score += task['points']
                    events.append({'task': key, 'points': task['points'], 'time': now,
                                   'score': self.score})
            else:
                self.since.pop(key, None)
            if self.design.get('ordered', False):
                break  # Later tasks cannot accumulate dwell before their turn.
        return events
