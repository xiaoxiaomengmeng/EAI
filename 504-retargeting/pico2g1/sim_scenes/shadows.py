"""Fit native directional shadow maps to geometry instead of camera clip distance."""
import mujoco
import numpy as np

_SIGNS = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])


def geometry_bounds(model, data):
    """Mesh AABBs rotated into world space substantially overestimate this office.

    Use actual vertices once at startup; include visible geometry groups only.
    Infinite floor planes are receivers, not finite shadow-volume boundaries.
    """
    low, high = np.full(3, np.inf), np.full(3, -np.inf)
    ids = np.flatnonzero((model.geom_type != mujoco.mjtGeom.mjGEOM_PLANE)
                         & (model.geom_group < 3))
    for gid in ids:
        if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = model.geom_dataid[gid]
            start = model.mesh_vertadr[mid]
            points = model.mesh_vert[start:start + model.mesh_vertnum[mid]]
        else:
            points = model.geom_aabb[gid, :3] + _SIGNS * model.geom_aabb[gid, 3:]
        world = points @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]
        low = np.minimum(low, world.min(axis=0))
        high = np.maximum(high, world.max(axis=0))
    if not len(ids):
        low, high = np.full(3, -1.), np.full(3, 1.)
    return low, high


def _radius(corners, center, direction):
    relative = corners - center
    # MuJoCo's frame for a vertical light is aligned with world X/Y.
    if np.linalg.norm(direction[:2]) < 1e-6:
        return float(np.abs(relative[:, :2]).max()) + .5
    # Rotation-independent bound for lights at other angles.
    along = relative @ direction
    return float(np.linalg.norm(relative - along[:, None] * direction, axis=1).max()) + .5


class DirectionalShadows:
    """One full-scene shadow pass, with <= 8 mm texels for the bundled scenes.

    Only the directional light's shadow projection origin changes. Its direction,
    color and intensity stay intact, so fitting does not move the illumination.
    """
    def __init__(self, model, data, *, enabled=True):
        self.model = model
        self.enabled = enabled
        self.low, self.high = geometry_bounds(model, data)
        self.dynamic = np.flatnonzero((model.geom_type != mujoco.mjtGeom.mjGEOM_PLANE)
                                     & (model.geom_group < 3)
                                     & (model.body_weldid[model.geom_bodyid] != 0))
        if not enabled:
            # Must happen before Renderer allocates its GL context, not merely
            # by clearing mjRND_SHADOW after a large map has already been created.
            model.vis.quality.shadowsize = 0
            return
        corners = (self.low + self.high) / 2 + _SIGNS * (self.high - self.low) / 2
        radius = 0.
        for lid in range(model.nlight):
            if model.light_castshadow[lid] and model.light_type[lid] == mujoco.mjtLightType.mjLIGHT_DIRECTIONAL:
                radius = max(radius, _radius(corners, (self.low + self.high) / 2,
                                             data.light_xdir[lid]))
        if radius:
            # Bound memory and keep the existing single render pass. Small scenes
            # retain 4096; the 63 m office needs 8192 (about 7.8 mm per texel).
            required = 2 ** int(np.ceil(np.log2(2 * radius / .008)))
            model.vis.quality.shadowsize = max(4096, model.vis.quality.shadowsize,
                                               min(8192, required))

    def prepare(self, renderer, data):
        if not self.enabled:
            return
        low, high = self.low.copy(), self.high.copy()
        # Moving robot/props must remain inside the map, including after replay
        # translations. Static mesh bounds need no per-frame vertex traversal.
        ids = self.dynamic
        if len(ids):
            local = self.model.geom_aabb[ids, None, :3] + _SIGNS * self.model.geom_aabb[ids, None, 3:]
            world = (np.einsum('nij,nkj->nki', data.geom_xmat[ids].reshape(-1, 3, 3), local)
                     + data.geom_xpos[ids, None, :])
            low = np.minimum(low, world.min(axis=(0, 1)))
            high = np.maximum(high, world.max(axis=(0, 1)))
        center = (low + high) / 2
        corners = center + _SIGNS * (high - low) / 2
        radius = 0.
        for lamp in renderer.scene.lights[:renderer.scene.nlight]:
            if not lamp.castshadow or lamp.type != mujoco.mjtLightType.mjLIGHT_DIRECTIONAL:
                continue
            direction = lamp.dir.astype(float)
            direction /= np.linalg.norm(direction)
            radius = max(radius, _radius(corners, center, direction))
            # Shadow rendering shares the camera near plane. Keep the light's
            # projection upstream of the tallest caster even in overview mode.
            near = max(c.frustum_near for c in renderer.scene.camera)
            distance = float(np.max((center - corners) @ direction)) + near + 1.
            lamp.pos[:] = center - direction * distance
        if radius:
            self.model.vis.map.shadowclip = radius / self.model.stat.extent
            renderer._mjr_context.shadowClip = radius
