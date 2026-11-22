"""Appearance shifts (the paper's perturbations), applied to the sim's model at run time.

Render-only (physics untouched), so the same state can be rendered with and without the shift:
the policy's action on the unshifted render is an exact, free label for repairing the shifted
policy (labels.cf_label). Spec: ';'-separated items
  light=s            scale the scene light's and the headlight's diffuse by s
  table=r,g,b        tint the table top material (multiplies its texture)
  cube=r,g,b         tint the cube material
  walls=r,g,b        tint the walls material (background)
  cam=dx,dy,dz,yaw   move the agentview camera (m) and turn it about world z (deg)
  wcam=dx,dy,dz,yaw  the same for the wrist camera, in the hand's frame
  roll=deg / wroll=deg   turn the agentview / wrist camera about its viewing axis
  fov=deg / wfov=deg     set the agentview / wrist camera's vertical field of view
  objalpha=a         task object's opacity in every camera (Lift cube, Square nut, ToolHang tool; 0: invisible;
                     physics unchanged); cubealpha is the same
robosuite rebuilds the model on every reset (hard_reset), so call on() again after each reset.
"""
from __future__ import annotations

import numpy as np

MATERIALS = {'table': 'table_ceramic', 'cube': 'cube_redwood_mat', 'walls': 'walls_mat'}
OBJECTS = (('cube_redwood_mat', 'cube_g'), ('SquareNut_bmetal', 'SquareNut_g'),   # (material, geom prefix)
           ('tool_hole1_steel_mat', 'tool_'), ('tool_hole2_steel_mat', 'tool_'),
           ('tool_handle_steel_mat', 'tool_'))                                    # ToolHang: the tool


def parse(spec: str) -> list[tuple[str, np.ndarray]]:
    out = []
    for item in filter(None, (s.strip() for s in spec.split(';'))):
        key, val = item.split('=')
        out.append((key.strip(), np.array([float(v) for v in val.split(',')])))
    return out


def yaw_quat(q: np.ndarray, deg: float) -> np.ndarray:
    """wxyz quaternion q turned by deg about world z: rz * q."""
    h = np.radians(deg) / 2
    c, s = np.cos(h), np.sin(h)
    w, x, y, z = q
    return np.array([c * w - s * z, c * x - s * y, c * y + s * x, c * z + s * w])


def roll_quat(q: np.ndarray, deg: float) -> np.ndarray:
    """wxyz quaternion q turned by deg about its own z (a camera's viewing axis): q * rz."""
    h = np.radians(deg) / 2
    c, s = np.cos(h), np.sin(h)
    w, x, y, z = q
    return np.array([w * c - z * s, x * c + y * s, y * c - x * s, z * c + w * s])


class Nuisance:
    """on(env) applies the spec (remembering what it overwrote), off(env) restores it."""

    def __init__(self, spec: str) -> None:
        self.items, self.saved = parse(spec), []

    def on(self, env) -> None:
        import mujoco
        m = env.sim.model._model
        self.saved = []
        for key, v in self.items:
            if key == 'light':
                self.saved += [('light_diffuse', None, m.light_diffuse.copy()),
                               ('headlight', None, m.vis.headlight.diffuse.copy())]
                m.light_diffuse[:] *= v[0]
                m.vis.headlight.diffuse[:] = m.vis.headlight.diffuse * v[0]
            elif key in MATERIALS:
                i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, MATERIALS[key])
                self.saved.append(('mat_rgba', i, m.mat_rgba[i].copy()))
                m.mat_rgba[i, :3] = v
            elif key in ('cubealpha', 'objalpha'):      # 0 = the object vanishes from every camera
                for mat, prefix in OBJECTS:
                    i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, mat)
                    if i < 0:
                        continue
                    self.saved.append(('mat_rgba', i, m.mat_rgba[i].copy()))
                    m.mat_rgba[i, 3] = v[0]
                    for g in range(m.ngeom):            # its plain-coloured geoms too
                        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ''
                        if name.startswith(prefix) and m.geom_matid[g] < 0:
                            self.saved.append(('geom_rgba', g, m.geom_rgba[g].copy()))
                            m.geom_rgba[g, 3] = min(m.geom_rgba[g, 3], v[0])
            elif key in ('roll', 'wroll', 'fov', 'wfov'):
                name = 'agentview' if key in ('roll', 'fov') else 'robot0_eye_in_hand'
                i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, name)
                if key.endswith('fov'):
                    self.saved.append(('cam_fovy', i, m.cam_fovy[i].copy()))
                    m.cam_fovy[i] = v[0]
                else:                                   # about the camera's own viewing axis
                    self.saved.append(('cam_quat', i, m.cam_quat[i].copy()))
                    m.cam_quat[i] = roll_quat(m.cam_quat[i], v[0])
            elif key in ('cam', 'wcam'):
                name = 'agentview' if key == 'cam' else 'robot0_eye_in_hand'
                i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, name)
                self.saved += [('cam_pos', i, m.cam_pos[i].copy()), ('cam_quat', i, m.cam_quat[i].copy())]
                m.cam_pos[i] += v[:3]
                m.cam_quat[i] = yaw_quat(m.cam_quat[i], v[3] if len(v) > 3 else 0.0)
            else:
                raise ValueError(f'unknown nuisance {key!r}')
        env.sim.forward()

    def off(self, env) -> None:
        m = env.sim.model._model
        for name, i, v in reversed(self.saved):
            if name == 'headlight':
                m.vis.headlight.diffuse[:] = v
            elif i is None:
                getattr(m, name)[:] = v
            else:
                getattr(m, name)[i] = v
        self.saved = []
        env.sim.forward()
