"""Serialize a MuJoCo model's visible geoms for the web renderer."""

import base64

import mujoco
import numpy as np


def _b64(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def visible_geoms(model: mujoco.MjModel) -> np.ndarray:
    """Indices of geoms worth drawing: visual groups 0-2, not fully transparent."""
    keep = (model.geom_group <= 2) & (model.geom_rgba[:, 3] > 0)
    return np.flatnonzero(keep)


def scene_description(model: mujoco.MjModel, geom_ids: np.ndarray) -> dict:
    geoms = []
    used_meshes = set()
    for gid in geom_ids:
        gtype = int(model.geom_type[gid])
        mesh_id = int(model.geom_dataid[gid]) if gtype == mujoco.mjtGeom.mjGEOM_MESH else -1
        rgba = model.geom_rgba[gid]
        # Geoms using a material take their color from it.
        mat_id = int(model.geom_matid[gid])
        if mat_id >= 0:
            rgba = model.mat_rgba[mat_id]
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        geoms.append(
            {
                # Drop the "<fly name>/" prefix: the renderer styles by body part.
                "name": name.rsplit("/", 1)[-1],
                "type": gtype,
                "size": model.geom_size[gid].tolist(),
                "rgba": [float(c) for c in rgba],
                "mesh": str(mesh_id) if mesh_id >= 0 else None,
                "body": int(model.geom_bodyid[gid]),
            }
        )
        if mesh_id >= 0:
            used_meshes.add(mesh_id)

    meshes = {}
    for mid in used_meshes:
        v0, nv = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        f0, nf = model.mesh_faceadr[mid], model.mesh_facenum[mid]
        meshes[str(mid)] = {
            "vertices": _b64(model.mesh_vert[v0 : v0 + nv].astype(np.float32)),
            "faces": _b64(model.mesh_face[f0 : f0 + nf].astype(np.int32)),
        }
    return {"geoms": geoms, "meshes": meshes}


# Binary WebSocket messages start with a uint32 tag.
TAG_POSE = 1
TAG_SPIKES = 2
FRAME_HEADER = 3  # time, rtf, number of projectiles


def frame_bytes(
    data: mujoco.MjData,
    geom_ids: np.ndarray,
    rtf: float,
    overrides: dict | None = None,
    projectiles: list | None = None,
) -> bytes:
    """[uint32 TAG_POSE] then float32: [header (FRAME_HEADER)],
    [(xpos[3], xmat[9]) per geom], [(x, y, z, radius) per projectile].

    `overrides` maps model geom ids to (pos, 3x3 mat) drawn instead of the
    simulated pose."""
    xpos = data.geom_xpos[geom_ids]
    xmat = data.geom_xmat[geom_ids]
    if overrides:
        xpos, xmat = xpos.copy(), xmat.copy()
        for row, gid in enumerate(geom_ids):
            if gid in overrides:
                xpos[row], m = overrides[gid]
                xmat[row] = m.reshape(9)
    projectiles = projectiles or []
    header = np.array([data.time, rtf, len(projectiles)], dtype=np.float32)
    pose = np.concatenate([xpos, xmat], axis=1).astype(np.float32)
    proj = np.array(projectiles, dtype=np.float32).reshape(-1, 4)
    return np.uint32(TAG_POSE).tobytes() + header.tobytes() + pose.tobytes() + proj.tobytes()


def spikes_bytes(indices: np.ndarray) -> bytes:
    """[uint32 TAG_SPIKES][uint32 neuron index]*"""
    return np.uint32(TAG_SPIKES).tobytes() + indices.astype(np.uint32).tobytes()
