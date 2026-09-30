"""Serialize MuJoCo models' visible geoms for the web renderer.

A scene is assembled from several "parts" (one MuJoCo model per fly). The
first part contributes the ground; flies share identical meshes, so each mesh
is sent once, keyed by its name without the per-fly prefix.
"""

import base64

import mujoco
import numpy as np

# Binary WebSocket messages start with a uint32 tag.
TAG_POSE = 1
TAG_SPIKES = 2
FRAME_HEADER = 3  # time, rtf, number of projectiles


def _b64(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def visible_geoms(model: mujoco.MjModel, include_world: bool = True) -> np.ndarray:
    """Indices of geoms worth drawing: visual groups 0-2, not fully transparent.
    `include_world=False` drops geoms of the world body (the ground plane)."""
    keep = (model.geom_group <= 2) & (model.geom_rgba[:, 3] > 0)
    if not include_world:
        keep &= model.geom_bodyid != 0
    return np.flatnonzero(keep)


def _unprefixed(name: str | None) -> str:
    return (name or "").rsplit("/", 1)[-1]


def scene_description(parts: list[tuple[mujoco.MjModel, np.ndarray, int]]) -> dict:
    """parts: (model, geom ids, fly index or -1 for scenery)."""
    geoms = []
    meshes = {}
    for model, geom_ids, fly in parts:
        for gid in geom_ids:
            gtype = int(model.geom_type[gid])
            mesh_key = None
            if gtype == mujoco.mjtGeom.mjGEOM_MESH:
                mid = int(model.geom_dataid[gid])
                mesh_key = _unprefixed(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mid)) or str(mid)
                if mesh_key not in meshes:
                    v0, nv = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
                    f0, nf = model.mesh_faceadr[mid], model.mesh_facenum[mid]
                    meshes[mesh_key] = {
                        "vertices": _b64(model.mesh_vert[v0 : v0 + nv].astype(np.float32)),
                        "faces": _b64(model.mesh_face[f0 : f0 + nf].astype(np.int32)),
                    }
            rgba = model.geom_rgba[gid]
            # Geoms using a material take their color from it.
            mat_id = int(model.geom_matid[gid])
            if mat_id >= 0:
                rgba = model.mat_rgba[mat_id]
            geoms.append({
                # Drop the "<fly name>/" prefix: the renderer styles by body part.
                "name": _unprefixed(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)),
                "fly": fly if model.geom_bodyid[gid] != 0 else -1,
                "type": gtype,
                "size": model.geom_size[gid].tolist(),
                "rgba": [float(c) for c in rgba],
                "mesh": mesh_key,
                "body": int(model.geom_bodyid[gid]),
            })
    return {"geoms": geoms, "meshes": meshes}


def frame_bytes(
    parts: list[tuple[mujoco.MjData, np.ndarray, dict]],
    time: float,
    rtf: float,
    projectiles: list | None = None,
) -> bytes:
    """[uint32 TAG_POSE] then float32: [header (FRAME_HEADER)],
    [(xpos[3], xmat[9]) per geom, parts in order], [(x, y, z, r) per projectile].

    parts: (data, geom ids, overrides), where overrides maps geom ids to
    (pos, 3x3 mat) drawn instead of the simulated pose."""
    poses = []
    for data, geom_ids, overrides in parts:
        xpos = data.geom_xpos[geom_ids]
        xmat = data.geom_xmat[geom_ids]
        if overrides:
            xpos, xmat = xpos.copy(), xmat.copy()
            for row, gid in enumerate(geom_ids):
                if gid in overrides:
                    xpos[row], m = overrides[gid]
                    xmat[row] = m.reshape(9)
        poses.append(np.concatenate([xpos, xmat], axis=1))
    projectiles = projectiles or []
    header = np.array([time, rtf, len(projectiles)], dtype=np.float32)
    pose = np.concatenate(poses).astype(np.float32)
    proj = np.array(projectiles, dtype=np.float32).reshape(-1, 4)
    return np.uint32(TAG_POSE).tobytes() + header.tobytes() + pose.tobytes() + proj.tobytes()


def spikes_bytes(indices: np.ndarray) -> bytes:
    """[uint32 TAG_SPIKES][uint32 neuron index]*"""
    return np.uint32(TAG_SPIKES).tobytes() + indices.astype(np.uint32).tobytes()
