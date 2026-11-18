"""Egocentric episodes -> VLA state/action arrays (pure numpy, shared by the dataset exporters and tests).

A raw episode (blender/gen_ego_episodes.py) logs, per frame, each wrist's world pose (hand bone: position of
the wrist joint + orientation), each hand's grip in [0, 1], and every camera's cam-to-world pose (OpenCV axes).

State/action conventions (LIBERO-compatible layout, so policies pretrained on LIBERO line up):
- state  [8]: hand position (3), hand orientation as axis-angle (3), gripper "finger widths" (w, -w),
              w = 0.04 * (1 - grip), i.e. 0.04 open .. 0 closed, like LIBERO's two finger joints.
- action [7]: delta position (3), delta rotation as axis-angle (3) from frame t to t+1, gripper -1 open / +1 closed
              (the grip at t+1, thresholded at 0.5).
Poses are expressed in a reference frame: `head` (the head camera at time t, the usual egocentric choice; deltas
for t -> t+1 are both expressed in the head frame at t, so head motion is not mistaken for hand motion) or `world`.

The both-hands variants (`state_both` [20], `action_both` [14]) use [pos, 6D rotation] per hand plus both grips,
and [dpos, drot axis-angle, grip] per hand.
"""
import numpy as np

GRIP_OPEN_WIDTH = 0.04


# ------------------------------------------------------------------ rotations
def quat_to_mat(q):
    """(..., 4) quaternions (w, x, y, z) -> (..., 3, 3) rotation matrices."""
    q = np.asarray(q, np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def mat_to_quat(R):
    """(..., 3, 3) -> (..., 4) quaternions (w, x, y, z), w >= 0 (Shepperd's method, stable everywhere)."""
    R = np.asarray(R, np.float64)
    m00, m11, m22 = R[..., 0, 0], R[..., 1, 1], R[..., 2, 2]
    cands = np.stack([
        np.stack([1 + m00 + m11 + m22, R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1),
        np.stack([R[..., 2, 1] - R[..., 1, 2], 1 + m00 - m11 - m22, R[..., 0, 1] + R[..., 1, 0], R[..., 0, 2] + R[..., 2, 0]], -1),
        np.stack([R[..., 0, 2] - R[..., 2, 0], R[..., 0, 1] + R[..., 1, 0], 1 - m00 + m11 - m22, R[..., 1, 2] + R[..., 2, 1]], -1),
        np.stack([R[..., 1, 0] - R[..., 0, 1], R[..., 0, 2] + R[..., 2, 0], R[..., 1, 2] + R[..., 2, 1], 1 - m00 - m11 + m22], -1),
    ], -2)  # (..., 4 candidates, 4); candidate k has its k-th component = 4 q_k^2
    k = np.argmax(np.stack([1 + m00 + m11 + m22, 1 + m00 - m11 - m22, 1 - m00 + m11 - m22, 1 - m00 - m11 + m22], -1), -1)
    q = np.take_along_axis(cands, k[..., None, None].repeat(4, -1), -2)[..., 0, :]
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    return q * np.where(q[..., :1] < 0, -1.0, 1.0)


def mat_to_rotvec(R):
    """(..., 3, 3) -> (..., 3) axis-angle (angle in [0, pi])."""
    q = mat_to_quat(R)
    v = q[..., 1:]
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    ang = 2 * np.arctan2(n, q[..., :1])
    return np.where(n > 1e-12, v / np.where(n > 1e-12, n, 1.0) * ang, 2 * v)


def rotvec_to_mat(v):
    v = np.asarray(v, np.float64)
    ang = np.linalg.norm(v, axis=-1, keepdims=True)
    k = v / np.where(ang > 1e-12, ang, 1.0)
    K = np.zeros(v.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 2] = -k[..., 2], k[..., 1], -k[..., 0]
    K[..., 1, 0], K[..., 2, 0], K[..., 2, 1] = k[..., 2], -k[..., 1], k[..., 0]
    a = ang[..., None]
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * (K @ K)


def mat_to_6d(R):
    """First two columns, flattened (continuous rotation representation)."""
    return np.concatenate([R[..., :, 0], R[..., :, 1]], -1)


def pose(pos, quat_wxyz):
    T = np.zeros(np.shape(pos)[:-1] + (4, 4))
    T[..., :3, :3] = quat_to_mat(quat_wxyz)
    T[..., :3, 3] = pos
    T[..., 3, 3] = 1.0
    return T


# ------------------------------------------------------------------ episode -> arrays
def hand_poses(meta, hand):
    fr = meta["frames"]
    key = f"{hand}_hand"
    return pose(np.array([f[key]["pos"] for f in fr]), np.array([f[key]["quat_wxyz"] for f in fr]))


def grips(meta, hand):
    return np.array([f["grip"][hand] for f in meta["frames"]], np.float64)


def reference_frames(meta, ref="head"):
    """(T, 4, 4) world <- reference transforms."""
    n = len(meta["frames"])
    if ref == "world":
        return np.repeat(np.eye(4)[None], n, 0)
    return np.array([f["cameras"][ref] for f in meta["frames"]], np.float64)


def _in_ref(T_ref, T_world):
    return np.linalg.inv(T_ref) @ T_world


def hand_state_actions(meta, hand="right", ref="head"):
    """state [T, 8], action [T, 7] for one hand (see module docstring)."""
    Tw = hand_poses(meta, hand)
    R = reference_frames(meta, ref)
    g = grips(meta, hand)
    T = len(Tw)
    local = _in_ref(R, Tw)
    width = GRIP_OPEN_WIDTH * (1 - g)
    state = np.concatenate([local[:, :3, 3], mat_to_rotvec(local[:, :3, :3]), width[:, None], -width[:, None]], 1)
    nxt = np.concatenate([Tw[1:], Tw[-1:]], 0)
    nxt_local = _in_ref(R, nxt)  # t+1 expressed in the reference frame at t
    dpos = nxt_local[:, :3, 3] - local[:, :3, 3]
    drot = mat_to_rotvec(nxt_local[:, :3, :3] @ np.swapaxes(local[:, :3, :3], -1, -2))
    g_next = np.concatenate([g[1:], g[-1:]])
    grip_a = np.where(g_next > 0.5, 1.0, -1.0)
    action = np.concatenate([dpos, drot, grip_a[:, None]], 1)
    assert state.shape == (T, 8) and action.shape == (T, 7)
    return state.astype(np.float32), action.astype(np.float32)


def both_hands(meta, ref="head"):
    """state [T, 20] = per hand [pos 3, rot6d 6] + [grip_r, grip_l]; action [T, 14] = per hand [dpos, drot, grip]."""
    R = reference_frames(meta, ref)
    states, acts = [], []
    gs = []
    for hand in ("right", "left"):
        Tw = hand_poses(meta, hand)
        local = _in_ref(R, Tw)
        states.append(np.concatenate([local[:, :3, 3], mat_to_6d(local[:, :3, :3])], 1))
        _, a = hand_state_actions(meta, hand, ref)
        acts.append(a)
        gs.append(grips(meta, hand)[:, None])
    state = np.concatenate(states + gs, 1)
    return state.astype(np.float32), np.concatenate(acts, 1).astype(np.float32)


def stats(x):
    x = np.asarray(x, np.float64)
    return {"mean": x.mean(0).tolist(), "std": x.std(0).tolist(), "min": x.min(0).tolist(), "max": x.max(0).tolist(),
            "q01": np.quantile(x, 0.01, 0).tolist(), "q99": np.quantile(x, 0.99, 0).tolist()}


def load_raw(root):
    """Sorted list of (episode_dir, meta) under a raw output folder."""
    import json
    import os
    out = []
    for d in sorted(os.listdir(root)):
        p = os.path.join(root, d, "episode.json")
        if d.startswith("episode_") and os.path.exists(p):
            with open(p) as f:
                out.append((os.path.join(root, d), json.load(f)))
    return out
