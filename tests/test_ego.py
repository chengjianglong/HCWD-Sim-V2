"""Egocentric kitchen data: spec, state/action maths and the two dataset exporters (no Blender needed)."""
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from worldsim import ego, room_spec  # noqa: E402

KITCHEN = os.path.join(ROOT, "configs", "kitchen.json")


# ------------------------------------------------------------------ spec
def test_kitchen_spec_loads_and_task_fits_on_counter():
    spec = room_spec.load_spec(KITCHEN)
    T = spec["task"]
    counter = next(f for f in spec["furniture"] if f["name"] == T["counter"])
    assert counter.get("rot", 0) == 0, "task frame assumes the cook faces north"
    cx, cy = counter["center"]
    w, d = counter["size"][:2]
    front = cy - d / 2
    bw, bd = T["board"]["size"][:2]
    jx, jy = T["board"]["jitter"][:2]
    for x in T["cook_x"]:  # both extremes of where the cook may stand
        root_y = front - T["cook_back"]
        bx, by = x + T["board"]["at"][0], root_y + T["board"]["at"][1]
        r = math.hypot(bw, bd) / 2  # any board yaw
        assert cx - w / 2 <= bx - r - jx and bx + r + jx <= cx + w / 2, f"board off the counter at cook_x={x}"
        assert front <= by - bd / 2 - jy, "board overhangs the counter front"
        assert by + bd / 2 + jy <= cy + d / 2 - 0.02, "board hits the backsplash"
        kx, ky = x + T["knife"]["at"][0], root_y + T["knife"]["at"][1]
        assert abs(kx - bx) < bw / 2 and abs(ky - by) < bd / 2, "knife must rest on the board"
    assert T["worktop_z"] == pytest.approx(counter["size"][2])


# ------------------------------------------------------------------ rotations
def test_rotation_roundtrips():
    rng = np.random.default_rng(0)
    v = rng.normal(size=(500, 3))
    v = v / np.linalg.norm(v, axis=1, keepdims=True) * rng.uniform(0, np.pi, (500, 1))
    v = np.concatenate([v, [[np.pi, 0, 0], [0, 0, 0], [0, 1e-9, 0]]])
    R = ego.rotvec_to_mat(v)
    assert np.allclose(ego.rotvec_to_mat(ego.mat_to_rotvec(R)), R, atol=1e-9)
    q = ego.mat_to_quat(R)
    assert np.allclose(ego.quat_to_mat(q), R, atol=1e-9)
    # 90 deg about z: x -> y
    Rz = ego.rotvec_to_mat(np.array([0, 0, np.pi / 2]))
    assert np.allclose(Rz @ [1, 0, 0], [0, 1, 0], atol=1e-12)
    assert np.allclose(ego.mat_to_rotvec(Rz), [0, 0, np.pi / 2])


# ------------------------------------------------------------------ synthetic episodes
def quat_from_rotvec(v):
    return ego.mat_to_quat(ego.rotvec_to_mat(np.asarray(v, float)))


def fake_meta(n=12, head_yaw=0.0, head_pos=(0.0, 0.0, 1.5), fps=10, views=("head", "right_wrist", "scene_front")):
    """Right hand moves +x 1 cm/frame and turns 0.05 rad/frame about z; grip closes at frame 4, opens at 9."""
    frames = []
    Rh = ego.rotvec_to_mat(np.array([0, 0, head_yaw]))
    cam = np.eye(4)
    cam[:3, :3] = Rh
    cam[:3, 3] = head_pos
    for t in range(n):
        grip = 1.0 if 4 <= t < 9 else 0.0
        frames.append(dict(
            frame_index=t, scene_frame=1 + 3 * t, timestamp=t / fps, phase="chop0" if 4 <= t < 9 else "idle",
            right_hand=dict(pos=[0.01 * t, 0.4, 0.95], quat_wxyz=list(quat_from_rotvec([0, 0, 0.05 * t]))),
            left_hand=dict(pos=[-0.1, 0.4, 0.95], quat_wxyz=[1, 0, 0, 0]),
            grip=dict(right=grip, left=grip), knife_tip=[0, 0, 0], ik_err=dict(right=0, left=0),
            cameras={v: cam.tolist() for v in views}))
    return dict(episode_index=0, instruction="chop the red apple into slices", fps=fps, n_frames=n, views=list(views),
                intrinsics={v: dict(model="pinhole", fx=100, fy=100, cx=8, cy=8, width=16, height=16) for v in views},
                task=dict(apple_color="red", n_cuts=3, pieces=4), frames=frames)


def test_world_frame_state_and_actions():
    m = fake_meta()
    s, a = ego.hand_state_actions(m, "right", "world")
    assert s.shape == (12, 8) and a.shape == (12, 7)
    assert np.allclose(s[:, 0], 0.01 * np.arange(12), atol=1e-6)
    assert np.allclose(a[:-1, :3], [[0.01, 0, 0]] * 11, atol=1e-6)
    assert np.allclose(a[:-1, 3:6], [[0, 0, 0.05]] * 11, atol=1e-6)
    assert np.allclose(a[-1, :6], 0)  # nothing after the last frame
    # gripper: action is the grip at t+1 (+1 closed); state carries finger widths 0.04 open .. 0 closed
    assert list(a[:, 6]) == [-1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, -1]
    assert np.allclose(s[0, 6:], [0.04, -0.04]) and np.allclose(s[5, 6:], [0, 0])


def test_head_frame_rotates_deltas_and_ignores_head_pose():
    yaw = np.pi / 2
    m = fake_meta(head_yaw=yaw, head_pos=(1.0, 2.0, 1.6))
    s, a = ego.hand_state_actions(m, "right", "head")
    # world +x motion seen from a camera yawed +90 deg about world z is -y in the camera frame
    assert np.allclose(a[:-1, :3], [[0, -0.01, 0]] * 11, atol=1e-6)
    # rotation about world z is rotation about camera z here (yaw only)
    assert np.allclose(a[:-1, 3:6], [[0, 0, 0.05]] * 11, atol=1e-6)
    # the head-frame position is the world position relative to the head, rotated into it
    p0 = np.array([0.0, 0.4, 0.95]) - np.array([1.0, 2.0, 1.6])
    Rz = ego.rotvec_to_mat(np.array([0, 0, yaw]))
    assert np.allclose(s[0, :3], Rz.T @ p0, atol=1e-6)


def test_both_hands_shapes():
    s, a = ego.both_hands(fake_meta())
    assert s.shape == (12, 20) and a.shape == (12, 14)
    assert np.allclose(s[:, 18], s[:, 19])  # both grips in the synthetic data


# ------------------------------------------------------------------ exporters
def write_raw(root, n_eps=2, n=12, res=16):
    rng = np.random.default_rng(1)
    for e in range(n_eps):
        m = fake_meta(n=n + e)
        m["episode_index"] = e
        if e:
            m["instruction"] = "slice the green apple on the cutting board"
        d = os.path.join(root, f"episode_{e:06d}")
        for v in m["views"]:
            os.makedirs(os.path.join(d, v))
            for i in range(m["n_frames"]):
                Image.fromarray(rng.integers(0, 255, (res, res, 3), dtype=np.uint8)).save(os.path.join(d, v, f"{i:06d}.png"))
        with open(os.path.join(d, "episode.json"), "w") as f:
            json.dump(m, f)


def run(script, *args):
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", script), *args], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout


def test_vlaqwen_export(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "vq"
    write_raw(str(raw))
    run("ego_to_vlaqwen.py", "--raw", str(raw), "--out", str(out), "--res", "32")
    meta = json.load(open(out / "meta.json"))
    assert meta["total"] == 25 and meta["res"] == 32
    assert meta["episode_starts"] == [0, 12] and meta["episode_lengths"] == [12, 13]
    assert set(meta["tasks"].values()) == {"chop the red apple into slices", "slice the green apple on the cutting board"}
    arr = np.load(out / "arrays.npz")
    assert arr["state"].shape == (25, 8) and arr["actions"].shape == (25, 7)
    assert list(arr["episode_index"][10:14]) == [0, 0, 1, 1] and arr["frame_index"][12] == 0
    for k in ("mean", "std", "q01", "q99"):
        assert len(meta["stats"]["state"][k]) == 8 and len(meta["stats"]["actions"][k]) == 7
    imgs = np.memmap(out / "images.u8", dtype=np.uint8, mode="r", shape=(25, 2, 32, 32, 3))
    assert imgs[13, 1].std() > 0  # wrist view written


def test_lerobot_export(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    raw, out = tmp_path / "raw", tmp_path / "lr"
    write_raw(str(raw))
    run("ego_to_lerobot.py", "--raw", str(raw), "--out", str(out))
    info = json.load(open(out / "meta" / "info.json"))
    assert info["total_episodes"] == 2 and info["total_frames"] == 25 and info["total_tasks"] == 2
    assert info["total_videos"] == 6
    for v in ("head", "right_wrist", "scene_front"):
        assert info["features"][f"observation.images.{v}"]["dtype"] == "video"
        assert (out / "videos" / "chunk-000" / f"observation.images.{v}" / "episode_000001.mp4").stat().st_size > 0
    t = pq.read_table(out / "data" / "chunk-000" / "episode_000001.parquet").to_pydict()
    assert len(t["index"]) == 13 and t["index"][0] == 12 and t["episode_index"][0] == 1
    assert len(t["observation.state"][0]) == 8 and len(t["action.both_hands"][0]) == 14
    assert len(t["observation.extrinsics.head"][0]) == 16 and t["next.done"][-1] and not t["next.done"][0]
    eps = [json.loads(l) for l in open(out / "meta" / "episodes.jsonl")]
    assert [e["length"] for e in eps] == [12, 13]
    stats = json.load(open(out / "meta" / "stats.json"))
    assert len(stats["action"]["q99"]) == 7
