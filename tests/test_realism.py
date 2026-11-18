"""Photo-realism / sim-to-real pieces that need no Blender, GPU or network: the asset manifest, the simulated camera
sensor (worldsim/sensor.py) and the gap metric maths (scripts/sim2real_metric.py)."""
import importlib.util
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from worldsim import sensor  # noqa: E402


# ------------------------------------------------------------------ assets manifest
def test_asset_manifest_is_consistent():
    m = json.load(open(os.path.join(ROOT, "configs", "assets.json")))
    assert m["resolution"] in ("1k", "2k", "4k")
    for slot, ids in m["textures"].items():
        assert ids and len(ids) == len(set(ids)), slot
    assert m["hdris"]["sky"], "need at least one daylight HDRI"
    for slot, s in m["models"].items():
        assert s["ids"] and s["fit"] in ("contain", "stretch"), slot
    # every furniture / object type in the specs that we replace with a model has a slot
    for t in ("sofa", "chair", "bookshelf", "plant", "stool", "tree", "shrub", "bench", "street_lamp", "bins"):
        assert t in m["models"], t


# ------------------------------------------------------------------ sensor
INTR = dict(fx=120.0, fy=120.0, cx=96.0, cy=80.0, width=192, height=160)


def clean_sensor(view="head", seed=1):
    cam = sensor.CameraSensor(view, INTR, seed)
    cam.p.update(full_well=0.0, read_noise=0.0, jpeg=0, vignette=0.0, gain=1.0, wb=(1.0, 1.0, 1.0), ca=0.0)
    cam._build_maps()
    return cam


def test_undistort_inverts_distort():
    rng = np.random.default_rng(0)
    x, y = rng.uniform(-1, 1, 500), rng.uniform(-0.8, 0.8, 500)
    for d in [(-0.2, 0.05, 0.001, -0.001), (-0.05, 0.0, 0.0, 0.0), (0.02, 0.01, 0.0005, 0.0)]:
        xd, yd = sensor.distort_norm(x, y, *d)
        xu, yu = sensor.undistort_norm(xd, yd, *d)
        assert np.allclose(xu, x, atol=1e-6) and np.allclose(yu, y, atol=1e-6)


@pytest.mark.parametrize("view", ["head", "right_wrist", "scene_front"])
def test_distorted_frame_matches_exported_intrinsics(view):
    """A point drawn in the clean render lands where the exported K + dist project it (cv2.projectPoints model)."""
    cam = clean_sensor(view)
    intr = cam.intrinsics()
    assert intr["fx"] >= INTR["fx"] and intr["dist"][0] <= 0.0  # zoomed in just enough; barrel distortion
    K_in = np.array([[INTR["fx"], 0, INTR["cx"]], [0, INTR["fy"], INTR["cy"]], [0, 0, 1.0]])
    K_out = [[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1]]
    for P in ([0.25, -0.18, 1.0], [-0.3, 0.2, 1.2], [0.0, 0.0, 2.0]):
        P = np.array([P])
        u0, v0 = (K_in @ (P[0] / P[0, 2]))[:2]  # where the ideal pinhole render has it
        img = np.zeros((INTR["height"], INTR["width"], 3), np.uint8)
        yy, xx = np.mgrid[:INTR["height"], :INTR["width"]]
        img[(xx - u0) ** 2 + (yy - v0) ** 2 <= 2.5 ** 2] = 255
        out = cam(img).astype(float)[..., 1]
        w = out / out.sum()
        cu, cv = (w * xx).sum(), (w * yy).sum()
        pu, pv = sensor.project(K_out, intr["dist"], P)[0]
        assert abs(cu - pu) < 0.6 and abs(cv - pv) < 0.6, (view, (cu, cv), (pu, pv))


def test_sensor_is_deterministic_and_changes_the_image():
    img = (np.random.default_rng(3).uniform(0, 255, (160, 192, 3))).astype(np.uint8)
    a = sensor.CameraSensor("head", INTR, sensor.seed_for(0, "episode_000001", "head"))
    b = sensor.CameraSensor("head", INTR, sensor.seed_for(0, "episode_000001", "head"))
    c = sensor.CameraSensor("head", INTR, sensor.seed_for(0, "episode_000002", "head"))
    oa, ob = a(img), b(img)
    assert oa.dtype == np.uint8 and oa.shape == img.shape
    assert np.array_equal(oa, ob)  # same episode + view -> same device and same noise sequence
    assert a.p != c.p  # another episode -> another device
    assert np.abs(oa.astype(int) - img).mean() > 2  # it does something
    none = sensor.CameraSensor("head", INTR, 0, profile="none")
    assert np.array_equal(none(img), img) and none.intrinsics()["dist"] == [0.0] * 5


def test_camera_types():
    assert [sensor.camera_type(v) for v in ("head", "chest", "left_wrist", "scene_left")] == ["head", "chest", "wrist", "scene"]


# ------------------------------------------------------------------ gap metric
def metric():
    spec = importlib.util.spec_from_file_location("sim2real_metric", os.path.join(ROOT, "scripts", "sim2real_metric.py"))
    mod = importlib.util.module_from_spec(spec)
    pytest.importorskip("torch")
    spec.loader.exec_module(mod)
    return mod


def test_kid_and_fid_separate_shifted_sets():
    mod = metric()
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(400, 16)), rng.normal(size=(400, 16))
    c = rng.normal(0.6, 1.0, size=(400, 16))
    same, _ = mod.kid(a, b, subsets=20, size=200)
    shifted, _ = mod.kid(a, c, subsets=20, size=200)
    assert abs(same) < 0.02 and shifted > 10 * abs(same) + 0.05
    if importlib.util.find_spec("scipy"):
        assert mod.fid(a, b) < mod.fid(a, c)
