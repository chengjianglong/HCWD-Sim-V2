"""Realistic camera sensors for rendered frames: the sim-to-real step that happens at export time.

Blender renders an ideal pinhole image. A real head-mounted, wrist or room camera adds lens distortion, vignetting,
exposure and white-balance error, sensor noise, chromatic aberration and compression. `CameraSensor` adds all of these
to a clean uint8 RGB frame, with parameters drawn once per (episode, view) from a per-camera-type profile, so a whole
episode is filmed by one consistent "device", and different episodes by different devices.

Geometry stays exact: the distortion is the OpenCV Brown-Conrady model (k1, k2, p1, p2, k3=0) and `intrinsics()`
returns the K and distortion coefficients of the output image, so a pose plus those intrinsics project correctly
into the distorted frame (cv2.projectPoints convention). The output focal length is reduced just enough that the
distorted image has no empty corners.

Pure numpy; the raw renders stay clean ground truth and the exporters call this (``--sensor realistic``).
"""
import hashlib
import io

import numpy as np

# Per camera type: ranges sampled uniformly per (episode, view). Distortion is barrel (k1 < 0) for wide lenses.
PROFILES = {
    "head": dict(k1=(-0.24, -0.08), k2=(0.02, 0.07), p=0.0015, vignette=(0.15, 0.40), exposure_ev=(-0.6, 0.4),
                 wb=0.08, full_well=(600, 2500), read_noise=(0.002, 0.008), ca=(0.0005, 0.0025), jpeg=(70, 92)),
    "chest": dict(k1=(-0.24, -0.08), k2=(0.02, 0.07), p=0.0015, vignette=(0.20, 0.45), exposure_ev=(-0.6, 0.4),
                  wb=0.08, full_well=(600, 2500), read_noise=(0.002, 0.008), ca=(0.0008, 0.003), jpeg=(70, 92)),
    "wrist": dict(k1=(-0.20, -0.06), k2=(0.01, 0.05), p=0.002, vignette=(0.15, 0.40), exposure_ev=(-0.7, 0.5),
                  wb=0.10, full_well=(400, 1800), read_noise=(0.003, 0.012), ca=(0.0008, 0.003), jpeg=(65, 90)),
    "scene": dict(k1=(-0.10, 0.0), k2=(0.0, 0.03), p=0.001, vignette=(0.05, 0.25), exposure_ev=(-0.4, 0.4),
                  wb=0.06, full_well=(1500, 6000), read_noise=(0.001, 0.005), ca=(0.0, 0.0015), jpeg=(75, 95)),
}
NONE = dict(k1=0.0, k2=0.0, p1=0.0, p2=0.0, vignette=0.0, gain=1.0, wb=(1.0, 1.0, 1.0), full_well=0.0,
            read_noise=0.0, ca=0.0, jpeg=0)


def camera_type(view):
    for t in ("wrist", "head", "chest"):
        if t in view:
            return t
    return "scene"


def seed_for(*keys):
    return int.from_bytes(hashlib.sha256("/".join(map(str, keys)).encode()).digest()[:8], "little")


def sample_params(view, rng):
    P = PROFILES[camera_type(view)]
    u = lambda r: float(rng.uniform(*r))  # noqa: E731
    wb = 1.0 + rng.uniform(-P["wb"], P["wb"], 3)
    wb[1] = 1.0  # white balance error relative to green
    return dict(k1=u(P["k1"]), k2=u(P["k2"]), p1=float(rng.uniform(-P["p"], P["p"])),
                p2=float(rng.uniform(-P["p"], P["p"])), vignette=u(P["vignette"]), gain=2.0 ** u(P["exposure_ev"]),
                wb=tuple(float(x) for x in wb), full_well=u(P["full_well"]), read_noise=u(P["read_noise"]),
                ca=u(P["ca"]), jpeg=int(rng.integers(*P["jpeg"])))


# ------------------------------------------------------------------------------------------------ distortion maths
def distort_norm(x, y, k1, k2, p1, p2):
    """Normalised undistorted -> distorted coordinates (OpenCV model, k3 = 0)."""
    r2 = x * x + y * y
    rad = 1 + k1 * r2 + k2 * r2 * r2
    return (x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x),
            y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y)


def undistort_norm(xd, yd, k1, k2, p1, p2, iters=20):
    """Distorted -> undistorted normalised coordinates (fixed-point iteration, as cv2.undistortPoints)."""
    x, y = np.array(xd, dtype=float), np.array(yd, dtype=float)
    for _ in range(iters):
        r2 = x * x + y * y
        rad = 1 + k1 * r2 + k2 * r2 * r2
        dx = 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        dy = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        x, y = (xd - dx) / rad, (yd - dy) / rad
    return x, y


def _bilinear(img, u, v):
    """Sample img [H, W, C] float at pixel coordinates (u = column, v = row); outside -> edge."""
    H, W = img.shape[:2]
    u = np.clip(u, 0, W - 1.001)
    v = np.clip(v, 0, H - 1.001)
    u0, v0 = np.floor(u).astype(int), np.floor(v).astype(int)
    fu, fv = (u - u0)[..., None], (v - v0)[..., None]
    a, b = img[v0, u0], img[v0, u0 + 1]
    c, d = img[v0 + 1, u0], img[v0 + 1, u0 + 1]
    return (a * (1 - fu) + b * fu) * (1 - fv) + (c * (1 - fu) + d * fu) * fv


def srgb_to_linear(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


class CameraSensor:
    """One camera for one episode: fixed parameters, cached remap grid, fresh noise every frame."""

    def __init__(self, view, intr, seed, profile="realistic"):
        self.view = view
        self.w, self.h = int(intr["width"]), int(intr["height"])
        self.K = np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1]], float)
        self.rng = np.random.default_rng(seed)
        self.p = sample_params(view, self.rng) if profile == "realistic" else dict(NONE)
        self._build_maps()

    def _build_maps(self):
        p, K = self.p, self.K
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        d = (p["k1"], p["k2"], p["p1"], p["p2"])
        # Output focal = alpha * input focal; pick the largest alpha <= 1 whose distorted border stays inside the
        # rendered (undistorted) image, so the frame has no empty corners.
        bu = np.concatenate([np.arange(self.w), np.arange(self.w), np.zeros(self.h), np.full(self.h, self.w - 1)])
        bv = np.concatenate([np.zeros(self.w), np.full(self.w, self.h - 1), np.arange(self.h), np.arange(self.h)])
        alpha = 1.0
        for _ in range(30):
            xu, yu = undistort_norm((bu - cx) / (alpha * fx), (bv - cy) / (alpha * fy), *d)
            u, v = xu * fx + cx, yu * fy + cy
            if u.min() >= -0.5 and v.min() >= -0.5 and u.max() <= self.w - 0.5 and v.max() <= self.h - 0.5:
                break
            alpha *= 1.03
        self.alpha = alpha
        self.K_out = K.copy()
        self.K_out[0, 0], self.K_out[1, 1] = alpha * fx, alpha * fy
        uu, vv = np.meshgrid(np.arange(self.w, dtype=float), np.arange(self.h, dtype=float))
        xd, yd = (uu - cx) / (alpha * fx), (vv - cy) / (alpha * fy)
        xu, yu = undistort_norm(xd, yd, *d)
        self.map_u, self.map_v = xu * fx + cx, yu * fy + cy
        # lateral chromatic aberration: red sampled slightly further out, blue slightly further in
        c = p["ca"]
        self.map_r = (cx + (self.map_u - cx) * (1 + c), cy + (self.map_v - cy) * (1 + c))
        self.map_b = (cx + (self.map_u - cx) * (1 - c), cy + (self.map_v - cy) * (1 - c))
        r2 = (xd * xd + yd * yd) / max(1e-9, ((self.w / 2) / (alpha * fx)) ** 2 + ((self.h / 2) / (alpha * fy)) ** 2)
        self.vig = (1 - p["vignette"] * r2)[..., None]

    def intrinsics(self):
        """Intrinsics of the frames this sensor outputs (OpenCV pinhole + Brown-Conrady)."""
        K, p = self.K_out, self.p
        return dict(model="opencv", fx=float(K[0, 0]), fy=float(K[1, 1]), cx=float(K[0, 2]), cy=float(K[1, 2]),
                    width=self.w, height=self.h, dist=[p["k1"], p["k2"], p["p1"], p["p2"], 0.0])

    def describe(self):
        return dict(view=self.view, type=camera_type(self.view), **{k: v for k, v in self.p.items()})

    def __call__(self, img):
        """Clean uint8 RGB [H, W, 3] -> what the real camera would have recorded (uint8 RGB)."""
        p = self.p
        lin = srgb_to_linear(np.asarray(img, dtype=np.float64) / 255.0)
        if p["k1"] or p["k2"] or p["p1"] or p["p2"] or p["ca"]:
            out = np.empty_like(lin)
            out[..., 1] = _bilinear(lin, self.map_u, self.map_v)[..., 1]
            out[..., 0] = _bilinear(lin, *self.map_r)[..., 0]
            out[..., 2] = _bilinear(lin, *self.map_b)[..., 2]
            lin = out
        lin = lin * self.vig * p["gain"] * np.asarray(p["wb"])
        if p["full_well"]:  # shot noise (Poisson in electrons) + read noise, in linear light
            e = self.rng.poisson(np.clip(lin, 0, 1) * p["full_well"]) / p["full_well"]
            lin = e + self.rng.normal(0, p["read_noise"], lin.shape)
        out = (linear_to_srgb(lin) * 255 + 0.5).astype(np.uint8)
        if p["jpeg"]:
            from PIL import Image
            buf = io.BytesIO()
            Image.fromarray(out).save(buf, format="JPEG", quality=p["jpeg"])
            out = np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))
        return out


def project(K, dist, pts_cam):
    """Project camera-frame points [N, 3] (OpenCV axes: x right, y down, z forward) with distortion -> pixels [N, 2]."""
    x, y = pts_cam[:, 0] / pts_cam[:, 2], pts_cam[:, 1] / pts_cam[:, 2]
    xd, yd = distort_norm(x, y, dist[0], dist[1], dist[2], dist[3])
    return np.stack([K[0][0] * xd + K[0][2], K[1][1] * yd + K[1][2]], 1)
