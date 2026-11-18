"""How far are the simulated images from real ones? KID and FID between rendered frames and a folder of real frames.

    .venv/bin/python scripts/sim2real_metric.py --real data/real/epic_kitchens_frames \\
        --sim procedural=data/ego_raw photoreal=data/ego_photoreal --views head [--sensor realistic] [--out gap.json]

--real   a folder of real images (searched recursively, .jpg/.jpeg/.png), e.g. a few thousand frames sampled from
         EPIC-KITCHENS or Ego4D (download them yourself under the dataset's licence). Head-camera footage is the
         closest match to the `head` view.
--sim    NAME=PATH pairs: a raw egocentric folder (episode_*/<view>/*.png, from blender/gen_ego_episodes.py) or any
         image folder. Frames are sampled evenly across episodes, up to --max per set.
--sensor realistic  passes raw frames through the simulated camera (worldsim/sensor.py) first, as the exporters do.

Features are Inception-v3 pool features (the standard for FID/KID; torchvision weights, downloaded once).
KID (kernel inception distance, x1000, mean +- std over random subsets) is unbiased and stable for a few hundred
images, so it is the headline number; FID is printed alongside. Lower = closer to real.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from worldsim import ego, sensor  # noqa: E402

EXT = (".jpg", ".jpeg", ".png")


def image_folder(path, n, rng):
    files = sorted(f for f in glob.glob(os.path.join(path, "**", "*"), recursive=True) if f.lower().endswith(EXT))
    if len(files) > n:
        files = [files[i] for i in sorted(rng.choice(len(files), n, replace=False))]
    return [(f, None) for f in files]


def raw_frames(path, views, n, rng, sensor_mode):
    """[(png, sensor_or_None)] sampled evenly over the episodes and views of a raw egocentric folder."""
    eps = ego.load_raw(path)
    out = []
    for d, m in eps:
        for v in views:
            if v not in m["views"]:
                continue
            cam = (sensor.CameraSensor(v, m["intrinsics"][v], sensor.seed_for("metric", os.path.basename(d), v))
                   if sensor_mode == "realistic" else None)
            out += [(os.path.join(d, v, f"{i:06d}.png"), cam) for i in range(m["n_frames"])]
    if len(out) > n:
        out = [out[i] for i in sorted(rng.choice(len(out), n, replace=False))]
    return out


def is_raw(path):
    return bool(glob.glob(os.path.join(path, "episode_*", "episode.json")))


class Features:
    def __init__(self, device):
        from torchvision.models import Inception_V3_Weights, inception_v3
        net = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
        net.fc = torch.nn.Identity()
        self.net = net.eval().to(device)
        self.device = device
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def __call__(self, items, batch=32):
        feats = []
        for k in range(0, len(items), batch):
            ims = []
            for path, cam in items[k:k + batch]:
                im = np.asarray(Image.open(path).convert("RGB"))
                if cam is not None:
                    im = cam(im)
                ims.append(np.asarray(Image.fromarray(im).resize((299, 299), Image.BICUBIC)))
            x = torch.from_numpy(np.stack(ims)).to(self.device).permute(0, 3, 1, 2).float() / 255.0
            feats.append(self.net((x - self.mean) / self.std).double().cpu().numpy())
        return np.concatenate(feats)


def kid(a, b, subsets=100, size=None, seed=0):
    """Unbiased MMD^2 with the cubic polynomial kernel (Binkowski et al. 2018), averaged over random subsets."""
    rng = np.random.default_rng(seed)
    m = size or min(len(a), len(b), 1000)
    d = a.shape[1]
    vals = []
    for _ in range(subsets):
        x = a[rng.choice(len(a), m, replace=False)]
        y = b[rng.choice(len(b), m, replace=False)]
        kxx = (x @ x.T / d + 1) ** 3
        kyy = (y @ y.T / d + 1) ** 3
        kxy = (x @ y.T / d + 1) ** 3
        vals.append((kxx.sum() - np.trace(kxx)) / (m * (m - 1)) + (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
                    - 2 * kxy.mean())
    return float(np.mean(vals)), float(np.std(vals))


def fid(a, b):
    from scipy import linalg
    mu1, mu2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    covmean, _ = linalg.sqrtm(s1 @ s2, disp=False)
    if not np.isfinite(covmean).all():
        eps = np.eye(len(s1)) * 1e-6
        covmean = linalg.sqrtm((s1 + eps) @ (s2 + eps))
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * covmean.real))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True)
    ap.add_argument("--sim", nargs="+", required=True, metavar="NAME=PATH")
    ap.add_argument("--views", nargs="*", default=["head"], help="views to take from raw egocentric folders")
    ap.add_argument("--sensor", choices=["none", "realistic"], default="none")
    ap.add_argument("--max", type=int, default=2000, help="images per set")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", help="write the results as JSON")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    F = Features(a.device)
    real = image_folder(a.real, a.max, rng)
    if len(real) < 50:
        sys.exit(f"only {len(real)} real images in {a.real}; need at least 50 (a few hundred or more is better)")
    fr = F(real)
    rows = []
    for spec in a.sim:
        name, path = spec.split("=", 1) if "=" in spec else (os.path.basename(spec.rstrip("/")), spec)
        items = raw_frames(path, a.views, a.max, rng, a.sensor) if is_raw(path) else image_folder(path, a.max, rng)
        if len(items) < 50:
            print(f"[gap] {name}: only {len(items)} images, skipped")
            continue
        fs = F(items)
        k, ks = kid(fs, fr, seed=a.seed)
        rows.append(dict(name=name, path=path, n=len(items), kid_x1000=1000 * k, kid_std_x1000=1000 * ks, fid=fid(fs, fr)))
        print(f"[gap] {name:20s} n={len(items):5d}  KID x1000 = {1000 * k:7.2f} +- {1000 * ks:5.2f}   FID = {rows[-1]['fid']:7.2f}",
              flush=True)
    res = dict(real=a.real, n_real=len(real), views=a.views, sensor=a.sensor, features="inception_v3 pool", sets=rows)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
    print("\n| set | images | KID ×1000 (↓) | FID (↓) |\n|---|---|---|---|")
    for r in rows:
        print(f"| {r['name']} | {r['n']} | {r['kid_x1000']:.1f} ± {r['kid_std_x1000']:.1f} | {r['fid']:.1f} |")


if __name__ == "__main__":
    main()
