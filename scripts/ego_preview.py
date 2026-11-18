"""Preview raw egocentric episodes: a labelled contact sheet of every view, and an animated GIF.

    .venv/bin/python scripts/ego_preview.py --raw data/ego_raw [--episodes 0 3 7] [--frames 0.15 0.45 0.7]
        [--sheet docs/media/ego_views.jpg] [--gif docs/media/ego_views.gif --gif-episode 0 --gif-views head ...]

Sheet: one row per (episode, frame fraction), one column per camera. GIF: the chosen views side by side over
a whole episode, with the instruction and phase as a caption.
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from worldsim import ego  # noqa: E402

NAMES = {"head": "head (glasses)", "chest": "chest", "left_wrist": "left wrist", "right_wrist": "right wrist",
         "scene_front": "scene front", "scene_left": "scene left", "scene_right": "scene right"}


def font(size):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def tile(d, view, i, r):
    return Image.open(os.path.join(d, view, f"{i:06d}.png")).convert("RGB").resize((r, r), Image.LANCZOS)


def sheet(eps, fracs, out, r=200):
    views = eps[0][1]["views"]
    head_h, cap_w = 26, 230
    rows = [(d, m, min(m["n_frames"] - 1, int(f * m["n_frames"]))) for d, m in eps for f in fracs]
    im = Image.new("RGB", (cap_w + r * len(views), head_h + r * len(rows)), (250, 250, 248))
    dr = ImageDraw.Draw(im)
    f_head, f_cap = font(14), font(12)
    for j, v in enumerate(views):
        dr.text((cap_w + j * r + 6, 6), NAMES.get(v, v), fill=(20, 20, 20), font=f_head)
    for k, (d, m, i) in enumerate(rows):
        y = head_h + k * r
        t = m["task"]
        lines = [f"\"{m['instruction']}\"", f"{t['avatar']}, {t['apple_color']} apple", f"{t['n_cuts']} cuts at {t['chop_hz']:.1f} Hz",
                 f"t = {i / m['fps']:.1f} s  ({m['frames'][i]['phase']})"]
        wrapped = []
        for ln in lines:
            while len(ln) > 30:
                cut = ln.rfind(" ", 0, 30)
                cut = cut if cut > 0 else 30
                wrapped.append(ln[:cut])
                ln = ln[cut:].lstrip()
            wrapped.append(ln)
        for n, ln in enumerate(wrapped):
            dr.text((8, y + 10 + 17 * n), ln, fill=(40, 40, 40), font=f_cap)
        for j, v in enumerate(views):
            im.paste(tile(d, v, i, r), (cap_w + j * r, y))
    im.save(out, quality=88)
    print(f"[preview] sheet {im.size[0]}x{im.size[1]} -> {out}")


def gif(d, m, views, out, r=240, step=1):
    f_cap = font(13)
    frames = []
    for i in range(0, m["n_frames"], step):
        im = Image.new("RGB", (r * len(views), r + 24), (250, 250, 248))
        for j, v in enumerate(views):
            im.paste(tile(d, v, i, r), (j * r, 24))
        ImageDraw.Draw(im).text((6, 5), f"\"{m['instruction']}\"   {m['frames'][i]['phase']}   "
                                f"{' | '.join(NAMES.get(v, v) for v in views)}", fill=(30, 30, 30), font=f_cap)
        frames.append(im.convert("P", palette=Image.ADAPTIVE, colors=200))
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000 * step / m["fps"]), loop=0, optimize=True)
    print(f"[preview] gif {len(frames)} frames -> {out} ({os.path.getsize(out) / 1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/ego_raw")
    ap.add_argument("--episodes", type=int, nargs="*", default=[0])
    ap.add_argument("--frames", type=float, nargs="*", default=[0.2, 0.5, 0.75], help="fractions of each episode")
    ap.add_argument("--sheet", default="ego_views.jpg")
    ap.add_argument("--tile", type=int, default=200)
    ap.add_argument("--gif")
    ap.add_argument("--gif-episode", type=int, default=0)
    ap.add_argument("--gif-views", nargs="*", default=["head", "left_wrist", "right_wrist", "scene_front"])
    ap.add_argument("--gif-step", type=int, default=1)
    ap.add_argument("--gif-tile", type=int, default=240)
    a = ap.parse_args()
    raw = {m["episode_index"]: (d, m) for d, m in ego.load_raw(a.raw)}
    eps = [raw[e] for e in a.episodes if e in raw]
    if not eps:
        sys.exit(f"none of episodes {a.episodes} in {a.raw}")
    sheet(eps, a.frames, a.sheet, a.tile)
    if a.gif:
        gif(*raw[a.gif_episode], a.gif_views, a.gif, r=a.gif_tile, step=a.gif_step)


if __name__ == "__main__":
    main()
