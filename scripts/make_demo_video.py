"""Cut the rendered rollouts into a ~90 s captioned project demo video.

    python scripts/make_demo_video.py [--out docs/media/demo.mp4]

Inputs (made by scripts/make_demo_media.sh): renders/demo/{cutaway,house,kitchen,street,garden,aerial}.mp4,
renders/demo/top_f420.png, a room-mode render, docs/media/training.png, and one or more raw egocentric episodes
(--ego, from blender/gen_ego_episodes.py). Title cards, captions, the self-improvement diagram and the multi-view
egocentric mosaic are drawn with PIL; ffmpeg (imageio-ffmpeg) does the editing.
"""
import argparse
import json
import os
import subprocess
import tempfile

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FF = imageio_ffmpeg.get_ffmpeg_exe()
W, H, FPS = 1280, 720, 30
NAVY, PAPER, ORANGE, MUTED, BLUE = (15, 26, 43), (246, 244, 238), (240, 138, 93), (201, 211, 224), (79, 134, 217)
FONT_DIRS = ["/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/truetype/liberation2",
             "/usr/share/fonts/truetype/liberation"]


def font(name, size):
    for d in FONT_DIRS:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


BOLD = lambda s: font("DejaVuSans-Bold.ttf", s)  # noqa: E731
REG = lambda s: font("DejaVuSans.ttf", s)  # noqa: E731
MONO = lambda s: font("DejaVuSansMono.ttf", s)  # noqa: E731


def wrap(draw, text, f, width):
    lines, cur = [], ""
    for w in text.split():
        t = (cur + " " + w).strip()
        if draw.textlength(t, font=f) <= width:
            cur = t
        else:
            lines.append(cur)
            cur = w
    return lines + [cur]


def run(args):
    subprocess.run([FF, "-v", "error", "-y", *args], check=True)


# ----------------------------------------------------------------------------------------- graphics
def caption_png(path, label, text):
    """Lower-third: a translucent navy panel with an orange label and white text."""
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    fl, ft = MONO(18), BOLD(28)
    lines = wrap(d, text, ft, 1040)
    tw = max(d.textlength(s, font=ft) for s in lines + [label.upper()])
    bw, bh = int(tw) + 64, 40 + 26 + 38 * len(lines)
    x0, y0 = 40, H - 40 - bh
    d.rounded_rectangle([x0, y0, x0 + bw, y0 + bh], 14, fill=NAVY + (215,))
    d.rectangle([x0, y0 + 14, x0 + 5, y0 + bh - 14], fill=ORANGE + (255,))
    d.text((x0 + 32, y0 + 18), label.upper(), font=fl, fill=ORANGE + (255,))
    for i, s in enumerate(lines):
        d.text((x0 + 32, y0 + 44 + 38 * i), s, font=ft, fill=PAPER + (255,))
    im.save(path)


def backdrop(src_video, t):
    """A darkened, blurred frame of a clip as a card background."""
    tmp = tempfile.mktemp(suffix=".png")
    run(["-ss", str(t), "-i", src_video, "-frames:v", "1", "-vf", f"scale={W}:{H}", tmp])
    im = Image.open(tmp).convert("RGB").filter(ImageFilter.GaussianBlur(10))
    os.remove(tmp)
    return Image.blend(im, Image.new("RGB", (W, H), NAVY), 0.72)


def title_card(path, bg, eyebrow, title, sub, foot=None):
    im = bg.copy()
    d = ImageDraw.Draw(im)
    x, y = 96, 200
    d.text((x, y), eyebrow.upper(), font=MONO(22), fill=ORANGE)
    y += 52
    for s in wrap(d, title, BOLD(64), 1080):
        d.text((x, y), s, font=BOLD(64), fill=PAPER)
        y += 78
    y += 18
    for s in wrap(d, sub, REG(30), 1000):
        d.text((x, y), s, font=REG(30), fill=MUTED)
        y += 44
    if foot:
        d.text((x, H - 84), foot, font=MONO(22), fill=(138, 151, 171))
    im.save(path)


def loop_card(path):
    """The recursive self-improvement loop, as on the slides."""
    im = Image.new("RGB", (W, H), NAVY)
    d = ImageDraw.Draw(im)
    d.text((80, 56), "HOW IT LEARNS", font=MONO(22), fill=ORANGE)
    d.text((80, 92), "Recursive self-improvement", font=BOLD(52), fill=PAPER)
    boxes = [("1 · Train", "8 members × PPO, self-play"), ("2 · Evaluate", "fixed-seed benchmark"),
             ("3 · Exploit & explore", "bottom 25% copy top 25%"), ("4 · Elite archive", "opponents and teachers"),
             ("5 · Distill", "worst reborn from top 3"), ("6 · Curriculum", "more people, more cars")]
    bw, bh, gx = 360, 150, 50
    xs = [80, 80 + bw + gx, 80 + 2 * (bw + gx)]
    pos = [(xs[0], 230), (xs[1], 230), (xs[2], 230), (xs[2], 470), (xs[1], 470), (xs[0], 470)]
    for i, ((t, s), (x, y)) in enumerate(zip(boxes, pos)):
        edge = BLUE if i < 3 else ORANGE
        d.rounded_rectangle([x, y, x + bw, y + bh], 16, fill=(27, 42, 66), outline=edge, width=3)
        d.text((x + 24, y + 30), t, font=BOLD(26), fill=PAPER)
        d.text((x + 24, y + 82), s, font=REG(22), fill=MUTED)

    def arrow(x1, y1, x2, y2):
        d.line([x1, y1, x2, y2], fill=PAPER, width=4)
        if x1 == x2:
            s = 1 if y2 > y1 else -1
            d.polygon([(x2, y2), (x2 - 10, y2 - 16 * s), (x2 + 10, y2 - 16 * s)], fill=PAPER)
        else:
            s = 1 if x2 > x1 else -1
            d.polygon([(x2, y2), (x2 - 16 * s, y2 - 10), (x2 - 16 * s, y2 + 10)], fill=PAPER)

    arrow(xs[0] + bw, 305, xs[1], 305)
    arrow(xs[1] + bw, 305, xs[2], 305)
    arrow(xs[2] + bw // 2, 380, xs[2] + bw // 2, 470)
    arrow(xs[2], 545, xs[1] + bw, 545)
    arrow(xs[1], 545, xs[0] + bw, 545)
    arrow(xs[0] + bw // 2, 470, xs[0] + bw // 2, 380)
    d.text((80, 660), "PPO inside, population-based training outside — on 2× RTX A2000", font=REG(24), fill=MUTED)
    im.save(path)


def chart_card(path, png):
    im = Image.new("RGB", (W, H), (252, 252, 251))
    ch = Image.open(png).convert("RGB")
    ch.thumbnail((1080, 540), Image.LANCZOS)
    im.paste(ch, ((W - ch.width) // 2, 12))
    im.save(path)


EGO_NAMES = {"head": "head (glasses)", "chest": "chest", "left_wrist": "left wrist", "right_wrist": "right wrist",
             "scene_front": "scene front", "scene_left": "scene left", "scene_right": "scene right"}


def ego_mosaic(ep_dir, out):
    """One raw egocentric episode as a 1280x720 clip: the head view large, the other six views in a 3x2 grid."""
    m = json.load(open(os.path.join(ep_dir, "episode.json")))
    others = [v for v in m["views"] if v != "head"][:6]
    big, small, gap, x0, y0 = 512, 200, 8, 64, 150
    xs = x0 + big + 16
    frames = tempfile.mkdtemp(prefix="ego_frames_")
    t = m.get("task", {})
    sub = f"{t.get('n_cuts', '?')} cuts at {t.get('chop_hz', 0):.1f} Hz · {t.get('apple_color', '')} apple · " \
          f"{'in the whole house' if m.get('scene') == 'house' else 'standalone kitchen'}"
    for i in range(m["n_frames"]):
        im = Image.new("RGB", (W, H), NAVY)
        d = ImageDraw.Draw(im)
        d.text((x0, 36), "EGOCENTRIC KITCHEN · 7 SYNCHRONISED CAMERAS", font=MONO(20), fill=ORANGE)
        d.text((x0, 66), f"\u201c{m['instruction']}\u201d", font=BOLD(34), fill=PAPER)
        d.text((x0, 112), sub, font=REG(20), fill=MUTED)
        tile = lambda v, r: Image.open(os.path.join(ep_dir, v, f"{i:06d}.png")).convert("RGB").resize((r, r), Image.LANCZOS)  # noqa: E731
        im.paste(tile("head", big), (x0, y0 + 26))
        d.text((x0, y0), EGO_NAMES["head"], font=MONO(18), fill=MUTED)
        for k, v in enumerate(others):
            x, y = xs + (k % 3) * (small + gap), y0 + (k // 3) * (small + 34)
            d.text((x, y), EGO_NAMES.get(v, v), font=MONO(18), fill=MUTED)
            im.paste(tile(v, small), (x, y + 26))
        fr = m["frames"][i]
        yb = y0 + 2 * (small + 34) + 6
        d.text((xs, yb), f"t = {i / m['fps']:4.1f} s   phase: {fr['phase']}", font=MONO(20), fill=PAPER)
        d.text((xs, yb + 32), "Logged per frame: both hands' 6-DoF pose, grip,", font=REG(19), fill=MUTED)
        d.text((xs, yb + 56), "every camera's intrinsics and pose.", font=REG(19), fill=MUTED)
        im.save(os.path.join(frames, f"{i:06d}.png"))
    run(["-framerate", str(m["fps"]), "-i", os.path.join(frames, "%06d.png"), "-c:v", "libx264", "-crf", "16",
         "-pix_fmt", "yuv420p", out])
    return m["n_frames"] / m["fps"]


# ----------------------------------------------------------------------------------------- editing
def segment(out, src, dur, captions=(), start=0.0, still=False, zoom=False):
    """One normalised 1280x720/30fps segment with timed captions and fades from/to black."""
    args = (["-loop", "1", "-t", str(dur), "-i", src] if still else ["-ss", str(start), "-t", str(dur), "-i", src])
    base = f"scale={W}:{H}:flags=lanczos,setsar=1,fps={FPS}"
    if zoom:
        n = int(dur * FPS)
        base = (f"scale={2 * W}:{2 * H},zoompan=z='min(1+0.10*on/{n},1.10)':x='iw/2-(iw/zoom/2)':"
                f"y='ih/2-(ih/zoom/2)':d={n}:s={W}x{H}:fps={FPS},setsar=1")
    chain, last = [f"[0:v]{base}[v0]"], "v0"
    for i, (png, t0, t1) in enumerate(captions):
        args += ["-loop", "1", "-t", str(dur), "-i", png]
        k = i + 1
        chain.append(f"[{k}:v]format=rgba,fade=t=in:st={t0}:d=0.4:alpha=1,fade=t=out:st={t1 - 0.4}:d=0.4:alpha=1[c{k}]")
        chain.append(f"[{last}][c{k}]overlay=0:0:enable='between(t,{t0},{t1})'[v{k}]")
        last = f"v{k}"
    chain.append(f"[{last}]fade=t=in:st=0:d=0.5,fade=t=out:st={dur - 0.5}:d=0.5,format=yuv420p[out]")
    run([*args, "-filter_complex", ";".join(chain), "-map", "[out]", "-t", str(dur), "-r", str(FPS),
         "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-an", out])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", default=os.path.join(ROOT, "renders", "demo"))
    ap.add_argument("--room", default=os.path.join(ROOT, "renders", "showcase", "rollout_corner.mp4"))
    ap.add_argument("--chart", default=os.path.join(ROOT, "docs", "media", "training.png"))
    ap.add_argument("--ego", nargs="*", default=[], help="raw egocentric episode folders to show (episode_NNNNNN)")
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "media", "demo.mp4"))
    a = ap.parse_args()
    v = lambda n: os.path.join(a.demo, n)  # noqa: E731
    tmp = tempfile.mkdtemp(prefix="demo_video_")
    p = lambda n: os.path.join(tmp, n)  # noqa: E731

    cap_specs = {
        "c1": ("Live rollout · trained policy", "Six digital humans share a house, a kitchen, a garden and a street with eight cars"),
        "c2": ("No scripted paths", "Each person picks a goal and plans a route through doors, the garden gate and the crossing"),
        "c4": ("The world · configs/world.json", "A living room, a kitchen built onto it, a fenced back garden, a front yard and a two-lane street"),
        "c11": ("The kitchen · configs/kitchen.json", "Built onto the house; an open door joins it to the living room"),
        "c12": ("New goals", "Fridge, stove, chopping counter and island, so people walk between the two rooms"),
        "c13": ("Inside the kitchen", "The same kitchen is the set for the egocentric chopping data"),
        "c5": ("Traffic model", "Cars follow each other (IDM) and stop for people in their lane"),
        "c6": ("Distracted drivers", "A quarter of drivers only notice people within 7 m, so people learn to look first"),
        "c7": ("Back garden", "Patio, BBQ, beds, bench and shed, reached through the back door or the side gate"),
        "c8": ("Navigation", "Routes come from a GPU navigation grid; road cells cost 4×, so people use the crossing"),
        "c9": ("Results · world run, 6 people and 8 cars", "99.9% goal success at 5.8 goals/min; collisions down from 7% to 0.7% of steps; car hits 0.01%"),
        "c10": ("Room mode · Cycles", "Photo-realistic Microsoft Rocketbox avatars with motion-captured gaits"),
    }
    for k, (label, text) in cap_specs.items():
        caption_png(p(k + ".png"), label, text)

    bg = backdrop(v("cutaway.mp4"), 6)
    title_card(p("title.png"), bg, "Technical demo", "Human-Centric World Simulation",
               "Digital humans that learn to share a house, a garden and a busy street, and first-person data for robots",
               "PPO + recursive self-improvement · PyTorch · Blender 4.2")
    title_card(p("end.png"), bg, "Open source on GitHub", "Human-Centric World Simulation",
               "github.com/chengjianglong/human-centric-world-simulation",
               "PyTorch · Blender 4.2 · 2× RTX A2000")
    loop_card(p("loop.png"))
    chart_card(p("chart.png"), a.chart)

    segs = [
        (p("title.png"), 5, [], dict(still=True, zoom=True)),
        (v("cutaway.mp4"), 12, [("c1", 0.5, 6.0), ("c2", 6.0, 11.6)], dict(start=0.5)),
        (v("top_f420.png"), 7, [("c4", 0.8, 6.6)], dict(still=True, zoom=True)),
        (v("house.mp4"), 12, [("c11", 0.5, 6.0), ("c12", 6.0, 11.6)], dict(start=3)),
        (v("kitchen.mp4"), 7, [("c13", 0.5, 6.6)], dict(start=9)),
        (v("street.mp4"), 9, [("c5", 0.5, 4.6), ("c6", 4.6, 8.6)], dict(start=0.5)),
        (v("garden.mp4"), 7, [("c7", 0.5, 6.6)], dict(start=0.3)),
        (v("aerial.mp4"), 7, [("c8", 0.5, 6.6)], dict(start=0.3)),
        (p("loop.png"), 8, [], dict(still=True)),
        (p("chart.png"), 7, [("c9", 0.6, 6.6)], dict(still=True)),
        (a.room, 9, [("c10", 0.5, 8.6)], dict(start=2)),
    ]
    if a.ego:
        title_card(p("ego_title.png"), backdrop(v("kitchen.mp4"), 12), "Egocentric data for VLA training",
                   "A woman chops an apple; a body-worn rig films it",
                   "Head, chest and both wrists, plus three scene cameras. Exported to LeRobot v2 and vla-qwen.",
                   "blender/gen_ego_episodes.py · --scene house")
        segs.append((p("ego_title.png"), 5, [], dict(still=True, zoom=True)))
        for j, ep in enumerate(a.ego):
            clip = p(f"ego{j}.mp4")
            dur = ego_mosaic(ep, clip)
            segs.append((clip, round(dur, 1), [], {}))
    segs.append((p("end.png"), 6, [], dict(still=True, zoom=True)))
    parts = []
    for i, (src, dur, caps, kw) in enumerate(segs):
        out = p(f"seg{i:02d}.mp4")
        segment(out, src, dur, [(p(c + ".png"), t0, t1) for c, t0, t1 in caps], **kw)
        parts.append(out)
        print(f"[video] segment {i + 1}/{len(segs)} ({dur}s) <- {os.path.basename(src)}")
    lst = p("list.txt")
    with open(lst, "w") as f:
        f.writelines(f"file '{x}'\n" for x in parts)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    run(["-f", "concat", "-safe", "0", "-i", lst, "-c:v", "libx264", "-crf", "21", "-preset", "slow",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", a.out])
    print(f"[video] {sum(s[1] for s in segs)} s -> {a.out} ({os.path.getsize(a.out) / 1e6:.1f} MB); frames in {tmp}")


if __name__ == "__main__":
    main()
