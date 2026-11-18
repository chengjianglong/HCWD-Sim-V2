"""Generate randomised egocentric chopping episodes (multi-view images + hand poses).

    tools/blender/blender -b -P blender/gen_ego_episodes.py -- --out data/ego_raw --episodes 20 --seed 0
        [--start 0] [--engine eevee|cycles] [--res 256] [--fps 10] [--samples 8] [--fisheye] [--views head ...]
        [--scene kitchen|house]

--scene kitchen (default) uses the standalone kitchen, assets/kitchen.blend (blender/build_kitchen.py). --scene house
uses the integrated house, assets/world.blend (blender/build_world.py), where the same kitchen is joined to the living
room through an open door: the whole house is moved so the kitchen's corner is the origin, so poses and actions are in
the same frame for both scenes. Each episode re-opens the blend, randomises the scene, the
cook and the task, animates the chopping (chop.py), keys the camera rig (ego_rig.py) and renders every view
in one multi-view pass. Output per episode:

    <out>/episode_000123/<view>/000000.png ...   one folder per camera
    <out>/episode_000123/episode.json             instruction, fps, per-frame hand/knife/camera poses, phases

scripts/ego_to_lerobot.py and scripts/ego_to_vlaqwen.py turn these into training datasets.
"""
import argparse
import json
import math
import os
import shutil
import sys
import time

import bpy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chop  # noqa: E402
import common as C  # noqa: E402
import ego_rig  # noqa: E402
import pbr  # noqa: E402
import rocketbox as RB  # noqa: E402
from worldsim import room_spec  # noqa: E402

FEMALE = ("Female_Adult_01", "Female_Adult_04", "Female_Adult_06")
PALETTE = dict(
    cabinet=[(0.42, 0.5, 0.44), (0.12, 0.17, 0.27), (0.9, 0.89, 0.86), (0.2, 0.2, 0.21), (0.55, 0.38, 0.22),
             (0.86, 0.8, 0.68), (0.55, 0.62, 0.66)],
    worktop=[(0.86, 0.85, 0.82), (0.12, 0.12, 0.13), (0.55, 0.38, 0.22), (0.7, 0.68, 0.64), (0.93, 0.92, 0.9)],
    backsplash=[(0.93, 0.93, 0.9), (0.35, 0.55, 0.6), (0.82, 0.72, 0.6), (0.18, 0.22, 0.3), (0.95, 0.93, 0.85)],
    floor=[(0.62, 0.6, 0.56), (0.3, 0.29, 0.28), (0.72, 0.66, 0.56), (0.5, 0.5, 0.52)],
    walls=[(0.9, 0.88, 0.84), (0.82, 0.86, 0.84), (0.93, 0.9, 0.82), (0.78, 0.8, 0.85)],
)


def set_color(mat_name, rgb, rng, var=0.06):
    mat = bpy.data.materials.get(mat_name)
    if mat is None:
        return
    c = np.clip(np.array(rgb) * rng.uniform(1 - var, 1 + var, 3), 0, 1)
    if mat.get("pbr_mean") is not None:  # photo-textured: keep the texture, give it this average colour
        pbr.set_albedo(mat, [float(x) for x in c])
        return
    nt = mat.node_tree
    for n in nt.nodes:
        if n.type == "BSDF_PRINCIPLED" and not n.inputs["Base Color"].is_linked:
            n.inputs["Base Color"].default_value = (*c, 1)
        if n.type == "TEX_BRICK":
            n.inputs["Color1"].default_value = (*c, 1)
            n.inputs["Color2"].default_value = (*np.clip(c * 1.06, 0, 1), 1)
        if n.type == "VALTORGB":  # noise-coloured materials (worktop stone)
            n.color_ramp.elements[0].color = (*np.clip(c * 0.88, 0, 1), 1)
            n.color_ramp.elements[-1].color = (*np.clip(c * 1.08, 0, 1), 1)
    mat.diffuse_color = (*c, 1)


# photo-texture slots re-drawn per episode (material name in the kitchen -> texture slot)
TEXTURE_SLOTS = dict(cabinet="cabinet", worktop="worktop_stone", backsplash="backsplash_tile", floor="kitchen_floor_tile",
                     walls="wall_plaster")


def randomise_domain(drng, spec):
    """Photo-real domain randomisation (own RNG, so the task draws stay identical to the procedural pipeline):
    texture variant per kitchen surface, HDRI sky + rotation + strength, worktop clutter. Returns a record."""
    dom = {}
    if pbr.ENABLED is False:
        return dom
    tex = {}
    for name, slot in TEXTURE_SLOTS.items():
        mat = bpy.data.materials.get(name)
        ids = pbr.slot_ids(slot)
        if mat is not None and mat.get("pbr_id") and ids:
            aid = ids[int(drng.integers(len(ids)))]
            if pbr.set_variant(mat, aid):
                tex[name] = aid
    if tex:
        dom["textures"] = tex
    w = bpy.context.scene.world
    if w is not None and w.get("hdri"):
        pool = pbr.hdri_ids("sky") + pbr.hdri_ids("overcast")
        aid = pool[int(drng.integers(len(pool)))]
        rot, strength = float(drng.uniform(0, 360)), float(drng.uniform(0.6, 1.5))
        pbr.hdri_world(w, aid, strength, rot)
        dom["hdri"] = dict(id=aid, rotation_deg=round(rot, 1), strength=round(strength, 3))
    import build_kitchen as BK  # clutter on the free worktops, re-sampled every episode
    old = [o for o in bpy.data.objects if o.get("clutter")]
    for o in old:
        for ch in o.children_recursive:
            bpy.data.objects.remove(ch)
        bpy.data.objects.remove(o)
    col = C.get_collection("props")
    placed = BK.build_clutter(spec, col, seed=int(drng.integers(1 << 31)), n=int(drng.integers(3, 9)))
    dom["clutter"] = [o.children[0].get("model") for o in placed if o.children]
    return dom


def randomise_scene(rng):
    for key, choices in PALETTE.items():
        set_color(key, choices[int(rng.integers(len(choices)))], rng)
    sun = bpy.data.objects.get("sun")
    if sun:
        el, az = math.radians(rng.uniform(15, 55)), math.radians(rng.uniform(100, 260))
        sun.rotation_euler = (math.pi / 2 - el, 0, az + math.pi / 2)
        sun.data.energy = rng.uniform(1.5, 4.5)
        k = rng.uniform(0, 1)
        sun.data.color = (1.0, 0.92 + 0.08 * k, 0.82 + 0.18 * k)
    light = bpy.data.objects.get("kitchen_light") or bpy.data.objects.get("ceiling_light")
    if light:
        light.data.energy = rng.uniform(80, 240)
        k = rng.uniform(0, 1)
        light.data.color = (1.0, 0.85 + 0.15 * k, 0.7 + 0.3 * k)
    w = bpy.context.scene.world
    strength = rng.uniform(0.2, 0.5)  # drawn in every case, so the task RNG sequence never changes
    if w and w.use_nodes and not w.get("hdri"):
        w.node_tree.nodes["Background"].inputs["Strength"].default_value = strength
    if w is not None and w.get("hdri") and sun:  # the HDRI carries the sun
        sun.data.energy = 0.0


def setup_render(sc, a, views):
    sc.render.resolution_x = sc.render.resolution_y = a.res
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGB"
    sc.view_settings.view_transform = "AgX"
    sc.view_settings.look = "AgX - Medium High Contrast"
    sc.view_settings.exposure = -0.5
    if a.engine == "cycles":
        import render_rollout as RR
        RR.setup_gpu_cycles(sc, a.samples)
        sc.cycles.use_adaptive_sampling = True
        sc.cycles.adaptive_threshold = 0.02
        sc.cycles.max_bounces, sc.cycles.diffuse_bounces, sc.cycles.glossy_bounces = 6, 3, 3
        sc.cycles.sample_clamp_indirect = 8.0  # no fireflies from the HDRI sun through the windows
        sc.render.use_persistent_data = True  # keep the BVH between frames: the static scene is most of it
        sc.render.use_motion_blur = not a.no_motion_blur  # a real camera integrates over its exposure
        sc.render.motion_blur_shutter = 0.5
    else:
        sc.render.engine = "BLENDER_EEVEE_NEXT"
        sc.eevee.taa_render_samples = a.samples
    # one multi-view render pass: view "<name>" uses camera "ego_<name>" (scene camera = ego_head)
    sc.render.use_multiview = True
    sc.render.views_format = "MULTIVIEW"
    for v in list(sc.render.views):
        v.use = False
    for name in views:
        v = sc.render.views.get(name) or sc.render.views.new(name)
        v.camera_suffix = f"_{name}"
        v.use = True
    sc.render.image_settings.views_format = "INDIVIDUAL"
    sc.camera = bpy.data.objects[f"ego_{views[0]}"]


def hand_pose(arm, side):
    pb = arm.pose.bones[f"Bip01 {side} Hand"]
    m = arm.matrix_world @ pb.matrix
    r = m.to_3x3().normalized()
    q = r.to_quaternion()
    return dict(pos=list(arm.matrix_world @ pb.head), quat_wxyz=list(q))


def house_origin(spec):
    """Where the kitchen's south-west interior corner sits in the house (assets/world_layout.json)."""
    with open(os.path.join(C.ASSETS, "world_layout.json")) as f:
        lay = json.load(f)
    an = next((an for an in lay.get("annexes", []) if an["name"] == spec["name"]), None)
    if an is None:
        sys.exit(f"assets/world_layout.json has no {spec['name']!r} annex: rebuild with blender/build_world.py")
    return an["origin"]


def shift_scene(origin):
    """Move the whole scene by -origin so the kitchen's own frame is the world frame."""
    for o in bpy.data.objects:
        if o.parent is None:
            o.location.x -= origin[0]
            o.location.y -= origin[1]


def run_episode(idx, a, spec, out_root):
    rng = np.random.default_rng([a.seed, idx])
    drng = np.random.default_rng([a.seed, idx, 1729])  # domain randomisation of the photo-real scene
    bpy.ops.wm.open_mainfile(filepath=a.blend)
    pbr.ENABLED = bool(bpy.context.scene.world and bpy.context.scene.world.get("hdri")) or any(
        m.get("pbr_id") for m in bpy.data.materials)
    if a.scene == "house":
        shift_scene(a.origin)
    RB._CLIPS.clear()
    RB._REST.clear()
    sc = bpy.context.scene
    randomise_scene(rng)
    domain = randomise_domain(drng, spec) if pbr.ENABLED else {}
    T = spec["task"]
    colors = list(T["apple"]["colors"])
    color = colors[int(rng.integers(len(colors)))]
    others = [c for c in colors if c != color]
    distract = [others[int(rng.integers(len(others)))] for _ in range(int(rng.integers(0, 4)))]
    jb = T["board"]["jitter"]
    ep = chop.Episode(
        spec, rng,
        avatar=FEMALE[int(rng.integers(len(FEMALE)))] if not a.avatar else a.avatar,
        apple_color=color,
        n_cuts=int(rng.integers(T["cuts"][0], T["cuts"][1] + 1)),
        chop_hz=float(rng.uniform(*T["chop_hz"])),
        cook_x=float(rng.uniform(*T["cook_x"])),
        board_jitter=(float(rng.uniform(-jb[0], jb[0])), float(rng.uniform(-jb[1], jb[1])), float(rng.uniform(-jb[2], jb[2]))),
        clip_start=int(rng.integers(chop.BODY_WINDOW[0], 220)),
        apple_radius=float(rng.uniform(*T["apple"]["radius"])),
        distractors=distract,
    )
    ep.build()
    info = ep.animate()
    rig = ego_rig.Rig(ep.arm, ep.work, C.get_collection("ego_rig"), res=a.res, rng=rng, jitter=not a.no_jitter,
                      fisheye=a.fisheye)
    step = chop.FPS // a.fps
    sampled = list(range(1, info["n_frames"] + 1, step))
    frames = []
    for f in range(1, info["n_frames"] + 1):
        sc.frame_set(f)
        mats = rig.key(f)
        if f not in sampled:
            continue
        tip = ep.knife.matrix_world @ __import__("mathutils").Vector(ep.knife["tip"])
        g_r, g_l = ep.gripper[f - 1]
        frames.append(dict(
            frame_index=len(frames), scene_frame=f, timestamp=(f - 1) / chop.FPS, phase=ep.timeline[f - 1],
            right_hand=hand_pose(ep.arm, "R"), left_hand=hand_pose(ep.arm, "L"),
            grip=dict(right=round(g_r, 4), left=round(g_l, 4)), knife_tip=list(tip),
            ik_err=dict(right=round(chop.ik_error(ep.arm, "R", ep.targets["R"][0]) * g_r, 4),
                        left=round(chop.ik_error(ep.arm, "L", ep.targets["L"][0]) * g_l, 4)),
            cameras={n: ego_rig.cam_to_world_cv(m) for n, m in mats.items()},
        ))
    for cam in rig.cams.values():
        for fc in cam.animation_data.action.fcurves:
            for k in fc.keyframe_points:
                k.interpolation = "LINEAR"

    views = a.views or ego_rig.VIEWS
    setup_render(sc, a, views)
    ep_dir = os.path.join(out_root, f"episode_{idx:06d}")
    tmp = os.path.join(ep_dir, "_render")
    os.makedirs(tmp, exist_ok=True)
    sc.frame_start, sc.frame_end, sc.frame_step = 1, info["n_frames"], step
    sc.render.filepath = os.path.join(tmp, "f_####")
    t0 = time.time()
    if not a.no_render:
        bpy.ops.render.render(animation=True)
    t_render = time.time() - t0
    for v in views:
        os.makedirs(os.path.join(ep_dir, v), exist_ok=True)
        for i, f in enumerate(sampled):
            src = os.path.join(tmp, f"f_{f:04d}_{v}.png")
            if os.path.exists(src):
                os.replace(src, os.path.join(ep_dir, v, f"{i:06d}.png"))
    shutil.rmtree(tmp, ignore_errors=True)
    meta = dict(
        episode_index=idx, instruction=ep.instruction(), fps=a.fps, n_frames=len(frames), views=views,
        intrinsics={v: rig.intrinsics(v) for v in views}, camera_convention="opencv cam-to-world (x right, y down, z fwd)",
        task=dict(apple_color=color, n_cuts=ep.n_cuts, pieces=ep.n_cuts + 1, chop_hz=ep.hz, apple_radius=ep.radius,
                  distractors=distract, avatar=ep.avatar, cook_x=ep.cook_x, clip_start=ep.clip_start),
        render=dict(engine=a.engine, res=a.res, samples=a.samples, seconds=round(t_render, 1)),
        scene=a.scene, scene_origin=list(a.origin), realism="photo" if pbr.ENABLED else "procedural", domain=domain,
        frames=frames,
    )
    with open(os.path.join(ep_dir, "episode.json"), "w") as f:
        json.dump(meta, f)
    if a.keep_blend:
        bpy.ops.wm.save_as_mainfile(filepath=os.path.join(ep_dir, "episode.blend"))
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default=os.path.join(C.ROOT, "configs", "kitchen.json"))
    ap.add_argument("--realism", choices=["auto", "photo", "procedural"], default="auto",
                    help="photo-asset scene build (assets/photoreal) when it exists, else the procedural one")
    ap.add_argument("--scene", choices=["kitchen", "house"], default="kitchen",
                    help="standalone kitchen, or the kitchen joined to the living room in the whole house")
    ap.add_argument("--blend", help="default: assets/kitchen.blend or assets/world.blend for --scene house")
    ap.add_argument("--out", default=os.path.join(C.ROOT, "data", "ego_raw"))
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--start", type=int, default=0, help="index of the first episode (for splitting across GPUs)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--engine", choices=["eevee", "cycles"], default="eevee")
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--fps", type=int, default=10, choices=[5, 10, 15, 30])
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--fisheye", action="store_true", help="equisolid fisheye head camera (Cycles only)")
    ap.add_argument("--views", nargs="*", choices=ego_rig.VIEWS)
    ap.add_argument("--avatar", help="force one Rocketbox avatar")
    ap.add_argument("--no-jitter", action="store_true", help="exact rig mounts (no per-episode offsets)")
    ap.add_argument("--no-render", action="store_true", help="poses only (fast check)")
    ap.add_argument("--keep-blend", action="store_true", help="save each episode's animated .blend")
    ap.add_argument("--no-motion-blur", action="store_true", help="Cycles: sharp frames (default: 0.5 shutter blur)")
    a = ap.parse_args(C.script_args())
    if a.fisheye and a.engine != "cycles":
        ap.error("--fisheye needs --engine cycles")
    a.blend = a.blend or C.scene_blend("world" if a.scene == "house" else "kitchen", a.realism)
    if not os.path.exists(a.blend):
        builder = "build_world.py" if a.scene == "house" else "build_kitchen.py"
        sys.exit(f"{a.blend} missing: run tools/blender/blender -b -P blender/{builder}")
    spec = room_spec.load_spec(a.spec)
    a.origin = house_origin(spec) if a.scene == "house" else (0.0, 0.0)
    os.makedirs(a.out, exist_ok=True)
    for i in range(a.start, a.start + a.episodes):
        t0 = time.time()
        meta = run_episode(i, a, spec, a.out)
        err = max(max(fr["ik_err"].values()) for fr in meta["frames"])
        print(f"[ego] episode {i}: {meta['n_frames']} frames x {len(meta['views'])} views, max IK error {err * 100:.1f} cm, "
              f"'{meta['instruction']}' ({time.time() - t0:.0f}s, render {meta['render']['seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
