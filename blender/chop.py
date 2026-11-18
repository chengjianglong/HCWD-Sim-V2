"""The chopping task: board, knife and a sliceable apple, and a Rocketbox cook who chops it.

There is no cutting mocap in Rocketbox, so the body comes from `f_work_table` (a woman working at a
worktop) and the arms are procedural. Each forearm gets an IK constraint to a target empty, each hand a
Copy Rotation from the same empty, and the constraint influences blend from the mocap pose to the IK pose,
so reaching looks like the mocap arm moving to the target. The right-hand target is the knife's grip
frame, so once she holds the knife the hand simply follows it. The left hand holds the apple in a claw
grip and steps back after every cut. Fingers are a separate NLA layer (open / grip / claw blends).

All task coordinates live in the `work` frame: origin on the floor under the cook, x = her right,
y = forward (she faces +y, i.e. the world frame shifted to her position), z up.

Phases (seconds at 1x): idle, reach, grasp, lift (knife flat -> upright; left hand takes the apple),
chop x n, return, release, retract, idle. `Episode.timeline` keeps the phase of every frame.
"""
import math
import os

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

import common as C
import rocketbox as RB

FPS = 30
BODY_CLIP = "f_work_table"
BODY_WINDOW = (60, 520)  # clip frames where she is bent over the worktop
FINGERS = [f"Finger{i}{j}" for i in range(5) for j in ("", "1", "2")]

# Hand bones: local x = wrist -> knuckles, z = thumb side (R) / -thumb side (L), -y = back of the hand.
# Fingers curl about local +z. Grip frames below give the hand bone pose in the knife / apple frames.


def hand_basis(finger_dir, thumb_dir, side):
    """Rotation whose columns are the hand bone's local axes, from finger and thumb directions."""
    x = Vector(finger_dir).normalized()
    t = Vector(thumb_dir)
    t = (t - t.dot(x) * x).normalized()
    z = t if side == "R" else -t
    y = z.cross(x)
    return Matrix((x, y, z)).transposed()


# knife frame: origin = handle centre, +y = towards the tip, +z = spine (edge at -z), +x = her right when upright
KNIFE_GRIP = Matrix.Translation((0.012, -0.075, 0.03)) @ hand_basis((-0.25, 0.55, -0.8), (-0.35, 0.8, 0.45), "R").to_4x4()
# apple claw frame: origin = top of the cut face the fingertips rest on, axes = work frame
APPLE_CLAW = Matrix.Translation((-0.085, -0.035, 0.05)) @ hand_basis((0.7, 0.3, -0.65), (0.15, -0.95, -0.1), "L").to_4x4()

POSES = {  # curl angles (deg) for [proximal, middle, distal] of thumb, index, middle, ring, little
    "relaxed": [[5, 5, 5], [12, 15, 10], [14, 16, 10], [16, 18, 12], [18, 20, 12]],
    "knife": [[20, 25, 15], [55, 60, 35], [80, 80, 45], [85, 85, 45], [88, 85, 45]],
    "claw": [[15, 20, 10], [45, 75, 55], [45, 80, 55], [45, 80, 55], [45, 75, 50]],
}


def smooth(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


# ------------------------------------------------------------------ props
def wood_board(name, size, col, parent):
    mat = C.wood_material("board_wood", (0.62, 0.42, 0.24), 0.5, plank_width=0.06)
    b = C.box(name, (0, 0, size[2] / 2), size, mat, parent=parent, bevel=0.006, collection=col)
    return b


def knife_object(col, blade=0.2, handle=0.12):
    """Chef's knife. Origin = handle centre; +y to the tip, +z spine, blade in the local yz plane."""
    root = C.empty("knife", collection=col)
    steel = C.material("blade_steel", (0.82, 0.83, 0.85), 0.18, metallic=1.0)
    grip = C.material("knife_handle", (0.05, 0.045, 0.04), 0.45)
    # blade outline (y, z): straight spine, edge rising to the tip
    y0, n = handle / 2 + 0.015, 14
    spine = [(y0 + blade * k / n, 0.008 - 0.012 * (k / n) ** 3) for k in range(n + 1)]
    edge = [(y0 + blade * k / n, -0.038 + 0.04 * (k / n) ** 2.2) for k in range(n + 1)]
    outline = spine + edge[::-1]
    th = 0.0009
    me = bpy.data.meshes.new("knife_blade")
    verts = [(x, y, z) for x in (-th, th) for y, z in outline]
    m = len(outline)
    faces = [tuple(range(m)), tuple(range(2 * m - 1, m - 1, -1))]
    faces += [(i, (i + 1) % m, m + (i + 1) % m, m + i) for i in range(m)]
    me.from_pydata(verts, [], faces)
    me.update()
    bl = C.link(bpy.data.objects.new("knife_blade", me), col)
    bl.data.materials.append(steel)
    bl.parent = root
    C.box("knife_bolster", (0, y0 - 0.008, -0.004), (0.012, 0.016, 0.03), steel, root, collection=col, bevel=0.003)
    C.box("knife_grip", (0, 0, -0.002), (0.016, handle, 0.024), grip, root, collection=col, bevel=0.006)
    for i, yy in enumerate((-0.035, 0.0, 0.035)):
        r = C.cylinder(f"knife_rivet{i}", (0, yy, -0.002), 0.0028, 0.0172, steel, root, verts=10, collection=col)
        r.rotation_euler = (0, math.pi / 2, 0)
    root["tip"] = (0.0, y0 + blade, 0.008 - 0.012)
    root["edge_mid"] = (0.0, y0 + 0.55 * blade, -0.038 + 0.04 * 0.55 ** 2.2)
    return root


def apple_materials(color, rng):
    skin = bpy.data.materials.new("apple_skin")
    skin.use_nodes = True
    nt = skin.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = 0.32
    bsdf.inputs["Coat Weight"].default_value = 0.3
    # streaks along z (apples are stripy) + a lighter blush
    tex = nt.nodes.new("ShaderNodeTexNoise")
    tex.inputs["Scale"].default_value = 9.0
    tex.inputs["Detail"].default_value = 4.0
    mapping = nt.nodes.new("ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (6.0, 6.0, 0.6)
    coord = nt.nodes.new("ShaderNodeTexCoord")
    nt.links.new(coord.outputs["Object"], mapping.inputs["Vector"])
    nt.links.new(mapping.outputs["Vector"], tex.inputs["Vector"])
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    c = np.array(color)
    alt = np.clip(c * rng.uniform(0.55, 0.8) + np.array([0.12, 0.08, 0.0]) * rng.uniform(0, 1), 0, 1)
    ramp.color_ramp.elements[0].color = (*alt, 1)
    ramp.color_ramp.elements[1].color = (*c, 1)
    nt.links.new(tex.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    skin.diffuse_color = (*c, 1)
    flesh = C.material("apple_flesh", (0.95, 0.9, 0.72), 0.55)
    return skin, flesh


def apple_mesh(radius, rng):
    """Apple-ish UV sphere (slightly squashed, dimples top and bottom), centred, base at z = 0."""
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=40, v_segments=28, radius=radius)
    sq = rng.uniform(0.86, 0.95)
    for v in bm.verts:
        x, y, z = v.co
        rr = math.hypot(x, y) / radius
        z *= sq
        if z > 0:
            z -= 0.22 * radius * math.exp(-(rr / 0.28) ** 2)  # stem well
        else:
            z += 0.12 * radius * math.exp(-(rr / 0.25) ** 2)
        w = 1.0 + 0.05 * (z / radius)  # wider shoulders
        v.co = (x * w, y * w, z)
    zmin = min(v.co.z for v in bm.verts)
    for v in bm.verts:
        v.co.z -= zmin
    return bm


def sliced_apple(col, parent, radius, cut_xs, color, rng):
    """Apple split into len(cut_xs)+1 pieces by planes x = cut_xs (descending). Piece k is cut off by cut k;
    the last piece is the part the left hand holds. Returns (root, pieces, stem height)."""
    skin, flesh = apple_materials(color, rng)
    root = C.empty("apple", collection=col)
    root.parent = parent
    base = apple_mesh(radius, rng)
    height = max(v.co.z for v in base.verts)
    bounds = [math.inf] + list(cut_xs) + [-math.inf]
    pieces = []
    for k in range(len(bounds) - 1):
        hi, lo = bounds[k], bounds[k + 1]
        bm = base.copy()
        for x0, keep_neg in ((hi, True), (lo, False)):
            if math.isinf(x0):
                continue
            geom = bm.verts[:] + bm.edges[:] + bm.faces[:]
            res = bmesh.ops.bisect_plane(bm, geom=geom, plane_co=(x0, 0, 0), plane_no=(1, 0, 0),
                                         clear_outer=keep_neg, clear_inner=not keep_neg)
            cut_edges = [e for e in res["geom_cut"] if isinstance(e, bmesh.types.BMEdge)]
            new = bmesh.ops.edgeloop_fill(bm, edges=cut_edges)["faces"] if cut_edges else []
            for f in new:
                f.material_index = 1
                f.smooth = False
        for f in bm.faces:
            if f.material_index == 0:
                f.smooth = True
        # origin at the piece's bottom edge on the cut side, so a tilt tips it over that edge
        xs = [v.co.x for v in bm.verts]
        px = max(xs) if k < len(bounds) - 2 else min(xs)
        for v in bm.verts:
            v.co.x -= px
        me = bpy.data.meshes.new(f"apple_piece{k}")
        bm.to_mesh(me)
        bm.free()
        ob = C.link(bpy.data.objects.new(f"apple_piece{k}", me), col)
        ob.data.materials.append(skin)
        ob.data.materials.append(flesh)
        ob.parent = root
        ob.location = (px, 0, 0)
        ob["rest_x"] = px
        pieces.append(ob)
    base.free()
    stem_mat = C.material("apple_stem", (0.25, 0.16, 0.07), 0.8)
    stem = C.cylinder("apple_stem", (0, 0, height - 0.22 * radius + 0.012), 0.0025, 0.03, stem_mat, verts=8, collection=col)
    stem.rotation_euler = (rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3), 0)
    last = pieces[-1] if cut_xs[-1] > 0 else next(p for k, p in enumerate(pieces) if bounds[k + 1] <= 0 <= bounds[k])
    stem.parent = last
    stem.location = Vector(stem.location) - Vector(last.location)
    return root, pieces, height


def whole_apple(name, col, radius, color, rng, location):
    skin, _ = apple_materials(color, rng)
    bm = apple_mesh(radius, rng)
    for f in bm.faces:
        f.smooth = True
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    ob = C.link(bpy.data.objects.new(name, me), col)
    ob.data.materials.append(skin)
    ob.location = location
    ob.rotation_euler = (rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4), rng.uniform(0, 6.28))
    return ob


# ------------------------------------------------------------------ cook
ARM_BONES = ("UpperArm", "Forearm", "Hand")


def ik_error(arm, side, target):
    """Distance (m) between the wrist and its target."""
    return (arm.matrix_world @ arm.pose.bones[f"Bip01 {side} Hand"].head - target.matrix_world.translation).length


def nearest_axis(v):
    """Damped Track axis enum closest to local direction v."""
    i = max(range(3), key=lambda k: abs(v[k]))
    return ("TRACK_" if v[i] > 0 else "TRACK_NEGATIVE_") + "XYZ"[i]


def _rot_about(m, pivot, q):
    """Rotate armature-space matrix m by quaternion q about point pivot."""
    return Matrix.Translation(pivot) @ q.to_matrix().to_4x4() @ Matrix.Translation(-pivot) @ m


def solve_arm(arm, side, target_w, pole_w, w):
    """Analytic two-bone IK from joint positions (the imported bones' tails are not the next joint, so
    Blender's IK can't be used). Returns local pose quaternions for upper arm, forearm and hand, blended
    with the current (mocap) pose by weight w. Works in armature space with the current evaluated pose."""
    pb = {b: arm.pose.bones[f"Bip01 {side} {b}"] for b in ARM_BONES}
    cur = {b: Quaternion(p.rotation_quaternion) for b, p in pb.items()}
    if w <= 1e-4:
        return cur
    inv = arm.matrix_world.inverted()
    tgt = inv @ target_w
    T, P = tgt.translation, inv @ pole_w
    M = {b: p.matrix.copy() for b, p in pb.items()}
    S, E, W = M["UpperArm"].translation, M["Forearm"].translation, M["Hand"].translation
    a, b = (E - S).length, (W - E).length
    d = T - S
    c = min(max(d.length, abs(a - b) + 1e-3), (a + b) * 0.999)
    d.normalize()
    m = (P - S) - (P - S).dot(d) * d
    m = m.normalized() if m.length > 1e-6 else (E - S - (E - S).dot(d) * d).normalized()
    cos_a = (a * a + c * c - b * b) / (2 * a * c)
    sin_a = math.sqrt(max(0.0, 1 - cos_a * cos_a))
    E2 = S + a * (cos_a * d + sin_a * m)
    T2 = S + c * d
    q1 = (E - S).rotation_difference(E2 - S)
    M_ua = _rot_about(M["UpperArm"], S, q1)
    M_fa = _rot_about(M["Forearm"], S, q1)
    W1 = S + q1 @ (W - S)
    q2 = (W1 - E2).rotation_difference(T2 - E2)
    M_fa = _rot_about(M_fa, E2, q2)
    M_h = Matrix.Translation(T2) @ tgt.to_quaternion().to_matrix().to_4x4()
    new = {"UpperArm": M_ua, "Forearm": M_fa, "Hand": M_h}
    out = {}
    for name in ARM_BONES:
        p = pb[name]
        parent_m = new.get(p.parent.name.split()[-1]) if p.parent.name.split()[-1] in new else p.parent.matrix
        rel = p.parent.bone.matrix_local.inverted() @ p.bone.matrix_local
        local = rel.inverted() @ parent_m.inverted() @ new[name]
        q = local.to_quaternion()
        if q.dot(cur[name]) < 0:
            q.negate()
        out[name] = cur[name].slerp(q, w)
    return out


def finger_action(arm, name, frames, pose_r, pose_l):
    """NLA layer for the finger bones: per-frame curl blends (lists of (pose_a, pose_b, w))."""
    act = bpy.data.actions.new(name)
    rot = {}
    for side, seq in (("R", pose_r), ("L", pose_l)):
        for fi, bone in enumerate(FINGERS):
            digit, joint = int(bone[6]), len(bone) - 7
            vals = []
            for a, b, w in seq:
                ang = (1 - w) * POSES[a][digit][joint] + w * POSES[b][digit][joint]
                q = Quaternion((0, 0, 1), math.radians(ang))
                if digit == 0:  # thumb: curl about its own axis, plus a bit of opposition
                    q = Quaternion((0, 1, 0), math.radians(0.4 * ang)) @ Quaternion((0, 0, 1), math.radians(0.6 * ang))
                vals.append(q)
            rot[f"Bip01 {side} {bone}"] = vals
    for bone, qs in rot.items():
        path = f'pose.bones["{bone}"].rotation_quaternion'
        for i in range(4):
            fc = act.fcurves.new(path, index=i, action_group=bone)
            fc.keyframe_points.add(len(frames))
            co = np.empty(2 * len(frames), dtype=np.float32)
            co[0::2] = frames
            co[1::2] = [q[i] for q in qs]
            fc.keyframe_points.foreach_set("co", co)
            fc.update()
    return act


def key_matrix(obj, frame, m):
    obj.matrix_world = m
    obj.keyframe_insert("location", frame=frame)
    obj.keyframe_insert("rotation_quaternion", frame=frame)


def key_value(owner, prop, frame, value):
    setattr(owner, prop, value)
    owner.keyframe_insert(prop, frame=frame)


# ------------------------------------------------------------------ episode
class Episode:
    """One randomised chopping episode built into the open kitchen scene."""

    def __init__(self, spec, rng, avatar, apple_color, n_cuts, chop_hz, cook_x, board_jitter, clip_start,
                 apple_radius, distractors=()):
        self.spec, self.rng = spec, rng
        self.task = spec["task"]
        self.avatar, self.color_name = avatar, apple_color
        self.n_cuts, self.hz = n_cuts, chop_hz
        self.cook_x, self.board_jitter, self.clip_start = cook_x, board_jitter, clip_start
        self.radius, self.distractors = apple_radius, distractors
        self.col = C.get_collection("episode")
        self.timeline = []  # phase name per scene frame
        self.gripper = []  # (right grip closure, left hand engagement) per scene frame

    # --- scene objects
    def build(self):
        T, rng = self.task, self.rng
        counter = next(f for f in self.spec["furniture"] if f["name"] == T["counter"])
        front_y = counter["center"][1] - counter["size"][1] / 2
        self.work = C.empty("work", (self.cook_x, front_y - T["cook_back"], 0.0), 0.0, self.col)
        z_top = T["worktop_z"]
        bs = T["board"]["size"]
        jx, jy, jyaw = self.board_jitter
        self.board = C.empty("board_root", (T["board"]["at"][0] + jx, T["board"]["at"][1] + jy, z_top), math.radians(jyaw), self.col)
        self.board.parent = self.work
        wood_board("board", bs, self.col, self.board)
        self.board_top = z_top + bs[2]
        # apple: cuts from its +x end (knife side) towards -x
        r = self.radius
        span = rng.uniform(0.75, 0.95) * r
        self.cut_xs = list(np.linspace(0.62 * r, 0.62 * r - span, self.n_cuts))
        ax, ay = T["apple"]["at"][0] + jx, T["apple"]["at"][1] + jy
        self.apple_root, self.pieces, self.apple_h = sliced_apple(self.col, None, r, self.cut_xs,
                                                                  T["apple"]["colors"][self.color_name], rng)
        self.apple_root.parent = self.work
        self.apple_root.location = (ax, ay, self.board_top)
        self.apple_xy = (ax, ay)
        # knife lying on its side on the board, edge facing away from the apple
        self.knife = knife_object(self.col, T["knife"]["blade"], T["knife"]["handle"])
        self.knife.rotation_mode = "QUATERNION"
        self.knife.parent = self.work
        kx, ky = T["knife"]["at"]
        yaw = math.radians(T["knife"]["yaw_deg"] + rng.uniform(-8, 8))
        self.knife_rest = (Matrix.Translation((kx + jx, ky + jy, self.board_top + 0.009))
                           @ Matrix.Rotation(-yaw, 4, "Z") @ Matrix.Rotation(math.radians(-90), 4, "Y"))
        # distractor apples in the fruit bowl
        bowl = next((p for p in self.spec.get("props", []) if p["type"] == "fruit_bowl"), None)
        for i, cname in enumerate(self.distractors):
            if bowl is None:
                break
            a = rng.uniform(0, 2 * math.pi) + i * 2.1
            rr = rng.uniform(0.035, 0.06)
            whole_apple(f"bowl_apple{i}", self.col, rng.uniform(0.034, 0.042), T["apple"]["colors"][cname], rng,
                        (bowl["at"][0] + rr * math.cos(a), bowl["at"][1] + rr * math.sin(a), bowl["at"][2] + 0.02))
        # cook
        self.cook_root = C.empty("cook_root", (0, 0, 0), math.pi, self.col)  # rocketbox faces -y
        self.cook_root.parent = self.work
        self.arm = RB.import_avatar(self.avatar, self.col, "cook")
        self.arm.parent = self.cook_root
        self.arm.matrix_parent_inverse.identity()
        self.targets = {}
        for side in "RL":
            t = C.empty(f"target_{side}", collection=self.col)
            t.rotation_mode = "QUATERNION"
            pole = C.empty(f"pole_{side}", collection=self.col)
            pole.parent = self.work
            pole.location = ((0.45 if side == "R" else -0.45), -0.35, 0.75)
            self.targets[side] = (t, pole)
        self.look = C.empty("look_target", collection=self.col)
        bpy.context.view_layer.update()

    # --- motion
    def animate(self):
        rng, sc = self.rng, bpy.context.scene
        info = RB.load_clip(BODY_CLIP)
        ad = self.arm.animation_data_create()
        tr = ad.nla_tracks.new()
        tr.name = "body"
        act = RB._per_avatar_action(info, self.arm, "cook")
        s0 = self.clip_start
        strip = tr.strips.new("body", 1, act)
        strip.action_frame_start = s0
        strip.action_frame_end = BODY_WINDOW[1]
        strip.frame_start, strip.frame_end = 1, 1 + BODY_WINDOW[1] - s0
        strip.extrapolation = "HOLD"

        Wm = self.work.matrix_world.copy()
        bt = self.board_top
        ax, ay = self.apple_xy
        knife_rest = self.knife_rest
        tilt = math.radians(rng.uniform(4, 10))  # tip-down rake while chopping

        def knife_up(x, z):  # upright knife, edge-mid over (x, ay), grip frame at height z above the edge
            em = Vector(self.knife["edge_mid"])
            rot = Matrix.Rotation(math.radians(rng_yaw), 4, "Z") @ Matrix.Rotation(-tilt, 4, "X")
            off = rot @ em
            return Matrix.Translation((x - off.x, ay - off.y, z - off.z)) @ rot

        rng_yaw = rng.uniform(-6, 6)
        clear = 0.03
        z_top_stroke = bt + self.apple_h + clear
        z_bottom = bt + 0.0015

        # phase durations (s)
        d = dict(idle=rng.uniform(0.4, 0.8), reach=rng.uniform(0.8, 1.1), grasp=0.35, lift=rng.uniform(0.8, 1.0),
                 ret=rng.uniform(0.8, 1.0), release=0.35, retract=rng.uniform(0.7, 0.9), tail=0.5)
        period = 1.0 / self.hz
        t_chop0 = d["idle"] + d["reach"] + d["grasp"] + d["lift"]
        t_end_chop = t_chop0 + self.n_cuts * period
        T_total = t_end_chop + d["ret"] + d["release"] + d["retract"] + d["tail"]
        n_frames = int(math.ceil(T_total * FPS)) + 1
        self.n_frames = n_frames
        sc.frame_start, sc.frame_end = 1, n_frames
        sc.render.fps = FPS

        t_reach0 = d["idle"]
        t_grasp0 = t_reach0 + d["reach"]
        t_lift0 = t_grasp0 + d["grasp"]
        t_ret0 = t_end_chop
        t_rel0 = t_ret0 + d["ret"]
        t_retr0 = t_rel0 + d["release"]

        ready = knife_up(ax + self.cut_xs[0], z_top_stroke)

        def knife_pose(t):
            if t < t_lift0:
                return knife_rest, "reach" if t >= t_reach0 else "idle"
            if t < t_chop0:
                u = smooth((t - t_lift0) / d["lift"])
                return lerp_pose(knife_rest, ready, u, lift=0.06), "lift"
            if t < t_end_chop:
                k = min(int((t - t_chop0) / period), self.n_cuts - 1)
                u = (t - t_chop0 - k * period) / period
                x0 = ax + self.cut_xs[k]
                x1 = ax + self.cut_xs[min(k + 1, self.n_cuts - 1)]
                if u < 0.45:  # down stroke, accelerating
                    z = z_top_stroke + (z_bottom - z_top_stroke) * (u / 0.45) ** 1.6
                    x = x0
                elif u < 0.52:
                    z, x = z_bottom, x0
                else:  # back up while stepping over to the next cut
                    w = smooth((u - 0.52) / 0.48)
                    z = z_bottom + (z_top_stroke - z_bottom) * w
                    x = x0 + (x1 - x0) * w
                return knife_up(x, z), f"chop{k}"
            if t < t_rel0:
                u = smooth((t - t_ret0) / d["ret"])
                return lerp_pose(knife_up(ax + self.cut_xs[-1], z_top_stroke), knife_rest, u, lift=0.05), "return"
            return knife_rest, "release" if t < t_retr0 else ("retract" if t < t_retr0 + d["retract"] else "idle_end")

        def cut_done(t):
            """Number of completed cuts at time t (a cut completes at the bottom of its stroke)."""
            if t < t_chop0:
                return 0
            return int(min(self.n_cuts, math.floor((t - t_chop0) / period + 0.55)))

        def claw_pose(t):
            k = min(cut_done(t), self.n_cuts - 1)
            u = 0.0
            if t >= t_chop0 and cut_done(t) < self.n_cuts:  # glide back during the up stroke
                kk = int((t - t_chop0) / period)
                uu = (t - t_chop0 - kk * period) / period
                u = smooth((uu - 0.55) / 0.4) if uu > 0.55 else 0.0
                k = kk
            xa = self.cut_xs[min(k, self.n_cuts - 1)]
            xb = self.cut_xs[min(k + 1, self.n_cuts - 1)]
            x = ax + xa + (xb - xa) * u - 0.012
            return Matrix.Translation((x, ay, bt + self.apple_h * 0.92)) @ APPLE_CLAW

        def ramp(t, t0, dur):
            return smooth((t - t0) / dur)

        import ego_rig
        cal = ego_rig.calibrate(self.arm, ("Bip01 Head", "Bip01 Spine2"))
        head = self.arm.pose.bones["Bip01 Head"].constraints.new("DAMPED_TRACK")
        head.target, head.track_axis = self.look, nearest_axis(cal["Bip01 Head"][0])

        fr_r, fr_l = [], []
        frames = np.arange(1, n_frames + 1, dtype=np.float32)
        tR, tL = self.targets["R"][0], self.targets["L"][0]
        sacc = Vector((0, 0, 0))
        weights = []
        for f in range(1, n_frames + 1):
            t = (f - 1) / FPS
            kp, phase = knife_pose(t)
            self.timeline.append(phase)
            key_matrix(self.knife, f, Wm @ kp)
            key_matrix(tR, f, Wm @ kp @ KNIFE_GRIP)
            key_matrix(tL, f, Wm @ claw_pose(t))
            # right arm: reach for the knife, hold, let go
            w_r = ramp(t, t_reach0, d["reach"]) * (1 - ramp(t, t_retr0, d["retract"]))
            # left arm: comes in while the knife is lifted, leaves on the return
            w_l = ramp(t, t_lift0, d["lift"] * 0.9) * (1 - ramp(t, t_ret0 + 0.2, d["ret"] + 0.3))
            weights.append((w_r, w_l))
            g_r = ramp(t, t_grasp0 - 0.1, d["grasp"]) * (1 - ramp(t, t_rel0, d["release"]))
            fr_r.append(("relaxed", "knife", g_r))
            fr_l.append(("relaxed", "claw", w_l))
            # gaze: the knife before it is picked up, then the cut line, small saccades
            if f % 9 == 1:
                sacc = Vector((rng.normal(0, 0.012), rng.normal(0, 0.01), rng.normal(0, 0.006)))
            kx = (Wm @ kp).translation
            cut = Wm @ Vector((ax + self.cut_xs[min(cut_done(t), self.n_cuts - 1)], ay, bt + 0.4 * self.apple_h))
            g = ramp(t, t_lift0, d["lift"]) * (1 - ramp(t, t_rel0, d["release"] + d["retract"]))
            key_matrix(self.look, f, Matrix.Translation(kx.lerp(cut, g) + sacc))
            key_value(head, "influence", f, 0.5)
            self.gripper.append((g_r, w_l))
        act = finger_action(self.arm, "fingers", frames, fr_r, fr_l)
        ftr = ad.nla_tracks.new()
        ftr.name = "fingers"
        fs = ftr.strips.new("fingers", 1, act)
        fs.blend_type = "REPLACE"
        fs.extrapolation = "HOLD"
        self._animate_pieces(t_chop0, period)
        for o in (self.knife, tR, tL, self.look):
            for fc in o.animation_data.action.fcurves:
                for k in fc.keyframe_points:
                    k.interpolation = "LINEAR"
        self._bake_arms(weights)
        return dict(durations=d, period=period, t_chop0=t_chop0, n_frames=n_frames)

    def _bake_arms(self, weights):
        """Second pass: solve both arms on top of the evaluated mocap and key them as an NLA layer."""
        sc = bpy.context.scene
        keys = {f"Bip01 {s} {b}": [] for s in "RL" for b in ARM_BONES}
        poles = {s: self.targets[s][1] for s in "RL"}
        W = self.work.matrix_world
        Winv = W.inverted()
        z_clear = self.task["worktop_z"] + 0.035  # wrist height that keeps the hand out of the worktop
        y_edge = self.task["cook_back"] - 0.04  # counter front edge (work frame), with margin
        for f, (w_r, w_l) in enumerate(weights, start=1):
            sc.frame_set(f)
            for side, w in (("R", w_r), ("L", w_l)):
                # rest target = the mocap wrist, lifted clear of the worktop when it is over the counter
                hb = self.arm.pose.bones[f"Bip01 {side} Hand"]
                rest = self.arm.matrix_world @ hb.matrix
                loc, rot, _ = rest.decompose()
                lw = Winv @ loc
                lift = max(0.0, z_clear - lw.z) if lw.y > y_edge else 0.0
                rest = Matrix.Translation(loc + Vector((0, 0, lift))) @ rot.to_matrix().to_4x4()
                task = self.targets[side][0].matrix_world
                if w <= 1e-4 and lift <= 0:
                    q = {b: Quaternion(self.arm.pose.bones[f"Bip01 {side} {b}"].rotation_quaternion) for b in ARM_BONES}
                else:
                    tgt = lerp_pose(rest, task, w)
                    q = solve_arm(self.arm, side, tgt, poles[side].matrix_world.translation, 1.0)
                for b in ARM_BONES:
                    keys[f"Bip01 {side} {b}"].append(q[b])
        act = bpy.data.actions.new("arms_ik")
        frames = np.arange(1, len(weights) + 1, dtype=np.float32)
        for bone, qs in keys.items():
            for k in range(1, len(qs)):  # keep hemispheres continuous for linear interpolation
                if qs[k].dot(qs[k - 1]) < 0:
                    qs[k].negate()
            path = f'pose.bones["{bone}"].rotation_quaternion'
            for i in range(4):
                fc = act.fcurves.new(path, index=i, action_group=bone)
                fc.keyframe_points.add(len(frames))
                co = np.empty(2 * len(frames), dtype=np.float32)
                co[0::2], co[1::2] = frames, [q[i] for q in qs]
                fc.keyframe_points.foreach_set("co", co)
                fc.update()
        ad = self.arm.animation_data
        tr = ad.nla_tracks.new(prev=ad.nla_tracks["body"])
        tr.name = "arms"
        st = tr.strips.new("arms", 1, act)
        st.blend_type = "REPLACE"
        st.extrapolation = "HOLD"

    def _animate_pieces(self, t_chop0, period):
        rng = self.rng
        for k, p in enumerate(self.pieces[:-1]):
            t_cut = t_chop0 + (k + 0.47) * period
            f_cut = 1 + int(round(t_cut * FPS))
            p.rotation_mode = "XYZ"
            rest = Vector(p.location)
            p.keyframe_insert("location", frame=f_cut)
            p.keyframe_insert("rotation_euler", frame=f_cut)
            thin = (self.cut_xs[k - 1] if k else self.radius) - self.cut_xs[k]
            ang = math.radians(rng.uniform(55, 85) if thin < 0.4 * self.radius else rng.uniform(8, 18))
            p.location = rest + Vector((rng.uniform(0.003, 0.008), rng.normal(0, 0.002), 0))
            p.rotation_euler = (rng.normal(0, 0.05), ang, rng.normal(0, 0.1))
            p.keyframe_insert("location", frame=f_cut + 5)
            p.keyframe_insert("rotation_euler", frame=f_cut + 5)
            # the knife's next strokes nudge earlier slices a little further away
            p.location = p.location + Vector((0.004 * (self.n_cuts - k - 1) * 0.5, 0, 0))
            p.keyframe_insert("location", frame=f_cut + int(FPS * period * (self.n_cuts - k)))

    def instruction(self):
        n = self.n_cuts + 1
        c = self.color_name
        pool = [f"chop the {c} apple into slices", f"slice the {c} apple on the cutting board",
                f"cut the {c} apple into {n} pieces", f"pick up the knife and chop the {c} apple",
                "slice the apple on the board", f"cut the {c} apple on the cutting board into slices"]
        return pool[int(self.rng.integers(len(pool)))]


def lerp_pose(a, b, u, lift=0.0):
    """Interpolate two 4x4 poses (lerp + slerp) with an upward arc of height `lift`."""
    ta, ra, _ = a.decompose()
    tb, rb, _ = b.decompose()
    t = ta.lerp(tb, u) + Vector((0, 0, lift * math.sin(math.pi * u)))
    return Matrix.Translation(t) @ ra.slerp(rb, u).to_matrix().to_4x4()
