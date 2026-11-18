"""Body-worn + fixed camera rig for egocentric data.

Body cameras follow bones (head, chest, wrists); their world pose is computed from the bone every frame
and keyframed, so the same matrices are rendered and logged. Fixed cameras look at the cutting board.

Bone axes from the FBX import are arbitrary (a bone's tail is not the next joint), so head and chest
frames are calibrated in the rest pose (`calibrate`) and wrist frames come from joint positions
(elbow -> wrist) plus the back-of-hand direction (hand bone -y; fingers = +x, thumb side = +z for R).

Blender cameras look down their local -Z with +Y up. Each mount is given in an anatomical frame
(right, forward, up) built from its bone, as an offset plus pitch-down / yaw angles.
"""
import math

import bpy
import numpy as np
from mathutils import Matrix, Vector

import common as C

# name: bone, offset (right, forward, up) in m from the bone's anchor, pitch down (deg), yaw (deg, + = right), FOV (deg)
# anchors: head = between the eyebrows (glasses), chest = Spine2 joint, wrists = the wrist joint (watch strap)
BODY_MOUNTS = {
    "head": dict(bone="Bip01 Head", anchor="Bip01 MMiddleEyebrow", offset=(0.0, 0.035, 0.0), pitch=30, yaw=-4, fov=110),
    "chest": dict(bone="Bip01 Spine2", offset=(0.0, 0.2, 0.04), pitch=50, yaw=-12, fov=120),
    "left_wrist": dict(bone="Bip01 L Hand", offset=(0.0, -0.06, 0.08), pitch=35, yaw=0, fov=100, wrist=True),
    "right_wrist": dict(bone="Bip01 R Hand", offset=(0.0, -0.06, 0.08), pitch=35, yaw=0, fov=100, wrist=True),
}
# name: position, look-at point, both in the work frame (x = her right, y = forward, z up), FOV
FIXED_MOUNTS = {
    "scene_front": dict(pos=(-0.05, 0.62, 1.42), at=(-0.03, 0.36, 0.92), fov=70),
    "scene_left": dict(pos=(-0.85, 0.3, 1.38), at=(-0.03, 0.38, 0.92), fov=60),
    "scene_right": dict(pos=(0.85, 0.3, 1.38), at=(-0.03, 0.38, 0.92), fov=60),
}
VIEWS = list(BODY_MOUNTS) + list(FIXED_MOUNTS)


def calibrate(arm, bones=("Bip01 Head", "Bip01 Spine2")):
    """Forward/up of each bone in its own local frame, measured in the rest pose (T-pose, upright).
    Facing = horizontal head -> nose direction in that pose (the FBX import rotates the armature object,
    so its axes don't tell). Bone axes from the import are arbitrary, so this is the robust way."""
    arm.data.pose_position = "REST"
    bpy.context.view_layer.update()
    mw = arm.matrix_world
    face = mw @ arm.pose.bones["Bip01 MNose"].head - mw @ arm.pose.bones["Bip01 Head"].head
    fwd_w, up_w = Vector((face.x, face.y, 0.0)).normalized(), Vector((0, 0, 1))
    cal = {}
    for b in bones:
        r = (arm.matrix_world @ arm.pose.bones[b].matrix).to_3x3().normalized()
        ri = r.inverted()
        cal[b] = (ri @ fwd_w, ri @ up_w)
    arm.data.pose_position = "POSE"
    bpy.context.view_layer.update()
    arm["rig_cal"] = {b: [list(f), list(u)] for b, (f, u) in cal.items()}
    return cal


def anatomical_frame(arm, mount):
    """(origin, right, forward, up) of a mount in world space."""
    mw = arm.matrix_world
    pb = arm.pose.bones[mount["bone"]]
    r = (mw @ pb.matrix).to_3x3().normalized()
    if mount.get("wrist"):
        # forward = elbow -> wrist, up = back of the hand (hand bone's -y)
        side = mount["bone"].split()[1]
        origin = mw @ pb.head
        fwd = (origin - mw @ arm.pose.bones[f"Bip01 {side} Forearm"].head).normalized()
        up = -r.col[1].normalized()
    else:
        f_l, u_l = (Vector(v) for v in arm["rig_cal"][mount["bone"]])
        fwd, up = (r @ f_l).normalized(), (r @ u_l).normalized()
        origin = mw @ arm.pose.bones[mount.get("anchor", mount["bone"])].head
    up = (up - up.dot(fwd) * fwd).normalized()
    right = fwd.cross(up).normalized()
    return origin, right, fwd, up


def mount_matrix(origin, right, fwd, up, offset, pitch, yaw, jitter=None):
    """World matrix of a camera at `offset` in the (right, fwd, up) frame, looking forward, pitched down."""
    j = jitter or dict(dp=(0, 0, 0), dr=(0, 0, 0))
    o = Vector(offset) + Vector(j["dp"])
    pos = origin + right * o.x + fwd * o.y + up * o.z
    p, y, roll = (math.radians(a) for a in (pitch + j["dr"][0], yaw + j["dr"][1], j["dr"][2]))
    look = (fwd * math.cos(y) + right * math.sin(y))
    look = (look * math.cos(p) - up * math.sin(p)).normalized()
    cam_up = (up - up.dot(look) * look).normalized()
    cam_up = Matrix.Rotation(roll, 3, look) @ cam_up
    cam_x = look.cross(cam_up).normalized()  # camera +X = right
    rot = Matrix((cam_x, cam_up, -look)).transposed()
    m = rot.to_4x4()
    m.translation = pos
    return m


def look_matrix(pos, at, up=(0, 0, 1)):
    pos, at = Vector(pos), Vector(at)
    look = (at - pos).normalized()
    cam_x = look.cross(Vector(up)).normalized()
    cam_up = cam_x.cross(look)
    m = Matrix((cam_x, cam_up, -look)).transposed().to_4x4()
    m.translation = pos
    return m


class Rig:
    def __init__(self, arm, work, col, res=256, rng=None, jitter=True, fisheye=False):
        self.arm, self.work, self.res = arm, work, res
        self.cams, self.jit = {}, {}
        rng = rng or np.random.default_rng(0)
        if "rig_cal" not in arm:
            calibrate(arm)
        for name in VIEWS:
            mount = BODY_MOUNTS.get(name) or FIXED_MOUNTS[name]
            cam = bpy.data.cameras.new(f"ego_{name}")
            cam.sensor_fit = "HORIZONTAL"
            cam.sensor_width = 36.0
            cam.angle = math.radians(mount["fov"])
            cam.clip_start = 0.01
            cam.clip_end = 30.0
            if fisheye and name == "head":  # Aria-style: Cycles only
                cam.type = "PANO"
                cam.panorama_type = "FISHEYE_EQUISOLID"
                cam.fisheye_fov = math.radians(mount["fov"])
                cam.fisheye_lens = 10.5
            obj = C.link(bpy.data.objects.new(f"ego_{name}", cam), col)
            obj.rotation_mode = "QUATERNION"
            self.cams[name] = obj
            s = 1.0 if jitter else 0.0
            self.jit[name] = dict(dp=tuple(rng.normal(0, 0.006 * s, 3)), dr=tuple(rng.normal(0, 2.0 * s, 3)))

    def matrices(self):
        """World matrix of every camera for the current (evaluated) frame."""
        out = {}
        W = self.work.matrix_world
        for name in VIEWS:
            if name in BODY_MOUNTS:
                m = BODY_MOUNTS[name]
                out[name] = mount_matrix(*anatomical_frame(self.arm, m), m["offset"], m["pitch"], m["yaw"], self.jit[name])
            else:
                m = FIXED_MOUNTS[name]
                j = self.jit[name]
                pos = W @ (Vector(m["pos"]) + Vector(j["dp"]) * 3)
                at = W @ (Vector(m["at"]) + Vector(j["dp"]))
                out[name] = look_matrix(pos, at)
        return out

    def key(self, frame):
        mats = self.matrices()
        for name, m in mats.items():
            obj = self.cams[name]
            obj.matrix_world = m
            obj.keyframe_insert("location", frame=frame)
            obj.keyframe_insert("rotation_quaternion", frame=frame)
        return mats

    def intrinsics(self, name):
        cam = self.cams[name].data
        if cam.type == "PANO":
            return dict(model="fisheye_equisolid", fov_deg=math.degrees(cam.fisheye_fov), width=self.res, height=self.res)
        f = self.res / (2 * math.tan(cam.angle / 2))
        return dict(model="pinhole", fx=f, fy=f, cx=self.res / 2, cy=self.res / 2, width=self.res, height=self.res,
                    fov_deg=math.degrees(cam.angle))


def cam_to_world_cv(m):
    """Blender camera matrix -> 4x4 camera-to-world in the OpenCV convention (x right, y down, z forward)."""
    flip = Matrix(((1, 0, 0, 0), (0, -1, 0, 0), (0, 0, -1, 0), (0, 0, 0, 1)))
    return [list(r) for r in (m @ flip)]
