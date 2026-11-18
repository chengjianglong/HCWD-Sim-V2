"""Photo-scanned PBR materials and HDRI skies (CC0, Poly Haven), with procedural fallbacks.

configs/assets.json maps semantic slots (floor_wood, worktop_stone, ...) to Poly Haven ids; scripts/fetch_assets.py
downloads them to assets/polyhaven. Everything here degrades gracefully: when a slot's files are missing,
`pbr_material` returns the caller's procedural fallback, and `hdri_world` leaves the Nishita sky in place.

Materials are box-projected (triplanar) in world space at the texture's real-world size, so they need no UVs (our
procedural boxes have none), stay continuous across adjacent wall segments, and keep true scale everywhere.
Each one has a MixRGB node named "tint" between the photo texture and the BSDF: recolouring a material (the
egocentric generator's randomisation) sets that node's colour instead of the base colour.
"""
import json
import math
import os

import bpy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PH = os.path.join(ROOT, "assets", "polyhaven")
_MANIFEST = None
ENABLED = True  # builders set this from --realism; False = procedural everywhere, even when assets are downloaded


def configure(realism):
    """--realism auto|photo|procedural -> whether photo assets are used; returns the output folder for the build."""
    global ENABLED
    have = os.path.exists(os.path.join(PH, "textures"))
    if realism == "photo" and not have:
        raise SystemExit("--realism photo: no assets in assets/polyhaven (run scripts/fetch_assets.py)")
    ENABLED = realism == "photo" or (realism == "auto" and have)
    return os.path.join(ROOT, "assets", "photoreal") if ENABLED else os.path.join(ROOT, "assets")


def manifest():
    global _MANIFEST
    if _MANIFEST is None:
        with open(os.path.join(ROOT, "configs", "assets.json")) as f:
            _MANIFEST = json.load(f)
    return _MANIFEST


def res():
    return manifest().get("resolution", "2k")


def texture_files(aid):
    if not ENABLED:
        return None
    d = os.path.join(PH, "textures", aid)
    files = {k: os.path.join(d, f"{aid}_{k}_{res()}.jpg") for k in ("diff", "nor_gl", "rough", "ao")}
    return {k: p for k, p in files.items() if os.path.exists(p)} if os.path.exists(files["diff"]) else None


def texture_size_m(aid):
    """Real-world width of one texture tile in metres (Poly Haven's dimensions are in mm)."""
    try:
        with open(os.path.join(PH, "textures", aid, "info.json")) as f:
            dims = json.load(f).get("dimensions") or [2000]
        return max(0.2, float(dims[0]) / 1000.0)
    except (OSError, ValueError):
        return 2.0


def slot_ids(slot):
    return manifest()["textures"].get(slot, [])


def available(slot, variant=0):
    ids = slot_ids(slot)
    return bool(ids) and texture_files(ids[variant % len(ids)]) is not None


def _image(path, non_color=False):
    img = bpy.data.images.load(path, check_existing=True)
    if non_color:
        img.colorspace_settings.name = "Non-Color"
    return img


def _tex(nt, img, vec, blend=0.25, x=0, y=0):
    t = nt.nodes.new("ShaderNodeTexImage")
    t.image = img
    t.projection = "BOX"
    t.projection_blend = blend
    t.location = (x, y)
    nt.links.new(vec, t.inputs["Vector"])
    return t


def mean_albedo(img):
    """Average linear colour of an image (cached on the image): what the photo texture contributes on average."""
    if "mean_lin" in img:
        return list(img["mean_lin"])
    import numpy as np
    px = np.empty(len(img.pixels), np.float32)
    img.pixels.foreach_get(px)
    rgb = px.reshape(-1, 4)[::7, :3].astype(np.float64)
    if img.colorspace_settings.name == "sRGB":  # byte images come back as sRGB-encoded values
        rgb = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    m = [float(x) for x in rgb.mean(0)]
    img["mean_lin"] = m
    return m


def albedo_tint(mat, color):
    """Tint that gives the photo texture of `mat` the average colour `color` (linear RGB), keeping its detail."""
    mean = mat.get("pbr_mean") or (0.5, 0.5, 0.5)
    return [min(4.0, c / max(m, 1e-3)) for c, m in zip(color[:3], mean)]


def pbr_material(name, slot, variant=0, tint=(1.0, 1.0, 1.0), albedo=None, scale=1.0, roughness_scale=1.0,
                 normal_strength=1.0, fallback=None):
    """Photo-textured material `name` from texture slot `slot` (variant index into its id list).

    tint     multiplied into the colour texture (keep it light; it is the recolouring knob)
    albedo   instead of a tint: the average linear colour the material should have (a spec / palette colour);
             the tint is computed from the texture's own mean, so detail stays and colour is under control
    scale    texture repeats per real-world tile size (2 = features twice as small)
    fallback zero-arg callable returning a procedural material when the slot's files are missing
    """
    mat = bpy.data.materials.get(name)
    if mat is not None:
        return mat
    ids = slot_ids(slot)
    aid = ids[variant % len(ids)] if ids else None
    files = texture_files(aid) if aid else None
    if files is None:
        if fallback is not None:
            return fallback()
        from common import material
        return material(name, tint)
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    mp = nt.nodes.new("ShaderNodeMapping")
    s = scale / texture_size_m(aid)
    mp.inputs["Scale"].default_value = (s, s, s)
    nt.links.new(geo.outputs["Position"], mp.inputs["Vector"])
    vec = mp.outputs["Vector"]
    diff = _tex(nt, _image(files["diff"]), vec, x=-600, y=300)
    diff.name = "pbr_diff"
    mix = nt.nodes.new("ShaderNodeMixRGB")
    mix.name = mix.label = "tint"
    mix.blend_type = "MULTIPLY"
    mix.inputs["Fac"].default_value = 1.0
    mix.inputs["Color2"].default_value = (*tint[:3], 1.0)
    nt.links.new(diff.outputs["Color"], mix.inputs["Color1"])
    col = mix.outputs["Color"]
    if "ao" in files:
        ao = _tex(nt, _image(files["ao"], True), vec, x=-600, y=0)
        aom = nt.nodes.new("ShaderNodeMixRGB")
        aom.blend_type = "MULTIPLY"
        aom.inputs["Fac"].default_value = 0.8
        nt.links.new(col, aom.inputs["Color1"])
        nt.links.new(ao.outputs["Color"], aom.inputs["Color2"])
        col = aom.outputs["Color"]
    nt.links.new(col, bsdf.inputs["Base Color"])
    if "rough" in files:
        r = _tex(nt, _image(files["rough"], True), vec, x=-600, y=-300)
        rm = nt.nodes.new("ShaderNodeMath")
        rm.operation = "MULTIPLY"
        rm.inputs[1].default_value = roughness_scale
        nt.links.new(r.outputs["Color"], rm.inputs[0])
        nt.links.new(rm.outputs["Value"], bsdf.inputs["Roughness"])
    if "nor_gl" in files:
        n = _tex(nt, _image(files["nor_gl"], True), vec, x=-600, y=-600)
        nm = nt.nodes.new("ShaderNodeNormalMap")
        nm.inputs["Strength"].default_value = normal_strength
        nt.links.new(n.outputs["Color"], nm.inputs["Color"])
        nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    mat["pbr_slot"], mat["pbr_id"] = slot, aid
    mat["pbr_mean"] = mean_albedo(diff.image)
    if albedo is not None:
        set_tint(mat, albedo_tint(mat, albedo))
        mat["albedo"] = list(albedo[:3])
    mat.diffuse_color = (*(albedo or tint)[:3], 1.0)
    return mat


def set_variant(mat, aid):
    """Swap a pbr_material's photo textures to another Poly Haven id of the same slot (domain randomisation)."""
    files = texture_files(aid)
    if files is None or mat.get("pbr_id") is None:
        return False
    for node in mat.node_tree.nodes:
        if node.type != "TEX_IMAGE" or node.image is None:
            continue
        for key, path in files.items():
            if node.image.name.endswith(f"_{key}_{res()}.jpg"):
                node.image = _image(path, key != "diff")
    mp = next(n for n in mat.node_tree.nodes if n.type == "MAPPING")
    s = mp.inputs["Scale"].default_value[0] * texture_size_m(mat["pbr_id"]) / texture_size_m(aid)
    mp.inputs["Scale"].default_value = (s, s, s)
    mat["pbr_id"] = aid
    mat["pbr_mean"] = mean_albedo(mat.node_tree.nodes["pbr_diff"].image)
    if mat.get("albedo"):
        set_tint(mat, albedo_tint(mat, mat["albedo"]))
    return True


def set_tint(mat, color):
    node = mat.node_tree.nodes.get("tint") if mat.use_nodes else None
    if node is None:
        return False
    node.inputs["Color2"].default_value = (*color[:3], 1.0)
    return True


def set_albedo(mat, color):
    """Recolour a photo material to average colour `color` (the egocentric generator's palette randomisation)."""
    if mat.get("pbr_mean") is None:
        return False
    mat["albedo"] = list(color[:3])
    mat.diffuse_color = (*color[:3], 1.0)
    return set_tint(mat, albedo_tint(mat, color))


# ------------------------------------------------------------------------------------------------ HDRI sky
def hdri_path(aid):
    if not ENABLED:
        return None
    p = os.path.join(PH, "hdris", f"{aid}_{res()}.hdr")
    return p if os.path.exists(p) else None


def hdri_ids(kind="sky"):
    return [i for i in manifest()["hdris"].get(kind, []) if hdri_path(i)]


def hdri_world(world, aid, strength=1.0, rotation_deg=0.0):
    """Light the scene with an HDRI: an Environment Texture feeds the existing node named "Background".

    The HDRI already contains the sun, so the caller turns the sun lamp off (EEVEE extracts a shadow-casting sun
    from the HDRI via world.sun_threshold; Cycles importance-samples it). Returns False if the file is missing.
    """
    path = hdri_path(aid) if aid else None
    if path is None:
        return False
    nt = world.node_tree
    bg = nt.nodes["Background"]
    env = nt.nodes.get("hdri_env")
    if env is None:
        env = nt.nodes.new("ShaderNodeTexEnvironment")
        env.name = "hdri_env"
        tc = nt.nodes.new("ShaderNodeTexCoord")
        mp = nt.nodes.new("ShaderNodeMapping")
        mp.name = "hdri_rot"
        nt.links.new(tc.outputs["Generated"], mp.inputs["Vector"])
        nt.links.new(mp.outputs["Vector"], env.inputs["Vector"])
    env.image = bpy.data.images.load(path, check_existing=True)
    nt.nodes["hdri_rot"].inputs["Rotation"].default_value = (0, 0, math.radians(rotation_deg))
    nt.links.new(env.outputs["Color"], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = strength
    world["hdri"] = aid
    if hasattr(world, "sun_threshold"):  # EEVEE Next: pull the sun out of the HDRI as a shadowed light
        world.sun_threshold = 2.0
        world.sun_angle = math.radians(1.0)
        world.use_sun_shadow = True
    return True
