"""Build the kitchen for the egocentric chopping task from configs/kitchen.json.

    tools/blender/blender -b -P blender/build_kitchen.py -- [--spec configs/kitchen.json] [--out assets]

Writes assets/kitchen.blend and assets/kitchen_layout.json. The kitchen uses the room.json schema, so the
walls, openings, lighting and plain furniture come from build_room.py; this file adds the kitchen pieces
(counters, sink, stove, fridge, wall cabinets, hood, island, stools) and the small static props.
Materials that the episode generator recolours are named: cabinet, worktop, backsplash, floor, walls, handle.
"""
import argparse
import math
import os
import sys

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_room as BR  # noqa: E402
import common as C  # noqa: E402
from worldsim import room_spec  # noqa: E402

DEFAULT_SPEC = os.path.join(C.ROOT, "configs", "kitchen.json")
TOE, TOP = 0.1, 0.035  # toe-kick height, worktop thickness


# ------------------------------------------------------------------ materials
def tile_material(name, color, roughness=0.35, scale=6.0, mortar=(0.8, 0.79, 0.76)):
    """Square tiles with grout lines (brick texture with no offset)."""
    mat = C.material(name, color, roughness)
    nt = mat.node_tree
    if "tiles" in nt.nodes:
        return mat
    tex = nt.nodes.new("ShaderNodeTexBrick")
    tex.name = "tiles"
    tex.offset = 0.0
    tex.squash = 1.0
    tex.inputs["Scale"].default_value = scale
    tex.inputs["Mortar Size"].default_value = 0.012
    tex.inputs["Brick Width"].default_value = 0.5
    tex.inputs["Row Height"].default_value = 0.5
    tex.inputs["Color1"].default_value = (*color[:3], 1)
    tex.inputs["Color2"].default_value = tuple(min(1.0, c * 1.06) for c in color[:3]) + (1,)
    tex.inputs["Mortar"].default_value = (*mortar, 1)
    coord = nt.nodes.new("ShaderNodeTexCoord")
    nt.links.new(coord.outputs["Object"], tex.inputs["Vector"])
    nt.links.new(tex.outputs["Color"], nt.nodes["Principled BSDF"].inputs["Base Color"])
    return mat


def stone_material(name, color, roughness=0.25):
    mat = C.material(name, color, roughness)
    C._noise_color(mat, color, scale=14.0, detail=6.0, var=0.12)
    return mat


def kitchen_materials(spec):
    M = spec.get("materials", {})
    return dict(
        cabinet=C.material("cabinet", (0.42, 0.5, 0.44), 0.45),
        cabinet_inner=C.material("cabinet_inner", (0.85, 0.84, 0.8), 0.6),
        worktop=stone_material("worktop", (0.86, 0.85, 0.82)),
        backsplash=tile_material("backsplash", (0.93, 0.93, 0.9), 0.15, scale=9.0),
        handle=C.material("handle", (0.75, 0.62, 0.38), 0.25, metallic=1.0),
        steel=C.material("steel", (0.78, 0.78, 0.8), 0.22, metallic=1.0),
        black_glass=C.material("black_glass", (0.01, 0.01, 0.012), 0.05),
        appliance=C.material("appliance", (0.82, 0.82, 0.83), 0.3, metallic=0.7),
        plinth=C.material("plinth", (0.08, 0.08, 0.08), 0.6),
        wood=C.wood_material("island_wood", (0.45, 0.3, 0.17), 0.45),
        ceramic=C.material("ceramic_white", (0.92, 0.91, 0.88), 0.2),
        terracotta=C.material("terracotta", (0.62, 0.3, 0.18), 0.8),
        leaf=C.material("herb_leaf", (0.12, 0.35, 0.08), 0.7),
        floor=tile_material("floor", M.get("floor", {}).get("color", (0.62, 0.6, 0.56)),
                            M.get("floor", {}).get("roughness", 0.35), scale=1.6, mortar=(0.45, 0.44, 0.42)),
    )


# ------------------------------------------------------------------ furniture (local frame: front = -y)
def _doors(f, p, mats, n, z0, z1, depth_front, gap=0.004, name="door", handle_top=True):
    w = f["size"][0]
    dw = w / n
    for i in range(n):
        x = -w / 2 + dw * (i + 0.5)
        C.box(f"{f['name']}_{name}{i}", (x, depth_front - 0.01, (z0 + z1) / 2), (dw - 2 * gap, 0.02, z1 - z0 - 2 * gap),
              mats["cabinet"], p, bevel=0.003)
        hx = x + (dw / 2 - 0.05) * (1 if i % 2 else -1)
        hz = z1 - 0.07 if handle_top else z0 + 0.07
        C.box(f"{f['name']}_{name}{i}_handle", (hx, depth_front - 0.03, hz), (0.012, 0.015, 0.14), mats["handle"], p)


def _base_cabinet(f, p, mats, doors=None, worktop=True):
    w, d, h = f["size"]
    C.box(f"{f['name']}_plinth", (0, 0.03, TOE / 2), (w, d - 0.06, TOE), mats["plinth"], p)
    C.box(f"{f['name']}_body", (0, 0.01, TOE + (h - TOP - TOE) / 2), (w, d - 0.02, h - TOP - TOE), mats["cabinet_inner"], p)
    if doors:
        _doors(f, p, mats, doors, TOE, h - TOP, -d / 2 + 0.01)
    if worktop:
        C.box(f"{f['name']}_top", (0, -0.01, h - TOP / 2), (w, d + 0.02, TOP), mats["worktop"], p, bevel=0.004)


def _backsplash(f, p, mats, height=0.55):
    w, d, h = f["size"]
    C.box(f"{f['name']}_splash", (0, d / 2 - 0.006, h + height / 2), (w, 0.012, height), mats["backsplash"], p)


def f_counter(f, p, mat, mats):
    _base_cabinet(f, p, mats, doors=f.get("doors", 2))
    _backsplash(f, p, mats)


def f_sink(f, p, mat, mats):
    w, d, h = f["size"]
    _base_cabinet(f, p, mats, doors=2, worktop=False)
    _backsplash(f, p, mats)
    bw, bd = w - 0.22, d - 0.24
    # worktop frame around the basin, then a steel basin set into it
    for i, (cx, sx) in enumerate(((-(w + bw) / 4, (w - bw) / 2), ((w + bw) / 4, (w - bw) / 2))):
        C.box(f"{f['name']}_top{i}", (cx, -0.01, h - TOP / 2), (sx, d + 0.02, TOP), mats["worktop"], p)
    front, back = -d / 2 - 0.02, d / 2
    y0, y1 = -bd / 2 - 0.02, bd / 2 - 0.02
    C.box(f"{f['name']}_top_f", (0, (front + y0) / 2, h - TOP / 2), (bw, y0 - front, TOP), mats["worktop"], p)
    C.box(f"{f['name']}_top_b", (0, (y1 + back) / 2, h - TOP / 2), (bw, back - y1, TOP), mats["worktop"], p)
    C.box(f"{f['name']}_basin", (0, -0.02, h - 0.11), (bw, bd, 0.012), mats["steel"], p)
    for i, (c, s) in enumerate((((0, y0 + 0.005, h - 0.055), (bw, 0.01, 0.11)), ((0, y1 - 0.005, h - 0.055), (bw, 0.01, 0.11)),
                                ((-bw / 2 + 0.005, -0.02, h - 0.055), (0.01, bd, 0.11)), ((bw / 2 - 0.005, -0.02, h - 0.055), (0.01, bd, 0.11)))):
        C.box(f"{f['name']}_basin_wall{i}", c, s, mats["steel"], p)
    # tap: column + spout
    C.cylinder(f"{f['name']}_tap", (0, d / 2 - 0.08, h + 0.14), 0.014, 0.28, mats["steel"], p, verts=16)
    spout = C.cylinder(f"{f['name']}_spout", (0, d / 2 - 0.17, h + 0.27), 0.011, 0.19, mats["steel"], p, verts=16)
    spout.rotation_euler = (math.pi / 2, 0, 0)
    C.box(f"{f['name']}_tap_lever", (0.03, d / 2 - 0.08, h + 0.22), (0.06, 0.012, 0.012), mats["steel"], p)


def f_stove(f, p, mat, mats):
    w, d, h = f["size"]
    _base_cabinet(f, p, mats, worktop=False)
    _backsplash(f, p, mats)
    C.box(f"{f['name']}_oven_door", (0, -d / 2 + 0.005, TOE + 0.33), (w - 0.06, 0.02, 0.52), mats["black_glass"], p, bevel=0.004)
    C.box(f"{f['name']}_oven_frame", (0, -d / 2 + 0.012, TOE + 0.33), (w - 0.02, 0.012, 0.58), mats["appliance"], p)
    C.box(f"{f['name']}_oven_handle", (0, -d / 2 - 0.025, TOE + 0.56), (w - 0.16, 0.02, 0.02), mats["steel"], p)
    C.box(f"{f['name']}_panel", (0, -d / 2 + 0.01, h - 0.12), (w - 0.02, 0.02, 0.12), mats["appliance"], p)
    for i in range(4):
        k = C.cylinder(f"{f['name']}_knob{i}", (-w / 2 + 0.12 + i * (w - 0.24) / 3, -d / 2 - 0.01, h - 0.12), 0.018, 0.03,
                       mats["black_glass"], p, verts=16)
        k.rotation_euler = (math.pi / 2, 0, 0)
    C.box(f"{f['name']}_hob", (0, 0, h - TOP / 2), (w, d + 0.02, TOP), mats["black_glass"], p, bevel=0.003)
    ring = C.material("hob_ring", (0.25, 0.25, 0.26), 0.4)
    for i, (x, y, r) in enumerate(((-0.17, -0.12, 0.09), (0.17, -0.12, 0.07), (-0.17, 0.14, 0.07), (0.17, 0.14, 0.1))):
        C.cylinder(f"{f['name']}_ring{i}", (x, y, h + 0.0005), r, 0.001, ring, p, verts=40)


def f_fridge(f, p, mat, mats):
    w, d, h = f["size"]
    C.box(f"{f['name']}_body", (0, 0, h / 2), (w, d, h), mats["appliance"], p, bevel=0.01)
    split = h * 0.62
    C.box(f"{f['name']}_gap", (0, -d / 2, split), (w - 0.01, 0.004, 0.006), mats["plinth"], p)
    for z0, z1 in ((0.05, split - 0.06), (split + 0.06, h - 0.05)):
        C.box(f"{f['name']}_handle_{z0:.2f}", (w / 2 - 0.06, -d / 2 - 0.03, (z0 + z1) / 2 if z1 - z0 < 0.6 else z1 - 0.3),
              (0.02, 0.025, min(0.5, z1 - z0)), mats["steel"], p)


def f_upper_cabinet(f, p, mat, mats):
    w, d, h = f["size"]
    z0 = f.get("z0", 1.45)
    C.box(f"{f['name']}_body", (0, 0.01, z0 + h / 2), (w, d - 0.02, h), mats["cabinet_inner"], p)
    _doors(f, p, mats, f.get("doors", 2), z0, z0 + h, -d / 2 + 0.01, handle_top=False)
    # under-cabinet LED strip lights the worktop
    C.box(f"{f['name']}_led", (0, -d / 2 + 0.05, z0 - 0.004), (w - 0.1, 0.02, 0.006),
          C.material("led", (1, 1, 1), 0.3, emission=(1.0, 0.92, 0.8, 6.0)), p)


def f_range_hood(f, p, mat, mats):
    w, d, h = f["size"]
    z0 = f.get("z0", 1.6)
    C.box(f"{f['name']}_canopy", (0, 0, z0 + 0.06), (w, d, 0.12), mats["steel"], p, bevel=0.004)
    C.box(f"{f['name']}_chimney", (0, d / 2 - 0.14, z0 + 0.12 + 0.5), (0.3, 0.26, 1.0), mats["steel"], p)


def f_island(f, p, mat, mats):
    w, d, h = f["size"]
    C.box(f"{f['name']}_plinth", (0, 0, TOE / 2), (w - 0.1, d - 0.1, TOE), mats["plinth"], p)
    C.box(f"{f['name']}_body", (0, 0, TOE + (h - TOP - TOE) / 2), (w - 0.06, d - 0.06, h - TOP - TOE), mats["wood"], p)
    _doors(dict(f, size=[w - 0.06, d, h]), p, mats, 3, TOE, h - TOP, -d / 2 + 0.04)
    C.box(f"{f['name']}_top", (0, 0, h - TOP / 2), (w + 0.04, d + 0.2, TOP), mats["worktop"], p, bevel=0.004)


def f_stool(f, p, mat, mats):
    w, d, h = f["size"]
    C.cylinder(f"{f['name']}_seat", (0, 0, h - 0.025), w / 2, 0.05, mats["wood"], p, verts=32)
    for i in range(4):
        a = math.pi / 4 + i * math.pi / 2
        C.cylinder(f"{f['name']}_leg{i}", (0.13 * math.cos(a), 0.13 * math.sin(a), (h - 0.05) / 2), 0.012, h - 0.05,
                   mats["plinth"], p, verts=12)
    C.cylinder(f"{f['name']}_ring", (0, 0, 0.22), 0.14, 0.012, mats["plinth"], p, verts=32)


KITCHEN_BUILDERS = dict(counter=f_counter, sink=f_sink, stove=f_stove, fridge=f_fridge, upper_cabinet=f_upper_cabinet,
                        range_hood=f_range_hood, island=f_island, stool=f_stool)


# ------------------------------------------------------------------ static props (world coordinates)
def p_fruit_bowl(pr, col, mats):
    x, y, z = pr["at"]
    r = pr.get("radius", 0.15)
    bowl = C.ellipsoid(pr["name"], (x, y, z + r * 0.42), (r, r, r * 0.42), mats["ceramic"], collection=col)
    bowl.modifiers.new("solid", "SOLIDIFY").thickness = 0.008
    _cut_top(bowl)
    return bowl


def _cut_top(obj):
    """Remove the upper half of a sphere-based mesh (bowls)."""
    import bmesh
    me = obj.data
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if v.co.z > 1e-4], context="VERTS")
    bm.to_mesh(me)
    bm.free()


def p_knife_block(pr, col, mats):
    x, y, z = pr["at"]
    blk = C.box(pr["name"], (x, y, z + 0.12), (0.11, 0.16, 0.24), mats["wood"], collection=col, bevel=0.008)
    blk.rotation_euler = (math.radians(-20), 0, math.radians(pr.get("rot", 0)))
    for i in range(4):
        C.cylinder(f"{pr['name']}_handle{i}", (-0.03 + 0.02 * i, 0.0, 0.16), 0.011, 0.1, mats["plinth"], blk, verts=12)
    return blk


def p_kettle(pr, col, mats):
    x, y, z = pr["at"]
    k = C.cylinder(pr["name"], (x, y, z + 0.11), 0.08, 0.2, mats["steel"], collection=col, verts=32)
    C.cylinder(f"{pr['name']}_base", (0, 0, -0.105), 0.085, 0.015, mats["plinth"], k, verts=32)
    C.box(f"{pr['name']}_grip", (0, 0.1, 0.03), (0.02, 0.03, 0.16), mats["plinth"], k)
    return k


def p_utensil_jar(pr, col, mats):
    x, y, z = pr["at"]
    j = C.cylinder(pr["name"], (x, y, z + 0.07), 0.05, 0.14, mats["ceramic"], collection=col, verts=24)
    for i, (dx, dy) in enumerate(((0.015, 0.0), (-0.012, 0.012), (0.0, -0.015))):
        u = C.cylinder(f"{pr['name']}_u{i}", (dx, dy, 0.12), 0.006, 0.24, mats["wood"], j, verts=8)
        u.rotation_euler = (dy * 8, -dx * 8, 0)
    return j


def p_plate_stack(pr, col, mats):
    x, y, z = pr["at"]
    for i in range(5):
        C.cylinder(f"{pr['name']}{i}", (x, y, z + 0.006 + i * 0.012), 0.12, 0.01, mats["ceramic"], collection=col, verts=40)


def p_herb_pot(pr, col, mats):
    x, y, z = pr["at"]
    pot = C.cylinder(pr["name"], (x, y, z + 0.06), 0.065, 0.12, mats["terracotta"], collection=col, verts=24)
    for i in range(7):
        a = i * 2.4
        C.ellipsoid(f"{pr['name']}_leaf{i}", (0.03 * math.cos(a), 0.03 * math.sin(a), 0.1 + 0.015 * (i % 3)),
                    (0.035, 0.02, 0.05), mats["leaf"], pot, segments=10)
    return pot


def p_pan(pr, col, mats):
    x, y, z = pr["at"]
    pan = C.cylinder(pr["name"], (x, y, z + 0.025), 0.13, 0.05, mats["black_glass"], collection=col, verts=40)
    C.box(f"{pr['name']}_handle", (0.0, -0.22, 0.01), (0.03, 0.2, 0.015), mats["plinth"], pan)
    pan.rotation_euler = (0, 0, math.radians(pr.get("rot", 90)))
    return pan


PROP_BUILDERS = dict(fruit_bowl=p_fruit_bowl, knife_block=p_knife_block, kettle=p_kettle, utensil_jar=p_utensil_jar,
                     plate_stack=p_plate_stack, herb_pot=p_herb_pot, pan=p_pan)


# ------------------------------------------------------------------ cameras
def build_cameras(spec, col):
    """Overview cameras for checking the build (the data rig lives in ego_rig.py)."""
    W, D, H = spec["width"], spec["depth"], spec["height"]
    cam = bpy.data.cameras.new("cam_overview")
    cam.lens = 14
    cam.clip_start = 0.05
    co = C.link(bpy.data.objects.new("cam_overview", cam), col)
    co.location = (0.35, 0.3, H - 0.3)
    C.look_at(co, (W * 0.62, D * 0.72, 0.9))
    top = bpy.data.cameras.new("cam_top")
    top.type = "ORTHO"
    top.ortho_scale = max(W, D) * 1.12
    to = C.link(bpy.data.objects.new("cam_top", top), col)
    to.location = (W / 2, D / 2, H + 4.0)
    bpy.context.scene.camera = co


# ------------------------------------------------------------------ main
def build_materials(spec):
    M = spec.get("materials", {})
    mats = dict(
        walls=C.material("walls", M.get("walls", {}).get("color", (0.9, 0.88, 0.84)), M.get("walls", {}).get("roughness", 0.9)),
        ceiling=C.material("ceiling", M.get("ceiling", {}).get("color", (0.94, 0.94, 0.93)), 0.95),
        trim=C.material("trim", (0.95, 0.95, 0.94), 0.4),
        door=C.material("door", (0.92, 0.91, 0.88), 0.45),
        glass=C.material("glass", (0.9, 0.95, 1.0), 0.02, transmission=1.0),
        metal=C.material("metal", (0.6, 0.6, 0.62), 0.3, metallic=1.0),
        wood_dark=C.material("wood_dark", (0.12, 0.08, 0.05), 0.5),
    )
    mats.update(kitchen_materials(spec))
    return mats


def build_interior(spec, mats, skip_walls=()):
    """Floor, ceiling, walls (minus `skip_walls`), openings, furniture and props, in the kitchen's own frame."""
    W, D, H = spec["width"], spec["depth"], spec["height"]
    BR.BUILDERS.update(KITCHEN_BUILDERS)
    shell = C.get_collection("room_shell")
    furn = C.get_collection("furniture")
    props = C.get_collection("props")
    C.box("floor", (W / 2, D / 2, -0.01), (W, D, 0.02), mats["floor"], collection=shell)
    # the ceiling overlaps into the walls, but not through a shared wall into the neighbouring room
    cx0, cx1 = (0.0 if "west" in skip_walls else -0.15), (W if "east" in skip_walls else W + 0.15)
    cy0, cy1 = (0.0 if "south" in skip_walls else -0.15), (D if "north" in skip_walls else D + 0.15)
    C.box("ceiling", ((cx0 + cx1) / 2, (cy0 + cy1) / 2, H + 0.01), (cx1 - cx0, cy1 - cy0, 0.02), mats["ceiling"], collection=shell)
    BR.build_walls(spec, shell, mats, skip=skip_walls)
    BR.build_openings(spec, shell, mats)
    # build_room's builders take (f, parent, mat, mats); kitchen ones ignore `mat` and use the shared mats
    for f in spec.get("furniture", []):
        parent = C.empty(f["name"], (f["center"][0], f["center"][1], 0.0), math.radians(f.get("rot", 0.0)), furn)
        BR.BUILDERS.get(f["type"], BR.f_generic)(f, parent, BR.furniture_material(f), mats)
        for ch in parent.children_recursive:
            if furn not in ch.users_collection:
                C._move_to(ch, furn)
    for pr in spec.get("props", []):
        before = set(bpy.data.objects)
        PROP_BUILDERS[pr["type"]](pr, props, mats)
        for o in set(bpy.data.objects) - before:
            if props not in o.users_collection:
                C._move_to(o, props)


def ceiling_light(spec, col, name="kitchen_light"):
    """Area light + emissive fixture in the middle of the ceiling (the annex's own; the house has the sun)."""
    W, D, H = spec["width"], spec["depth"], spec["height"]
    area = bpy.data.lights.new(name, "AREA")
    area.shape = "DISK"
    area.size = 0.5
    area.energy = spec.get("lights", {}).get("ceiling_light_power", 120.0)
    area.color = (1.0, 0.9, 0.78)
    ao = C.link(bpy.data.objects.new(name, area), col)
    ao.location = (W / 2, D / 2, H - 0.06)
    fixture = C.material("fixture", (1, 1, 1), 0.3, emission=(1.0, 0.93, 0.82, 4.0))
    C.cylinder(f"{name.removesuffix('_light')}_ceiling_fixture", (W / 2, D / 2, H - 0.025), 0.26, 0.05, fixture, collection=col)
    return ao


def build(spec):
    """The standalone kitchen (assets/kitchen.blend): interior + sun/sky + ceiling light + check cameras."""
    mats = build_materials(spec)
    build_interior(spec, mats)
    rig = C.get_collection("lights_cameras")
    BR.build_lighting(spec, rig)
    build_cameras(spec, rig)
    return mats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default=DEFAULT_SPEC)
    ap.add_argument("--out", default=C.ASSETS)
    a = ap.parse_args(C.script_args())
    spec = room_spec.load_spec(a.spec)
    os.makedirs(a.out, exist_ok=True)
    C.clear_scene()
    build(spec)
    lay = room_spec.layout(spec)
    lay.update(kind="kitchen", task=spec.get("task", {}))
    import json
    with open(os.path.join(a.out, "kitchen_layout.json"), "w") as f:
        json.dump(lay, f, indent=2)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(a.out, "kitchen.blend"))
    print(f"[build_kitchen] {len(bpy.data.objects)} objects -> {a.out}")


if __name__ == "__main__":
    main()
