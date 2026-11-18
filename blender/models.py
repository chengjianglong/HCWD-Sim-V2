"""Real-world 3D models (CC0, Poly Haven glTF) placed inside the JSON boxes, and instance scattering (grass, flowers).

A model never changes physics: it is fitted inside the furniture/object box the spec already gives
(centre, size, rot), which is exactly what the colliders use. Models face -y like all our furniture.

Many Poly Haven files hold several variants side by side (a row of shrubs, 17 grass clumps, a clean and a rusty
bin). Roots whose bounding boxes overlap are grouped into one "variant"; `place` picks one by index.
Imported files go to a hidden "model_library" collection once; placements are linked duplicates (shared meshes).
"""
import glob
import os

import bpy
from mathutils import Matrix, Vector

import common as C
import pbr

LIB = "model_library"
_variants = {}  # asset id -> list of variants; each variant = (list of root objects, bbox min, bbox max)


def model_path(aid):
    if not pbr.ENABLED:
        return None
    f = glob.glob(os.path.join(pbr.PH, "models", aid, f"*_{pbr.res()}.gltf"))
    return f[0] if f else None


def slot(name):
    return pbr.manifest()["models"].get(name)


def ids(name):
    s = slot(name)
    return [i for i in (s["ids"] if s else [name]) if model_path(i)]


def available(name):
    return bool(ids(name))


def _library():
    col = bpy.data.collections.get(LIB)
    if col is None:
        col = bpy.data.collections.new(LIB)
        bpy.context.scene.collection.children.link(col)
        col.hide_render = True
        col.hide_viewport = True
    return col


def _bbox(objs):
    pts = [o.matrix_world @ Vector(c) for r in objs for o in [r, *r.children_recursive] if o.type == "MESH"
           for c in o.bound_box]
    return (Vector([min(p[i] for p in pts) for i in range(3)]), Vector([max(p[i] for p in pts) for i in range(3)]))


def _overlap(a, b, m=0.02):
    return all(a[0][i] - m <= b[1][i] and b[0][i] - m <= a[1][i] for i in range(2))


def load(aid):
    """Import a model once; return its variants [(roots, bbox_min, bbox_max)], largest first."""
    if aid in _variants:
        return _variants[aid]
    lib = _library()
    new = [o for o in lib.objects if o.name.startswith(f"lib_{aid}_")]  # already in this .blend (a saved build)
    if not new:
        path = model_path(aid)
        if path is None:
            return []
        before = set(bpy.data.objects)
        bpy.ops.import_scene.gltf(filepath=path, import_pack_images=False)  # textures stay files in assets/polyhaven
        new = [o for o in bpy.data.objects if o not in before]
        for o in new:
            C._move_to(o, lib)
            o.name = f"lib_{aid}_{o.name}"
    roots = [o for o in new if o.parent is None and (o.type == "MESH" or any(c.type == "MESH" for c in o.children_recursive))]
    bpy.context.view_layer.update()
    groups = []  # union roots whose footprints overlap: parts of one object
    for r in roots:
        bb = _bbox([r])
        hit = [g for g in groups if _overlap(g[1], bb)]
        merged = [r] + [x for g in hit for x in g[0]]
        for g in hit:
            groups.remove(g)
        groups.append((merged, _bbox(merged)))
    var = sorted([(g, bb[0], bb[1]) for g, bb in groups], key=lambda v: -(v[2] - v[1]).length)
    _variants[aid] = var
    return var


def _dup(obj, parent, collection):
    d = obj.copy()  # linked duplicate: shares mesh + materials
    collection.objects.link(d)
    d.hide_render = d.hide_viewport = False
    d.parent = parent
    for ch in obj.children:
        _dup(ch, d, collection).matrix_parent_inverse = ch.matrix_parent_inverse.copy()
    return d


def place(name, size, parent, collection, fit=None, variant=0, choice=0, label=None):
    """Fit model `name` (a models slot from configs/assets.json, or a Poly Haven id) into the box `size` = (w, d, h)
    centred on `parent`'s origin, sitting on z = 0, front towards -y. Returns the model's root empty, or None.

    fit: "contain" (uniform, inside the box), "stretch" (fill the box), "height" (uniform, match the height only:
         trees, whose JSON box is the trunk). Defaults to the slot's fit. choice picks among the slot's ids,
         variant among the variants inside the file.
    """
    s = slot(name)
    aids = ids(name)
    if not aids:
        return None
    aid = aids[choice % len(aids)]
    var = load(aid)
    if not var:
        return None
    roots, lo, hi = var[variant % len(var)]
    fit = fit or (s["fit"] if s else "contain")
    ext = hi - lo
    w, d, h = size
    sx, sy, sz = w / max(ext.x, 1e-6), d / max(ext.y, 1e-6), h / max(ext.z, 1e-6)
    if fit == "contain":
        sx = sy = sz = min(sx, sy, sz)
    elif fit == "height":
        sx = sy = sz
    root = C.empty(label or f"{parent.name}_model", (0, 0, 0), 0.0, collection)
    root.parent = parent
    centre = Vector(((lo.x + hi.x) / 2, (lo.y + hi.y) / 2, lo.z))
    offset = Matrix.Diagonal((sx, sy, sz, 1.0)) @ Matrix.Translation(-centre)
    for r in roots:
        dup = _dup(r, root, collection)
        dup.matrix_parent_inverse = Matrix.Identity(4)
        dup.matrix_basis = offset @ r.matrix_world
    root["model"] = aid
    return root


def scatter(name, points, collection, label, rot=None, scale=None, seed=0):
    """Instance every variant of model `name` over `points` [(x, y, z)] with a geometry-nodes modifier.

    rot / scale: per-point lists (radians about z / uniform scale); random variant per point. Used for grass clumps
    and flowers: thousands of instances, one small mesh in the file.
    """
    aids = ids(name)
    if not aids or not points:
        return None
    src = bpy.data.collections.new(f"{label}_src")
    _library().children.link(src)
    n_var = 0
    for aid in aids:
        for roots, lo, hi in load(aid):  # one child collection per variant = one instance choice
            centre = Vector(((lo.x + hi.x) / 2, (lo.y + hi.y) / 2, lo.z))
            sub = bpy.data.collections.new(f"{label}_v{n_var}")
            src.children.link(sub)
            n_var += 1
            for r in roots:
                dup = _dup(r, None, sub)
                dup.matrix_basis = Matrix.Translation(-centre) @ r.matrix_world
    me = bpy.data.meshes.new(label)
    me.from_pydata([tuple(p) for p in points], [], [])
    if rot is not None:
        a = me.attributes.new("rot_z", "FLOAT", "POINT")
        a.data.foreach_set("value", list(map(float, rot)))
    if scale is not None:
        a = me.attributes.new("scale", "FLOAT", "POINT")
        a.data.foreach_set("value", list(map(float, scale)))
    obj = C.link(bpy.data.objects.new(label, me), collection)
    ng = bpy.data.node_groups.new(f"{label}_scatter", "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    N = ng.nodes
    gi, go = N.new("NodeGroupInput"), N.new("NodeGroupOutput")
    ci = N.new("GeometryNodeCollectionInfo")
    ci.inputs["Collection"].default_value = src
    ci.inputs["Separate Children"].default_value = True
    ci.inputs["Reset Children"].default_value = True
    ri = N.new("FunctionNodeRandomValue")
    ri.data_type = "INT"
    ri.inputs["Min"].default_value = 0
    ri.inputs["Max"].default_value = n_var - 1
    ri.inputs["Seed"].default_value = seed
    iop = N.new("GeometryNodeInstanceOnPoints")
    iop.inputs["Pick Instance"].default_value = True
    L = ng.links
    L.new(gi.outputs["Geometry"], iop.inputs["Points"])
    L.new(ci.outputs["Instances"], iop.inputs["Instance"])
    L.new(ri.outputs[2], iop.inputs["Instance Index"])  # the INT output
    if rot is not None:
        ra = N.new("GeometryNodeInputNamedAttribute")
        ra.data_type = "FLOAT"
        ra.inputs["Name"].default_value = "rot_z"
        cv = N.new("ShaderNodeCombineXYZ")
        L.new(ra.outputs["Attribute"], cv.inputs["Z"])
        L.new(cv.outputs["Vector"], iop.inputs["Rotation"])
    if scale is not None:
        sa = N.new("GeometryNodeInputNamedAttribute")
        sa.data_type = "FLOAT"
        sa.inputs["Name"].default_value = "scale"
        L.new(sa.outputs["Attribute"], iop.inputs["Scale"])
    L.new(iop.outputs["Instances"], go.inputs["Geometry"])
    mod = obj.modifiers.new("scatter", "NODES")
    mod.node_group = ng
    return obj
