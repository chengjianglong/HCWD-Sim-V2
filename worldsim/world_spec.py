"""Pure-Python description of the house + garden + street world (configs/world.json).

The house interior comes from the referenced room.json, plus optional `annexes`: further rooms (e.g. the kitchen,
configs/kitchen.json) built flush against one of its walls and joined to it by a door. The shared wall is drawn and
collided once, by the main room, with the annex's door cut into it. Like room_spec, this is shared
by Blender (visuals) and the torch env (physics), so the colliders match what is drawn.
"""
import copy
import json
import math
import os

from . import room_spec


def is_world(path):
    with open(path) as f:
        return "room" in json.load(f)


OPPOSITE = {"west": "east", "east": "west", "south": "north", "north": "south"}


def load_world(path):
    with open(path) as f:
        w = json.load(f)
    here = os.path.dirname(os.path.abspath(path))
    w["_room"] = room_spec.load_spec(os.path.join(here, w["room"]))
    w["_annexes"] = [join_annex(w["_room"], room_spec.load_spec(os.path.join(here, a["spec"])), a)
                     for a in w.get("annexes", [])]
    validate(w)
    return w


def join_annex(room, spec, a):
    """Attach annex `spec` to `room`: `room` is edited in place, the annex is returned as a modified copy.

    a = {spec, origin: [x, y] of the annex's interior south-west corner in room coordinates,
         shared_wall: the annex wall that touches the room, door: name of the annex door in that wall}.
    The door moves onto the room's facing wall (same position, opening swung into the annex); room windows on the
    covered stretch are dropped; the annex keeps its own walls except the shared one."""
    wall, ox, oy = a["shared_wall"], *a["origin"]
    rw = OPPOSITE[wall]
    W, D, t = room["width"], room["depth"], room["wall_thickness"]
    Wk, Dk = spec["width"], spec["depth"]
    if abs(spec["wall_thickness"] - t) > 1e-6:
        raise ValueError(f"annex {spec['name']}: wall_thickness must match the room's ({t})")
    flush = {"west": abs(ox - (W + t)), "east": abs(ox + Wk + t), "south": abs(oy - (D + t)), "north": abs(oy + Dk + t)}[wall]
    if flush > 1e-6:
        raise ValueError(f"annex {spec['name']}: origin {a['origin']} is not flush against the room's {rw} wall")
    along_y = wall in ("west", "east")
    shift = oy if along_y else ox
    lo, hi = shift - t, shift + (Dk if along_y else Wk) + t           # stretch of the room wall the annex covers
    door = next((o for o in spec.get("openings", []) if o["name"] == a["door"]), None)
    if door is None or door["wall"] != wall:
        raise ValueError(f"annex {spec['name']}: no door {a['door']!r} on its {wall} wall")
    spec = copy.deepcopy(spec)
    spec["openings"] = [o for o in spec["openings"] if o["name"] != a["door"]]
    spec["_skip_walls"] = [wall]
    rdoor = dict(door, name=f"{spec['name']}_door", wall=rw, offset=door["offset"] + shift)
    if door.get("open"):
        rdoor["open"] = {"in": "out", "out": "in"}[door["open"]]  # same leaf, seen from the room's side
    room["openings"] = [o for o in room.get("openings", [])
                        if not (o["wall"] == rw and o["offset"] < hi and o["offset"] + o["width"] > lo)] + [rdoor]
    room_spec.validate(room)
    return dict(spec=spec, origin=(float(ox), float(oy)), shared_wall=wall, door=rdoor["name"],
                rect=[ox, oy, ox + Wk, oy + Dk])


def validate(w):
    x0, y0, x1, y1 = w["bounds"]
    r = w["_room"]
    if not (x0 < -r["wall_thickness"] and y0 < -r["wall_thickness"]
            and x1 > r["width"] + r["wall_thickness"] and y1 > r["depth"] + r["wall_thickness"]):
        raise ValueError("world bounds must enclose the house")
    for an in w.get("_annexes", []):
        t = an["spec"]["wall_thickness"]
        ax0, ay0, ax1, ay1 = an["rect"][0] - t, an["rect"][1] - t, an["rect"][2] + t, an["rect"][3] + t
        if not (x0 < ax0 and y0 < ay0 and ax1 < x1 and ay1 < y1):
            raise ValueError(f"world bounds must enclose annex {an['spec']['name']}")
        inside = lambda p: ax0 + 1e-3 < p[0] < ax1 - 1e-3 and ay0 + 1e-3 < p[1] < ay1 - 1e-3  # noqa: E731
        for f in w.get("objects", []):
            cs = room_spec.obb_corners(f)
            mid = (sum(c[0] for c in cs) / 4, sum(c[1] for c in cs) / 4)
            if any(inside(c) for c in cs + [mid]):
                raise ValueError(f"object {f['name']} overlaps annex {an['spec']['name']}")
        for fe in w.get("fences", []):
            (fx0, fy0), (fx1, fy1) = fe["from"], fe["to"]
            if any(inside((fx0 + (fx1 - fx0) * k / 50, fy0 + (fy1 - fy0) * k / 50)) for k in range(51)):
                raise ValueError(f"fence {fe['name']} runs through annex {an['spec']['name']}")
    for f in w.get("objects", []):
        for c in room_spec.obb_corners(f):
            if not (x0 - 1e-3 <= c[0] <= x1 + 1e-3 and y0 - 1e-3 <= c[1] <= y1 + 1e-3):
                raise ValueError(f"object {f['name']} pokes outside the world bounds: corner {c}")
    for fe in w.get("fences", []):
        length = math.dist(fe["from"], fe["to"])
        for g in fe.get("gates", []):
            if g["offset"] < 0 or g["offset"] + g["width"] > length + 1e-6:
                raise ValueError(f"gate {g['name']} does not fit on fence {fe['name']}")


FENCE_T = 0.08


def fence_pieces(fe):
    """Solid runs of a fence (gates cut out): list of (name, (ax, ay), (bx, by))."""
    (ax, ay), (bx, by) = fe["from"], fe["to"]
    length = math.dist(fe["from"], fe["to"])
    ux, uy = (bx - ax) / length, (by - ay) / length
    cuts = sorted((g["offset"], g["offset"] + g["width"]) for g in fe.get("gates", []))
    out, u = [], 0.0
    for i, (a, b) in enumerate(cuts + [(length, length)]):
        if a > u + 1e-6:
            out.append((f"{fe['name']}_{i}", (ax + ux * u, ay + uy * u), (ax + ux * a, ay + uy * a)))
        u = b
    return out


def fence_colliders(w):
    out = []
    for fe in w.get("fences", []):
        for name, (ax, ay), (bx, by) in fence_pieces(fe):
            length = math.dist((ax, ay), (bx, by))
            out.append(dict(name=name, cx=(ax + bx) / 2, cy=(ay + by) / 2, hx=length / 2, hy=FENCE_T / 2,
                            theta=math.atan2(by - ay, bx - ax), height=fe.get("height", 1.8)))
    return out


def gates(w):
    """Gate openings: dict(name, x, y) centre points (also used as route landmarks / POIs)."""
    out = []
    for fe in w.get("fences", []):
        (ax, ay), (bx, by) = fe["from"], fe["to"]
        length = math.dist(fe["from"], fe["to"])
        ux, uy = (bx - ax) / length, (by - ay) / length
        for g in fe.get("gates", []):
            m = g["offset"] + g["width"] / 2
            out.append(dict(name=g["name"], x=ax + ux * m, y=ay + uy * m, width=g["width"],
                            theta=math.atan2(uy, ux), fence=fe["name"]))
    return out


def object_colliders(w):
    out = []
    for f in w.get("objects", []):
        if not f.get("collide", True):
            continue
        out.append(dict(name=f["name"], cx=f["center"][0], cy=f["center"][1], hx=f["size"][0] / 2, hy=f["size"][1] / 2,
                        theta=math.radians(f.get("rot", 0.0)), height=f["size"][2]))
    return out


def annex_colliders(an):
    """An annex's walls (minus the shared one), furniture and door leaves, moved into room coordinates."""
    spec, (ox, oy) = an["spec"], an["origin"]
    cols = room_spec.wall_colliders(spec, skip=spec["_skip_walls"]) + room_spec.colliders(spec)
    return [dict(c, name=f"{spec['name']}_{c['name']}", cx=c["cx"] + ox, cy=c["cy"] + oy) for c in cols]


def annex_pois(an):
    spec, (ox, oy) = an["spec"], an["origin"]
    return [dict(p, name=f"{spec['name']}_{p['name']}", x=p["x"] + ox, y=p["y"] + oy) for p in room_spec.pois(spec)]


def colliders(w):
    room = w["_room"]
    return (room_spec.wall_colliders(room) + room_spec.colliders(room)
            + [c for an in w.get("_annexes", []) for c in annex_colliders(an)] + fence_colliders(w) + object_colliders(w))


def pois(w):
    out = list(room_spec.pois(w["_room"]))
    out.extend(p for an in w.get("_annexes", []) for p in annex_pois(an))
    for f in w.get("objects", []):
        if "poi" in f:
            x, y = room_spec.local_to_world(f, *f["poi"])
            out.append(dict(name=f["name"], x=x, y=y))
    out.extend(w.get("extra_pois", []))
    return out


def layout(w):
    room = w["_room"]
    x0, y0, x1, y1 = w["bounds"]
    return dict(kind="world", name=w["name"], width=x1 - x0, depth=y1 - y0, height=room["height"], bounds=w["bounds"],
                colliders=colliders(w), pois=pois(w), openings=room.get("openings", []), gates=gates(w),
                indoor=[[0.0, 0.0, room["width"], room["depth"]]] + [an["rect"] for an in w.get("_annexes", [])],
                annexes=[dict(name=an["spec"]["name"], origin=list(an["origin"]), rect=an["rect"], door=an["door"],
                              height=an["spec"]["height"], openings=an["spec"].get("openings", []))
                         for an in w.get("_annexes", [])],
                road=w.get("road", []),
                crosswalks=w.get("crosswalks", []), sidewalks=w.get("sidewalks", []),
                nav=w.get("nav", {}), traffic=w.get("traffic"))


def write_layout(w, path):
    with open(path, "w") as f:
        json.dump(layout(w), f, indent=2)
