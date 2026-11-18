#!/usr/bin/env python3
"""Download the photo-realistic assets listed in configs/assets.json (all CC0, from Poly Haven) into assets/polyhaven.

    python scripts/fetch_assets.py                    # everything (~1.5 GB at 2k)
    python scripts/fetch_assets.py --only textures hdris
    python scripts/fetch_assets.py --list             # what would be fetched, and what is already here

Layout (read by blender/pbr.py and blender/models.py):
    assets/polyhaven/textures/<id>/<id>_{diff,nor_gl,rough,ao}_2k.jpg  + info.json (real-world size in mm)
    assets/polyhaven/hdris/<id>_2k.hdr
    assets/polyhaven/models/<id>/<id>_2k.gltf  + .bin + textures/... + info.json
Files that already exist are skipped, so the script can be re-run to resume. Source: https://polyhaven.com (CC0).
"""
import argparse
import json
import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(ROOT, "assets", "polyhaven")
API = "https://api.polyhaven.com"
TEX_MAPS = {"Diffuse": "diff", "nor_gl": "nor_gl", "Rough": "rough", "AO": "ao"}


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "worldsim-asset-fetch"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return r.read()


def _json(url):
    return json.loads(_get(url))


def download(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return 0
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    data = _get(url)
    tmp = dest + ".part"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, dest)
    return len(data)


def manifest():
    with open(os.path.join(ROOT, "configs", "assets.json")) as f:
        return json.load(f)


def wanted(m, only):
    out = []
    if "textures" in only:
        out += [("textures", i) for i in sorted({i for ids in m["textures"].values() for i in ids})]
    if "hdris" in only:
        out += [("hdris", i) for i in sorted({i for ids in m["hdris"].values() for i in ids})]
    if "models" in only:
        out += [("models", i) for i in sorted({i for s in m["models"].values() for i in s["ids"]})]
    return out


def have(kind, aid, res):
    if kind == "hdris":
        return os.path.exists(os.path.join(DEST, "hdris", f"{aid}_{res}.hdr"))
    if kind == "models":
        return os.path.exists(os.path.join(DEST, kind, aid, f"{aid}_{res}.gltf"))
    return os.path.exists(os.path.join(DEST, kind, aid, "info.json"))


def fetch(kind, aid, res):
    files = _json(f"{API}/files/{aid}")
    n = 0
    if kind == "hdris":
        return download(files["hdri"][res]["hdr"]["url"], os.path.join(DEST, "hdris", f"{aid}_{res}.hdr"))
    d = os.path.join(DEST, kind, aid)
    if kind == "textures":
        for key, short in TEX_MAPS.items():
            if key in files:
                n += download(files[key][res]["jpg"]["url"], os.path.join(d, f"{aid}_{short}_{res}.jpg"))
    else:  # models: glTF (+ .bin and textures, relative paths); Poly Haven's .blend files may need a newer Blender
        b = files["gltf"][res]["gltf"]
        n += download(b["url"], os.path.join(d, os.path.basename(b["url"])))
        for rel, inc in b.get("include", {}).items():
            n += download(inc["url"], os.path.join(d, rel))
    info = _json(f"{API}/info/{aid}")
    with open(os.path.join(d, "info.json"), "w") as f:
        json.dump({k: info.get(k) for k in ("name", "dimensions", "categories", "authors")} | {"license": "CC0"}, f)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=["textures", "hdris", "models"], choices=["textures", "hdris", "models"])
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    m = manifest()
    res = m.get("resolution", "2k")
    todo = wanted(m, a.only)
    if a.list:
        for kind, aid in todo:
            print(f"{'ok ' if have(kind, aid, res) else '-- '}{kind:9s} {aid}")
        return
    os.makedirs(DEST, exist_ok=True)
    with open(os.path.join(DEST, "LICENSE.txt"), "w") as f:
        f.write("All assets in this folder are from Poly Haven (https://polyhaven.com) and are CC0 (public domain).\n")
    total, failed = 0, []
    for k, (kind, aid) in enumerate(todo):
        if have(kind, aid, res):
            continue
        try:
            n = fetch(kind, aid, res)
            total += n
            print(f"[fetch] {k + 1}/{len(todo)} {kind} {aid} ({n / 1e6:.1f} MB)", flush=True)
        except Exception as e:  # keep going; the builders fall back for anything missing
            failed.append(aid)
            print(f"[fetch] {kind} {aid} FAILED: {e}", file=sys.stderr, flush=True)
    print(f"[fetch] done: {total / 1e6:.0f} MB downloaded into {DEST}" + (f"; failed: {failed}" if failed else ""))


if __name__ == "__main__":
    main()
