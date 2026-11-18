"""Raw egocentric episodes -> the vla-qwen training format (same files as vla-qwen's prepare_data.py).

    .venv/bin/python scripts/ego_to_vlaqwen.py --raw data/ego_raw --out data/ego_vlaqwen \\
        [--views head right_wrist] [--res 224] [--ref head|world] [--hand right]

Writes:
    images.u8    uint8 memmap [N, 2, R, R, 3]   view 0 takes the place of LIBERO's agentview, view 1 the wrist
    arrays.npz   state [N, 8], actions [N, 7], episode_index / frame_index / task_index [N] (int64)
    meta.json    res, total, tasks {int: str}, episode_starts, episode_lengths, stats {state, actions}

Then, in vla-qwen:  python train.py --data <out>
State/action layout matches LIBERO's (eef pos, axis-angle, 2 finger widths / delta pose + gripper +-1); see
worldsim/ego.py. Normalisation stats (q01/q99) are computed from this data, as prepare_data.py does.
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from worldsim import ego, sensor  # noqa: E402


def load(path, res, cam=None):
    im = Image.open(path).convert("RGB")
    if cam is not None:
        im = Image.fromarray(cam(np.asarray(im)))
    if im.size != (res, res):
        im = im.resize((res, res), Image.BICUBIC)
    return np.asarray(im, dtype=np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/ego_raw")
    ap.add_argument("--out", default="data/ego_vlaqwen")
    ap.add_argument("--views", nargs=2, default=["head", "right_wrist"], metavar=("SCENE", "WRIST"))
    ap.add_argument("--res", type=int, default=224)
    ap.add_argument("--ref", choices=["head", "world"], default="head")
    ap.add_argument("--hand", choices=["right", "left"], default="right")
    ap.add_argument("--sensor", choices=["realistic", "none"], default="realistic",
                    help="simulated camera per episode and view (worldsim/sensor.py), applied before resizing")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    eps = ego.load_raw(a.raw)
    if not eps:
        sys.exit(f"no episodes in {a.raw}")
    for d, m in eps:
        missing = [v for v in a.views if v not in m["views"]]
        if missing:
            sys.exit(f"{d} has no {missing} images")
    os.makedirs(a.out, exist_ok=True)
    lengths = [m["n_frames"] for _, m in eps]
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(int).tolist()
    total = int(sum(lengths))
    img_path = os.path.join(a.out, "images.u8")
    mm = np.memmap(img_path, dtype=np.uint8, mode="w+", shape=(total, 2, a.res, a.res, 3))
    state = np.zeros((total, 8), np.float32)
    actions = np.zeros((total, 7), np.float32)
    ep_i, fr_i, tk_i = (np.zeros(total, np.int64) for _ in range(3))
    tasks, sensors = {}, []
    for ei, ((d, m), st, n) in enumerate(zip(eps, starts, lengths)):
        ti = tasks.setdefault(m["instruction"], len(tasks))
        s, act = ego.hand_state_actions(m, a.hand, a.ref)
        state[st:st + n], actions[st:st + n] = s, act
        ep_i[st:st + n], fr_i[st:st + n], tk_i[st:st + n] = ei, np.arange(n), ti
        cams = [sensor.CameraSensor(v, m["intrinsics"][v], sensor.seed_for(a.seed, os.path.basename(d), v), a.sensor)
                if a.sensor == "realistic" else None for v in a.views]
        sensors.append({v: dict(c.describe(), intrinsics=c.intrinsics()) for v, c in zip(a.views, cams) if c})
        for i in range(n):
            for k, v in enumerate(a.views):
                mm[st + i, k] = load(os.path.join(d, v, f"{i:06d}.png"), a.res, cams[k])
        print(f"[vla-qwen] {ei + 1}/{len(eps)} {os.path.basename(d)}: {n} frames", flush=True)
    mm.flush()
    del mm
    np.savez(os.path.join(a.out, "arrays.npz"), state=state, actions=actions,
             episode_index=ep_i, frame_index=fr_i, task_index=tk_i)

    def stats(x):
        s = ego.stats(x)
        return {k: s[k] for k in ("mean", "std", "q01", "q99")}

    meta = {
        "res": a.res, "total": total,
        "tasks": {int(i): t for t, i in tasks.items()},
        "episode_starts": starts, "episode_lengths": lengths,
        "stats": {"state": stats(state), "actions": stats(actions)},
        "source": {"generator": "human-centric-world-simulation ego kitchen", "views": a.views, "ref": a.ref, "hand": a.hand,
                   "sensor": a.sensor},
        "sensors": sensors,  # per episode: the simulated camera of each view (native-resolution intrinsics)
    }
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"[vla-qwen] {len(eps)} episodes, {total} frames, {len(tasks)} tasks, views {a.views} -> {a.out}")


if __name__ == "__main__":
    main()
