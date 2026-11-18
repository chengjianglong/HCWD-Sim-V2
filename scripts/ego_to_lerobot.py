"""Raw egocentric episodes -> a LeRobot v2.0 dataset (parquet + one mp4 per camera per episode).

    .venv/bin/python scripts/ego_to_lerobot.py --raw data/ego_raw --out data/ego_lerobot [--ref head|world]
        [--views head chest left_wrist right_wrist scene_front scene_left scene_right] [--hand right]

Layout (LeRobot v2.0):
    meta/info.json  meta/episodes.jsonl  meta/tasks.jsonl  meta/stats.json  meta/cameras.json
    data/chunk-000/episode_000000.parquet
    videos/chunk-000/observation.images.<view>/episode_000000.mp4

Parquet columns:
    observation.state [8], action [7]            main hand (default right), see worldsim/ego.py
    observation.state.both_hands [20], action.both_hands [14]
    observation.extrinsics.<view> [16]           cam-to-world 4x4 (OpenCV axes), row-major
    phase_index, cuts_done, timestamp, frame_index, episode_index, index, task_index, next.done
Intrinsics per view are in meta/cameras.json (and the phase names in meta/info.json).
"""
import argparse
import json
import os
import shutil
import sys

import imageio_ffmpeg
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from worldsim import ego  # noqa: E402

CHUNK = 1000
PHASES = ["idle", "reach", "lift", "chop", "return", "release", "retract", "idle_end"]


def phase_index(name):
    return PHASES.index("chop" if name.startswith("chop") else name)


def cuts_done(phases, n_cuts):
    """Cuts finished by each frame: K while chopping cut K (0-based), all of them once chopping is over."""
    out = []
    for p in phases:
        if p.startswith("chop"):
            out.append(int(p[4:]))
        elif p in ("return", "release", "retract", "idle_end"):
            out.append(n_cuts)
        else:
            out.append(0)
    return out


def write_video(frames_dir, n, path, fps, crf=18):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    first = Image.open(os.path.join(frames_dir, f"{0:06d}.png"))
    w, h = first.size
    wr = imageio_ffmpeg.write_frames(path, (w, h), fps=fps, codec="libx264", pix_fmt_out="yuv420p",
                                     output_params=["-crf", str(crf), "-g", "2"], macro_block_size=1)
    wr.send(None)
    for i in range(n):
        wr.send(np.ascontiguousarray(np.asarray(Image.open(os.path.join(frames_dir, f"{i:06d}.png")).convert("RGB"))))
    wr.close()
    return w, h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/ego_raw")
    ap.add_argument("--out", default="data/ego_lerobot")
    ap.add_argument("--ref", choices=["head", "world"], default="head", help="frame the hand poses are expressed in")
    ap.add_argument("--hand", choices=["right", "left"], default="right", help="hand for observation.state / action")
    ap.add_argument("--views", nargs="*", help="cameras to export (default: all in the raw data)")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    eps = ego.load_raw(a.raw)
    if not eps:
        sys.exit(f"no episodes in {a.raw}")
    if os.path.exists(a.out):
        if not a.overwrite:
            sys.exit(f"{a.out} exists (use --overwrite)")
        shutil.rmtree(a.out)
    views = a.views or eps[0][1]["views"]
    fps = eps[0][1]["fps"]
    tasks, episodes, all_state, all_act, all_sb, all_ab = {}, [], [], [], [], []
    cameras = {}
    index = 0
    for ei, (d, m) in enumerate(eps):
        n = m["n_frames"]
        task = m["instruction"]
        ti = tasks.setdefault(task, len(tasks))
        state, act = ego.hand_state_actions(m, a.hand, a.ref)
        sb, ab = ego.both_hands(m, a.ref)
        phases = [f["phase"] for f in m["frames"]]
        cols = {
            "observation.state": list(state), "action": list(act),
            "observation.state.both_hands": list(sb), "action.both_hands": list(ab),
            "phase_index": np.array([phase_index(p) for p in phases], np.int64),
            "cuts_done": np.array(cuts_done(phases, m["task"]["n_cuts"]), np.int64),
            "timestamp": np.arange(n, dtype=np.float32) / fps,
            "frame_index": np.arange(n, dtype=np.int64),
            "episode_index": np.full(n, ei, np.int64),
            "index": np.arange(index, index + n, dtype=np.int64),
            "task_index": np.full(n, ti, np.int64),
            "next.done": np.arange(n) == n - 1,
        }
        for v in views:
            cols[f"observation.extrinsics.{v}"] = [np.asarray(f["cameras"][v], np.float32).reshape(-1) for f in m["frames"]]
        table = pa.table({k: (pa.array([np.asarray(x, np.float32) for x in c], type=pa.list_(pa.float32()))
                              if isinstance(c, list) else pa.array(c)) for k, c in cols.items()})
        chunk = ei // CHUNK
        p = os.path.join(a.out, "data", f"chunk-{chunk:03d}", f"episode_{ei:06d}.parquet")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        pq.write_table(table, p)
        for v in views:
            w, h = write_video(os.path.join(d, v), n, os.path.join(
                a.out, "videos", f"chunk-{chunk:03d}", f"observation.images.{v}", f"episode_{ei:06d}.mp4"), fps)
            cameras.setdefault(v, m["intrinsics"][v])
        episodes.append({"episode_index": ei, "tasks": [task], "length": n, "source": os.path.basename(d),
                         "task_info": m["task"]})
        all_state.append(state), all_act.append(act), all_sb.append(sb), all_ab.append(ab)
        index += n
        print(f"[lerobot] {ei + 1}/{len(eps)} {os.path.basename(d)}: {n} frames, '{task}'", flush=True)

    res = cameras[views[0]]["width"], cameras[views[0]]["height"]
    vec = lambda n, names=None: {"dtype": "float32", "shape": [n], "names": names}  # noqa: E731
    features = {
        "observation.state": vec(8, ["x", "y", "z", "rx", "ry", "rz", "finger_l", "finger_r"]),
        "action": vec(7, ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"]),
        "observation.state.both_hands": vec(20),
        "action.both_hands": vec(14),
        "phase_index": {"dtype": "int64", "shape": [1], "names": None},
        "cuts_done": {"dtype": "int64", "shape": [1], "names": None},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
        "next.done": {"dtype": "bool", "shape": [1], "names": None},
    }
    for v in views:
        features[f"observation.images.{v}"] = {
            "dtype": "video", "shape": [res[1], res[0], 3], "names": ["height", "width", "channels"],
            "info": {"video.fps": fps, "video.height": res[1], "video.width": res[0], "video.channels": 3,
                     "video.codec": "h264", "video.pix_fmt": "yuv420p", "video.is_depth_map": False, "has_audio": False}}
        features[f"observation.extrinsics.{v}"] = vec(16)
    total = int(sum(e["length"] for e in episodes))
    info = {
        "codebase_version": "v2.0", "robot_type": "human_egocentric", "fps": fps,
        "total_episodes": len(episodes), "total_frames": total, "total_tasks": len(tasks),
        "total_videos": len(episodes) * len(views), "total_chunks": (len(episodes) - 1) // CHUNK + 1,
        "chunks_size": CHUNK, "splits": {"train": f"0:{len(episodes)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
        "ego": {"reference_frame": a.ref, "main_hand": a.hand, "phases": PHASES,
                "state": "hand pos, axis-angle, finger widths (0.04 open..0 closed) - see worldsim/ego.py",
                "action": "delta pos, delta axis-angle (t->t+1, in the reference frame at t), gripper -1 open/+1 closed"},
    }
    meta = os.path.join(a.out, "meta")
    os.makedirs(meta, exist_ok=True)
    with open(os.path.join(meta, "info.json"), "w") as f:
        json.dump(info, f, indent=2)
    with open(os.path.join(meta, "tasks.jsonl"), "w") as f:
        f.writelines(json.dumps({"task_index": i, "task": t}) + "\n" for t, i in sorted(tasks.items(), key=lambda kv: kv[1]))
    with open(os.path.join(meta, "episodes.jsonl"), "w") as f:
        f.writelines(json.dumps(e) + "\n" for e in episodes)
    st = {"observation.state": ego.stats(np.concatenate(all_state)), "action": ego.stats(np.concatenate(all_act)),
          "observation.state.both_hands": ego.stats(np.concatenate(all_sb)),
          "action.both_hands": ego.stats(np.concatenate(all_ab))}
    with open(os.path.join(meta, "stats.json"), "w") as f:
        json.dump(st, f, indent=1)
    with open(os.path.join(meta, "cameras.json"), "w") as f:
        json.dump(cameras, f, indent=1)
    print(f"[lerobot] {len(episodes)} episodes, {total} frames, {len(tasks)} tasks, {len(views)} views -> {a.out}")


if __name__ == "__main__":
    main()
