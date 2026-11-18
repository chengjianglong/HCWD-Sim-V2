#!/usr/bin/env bash
# Re-create the README demo media in docs/media from trained checkpoints.
#   scripts/make_demo_media.sh [world_ckpt] [room_ckpt]
# Needs: tools/blender, Rocketbox avatars, `pip install imageio-ffmpeg matplotlib` in .venv.
set -euo pipefail
cd "$(dirname "$0")/.."
WORLD_CKPT=${1:-runs/world/best.pt}
ROOM_CKPT=${2:-runs/main/best.pt}
PY=.venv/bin/python
BL=tools/blender/blender
FF=$($PY -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())")
OUT=docs/media
TMP=renders/demo
mkdir -p "$OUT" "$TMP"

gif() {  # gif <video> <out.gif> <start s> <length s> <width>
  "$FF" -v error -y -ss "$3" -t "$4" -i "$1" -vf \
    "fps=12,scale=$5:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=160:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle" "$2"
}
mp4() {  # mp4 <video> <out.mp4>: small web-friendly copy
  "$FF" -v error -y -i "$1" -c:v libx264 -crf 23 -preset slow -pix_fmt yuv420p -movflags +faststart -an "$2"
}

# --- world: one rollout, six cameras --------------------------------------------------------
# Seed 1 sends three people into the kitchen in its second half, so the house and kitchen cameras render that half
# (from step 180). Clips are rendered only as long as the README and the video use them.
$PY evaluate.py --ckpt "$WORLD_CKPT" --spec configs/world.json --steps 500 --seed 1 --out $TMP/world.npz
$PY -c "
import numpy as np, sys
d = dict(np.load('$TMP/world.npz', allow_pickle=True)); T = d['pos'].shape[0]
np.savez_compressed('$TMP/world_late.npz', **{k: v[180:] if getattr(v, 'ndim', 0) and v.shape[0] == T else v for k, v in d.items()})"
for spec in "house world_late 400" "kitchen world_late 400" "cutaway world 260" "street world 200" "garden world 150" "aerial world 150"; do
  set -- $spec
  $BL -b -P blender/render_rollout.py -- --rollout $TMP/$2.npz --camera $1 --engine eevee --samples 8 --res 960x540 \
    --frames $3 --out $TMP/$1.mp4
done
for cam in cutaway house street aerial garden; do mp4 $TMP/$cam.mp4 $OUT/world_$cam.mp4; done
mp4 $TMP/kitchen.mp4 $OUT/world_kitchen_inside.mp4
$BL -b -P blender/render_rollout.py -- --rollout $TMP/world.npz --camera top --engine eevee --res 1280x720 --still 420 --out $TMP/top.mp4
"$FF" -v error -y -i $TMP/top_f420.png -q:v 3 $OUT/world_top.jpg
"$FF" -v error -y -ss 12 -i $TMP/house.mp4 -frames:v 1 -q:v 3 $OUT/world_kitchen.jpg
gif $TMP/cutaway.mp4 $OUT/world_cutaway.gif 1 10 720
gif $TMP/house.mp4   $OUT/world_house.gif 5 8 480
gif $TMP/kitchen.mp4 $OUT/world_kitchen_inside.gif 8 8 480
gif $TMP/street.mp4  $OUT/world_street.gif 1 8 480
gif $TMP/garden.mp4  $OUT/world_garden.gif 0.3 7 480
gif $TMP/aerial.mp4  $OUT/world_aerial.gif 0.3 7 480

# --- room: Rocketbox avatars, Cycles ----------------------------------------------------------
$PY evaluate.py --ckpt "$ROOM_CKPT" --steps 300 --out $TMP/room.npz
$BL -b -P blender/render_rollout.py -- --rollout $TMP/room.npz --camera corner --engine cycles --out $TMP/room.mp4
mp4 $TMP/room.mp4 $OUT/room_corner.mp4
gif $TMP/room.mp4 $OUT/room.gif 2 8 480

# --- training curves --------------------------------------------------------------------------
$PY scripts/plot_training.py --log "runs/main.log:Room mode" --log "runs/world.log:World mode" --out $OUT/training.png

# --- egocentric episodes for the video (house scene, two of them) -----------------------------
for ep in 3 12; do
  [ -f $TMP/ego/episode_$(printf %06d $ep)/episode.json ] || $BL -b -P blender/gen_ego_episodes.py -- --scene house \
    --out $TMP/ego --start $ep --episodes 1 --res 384
done

# --- captioned demo video (uses the renders above and training.png) ---------------------------
$PY scripts/make_demo_video.py --demo $TMP --room $TMP/room.mp4 --ego $TMP/ego/episode_000003 $TMP/ego/episode_000012 \
  --out $OUT/demo.mp4
$PY -c "
from PIL import Image, ImageDraw
import subprocess, imageio_ffmpeg
subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-y', '-ss', '29', '-i', '$OUT/demo.mp4', '-frames:v', '1', '$TMP/thumb.png'], check=True)
im = Image.open('$TMP/thumb.png').convert('RGBA'); W, H = im.size
ov = Image.new('RGBA', im.size, (0, 0, 0, 0)); d = ImageDraw.Draw(ov); cx, cy, r = W // 2, H // 2 - 40, 70
d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(15, 26, 43, 200))
d.polygon([(cx - 22, cy - 36), (cx - 22, cy + 36), (cx + 40, cy)], fill=(246, 244, 238, 255))
Image.alpha_composite(im, ov).convert('RGB').resize((960, 540), Image.LANCZOS).save('$OUT/demo_thumb.jpg', quality=88)"
ls -lh $OUT
