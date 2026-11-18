#!/usr/bin/env bash
# Generate an egocentric chopping dataset and export it for training.
#
#   scripts/gen_ego_data.sh [EPISODES] [JOBS]        e.g. scripts/gen_ego_data.sh 200 2
#
# Env overrides: SCENE (kitchen|house), SEED (0), RES (256), FPS (10), ENGINE (cycles|eevee), SAMPLES (32), RAW (data/ego_raw),
#                LEROBOT (data/ego_lerobot), VLAQWEN (data/ego_vlaqwen), VIEWS ("head right_wrist" for vla-qwen).
# JOBS Blender processes render disjoint episode ranges in parallel (job j gets CUDA_VISIBLE_DEVICES=j, which
# picks the GPU for Cycles; EEVEE draws on the first GPU it finds). Episodes already rendered are skipped,
# so the script can be re-run to resume.
set -euo pipefail
cd "$(dirname "$0")/.."
EPISODES=${1:-20}
JOBS=${2:-2}
SCENE=${SCENE:-kitchen}
SEED=${SEED:-0}
RES=${RES:-256}
FPS=${FPS:-10}
ENGINE=${ENGINE:-cycles}
SAMPLES=${SAMPLES:-32}
RAW=${RAW:-data/ego_raw}
LEROBOT=${LEROBOT:-data/ego_lerobot}
VLAQWEN=${VLAQWEN:-data/ego_vlaqwen}
VIEWS=${VIEWS:-head right_wrist}
BL=tools/blender/blender
PY=.venv/bin/python

if [ "$SCENE" = house ]; then
  [ -f assets/world.blend ] || $BL -b -P blender/build_world.py
else
  [ -f assets/kitchen.blend ] || $BL -b -P blender/build_kitchen.py
fi
mkdir -p "$RAW"

per=$(( (EPISODES + JOBS - 1) / JOBS ))
pids=()
for ((j = 0; j < JOBS; j++)); do
  start=$(( j * per ))
  n=$(( EPISODES - start < per ? EPISODES - start : per ))
  [ "$n" -gt 0 ] || continue
  todo=()  # skip episodes that already have an episode.json
  for ((i = start; i < start + n; i++)); do
    [ -f "$RAW/$(printf 'episode_%06d' $i)/episode.json" ] || todo+=("$i")
  done
  [ ${#todo[@]} -gt 0 ] || continue
  (
    for i in "${todo[@]}"; do
      CUDA_VISIBLE_DEVICES=$j $BL -b -P blender/gen_ego_episodes.py -- --out "$RAW" --start "$i" --episodes 1 --scene "$SCENE" \
        --seed "$SEED" --res "$RES" --fps "$FPS" --engine "$ENGINE" --samples "$SAMPLES" 2>&1 | grep -E '^\[ego\]|Error|Traceback' || true
    done
  ) > "$RAW/gen_job$j.log" 2>&1 &
  pids+=($!)
  echo "[gen] job $j: episodes ${todo[0]}..$(( start + n - 1 )) (${#todo[@]} to render) -> $RAW/gen_job$j.log"
done
for p in "${pids[@]}"; do wait "$p"; done
cat "$RAW"/gen_job*.log 2>/dev/null | grep -c '^\[ego\]' | xargs -I{} echo "[gen] {} episodes rendered this run"

$PY scripts/ego_to_lerobot.py --raw "$RAW" --out "$LEROBOT" --overwrite
$PY scripts/ego_to_vlaqwen.py --raw "$RAW" --out "$VLAQWEN" --views $VIEWS
echo "[gen] done: $RAW -> $LEROBOT (LeRobot v2, all views) and $VLAQWEN (vla-qwen: $VIEWS)"
