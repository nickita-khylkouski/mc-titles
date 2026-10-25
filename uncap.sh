#!/bin/zsh
# Uncapped re-pull of the 95 channels over 1,500 videos, then everything downstream. One nice'd job at a time.
# Phase 1 (today): wait for fetch_videos.py, then embed/tag only the new videos (no YouTube quota needed).
# Phase 2 (after the midnight-PT quota reset): list the rest, embed those, fetch details, rebuild the 3 maps.
cd ~/mc-titles
L=data/uncap.log
log() { echo "== $* $(date)" >> $L; }
step() { log "$1"; shift; "$@" >> $L 2>&1; log "exit $?"; }
downstream() {
  step titles     nice uv run python embed.py
  step rich       nice uv run python embed_rich.py
  step thumbs-dl  env WORKERS=12 nice python3 thumbs.py
  step thumbs-emb env WORKERS=6 nice uv run python embed_thumbs.py
  step is-mc      env WORKERS=3 nice uv run python tag_mc.py
}
while pgrep -f "fetch_videos.py" >/dev/null; do sleep 60; done
downstream
log "phase 1 done"

target=$(date -v+1d -v0H -v10M -v0S +%s)
[ $(date +%H) -lt 1 ] && target=$(date -v0H -v10M -v0S +%s)
while [ $(date +%s) -lt $target ]; do sleep 60; done  # polls the clock: a single long sleep stalls while the Mac sleeps
step list env CAP=0 REDO_CAPPED=1 QUOTA_BUDGET=9000 WORKERS=4 nice python3 fetch_videos.py
downstream
used=$(sed -n '/^== list /,$p' $L | grep -o "quota used [0-9]*" | tail -1 | grep -o "[0-9]*$"); used=${used:-0}
step details env QUOTA_BUDGET=$((9600 - used)) WORKERS=4 nice python3 fetch_details.py
for src in title rich thumb; do
  step "map $src" env MAP_SRC=$src nice uv run python map.py
done
log "phase 2 done: review auto cluster names, then eras/insights/analyze + restart serve.py"
