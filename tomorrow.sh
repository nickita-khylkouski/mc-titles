#!/bin/zsh
# after the YouTube quota resets (midnight PT): finish details for the last channels, then put views on every map
cd ~/mc-titles
target=$(date -v+1d -v0H -v10M -v0S +%s); now=$(date +%s)
[ $(date +%H) -lt 1 ] && target=$(date -v0H -v10M -v0S +%s)
while [ $(date +%s) -lt $target ]; do sleep 60; done  # polls the clock: a single long sleep stalls while the Mac sleeps
echo "== details $(date)" >> data/tomorrow.log
WORKERS=4 python3 fetch_details.py >> data/details.log 2>&1
echo "== details exit $? $(date)" >> data/tomorrow.log
while pgrep -f "python map.py" >/dev/null || pgrep -f queue_maps.sh >/dev/null; do sleep 60; done
for src in title thumb rich; do
  [ -f data/map_state$([ $src = title ] || echo _$src).npz ] && MAP_SRC=$src nice uv run python map.py --apply >> data/tomorrow.log 2>&1
done
echo "== applied $(date)" >> data/tomorrow.log
