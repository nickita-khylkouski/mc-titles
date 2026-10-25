#!/bin/zsh
# Finish the pull after the YouTube quota resets (midnight Pacific), then embed + rebuild the map.
cd "${0:A:h}"
while pgrep -f fetch_videos.py >/dev/null; do sleep 60; done
now=$(date +%s)
target=$(TZ=America/Los_Angeles date -j -v+1d -f "%Y-%m-%d %H:%M:%S" "$(TZ=America/Los_Angeles date +%Y-%m-%d) 00:10:00" +%s)
echo "$(date) sleeping $(( (target - now) / 60 )) min until quota reset"
sleep $(( target - now ))
echo "$(date) resuming fetch"
QUOTA_BUDGET=9500 WORKERS=4 nice python3 fetch_videos.py >> data/fetch.log 2>&1
echo "$(date) fetch exit $?"
nice uv run python embed.py && nice uv run python map.py > data/map.log 2>&1
echo "$(date) map exit $?"
