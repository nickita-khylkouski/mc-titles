#!/bin/zsh
# wait for the overnight resume.sh (fetch -> embed -> map), then grab thumbnails for the new channels
cd "$(dirname "$0")"
while pgrep -f resume.sh >/dev/null; do sleep 60; done
WORKERS=96 python3 thumbs.py >> data/thumbs.log 2>&1
