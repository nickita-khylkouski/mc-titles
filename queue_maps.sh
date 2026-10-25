#!/bin/zsh
# run map builds one after another (one CPU job at a time): waits for any running map.py first
cd ~/mc-titles
while pgrep -f "python map.py" >/dev/null; do sleep 30; done
for src in "$@"; do
  echo "== $src $(date)" >> data/queue.log
  MAP_SRC=$src nice uv run python map.py > data/map_$src.log 2>&1
  echo "== $src exit $? $(date)" >> data/queue.log
done
