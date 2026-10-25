#!/bin/zsh
# after the rich rebuild: title names + numbers, then the thumbnail rebuild. One nice'd job at a time.
cd ~/mc-titles
L=data/finish.log
step() { echo "== $1 $(date)" >> $L; shift; "$@" >> $L 2>&1; echo "== exit $? $(date)" >> $L; }
while kill -0 26354 2>/dev/null; do sleep 20; done
step "apply title" env MAP_SRC=title nice uv run python map.py --apply
step "insights title" env MAP_SRC=title nice uv run python insights.py
step "analyze" nice uv run python analyze.py
step "eras" nice uv run python eras.py
step "map thumb" env MAP_SRC=thumb nice uv run python map.py
echo "== finish done $(date)" >> $L
