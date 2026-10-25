#!/bin/zsh
# title map steps while the boxes build rich + thumb layouts. One nice'd job at a time.
cd ~/mc-titles
L=data/local_title.log
step() { echo "== $1 $(date)" >> $L; shift; "$@" >> $L 2>&1; echo "== exit $? $(date)" >> $L; }
step "apply title" env MAP_SRC=title nice uv run python pipeline/map.py --apply
step "insights title" env MAP_SRC=title nice uv run python pipeline/insights.py
step "analyze" nice uv run python pipeline/analyze.py
step "eras" nice uv run python pipeline/eras.py
echo "== done $(date)" >> $L
