#!/bin/sh
# Copies the map into ~/mc, the public repo nickita-khylkouski/mc that GitHub Pages serves at nickita.org/mc.
# No server there: search matches title words, and clicks read data*/nn/ from pipeline/static_nn.py.
# Then commit + push in ~/mc.
set -e
cd "$(dirname "$0")/.."
OUT=${1:-$HOME/mc}
mkdir -p "$OUT"
rsync -a --delete --exclude .git --exclude README.md --exclude .nojekyll --exclude 'nn/' web/ "$OUT/"
for s in "" _rich _thumb; do
  [ -d "data/static_nn$s" ] && rsync -a --delete "data/static_nn$s/" "$OUT/data$s/nn/"
done
touch "$OUT/.nojekyll"
cat > "$OUT/README.md" <<'MD'
# Minecraft YouTube, mapped

The public copy of the map at [nickita.org/mc](https://nickita.org/mc). About 487,000 videos from the 523 biggest Minecraft channels, laid out by what their titles, descriptions or thumbnails mean, with about 200 hand-named topics.

The code, the pipeline and how it's built are in [mc-titles](https://github.com/nickita-khylkouski/mc-titles). This repo is only the built site (`scripts/publish.sh` there writes it).
MD
du -sh "$OUT"
