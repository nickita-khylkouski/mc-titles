#!/bin/zsh
# Words map: once embed_rich.py is done and the thumbnail queue has finished, build MAP_SRC=rich
cd ~/mc-titles
while pgrep -f embed_rich.py >/dev/null || pgrep -f queue_maps.sh >/dev/null; do sleep 30; done
./queue_maps.sh rich
