#!/bin/zsh
# yt-dlp scrapes of Minecraft searches sorted by view count (no API quota). One at a time, nice'd.
cd "${0:A:h}"
queries=(
  "minecraft 100 days" "minecraft hardcore" "minecraft smp" "minecraft but" "minecraft manhunt"
  "minecraft speedrun" "minecraft mods" "minecraft modpack" "minecraft animation" "minecraft roleplay"
  "minecraft build" "minecraft redstone" "minecraft survival lets play" "minecraft hunger games"
  "minecraft minigames" "hypixel bedwars" "hypixel skywars" "minecraft pvp" "minecraft parkour"
  "minecraft prank" "minecraft challenge" "minecraft horror" "minecraft lore" "hermitcraft"
  "minecraft funny moments" "minecraft trolling" "minecraft 1.0" "minecraft machinima" "minecraft series"
)
for q in $queries; do
  f="data/search_${q// /_}.txt"
  [[ -s $f ]] && continue
  timeout 200 nice yt-dlp --flat-playlist --playlist-end 600 \
    --print "%(channel_id)s|%(channel)s|%(id)s|%(title)s" \
    "https://www.youtube.com/results?search_query=${q// /+}&sp=CAM%253D" > "$f" 2>/dev/null
  echo "$q: $(wc -l < $f)"
done
