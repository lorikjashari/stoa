#!/bin/bash
# render.sh <pages dir> <out dir> name:w:h ...
P=$1; O=$2; shift 2
for spec in "$@"; do IFS=: read n w h <<<"$spec"
  chromium --headless=new --no-sandbox --disable-gpu --hide-scrollbars --force-device-scale-factor=2 --default-background-color=00000000 \
    --virtual-time-budget=8000 --window-size=$w,$h --screenshot="$O/$n.png" "file://$P/$n.html" >/dev/null 2>&1 &
done; wait
