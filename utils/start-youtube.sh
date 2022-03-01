#!/bin/bash

if [ "$1" = "" ]; then
	echo -n "Insert Youtube URL: "
	read URL
else
	URL="$1"
fi

ID=$(echo "$URL" | tr "?&#" "\n\n" | grep -E "^v=.{11}$" | cut -d "=" -f 2)
EMBED_URL=https://www.youtube.com/embed/$ID?autoplay=1

echo $EMBED_URL

# Set the volume to 50%
amixer -qD pulse sset Master 50% 

# Start the browser
function ctrl_c() {
	pkill chromium-browser
	amixer -qD pulse sset Master 100% # Restore the volume to 100%
}
trap ctrl_c INT SIGINT SIGKILL SIGHUP
export DISPLAY=:0.1
chromium-browser $EMBED_URL --kiosk --window-size=2560,1440 --start-fullscreen --noerrdialogs --allow-running-insecure-content --remember-cert-error-decisions --autoplay-policy=no-user-gesture-required
YT=$!


