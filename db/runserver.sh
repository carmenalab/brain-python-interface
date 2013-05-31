#!/bin/bash
DISPLAY=`ps aux | grep -o "/usr/bin/X :[0-9]" | grep -o ":[0-9]"`
if [ -z `mount | grep /storage/plexon` ]
    then
    sudo mount /storage/plexon
fi
trap ctrl_c INT SIGINT SIGKILL SIGHUP

MANAGER=/home/helene/code/bmi3d/db/manage.py
#MANAGER=manage.py

python $MANAGER runserver 0.0.0.0:8000 --noreload &
DJANGO=$!
python $MANAGER celery worker &
CELERY=$!
python $MANAGER celery flower --address=0.0.0.0 &
FLOWER=$!

function ctrl_c() {
	kill -9 $DJANGO
	kill $CELERY
	kill $FLOWER
	kill -9 `ps -C 'python manage.py' -o pid --no-headers`
}

wait $DJANGO
kill $CELERY
kill $FLOWER
