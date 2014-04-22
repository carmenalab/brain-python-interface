#!/bin/bash
DISPLAY=`ps aux | grep -o "/usr/bin/X :[0-9]" | grep -o ":[0-9]"`
# if [[ -z `mount | grep /storage/blackrock` ]]
#     then
#     sudo mount /storage/blackrock
# fi
trap ctrl_c INT SIGINT SIGKILL SIGHUP

MANAGER=$HOME/code/bmi3d/db/manage.py
#MANAGER=manage.py

# Start python processes and save their PIDs (stored in the bash '!' variable 
# immediately after the command is executed)
python $MANAGER runserver 0.0.0.0:8000 --noreload &
DJANGO=$!
python $MANAGER celery worker &
CELERY=$!
python $MANAGER celery flower --address=0.0.0.0 &
FLOWER=$!

# Define what happens when you hit control-C
function ctrl_c() {
	kill -9 $DJANGO
	kill $CELERY
	kill $FLOWER
	kill -9 `ps -C 'python manage.py' -o pid --no-headers`
}

# Run until the PID stored in $DJANGO is dead
wait $DJANGO
kill $CELERY
kill $FLOWER
