#!/bin/bash

trap ctrl_c INT

python manage.py runserver 0.0.0.0:8000 &
DJANGO=$!
python manage.py celery worker &
CELERY=$!
python manage.py celery flower &
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
