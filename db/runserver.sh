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
	kill -9 `ps a | grep 'python manage.py' | cut -f 1 -d ' '`
}

wait $DJANGO
kill $CELERY
kill $FLOWER
