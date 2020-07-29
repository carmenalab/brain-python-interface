#!/bin/bash
SOURCE_PATH=`realpath ..`
WORK_VOLUME=bmi3d_vol
DOCKER_IMG=bmi3d:latest

# for graphics
export DISPLAY=:0
xhost +

docker volume create $WORK_VOLUME     # this will be persistent every time the image is invoked

docker run --rm -ti \
    -v $WORK_VOLUME:/work \
    -v $SOURCE_PATH:/src \
    -w /work \
    -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
    -p 8000:8000 \
    $DOCKER_IMG bash
