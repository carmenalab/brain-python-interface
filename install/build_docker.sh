SOURCE_PATH=`realpath ..`
WORK_VOLUME=bmi3d_vol

docker volume create $WORK_VOLUME     # this will be persistent every time the image is invoked

docker build -t bmi3d \
	-f Dockerfile docker_context
