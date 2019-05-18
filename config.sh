#! /bin/bash
if [$# -eq 0]; then
	echo "No docker image name given!"
else
	docker run -it --name configuration-inator $1 python config/make_config.py;
	docker commit configuration-inator $1;
fi