@echo off

IF "%~1"=="" (
	echo "No docker container name given!"
	EXIT /B 2
) ELSE (
	docker run -it --name configuration-inator %1 python config/make_config.py
	docker commit configuration-inator %1
)