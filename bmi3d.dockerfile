FROM python:3

#### Connect up third-party repositories (rabbitmq and erlang)
RUN curl -s https://packagecloud.io/install/repositories/rabbitmq/rabbitmq-server/script.deb.sh | bash
RUN echo "deb http://dl.bintray.com/rabbitmq-erlang/debian bionic erlang" \
	>> /etc/apt/sources.list.d/bintray.erlang.list
RUN apt-get -y update


##### Install required ubuntu packages
RUN apt-get install -y \
	smbclient \
	cifs-utils \
	bison \
	flex \
	openssh-server \
	libusb-dev\
	libcomedi-dev \
	python-comedilib  \
	swig \
	isc-dhcp-server \
	sqlite3 \
	vim

# Install rabbitmq with it's erlang dependencies
RUN apt-get install -y --allow-unauthenticated \
    erlang-base-hipe \
    erlang-asn1 \
    erlang-crypto \
    erlang-eldap \
    erlang-ftp \
    erlang-inets \
    erlang-mnesia \
    erlang-os-mon \
    erlang-parsetools \
    erlang-public-key \
    erlang-runtime-tools \
    erlang-snmp \
    erlang-ssl \
    erlang-syntax-tools \
    erlang-tftp \
    erlang-tools \
    erlang-xmerl \
    rabbitmq-server


####### Set up directories and copy source code
RUN mkdir -v -p /code/src/
RUN mkdir -v -p /backup && chown root /backup
RUN mkdir -v -p /storage/plots && chown -R root /storage

# --- Expect cache invalidation here if source files have changed --- #
COPY bmi3d/ /code/bmi3d/
COPY bmi3d_tasks_analysis/ /code/bmi3d_tasks_analysis/

# Replace windows symlinks with unix ones in the new env
RUN rm /code/bmi3d/analysis && ln -s /code/bmi3d_tasks_analysis/analysis/ /code/bmi3d/analysis \
 && rm /code/bmi3d/tasks    && ln -s /code/bmi3d_tasks_analysis/tasks/    /code/bmi3d/tasks

# Fix all .sh files that might have aquired windows line endings
RUN for f in $(find ./code/ -name "*.sh"); \
	do echo "fixing: $f" && sed -i 's/\r$//' $f; \
	done  

WORKDIR /code/bmi3d/
RUN mkdir -v logs


###### Install python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Set env vars for future reference
ENV BMI3D="/code/bmi3d" \
	PYTHONPATH="${PYTHONPATH}:/code/bmi3d/:/code/bmi3d_tasks_analysis"

# RUN python config/make_config.py 


CMD [ "/bin/bash", "./db/runserver.sh" ]
