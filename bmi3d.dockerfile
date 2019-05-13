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
	sqlite3

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

COPY . /code/bmi3d/
WORKDIR /code/bmi3d/
RUN mkdir -v logs


###### Install python dependencies
RUN echo $PWD
RUN pip install --upgrade pip
RUN pip install -r requirements.txt


# Prepare scripts: Fix line endings because windows breaks bash
RUN sed -i 's/\r$//' install/docker/src_code_install.sh	



RUN ./install/docker/src_code_install.sh 
 
#CMD [ "python", "./your-daemon-or-script.py" ]