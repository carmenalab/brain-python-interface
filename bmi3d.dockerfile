FROM python:3

##### Install required ubuntu packages
# Add the repository to get the rabbitmq server
RUN curl -s https://packagecloud.io/install/repositories/rabbitmq/rabbitmq-server/script.deb.sh | bash
RUN apt-get update && apt-get -y upgrade

# Install dependencies
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

# Rabbitmq can run into issues so run as a separate command
RUN apt-get install rabbitmq-server 


###### Install python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt


####### Set up directories and copy source code
#RUN mkdir -v -p /code/src/
#RUN mkdir -v -p /backup && chown root /backup
#RUN mkdir -v -p /storage/plots && chown -R root /storage


# ----- Expected cache invalidation here ------- #

COPY . /code/bmi3d/
WORKDIR /code/bmi3d/
RUN mkdir -v logs




# Prepare scripts: Fix line endings because windows breaks bash
# RUN sed -i 's/\r$//' install/docker/src_code_install.sh	



#RUN ./install/docker/src_code_install.sh 
 
#CMD [ "python", "./your-daemon-or-script.py" ]