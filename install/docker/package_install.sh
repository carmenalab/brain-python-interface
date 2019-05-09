#!/bin/bash

# Add the repository to get the rabbitmq server
curl -s https://packagecloud.io/install/repositories/rabbitmq/rabbitmq-server/script.deb.sh | bash
apt-get update
apt-get -y upgrade


####### Install Ubuntu dependencies
# apt-get -y install python-pip libhdf5-serial-dev
# setup the CIFS 
apt-get -y install smbclient cifs-utils
# pygame
# apt-get -y install mercurial python-dev python-numpy ffmpeg libsdl-image1.2-dev libsdl-mixer1.2-dev libsdl-ttf2.0-dev libsmpeg-dev libsdl1.2-dev  libportmidi-dev libswscale-dev libavformat-dev libavcodec-dev
# install tools
apt-get -y install bison flex
# ssh
apt-get -y install openssh-server
# text editors
# apt-get -y install sublime-text vim-gnome
apt-get -y install rabbitmq-server
apt-get -y install libusb-dev
# apt-get -y install ipython
# NIDAQ
apt-get -y install libcomedi-dev python-comedilib  swig
# DHCP server
apt-get -y install isc-dhcp-server
apt-get -y install sqlite3
# Arduino IDE
# apt-get -y install arduino arduino-core  
# Serial lib
# apt-get -y install setserial