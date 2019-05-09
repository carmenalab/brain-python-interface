#!/bin/bash
####### Declare environment variables
CODE=/code
BMI3D=$CODE/bmi3d ### Directory in which to install the bmi3d software
USER=root  # We're in a docker container so root is safe 

####### Set up directories
mkdir -p $CODE
mkdir /backup
chown $USER /backup

mkdir /storage
chown -R $USER /storage
mkdir /storage/plots
mkdir $CODE/src/

# make log directory
mkdir $BMI3D/log

# Add the repository to get the rabbitmq server
curl -s https://packagecloud.io/install/repositories/rabbitmq/rabbitmq-server/script.deb.sh | bash
apt-get update


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
apt-get -y install ipython
# NIDAQ
apt-get -y install libcomedi-dev python-comedilib  swig
# DHCP server
apt-get -y install isc-dhcp-server
apt-get -y install sqlite3
# Arduino IDE
apt-get install arduino arduino-core  
# Serial lib
apt-get install setserial

####### Install Python dependencies
pip install -r requirements.txt


####### Download any src code
git clone https://github.com/sgowda/plot $HOME/code/plotutil
git clone https://github.com/sgowda/robotics_toolbox $HOME/code/robotics
# pygame
hg clone https://bitbucket.org/pygame/pygame $HOME/code/pygame
# Phidgets code
wget http://www.phidgets.com/downloads/libraries/libphidget.tar.gz
wget http://www.phidgets.com/downloads/libraries/PhidgetsPython.zip




####### Install source code, configure software
# plexread module
cd $BMI3D/riglib
python setup.py install

# pygame
cd $HOME/code/pygame
python setup.py install

# symlink for iPython
ln -s /usr/bin/ipython /usr/bin/ipy

# NIDAQ software -- deprecated!
# $HOME/code/bmi3d/riglib/nidaq/build.sh

# Phidgets libraries
cd $CODE/src/
tar xzf libphidget.tar.gz 
cd libphidget*
./configure
make
make install

cd $CODE/src/
unzip PhidgetsPython.zip  
cd PhidgetsPython
python setup.py install



####### Configure udev rules, permissions
# Phidgets
cp $CODE/src/libphidget*/udev/99-phidgets.rules /etc/udev/rules.d
chmod a+r /etc/udev/rules.d/99-phidgets.rules
# NIDAQ
cp $HOME/code/bmi3d/install/udev/comedi.rules /etc/udev/rules.d/
chmod a+r /etc/udev/rules.d/comedi.rules 
udevadm control --reload-rules
# Group permissions
usermod -a -G iocard $USER # NIDAQ card belongs to iocard group
usermod -a -G dialout $USER # Serial ports belong to 'dialout' group


####### Reconfigure .bashrc
sed -i '$a export PYTHONPATH=$PYTHONPATH:$HOME/code/robotics' $HOME/.bashrc
sed -i '$a export BMI3D=/home/lab/code/bmi3d' $HOME/.bashrc
sed -i '$a source $HOME/code/bmi3d/pathconfig.sh' $HOME/.bashrc
source $HOME/.bashrc

chown -R $USER ~/.matplotlib

cd $HOME/code/bmi3d/db
python manage.py syncdb
# Add superuser 'lab' with password 'lab'
