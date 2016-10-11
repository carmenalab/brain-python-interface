#!/bin/bash
####### Declare environment variables
CODE=$HOME/code
BMI3D=$CODE/bmi3d ### Directory in which to install the bmi3d software


####### Set up directories
mkdir -p $CODE
sudo mkdir /backup
sudo chown $USER /backup

sudo mkdir /storage
sudo chown -R $USER /storage
mkdir /storage/plots
mkdir $CODE/src/

sudo apt-get -y install git gitk
if [ ! -d "$HOME/code/bmi3d" ]; then
    git clone https://github.com/carmenalab/bmi3d.git $HOME/code/bmi3d

    #Add tasks & analysis, if desired
    git clone https://github.com/carmenalab/bmi3d_tasks_analysis.git $HOME/code/bmi3d_tasks_analysis

    #Make symlinks to tasks/analysis in main bmi3d repository
    ln -s $HOME/code/bmi3d_tasks_analysis/analysis $HOME/code/bmi3d/analysis
	ln -s $HOME/code/bmi3d_tasks_analysis/tasks $HOME/code/bmi3d/tasks

fi

# make log directory
mkdir $BMI3D/log

####### Reconfigure Ubuntu package manager
sudo apt-add-repository "deb http://www.rabbitmq.com/debian/ testing main"
cd $HOME
wget http://www.rabbitmq.com/rabbitmq-signing-key-public.asc
sudo apt-key add rabbitmq-signing-key-public.asc
sudo add-apt-repository ppa:webupd8team/sublime-text-2

## Refresh the package manager's list of available packages
sudo apt-get update


####### Install Ubuntu dependencies
sudo apt-get -y install python-pip libhdf5-serial-dev
sudo apt-get -y install python-numpy
sudo apt-get -y install python-scipy
# setup the CIFS 
sudo apt-get -y install smbclient cifs-utils smbfs
# matplotlib
sudo apt-get -y install python-matplotlib
# pygame
sudo apt-get -y install mercurial python-dev python-numpy ffmpeg libsdl-image1.2-dev libsdl-mixer1.2-dev libsdl-ttf2.0-dev libsmpeg-dev libsdl1.2-dev  libportmidi-dev libswscale-dev libavformat-dev libavcodec-dev
# install tools
sudo apt-get -y install libtool automake bison flex
# ssh
sudo apt-get -y install openssh-server
# text editors
sudo apt-get -y install sublime-text vim-gnome
sudo apt-get -y install rabbitmq-server
sudo apt-get -y install libusb-dev
sudo apt-get -y install ipython
# NIDAQ
sudo apt-get -y install libcomedi-dev
sudo apt-get -y install python-comedilib
sudo apt-get -y install swig
# DHCP server
sudo apt-get -y install isc-dhcp-server
# cURL: command line utility for url transfer
sudo apt-get -y install curl
sudo apt-get -y install sqlite3
# Arduino IDE
sudo apt-get install arduino arduino-core  
# Serial lib
sudo apt-get install setserial

####### Install Python dependencies
sudo pip install numexpr 
sudo pip install cython 
sudo pip install django-celery 
sudo pip install traits 
sudo pip install pandas 
sudo pip install patsy 
sudo pip install statsmodels 
sudo pip install PyOpenGL PyOpenGL_accelerate
sudo pip install Django==1.6 
sudo pip install pylibftdi
sudo pip install nitime
sudo pip install sphinx
sudo pip install numpydoc
sudo pip install tornado
sudo pip install tables==2.4.0
sudo pip install sklearn


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
sudo python setup.py install

# pygame
cd $HOME/code/pygame
sudo python setup.py install

# symlink for iPython
sudo ln -s /usr/bin/ipython /usr/bin/ipy

# NIDAQ software -- deprecated!
# $HOME/code/bmi3d/riglib/nidaq/build.sh

# Phidgets libraries
cd $CODE/src/
tar xzf libphidget.tar.gz 
cd libphidget*
./configure
make
sudo make install

cd $CODE/src/
unzip PhidgetsPython.zip  
cd PhidgetsPython
sudo python setup.py install



####### Configure udev rules, permissions
# Phidgets
sudo cp $CODE/src/libphidget*/udev/99-phidgets.rules /etc/udev/rules.d
sudo chmod a+r /etc/udev/rules.d/99-phidgets.rules
# NIDAQ
sudo cp $HOME/code/bmi3d/install/udev/comedi.rules /etc/udev/rules.d/
sudo chmod a+r /etc/udev/rules.d/comedi.rules 
sudo udevadm control --reload-rules
# Group permissions
sudo usermod -a -G iocard $USER # NIDAQ card belongs to iocard group
sudo usermod -a -G dialout $USER # Serial ports belong to 'dialout' group


####### Reconfigure .bashrc
sed -i '$a export PYTHONPATH=$PYTHONPATH:$HOME/code/robotics' $HOME/.bashrc
sed -i '$a export BMI3D=/home/lab/code/bmi3d' $HOME/.bashrc
sed -i '$a source $HOME/code/bmi3d/pathconfig.sh' $HOME/.bashrc
source $HOME/.bashrc

sudo chown -R $USER ~/.matplotlib

cd $HOME/code/bmi3d/db
python manage.py syncdb
# Add superuser 'lab' with password 'lab'
