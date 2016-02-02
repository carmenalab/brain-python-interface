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
    git clone https://github.com/hgm110/bmi3d.git $HOME/code/bmi3d
fi

# ADDED 2 LINES BELOW
cd $BMI3D
git checkout task_dev

# make log directory
mkdir $BMI3D/log

####### Reconfigure Ubuntu package manager
sudo apt-add-repository "deb http://www.rabbitmq.com/debian/ testing main"
cd $HOME
wget http://www.rabbitmq.com/rabbitmq-signing-key-public.asc
sudo apt-key add rabbitmq-signing-key-public.asc
# COMMENTED OUT LINE BELOW
# sudo add-apt-repository ppa:webupd8team/sublime-text-2

## Refresh the package manager's list of available packages
sudo apt-get update


####### Install Ubuntu dependencies
# COMMENTED OUT LINE BELOW (don't want libhdf5-serial-dev; want to install from source)
# sudo apt-get -y install python-pip libhdf5-serial-dev
sudo apt-get -y install python-pip
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
# COMMENTED OUT LINE BELOW
# sudo apt-get -y install sublime-text vim-gnome
sudo apt-get -y install rabbitmq-server
sudo apt-get -y install libusb-dev
sudo apt-get -y install ipython
# NIDAQ
sudo apt-get -y install libcomedi-dev
sudo apt-get -y install python-comedilib
sudo apt-get -y install swig
# DHCP server
sudo apt-get -y install isc-dhcp-server
# for playing audio files
sudo apt-get -y install sox


####### Install Python dependencies
yes | sudo pip install numexpr 
yes | sudo pip install cython 
yes | sudo pip install django-celery 
yes | sudo pip install traits 
yes | sudo pip install pandas 
yes | sudo pip install patsy 
yes | sudo pip install statsmodels 
yes | sudo pip install PyOpenGL PyOpenGL_accelerate
yes | sudo pip install Django==1.6 
yes | sudo pip install pylibftdi
yes | sudo pip install nitime==0.4
yes | sudo pip install sphinx
yes | sudo pip install numpydoc
yes | sudo pip install tornado
# COMMENTED OUT LINE BELOW (cannot install tables without HDF5, which we install from source later)
# yes | sudo pip install tables


####### Download any src code
git clone https://github.com/sgowda/plot $HOME/code/plotutil
git clone https://github.com/sgowda/robotics_toolbox $HOME/code/robotics
# pygame
hg clone https://bitbucket.org/pygame/pygame $HOME/code/pygame




####### Install source code, configure software
# plexread module
cd $BMI3D/riglib
sudo python setup.py install

# pygame
cd $HOME/code/pygame
sudo python setup.py install

# symlink for iPython
sudo ln -s /usr/bin/ipython /usr/bin/ipy

# NIDAQ software
# COMMENTED OUT LINE BELOW - want to use nidaq/build_blackrock.sh instead
# $HOME/code/bmi3d/riglib/nidaq/build.sh
$HOME/code/bmi3d/riglib/nidaq/build_blackrock.sh

# Phidgets libraries
cd $CODE/src/
wget http://www.phidgets.com/downloads/libraries/libphidget.tar.gz
tar xzf libphidget.tar.gz 
cd libphidget*
./configure
make
sudo make install

cd $CODE/src/
wget http://www.phidgets.com/downloads/libraries/PhidgetsPython.zip
unzip PhidgetsPython.zip  
cd PhidgetsPython
sudo python setup.py install



####### Configure udev rules, permissions
# Phidgets
sudo cp udev/99-phidgets.rules /etc/udev/rules.d
sudo chmod a+r /etc/udev/rules.d/99-phidgets.rules
# NIDAQ
sudo cp $HOME/code/bmi3d/install/udev/comedi.rules /etc/udev/rules.d/
sudo chmod a+r /etc/udev/rules.d/comedi.rules 
sudo udevadm control --reload-rules
# Group permissions
sudo usermod -a -G iocard lab # NIDAQ card belongs to iocard group
sudo usermod -a -G dialout lab # Serial ports belong to 'dialout' group


####### Reconfigure .bashrc
sed -i '$a export PYTHONPATH=$PYTHONPATH:$HOME/code/robotics' $HOME/.bashrc
sed -i '$a source $HOME/code/bmi3d/pathconfig.sh' $HOME/.bashrc
source $HOME/.bashrc

sudo chown -R $USER ~/.matplotlib

# COMMENTED OUT LINES BELOW (need tables first, and need HDF5 before that)
# cd $HOME/code/bmi3d/db
# python manage.py syncdb
# Add superuser 'lab' with password 'lab'




###########################################################################
########################### iBMI specific stuff ###########################
###########################################################################

# build/install HDF5 libraries from source so that Default API Mapping is v18
# (need v18 in order to build n2h5 conversion utility)
if [ ! -f "/usr/lib/libhdf5.so" ]; then
    cd $CODE/src
    wget http://www.hdfgroup.org/ftp/HDF5/current/src/hdf5-1.8.13.tar.gz
    tar zxvf hdf5-1.8.13.tar.gz
    cd hdf5-1.8.13
    ./configure --prefix=/usr
    make
    make check
    sudo make install
    sudo make check-install
fi

yes | sudo pip install tables

# add following line to end of .bashrc
sed -i '$a export PYTHONPATH=$PYTHONPATH:$HOME/code' $HOME/.bashrc
source $HOME/.bashrc


sudo apt-get -y install cmake
sudo apt-get -y install build-essential
sudo apt-get -y install libqt4-*
sudo apt-get -y install python-setuptools

yes | sudo pip install flower

# the pip-installed h5py doesn't currently work with hdf5-1.8.13
# yes | sudo pip install h5py

# instead, clone h5py git repo and install from source
if [ ! -d "$HOME/code/src/h5py" ]; then
    git clone https://github.com/h5py/h5py.git $HOME/code/src/h5py
    cd $HOME/code/src/h5py
    python setup.py build
    python setup.py test
    sudo python setup.py install
fi

# Sublime Text editor
sudo add-apt-repository -y ppa:webupd8team/sublime-text-3
sudo apt-get update
sudo apt-get -y install sublime-text-installer
sudo ln -s /usr/bin/subl /usr/bin/sublime  # create symlink


mkdir /storage/blackrock

# first setup static IP connection

# add line below to end of /etc/fstab; adjust as needed
# may have to add option sec=ntlmssp too (e.g., ...,gid=1000,sec=ntlmssp)
# //192.168.137.1/BlackrockData /storage/blackrock  smbfs   user=Siddharth,pass=Sidshow24,uid=1000,gid=1000 0 0


cd $HOME/code

# manual step needed -- copy "assist_params" folder to /storage/

# add following line to end of .bashrc
sed -i '$a export PYTHONPATH=$PYTHONPATH:$HOME/code/bmi3d/riglib/ismore/sim' $HOME/.bashrc
source $HOME/.bashrc

# clone CereLink git repo
if [ ! -d "$HOME/code/CereLink" ]; then
    git clone https://github.com/sdangi/CereLink $HOME/code/CereLink

    cd $HOME/code/CereLink
    git remote add upstream https://github.com/dashesy/CereLink.git

    # build old cbpy (CereLink.cbpy) and n2h5 conversion utility
    cd $HOME/code/CereLink/build
    cmake .
    make all
    sudo make install

    # build new cbpy (cerebus.cbpy)
    cd $HOME/code/CereLink
    python setup.py build_ext --inplace

    # increase length of socket buffer (UDP memory buffer)
    # add this line to the end of the /etc/sysctl.conf file
    net.core.rmem_max=8388608
fi

# add following line to end of .bashrc
sed -i '$a export PYTHONPATH=$PYTHONPATH:$HOME/code/CereLink' $HOME/.bashrc
source $HOME/.bashrc

cd $HOME/code/bmi3d/db
python manage.py syncdb
# Add superuser 'lab' with password 'lab'

sudo apt-get -y install apt-show-versions

mkdir -p /storage/rawdata/hdf
mkdir -p /storage/rawdata/bmi
mkdir -p /storage/bmi_params

# add following line to end of .bashrc
sed -i '$a export DJANGO_SETTINGS_MODULE=db.settings' $HOME/.bashrc
source $HOME/.bashrc

# add following line to end of .bashrc
sed -i '$a export PYTHONPATH=$PYTHONPATH:$HOME/code/bmi3d/riglib' $HOME/.bashrc
sed -i '$a export PYTHONPATH=$PYTHONPATH:$HOME/code/bmi3d/db/tracker' $HOME/.bashrc
source $HOME/.bashrc

# TODO: add sample001.nev and .ns5 files to /storage/blackrock
# (to be able to use add_blackrock_....py script)

# TODO: create empty plexfile.py and psth.py files in riglib/plexon

# in Django admin, change path to /storage/blackrock for System "blackrock"
# make folder /storage/decoders
# in Django admin, change path to /storage/decoders for System "bmi"

# if you encounter a mounting error due to memory:
# http://boinst.wordpress.com/2012/03/20/mount-cifs-cannot-allocate-memory-mounting-windows-share/
