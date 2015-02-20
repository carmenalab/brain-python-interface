#!/bin/bash
####### Declare environment variables
CODE=$HOME/code
BMI3D=$CODE/bmi3d ### Directory in which to install the bmi3d software

# numpy and numexpr needed to be removed and reinstalled
# numpy=1.6.1
# numexpr=2.4

####### Set up directories
mkdir -p $CODE
sudo mkdir /backup
sudo chown $USER /backup

sudo mkdir /storage
sudo chown -R $USER /storage
mkdir /storage/plots
mkdir $CODE/src/

sudo yum -y install git gitk
if [ ! -d "$HOME/code/bmi3d" ]; then
    git clone https://github.com/hgm110/bmi3d.git $HOME/code/bmi3d
fi

# make log directory
mkdir $BMI3D/log

# Add the EPEL repo, where some of the required packages live
cd /tmp
wget https://dl.fedoraproject.org/pub/epel/7/x86_64/e/epel-release-7-5.noarch.rpm
sudo yum install epel-release-7-5.noarch.rpm

####### Reconfigure RHEL package manager: have to install erlang before rabbitmq
'''
sudo apt-add-repository "deb http://www.rabbitmq.com/debian/ testing main"
cd $HOME
wget http://www.rabbitmq.com/rabbitmq-signing-key-public.asc
sudo apt-key add rabbitmq-signing-key-public.asc
sudo add-apt-repository ppa:webupd8team/sublime-text-2
'''
sudo wget http://packages.erlang-solutions.com/erlang-solutions_1.0_all.deb
sudo dpkg -i erlang-solutions_1.0_all.deb
sudo yum update
sudo yum -y install erlang-solutions
sudo rpm --import http://www.rabbitmq.com/rabbitmq-signing-key-public.asc
sudo yum -y install rabbitmq-server--1.noarch.rpm
sudo wget http://repo.cloudhike.com/sublime2/fedora/sublime2.repo -O /etc/yum.repos.d/sublime2.repo
## Refresh the package manager's list of available packages
sudo yum update


####### Install RHEL dependencies
sudo yum -y install libpng-devel lapack-devel
sudo yum -y install python-pip hdf5-devel
sudo yum -y install numpy
sudo yum -y install scipy
# setup the CIFS 
sudo yum -y install samba-client samba-common cifs-utils
# matplotlib
sudo yum -y install python-matplotlib
# pygame: check these
sudo yum -y python-devel SDL_image-devel SDL_mixer-devel SDL_ttf-devel SDL-devel smpeg-devel numpy subversion portmidi-devel
# install tools
sudo yum -y install libtool automake bison flex
# ssh
sudo yum -y install openssh-server
# text editors: need vim gnome?
sudo yum -y install sublime-text vim-X11 vim-enhanced
sudo yum -y install erlang rabbitmq-server
sudo yum -y install libusb-devel
sudo yum -y install ipython
# NIDAQ
sudo yum -y install comedilib-devel
sudo yum -y install comedilib
sudo yum -y install swig
# DHCP server
sudo yum -y install dhcp
# cURL: command line utility for url transfer
sudo yum -y install curl



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
sudo pip install pyserial


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

# NIDAQ software
$HOME/code/bmi3d/riglib/nidaq/build.sh

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
sed -i '$a source $HOME/code/bmi3d/pathconfig.sh' $HOME/.bashrc
source $HOME/.bashrc

sudo chown -R $USER ~/.matplotlib

cd $HOME/code/bmi3d/db
python manage.py syncdb
# Add superuser 'lab' with password 'lab'
