#!/bin/bash
####### Declare environment variables
CODE=/code
BMI3D=$CODE/bmi3d ### Directory in which to install the bmi3d software
USER=root  # We're in a docker container so root is safe 

####### Download any src code
git clone https://github.com/sgowda/plot $HOME/code/plotutil
git clone https://github.com/sgowda/robotics_toolbox $HOME/code/robotics
# pygame
hg clone https://bitbucket.org/pygame/pygame $HOME/code/pygame
# Phidgets code
#wget https://www.phidgets.com/downloads/phidget22/libraries/linux/libphidget22/libphidget22-1.1.20190417.tar.gz
#wget https://www.phidgets.com/downloads/phidget22/libraries/any/Phidget22Python/Phidget22Python_1.1.20190418.zip


####### Install source code, configure software
# plexread module
#cd $BMI3D/riglib
which python
#python setup.py install

# pygame
cd $HOME/code/pygame
python setup.py install

# NIDAQ software -- deprecated!
# $HOME/code/bmi3d/riglib/nidaq/build.sh

echo "TESTED IF HERE"

# Phidgets libraries
#cd $CODE/src/
#tar xzf libphidget.tar.gz 
#cd libphidget*
#./configure
#make
#make install

#cd $CODE/src/
#unzip PhidgetsPython.zip  
#cd PhidgetsPython
#python setup.py install



####### Configure udev rules, permissions
# Phidgets
#cp $CODE/src/libphidget*/udev/99-phidgets.rules /etc/udev/rules.d
#chmod a+r /etc/udev/rules.d/99-phidgets.rules
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

chown -R $USER ~/.matplotlibs

cd $HOME/code/bmi3d/db
python manage.py syncdb
# Add superuser 'lab' with password 'lab'
