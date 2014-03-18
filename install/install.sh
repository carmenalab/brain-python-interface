mkdir -p $HOME/code
cd $HOME/code

sudo apt-get -y install git
if [ ! -d "$HOME/code/bmi3d" ]; then
    git clone https://github.com/hgm110/bmi3d.git $HOME/code/bmi3d
fi

sudo apt-get -y install python-pip libhdf5-serial-dev
sudo apt-get -y install python-numpy  # "sudo pip install numpy" didn't work
sudo apt-get -y install python-scipy
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

# setup the CIFS 
sudo apt-get -y install smbclient cifs-utils
sudo mkdir /backup
sudo chown lab /backup

MOUNTSTORAGE="/dev/sdb1 /storage        ext4    defaults 0       0"
MOUNTBACKUP="//project.eecs.berkeley.edu/carmena /backup cifs noauto,username=sgowda,domain=EECS,sec=ntlmssp,uid=localuser,dir_mode=0777,file_mode=0777 0 0"
MOUNTPLEXON="//10.0.0.13/PlexonData /storage/plexon  smbfs   user=arc,pass=c@rmena,uid=1000,gid=1000 0 0"

## edit fstab as instructed in the email from EECS
if [ ! `cat /etc/fstab | tr -s ' ' | cut -d " " -f 2 | grep "/backup"` ]; then
    sudo sed -i '$a $MOUNTBACKUP' /etc/fstab
fi

if [ ! `cat /etc/fstab | tr -s ' ' | cut -d " " -f 2 | grep "/storage"` ]; then
    sudo sed -i '$a $MOUNTSTORAGE' /etc/fstab
fi

if [ ! `cat /etc/fstab | tr -s ' ' | cut -d " " -f 2 | grep "/storage/plexon"` ]; then
    sudo sed -i '$a $MOUNTPLEXON' /etc/fstab
fi


# tornado web framework
sudo pip install tornado

# matplotlib
sudo apt-get -y install python-matplotlib

# make log directory
mkdir $HOME/code/bmi3d/log

# Make directory to store plots and other data analysis things
sudo mkdir /storage
sudo chown lab /storage
mkdir /storage/plots

# Add the following lines to .bashrc
sed -i '$a source $HOME/code/bmi3d/pathconfig.sh' $HOME/.bashrc

# get version 2.4.0 of pytables
sudo pip install git+https://github.com/PyTables/PyTables.git@v.2.4.0#egg=tables

# Install plexread module
cd ~/code/bmi3d/riglib
sudo python setup.py install

# pygame -- copied from http://www.pygame.org/wiki/CompileUbuntu
# pygame - install dependencies

sudo apt-get -y install mercurial python-dev python-numpy ffmpeg \
    libsdl-image1.2-dev libsdl-mixer1.2-dev libsdl-ttf2.0-dev libsmpeg-dev \
    libsdl1.2-dev  libportmidi-dev libswscale-dev libavformat-dev libavcodec-dev
# pygame - Grab source
hg clone https://bitbucket.org/pygame/pygame $HOME/pygame
cd $HOME/pygame
sudo python setup.py install

git clone https://github.com/sgowda/robotics_toolbox $HOME/code/robotics
sed -i '$a export PYTHONPATH=$PYTHONPATH:$HOME/code/robotics' $HOME/.bashrc

cd $HOME/code/bmi3d/db
python manage.py syncdb
# Add superuser 'lab' with password 'lab'

sudo apt-get -y install ipython
sudo ln -s /usr/bin/ipython /usr/bin/ipy

# NIDAQ software
sudo apt-get -y install libcomedi-dev
sudo apt-get -y install python-comedilib
sudo apt-get -y install swig
$HOME/code/bmi3d/riglib/nidaq/build.sh
sudo cp $HOME/code/bmi3d/install/comedi.rules /etc/udev/rules.d/
sudo chmod a+r /etc/udev/rules.d/comedi.rules 
sudo chown lab /dev/comedi0
sudo udevadm control --reload-rules
# Add 'lab' user to the iocard group to give write access to the NIDAQ card
sudo usermod -a -G iocard lab
# Will need to log out and log back in for the change to take effect?

# RabbitMQ
sudo apt-add-repository "deb http://www.rabbitmq.com/debian/ testing main"
# sudo sed -i '$a deb http://www.rabbitmq.com/debian/ testing main' /etc/apt/sources.list
cd $HOME
wget http://www.rabbitmq.com/rabbitmq-signing-key-public.asc
sudo apt-key add rabbitmq-signing-key-public.asc
sudo apt-get -y install rabbitmq-server

# DHCP for neural recording platform
sudo apt-get -y install isc-dhcp-server
## Add 'eth1' to interfaces in /etc/default/isc-dhcp-server
# Edit /etc/dhcp/dhcpd.conf
sudo /etc/init.d/isc-dhcp-server restart


sudo apt-get install libusb-dev
mkdir $HOME/src/
cd $HOME/src/
wget http://www.phidgets.com/downloads/libraries/libphidget.tar.gz
tar xzf libphidget.tar.gz 
cd libphidget*
./configure
make
sudo make install
# Copy udev rules
sudo cp udev/99-phidgets.rules /etc/udev/rules.d

cd $HOME/src/
wget http://www.phidgets.com/downloads/libraries/PhidgetsPython.zip
unzip PhidgetsPython.zip  
