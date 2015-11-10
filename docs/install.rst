Installation
============
This document describes how to set up the BMI3D code for use in
neurophysiology experiments. 

Downloading the software
------------------------
The software should reside at the path $HOME/code/bmi3d, where $HOME is your home directory (this should be a variable already defined in your bash environment). If the bmi3d path does not already exist, create it by executing the commands::

    mkdir $HOME/code
    cd $HOME/code
    git clone https://github.com/carmenalab/brain-python-interface.git bmi3d

This will also clone the core software in the repository and put it in the appropriately named folder


Library dependencies
--------------------
Many of the commands to execute below are available in the file $BMI3D/install/install.sh. It's highly recommended that you read through the instructions rather than blindly executing the script as every BMI rig will have its own slight differences

Directory structure
-------------------
We make several directories. The main BMI3D package code base will be installed to $HOME/code/bmi3d. But if you're already reading these instructions from the git repository, you can change the installation location(s) if you want by modifying the environment variables::
    CODE=$HOME/code
    BMI3D=$HOME/code/bmi3d

We also make the directories /backup and /storage. /backup is the mount point for off-site data backup. /storage is where files created during the process of running experiments, e.g. HDF files containing BMI kinematics, will be stored. Most of the file handling in /storage is done through Django::

    mkdir -p $CODE
    sudo mkdir /backup
    sudo chown $USER /backup
    # make log directory
    mkdir $BMI3D/log
    sudo mkdir /storage
    sudo chown -R lab /storage
    mkdir /storage/plots

We also make a directory to dump miscellaneous source code bases required to run the experiments::

    mkdir $CODE/src/

Configuring the Ubuntu package manager
--------------------------------------
We need apt-get to search for packages in a couple of repositories that are not managed by Ubuntu, so we have to add these manually. Specifically we need to add repositories for the RabbitMQ and Sublime Text packages (the second is optional)::

    sudo apt-add-repository "deb http://www.rabbitmq.com/debian/ testing main"
    cd $HOME
    wget http://www.rabbitmq.com/rabbitmq-signing-key-public.asc
    sudo apt-key add rabbitmq-signing-key-public.asc
    sudo add-apt-repository ppa:webupd8team/sublime-text-2

    ## Refresh the package manager's list of available packages
    sudo apt-get update

Install apt-get dependencies
----------------------------
We install all the apt-get dependencies at once. All the commands are given with the "-y" flag so that they install automatically, so read through the list if you need to be careful of collisions::

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


Install Python dependencies
---------------------------
For the remainder of the Python dependencies, we use the python package-manager "pip". This makes the installation procedure marginally less platform dependent. Many of these packages will be necessary if you want to take full advantage of the BMI3D software for the purpose of analyzing data::

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
    sudo pip install tables

Additional source code
----------------------
Suraj's matplotlib code. Obviously optional::
    git clone https://github.com/sgowda/plot $HOME/code/plotutil

Robotics toolbox. Necessary prerequisite due to import statements, but can be removed if your experiment does not use any of the kinematic chain functionality built in (and you remove the appropriate import statements)::
    git clone https://github.com/sgowda/robotics_toolbox $HOME/code/robotics

All the graphics run through pygame::
    hg clone https://bitbucket.org/pygame/pygame $HOME/code/pygame

See install.sh for the detailed commands to install each of these packages

udev and groups
---------------
udev rules are necessary for any user to interact with the Phidgets board (optional) and the NIDAQ card. In addition, to allow the BMI3D software to be run as a regular user but still interact with various hardware devices/serial ports, it is necessary to add the experimenter user ('lab', in our case) to the 'dialout' group ()for serial port access) and the 'iocard' group (NIDAQ board access). 

Path configuration
------------------
After all the packages have been installed through their various mechanisms (apt-get, pip, source, etc.), we must add the appropriate BMI3D folders to the system's python path. At the top of the BMI3D folder lives a file called 'pathconfig.sh'. If you 'source' this script, it will make the appropriate modifications to the PYTHONPATH. To avoid having to source this constantly, add a line to the .bashrc file so that it gets sourced every time you open a new shell::
    sed -i '$a source $HOME/code/bmi3d/pathconfig.sh' $HOME/.bashrc

We also need to make sure the robotics toolbox is on the PYTHONPATH::
    sed -i '$a export PYTHONPATH=$PYTHONPATH:$HOME/code/robotics' $HOME/.bashrc

Reboot
------
At this point, reboot so that changes can take effect (ssh server running, hardware drivers get loaded into kernel, etc.). After the machine boots back up again, check that you are a member of the 'iocard' and 'dialout' groups. 


Managing fstab
--------------
Entries in the fstab (file system table) detail information when you give the command ``mount $MOUNTPOINT``, so that you don't have to specify all the permissions, etc. explicitly every time you issue the mount command. We add 3 entries to fstab::

    /dev/sdb1 /storage        ext4    defaults 0       0
    //project.eecs.berkeley.edu/carmena /backup cifs noauto,username=sgowda,domain=EECS,sec=ntlmssp,uid=localuser,dir_mode=0777,file_mode=0777 0 0
    //10.0.0.13/PlexonData /storage/plexon  smbfs   user=arc,pass=c@rmena,uid=1000,gid=1000 0 0

The first specifies '/storage' as the mount-point for a second hard drive (/dev/sdb1 is the first partition on hard drive 'b'). In our system, we run the operating system off a small solid-state drive (/dev/sda) and store data on a larger regular hard drive (/dev/sdb). 

The second specifies how to mount the offsite backup. In this case, the protocol is CIFS. 

The third entry specifies how to mount the data directory of the neural recording PC (in our case, this is a Windows PC provided by plexon). You may wish to also assign a different IP address to the neural recording PC. This line also will not work until you set up DHCP in the next step

rsync & crontab
---------------
rsync is a unix command-line utility which can synchronize two folders (including remote folders). This funcitonality is similar to how cloud storage services (like Dropbox) operate, but for a single user. An example rsync command::

    rsync -rlt --partial /home/lab/code/bmi3d/db/db.sql /backup/exorig/db_exorig.sql
    rsync -rlt --exclude=plexonlocal --exclude=video --partial /storage/ /backup/exorig/

These two commands force ``/backup/exorig/db_exorig.sql`` to match the ``db.sql`` which is modified by the rig, and also to synchronize ``/storage`` with ``/backup``, where in our earlier modified ``/etc/fstab``, ``/backup`` is an off-site storage directory which we network mounted. 

You can used the ``crontab`` utility to automatically run rsync commands at a certain time of the day, to make the synchronization automatic without human intervention. 


Network configuration
=====================
In our setup, the main PC (named 'arc') has two network cards. One faces the outside internet (interface eth0) and the other is used for communicating with other devices through a local switch (interface eth1). Other devices might include the neural recording PC, an eyetracker, a motiontracker, etc. In order for all these devices to talk to each other, they must all have a unique IP on the local subnet assigned by a DHCP server running on the main computer, arc. 

DHCP
----
First, we specify eth1 as our DHCP server interface by editing /etc/default/isc-dhcp-server to read::

    INTERFACES="eth1"

The default for the INTERFACES line should be an empty string

Second, we have to configure /etc/network/interfaces to use eth0 as the "outward" facing interface and eth1 as the internal interface. For eth0, we add the following lines to /etc/network/interfaces::
    # The primary network interface
    auto eth0
    iface eth0 inet dhcp

which essentially tells it to ask some external DHCP server for a license for eth0. For eth1, where arc is acting as the server rather than the client, the configuration is slightly more complex::

    # Set up the internal wired network
    auto eth1
    iface eth1 inet static
        address 10.0.0.1
        network 10.0.0.0
        netmask 255.255.255.0
        broadcast 10.0.0.255

If your two network configuration is the same as ours, you can simply run the command::

    sudo cp $BMI3D/install/arc_interfaces /etc/network/interfaces

Third, we have to edit the DHCP configuration file, /etc/dhcp/dhcpd.conf. For each mac address on the internal network, we assign a name and an IP address to that machine in a static configuration (so that they're always the same when the experiments are running). We add lines to declare the subnet IP space::

    subnet 10.0.0.0 netmask 255.255.255.0 {
      range 10.0.0.10 10.0.0.254;
      option routers 10.0.0.1;
    }

and then the specific IP address for each machine, e.g. for the plexon PC below::

    host plexon {
        hardware ethernet b8:ac:6f:92:50:e1;
        fixed-address 10.0.0.13;
    }

Now we restart the DHCP server (which should be running in the background automatically if you've rebooted the machine as specified above) with::

    sudo service isc-dhcp-server restart

All the instructions above can also be automated by running the script $BMI3D/install/install_dhcp.sh. To force the windows computer to renew its DHCP lease and change IP address, run "ipconfig /release" followed by "ipconfig /renew" on the command line.

You may have to reboot the machine to make the DHCP server changes go into effect before continuing on to configure network address translation. 

NAT
---
In our setup, the external internet is used only sparingly by the neural recording PC. So we would like to have some internet access but we don't necessarily want to have a direct interface dedicated to external internet (cost, security, etc.). So we use network address translation (NAT) to route internet traffic through the main experimental computer (arc). The configuration script used for arc was copied from the internet (https://help.ubuntu.com/community/Router). To execute it, run::

    sudo $BMI3D/install/arc_install_nat.sh

At this point, you should be able to get external internet on the Windows PC. You may need to renew the DHCP license to do so. 

NOTE: every time you add a new machine to the dchp config file, it seems that you must re-run the NAT setup script. Otherwise the new machine will get an
IP address from DHCP but will not be able to reach the outside internet. 


Setting up a gateway machine
----------------------------
You may not want to allow direct SSH to your rig machine and instead force ssh traffic through a gateway machine. This is a great idea for security, since if your rig machine is compromised, you'll may have to redo many of the steps above (and you might lose data!). But making a gateway means that copying files over to your analysis machine is annoying, since you basically have to execute twice as many ``scp`` commands. A nice alternative is an SSH tunnel that you can create on your analysis machine. If 'portal' is the gateway and 'nucleus' is the rig machine, then on you analysis machine you can execute the commands

.. code :: bash

    kill `ps aux | grep 8000 | grep ssh | tr -s ' ' | cut -d ' ' -f 2`
    kill `ps aux | grep 22 | grep ssh | tr -s ' ' | cut -d ' ' -f 2`
    ssh -f -N -L 43002:nucleus:8000 portal
    ssh -f -N -L 43001:nucleus:22 portal

This forwards port 8000 (for Django) to local port 43002 and port 22 (for ssh) to local port 43001. Then in your local ssh config file (~/.ssh/config), 

    Host nucleus_tunnel 
        HostName localhost
        Port 43001 
        User helene

Then any subsequent ssh/scp commands can use 'nucleus_tunnel' in place of 'nucleus' and just work as if they were on the same local network as 'nucleus'. Similarly, you can remotely view the web interface by pointing your browser to localhost:43002




Peripheral devices
==================

Testing the NIDAQ interface
---------------------------
.. note :: the NIDAQ card is deprecated!

The NIDAQ card uses the 'comedi' device driver for linux, written in C. There is a wrapper for the library, pycomedi. Unfortunately we don't seem to have properly configured things, so initializing the device doesn't seem to work form python. Instead, the C version of the code must be used for initializing the device, after which the IO lanes can be read/written from python. 


**Setting up the operating system**

.. note :: This step is only if you are using the NI PCI 6503 card. If not, skip it! Or if you need a particular kernel version for a different reason, you should be able to adapt these instructions

The bmi3d software is written primarily in Python. It has
primarily been tested in operation in Ubuntu linux, though some testing has been
done for CentOS 6. Ubuntu is substantially easier to set up, and these instructions 
are geared toward an Ubuntu setup. 

If you are using the NI6503 card to send digital data, we recommend specifically
using Ubuntu 12.04 LTS with kernel version 3.2.x-x. The version of
Ubuntu may not be very important, but the specific kernel version 
is important *if you are using the NI6503 card to send digital data*.
If you are using an arduino microcontroller, you can skip this step. 

The instructions below are intended for a clean install of Ubuntu. If 
you do not do this on a clean installation, you must be careful with
the instructions below, which actually remove later versions of the kernel.
Although multiple versions of the kernel can coexist on the same system
if you use a bootloader such as grub, we have chosen to keep only
one version so as to avoid the possiblity in a deployment environment
of accidentally rebooting the machine and using the wrong version.

When installing Ubuntu, either create the installer with the correct
kernel version (not sure exactly how to do this, but of course at some
point the default installer was the version of the kernel we want) OR
install any later version of the kernel and downgrade the image. Rough 
instructions are below downgrade the kernel image::

    sudo apt-get install linux-headers-3.2.0-60-generic
    sudo apt-get install linux-image-3.2.0-60-generic
    sudo apt-get remove --purge linux-image-3.11.0-*    
    sudo apt-get remove --purge linux-headers-3.11.0-*
    sudo apt-get install grub
    
    # Update the bootloader; a bootloader may not be strictly necessary..
    sudo grub-mkconfig -o /boot/grub/grub.cfg
    sudo update-grub

These can also be found in the script $BMI3D/install/ubuntu_kernel_mod.sh.
Of course if you install with a kernel version other than 3.11.0, make
the appropriate replacement. You may have to run the purge commands more than
once if apt-get decides to install another kernel version automatically when
you remove the one you installed. 

After checking in /boot/grub/grub.cfg that only kernel version 3.2.0-60 appears
in the bootloader menu, restart the machine so that the kernel changes take
effect. 


Renaming serial ports
---------------------
Certain peripheral devices make use of serial ports. Linux assigns somewhat random names to serial ports, which can change based on the order in which they're plugged in. This makes it annoying to write the name of the serial port in the code. 

These instructions help automatically rename serial ports: http://hintshop.ludvig.co.nz/show/persistent-names-usb-serial-devices/


Running the first tasks
=======================


Running the Django server for the first time
--------------------------------------------
First, for some reason the matplotlib configuration file directory appears to be owned by root when making these instructions. The Django software needs matplotlib for some reason, so we change ownership of the directorh $HOME/.matplotlib back to the user, which is what it should be anyway::
    sudo chown -R $USER ~/.matplotlib

Before running the experiment server, we have to create an empty database file. This is done by::

    cd $BMI3D/db
    python manage.py syncdb

You will be prompted to create a superuser account for the database. Since our database will never be publicly visible (it's more for record-keeping purposes than for building a website, which is what Django was intended for), there's no need to worry too much about password security here. 

To fire up the experimental rig, from the same 'db' directory run::

    ./runserver.sh

Then, from your browser, point to the address::

    localhost:8000


Running a simple task
---------------------
From the browser, start the visual_feedback_multi task. Make sure to check the 'autostart' and 'saveHDF' features (otherwise the task will not run), select the 'centerout_2D_discerete' generator from the Sequence menu, and select 'CursorPlant' in the arm_class. 









Configuration files
-------------------
config : rig-specific configurations, e.g., data paths, neural recording system. 
tasklist : List of tasks which can be started through the web interface
featurelist : list of features which can be selected through the web interface
bmilist : 
    type of decoding algorithm, plant type, signal source (spike counts, lfp)
    BMI menu only shows up for task classes which are marked as bmi, with the task class attribute is_bmi_seed






Automatic testing
-----------------
[(]this section is still incomplete]
Use the Ubuntu GUI to add a "testing" user
As the testing user
- clone the BMI3D repo
- run make_config.py; make sure data paths are correct; other options don't matter
- install the robitics toolbox and add to python path
- make sure group ownership and file permissions allow the testing user to read data files
- mkdir $BMI3D/test_output


Configuring ipython
-------------------
For quick analyses from the command line, it can be useful to have ipython pre-loaded with some commonly used modules. For complete instructions on how to set this up, see https://ipython.org/ipython-doc/dev/config/intro.html. You can make a default ipython profile by executing the shell command::

    ipython profile create

Then edit the newly create configuration file to look something like 

c.InteractiveShellApp.exec_lines = [
    "from db import dbfunctions as dbfn",
    "from db.tracker import models",
    "import numpy as np",
    "import matplotlib.pyplot as plt",
]



Graphics
--------
.. note :: This step is only needed if (1) you have an nvidia graphics card and (2) the 3-D stereo graphics do not render properly. If they do render okay, then don't change the driver!

After downgrading the kernel on arc, the graphics are all messed up. The second monitor isn't detected and there's a weird border around the first screen. It seems like this has something to do with the kernel version change (i.e., the driver was kernel-version specific). So we do a::

    sudo apt-get install nvidia-331

where at the time of this writing, "331" was the latest stable version of the library available in the standard Ubuntu repository. At some point during the installation says it's "Building initial module for 3.2.0-60-generic". After installing, we reboot again. After reboot, the second monitor has things displayed on it again and everything looks normal!