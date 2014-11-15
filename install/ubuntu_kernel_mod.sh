#!/bin/bash
## This script downgrades the kernel to version 3.2.0-60 and removes newer kernel 
## images. In this script in particular, it is assumed that the installation of
## ubuntu was with kernel version 3.11.*-* (set by the installer)
sudo apt-get install linux-headers-3.2.0-60-generic
sudo apt-get install linux-image-3.2.0-60-generic
sudo apt-get remove --purge 3.11.0-*
sudo apt-get remove --purge 3.11.0-*
sudo apt-get install grub

# Update the bootloader; a bootloader may not be strictly necessary..
sudo grub-mkconfig -o /boot/grub/grub.cfg
sudo update-grub


sudo apt-get -y install python-comedilib
sudo apt-get -y install libcomedi-dev
sudo usermod -a -G iocard lab
sudo apt-get install vim
sudo apt-get install git
sudo cp $HOME/code/bmi3d/install/udev/comedi.rules /etc/udev/rules.d/
sudo chmod a+r /etc/udev/rules.d/comedi.rules
sudo udevadm control --reload-rules

mkdir ~/code
cd ~/code
git clone https://github.com/hgm110/bmi3d.git

# sudo reboot
echo
echo "REBOOT THE MACHINE !"
