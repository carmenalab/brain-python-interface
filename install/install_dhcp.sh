#!/bin/bash
sudo cp arc-isc-dhcp-server /etc/default/isc-dhcp-server
sudo cp arc_interfaces /etc/network/interfaces
sudo cp arc_dhcpd.conf /etc/dhcp/dhcpd.conf
sudo service isc-dhcp-server restart

###- To force a windows computer to renew its DHCP lease and change IP address, run "ipconfig /release" followed by "ipconfig /renew" on the command line.
sudo cp arc_fstab /etc/fstab