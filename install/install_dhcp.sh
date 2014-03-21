#!/bin/bash
###The DHCP server on arc is working, we can ping back and forth between arc and plexon, and the firewall is configured to allow plexnet data to stream. 
###
###Other fun/useful things we learned about network settings:
###
###- To specify which network card to use for the DHCP server on arc, add "eth1" to /etc/default/isc-dhcp-server to prevent conflicts with the campus network on the card that accesses the external network.
###
###- To assign a specific static IP address to a computer on the rig network, add the following to /etc/dhcp/dhcpd.conf:
### 
### host plexon {
###     hardware ethernet b8:ac:6f:92:50:e1;
###     fixed-address 10.0.0.13;
### }
###
###where plexon is an arbitrary name for the computer you want to assign, ethernet is its MAC address, fixed address is the IP you want to assign.
###
###- To restart the DHCP server on arc to apply configuration changes, run "sudo /etc/init.d/isc-dhcp-server restart"
###
###- To force a windows computer to renew its DHCP lease and change IP address, run "ipconfig /release" followed by "ipconfig /renew" on the command line.
sudo cp arc_interfaces /etc/network/interfaces
sudo cp arc_dhcpd.conf /etc/dhcp/dhcpd.conf
sudo cp arc-isc-dhcp-server /etc/default/isc-dhcp-server
sudo service isc-dhcp-server restart
