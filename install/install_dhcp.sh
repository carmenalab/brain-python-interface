#!/bin/bash
sudo cp arc-isc-dhcp-server /etc/default/isc-dhcp-server
sudo cp arc_interfaces /etc/network/interfaces
sudo cp arc_dhcpd.conf /etc/dhcp/dhcpd.conf
sudo service isc-dhcp-server restart

###- To force a windows computer to renew its DHCP lease and change IP address, run "ipconfig /release" followed by "ipconfig /renew" on the command line.
sudo cp arc_fstab /etc/fstab


# some useful debugging links for problems setting up the machine in Spain
# http://askubuntu.com/questions/57155/dhcpd-fails-to-start-on-eth1
# http://serverfault.com/questions/78240/debugging-rules-in-iptables
# http://www.linuxquestions.org/questions/linux-networking-3/need-help-debugging-iptables-firewall-nat-gateway-681930/
# http://prefetch.net/articles/iscdhcpd.html
# http://askubuntu.com/questions/590550/dhcp-server-not-running
# http://askubuntu.com/questions/579773/dhcp-server-setting
# https://codeghar.wordpress.com/2012/05/02/ubuntu-12-04-ipv4-nat-gateway-and-dhcp-server/

# start server in foreground (for debugging)
# sudo dhcpd

# to enable ip_forwarding
# edit /etc/sysctl.conf 
# sysctl -w net.ipv4.ip_forward=1


# to figure out what your DNS servers are for the main network interface
# cat /etc/resolv.conf

# to view your iptables rules after running the "arc_install_nat.sh" script
# iptables -nvL -t nat