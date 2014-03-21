#!/bin/bash
sudo cp arc_interfaces /etc/network/interfaces
sudo cp arc_dhcpd.conf /etc/dhcp/dhcpd.conf
sudo cp arc-isc-dhcp-server /etc/default/isc-dhcp-server
sudo service isc-dhcp-server restart
