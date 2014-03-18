#!/bin/bash
sudo cp arc_interfaces /etc/network/interfaces
sudo cp arc_dhcpd.conf /etc/dhcp/dhcpd.conf
sudo service isc-dhcp-server restart
