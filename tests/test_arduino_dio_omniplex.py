#!/usr/bin/python
'''
Test the remote start and data sending capabilities of the arduino for omniplex
'''
from features.arduino_features import PlexonSerialDIORowByte
import serial, glob

f = PlexonSerialDIORowByte()
f.pre_init()
# f.cleanup()