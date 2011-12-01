import serial
import reward_msg
from options import reset as resetoptions
from options import save as saveoptions
from options import options

try:
    port = serial.Serial("/dev/ttyUSB0", baudrate=38400)
    reward = reward_msg.ReadMsg(port)
except:
    print "Cannot find reward system"
    reward = None