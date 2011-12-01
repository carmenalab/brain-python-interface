import serial
import reward_msg

try:
	port
	reward
except NameError:
	try:
	    port = serial.Serial("/dev/ttyUSB0", baudrate=38400)
	    reward = reward_msg.ReadMsg(port)
	except:
	    print "Cannot find reward system"
	    reward = None