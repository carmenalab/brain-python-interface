
import serial, time
port = serial.Serial('/dev/arduino_neurosync')
port.write('p')
time.sleep(0.5)
port.write('r')
time.sleep(3)

#port.close()
