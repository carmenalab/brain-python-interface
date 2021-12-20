import serial
import pyfirmata, time
import serial.tools.list_ports
import numpy as np
from riglib import qwalor_laser

# The following command can be used for finding available ports
# python -m serial.tools.list_ports

frequency_table = [0.5, 0.5137, 0.5278, 0.5423, 0.5572, 0.5725, 0.5882, 0.6044, 0.621, 0.638, 0.6556, 0.6736, 0.6921,
                         0.7111, 0.7306, 0.7507, 0.7713, 0.7924, 0.8142, 0.8366, 0.8595, 0.8831, 0.9074, 0.9323, 0.9579,
                         0.9842, 1.0112, 1.039, 1.0675, 1.0968, 1.127, 1.1579, 1.1897, 1.2224, 1.2559, 1.2904, 1.3259,
                         1.3623, 1.3997, 1.4381, 1.4776, 1.5182, 1.5599, 1.6027, 1.6467, 1.6919, 1.7384, 1.7861, 1.8352,
                         1.8856, 1.9373, 1.9905, 2.0452, 2.1014, 2.1591, 2.2183, 2.2793, 2.3418, 2.4062, 2.4722, 2.5401,
                         2.6099, 2.6815, 2.7552, 2.8308, 2.9085, 2.9884, 3.0705, 3.1548, 3.2414, 3.3304, 3.4219, 3.5158,
                         3.6124, 3.7116, 3.8135, 3.9182, 4.0258, 4.1363, 4.2499, 4.3666, 4.4865, 4.6097, 4.7363, 4.8664,
                         5, 5.1373, 5.2784, 5.4233, 5.5722, 5.7252, 5.8824, 6.044, 6.2099, 6.3805, 6.5557, 6.7357, 6.9206,
                         7.1107, 7.3059, 7.5066, 7.7127, 7.9245, 8.1421, 8.3656, 8.5954, 8.8314, 9.0739, 9.3231, 9.5791,
                         9.8421, 10.1124, 10.39, 10.6753, 10.9685, 11.2697, 11.5791, 11.8971, 12.2238, 12.5594, 12.9043,
                         13.2587, 13.6227, 13.9968, 14.3811, 14.776, 15.1818, 15.5987, 16.027, 16.4671, 16.9193, 17.3839,
                         17.8612, 18.3517, 18.8556, 19.3734, 19.9054, 20.4519, 21.0135, 21.5906, 22.1834, 22.7926, 23.4185,
                         24.0615, 24.7222, 25.4011, 26.0986, 26.8152, 27.5516, 28.3081, 29.0855, 29.8841, 30.7047, 31.5479,
                         32.4142, 33.3042, 34.2187, 35.1584, 36.1238, 37.1157, 38.1349, 39.1821, 40.258, 41.3635, 42.4993,
                         43.6663, 44.8654, 46.0973, 47.3632, 48.6637, 50, 51.373, 52.7836, 54.2331, 55.7223, 57.2524,
                         58.8245, 60.4398, 62.0994, 63.8047, 65.5567, 67.3569, 69.2064, 71.1068, 73.0594, 75.0655, 77.1268
                         , 79.2447, 81.4207, 83.6564, 85.9536, 88.3139, 90.7389, 93.2305, 95.7906, 98.421, 101.1236,
                         103.9004, 106.7534, 109.6848, 112.6967, 115.7913, 118.9709, 122.2377, 125.5943, 129.0431, 132.5865,
                         136.2273, 139.968, 143.8115, 147.7605, 151.8179, 155.9867, 160.27, 164.671, 169.1928, 173.8387,
                         178.6122, 183.5168, 188.5561, 193.7338, 199.0536, 204.5195, 210.1355, 215.9057, 221.8344, 227.9258,
                         234.1845, 240.6151, 247.2223, 254.0109, 260.9859, 268.1524, 275.5158, 283.0813, 290.8546, 298.8413,
                         307.0473, 315.4787, 324.1416, 333.0423, 342.1875, 351.5838, 361.2381, 371.1575, 381.3493, 391.8209,
                         402.5801, 413.6348, 424.993, 436.6631, 448.6536, 460.9734, 473.6315, 486.6372, 500]

# condigPacket is 4 bytes configuration command and contains the laser information
# configPacket[0] : Channel (red, green, blue, white) & Mode(off, continuous, sin, inverted sin, square waveform)
# configPacket[1] : Frequency
# configPacket[2]~[3] : Laser beam gain

#configPacket = [0b01000100, 0b00000000, 0b00000000, 0b10000000]

def setConfig(channel, mode, freq, gain):
    """
    Args:
    channel (int): Laser channel. The value has to be 1, 2, 3, or 4.
    mode (int): Laser mode. 
    freq (int): 0 ~ 255
    gain (float): Laser gain. The value has to be from 0 to 1

    Returns:
    configPacket (int) : The signal to send the laser. It contains all information of 4 args.
    """

    # configure setting for channel and mode
    channel_byte = format(channel - 1, '02b')

    if mode == 'off':
        mode_byte = format(0, '03b')
    elif mode == 'CW':
        mode_byte = format(1, '03b')
    elif mode == 'SW':
        mode_byte = format(2, '03b')
    elif mode == 'ISW':
        mode_byte = format(3, '03b')
    elif mode == 'SQW':
        mode_byte = format(4, '03b')

    byte1 = '0b' + channel_byte + '000' + mode_byte
    byte1 = int(byte1, 0)

    # configure setting for frequency
    frequency_byte = format(freq, '08b')
    byte2 = '0b' + frequency_byte
    byte2 = int(byte2, 0)

    # configure setting for gain (byte3 is LSB and byte4 is MSB)
    gain_byte = format(round(gain*65535), '016b')
    gain_byte3 = gain_byte[8:] #LSB
    gain_byte4 = gain_byte[0:8] #MSB
    byte3 = '0b' + gain_byte3
    byte3 = int(byte3, 0)
    byte4 = '0b' + gain_byte4
    byte4 = int(byte4, 0)

    configPacket = [byte1, byte2, byte3, byte4]

    return configPacket


# Arduino setting
LED_pin = 13
pin = 12
port_arduino = "/dev/ttyACM1"

# laser setting
channel = 1
mode = 'CW'
freq = 0
gain = 0.5
b_rate = 115200
port_laser = '/dev/ttyUSB0'

def test_manual_config():

    # set configuration signal
    configPacket = setConfig(channel, mode, freq, gain)

    # open serial port
    board = pyfirmata.Arduino(port_arduino)
    ser = serial.Serial(port_laser, b_rate) 

    # send signal to laser
    ser.write(configPacket)

    #board.digital[LED_pin].write(0)
    time.sleep(0.1)

    # trigger
    pulse_width_list = [0.00001, 0.00002, 0.00003, 0.00005, 0.0001, 0.0002, 0.0003, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1]
    for idx, width in enumerate(pulse_width_list):
        t0 = time.perf_counter()
        board.digital[pin].write(1)
        while (time.perf_counter() - t0 < width):
            pass
        board.digital[pin].write(0)
        time.sleep(0.5)

    #gain_list = np.linspace(0.1, 1, num=10, endpoint=True)

    #for gain in gain_list:
    #    configPacket = setConfig(channel, mode, freq, gain)
    #    ser.write(configPacket)

    #    time.sleep(1)
    #    board.digital[pin].write(1)
    #    time.sleep(1)
    #    board.digital[pin].write(0)

    # close serial port
    board.exit()
    ser.close()

def test_class_config():

    laser = qwalor_laser.QwalorLaserSerial(port_laser, channel, port_arduino, pin)
    laser.set_power(gain)

    pulse_width_list = [0.00001, 0.00002, 0.00003, 0.00005, 0.0001, 0.0002, 0.0003, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1]
    for idx, width in enumerate(pulse_width_list):
        t0 = time.perf_counter()
        laser.on()
        while (time.perf_counter() - t0 < width):
            pass
        laser.off()
        time.sleep(0.5)

test_class_config()