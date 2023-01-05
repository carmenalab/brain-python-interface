import serial
from .gpio import CustomBoard
import time
import numpy as np
from . import singleton

# condigPacket is 4 bytes configuration command and contains the laser information
# configPacket[0] : Channel (red, green, blue, white) & Mode (off, continuous, sin, inverted sin, square waveform)
# configPacket[1] : Frequency index
# configPacket[2]~[3] : Laser beam gain (LSB, then MSB)

#configPacket = [0b01000100, 0b00000000, 0b00000000, 0b10000000]

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

def get_config_packet(channel, freq, gain, mode='off', debug=False):
    """
    Args:
    channel (int): Laser channel. The value has to be 1, 2, 3, or 4.
    freq (float): 0.5 ~ 500 Hz. selects closest from above table
    gain (float): Laser gain. The value has to be from 0 to 1
    mode (str): Laser mode. CW - continous wave, SW - sine wave, ISW - inverse sine wave, SQW - square wave. Defaults to off
    debug (bool): debug mode prints the config byte. Defaults to False.

    Returns:
    configPacket (int) : The signal to send the laser. It contains all information of 4 args.
    """

    gain_16_bit = int(gain*65535)
    mode_int = 0
    if mode == 'CW':
        mode_int = 1
    elif mode == 'SW':
        mode_int = 2
    elif mode == 'ISW':
        mode_int = 3
    elif mode == 'SQW':
        mode_int = 4

    byte1 = (channel - 1) << 6 | mode_int
    try:
        byte2 = np.searchsorted(frequency_table, freq)
    except:
        byte2 = 0
    byte3 = gain_16_bit & 0xff # LSB
    byte4 = gain_16_bit >> 8 # MSB

    configPacket = [byte1, byte2, byte3, byte4]

    if debug:
        print(f"Config packet for ch {channel}, mode {mode}, freq {freq}, gain {gain}:")
        for byte in configPacket:
            print(np.binary_repr(byte, width=8))

    return configPacket

class QwalorLink(singleton.Singleton):
    '''
    Helper singleton that acts as the serial link to the laser.
    '''

    __instance = None
        
    def __init__(self, laser_port='/dev/qwalorlaser', laser_baud_rate=115200):
        super().__init__()
        self.link = serial.Serial(laser_port, laser_baud_rate, timeout=3)

    def send(self, config_packet):
        self.link.write(config_packet)
        self.link.flush()
        time.sleep(0.005)

class QwalorLaserSerial:
    '''
    Implentation of the quad-wavelength laser modulator. By default, the laser starts in
    the 'off' mode, which can be used for TTL operation. Only need to set the power.
    '''

    channel = 2
    mode = 'off'
    gain = 0.
    freq = 0.

    def __init__(self, laser_channel, arduino_port=None, arduino_pin=12):
        '''
        Hoping to change this in the future to use a raspberry pi or similar connected to the network to relay the
        config packets to the laser. That way we can have multiple computers talking to the laser at once.
        '''
        if arduino_port == None:
            arduino_port = f"/dev/laser_ch{laser_channel}"
        self.trigger_pin = arduino_pin
        self.channel = laser_channel
        self._set_config()
        self.board = CustomBoard(arduino_port, baudrate=57600, timeout=10)

    def _set_config(self):
        config_packet = get_config_packet(self.channel, self.freq, self.gain, self.mode)
        link = QwalorLink.get_instance()
        link.send(config_packet)
            
    def set_mode(self, mode):
        self.mode = mode
        self._set_config()

    def set_freq(self, freq):
        self.freq = freq
        self._set_config()

    def set_power(self, gain):
        self.gain = gain
        self._set_config()

    def write_many(self, mask, data):
        '''
        Somewhat strange, but the way the lasers are implemented in othertasks.py requires they implement
        a `write_many` method, which should write the given data to the given masked outputs. In our case,
        this will be a single channel so we can safely ignore the mask and just turn off and on the laser.
        '''
        if data:
            self.on()
        else:
            self.off()

    def on(self):
        self.board.digital[self.trigger_pin].write(1)

    def off(self):
        self.board.digital[self.trigger_pin].write(0)

    def __del__(self):
        if hasattr(self, 'ser'):
            self.ser.close()
        if hasattr(self, 'board'):
            self.board.exit()


