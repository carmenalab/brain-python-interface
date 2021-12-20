import serial
import pyfirmata
import time

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

class QwalorLaserSerial():
    '''
    Implentation of single-channel CW mode for the quad-wavelength laser modulator
    '''

    def __init__(self, laser_port, laser_channel, arduino_port, arduino_pin, laser_baud_rate=115200):

        self.ser = serial.Serial(laser_port, laser_baud_rate) 
        self.board = pyfirmata.Arduino(arduino_port)
        self.trigger_pin = arduino_pin
        self.channel = laser_channel

    def set_power(self, gain):

        # make and send signal to laser
        mode = 'CW' # always use continuous wave mode
        freq = 0 # this doesn't matter in CW
        config_packet = setConfig(self.channel, mode, freq, gain)
        self.ser.write(config_packet)
        time.sleep(0.1)

    def write_many(self, mask, data):
        # TODO fix this!
        if data:
            self.on()
        else:
            self.off()

    def on(self):
        self.board.digital[self.trigger_pin].write(1)

    def off(self):
        self.board.digital[self.trigger_pin].write(0)

    def __del__(self):
        if self.ser:
            self.ser.close()
        if self.board:
            self.board.exit()


