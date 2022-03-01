'''
DIO using a serial port + microcontroller instead of the NIDAQ card
'''
import serial
from collections import defaultdict
import struct
from numpy import binary_repr
from .dio.parse import MSG_TYPE_ROWBYTE, MSG_TYPE_REGISTER

def construct_word(aux, msg_type, data, n_bits_data=8, n_bits_msg_type=3):
    word = (aux << (n_bits_data + n_bits_msg_type)) | (msg_type << n_bits_data) | data
    return word

def parse_word(word, n_bits_data=8, n_bits_msg_type=3):
    data = word & ((1 << n_bits_data) - 1)
    msg_type = (word >> n_bits_data) & ((1 << n_bits_msg_type) - 1) 
    aux = word >> n_bits_msg_type + n_bits_data
    return aux, msg_type, data

baudrate = 115200

class SendRowByte(object):
    '''
    Send only an 8-bit data word corresponding to the 8 lower 
    bits of the current row number of the HDF table
    '''
    '''
    Interface for sending all the task-generated data through the NIDAQ interface card
    '''
    def __init__(self, device=None):
        '''
        Constructor for SendRowByte

        Parameters
        ----------
        device : string, optional
            Linux name of the serial port for the Arduino board, defined by setserial

        Returns
        -------
        SendAll instance
        '''
        self.systems = dict()
        self.port = serial.Serial('/dev/arduino_neurosync', baudrate=baudrate)
        self.n_systems = 0
        self.rowcount = defaultdict(int)

    def close(self):
        '''
        Release access to the Arduino serial port
        '''
        # stop recording
        self.port.write('p')
        self.port.close()

    def register(self, system, dtype):
        '''
        Send information about the registration system (name and datatype) in string form, one byte at a time.

        Parameters
        ----------
        system : string
            Name of the system being registered
        dtype : np.dtype instance
            Datatype of incoming data, for later decoding of the binary data during analysis

        Returns
        -------
        None
        '''
        # Save the index of the system being registered (arbitrary number corresponding to the order in which systems were registered)
        self.n_systems += 1
        self.systems[system] = self.n_systems


        # if self.n_systems > 1:
        #     raise Exception("This currently only works for one system!")
        
        #print "System Register: %s" % system, self.systems[system]
        #print "Arduino register %s" % system, self.systems[system]

        #if self.n_systems > 1:
        #    raise Exception("This currently only works for one system!")

        print("Arduino register %s" % system, self.systems[system])

        for sys_name_chr in system:
            reg_word = construct_word(self.systems[system], MSG_TYPE_REGISTER, ord(sys_name_chr))
            self._send_data_word_to_serial_port(reg_word)

        null_term_word = construct_word(self.systems[system], MSG_TYPE_REGISTER, 0) # data payload is 0 for null terminator
        self._send_data_word_to_serial_port(null_term_word)

    def sendMsg(self, msg):
        '''
        Do nothing. Messages are stored with row numbers in the HDF table, so no need to also send the message over to the recording system.

        Parameters
        ----------
        msg : string
            Message to send

        Returns
        -------
        None
        '''
        # there's no point in sending a message, since every message is 
        # stored in the HDF table anyway with a row number, 
        # and every row number is automatically synced.
        pass

    def send(self, system, data):
        '''
        Send the row number for a data word to the neural system

        Parameters
        ----------
        system : string 
            Name of system 
        data : object
            This is unused. Only used in the parent's version where the actual data, and not just the HDF row number, is sent.

        Returns
        -------
        None
        '''
        
        if not (system in self.systems):
            # if the system is not registered, do nothing
            return

        current_sys_rowcount = self.rowcount[system]
        self.rowcount[system] += 1

        # construct the data packet
        word = construct_word(self.systems[system], MSG_TYPE_ROWBYTE, current_sys_rowcount % 256)
        self._send_data_word_to_serial_port(word)
        
        # if verbose:
        #     print binary_repr(word, 16)
        # word_str = 'd' + struct.pack('<H', word)
        # self.port.write(word_str)

    def _send_data_word_to_serial_port(self, word, verbose=False):
        #self.port.write(word)
        if verbose:
            print(binary_repr(word, 16))
        word_str = 'd' + struct.pack('<H', word)
        self.port.write(word_str)
