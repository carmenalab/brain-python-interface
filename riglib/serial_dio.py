'''
DIO using a serial port + microcontroller instead of the NIDAQ card
'''
import serial
import glob
from collections import defaultdict

class SendAll(object):
    '''
    Interface for sending all the task-generated data through the NIDAQ interface card
    '''
    def __init__(self, device=None):
        '''
        Constructor for SendAll

        Parameters
        ----------
        device : string, optional
            Linux name of the serial port for the Arduino board, defined by setserial

        Returns
        -------
        SendAll instance
        '''
        self.systems = dict()
        self.port = serial.Serial(glob.glob("/dev/ttyACM*")[0], baudrate=9600)
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

        # TODO
        # Save the index of the system being registered (arbitrary number corresponding to the order in which systems were registered)
        self.n_systems += 1
        self.systems[system] = self.n_systems

        if self.n_systems > 1:
        	raise Exception("This currently only works for one system!")

        print "Arduino register %s" % system, self.systems[system]

    def sendMsg(self, msg):
        '''
        Send a string mesasge to the recording system, e.g., as related to the task_msgs HDF table

        Parameters
        ----------
        msg : string
            Message to send

        Returns
        -------
        None
        '''
        pass
        # TODO
        # pcidio.sendMsg(str(msg))

import struct
from numpy import binary_repr
def construct_word(aux, msg_type, data, n_bits_data=8, n_bits_msg_type=3):
    word = (aux << (n_bits_data + n_bits_msg_type)) | (msg_type << n_bits_data) | data
    return word

class SendRowByte(SendAll):
    '''
    Send only an 8-bit data word corresponding to the 8 lower bits of the current row number of the HDF table
    '''
    def send(self, system, data):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        if not (system in self.systems):
            return

        current_sys_rowcount = self.rowcount[system]
        self.rowcount[system] += 1

        # construct the data packet        
        msg_type = 5
        word = construct_word(self.systems[system], msg_type, current_sys_rowcount % 256)
        
        verbose = 0
        if verbose:
            print binary_repr(word, 16)
        word_str = 'd' + struct.pack('<H', word)
        self.port.write(word_str)
        

# extern uchar sendRowByte(uchar idx) {
#     uint flush = 2, msg = (idx << 3 | SEND_ROWBYTE) << 8 | (255 & rowcount[idx]);
#     comedi_dio_bitfield2(ni, 0, writemask, &msg, 0);
#     comedi_dio_bitfield2(ni, 0, 2, &flush, 16);
#     rowcount[idx]++;
#     return 0;
# }            
