'''
DIO using a serial port + microcontroller instead of the NIDAQ card
'''
import serial

class SendAll(object):
    '''
    Interface for sending all the task-generated data through the NIDAQ interface card
    '''
    def __init__(self, device="/dev/ttyACM0"):
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
        self.port = serial.Serial(device)
        self.n_systems = 0
    
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
        print "Arduino register %s" % system

        # TODO
        # Save the index of the system being registered (arbitrary number corresponding to the order in which systems were registered)
        self.n_systems += 1
        self.systems[system] = self.n_systems

        if self.n_systems > 1:
        	raise Exception("This currently only works for one system!")

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
        self.port.write('00') # stub data
        # if system in self.systems:
        #     pcidio.sendRowByte(self.systems[system])

# extern uchar sendRowByte(uchar idx) {
#     uint flush = 2, msg = (idx << 3 | SEND_ROWBYTE) << 8 | (255 & rowcount[idx]);
#     comedi_dio_bitfield2(ni, 0, writemask, &msg, 0);
#     comedi_dio_bitfield2(ni, 0, 2, &flush, 16);
#     rowcount[idx]++;
#     return 0;
# }            
