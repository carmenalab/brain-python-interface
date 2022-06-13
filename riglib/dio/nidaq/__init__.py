'''
NIDAQ Digital I/O interface code. Higher-level Python wrapper for pcidio 
'''

try:
    import pcidio
except:
    pass

class SendAll(object):
    '''
    Interface for sending all the task-generated data through the NIDAQ interface card
    '''
    def __init__(self, device="/dev/comedi0"):
        '''
        Constructor for SendAll

        Parameters
        ----------
        device : string, optional
            comedi device to open and use for data sending; only tested for NI PCI-6503

        Returns
        -------
        SendAll instance
        '''
        self.systems = dict()

        try:
            pcidio
        except:
            raise Exception('Cannot import pcidio. Did you run its build script?')
        if pcidio.init(device) != 0:
            raise ValueError("Could not initialize comedi system")
    
    def close(self):
        '''
        Release access to the NIDAQ card so that other processes can access the device later.
        '''
        if pcidio.closeall() != 0:
            raise ValueError("Unable to close comedi system")

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
        print("nidaq register %s" % system)

        # Save the index of the system being registered (arbitrary number corresponding to the order in which systems were registered)
        self.systems[system] = pcidio.register_sys(system, str(dtype.descr))

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
        pcidio.sendMsg(str(msg))

    def sendRow(self, system, idx):
        '''
        The function pcidio.sendRow tries to send an array through the card one byte at a time. 
        It's unclear how the underlying code is supposed to get the 'idx' pointer that is required to use this function...
        .. and this function appears to be unused and probably should be removed. 
        Abandoned in place in case there is a future unknown use for this functionality
        '''
        if system in self.systems:
            pcidio.sendRow(self.systems[system], idx)

    def rstart(self, state):
        '''
        Remotely start recording from the plexon system

        Parameters
        ----------
        state : int
            0 or 1 depending on whether you want the system to start or stop
            For the plexon system, this is actually not used, and instead the comedi python bindings generate the required pulse.

        Returns
        -------
        None
        '''
        print("Sending rstart command")
        pcidio.rstart(state)

    def send(self, system, data):
        '''
        Send data through the DIO device

        Parameters
        ----------
        system : string
            Name of system where the data originated
        data : object
            Data to send. Must have a '.tostring()' attribute

        Returns
        -------
        None
        '''
        if system in self.systems:
            pcidio.sendData(self.systems[system], data.tostring())

class SendRow(SendAll):
    '''
    Send the number of rows to the plexon system. Used by riglib.hdfwriter.PlexRelayWriter, which is never actually used anywhere....
    '''
    def send(self, system, data):
        '''
        Send data to a registered system

        Parameters
        ----------
        system : string
            Name of system where the data originated
        data : object
            Argument is ignored, since only the count is sent and not the actual data

        Returns
        -------
        None
        '''
        if system in self.systems:
            pcidio.sendRowCount(self.systems[system])

class SendRowByte(SendAll):
    '''
    Send only an 8-bit data word corresponding to the 8 lower bits of the current row number of the HDF table
    '''
    def send(self, system, data):
        '''
        Send data to a registered system

        Parameters
        ----------
        system : string
            Name of system where the data originated
        data : object
            Argument is ignored, since only the count is sent and not the actual data

        Returns
        -------
        None
        '''
        if system in self.systems:
            pcidio.sendRowByte(self.systems[system])
