'''
NIDAQ Digital I/O interface code. Higher-level Python wrapper for pcidio 
'''

import parse
try:
    import pcidio
except:
    import warnings
    warnings.warn('Cannot import pcidio. Did you run its build script?')

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
        if pcidio.init(device) != 0:
            raise ValueError("Could not initialize comedi system")
    
    def close(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        if pcidio.closeall() != 0:
            raise ValueError("Unable to close comedi system")

    def register(self, system, dtype):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        print "nidaq register %s" % system
        self.systems[system] = pcidio.register_sys(system, str(dtype.descr))

    def sendMsg(self, msg):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        pcidio.sendMsg(str(msg))

    def sendRow(self, system, idx):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        if system in self.systems:
            pcidio.sendRow(self.systems[system], idx)

    def rstart(self, state):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        print "Sending rstart command"
        pcidio.rstart(state)

    def send(self, system, data):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        if system in self.systems:
            pcidio.sendData(self.systems[system], data.tostring())

class SendRow(SendAll):
    '''
    Send the number of rows to the plexon system. Used by HDFWriter
    '''
    def send(self, system, data):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        if system in self.systems:
            pcidio.sendRowCount(self.systems[system])

class SendRowByte(SendAll):
    ''' Docstring '''
    def send(self, system, data):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        if system in self.systems:
            pcidio.sendRowByte(self.systems[system])
