'''
Base code for 'saveHDF' feature in experiments for periodically writing data to an HDF file during experiment
'''

import tables
import numpy as np

compfilt = tables.Filters(complevel=5, complib="zlib", shuffle=True)

class MsgTable(tables.IsDescription):
    '''
    Pytables custom table atom type used for the HDF tables named *_msgs
    '''
    time = tables.UIntCol()
    msg = tables.StringCol(256)

class HDFWriter(object):
    ''' Docstring '''
    def __init__(self, filename):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        print "Saving datafile to %s"%filename
        self.h5 = tables.openFile(filename, "w")
        self.data = {}
        self.msgs = {}
    
    def register(self, name, dtype, include_msgs=True):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        print "HDFWriter registered %r" % name
        if dtype.subdtype is not None:
            #just a simple dtype with a shape
            dtype, sliceshape = dtype.subdtype
            arr = self.h5.createEArray("/", name, tables.Atom.from_dtype(dtype), 
                shape=(0,)+sliceshape, filters=compfilt)
        else:
            arr = self.h5.createTable("/", name, dtype, filters=compfilt)

        self.data[name] = arr
        if include_msgs:
            msg = self.h5.createTable("/", name+"_msgs", MsgTable, filters=compfilt)
            self.msgs[name] = msg
    
    def send(self, system, data):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''

        if system in self.data:
            if len(data) != 1:
                # this might not be necessary
                data = np.array(data)[np.newaxis]
            self.data[system].append(data)

    def sendMsg(self, msg):
        '''
        Write a string to the *_msgs table for each system registered with the HDF sink

        Parameters
        ----------
        msg : string
            Message to link to the current row of the HDF table

        Returns
        -------
        None
        '''
        for system in self.msgs.keys():
            row = self.msgs[system].row
            row['time'] = len(self.data[system])
            row['msg'] = msg
            row.append()

    def sendAttr(self, system, attr, value):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        if system in self.data:
            self.data[system].attrs[attr] = value
    
    def close(self):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        self.h5.close()
        print "Closed hdf"

class PlexRelayWriter(HDFWriter):
    ''' Docstring '''
    def __init__(self, filename, device="/dev/comedi0"):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        import nidaq
        self.ni = nidaq.SendRow(device)
        super(PlexRelayWriter, self).__init__(filename)

    def register(self, system, dtype):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        self.ni.register(system, dtype)
        super(PlexRelayWriter, self).register(system, dtype)

    def send(self, system, data):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        row = len(self.data[system])
        self.ni.send(system, row)
        super(PlexRelayWriter, self).send(system, data)

    def sendMsg(self, msg):
        '''
        Docstring

        Parameters
        ----------

        Returns
        -------
        '''
        self.ni.sendMsg(msg)
        super(PlexRelayWriter, self).sendMsg(msg)
