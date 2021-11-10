'''
Base code for 'saveHDF' feature in experiments for writing data to an HDF file during experiment
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
    ''' 
    Used by the SaveHDF feature (features.hdf_features.SaveHDF) to save data 
    to an HDF file in "real-time", as the task is running
    '''
    def __init__(self, filename):
        '''
        Constructor for HDFWriter

        Parameters
        ----------
        filename : string
            Name of file to use to send data

        Returns
        -------
        HDFWriter instance
        '''
        print("HDFWriter: Saving datafile to %s"%filename)
        self.h5 = tables.open_file(filename, "w")
        self.data = {}
        self.msgs = {}
        self.f = []
    
    def register(self, name, dtype, include_msgs=True):
        '''
        Create a table in the HDF file corresponding to the specified source name and data type

        Parameters
        ----------
        system : string
            Name of the system being registered
        dtype : np.dtype instance
            Datatype of incoming data, for later decoding of the binary data during analysis
        include_msgs : boolean, optional, default=True
            Flag to indicated whether a table should be created for "msgs" from the current source (default True)

        Returns
        -------
        None
        '''
        print("HDFWriter registered %r" % name)
        print(dtype)
        if dtype.subdtype is not None:
            #just a simple dtype with a shape
            dtype, sliceshape = dtype.subdtype
            arr = self.h5.create_earray("/", name, tables.Atom.from_dtype(dtype), 
                shape=(0,)+sliceshape, filters=compfilt)
        else:
            arr = self.h5.create_table("/", name, dtype, filters=compfilt)

        self.data[name] = arr
        if include_msgs:
            msg = self.h5.create_table("/", name+"_msgs", MsgTable, filters=compfilt)
            self.msgs[name] = msg
    
    def send(self, system, data):
        '''
        Add a new row to the HDF table for 'system' and fill it with the 'data' values

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
        if system in self.data:
            if data is not None:
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
        for system in list(self.msgs.keys()):
            row = self.msgs[system].row
            row['time'] = len(self.data[system])
            row['msg'] = msg
            row.append()

    def sendAttr(self, system, attr, value):
        '''
        While the HDF writer process is running, set an attribute of the table
        (not sure that this has ever been tested..)

        Parameters
        ----------
        system : string
            Name of the table where the attribute should be set
        attr : string 
            Name of the attribute
        value : object
            Value of the attribute to set

        Returns
        -------
        None
        '''
        if system in self.data:
            self.data[system].attrs[attr] = value
    
    def close(self):
        '''
        Close the HDF file so that it saves properly after the process terminates
        '''
        self.h5.close()
        print("Closed hdf")

