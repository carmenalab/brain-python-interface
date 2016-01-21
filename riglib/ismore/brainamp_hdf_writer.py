import tables
import numpy as np
import tempfile
import datetime
import os

compfilt = tables.Filters(complevel=5, complib="zlib", shuffle=True)

class BrainampData(object):
    
    def __init__(self, brainamp_channels, sink_dtype, *args, **kwargs):

        dt = datetime.datetime.now()
        tm = dt.time()
        self.filename = '/storage/supp_hdf/tmp_'+str(dt.year)+str(dt.month)+str(dt.day)+'_'+tm.isoformat()+'.hdf'
        self.ba_h5 = tables.openFile(self.filename, "w")

        #If sink datatype is not specified: 
        if sink_dtype is None:
            self.dtype = np.dtype([('data',       np.float64),
                              ('ts_arrival', np.float64)])

            self.send_to_sinks_dtype = np.dtype([('chan'+str(chan), self.dtype) for chan in brainamp_channels])

        else:
            self.send_to_sinks_dtype = sink_dtype
        self.ba_data = self.ba_h5.createTable("/", 'brainamp', self.send_to_sinks_dtype, filters=compfilt)

    def add_data(self, data):

        #if len(data) != 1:
        # this might not be necessary
            #data = np.array(data)[np.newaxis]

        self.ba_data.append(data)

    def close_data(self):
        self.ba_h5.close()
        print "Closed brainamp hdf"



