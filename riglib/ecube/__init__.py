from pyeCubeStream import eCubeStream
import numpy as np
import time

'''
#to do list
#a method to check connection instead of pulling data once
# modify the return type of the get function
'''

from riglib.source import DataSourceSystem
class LFP(DataSourceSystem):
    '''
    wrapper class for pyecubestream
    pyecube already implemented, start, stop, and get
    here, we just wrap it under DataSourceSystem
    '''
    #as required by a DataSourceSystem
    update_freq = 1000.
    dtype = np.dtype('float')

    #need to decide if we do processing here? 
    #for future work, we need to decide 
    # which channels
    # chan_offset if there needs to be

    def __init__(self):
        '''
        Constructor for ecube.LFP

        Parameters
        ----------


        Returns
        -------
        ecube.LFP instance
        '''

        #ecubeStream by defaults stream from HS, but just to make it super clear
        self.conn = eCubeStream(source='Headstages')



        #then we can select channels if channel selection is available

    def check_conn_by_pull_data(self):
            '''
            quickly pull data twice 
            print out the process time and the  
            '''
            t1 = time.perf_counter()

            test_conn_data = self.get()
            ecube_timestamp_1 = test_conn_data[0]

            test_conn_data = self.get()
            ecube_timestamp_2 = test_conn_data[0]

            t2 = time.perf_counter() 
            ecube_ts_in_ms = (ecube_timestamp_2 - ecube_timestamp_1)/1e6

            print(f'takes {(t2 - t1)*1000} ms to pull two frames')
            print(f'the delta time stamp difference is {ecube_ts_in_ms} ms')
            print(f'data has the shape {test_conn_data[1].shape}')

    #wrapper functions to  pyecubestream

    def start(self):
        self.conn.start()
        #quickly to pull data twice to check connections
        try:
            self.check_conn_by_pull_data()
        except:
            raise Exception('Connection to ecube failed')
    
    def stop(self):
        self.conn.stop()

    def get(self):
        '''
        do some data attenuation before returning
        '''
        data_block = self.conn.get() #in the form of (time_stamp, data)
        return data_block
        


#quick test
if __name__ == "__main__":
    #this will pull data twice and pull bunch of things
    lfp = LFP()
    lfp.start()
    lfp.stop()

    


