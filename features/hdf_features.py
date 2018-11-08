'''
HDF-saving features
'''
import time
import tempfile
import random
import traceback
import numpy as np
import fnmatch
import os
import subprocess
from riglib import calibrations, bmi
from riglib.bmi import extractor
from riglib.experiment import traits


class SaveHDF(object):
    '''
    Saves data from registered sources into tables in an HDF file
    '''
    def init(self):
        '''
        Secondary init function. See riglib.experiment.Experiment.init()
        Prior to starting the task, this 'init' starts an HDFWriter sink.
        '''
        from riglib import sink
        self.sinks = sink.sinks
        self.h5file = tempfile.NamedTemporaryFile()
        self.hdf = sink.sinks.start(self.sink_class, filename=self.h5file.name)

        super(SaveHDF, self).init()    

    @property
    def sink_class(self):
        '''
        Specify the sink class as a function in case future descendant classes want to use a different type of sink
        '''
        from riglib import hdfwriter
        return hdfwriter.HDFWriter

    def run(self):
        '''
        Code to execute immediately prior to the beginning of the task FSM executing, or after the FSM has finished running. 
        See riglib.experiment.Experiment.run(). This 'run' method stops the HDF sink after the FSM has finished running        
        '''
        try:
            super(SaveHDF, self).run()
        finally:
            self.hdf.stop()
    
    def join(self):
        '''
        Re-join any spawned process for cleanup
        '''
        self.hdf.join()
        super(SaveHDF, self).join()

    def set_state(self, condition, **kwargs):
        '''
        Save task state transitions to HDF

        Parameters
        ----------
        condition: string
            Name of new state to transition into. The state name must be a key in the 'status' dictionary attribute of the task

        Returns
        -------
        None
        '''
        self.hdf.sendMsg(condition)
        super(SaveHDF, self).set_state(condition, **kwargs)

    def cleanup(self, database, saveid, **kwargs):
        '''
        See LogExperiment.cleanup for documentation
        '''
        super(SaveHDF, self).cleanup(database, saveid, **kwargs)
        print("Beginning HDF file cleanup")
        print("\tHDF data currently saved to temp file: %s" % self.h5file.name)
        try:
            print("\tRunning self.cleanup_hdf()")
            self.cleanup_hdf()
        except:
            print("\n\n\n\n\nError cleaning up HDF file!")
            import traceback
            traceback.print_exc()

        dbname = kwargs['dbname'] if 'dbname' in kwargs else 'default'
        if dbname == 'default':
            database.save_data(self.h5file.name, "hdf", saveid)
        else:
            database.save_data(self.h5file.name, "hdf", saveid, dbname=dbname)
