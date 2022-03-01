'''
HDF-saving features
'''
import tempfile
import os
from riglib import sink
from riglib.experiment.experiment import control_decorator
import hdfwriter # from riglib.hdfwriter import hdfwriter # <-- now a package on PyPI

log_path = os.path.join(os.path.dirname(__file__), '../log/hdf_sink.log')

class SaveHDF():
    '''
    Saves data from registered sources into tables in an HDF file
    '''

    def init(self):
        '''
        Secondary init function. See riglib.experiment.Experiment.init()
        Prior to starting the task, this 'init' starts an HDFWriter sink.
        '''
        self.h5file = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
        self.h5file.flush()
        self.h5file.close()

        sink_manager = sink.SinkManager.get_instance()
        self.hdf = sink_manager.start(hdfwriter.HDFWriter, filename=self.h5file.name, log_filename=log_path)

        self.h5file_name = self.h5file.name

        super(SaveHDF, self).init()    

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

    @control_decorator
    def record_annotation(self, msg=""):
        """ Record a user-input annotation """
        self.hdf.sendMsg("annotation: " + msg)
        print("Saved annotation to HDF: " + msg)

    def get_h5_filename(self):
        return self.h5file.name        

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

        # this 'if' is needed because the remote procedure call to save_data doesn't like kwargs
        dbname = kwargs['dbname'] if 'dbname' in kwargs else 'default'
        if dbname == 'default':
            database.save_data(self.h5file.name, "hdf", saveid)
        else:
            database.save_data(self.h5file.name, "hdf", saveid, dbname=dbname)
